"""
Overfit test: verify the model architecture can learn by overfitting on a single batch.

If loss doesn't drop below 0.1 within 1000 steps, there's an architecture problem.
If it does, the model is capable of learning and just needs more training.

Usage:
    python overfit_test.py --gpu 0
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

import models
from warp_utils import temporal_blend


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patch', type=int, default=64)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda')

    # Build model
    model = models.make({
        'name': 'gaussian-fusion',
        'args': {
            'encoder_spec': {
                'name': 'edsr-baseline',
                'args': {
                    'n_resblocks': 16, 'n_feats': 64, 'res_scale': 1,
                    'scale': [1], 'no_upsampling': True,
                    'n_colors': 64, 'rgb_range': 1, 'pretrained_path': None,
                }
            },
            'spynet_pretrained': 'sintel-final',
            'n_feats': 64,
            'freeze_spynet': True,
        }
    }).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable params: {n_params:,}')

    # Create a fixed batch from real data
    to_tensor = transforms.ToTensor()
    p = args.patch

    # Load real frames
    vis_path = 'data/M3SVD/test/visible_Enhance/0111_1716'
    ir_path = 'data/M3SVD/test/infrared_Enhance/0111_1716'

    vis_0 = to_tensor(Image.open(f'{vis_path}/000001.png').convert('RGB'))[:, :p, :p].unsqueeze(0).to(device)
    ir_0 = to_tensor(Image.open(f'{ir_path}/000001.png').convert('RGB'))[:, :p, :p].unsqueeze(0).to(device)
    vis_N = to_tensor(Image.open(f'{vis_path}/000010.png').convert('RGB'))[:, :p, :p].unsqueeze(0).to(device)
    ir_N = to_tensor(Image.open(f'{ir_path}/000010.png').convert('RGB'))[:, :p, :p].unsqueeze(0).to(device)
    vis_gt = to_tensor(Image.open(f'{vis_path}/000005.png').convert('RGB'))[:, :p, :p].unsqueeze(0).to(device)
    ir_gt = to_tensor(Image.open(f'{ir_path}/000005.png').convert('RGB'))[:, :p, :p].unsqueeze(0).to(device)

    tau = 0.5  # frame 5 is midpoint of [1, 10]
    scale = (1.0, 1.0)

    print(f'Input shape: {vis_0.shape}')
    print(f'vis_0 range: [{vis_0.min():.3f}, {vis_0.max():.3f}]')
    print(f'vis_gt range: [{vis_gt.min():.3f}, {vis_gt.max():.3f}]')
    print(f'ir_gt range: [{ir_gt.min():.3f}, {ir_gt.max():.3f}]')

    # Optimizer - no warmup, high LR for overfit
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )
    l1_loss = nn.L1Loss()

    # Save GT for reference
    os.makedirs('overfit_vis', exist_ok=True)
    save_image(torch.cat([vis_0[0], ir_0[0], vis_gt[0], ir_gt[0]], dim=2),
               'overfit_vis/gt_reference.png')

    # Overfit loop
    model.train()
    target = (vis_gt + ir_gt) / 2.0  # Fused target (average of modalities)
    print(f'Fused target range: [{target.min():.3f}, {target.max():.3f}], mean={target.mean():.3f}')
    for step in range(1, args.steps + 1):
        optimizer.zero_grad()
        output = model(vis_0, ir_0, vis_N, ir_N, scale=scale, tau=tau)
        # Fused target (same as train.py FusionLoss): avoids conflicting gradients
        loss = l1_loss(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % 50 == 0 or step == 1:
            with torch.no_grad():
                out_clamped = output.clamp(0, 1)
                stats = getattr(model, '_debug_stats', {})
                print(f'Step {step:4d} | Loss: {loss.item():.4f} | '
                      f'Out [{output.min():.3f}, {output.max():.3f}] mean={output.mean():.3f} | '
                      f'Radii [{stats.get("radii_min",0)}, {stats.get("radii_max",0)}] '
                      f'zero={stats.get("radii_zero_frac",0):.2%} | '
                      f'L11 [{stats.get("cholesky_L11_min",0):.3f}, {stats.get("cholesky_L11_max",0):.3f}] '
                      f'L22 [{stats.get("cholesky_L22_min",0):.3f}, {stats.get("cholesky_L22_max",0):.3f}]')

            if step % 200 == 0:
                grid = torch.cat([vis_gt[0], ir_gt[0], out_clamped[0]], dim=2)
                save_image(grid, f'overfit_vis/step_{step:04d}.png')

    # Final output
    model.eval()
    with torch.no_grad():
        final = model(vis_0, ir_0, vis_N, ir_N, scale=scale, tau=tau).clamp(0, 1)
    save_image(torch.cat([vis_gt[0], ir_gt[0], final[0]], dim=2),
               'overfit_vis/final.png')
    print(f'\nFinal loss: {loss.item():.4f}')
    print(f'Final output range: [{final.min():.3f}, {final.max():.3f}]')
    print(f'Saved visualizations to overfit_vis/')

    if loss.item() < 0.1:
        print('\n✓ Model CAN learn - architecture is OK. Need more training.')
    else:
        print('\n✗ Model CANNOT overfit - architecture issue likely.')


if __name__ == '__main__':
    main()

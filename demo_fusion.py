"""
Demo script for GaussianFusion model.

Fuses visible + infrared frame pairs at arbitrary temporal position and spatial scale.

Usage:
    python demo_fusion.py \
        --vis0 frame_vis_0.png --ir0 frame_ir_0.png \
        --vis1 frame_vis_N.png --ir1 frame_ir_N.png \
        --model save/gaussian_fusion/best.pth \
        --tau 0.5 --scale 2.0 \
        --output fused_output.png
"""

import argparse
import torch
from torchvision import transforms
from PIL import Image

import models
import yaml


def main():
    parser = argparse.ArgumentParser(description='GaussianFusion Demo')
    parser.add_argument('--vis0', type=str, required=True, help='Visible anchor frame 0')
    parser.add_argument('--ir0', type=str, required=True, help='Infrared anchor frame 0')
    parser.add_argument('--vis1', type=str, required=True, help='Visible anchor frame N')
    parser.add_argument('--ir1', type=str, required=True, help='Infrared anchor frame N')
    parser.add_argument('--model', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--config', type=str, default=None,
                        help='Config YAML (optional, for model spec)')
    parser.add_argument('--tau', type=float, default=0.5,
                        help='Temporal position in (0, 1)')
    parser.add_argument('--scale', type=str, default='1.0',
                        help='Spatial scale (e.g., "2.0" or "2.0,3.0" for h,w)')
    parser.add_argument('--output', type=str, default='fused_output.png',
                        help='Output file path')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    to_tensor = transforms.ToTensor()

    # Parse scale
    if ',' in args.scale:
        scale_h, scale_w = [float(x) for x in args.scale.split(',')]
    else:
        scale_h = scale_w = float(args.scale)

    # Load images
    vis_0 = to_tensor(Image.open(args.vis0).convert('RGB')).unsqueeze(0).to(device)
    ir_0 = to_tensor(Image.open(args.ir0).convert('RGB')).unsqueeze(0).to(device)
    vis_N = to_tensor(Image.open(args.vis1).convert('RGB')).unsqueeze(0).to(device)
    ir_N = to_tensor(Image.open(args.ir1).convert('RGB')).unsqueeze(0).to(device)

    # Load model
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        model = models.make(config['model']).to(device)
    else:
        # Default model config
        model = models.make({
            'name': 'gaussian-fusion',
            'args': {
                'encoder_spec': {
                    'name': 'edsr-baseline',
                    'args': {
                        'n_resblocks': 16,
                        'n_feats': 64,
                        'res_scale': 1,
                        'scale': [1],
                        'no_upsampling': True,
                        'n_colors': 64,
                        'rgb_range': 1,
                        'pretrained_path': None,
                    }
                },
                'spynet_pretrained': 'sintel-final',
                'n_feats': 64,
                'freeze_spynet': True,
            }
        }).to(device)

    # Load checkpoint
    ckpt = torch.load(args.model, map_location=device)
    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    # Inference
    with torch.no_grad():
        output = model(vis_0, ir_0, vis_N, ir_N,
                       scale=(scale_h, scale_w), tau=args.tau)
        output = output.clamp(0, 1)

    # Save
    out_img = transforms.ToPILImage()(output.squeeze(0).cpu())
    out_img.save(args.output)
    print(f'Saved fused output to {args.output}')
    print(f'  Input size: {vis_0.shape[2]}x{vis_0.shape[3]}')
    print(f'  Output size: {output.shape[2]}x{output.shape[3]}')
    print(f'  Scale: ({scale_h}, {scale_w}), Tau: {args.tau}')


if __name__ == '__main__':
    main()

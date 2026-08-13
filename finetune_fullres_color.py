"""
Fine-tune the full-resolution color head (conv_color_full) on a small set of
test pairs, freezing all other weights. This enables the `full_res_color`
inference switch that restores small moving objects which the coarse
PixelUnshuffle grid previously averaged away.

Usage:
    python finetune_fullres_color.py --model save/gaussian_fusion/epoch_30.pth \
        --out save/gaussian_fusion/epoch_30_fullres_color.pth --steps 600
"""
import argparse
import os
import random

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

import yaml
import models


def parse_scale(s):
    if ',' in s:
        h, w = [float(x) for x in s.split(',')]
    else:
        h = w = float(s)
    return h, w


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, help='Base checkpoint to load')
    p.add_argument('--out', required=True, help='Output checkpoint path')
    p.add_argument('--config', default='configs/train/train-m3svd-fusion.yaml')
    p.add_argument('--data_root', default='data/M3SVD/test')
    p.add_argument('--video', default=None, help='Comma-separated test videos (default: all)')
    p.add_argument('--interval', type=int, default=8, help='Anchor frame interval')
    p.add_argument('--patch', type=int, default=224, help='Crop patch size')
    p.add_argument('--steps', type=int, default=600)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--gpu', default='0')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg['model']['args']['full_res_color'] = True
    model = models.make(cfg['model']).to(device)
    ckpt = torch.load(args.model, map_location=device)
    state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    model.load_state_dict(state, strict=False)

    # Freeze everything except conv_color_full
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.conv_color_full.parameters():
        p.requires_grad_(True)
    model.full_res_color = True
    model.full_res_color_trained = True
    model.train()

    opt = torch.optim.Adam(model.conv_color_full.parameters(), lr=args.lr)

    # Collect test clips
    root = os.path.join(args.data_root, 'visible_Enhance')
    videos = sorted(os.listdir(root)) if not args.video else args.video.split(',')
    clips = []
    for v in videos:
        d = os.path.join(root, v)
        if not os.path.isdir(d):
            continue
        frames = sorted([x for x in os.listdir(d) if x.endswith('.png')])
        if len(frames) > args.interval:
            clips.append((v, frames))
    if not clips:
        raise RuntimeError('No clips found')

    to_tensor = transforms.ToTensor()

    def load_pair(clip):
        v, frames = clip
        left = random.randint(0, len(frames) - args.interval - 1)
        right = left + args.interval
        mid = random.randint(left + 1, right - 1)
        vis_dir = os.path.join(args.data_root, 'visible_Enhance', v)
        ir_dir = os.path.join(args.data_root, 'infrared_Enhance', v)
        imgs = []
        for idx in (left, right, mid):
            for mod_dir in (vis_dir, ir_dir):
                pth = os.path.join(mod_dir, frames[idx])
                imgs.append(to_tensor(Image.open(pth).convert('RGB')))
        vis0, ir0, visN, irN, vis_gt, ir_gt = imgs
        # random crop
        _, H, W = vis0.shape
        if H < args.patch or W < args.patch:
            return None
        top = random.randint(0, H - args.patch)
        lft = random.randint(0, W - args.patch)
        def crop(t):
            return t[:, top:top + args.patch, lft:lft + args.patch]
        tau = (mid - left) / (right - left)
        return {
            'vis0': crop(vis0).unsqueeze(0).to(device),
            'ir0': crop(ir0).unsqueeze(0).to(device),
            'visN': crop(visN).unsqueeze(0).to(device),
            'irN': crop(irN).unsqueeze(0).to(device),
            'vis_gt': crop(vis_gt).unsqueeze(0).to(device),
            'ir_gt': crop(ir_gt).unsqueeze(0).to(device),
            'tau': torch.tensor([tau], dtype=torch.float32, device=device),
        }

    print(f'Fine-tuning conv_color_full on {len(clips)} clips, {args.steps} steps')
    loss_ema = None
    for step in range(1, args.steps + 1):
        b = load_pair(random.choice(clips))
        if b is None:
            continue
        out = model(b['vis0'], b['ir0'], b['visN'], b['irN'],
                    scale=(1.0, 1.0), tau=b['tau']).clamp(0, 1)
        loss = F.l1_loss(out, b['vis_gt']) + 0.5 * F.l1_loss(out, b['ir_gt'])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.conv_color_full.parameters(), max_norm=1.0)
        opt.step()
        loss_ema = loss.item() if loss_ema is None else 0.98 * loss_ema + 0.02 * loss.item()
        if step % 100 == 0 or step == 1:
            print(f'step {step:4d}: loss={loss.item():.4f} ema={loss_ema:.4f}')

    # Save: full model weights (only conv_color_full changed) + flag
    model.eval()
    new_state = model.state_dict()
    torch.save({'model': new_state, 'full_res_color_trained': True}, args.out)
    print(f'Saved fine-tuned checkpoint -> {args.out}')
    print('Inference: demo_fusion.py will auto-enable full_res_color when it finds conv_color_full weights.')


if __name__ == '__main__':
    main()

"""
Demo/test script for GaussianFusion model.

Single pair usage:
    python demo_fusion.py \
        --vis0 frame_vis_0.png --ir0 frame_ir_0.png \
        --vis1 frame_vis_N.png --ir1 frame_ir_N.png \
        --model save/gaussian_fusion/best.pth \
        --tau 0.5 --scale 2.0 \
        --output fused_output.png

M3SVD test GIF usage:
    python demo_fusion.py \
        --data_root data/M3SVD/test \
        --model save/gaussian_fusion/best.pth \
        --config configs/train/train-m3svd-fusion.yaml \
        --output_dir save/m3svd_test_gifs \
        --seconds 2 --fps 30 --anchor_interval 4 --scale 1.0
"""

import argparse
import os
from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

import models
import yaml


ENHANCE_VIS = 'visible_Enhance'
ENHANCE_IR = 'infrared_Enhance'


def parse_scale(scale_arg):
    if ',' in scale_arg:
        scale_h, scale_w = [float(x) for x in scale_arg.split(',')]
    else:
        scale_h = scale_w = float(scale_arg)
    return scale_h, scale_w


def load_model(model_path, config_path, device):
    if config_path:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        model = models.make(config['model']).to(device)
    else:
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

    ckpt = torch.load(model_path, map_location=device)
    state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


def load_rgb(path):
    return Image.open(path).convert('RGB')


def save_gif(frames, path, fps=30, loop=0):
    if len(frames) == 0:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    duration = max(1, round(1000 / fps))
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=loop,
        optimize=False,
    )


def tensor_to_pil(x):
    return transforms.ToPILImage()(x.detach().clamp(0, 1).cpu())


def select_anchor_indices(target_idx, num_frames, interval):
    """Choose two anchor frames around a target index inside a continuous clip."""
    if num_frames < 2:
        raise ValueError('Need at least 2 frames for temporal interpolation')
    interval = min(max(1, interval), num_frames - 1)
    left = target_idx - interval // 2
    left = max(0, min(left, num_frames - 1 - interval))
    right = left + interval
    tau = (target_idx - left) / interval
    return left, right, tau


def list_test_videos(data_root):
    vis_root = Path(data_root) / ENHANCE_VIS
    ir_root = Path(data_root) / ENHANCE_IR
    if not vis_root.exists():
        raise FileNotFoundError(f'Visible Enhance folder not found: {vis_root}')
    if not ir_root.exists():
        raise FileNotFoundError(f'Infrared Enhance folder not found: {ir_root}')
    videos = []
    for video_dir in sorted(vis_root.iterdir()):
        if video_dir.is_dir() and (ir_root / video_dir.name).is_dir():
            videos.append(video_dir.name)
    return videos


def common_frames(data_root, video):
    vis_dir = Path(data_root) / ENHANCE_VIS / video
    ir_dir = Path(data_root) / ENHANCE_IR / video
    vis_files = {p.name: p for p in vis_dir.glob('*.png')}
    ir_files = {p.name: p for p in ir_dir.glob('*.png')}
    names = sorted(set(vis_files) & set(ir_files))
    return [(vis_files[name], ir_files[name]) for name in names]


def run_m3svd_gif_test(args, model, device):
    """Run Enhance-only M3SVD test videos and save original/fused GIFs."""
    to_tensor = transforms.ToTensor()
    scale_h, scale_w = parse_scale(args.scale)
    all_videos = list_test_videos(args.data_root)

    if args.videos:
        requested = {v.strip() for v in args.videos.split(',') if v.strip()}
        videos = [v for v in all_videos if v in requested]
        missing = sorted(requested - set(videos))
        if missing:
            print(f'Warning: videos not found and skipped: {missing}')
    else:
        videos = all_videos

    if args.max_videos is not None:
        videos = videos[:args.max_videos]
    if len(videos) == 0:
        raise RuntimeError('No test videos selected')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    clip_len = max(2, round(args.seconds * args.fps))
    start_idx = max(0, args.start_frame - 1)

    print(f'Using Enhance modalities only: {ENHANCE_VIS}, {ENHANCE_IR}')
    print(f'Selected {len(videos)} video(s), clip_len≈{clip_len} frames, scale=({scale_h}, {scale_w})')

    for video in tqdm(videos, desc='M3SVD test videos'):
        frame_pairs = common_frames(args.data_root, video)
        if len(frame_pairs) < 2:
            print(f'Warning: skip {video}, not enough paired frames')
            continue

        end_idx = min(start_idx + clip_len, len(frame_pairs))
        if end_idx - start_idx < 2:
            print(f'Warning: skip {video}, selected range has <2 frames')
            continue
        clip_pairs = frame_pairs[start_idx:end_idx]

        # Save original continuous GIFs for both modalities.
        vis_frames = [load_rgb(vis_path) for vis_path, _ in clip_pairs]
        ir_frames = [load_rgb(ir_path) for _, ir_path in clip_pairs]
        save_gif(vis_frames, output_dir / f'{video}_visible_original.gif', fps=args.fps)
        save_gif(ir_frames, output_dir / f'{video}_infrared_original.gif', fps=args.fps)

        # Model-based spatiotemporal SR + fusion. One fused frame is generated for
        # each time index by selecting two surrounding anchors from the 2s clip.
        fused_frames = []
        if args.save_frames:
            frame_dir = output_dir / video / 'fused_frames'
            frame_dir.mkdir(parents=True, exist_ok=True)

        for target_idx in tqdm(range(len(clip_pairs)), desc=f'{video} fused', leave=False):
            left, right, tau = select_anchor_indices(target_idx, len(clip_pairs), args.anchor_interval)
            vis0 = to_tensor(load_rgb(clip_pairs[left][0])).unsqueeze(0).to(device)
            ir0 = to_tensor(load_rgb(clip_pairs[left][1])).unsqueeze(0).to(device)
            vis1 = to_tensor(load_rgb(clip_pairs[right][0])).unsqueeze(0).to(device)
            ir1 = to_tensor(load_rgb(clip_pairs[right][1])).unsqueeze(0).to(device)
            tau_t = torch.tensor([tau], dtype=torch.float32, device=device)

            with torch.inference_mode():
                fused = model(vis0, ir0, vis1, ir1, scale=(scale_h, scale_w), tau=tau_t)
                fused_img = tensor_to_pil(fused.squeeze(0))

            fused_frames.append(fused_img)
            if args.save_frames:
                fused_img.save(frame_dir / f'{target_idx + 1:06d}_tau{tau:.3f}.png')

        save_gif(
            fused_frames,
            output_dir / f'{video}_fused_interp_x{scale_h:g}_N{args.anchor_interval}.gif',
            fps=args.fps,
        )

        print(f'Saved GIFs for {video} -> {output_dir}')


def run_single_pair(args, model, device):
    to_tensor = transforms.ToTensor()
    scale_h, scale_w = parse_scale(args.scale)

    required = [args.vis0, args.ir0, args.vis1, args.ir1]
    if any(x is None for x in required):
        raise ValueError('Single-pair mode requires --vis0 --ir0 --vis1 --ir1')

    vis_0 = to_tensor(load_rgb(args.vis0)).unsqueeze(0).to(device)
    ir_0 = to_tensor(load_rgb(args.ir0)).unsqueeze(0).to(device)
    vis_N = to_tensor(load_rgb(args.vis1)).unsqueeze(0).to(device)
    ir_N = to_tensor(load_rgb(args.ir1)).unsqueeze(0).to(device)

    with torch.inference_mode():
        output = model(vis_0, ir_0, vis_N, ir_N,
                       scale=(scale_h, scale_w), tau=args.tau)
        output = output.clamp(0, 1)

    out_img = tensor_to_pil(output.squeeze(0))
    out_img.save(args.output)
    print(f'Saved fused output to {args.output}')
    print(f'  Input size: {vis_0.shape[2]}x{vis_0.shape[3]}')
    print(f'  Output size: {output.shape[2]}x{output.shape[3]}')
    print(f'  Scale: ({scale_h}, {scale_w}), Tau: {args.tau}')


def main():
    parser = argparse.ArgumentParser(description='GaussianFusion Demo')
    parser.add_argument('--vis0', type=str, default=None, help='Visible anchor frame 0')
    parser.add_argument('--ir0', type=str, default=None, help='Infrared anchor frame 0')
    parser.add_argument('--vis1', type=str, default=None, help='Visible anchor frame N')
    parser.add_argument('--ir1', type=str, default=None, help='Infrared anchor frame N')
    parser.add_argument('--model', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--config', type=str, default=None,
                        help='Config YAML (optional, for model spec)')
    parser.add_argument('--tau', type=float, default=0.5,
                        help='Temporal position in (0, 1)')
    parser.add_argument('--scale', type=str, default='1.0',
                        help='Spatial scale (e.g., "2.0" or "2.0,3.0" for h,w)')
    parser.add_argument('--output', type=str, default='fused_output.png',
                        help='Output file path')
    parser.add_argument('--gpu', type=str, default=None, help='GPU id, e.g. 0')

    # M3SVD Enhance-only GIF test mode.
    parser.add_argument('--data_root', type=str, default=None,
                        help='M3SVD test root containing visible_Enhance and infrared_Enhance')
    parser.add_argument('--output_dir', type=str, default='save/m3svd_test_gifs',
                        help='Directory for original and fused GIFs')
    parser.add_argument('--videos', type=str, default=None,
                        help='Comma-separated video ids to test, e.g. 0111_1716,0111_1753')
    parser.add_argument('--max_videos', type=int, default=None,
                        help='Limit number of test videos')
    parser.add_argument('--seconds', type=float, default=2.0,
                        help='Approximate duration of continuous test clip')
    parser.add_argument('--fps', type=int, default=30,
                        help='FPS used to select about seconds*fps frames and GIF playback speed')
    parser.add_argument('--start_frame', type=int, default=1,
                        help='1-indexed first frame in each test video')
    parser.add_argument('--anchor_interval', type=int, default=4,
                        help='Temporal distance between the two anchor frames used for each fused output')
    parser.add_argument('--save_frames', action='store_true',
                        help='Also save fused PNG frames for each video')
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_model(args.model, args.config, device)

    if args.data_root:
        run_m3svd_gif_test(args, model, device)
    else:
        run_single_pair(args, model, device)


if __name__ == '__main__':
    main()

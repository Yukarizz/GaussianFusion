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
        --seconds 2 --fps 30 --interp_factor 2 --anchor_interval 4 --scale 1.0

Random pair tau sweep usage:
    python demo_fusion.py \
        --data_root data/M3SVD/test \
        --model save/gaussian_fusion/best.pth \
        --config configs/train/train-m3svd-fusion.yaml \
        --output_dir save/m3svd_tau_sweep \
        --random_pair_sweep --anchor_interval 8 --scale 1.0
"""

import argparse
import os
import random
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
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys:
        print(f'Warning: missing checkpoint keys initialized randomly: {incompatible.missing_keys}')
    if incompatible.unexpected_keys:
        print(f'Warning: ignored unexpected checkpoint keys: {incompatible.unexpected_keys}')
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


def select_anchor_indices_at_time(target_pos, num_frames, interval):
    """Choose two anchors around a fractional target time inside a clip.

    Args:
        target_pos: Fractional frame position, e.g. 10.5 means halfway between
            source frames 10 and 11.
        num_frames: Number of source frames in the clip.
        interval: Anchor distance in source-frame units.

    Returns:
        left, right, tau where tau=(target_pos-left)/(right-left).
    """
    if num_frames < 2:
        raise ValueError('Need at least 2 frames for temporal interpolation')
    interval = min(max(1, interval), num_frames - 1)
    left = int(round(target_pos - interval / 2.0))
    left = max(0, min(left, num_frames - 1 - interval))
    right = left + interval
    tau = (target_pos - left) / interval
    tau = max(0.0, min(1.0, tau))
    return left, right, tau


def make_interpolated_positions(num_frames, interp_factor):
    """Return fractional target positions for true temporal upsampling.

    interp_factor=1 returns original positions: 0, 1, 2, ...
    interp_factor=2 returns: 0, 0.5, 1, 1.5, ...
    interp_factor=4 returns: 0, 0.25, 0.5, 0.75, 1, ...
    """
    if num_frames < 2:
        return [0.0]
    interp_factor = max(1, int(interp_factor))
    positions = []
    for idx in range(num_frames - 1):
        for sub_idx in range(interp_factor):
            positions.append(idx + sub_idx / interp_factor)
    positions.append(float(num_frames - 1))
    return positions


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


def select_videos(data_root, videos_arg=None, max_videos=None):
    all_videos = list_test_videos(data_root)

    if videos_arg:
        requested = {v.strip() for v in videos_arg.split(',') if v.strip()}
        videos = [v for v in all_videos if v in requested]
        missing = sorted(requested - set(videos))
        if missing:
            print(f'Warning: videos not found and skipped: {missing}')
    else:
        videos = all_videos

    if max_videos is not None:
        videos = videos[:max_videos]
    if len(videos) == 0:
        raise RuntimeError('No test videos selected')
    return videos


def run_m3svd_gif_test(args, model, device):
    """Run Enhance-only M3SVD test videos and save original/fused GIFs."""
    to_tensor = transforms.ToTensor()
    scale_h, scale_w = parse_scale(args.scale)
    videos = select_videos(args.data_root, args.videos, args.max_videos)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    clip_len = max(2, round(args.seconds * args.fps))
    interp_factor = max(1, args.interp_factor)
    fused_fps = args.fps * interp_factor
    start_idx = max(0, args.start_frame - 1)

    print(f'Using Enhance modalities only: {ENHANCE_VIS}, {ENHANCE_IR}')
    print(f'Selected {len(videos)} video(s), clip_len≈{clip_len} source frames, scale=({scale_h}, {scale_w})')
    print(f'True temporal upsampling: interp_factor={interp_factor}, fused_gif_fps={fused_fps}')

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

        # Model-based spatiotemporal SR + fusion. Generate true extra frames by
        # evaluating the model at fractional target times between source frames.
        fused_frames = []
        if args.save_frames:
            frame_dir = output_dir / video / 'fused_frames'
            frame_dir.mkdir(parents=True, exist_ok=True)

        target_positions = make_interpolated_positions(len(clip_pairs), interp_factor)
        for out_idx, target_pos in enumerate(tqdm(target_positions, desc=f'{video} fused', leave=False), start=1):
            left, right, tau = select_anchor_indices_at_time(target_pos, len(clip_pairs), args.anchor_interval)
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
                fused_img.save(frame_dir / f'{out_idx:06d}_pos{target_pos:.3f}_tau{tau:.3f}.png')

        save_gif(
            fused_frames,
            output_dir / f'{video}_fused_interp_x{scale_h:g}_N{args.anchor_interval}_f{interp_factor}.gif',
            fps=fused_fps,
        )

        print(f'Saved GIFs for {video} -> {output_dir} ({len(clip_pairs)} source frames, {len(fused_frames)} fused frames)')


def run_random_pair_tau_sweep(args, model, device):
    """For each video, randomly choose two frames N apart and save τ=0.1..0.9 outputs."""
    to_tensor = transforms.ToTensor()
    scale_h, scale_w = parse_scale(args.scale)
    interval = max(1, args.anchor_interval)
    tau_values = [i / 10.0 for i in range(1, 10)]
    videos = select_videos(args.data_root, args.videos, args.max_videos)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.random_seed is not None:
        random.seed(args.random_seed)

    print(f'Using Enhance modalities only: {ENHANCE_VIS}, {ENHANCE_IR}')
    print(f'Random pair tau sweep: interval={interval} frames, tau={tau_values}, scale=({scale_h}, {scale_w})')

    for video in tqdm(videos, desc='Random pair tau sweep'):
        frame_pairs = common_frames(args.data_root, video)
        if len(frame_pairs) <= interval:
            print(f'Warning: skip {video}, need > {interval} paired frames but got {len(frame_pairs)}')
            continue

        left = random.randint(0, len(frame_pairs) - interval - 1)
        right = left + interval
        video_dir = output_dir / video / f'random_pair_{left + 1:06d}_{right + 1:06d}_N{interval}'
        video_dir.mkdir(parents=True, exist_ok=True)

        vis0_img = load_rgb(frame_pairs[left][0])
        ir0_img = load_rgb(frame_pairs[left][1])
        vis1_img = load_rgb(frame_pairs[right][0])
        ir1_img = load_rgb(frame_pairs[right][1])
        vis0_img.save(video_dir / f'anchor0_visible_{left + 1:06d}.png')
        ir0_img.save(video_dir / f'anchor0_infrared_{left + 1:06d}.png')
        vis1_img.save(video_dir / f'anchor1_visible_{right + 1:06d}.png')
        ir1_img.save(video_dir / f'anchor1_infrared_{right + 1:06d}.png')

        vis0 = to_tensor(vis0_img).unsqueeze(0).to(device)
        ir0 = to_tensor(ir0_img).unsqueeze(0).to(device)
        vis1 = to_tensor(vis1_img).unsqueeze(0).to(device)
        ir1 = to_tensor(ir1_img).unsqueeze(0).to(device)

        fused_frames = []
        for tau in tau_values:
            tau_t = torch.tensor([tau], dtype=torch.float32, device=device)
            with torch.inference_mode():
                fused = model(vis0, ir0, vis1, ir1, scale=(scale_h, scale_w), tau=tau_t)
                fused_img = tensor_to_pil(fused.squeeze(0))
            fused_img.save(video_dir / f'fused_tau_{tau:.1f}.png')
            fused_frames.append(fused_img)

        save_gif(
            fused_frames,
            video_dir / f'fused_tau_0.1_0.9_x{scale_h:g}_N{interval}.gif',
            fps=args.sweep_fps,
        )
        print(f'Saved tau sweep for {video}: frames {left + 1} -> {right + 1}, output={video_dir}')


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
                        help='Source FPS used to select about seconds*fps frames; original GIF playback FPS')
    parser.add_argument('--interp_factor', type=int, default=1,
                        help='True temporal upsampling factor for fused GIF. 1=no extra frames, 2=2x FPS, 4=4x FPS')
    parser.add_argument('--start_frame', type=int, default=1,
                        help='1-indexed first frame in each test video')
    parser.add_argument('--anchor_interval', type=int, default=4,
                        help='Temporal distance between the two anchor frames used for each fused output')
    parser.add_argument('--save_frames', action='store_true',
                        help='Also save fused PNG frames for each video')
    parser.add_argument('--random_pair_sweep', action='store_true',
                        help='New test mode: for each video randomly choose two frames N=anchor_interval apart and save tau=0.1..0.9 fused outputs')
    parser.add_argument('--random_seed', type=int, default=None,
                        help='Random seed for --random_pair_sweep frame selection')
    parser.add_argument('--sweep_fps', type=int, default=5,
                        help='GIF playback FPS for --random_pair_sweep tau sweep outputs')
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_model(args.model, args.config, device)

    if args.data_root and args.random_pair_sweep:
        run_random_pair_tau_sweep(args, model, device)
    elif args.data_root:
        run_m3svd_gif_test(args, model, device)
    else:
        run_single_pair(args, model, device)


if __name__ == '__main__':
    main()

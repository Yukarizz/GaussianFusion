"""
Metric evaluation script for GaussianFusion on the M3SVD test set.

Protocol (aligned with training, n_min=2 -> anchors (t-1, t+1), tau=0.5):
    For every interior frame t of a test video, predict the fused frame
        F_t = model(vis_{t-1}, ir_{t-1}, vis_{t+1}, ir_{t+1}, tau=0.5)
    Reference-based metrics (SSIM / MEF-SSIM / VIF / MI / Qabf) compare F_t
    against the ground-truth frames of BOTH modalities (vis_t, ir_t).
    Flow-based metrics (BiSWE / MS2R) use 3-frame clips
        fused: [F_{t-1}, F_t, F_{t+1}]
        ref1 : [vis_{t-1}, vis_t, vis_{t+1}]   (visible)
        ref2 : [ir_{t-1}, ir_t, ir_{t+1}]      (infrared)
    and therefore require t-1 and t+1 to also be interior frames.

Usage (inline inference, model runs during evaluation):
    python evaluate_metrics.py \
        --data_root data/M3SVD/test \
        --model save/gaussian_fusion/best.pth \
        --config configs/train/train-m3svd-fusion.yaml \
        --metrics ssim,mef_ssim,vif,mi,qabf,biswe,ms2r \
        --output save/metric_results.json

Usage (evaluate saved inference results; no model needed):
    python evaluate_metrics.py \
        --data_root data/M3SVD/test \
        --pred_root save/my_fused_results \
        --metrics ssim,mef_ssim,vif,mi,qabf,biswe,ms2r \
        --output save/metric_results.json

    --pred_root expects the fused video split into frames, one folder per
    video (same video names as the test set):
        pred_root/<video>/*.png
    or a nested layout such as demo_fusion.py's
        pred_root/<video>/fused_frames/*.png
    A flat pred_root with frames directly inside also works when exactly one
    video is selected (--videos <name>).

    Saved frames are aligned to GT indices with --pred_align:
        auto     m == n     -> full
                 m == n - 2 -> interior
                 otherwise  -> try filename numbers (name)
        full     fused[i] <-> GT frame i          (m must equal n)
        interior fused[i] <-> GT frame i + 1      (m must equal n - 2; the
                 same interior range the inline protocol predicts)
        offset   fused[i] <-> GT frame i + K      (K = --pred_offset)
        name     leading number in the fused filename must match the GT
                 frame number (e.g. 000007.png <-> GT 000007.png)
    The evaluated frame ranges are identical to the inline protocol
    (reference metrics: interior frames; flow metrics: interior 3-frame
    clips), so numbers are comparable across both modes; boundary frames of
    a "full" dump are ignored on purpose.

Notes:
    - Reference metrics are ported from VF-Bench for comparable numbers and
      run on GPU tensors in [0, 255].
    - BiSWE / MS2R use this project's own SEA-RAFT flow (models/searaft.py),
      not VF-Bench's RAFT, so no external model copy is needed.
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

import metrics as M
from demo_fusion import (
    load_model, load_rgb, list_test_videos, common_frames, select_videos,
    parse_scale,
)


ENHANCE_VIS = 'visible_Enhance'
ENHANCE_IR = 'infrared_Enhance'


def to_device_tensor(img, device, to_tensor):
    return to_tensor(img).unsqueeze(0).to(device)


# ---------------------------------------------------------------------------
# Loading saved inference results (fused video split into frames)
# ---------------------------------------------------------------------------
def find_pred_dir(pred_root, video, allow_flat=False):
    """Locate the folder holding fused frames for one video under pred_root.

    Accepted layouts (checked in order):
        pred_root/<video>/*.png
        pred_root/<video>/fused_frames/*.png
        pred_root/<video>/*/*.png  (single sub-directory, e.g. fused_frames)
        pred_root/*.png             (flat layout, only when allow_flat=True,
                                     i.e. exactly one video is selected)
    """
    root = Path(pred_root)
    candidates = [
        root / video,
        root / video / 'fused_frames',
    ]
    for cand in candidates:
        if cand.is_dir() and list(cand.glob('*.png')):
            return cand
    video_dir = root / video
    if video_dir.is_dir():
        subdirs = [d for d in sorted(video_dir.iterdir())
                   if d.is_dir() and list(d.glob('*.png'))]
        if len(subdirs) == 1:
            return subdirs[0]
        if len(subdirs) > 1:
            print(f'Warning: multiple frame folders under {video_dir}, '
                  f'using {subdirs[0]}')
            return subdirs[0]
    # Flat layout: frames directly under pred_root (single-video mode only).
    if allow_flat and root.is_dir() and list(root.glob('*.png')):
        return root
    return None


def list_pred_frames(pred_dir):
    """Return fused frames as (path, leading_number_or_None) sorted by name."""
    items = []
    for p in sorted(pred_dir.glob('*.png')):
        items.append((p, _leading_number(p)))
    return items


def _leading_number(path):
    """Extract the leading integer from a filename like 000007.png."""
    stem = Path(path).stem
    digits = ''
    for ch in stem:
        if ch.isdigit():
            digits += ch
        else:
            break
    return int(digits) if digits else None


def resolve_pred_alignment(pred_items, n_gt, align, offset):
    """Map each saved fused frame to a GT index (0-based).

    Args:
        pred_items: list of (path, frame_number_or_None).
        n_gt: number of GT frames for the video.
        align: 'auto' | 'full' | 'interior' | 'offset' | 'name'.
        offset: K for 'offset' mode (fused[i] <-> GT i + K).

    Returns:
        list of (gt_index, path); frames that cannot be mapped are dropped
        (they are outside the GT range and would not be scored anyway).
    """
    m = len(pred_items)
    paths = [p for p, _ in pred_items]
    numbers = [num for _, num in pred_items if num is not None]

    if align == 'auto':
        if m == n_gt:
            align = 'full'
        elif m == n_gt - 2:
            align = 'interior'
        elif numbers and len(numbers) == m:
            # Filenames carry usable frame numbers (e.g. an interpolated dump
            # whose count matches no GT protocol).
            align = 'name'
        else:
            align = 'offset'
            offset = 1 if m >= n_gt else 0
            print(f'Warning: pred frame count {m} vs GT {n_gt} does not match '
                  f'full/interior protocols; assuming offset={offset}. '
                  f'Override with --pred_align/--pred_offset.')

    if align == 'full':
        if m != n_gt:
            print(f'Warning: full alignment expects {n_gt} frames, got {m}; '
                  f'extra frames ignored / missing frames skipped.')
        gt_idx = list(range(m))
    elif align == 'interior':
        if m != n_gt - 2:
            print(f'Warning: interior alignment expects {n_gt - 2} frames, '
                  f'got {m}; extra frames ignored / missing frames skipped.')
        gt_idx = list(range(1, 1 + m))
    elif align == 'offset':
        gt_idx = [i + offset for i in range(m)]
    elif align == 'name':
        gt_idx = []
        for p, num in pred_items:
            if num is None:
                num = _leading_number(p)
            if num is None:
                raise ValueError(
                    f'Cannot extract frame number from {p.name} in name mode')
            gt_idx.append(num - 1)  # filenames are 1-based
    else:
        raise ValueError(f'Unknown --pred_align: {align}')

    return [(gi, p) for gi, p in zip(gt_idx, paths)
            if 0 <= gi < n_gt]


@torch.inference_mode()
def load_fused_frames_from_disk(pred_items, device, to_tensor, max_frames=None):
    """Load saved fused frames as tensors.

    Returns:
        fused: list of [3, H, W] tensors indexed by GT position; entries
               without a saved frame are None (boundary or missing).
        n: number of GT positions covered (max gt index + 1, capped by
           max_frames).
    """
    max_gt = max(gi for gi, _ in pred_items)
    if max_frames is not None:
        max_gt = min(max_gt, max_frames - 1)
    n = max_gt + 1

    fused = [None] * n
    for gi, path in pred_items:
        if gi < n:
            fused[gi] = to_tensor(load_rgb(path)).to(device)
    return fused, n


@torch.inference_mode()
def generate_fused_frames(model, frame_pairs, device, scale, to_tensor,
                          max_frames=None):
    """Predict one fused frame per interior index t using anchors (t-1, t+1).

    Returns:
        fused: list of [3, H, W] tensors in [0, 1] aligned to frame_pairs
               indices; boundary entries are None.
        vis:   list of [3, H, W] GT visible tensors (all indices).
        ir:    list of [3, H, W] GT infrared tensors (all indices).
    """
    n = len(frame_pairs)
    if max_frames is not None:
        n = min(n, max_frames)

    vis = [None] * n
    ir = [None] * n
    fused = [None] * n

    # Preload all GT frames as tensors.
    for i in range(n):
        vis[i] = to_tensor(load_rgb(frame_pairs[i][0])).to(device)
        ir[i] = to_tensor(load_rgb(frame_pairs[i][1])).to(device)

    tau_t = torch.tensor([0.5], dtype=torch.float32, device=device)
    for t in tqdm(range(1, n - 1), desc='fuse', leave=False):
        vis0 = vis[t - 1].unsqueeze(0)
        ir0 = ir[t - 1].unsqueeze(0)
        vis1 = vis[t + 1].unsqueeze(0)
        ir1 = ir[t + 1].unsqueeze(0)
        out = model(vis0, ir0, vis1, ir1, scale=scale, tau=tau_t)
        fused[t] = out.squeeze(0).clamp(0, 1)

    # Keep GT lists consistent with the (possibly capped) frame count.
    vis = vis[:n]
    ir = ir[:n]
    return fused, vis, ir, n


def evaluate_video(model, flow_eval, frame_pairs, metric_set, device,
                   scale, to_tensor, max_frames=None, biswe_occ=True,
                   preloaded_fused=None):
    """Compute all requested metrics for one video. Returns dict of lists.

    preloaded_fused: optional (fused, n) tuple from
        load_fused_frames_from_disk(); when given, the fusion model is not
        run and these saved inference results are scored instead.
    """
    if preloaded_fused is not None:
        fused, n = preloaded_fused
        # Load GT frames on the fly to the same length as the predictions.
        vis = [None] * n
        ir = [None] * n
        for i in range(n):
            vis[i] = to_tensor(load_rgb(frame_pairs[i][0])).to(device)
            ir[i] = to_tensor(load_rgb(frame_pairs[i][1])).to(device)
    else:
        fused, vis, ir, n = generate_fused_frames(
            model, frame_pairs, device, scale, to_tensor, max_frames)

    per_metric = defaultdict(list)
    has_ref = bool(metric_set & set(M.REFERENCE_METRICS))
    has_flow = bool(metric_set & set(M.FLOW_METRICS)) and flow_eval is not None

    # ---- Reference-based metrics (per interior frame) ----
    if has_ref:
        for t in range(1, n - 1):
            if fused[t] is None:
                continue  # boundary of a full dump or a missing saved frame
            pred255 = torch.round(fused[t].unsqueeze(0) * 255)
            vis255 = torch.round(vis[t].unsqueeze(0) * 255)
            ir255 = torch.round(ir[t].unsqueeze(0) * 255)
            for name in metric_set & set(M.REFERENCE_METRICS):
                fn = M.REFERENCE_METRICS[name]
                val = fn(pred255, vis255, ir255)  # [1]
                per_metric[name].append(val.item())

    # ---- Flow-based metrics (3-frame clips) ----
    if has_flow:
        for t in range(2, n - 2):
            if fused[t - 1] is None or fused[t] is None or fused[t + 1] is None:
                continue  # need a complete 3-frame clip
            g_clip = torch.stack([fused[t - 1], fused[t], fused[t + 1]]).unsqueeze(0)
            r1_clip = torch.stack([vis[t - 1], vis[t], vis[t + 1]]).unsqueeze(0)
            r2_clip = torch.stack([ir[t - 1], ir[t], ir[t + 1]]).unsqueeze(0)
            if 'biswe' in metric_set:
                val = flow_eval.biswe(g_clip, r1_clip, r2_clip,
                                      use_occlusion=biswe_occ)
                per_metric['biswe'].append(val.item())
            if 'ms2r' in metric_set:
                val = flow_eval.ms2r(g_clip, r1_clip, r2_clip)
                per_metric['ms2r'].append(val.item())

    return per_metric


def main():
    parser = argparse.ArgumentParser(description='GaussianFusion metric evaluation')
    parser.add_argument('--data_root', type=str, default='data/M3SVD/test')
    parser.add_argument('--model', type=str, default=None, help='checkpoint path')
    parser.add_argument('--config', type=str, default=None, help='model config yaml')
    parser.add_argument('--pred_root', type=str, default=None,
                        help='evaluate saved inference results from this folder '
                             'instead of running the model; expects per-video '
                             'frame folders (see module docstring). --model is '
                             'then not required.')
    parser.add_argument('--pred_align', type=str, default='auto',
                        choices=['auto', 'full', 'interior', 'offset', 'name'],
                        help='how saved fused frames map to GT frame indices')
    parser.add_argument('--pred_offset', type=int, default=0,
                        help='GT index offset K for --pred_align offset '
                             '(fused[i] <-> GT i + K)')
    parser.add_argument('--metrics', type=str,
                        default='ssim,mef_ssim,vif,mi,qabf,biswe,ms2r',
                        help='comma-separated subset of: '
                             'ssim,mef_ssim,vif,mi,qabf,biswe,ms2r')
    parser.add_argument('--output', type=str, default='save/metric_results.json')
    parser.add_argument('--videos', type=str, default=None,
                        help='comma-separated video ids (default: all)')
    parser.add_argument('--max_videos', type=int, default=None)
    parser.add_argument('--max_frames', type=int, default=None,
                        help='limit frames per video (for quick debugging)')
    parser.add_argument('--scale', type=str, default='1.0',
                        help='spatial scale, e.g. "1.0" or "2.0,2.0"')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--searaft_weights', type=str, default=None,
                        help='SEA-RAFT weights for flow metrics (default: bundled)')
    parser.add_argument('--occ_threshold', type=float, default=1.0,
                        help='BiSWE forward-backward occlusion threshold (px)')
    parser.add_argument('--no_occlusion', action='store_true',
                        help='disable occlusion masking in BiSWE')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    metric_set = {m.strip().lower() for m in args.metrics.split(',') if m.strip()}
    unknown = metric_set - set(M.REFERENCE_METRICS) - set(M.FLOW_METRICS)
    if unknown:
        raise ValueError(f'Unknown metrics: {sorted(unknown)}')
    print(f'Metrics to evaluate: {sorted(metric_set)}')

    scale_h, scale_w = parse_scale(args.scale)
    scale = (scale_h, scale_w)
    to_tensor = transforms.ToTensor()

    # ---- Load fusion model (skipped when scoring saved inference results) ----
    if args.pred_root:
        if args.model:
            print('Note: --pred_root given, --model is ignored '
                  '(evaluating saved inference results).')
        model = None
    else:
        if not args.model:
            parser.error('--model is required unless --pred_root is given')
        model = load_model(args.model, args.config, device)

    # ---- Load flow model only if needed ----
    flow_eval = None
    if metric_set & set(M.FLOW_METRICS):
        print('Loading SEA-RAFT flow model for flow-based metrics ...')
        flow_eval = M.build_flow_metrics(
            device, searaft_weights=args.searaft_weights,
            occ_threshold=args.occ_threshold)

    # ---- Iterate videos ----
    videos = select_videos(args.data_root, args.videos, args.max_videos)
    print(f'Evaluating {len(videos)} video(s), scale={scale}')
    if args.pred_root:
        print(f'Mode: saved inference results from {args.pred_root} '
              f'(align={args.pred_align}, offset={args.pred_offset})')

    all_results = {}
    global_acc = defaultdict(list)

    for video in tqdm(videos, desc='videos'):
        frame_pairs = common_frames(args.data_root, video)
        if len(frame_pairs) < 5:
            print(f'Warning: skip {video}, need >=5 paired frames, got {len(frame_pairs)}')
            continue

        preloaded_fused = None
        if args.pred_root:
            pred_dir = find_pred_dir(args.pred_root, video,
                                     allow_flat=(len(videos) == 1))
            if pred_dir is None:
                print(f'Warning: skip {video}, no fused frames found under '
                      f'{Path(args.pred_root) / video} (or flat '
                      f'{args.pred_root}); expected <video>/*.png, '
                      f'<video>/fused_frames/*.png, or a flat single-video dir.')
                continue
            pred_items = list_pred_frames(pred_dir)
            mapped = resolve_pred_alignment(
                pred_items, len(frame_pairs), args.pred_align, args.pred_offset)
            if not mapped:
                print(f'Warning: skip {video}, no saved frames map into the '
                      f'GT range ({len(pred_items)} frames found in {pred_dir}).')
                continue
            print(f'[{video}] {len(pred_items)} saved fused frames in {pred_dir}'
                  f' -> {len(mapped)} mapped to GT indices')
            preloaded_fused = load_fused_frames_from_disk(
                mapped, device, to_tensor, max_frames=args.max_frames)

        per_metric = evaluate_video(
            model, flow_eval, frame_pairs, metric_set, device, scale,
            to_tensor, max_frames=args.max_frames,
            biswe_occ=not args.no_occlusion,
            preloaded_fused=preloaded_fused)

        video_mean = {k: (sum(v) / len(v) if v else float('nan'))
                      for k, v in per_metric.items()}
        all_results[video] = video_mean
        for k, v in per_metric.items():
            global_acc[k].extend(v)
        tqdm.write(f'[{video}] ' + '  '.join(
            f'{k}={video_mean[k]:.4f}' for k in sorted(video_mean)))

    # ---- Aggregate ----
    overall = {k: (sum(v) / len(v) if v else float('nan'))
               for k, v in global_acc.items()}

    print('\n================ Overall ================')
    for k in sorted(overall):
        print(f'  {k:>9s}: {overall[k]:.6f}')

    out = {
        'overall': overall,
        'per_video': all_results,
        'config': {
            'data_root': args.data_root,
            'model': args.model,
            'pred_root': args.pred_root,
            'pred_align': args.pred_align,
            'pred_offset': args.pred_offset,
            'metrics': sorted(metric_set),
            'scale': [scale_h, scale_w],
            'occ_threshold': args.occ_threshold,
            'use_occlusion': not args.no_occlusion,
        },
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved results to {out_path}')


if __name__ == '__main__':
    main()

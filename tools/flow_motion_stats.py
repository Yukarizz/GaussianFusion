"""
M3SVD 大尺度运动体检脚本  (FILM 改造方案 Phase 0)

目的
----
在动手改网络结构之前，先量清楚训练数据里「运动到底有多大」。产出两类关键数字：

  1. 端点间位移 |flow(I0 -> IN)|
     ——视频本身的运动剧烈程度。

  2. 建模所需位移  max(tau, 1-tau) * |flow(I0 -> IN)|
     ——**这个才是高斯核 offset 真正要覆盖的距离**。
     插帧需要的是 It->I0 与 It->IN 的 task-oriented flow，而 SEA-RAFT 给出的
     是 I0->IN 的常规光流；线性假设下两者相差一个 tau 系数。
     只统计端点间位移会高估约一倍（tau=0.5 时恰好是 0.5 倍）。

所有统计都换算到 GT 分辨率（scale>1 时 anchor 是 LR，需乘回 scale）。

用法
----
    python tools/flow_motion_stats.py \
        --config configs/train/train-m3svd-fusion.yaml \
        --num-samples 2000 \
        --output .workbuddy/motion_stats.json

显存吃紧时加 --batch-size 1 --amp。
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import yaml

# 允许从任意位置运行：把仓库根目录塞进 sys.path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from datasets import make as make_dataset   # noqa: E402
from models.searaft import SeaRAFT          # noqa: E402

PERCENTILES = (50, 75, 90, 95, 99, 99.9)


def percentile_block(values):
    lines = [f'  P{p:<8}: {np.percentile(values, p):8.2f}' for p in PERCENTILES]
    lines.append(f'  {"max":<10}: {values.max():8.2f}')
    lines.append(f'  {"mean":<10}: {values.mean():8.2f}')
    return '\n'.join(lines)


def parse_frame_info(frame_info):
    """frame_info 格式 'i,i+N,i+k,N' -> N。解析失败返回 None。"""
    try:
        parts = str(frame_info).split(',')
        return int(parts[3])
    except Exception:
        return None


def recommend_levels(task_p99, patch_size, n_convs=3, safety=1.5):
    """按 FILM options.py 的公式 2^(L-1) * flow_convs[-1] 给出层数建议。

    safety: 安全系数。P99 之外还有 P99.9 和 max 的长尾，且训练过程中
    样本分布会漂移，所以要求层数上限留出余量，而不是贴着 P99 卡。
    """
    required = task_p99 * safety
    print('\n--- 4. 金字塔层数建议 ---')
    print(f'  依据：建模所需位移 P99 = {task_p99:.1f} px，'
          f'按 safety={safety} 留出余量 -> 需 ≥ {required:.1f} px')
    print('  FILM 公式：max_motion = 2^(L-1) * convs，且 H/W 须被 2^(L-1) 整除\n')
    print(f'  {"层数":<8}{"整除要求":<11}{"位移上限":<11}{"patch 兼容性":<14}结论')
    print('  ' + '-' * 58)
    best = None
    for L in range(4, 9):
        div = 2 ** (L - 1)
        cap = div * n_convs
        ok = (patch_size % div == 0)
        fits = cap >= required
        if not fits:
            comment = '上限不足'
        elif ok:
            comment = '✓ 推荐'
        else:
            comment = '✗ patch 无法整除'
        print(f'  L={L:<6}{div:<11}{cap:<11}{"兼容" if ok else "不兼容":<14}{comment}')
        if ok and fits and best is None:
            best = L

    print()
    if best is None:
        print(f'  ⚠ patch_size={patch_size} 无法整除满足上限的层数')
        print('    建议改用 192 或 256（需被 2^(L-1) 整除）')
    else:
        print(f'  ==> 建议从 L={best} 起步')
    return best


def text_histogram(values, bins=12, width=46):
    counts, edges = np.histogram(values, bins=bins)
    if counts.max() == 0:
        return '  (无数据)'
    return '\n'.join(
        f'  {lo:7.1f}-{hi:7.1f} | {"#" * int(width * c / counts.max())} {c}'
        for c, lo, hi in zip(counts, edges[:-1], edges[1:]))


def main():
    ap = argparse.ArgumentParser(description='M3SVD large-motion diagnostics')
    ap.add_argument('--config', default='configs/train/train-m3svd-fusion.yaml')
    ap.add_argument('--num-samples', type=int, default=2000,
                    help='统计样本数，默认 2000 已足够稳定')
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--iters', type=int, default=None,
                    help='覆盖 SEA-RAFT 迭代次数，体检用默认 12 即可')
    ap.add_argument('--weights', default=None, help='覆盖预训练权重路径')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--amp', action='store_true', help='用 fp16 加速体检')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--output', default='.workbuddy/motion_stats.json')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    ds_cfg = dict(config['train_dataset'])
    ds_args = ds_cfg['args']
    m_cfg = dict(config['model']['args'])

    print('=' * 58)
    print(' M3SVD 运动幅值体检')
    print('=' * 58)
    print(f'  配置文件 : {args.config}')
    print(f'  数据根   : {ds_args.get("root_path")}')
    print(f'  设备     : {args.device}   AMP: {args.amp}')

    dataset = make_dataset(ds_cfg)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=False)

    weights = args.weights or m_cfg.get('searaft_pretrained')
    if not weights:
        raise SystemExit('未找到 searaft_pretrained，请用 --weights 指定')
    print(f'  光流模型 : searaft')
    print(f'  权重     : {weights}')

    net = SeaRAFT(pretrained=weights, iters=args.iters, freeze=True)
    net = net.to(args.device).eval()

    endpoint_all, task_all = [], []
    per_n = {}
    seen, failed = 0, 0
    t0 = time.time()

    print('\n开始采样（Ctrl+C 中断后仍会统计已采部分）...')
    try:
        for batch in loader:
            vis_0 = batch['vis_anchor0'].to(args.device, non_blocking=True)
            vis_N = batch['vis_anchor1'].to(args.device, non_blocking=True)
            tau = batch['tau'].to(args.device)
            scale = batch['scale'].to(args.device)

            with torch.no_grad():
                if args.amp:
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        flow = net(vis_0, vis_N)
                else:
                    flow = net(vis_0, vis_N)

            # anchor 可能是 LR：vis_0 的 H = patch/scale_h，故 GT 位移 = LR 位移 * scale_h
            s = scale[:, 0].to(flow.dtype).view(-1, 1, 1, 1)
            mag = torch.linalg.vector_norm(flow.float(), dim=1) * s      # [B,H,W]

            # 中间帧到两个端点的位移，取较大者
            tau_ = tau.to(mag.dtype).view(-1, 1, 1)
            task_mag = torch.maximum(tau_, 1.0 - tau_) * mag

            m_np = mag.flatten(1).cpu().numpy()
            t_np = task_mag.flatten(1).cpu().numpy()
            endpoint_all.append(m_np.ravel())
            task_all.append(t_np.ravel())

            for bi, finfo in enumerate(batch['frame_info']):
                N = parse_frame_info(finfo)
                if N is not None:
                    per_n.setdefault(N, []).append(t_np[bi].ravel())

            seen += vis_0.size(0)
            el = time.time() - t0
            print(f'  已处理 {seen}/{args.num_samples}  '
                  f'({el:.0f}s, {seen / max(el, 1e-6):.1f} it/s)',
                  end='\r', flush=True)
            if seen >= args.num_samples:
                break
    except KeyboardInterrupt:
        print('\n  [中断] 用已采部分继续统计')
    except Exception as e:
        failed += 1
        print(f'\n  [错误] {type(e).__name__}: {e}')
        print('  显存不足时尝试 --batch-size 1 --amp；权重路径用 --weights 覆盖')
        if not endpoint_all:
            raise

    print()
    endpoint = np.concatenate(endpoint_all)
    task = np.concatenate(task_all)
    patch_size = ds_args.get('patch_size', 224)

    print('\n' + '=' * 58)
    print('--- 1. 端点间位移 |flow(I0 -> IN)|  (px @ GT 分辨率) ---')
    print(percentile_block(endpoint))

    print('\n--- 2. 建模所需位移 max(tau,1-tau)*|flow|  <== 关键 ---')
    print('   高斯核 offset 实际需要覆盖的距离')
    print(percentile_block(task))

    print('\n--- 3. 按帧间隔 N 分组（建模所需位移）---')
    print(f'  {"N":<5}{"样本数":<9}{"mean":<10}{"P90":<10}{"P95":<10}{"max":<10}')
    print('  ' + '-' * 52)
    for N in sorted(per_n.keys()):
        v = np.concatenate(per_n[N])
        print(f'  {N:<5}{len(per_n[N]):<9}{v.mean():<10.2f}'
              f'{np.percentile(v, 90):<10.2f}'
              f'{np.percentile(v, 95):<10.2f}{v.max():<10.2f}')

    print('\n--- 分布直方图（建模所需位移）---')
    print(text_histogram(task))

    best_L = recommend_levels(float(np.percentile(task, 99)), patch_size)

    print('\n--- 5. 训练范围建议 ---')
    p90, p95, p99 = (np.percentile(task, q) for q in (90, 95, 99))
    print(f'  你的数据 P90/P95/P99 = {p90:.1f} / {p95:.1f} / {p99:.1f} px')
    print('  FILM Sec 5.3：训练运动范围须与测试分布匹配，并非越大越好——')
    print('  范围开太大会挤压模型在小运动上的容量。')
    n_max = ds_args.get('n_max', 0)
    if n_max >= 10:
        print(f'\n  ⚠ 当前 n_max={n_max}，跨度偏大。若 P95 已充分覆盖，')
        print('    可考虑降到 8，或改为「先小跨度收敛、再放开」的课程式分档。')

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or '.', exist_ok=True)
    summary = {
        'num_samples': seen,
        'failed_batches': failed,
        'patch_size': patch_size,
        'n_min': ds_args.get('n_min'),
        'n_max': n_max,
        'flow_model': 'searaft',
        'endpoint_flow_px': {str(p): float(np.percentile(endpoint, p))
                             for p in PERCENTILES},
        'endpoint_flow_max_px': float(endpoint.max()),
        'task_displacement_px': {str(p): float(np.percentile(task, p))
                                 for p in PERCENTILES},
        'task_displacement_max_px': float(task.max()),
        'task_displacement_mean_px': float(task.mean()),
        'per_N': {str(N): {
            'count': len(per_n[N]),
            'mean': float(np.concatenate(per_n[N]).mean()),
            'p90': float(np.percentile(np.concatenate(per_n[N]), 90)),
            'p95': float(np.percentile(np.concatenate(per_n[N]), 95)),
            'max': float(np.concatenate(per_n[N]).max()),
        } for N in sorted(per_n.keys())},
        'recommended_pyramid_levels': best_L,
        'elapsed_sec': round(time.time() - t0, 1),
    }
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    np.save(args.output.replace('.json', '_samples.npy'), task.astype(np.float16))

    print(f'\n结果已保存: {args.output}')
    print(f'原始样本  : {args.output.replace(".json", "_samples.npy")}')
    print('=' * 58)
    print('\n把上面 1~5 的输出发我即可，我会据此定 L 层数、n_max 与 AOW 窗口量程。')


if __name__ == '__main__':
    main()

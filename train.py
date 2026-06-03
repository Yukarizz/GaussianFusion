"""
Training script for GaussianFusion model.

Multi-modal temporal fusion with 2D Gaussian splatting.
Supports arbitrary spatiotemporal super-resolution.
"""

import argparse
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR

import yaml
from tqdm import tqdm
import wandb
from torchvision.utils import save_image

import models
import datasets
from utils import Averager, Timer, time_text, set_log_path, log
import kornia

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def setup_distributed():
    """Initialize distributed training if launched with torchrun.
    
    Note: gsplat CUDA kernels only work on cuda:0. For multi-GPU training,
    each process must be launched with CUDA_VISIBLE_DEVICES set to a single GPU
    BEFORE Python starts (see jobs/train_gaussian_fusion.bsub for the wrapper).
    """
    if 'RANK' in os.environ:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(0)  # Each process should see only 1 GPU
        return 0
    return 0


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


class GradientLoss(nn.Module):
    """
    使用 Sobel 算子提取图像的梯度幅值，用于计算纹理损失。
    支持多通道（如 RGB = 3通道）的深度可分离卷积。
    """
    def __init__(self, channels=3):
        super().__init__()
        # 定义 Sobel 卷积核
        kernel_x = torch.tensor([[-1.,  0.,  1.], 
                                 [-2.,  0.,  2.], 
                                 [-1.,  0.,  1.]])
        kernel_y = torch.tensor([[-1., -2., -1.], 
                                 [ 0.,  0.,  0.], 
                                 [ 1.,  2.,  1.]])
        
        # 调整形状为 [out_channels, in_channels/groups, kH, kW]
        # 这里使用 depthwise 卷积，所以 in_channels/groups = 1
        self.weight_x = nn.Parameter(kernel_x.view(1, 1, 3, 3).repeat(channels, 1, 1, 1), requires_grad=False)
        self.weight_y = nn.Parameter(kernel_y.view(1, 1, 3, 3).repeat(channels, 1, 1, 1), requires_grad=False)

    def forward(self, x):
        # groups=x.shape[1] 确保分别对 R, G, B 通道计算梯度
        grad_x = F.conv2d(x, self.weight_x.to(x.device), padding=1, groups=x.shape[1])
        grad_y = F.conv2d(x, self.weight_y.to(x.device), padding=1, groups=x.shape[1])
        
        # 计算梯度幅值 (L1 范数近似)
        return torch.abs(grad_x) + torch.abs(grad_y)



class FusionLoss(nn.Module):
    """
    Decoupled combined loss for multi-modal temporal fusion.
    
    L = λ_int * L1_intensity + λ_color * L1_color + λ_grad * L1_gradient
        + λ_temporal * L_temporal
    
    1. Intensity Target: max(vis_y, ir_y) on Y channel to preserve IR thermal targets and Vis highlights.
    2. Color Target: L1 distance between fused CbCr and Vis CbCr to maintain natural colors.
    3. Gradient Target: max(|grad_vis|, |grad_ir|) to preserve sharpest textures from both modalities.
    """

    def __init__(self, lambda_int=1.0, lambda_color=1.0, lambda_grad=5.0,
                 lambda_temporal=0.0, channels=3):
        super().__init__()
        self.lambda_int = lambda_int
        self.lambda_color = lambda_color  # 色彩损失权重
        self.lambda_grad = lambda_grad 
        self.lambda_temporal = lambda_temporal
        
        self.l1 = nn.L1Loss()
        self.grad_operator = GradientLoss(channels=channels)

    def forward(self, fused, vis_gt, ir_gt, temporal_loss=None):
        """
        Args:
            fused: Model output [B, 3, H, W] (RGB).
            vis_gt: Visible GT [B, 3, H, W] (RGB).
            ir_gt: Infrared GT [B, 3, H, W] (1-channel repeated 3 times).
        """
        # ==========================================
        # 1. 强度损失 (Intensity Loss) & 色彩损失 (Color Loss)
        # ==========================================
        # 使用 kornia 将 RGB 转换到 YCbCr 色彩空间
        fused_ycbcr = kornia.color.rgb_to_ycbcr(fused)
        vis_ycbcr = kornia.color.rgb_to_ycbcr(vis_gt)
        
        # 提取 Y 通道 (亮度)，保持 shape 为 [B, 1, H, W]
        fused_y = fused_ycbcr[:, 0:1, :, :]
        vis_y = vis_ycbcr[:, 0:1, :, :]
        
        # 红外图像为单通道重复3次，提取第1个通道作为红外强度
        ir_int = ir_gt[:, 0:1, :, :]
        
        # 采用像素级最大值，保留 Vis 高光和 IR 热源 (仅在亮度通道进行)
        target_int = torch.max(vis_y, ir_int)
        loss_int = self.l1(fused_y, target_int)
        
        # [新增] 色彩损失：提取 Cb 和 Cr 通道，约束融合图像的色彩与可见光一致
        fused_cbcr = fused_ycbcr[:, 1:, :, :]
        vis_cbcr = vis_ycbcr[:, 1:, :, :]
        loss_color = self.l1(fused_cbcr, vis_cbcr)
        
        # ==========================================
        # 2. 梯度/纹理损失 (Gradient Loss)
        # ==========================================
        # 梯度损失依然可以在原始 RGB 空间计算，或者也可以改为仅在 Y 通道计算
        # 这里保持在你原有的设定（对全部通道算梯度）
        grad_fused = self.grad_operator(fused)
        with torch.no_grad():
            grad_vis = self.grad_operator(vis_gt)
            grad_ir = self.grad_operator(ir_gt)
            target_grad = torch.max(grad_vis, grad_ir) 
            
        loss_grad = self.l1(grad_fused, target_grad)
        
        # ==========================================
        # 综合计算
        # ==========================================
        total = (self.lambda_int * loss_int + 
                 self.lambda_color * loss_color + 
                 self.lambda_grad * loss_grad)
        if temporal_loss is None:
            temporal_loss = fused.new_tensor(0.0)
        total = total + self.lambda_temporal * temporal_loss
        
        return total, {
            'loss_int': loss_int.item(),
            'loss_color': loss_color.item(),
            'loss_grad': loss_grad.item(),
            'loss_temporal': temporal_loss.detach().item(),
            'loss_total': total.item(),
        }


def _resize_like(x, ref):
    if x.shape[-2:] == ref.shape[-2:]:
        return x
    return F.interpolate(x, size=ref.shape[-2:], mode='bilinear', align_corners=False)


@torch.no_grad()
def _psnr(pred, target):
    mse = F.mse_loss(pred, target, reduction='none').flatten(1).mean(dim=1).clamp_min(1e-10)
    return (-10.0 * torch.log10(mse)).mean().item()


@torch.no_grad()
def collapse_diagnostics(fused, vis_0, ir_0, vis_N, ir_N, vis_gt, ir_gt, tau):
    """Metrics that reveal endpoint-copying or naive averaging behavior."""
    vis_0 = _resize_like(vis_0, fused)
    ir_0 = _resize_like(ir_0, fused)
    vis_N = _resize_like(vis_N, fused)
    ir_N = _resize_like(ir_N, fused)
    vis_avg = 0.5 * (vis_0 + vis_N)
    ir_avg = 0.5 * (ir_0 + ir_N)
    fusion_avg = 0.5 * (vis_gt + ir_gt)

    l1_vis0 = F.l1_loss(fused, vis_0).item()
    l1_visN = F.l1_loss(fused, vis_N).item()
    l1_ir0 = F.l1_loss(fused, ir_0).item()
    l1_irN = F.l1_loss(fused, ir_N).item()
    l1_vis0_sample = (fused - vis_0).abs().flatten(1).mean(dim=1)
    l1_visN_sample = (fused - vis_N).abs().flatten(1).mean(dim=1)
    l1_ir0_sample = (fused - ir_0).abs().flatten(1).mean(dim=1)
    l1_irN_sample = (fused - ir_N).abs().flatten(1).mean(dim=1)

    return {
        'diag/l1_to_vis0': l1_vis0,
        'diag/l1_to_visN': l1_visN,
        'diag/l1_to_ir0': l1_ir0,
        'diag/l1_to_irN': l1_irN,
        'diag/l1_to_nearest_endpoint': torch.stack([
            l1_vis0_sample, l1_visN_sample, l1_ir0_sample, l1_irN_sample
        ], dim=1).min(dim=1).values.mean().item(),
        'diag/l1_to_vis_endpoint_avg': F.l1_loss(fused, vis_avg).item(),
        'diag/l1_to_ir_endpoint_avg': F.l1_loss(fused, ir_avg).item(),
        'diag/l1_to_fusion_gt_avg': F.l1_loss(fused, fusion_avg).item(),
        'diag/l1_to_vis_gt': F.l1_loss(fused, vis_gt).item(),
        'diag/l1_to_ir_gt': F.l1_loss(fused, ir_gt).item(),
        'diag/psnr_to_vis_gt': _psnr(fused, vis_gt),
        'diag/psnr_to_fusion_gt_avg': _psnr(fused, fusion_avg),
        'diag/tau_mean': tau.mean().item(),
        'diag/tau_mid_frac': ((tau > 0.375) & (tau < 0.625)).float().mean().item(),
    }


def make_data_loader(config, tag='train'):
    """Create data loader from config."""
    dataset = datasets.make(config[f'{tag}_dataset'])
    sampler = None
    shuffle = (tag == 'train')
    if dist.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        shuffle = False
    loader = DataLoader(
        dataset,
        batch_size=config.get('batch_size', 4),
        shuffle=shuffle,
        sampler=sampler,
        num_workers=config.get('num_workers', 4),
        pin_memory=True,
        drop_last=(tag == 'train'),
        persistent_workers=config.get('num_workers', 4) > 0,
    )
    return loader


def make_model(config):
    """Create model from config."""
    model = models.make(config['model'])
    return model


def train_epoch(model, loader, optimizer, criterion, device, epoch, global_step, scaler=None):
    """Train one epoch."""
    model.train()
    if hasattr(loader, 'sampler') and hasattr(loader.sampler, 'set_epoch'):
        loader.sampler.set_epoch(epoch)
    loss_avg = Averager()
    loss_ema = None
    ema_decay = 0.99
    loss_components = {
        'loss_int': Averager(),
        'loss_color': Averager(),
        'loss_grad': Averager(),
        'loss_temporal': Averager(),
    }

    use_amp = scaler is not None

    pbar = tqdm(loader, desc=f'Epoch {epoch}', leave=False, disable=not is_main_process())
    for batch in pbar:
        vis_0 = batch['vis_anchor0'].to(device, non_blocking=True)
        ir_0 = batch['ir_anchor0'].to(device, non_blocking=True)
        vis_N = batch['vis_anchor1'].to(device, non_blocking=True)
        ir_N = batch['ir_anchor1'].to(device, non_blocking=True)
        vis_gt = batch['vis_gt'].to(device, non_blocking=True)
        ir_gt = batch['ir_gt'].to(device, non_blocking=True)
        tau = batch['tau'].to(device, non_blocking=True)
        scale = batch['scale'].to(device, non_blocking=True)

        # Forward with AMP (pass full tau tensor for per-sample temporal blending)
        raw_model = model.module if dist.is_initialized() else model
        with torch.amp.autocast('cuda', enabled=use_amp):
            fused = model(vis_0, ir_0, vis_N, ir_N, scale=scale, tau=tau)
            temporal_loss = raw_model.get_temporal_regularization() if hasattr(raw_model, 'get_temporal_regularization') else None
            loss, loss_dict = criterion(fused, vis_gt, ir_gt, temporal_loss=temporal_loss)

        # Skip step if loss is NaN/Inf or spikes too far above EMA
        cur_loss = loss.item()
        if not math.isfinite(cur_loss):
            optimizer.zero_grad(set_to_none=True)
            continue
        if loss_ema is not None and cur_loss > 10 * loss_ema + 0.1:
            optimizer.zero_grad(set_to_none=True)
            continue

        # Backward with gradient scaling + clipping
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        loss_avg.add(loss.item())
        loss_ema = loss.item() if loss_ema is None else ema_decay * loss_ema + (1 - ema_decay) * loss.item()
        for k in loss_components:
            loss_components[k].add(loss_dict[k])
    
        # wandb per-step logging
        global_step += 1
        if global_step % 50 == 0 and is_main_process():
            log_dict = {
                'train/loss_total': loss_dict['loss_total'],
                'train/loss_int': loss_dict['loss_int'],
                'train/loss_color': loss_dict['loss_color'],
                'train/loss_grad': loss_dict['loss_grad'],
                'train/loss_temporal': loss_dict['loss_temporal'],
                'train/lr': optimizer.param_groups[0]['lr'],
            }
            log_dict.update(collapse_diagnostics(fused.detach(), vis_0, ir_0, vis_N, ir_N, vis_gt, ir_gt, tau))
            # Log Gaussian health metrics
            if hasattr(raw_model, '_debug_stats'):
                for k, v in raw_model._debug_stats.items():
                    prefix = 'temporal' if k.startswith('temporal_') else 'gaussian'
                    metric_name = k[len('temporal_'):] if k.startswith('temporal_') else k
                    log_dict[f'{prefix}/{metric_name}'] = v
            wandb.log(log_dict, step=global_step)

        pbar.set_postfix(
            total=f'{loss_dict["loss_total"]:.4f}',
            int=f'{loss_dict["loss_int"]:.4f}',
            color=f'{loss_dict["loss_color"]:.4f}',
            grad=f'{loss_dict["loss_grad"]:.4f}',
            temp=f'{loss_dict["loss_temporal"]:.4f}',
        )

    # Epoch-level averages
    if is_main_process():
        epoch_metrics = {
            'epoch/train_loss': loss_avg.item(),
            'epoch/loss_int': loss_components['loss_int'].item(),
            'epoch/loss_color': loss_components['loss_color'].item(),
            'epoch/loss_grad': loss_components['loss_grad'].item(),
            'epoch/loss_temporal': loss_components['loss_temporal'].item(),
            'epoch/lr': optimizer.param_groups[0]['lr'],
        }
        wandb.log(epoch_metrics, step=global_step)

    return loss_avg.item(), global_step


@torch.no_grad()
def validate(model, loader, criterion, device, global_step, log_images=False):
    """Validation step with wandb logging."""
    model.eval()
    loss_avg = Averager()
    loss_components = {
        'loss_int': Averager(),
        'loss_color': Averager(),
        'loss_grad': Averager(),
        'loss_temporal': Averager(),
    }
    sample_logged = False
    diag_logged = False

    for batch in loader:
        vis_0 = batch['vis_anchor0'].to(device)
        ir_0 = batch['ir_anchor0'].to(device)
        vis_N = batch['vis_anchor1'].to(device)
        ir_N = batch['ir_anchor1'].to(device)
        vis_gt = batch['vis_gt'].to(device)
        ir_gt = batch['ir_gt'].to(device)
        tau = batch['tau'].to(device)

        fused = model(vis_0, ir_0, vis_N, ir_N, scale=(1.0, 1.0), tau=tau)
        temporal_loss = model.get_temporal_regularization() if hasattr(model, 'get_temporal_regularization') else None
        loss, loss_dict = criterion(fused, vis_gt, ir_gt, temporal_loss=temporal_loss)
        loss_avg.add(loss.item())
        for k in loss_components:
            loss_components[k].add(loss_dict[k])

        if not diag_logged:
            diag = collapse_diagnostics(fused, vis_0, ir_0, vis_N, ir_N, vis_gt, ir_gt, tau)
            wandb.log({f'val/{k.split("/", 1)[1]}': v for k, v in diag.items()}, step=global_step)
            diag_logged = True

        # Log sample images (first batch only)
        if log_images and not sample_logged:
            n = min(4, fused.shape[0])
            images = []
            for i in range(n):
                images.append(wandb.Image(
                    torch.cat([vis_gt[i], ir_gt[i], fused[i]], dim=2).clamp(0, 1).cpu(),
                    caption=f'vis_gt | ir_gt | fused (tau={tau[0]:.2f})'))
            wandb.log({'val/samples': images}, step=global_step)
            sample_logged = True

    val_loss = loss_avg.item()
    wandb.log({
        'val/loss': val_loss,
        'val/loss_int': loss_components['loss_int'].item(),
        'val/loss_color': loss_components['loss_color'].item(),
        'val/loss_grad': loss_components['loss_grad'].item(),
        'val/loss_temporal': loss_components['loss_temporal'].item(),
    }, step=global_step)
    return val_loss


def main():
    parser = argparse.ArgumentParser(description='Train GaussianFusion')
    parser.add_argument('--config', type=str, required=True, help='Config YAML file')
    parser.add_argument('--gpu', type=str, default='0', help='GPU id (single-GPU only)')
    parser.add_argument('--save_dir', type=str, default='./save/gaussian_fusion',
                        help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--wandb_project', type=str, default='GaussianFusion',
                        help='wandb project name')
    parser.add_argument('--wandb_run', type=str, default=None, help='wandb run name')
    parser.add_argument('--wandb_offline', action='store_true', help='Run wandb offline')
    args = parser.parse_args()

    # Distributed setup (auto-detected via torchrun env vars)
    local_rank = setup_distributed()

    # Single-GPU fallback: set CUDA_VISIBLE_DEVICES if not managed by scheduler/torchrun
    if not dist.is_initialized() and 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda', 0)

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Create save directory (only rank 0)
    if is_main_process():
        os.makedirs(args.save_dir, exist_ok=True)
        set_log_path(args.save_dir)
        log(f'Config: {args.config}')
        # wandb init (rank 0 only)
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run,
            config=config,
            dir=args.save_dir,
            mode='offline' if args.wandb_offline else 'online',
        )
    else:
        os.makedirs(args.save_dir, exist_ok=True)

    # Seed
    seed = config.get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

    # Data
    train_loader = make_data_loader(config, 'train')
    if is_main_process():
        log(f'Train dataset: {len(train_loader.dataset)} samples')

    # Model
    model = make_model(config).to(device)
    if dist.is_initialized():
        model = DDP(model, device_ids=[0], find_unused_parameters=False)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main_process():
        log(f'Model parameters: {n_params:,}')

    # Optimizer & Scheduler
    lr = config.get('lr', 1e-4)
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=config.get('weight_decay', 0)
    )
    total_epochs = config.get('epochs', 100)
    warmup_epochs = config.get('warmup_epochs', 5)
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda ep: min(1.0, (ep + 1) / warmup_epochs))
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs, eta_min=lr * 0.01)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
                             milestones=[warmup_epochs])

    # Loss
    criterion = FusionLoss(
        lambda_int=config.get('lambda_int', config.get('lambda_l1', 1.0)),
        lambda_color=config.get('lambda_color', 1.0),
        lambda_grad=config.get('lambda_grad', 5.0),
        lambda_temporal=config.get('lambda_temporal', 0.0),
    )
    if is_main_process():
        log('Loss weights: '
            f'lambda_int={criterion.lambda_int}, '
            f'lambda_color={criterion.lambda_color}, '
            f'lambda_grad={criterion.lambda_grad}, '
            f'lambda_temporal={criterion.lambda_temporal}')

    # Resume
    start_epoch = 1
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        log(f'Resumed from epoch {start_epoch - 1}')

    # Training loop
    timer = Timer()
    best_loss = float('inf')
    global_step = 0
    use_amp = config.get('use_amp', True)
    scaler = torch.amp.GradScaler('cuda', growth_interval=5000) if use_amp else None

    for epoch in range(start_epoch, total_epochs + 1):
        timer.s()
        train_loss, global_step = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch, global_step, scaler)
        scheduler.step()

        elapsed = timer.t()
        if is_main_process():
            log(f'Epoch {epoch}/{total_epochs} | Loss: {train_loss:.4f} | '
                f'LR: {scheduler.get_last_lr()[0]:.2e} | Time: {time_text(elapsed)}')

        # Save visualization samples
        if is_main_process() and epoch % config.get('val_every', 5) == 0:
            vis_dir = os.path.join(args.save_dir, 'vis')
            os.makedirs(vis_dir, exist_ok=True)
            vis_model = model.module if dist.is_initialized() else model
            vis_model.eval()
            with torch.no_grad():
                sample_batch = next(iter(train_loader))
                v0 = sample_batch['vis_anchor0'][:4].to(device)
                i0 = sample_batch['ir_anchor0'][:4].to(device)
                vN = sample_batch['vis_anchor1'][:4].to(device)
                iN = sample_batch['ir_anchor1'][:4].to(device)
                tau_s = sample_batch['tau'][:4].to(device)
                sc = sample_batch['scale'][:4].to(device)
                fused = vis_model(v0, i0, vN, iN, scale=sc, tau=tau_s)
                # Layout: Row1: vis_t0 | ir_t0 | vis_tN | ir_tN
                #          Row2: fused  | avg_gt | vis_gt | ir_gt
                vgt = sample_batch['vis_gt'][:4].to(device)
                igt = sample_batch['ir_gt'][:4].to(device)
                avg_gt = (vgt + igt) / 2.0
                for j in range(min(4, fused.shape[0])):
                    row1 = torch.cat([v0[j], i0[j], vN[j], iN[j]], dim=2)
                    row2 = torch.cat([fused[j].clamp(0,1), avg_gt[j], vgt[j], igt[j]], dim=2)
                    grid = torch.cat([row1, row2], dim=1)
                    save_image(grid, os.path.join(vis_dir, f'epoch{epoch:03d}_sample{j}.png'))
                try:
                    wandb_images = [wandb.Image(
                        torch.cat([
                            torch.cat([v0[j], i0[j], vN[j], iN[j]], dim=2),
                            torch.cat([fused[j].clamp(0,1), avg_gt[j], vgt[j], igt[j]], dim=2),
                        ], dim=1).cpu(),
                        caption=f'Row1: vis_t0|ir_t0|vis_tN|ir_tN  Row2: fused|avg_gt|vis_gt|ir_gt (tau={tau_s[j]:.2f})')
                        for j in range(min(4, fused.shape[0]))]
                    wandb.log({'train/vis_samples': wandb_images}, step=global_step)
                except Exception as e:
                    log(f'  Warning: wandb image logging failed: {e}')
            vis_model.train()

        # Validation (if val loader exists)
        val_every = config.get('val_every', 5)
        if epoch % val_every == 0 and 'val_dataset' in config:
            val_loader = make_data_loader(config, 'val')
            if is_main_process():
                val_model = model.module if dist.is_initialized() else model
                val_loss = validate(val_model, val_loader, criterion, device,
                                  global_step, log_images=(epoch % (val_every * 4) == 0))
                log(f'  Val Loss: {val_loss:.4f}')

        # Save checkpoint (rank 0 only)
        if is_main_process():
            model_state = model.module.state_dict() if dist.is_initialized() else model.state_dict()
            if epoch % config.get('save_every', 10) == 0:
                ckpt = {
                    'epoch': epoch,
                    'model': model_state,
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(ckpt, os.path.join(args.save_dir, f'epoch_{epoch}.pth'))
                wandb.save(os.path.join(args.save_dir, f'epoch_{epoch}.pth'))

            if train_loss < best_loss:
                best_loss = train_loss
                torch.save({'model': model_state},
                           os.path.join(args.save_dir, 'best.pth'))

        if dist.is_initialized():
            dist.barrier()

    if is_main_process():
        log(f'Training complete. Best loss: {best_loss:.4f}')
        wandb.finish()
    cleanup_distributed()


if __name__ == '__main__':
    main()

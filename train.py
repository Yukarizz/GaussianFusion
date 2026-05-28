"""
Training script for GaussianFusion model.

Multi-modal temporal fusion with 2D Gaussian splatting.
Supports arbitrary spatiotemporal super-resolution.
"""

import argparse
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

import models
import datasets
from utils import Averager, Timer, time_text, set_log_path, log


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


def ssim_loss(pred, target, window_size=11, channel=3):
    """Compute 1 - SSIM as a loss (differentiable)."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Create Gaussian window
    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = g.unsqueeze(1) @ g.unsqueeze(0)
    window = window.unsqueeze(0).unsqueeze(0).expand(channel, 1, -1, -1).contiguous()
    window = window.to(pred.device)

    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size // 2, groups=channel) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return 1 - ssim_map.mean()


class FusionLoss(nn.Module):
    """
    Combined loss for multi-modal temporal fusion.

    L = λ₁·L1(fused, vis_gt) + λ₁·L1(fused, ir_gt)
      + λ₂·SSIM(fused, vis_gt) + λ₂·SSIM(fused, ir_gt)
      + λ₃·MSE(fused, max(vis_gt, ir_gt))
    """

    def __init__(self, lambda_l1=1.0, lambda_ssim=0.5, lambda_max=0.5):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim
        self.lambda_max = lambda_max
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()

    def forward(self, fused, vis_gt, ir_gt):
        """
        Args:
            fused: Model output [B, 3, H, W].
            vis_gt: Visible GT [B, 3, H, W].
            ir_gt: Infrared GT [B, 3, H, W].
        """
        # L1 temporal accuracy
        loss_l1 = self.l1(fused, vis_gt) + self.l1(fused, ir_gt)

        # SSIM structural preservation
        loss_ssim = ssim_loss(fused, vis_gt) + ssim_loss(fused, ir_gt)

        # Max-intensity preservation (thermal radiation)
        max_gt = torch.max(vis_gt, ir_gt)
        loss_max = self.mse(fused, max_gt)

        total = self.lambda_l1 * loss_l1 + self.lambda_ssim * loss_ssim + self.lambda_max * loss_max
        return total, {
            'loss_l1': loss_l1.item(),
            'loss_ssim': loss_ssim.item(),
            'loss_max': loss_max.item(),
            'loss_total': total.item(),
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
    loss_components = {'loss_l1': Averager(), 'loss_ssim': Averager(),
                       'loss_max': Averager()}

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

        # Forward with AMP
        with torch.amp.autocast('cuda', enabled=use_amp):
            fused = model(vis_0, ir_0, vis_N, ir_N, scale=scale, tau=tau[0])
            loss, loss_dict = criterion(fused, vis_gt, ir_gt)

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
        for k in loss_components:
            loss_components[k].add(loss_dict[k])

        # wandb per-step logging
        global_step += 1
        if global_step % 50 == 0 and is_main_process():
            wandb.log({
                'train/loss_total': loss_dict['loss_total'],
                'train/loss_l1': loss_dict['loss_l1'],
                'train/loss_ssim': loss_dict['loss_ssim'],
                'train/loss_max': loss_dict['loss_max'],
                'train/lr': optimizer.param_groups[0]['lr'],
            }, step=global_step)

        pbar.set_postfix(loss=f'{loss_avg.item():.4f}')

    # Epoch-level averages
    if is_main_process():
        epoch_metrics = {
            'epoch/train_loss': loss_avg.item(),
            'epoch/loss_l1': loss_components['loss_l1'].item(),
            'epoch/loss_ssim': loss_components['loss_ssim'].item(),
            'epoch/loss_max': loss_components['loss_max'].item(),
            'epoch/lr': optimizer.param_groups[0]['lr'],
        }
        wandb.log(epoch_metrics, step=global_step)

    return loss_avg.item(), global_step


@torch.no_grad()
def validate(model, loader, criterion, device, global_step, log_images=False):
    """Validation step with wandb logging."""
    model.eval()
    loss_avg = Averager()
    sample_logged = False

    for batch in loader:
        vis_0 = batch['vis_anchor0'].to(device)
        ir_0 = batch['ir_anchor0'].to(device)
        vis_N = batch['vis_anchor1'].to(device)
        ir_N = batch['ir_anchor1'].to(device)
        vis_gt = batch['vis_gt'].to(device)
        ir_gt = batch['ir_gt'].to(device)
        tau = batch['tau'].to(device)

        fused = model(vis_0, ir_0, vis_N, ir_N, scale=(1.0, 1.0), tau=tau[0])
        loss, _ = criterion(fused, vis_gt, ir_gt)
        loss_avg.add(loss.item())

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
    wandb.log({'val/loss': val_loss}, step=global_step)
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
        lambda_l1=config.get('lambda_l1', 1.0),
        lambda_ssim=config.get('lambda_ssim', 0.5),
        lambda_max=config.get('lambda_max', 0.5),
    )

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
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    for epoch in range(start_epoch, total_epochs + 1):
        timer.s()
        train_loss, global_step = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch, global_step, scaler)
        scheduler.step()

        elapsed = timer.t()
        if is_main_process():
            log(f'Epoch {epoch}/{total_epochs} | Loss: {train_loss:.4f} | '
                f'LR: {scheduler.get_last_lr()[0]:.2e} | Time: {time_text(elapsed)}')

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

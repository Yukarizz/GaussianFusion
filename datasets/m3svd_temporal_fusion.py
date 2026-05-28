"""
M3SVD Temporal Fusion Dataset.

Large-interval anchor frame sampling for continuous temporal super-resolution
combined with multi-modal (visible + infrared) image fusion.

Sampling strategy:
    1. Pick a random video V
    2. Pick random interval N ∈ [N_min, N_max]
    3. Pick random start frame i ∈ [1, len(V) - N]
    4. Anchor frames: frame[i], frame[i+N]  → model input
    5. Pick random intermediate frame k ∈ [1, N-1]
    6. GT frame: frame[i+k]                 → supervision
    7. Temporal position: τ = k / N ∈ (0, 1)
"""

import os
import random
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF

from datasets import register


@register('m3svd-temporal-fusion')
class M3SVDTemporalFusion(Dataset):
    """
    M3SVD dataset for temporal fusion training.

    Returns anchor frame pairs (large interval) + intermediate GT frame,
    with random spatial cropping and scale.

    Args:
        root_path: Path to M3SVD root directory.
        split: 'train' or 'test'.
        vis_modality: Visible modality folder name.
        ir_modality: Infrared modality folder name.
        n_min: Minimum frame interval.
        n_max: Maximum frame interval.
        scale_min: Minimum spatial scale.
        scale_max: Maximum spatial scale.
        patch_size: LR patch size for training crops.
        augment: Whether to apply data augmentation.
    """

    def __init__(self, root_path, split='train',
                 vis_modality='visible_Enhance', ir_modality='infrared_Enhance',
                 n_min=2, n_max=16, scale_min=1.0, scale_max=4.0,
                 patch_size=64, augment=True):
        self.root = Path(root_path) / split
        self.vis_modality = vis_modality
        self.ir_modality = ir_modality
        self.n_min = n_min
        self.n_max = n_max
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.patch_size = patch_size
        self.augment = augment
        self.to_tensor = transforms.ToTensor()

        # Build index: list of (video_id, num_frames)
        self.videos = []
        vis_dir = self.root / vis_modality
        if not vis_dir.exists():
            raise FileNotFoundError(f"Visible modality directory not found: {vis_dir}")

        for vid in sorted(os.listdir(vis_dir)):
            vid_path_vis = vis_dir / vid
            vid_path_ir = self.root / ir_modality / vid
            if not vid_path_vis.is_dir() or not vid_path_ir.is_dir():
                continue
            frames = sorted(os.listdir(vid_path_vis))
            n_frames = len(frames)
            if n_frames > self.n_max:  # Need at least n_max+1 frames
                self.videos.append((vid, n_frames))

        if len(self.videos) == 0:
            raise RuntimeError(f"No valid videos found in {vis_dir}")

        # For deterministic length: each video contributes (n_frames - n_max) samples
        self.cumulative_lengths = []
        total = 0
        for vid, n_frames in self.videos:
            count = n_frames - self.n_max
            total += count
            self.cumulative_lengths.append(total)
        self.total_samples = total

    def __len__(self):
        return self.total_samples

    def _frame_path(self, modality, video_id, frame_idx):
        """Get frame file path (1-indexed)."""
        return self.root / modality / video_id / f"{frame_idx:06d}.png"

    def _load_frame(self, modality, video_id, frame_idx):
        """Load a single frame as tensor [3, H, W]."""
        path = self._frame_path(modality, video_id, frame_idx)
        img = Image.open(path).convert('RGB')
        return self.to_tensor(img)

    def _random_crop(self, *tensors, patch_size):
        """Apply the same random crop to all tensors."""
        _, h, w = tensors[0].shape
        if h < patch_size or w < patch_size:
            # Resize if too small
            scale = max(patch_size / h, patch_size / w) + 0.01
            new_h, new_w = int(h * scale), int(w * scale)
            tensors = [TF.resize(t, [new_h, new_w], antialias=True) for t in tensors]
            _, h, w = tensors[0].shape

        top = random.randint(0, h - patch_size)
        left = random.randint(0, w - patch_size)
        return [t[:, top:top + patch_size, left:left + patch_size] for t in tensors]

    def _augment(self, *tensors):
        """Random horizontal/vertical flip and 90° rotation."""
        if random.random() > 0.5:
            tensors = [TF.hflip(t) for t in tensors]
        if random.random() > 0.5:
            tensors = [TF.vflip(t) for t in tensors]
        if random.random() > 0.5:
            tensors = [torch.rot90(t, 1, [1, 2]) for t in tensors]
        return tensors

    def __getitem__(self, idx):
        # Find which video this index belongs to
        video_idx = 0
        for i, cum_len in enumerate(self.cumulative_lengths):
            if idx < cum_len:
                video_idx = i
                break

        video_id, n_frames = self.videos[video_idx]

        # Random interval N
        N = random.randint(self.n_min, self.n_max)

        # Random start frame (1-indexed)
        max_start = n_frames - N
        i = random.randint(1, max_start)

        # Random intermediate frame
        k = random.randint(1, N - 1)
        tau = k / N

        # Random spatial scale
        scale_h = random.uniform(self.scale_min, self.scale_max)
        scale_w = scale_h  # Uniform scaling (can be made non-uniform)

        # Load frames
        vis_0 = self._load_frame(self.vis_modality, video_id, i)
        ir_0 = self._load_frame(self.ir_modality, video_id, i)
        vis_N = self._load_frame(self.vis_modality, video_id, i + N)
        ir_N = self._load_frame(self.ir_modality, video_id, i + N)
        vis_gt = self._load_frame(self.vis_modality, video_id, i + k)
        ir_gt = self._load_frame(self.ir_modality, video_id, i + k)

        # Random crop at GT resolution (same crop for all frames)
        all_frames = self._random_crop(
            vis_0, ir_0, vis_N, ir_N, vis_gt, ir_gt,
            patch_size=self.patch_size
        )
        vis_0, ir_0, vis_N, ir_N, vis_gt, ir_gt = all_frames

        # Downsample anchor inputs by scale factor to create LR inputs
        # GT stays at patch_size; model must super-resolve from LR back to patch_size
        if scale_h > 1.0:
            lr_h = round(self.patch_size / scale_h)
            lr_w = round(self.patch_size / scale_w)
            vis_0 = TF.resize(vis_0, [lr_h, lr_w], antialias=True)
            ir_0 = TF.resize(ir_0, [lr_h, lr_w], antialias=True)
            vis_N = TF.resize(vis_N, [lr_h, lr_w], antialias=True)
            ir_N = TF.resize(ir_N, [lr_h, lr_w], antialias=True)

        # Augmentation
        if self.augment:
            vis_0, ir_0, vis_N, ir_N, vis_gt, ir_gt = self._augment(
                vis_0, ir_0, vis_N, ir_N, vis_gt, ir_gt
            )

        return {
            'vis_anchor0': vis_0,
            'ir_anchor0': ir_0,
            'vis_anchor1': vis_N,
            'ir_anchor1': ir_N,
            'vis_gt': vis_gt,
            'ir_gt': ir_gt,
            'tau': torch.tensor(tau, dtype=torch.float32),
            'scale': torch.tensor([scale_h, scale_w], dtype=torch.float32),
            'video_id': video_id,
            'frame_info': f"{i},{i+N},{i+k},{N}",
        }

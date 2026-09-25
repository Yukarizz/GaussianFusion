"""
M3SVD Temporal Fusion Dataset.

Large-interval anchor frame sampling for continuous temporal super-resolution
combined with multi-modal (visible + infrared) image fusion.

Non-overlapping window sampling:
    Each video is split into disjoint slots of at most n_max+1 frames. Each
    dataset index corresponds to one slot, so two samples in the same epoch
    can NEVER have overlapping windows (avoids near-duplicate inputs such as
    frames 1-9 vs frames 2-10, which would confuse the model).

Sampling strategy (per slot):
    1. Pick a random video V
    2. Pick random interval N ∈ [N_min, N_max]
    3. Anchor frames: frame[i], frame[i+N]  → model input (window inside slot)
    4. Pick random intermediate frame k ∈ [1, N-1]
    5. GT frame: frame[i+k]                 → supervision
    6. Temporal position: τ = k / N ∈ (0, 1)
"""

import os
import random
from pathlib import Path

from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF

from datasets import register


@register('m3svd-temporal-fusion')
class M3SVDTemporalFusion(Dataset):
    """
    M3SVD dataset for temporal fusion training.

    Returns anchor frame pairs (large interval) + intermediate GT frame,
    with configurable spatial cropping and scale.

    Spatial sampling (see `crop_strategy`):
        A uniformly random crop is a poor match for this dataset. On M3SVD the
        frames are 480x640 and only ~2.6% of pixels move more than 10px, so a
        224x224 window (16% of the scene) drawn at a uniform random location
        lands on static background most of the time. The 'motion' strategy
        biases the crop towards moving content, and 'resize' keeps the whole
        scene (at reduced motion magnitude) for scale diversity.

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
        crop_strategy: 'random' | 'motion' | 'resize' | 'hybrid'.
            random - uniformly random crop location (legacy behaviour).
            motion - crop location sampled proportionally to inter-frame
                motion energy, so patches cover the moving content.
            resize - aspect-preserving downscale so the short side equals
                patch_size, then crop the long side (keeps most of the scene).
            hybrid - mix of the three, see motion_crop_prob / resize_crop_prob.
        motion_crop_prob: In 'hybrid', probability of using 'motion'.
        resize_crop_prob: In 'hybrid', probability of using 'resize'.
            The remainder falls back to 'random'.
        motion_crop_gamma: Exponent on the motion energy weights. 0 degrades
            to uniform sampling; larger values are greedier (always the single
            most moving window in the limit).
        motion_crop_downscale: Factor by which frames are downsampled before
            computing the motion energy map. Downsampling suppresses texture
            noise and compression artefacts, which would otherwise dominate
            the raw frame difference.
    """

    def __init__(self, root_path, split='train',
                 vis_modality='visible_Enhance', ir_modality='infrared_Enhance',
                 n_min=2, n_max=16, scale_min=1.0, scale_max=4.0,
                 patch_size=64, augment=True,
                 crop_strategy='random',
                 motion_crop_prob=0.75, resize_crop_prob=0.15,
                 motion_crop_gamma=1.0, motion_crop_downscale=4):
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

        if crop_strategy not in ('random', 'motion', 'resize', 'hybrid'):
            raise ValueError(f"Unknown crop_strategy: {crop_strategy}")
        self.crop_strategy = crop_strategy
        self.motion_crop_prob = float(motion_crop_prob)
        self.resize_crop_prob = float(resize_crop_prob)
        self.motion_crop_gamma = float(motion_crop_gamma)
        self.motion_crop_downscale = max(1, int(motion_crop_downscale))

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

        # Non-overlapping window slots.
        # Each video is partitioned into disjoint windows of at most n_max+1 frames
        # (slot span = n_max gaps). A window sampled inside a slot never overlaps a
        # window from another slot, so within one epoch two samples can never be
        # near-duplicate (e.g. frames 1-9 AND frames 2-10), which would otherwise
        # let the model collapse onto near-identical inputs. Each slot yields
        # exactly ONE sample per epoch.
        self.locations = []  # list of (video_idx, slot_start) with 1-indexed frames
        for vi, (_vid, n_frames) in enumerate(self.videos):
            for start in range(1, n_frames - self.n_max + 1, self.n_max + 1):
                self.locations.append((vi, start))
        self.total_samples = len(self.locations)
        if self.total_samples == 0:
            raise RuntimeError(f"No non-overlapping windows available in {vis_dir}")

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

    def _ensure_min_size(self, tensors, patch_size):
        """Upscale every tensor so that both sides are at least patch_size."""
        _, h, w = tensors[0].shape
        if h < patch_size or w < patch_size:
            scale = max(patch_size / h, patch_size / w) + 0.01
            new_h, new_w = int(h * scale), int(w * scale)
            tensors = [TF.resize(t, [new_h, new_w], antialias=True) for t in tensors]
        return tensors

    def _crop_at(self, tensors, patch_size, top, left):
        """Apply the same crop at a fixed location to all tensors."""
        _, h, w = tensors[0].shape
        top = max(0, min(top, h - patch_size))
        left = max(0, min(left, w - patch_size))
        return [t[:, top:top + patch_size, left:left + patch_size] for t in tensors]

    def _random_crop(self, *tensors, patch_size):
        """Apply the same uniformly random crop to all tensors."""
        tensors = self._ensure_min_size(list(tensors), patch_size)
        _, h, w = tensors[0].shape
        top = random.randint(0, h - patch_size)
        left = random.randint(0, w - patch_size)
        return self._crop_at(tensors, patch_size, top, left)

    def _motion_energy(self, vis_0, vis_N):
        """Low-resolution inter-frame motion energy map, shape [h, w].

        Frames are average-pooled before differencing so that pixel-level
        texture detail and compression noise cancel out, leaving only the
        structural displacement the crop should cover.
        """
        ds = self.motion_crop_downscale
        if ds > 1:
            low_0 = F.avg_pool2d(vis_0.unsqueeze(0), ds, ds).squeeze(0)
            low_N = F.avg_pool2d(vis_N.unsqueeze(0), ds, ds).squeeze(0)
        else:
            low_0, low_N = vis_0, vis_N
        return (low_0 - low_N).abs().mean(0)

    def _motion_crop_topleft(self, energy, patch_size):
        """Sample a crop location with weight proportional to window motion energy.

        Uses an integral image so every candidate window is scored in O(1).
        Returns (top, left) in full-resolution coordinates, or None when the
        energy map is degenerate — the caller then falls back to a random crop.
        """
        ds = self.motion_crop_downscale
        ps = max(1, patch_size // ds)
        h, w = energy.shape
        if h < ps or w < ps:
            return None

        integ = F.pad(energy.cumsum(0).cumsum(1), (1, 0, 1, 0))
        step = max(1, ps // 8)
        tops = torch.arange(0, h - ps + 1, step)
        lefts = torch.arange(0, w - ps + 1, step)
        t0, t1 = tops, tops + ps
        l0, l1 = lefts, lefts + ps
        sums = (integ[t1[:, None], l1[None, :]]
                - integ[t0[:, None], l1[None, :]]
                - integ[t1[:, None], l0[None, :]]
                + integ[t0[:, None], l0[None, :]])

        if self.motion_crop_gamma > 0:
            # Weighting by energy**gamma is numerically identical to
            # softmax(gamma * log energy). Doing it in log space and
            # subtracting the maximum keeps float32 from overflowing to inf
            # once gamma grows past ~10, which would otherwise crash
            # torch.multinomial. Normalising by the mean first also makes
            # gamma a pure contrast knob, independent of the energy scale.
            norm = sums.float().clamp(min=0)
            mean = norm.mean()
            if not torch.isfinite(mean) or mean <= 0:
                return None
            log_w = self.motion_crop_gamma * torch.log(norm / mean + 1e-6)
            log_w = log_w - log_w.max()
            weights = torch.exp(log_w)
        else:
            weights = torch.ones_like(sums, dtype=torch.float32)
        if not torch.isfinite(weights).any() or weights.sum() <= 0:
            return None

        idx = torch.multinomial(weights.flatten(), 1).item()
        ti, li = divmod(idx, len(lefts))
        return int(tops[ti]) * ds, int(lefts[li]) * ds

    def _resize_then_crop(self, tensors, patch_size):
        """Aspect-preserving downscale (short side -> patch_size), then crop the long side.

        Keeps most of the scene inside the patch rather than ~16% of it, at the
        cost of shrinking absolute motion magnitude by the same factor. Useful
        for scale diversity: the model still sees whole-scene layout.
        """
        _, h, w = tensors[0].shape
        scale = patch_size / min(h, w)
        new_h = max(patch_size, round(h * scale))
        new_w = max(patch_size, round(w * scale))
        tensors = [TF.resize(t, [new_h, new_w], antialias=True) for t in tensors]
        _, h, w = tensors[0].shape
        top = 0 if h <= patch_size else random.randint(0, h - patch_size)
        left = 0 if w <= patch_size else random.randint(0, w - patch_size)
        return self._crop_at(tensors, patch_size, top, left)

    def _pick_crop_strategy(self):
        """Resolve the 'hybrid' strategy into one of the concrete ones."""
        if self.crop_strategy != 'hybrid':
            return self.crop_strategy
        r = random.random()
        if r < self.motion_crop_prob:
            return 'motion'
        if r < self.motion_crop_prob + self.resize_crop_prob:
            return 'resize'
        return 'random'

    def _spatial_sample(self, frames):
        """Crop (or rescale) all frames consistently according to crop_strategy.

        `frames` must be ordered [vis_0, ir_0, vis_N, ir_N, vis_gt, ir_gt];
        indices 0 and 2 are the visible anchors used to build the motion
        energy map. The energy is computed *after* any size normalisation so
        the sampled coordinates always refer to the tensors being cropped.
        """
        ps = self.patch_size
        # Evaluation keeps the legacy random crop so that numbers stay
        # comparable with existing baselines.
        strategy = self._pick_crop_strategy() if self.augment else 'random'

        if strategy == 'resize':
            return self._resize_then_crop(list(frames), ps)

        tensors = self._ensure_min_size(list(frames), ps)

        top, left = None, None
        if strategy == 'motion':
            energy = self._motion_energy(tensors[0], tensors[2])
            coords = self._motion_crop_topleft(energy, ps)
            if coords is not None:
                top, left = coords

        if top is None:
            _, h, w = tensors[0].shape
            top = random.randint(0, h - ps)
            left = random.randint(0, w - ps)
        return self._crop_at(tensors, ps, top, left)

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
        # Deterministic slot -> (video, window), guaranteeing that windows from
        # different slots never overlap (non-overlapping epoch sampling).
        video_idx, slot_start = self.locations[idx]
        video_id, n_frames = self.videos[video_idx]

        # Random interval N within the slot (window span <= slot span).
        N = random.randint(self.n_min, self.n_max)

        # Random start frame (1-indexed) such that window [i, i+N] stays inside
        # [slot_start, slot_start + n_max].
        i = random.randint(slot_start, slot_start + self.n_max - N)

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

        # Spatial sampling at GT resolution (same window for all frames).
        # Biased towards moving content: a uniformly random 224x224 window on
        # 480x640 frames covers only ~16% of the scene, while fewer than 3% of
        # pixels move, so uniform cropping lands on static background most of
        # the time and the model rarely sees the large motion it is judged on.
        vis_0, ir_0, vis_N, ir_N, vis_gt, ir_gt = self._spatial_sample(
            [vis_0, ir_0, vis_N, ir_N, vis_gt, ir_gt]
        )

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

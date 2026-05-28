import os
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from datasets import register


@register('m3svd-video-paired')
class M3SVDVideoPaired(Dataset):
    """M3SVD multi-modal video dataset.

    Returns paired frames (current + previous) from visible and infrared modalities.

    Directory structure:
        root_path/{split}/{modality}/{video_id}/{frame_idx:06d}.png

    Each sample returns:
        - vis_curr:  visible modality, current frame (t)
        - vis_prev:  visible modality, previous frame (t-1)
        - ir_curr:   infrared modality, current frame (t)
        - ir_prev:   infrared modality, previous frame (t-1)
    """

    def __init__(self, root_path, split='train',
                 vis_degraded='visible_Blur', vis_clean='visible_Enhance',
                 ir_degraded='infrared_noise', ir_clean='infrared_Enhance',
                 return_clean=True):
        """
        Args:
            root_path: path to M3SVD root (containing train/ and test/).
            split: 'train' or 'test'.
            vis_degraded: degraded visible modality folder name.
            vis_clean: clean visible modality folder name.
            ir_degraded: degraded infrared modality folder name.
            ir_clean: clean infrared modality folder name.
            return_clean: if True, also return the clean (Enhance) counterparts.
        """
        self.root = Path(root_path) / split
        self.vis_degraded = vis_degraded
        self.vis_clean = vis_clean
        self.ir_degraded = ir_degraded
        self.ir_clean = ir_clean
        self.return_clean = return_clean
        self.to_tensor = transforms.ToTensor()

        # Collect all valid (video_id, frame_idx) pairs where frame_idx >= 2
        # so that frame_idx - 1 always exists.
        self.samples = []

        vis_deg_dir = self.root / vis_degraded
        video_ids = sorted(os.listdir(vis_deg_dir))

        for vid in video_ids:
            # Verify the video exists in all modality folders
            if not all((self.root / mod / vid).is_dir() for mod in
                       [vis_degraded, ir_degraded] +
                       ([vis_clean, ir_clean] if return_clean else [])):
                continue

            frames = sorted(os.listdir(vis_deg_dir / vid))
            # Skip the first frame since it has no previous frame
            for frame_name in frames[1:]:
                self.samples.append((vid, frame_name))

    def __len__(self):
        return len(self.samples)

    def _load_frame(self, modality, video_id, frame_name):
        path = self.root / modality / video_id / frame_name
        return self.to_tensor(Image.open(path).convert('RGB'))

    def _prev_frame_name(self, frame_name):
        """Get the previous frame filename (e.g., 000005.png -> 000004.png)."""
        stem, ext = os.path.splitext(frame_name)
        idx = int(stem)
        return f"{idx - 1:06d}{ext}"

    def __getitem__(self, idx):
        video_id, frame_name = self.samples[idx]
        prev_name = self._prev_frame_name(frame_name)

        vis_curr = self._load_frame(self.vis_degraded, video_id, frame_name)
        vis_prev = self._load_frame(self.vis_degraded, video_id, prev_name)
        ir_curr = self._load_frame(self.ir_degraded, video_id, frame_name)
        ir_prev = self._load_frame(self.ir_degraded, video_id, prev_name)

        sample = {
            'vis_curr': vis_curr,
            'vis_prev': vis_prev,
            'ir_curr': ir_curr,
            'ir_prev': ir_prev,
            'video_id': video_id,
            'frame_name': frame_name,
        }

        if self.return_clean:
            sample['vis_clean_curr'] = self._load_frame(self.vis_clean, video_id, frame_name)
            sample['vis_clean_prev'] = self._load_frame(self.vis_clean, video_id, prev_name)
            sample['ir_clean_curr'] = self._load_frame(self.ir_clean, video_id, frame_name)
            sample['ir_clean_prev'] = self._load_frame(self.ir_clean, video_id, prev_name)

        return sample

"""
SEA-RAFT optical flow wrapper.

Wraps the SEA-RAFT model (from VF-Bench / UniVF) behind the same interface as
SpyNet: forward(im1, im2) -> flow [B, 2, H, W], where flow[:,0]=dx, flow[:,1]=dy.

The RAFT module expects inputs in [0, 255] (it divides by 255 internally), so
this wrapper scales [0,1] inputs to [0,255] before calling RAFT.
"""

import json
import os
from pathlib import Path

import torch
import torch.nn as nn

from models import register
from models.raft.raft import RAFT


def _default_args():
    """Default SEA-RAFT (Spring-S) config as a simple namespace."""
    class Args:
        pass
    args = Args()
    args.name = 'spring-S'
    args.dataset = 'spring'
    args.use_var = True
    args.var_min = 0
    args.var_max = 10
    args.pretrain = 'resnet18'
    args.initial_dim = 64
    args.block_dims = [64, 128, 256]
    args.radius = 4
    args.dim = 128
    args.num_blocks = 2
    args.iters = 12
    args.image_size = [540, 960]
    args.scale = -1
    args.epsilon = 1e-8
    args.path = None
    args.device = 'cuda'
    return args


@register('searaft')
class SeaRAFT(nn.Module):
    """Frozen SEA-RAFT optical flow with a SpyNet-compatible interface."""

    def __init__(self, pretrained=None, iters=None, freeze=True):
        super().__init__()
        args = _default_args()
        if iters is not None:
            args.iters = iters
        self.raft = RAFT(args)
        # RAFT expects inputs in [0,255]; our pipeline uses [0,1].
        self.rgb_scale = 255.0

        if pretrained:
            self.load_weights(pretrained)
        if freeze:
            for p in self.raft.parameters():
                p.requires_grad = False

    def load_weights(self, path):
        """Load RAFT checkpoint (state_dict with or without 'model' key)."""
        path = str(Path(path).expanduser())
        if not os.path.exists(path):
            raise FileNotFoundError(f'SEA-RAFT weights not found: {path}')
        state = torch.load(path, map_location='cpu')
        if isinstance(state, dict) and 'model' in state:
            state = state['model']
        missing, unexpected = self.raft.load_state_dict(state, strict=False)
        if missing:
            print(f'[searaft] missing keys: {len(missing)}')
        if unexpected:
            print(f'[searaft] unexpected keys: {len(unexpected)}')
        print(f'[searaft] loaded weights from {path}')

    def forward(self, im1, im2):
        """
        Args:
            im1: [B, 3, H, W] in [0,1].
            im2: [B, 3, H, W] in [0,1].
        Returns:
            flow [B, 2, H, W] (pixel displacement).
        """
        x1 = (im1 * self.rgb_scale).contiguous()
        x2 = (im2 * self.rgb_scale).contiguous()
        with torch.no_grad():
            out = self.raft(x1, x2, test_mode=True)
        return out['final']


def load_args_from_json(json_path):
    with open(json_path, 'r') as f:
        config_dict = json.load(f)
    class Args:
        pass
    args = Args()
    args.__dict__.update(config_dict)
    return args

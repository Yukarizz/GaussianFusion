"""
SpyNet: Spatial Pyramid Network for Optical Flow.
Ported from https://github.com/sniklaus/pytorch-spynet (PyTorch reimplementation).

Reference:
    Ranjan, Anurag, and Michael J. Black.
    "Optical Flow Estimation Using a Spatial Pyramid Network."
    IEEE Conference on Computer Vision and Pattern Recognition, 2017.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import register


backwarp_tenGrid = {}


def backwarp(tenInput, tenFlow):
    """Backward warp an image using optical flow (grid_sample based)."""
    key = str(tenFlow.shape)
    if key not in backwarp_tenGrid:
        tenHor = torch.linspace(-1.0, 1.0, tenFlow.shape[3]).view(
            1, 1, 1, -1).repeat(1, 1, tenFlow.shape[2], 1)
        tenVer = torch.linspace(-1.0, 1.0, tenFlow.shape[2]).view(
            1, 1, -1, 1).repeat(1, 1, 1, tenFlow.shape[3])
        backwarp_tenGrid[key] = torch.cat([tenHor, tenVer], 1)

    tenFlow = torch.cat([
        tenFlow[:, 0:1, :, :] * (2.0 / (tenInput.shape[3] - 1.0)),
        tenFlow[:, 1:2, :, :] * (2.0 / (tenInput.shape[2] - 1.0))
    ], 1)

    grid = (backwarp_tenGrid[key].to(tenFlow.device) + tenFlow).permute(0, 2, 3, 1)
    return F.grid_sample(
        input=tenInput, grid=grid,
        mode='bilinear', padding_mode='border', align_corners=True
    )


class SpyNetBasic(nn.Module):
    """Basic module at each pyramid level."""

    def __init__(self):
        super().__init__()
        self.netBasic = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3),
        )

    def forward(self, tenInput):
        return self.netBasic(tenInput)


class SpyNetPreprocess(nn.Module):
    """Preprocess images: BGR flip + ImageNet normalization."""

    def __init__(self):
        super().__init__()
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std_inv', torch.tensor([1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.225]).view(1, 3, 1, 1))

    def forward(self, tenInput):
        # RGB to BGR flip, then normalize
        tenInput = tenInput.flip([1])
        tenInput = (tenInput - self.mean) * self.std_inv
        return tenInput


@register('spynet')
class SpyNet(nn.Module):
    """SpyNet optical flow estimation network.

    A 6-level spatial pyramid network. Each level estimates a residual flow
    on top of the upsampled flow from the previous (coarser) level.

    Args:
        pretrained: URL or path to pretrained weights. If 'sintel-final',
                    loads from the default URL.
    """

    PRETRAINED_URLS = {
        'sintel-final': 'http://content.sniklaus.com/github/pytorch-spynet/network-sintel-final.pytorch',
        'sintel-clean': 'http://content.sniklaus.com/github/pytorch-spynet/network-sintel-clean.pytorch',
        'chairs-final': 'http://content.sniklaus.com/github/pytorch-spynet/network-chairs-final.pytorch',
        'chairs-clean': 'http://content.sniklaus.com/github/pytorch-spynet/network-chairs-clean.pytorch',
        'kitti-final': 'http://content.sniklaus.com/github/pytorch-spynet/network-kitti-final.pytorch',
    }

    def __init__(self, pretrained='sintel-final', num_levels=6):
        super().__init__()
        self.num_levels = num_levels
        self.netPreprocess = SpyNetPreprocess()
        self.netBasic = nn.ModuleList([SpyNetBasic() for _ in range(num_levels)])

        if pretrained:
            self._load_pretrained(pretrained)

    def _load_pretrained(self, pretrained):
        """Load pretrained weights."""
        try:
            if pretrained in self.PRETRAINED_URLS:
                url = self.PRETRAINED_URLS[pretrained]
                state_dict = torch.hub.load_state_dict_from_url(
                    url=url, file_name='spynet-' + pretrained
                )
            else:
                state_dict = torch.load(pretrained, map_location='cpu')

            # Convert key format: 'module_basic.0.netBasic.0.weight' -> 'netBasic.0.netBasic.0.weight'
            mapped_state_dict = {}
            for key, value in state_dict.items():
                new_key = key.replace('module', 'net')
                mapped_state_dict[new_key] = value

            self.load_state_dict(mapped_state_dict, strict=False)
            print(f'SpyNet: loaded pretrained weights ({pretrained})')
        except Exception as e:
            print(f'SpyNet: failed to load pretrained weights ({pretrained}): {e}')
            print('SpyNet: using random initialization')

    def forward(self, tenOne, tenTwo):
        """
        Estimate optical flow from tenOne to tenTwo.

        Args:
            tenOne: First image [B, 3, H, W] in range [0, 1].
            tenTwo: Second image [B, 3, H, W] in range [0, 1].

        Returns:
            Optical flow [B, 2, H, W] in pixel units.
        """
        # Build image pyramids
        tenOne_pyr = [self.netPreprocess(tenOne)]
        tenTwo_pyr = [self.netPreprocess(tenTwo)]

        for _ in range(self.num_levels - 1):
            if tenOne_pyr[0].shape[2] > 32 or tenOne_pyr[0].shape[3] > 32:
                tenOne_pyr.insert(0, F.avg_pool2d(
                    input=tenOne_pyr[0], kernel_size=2, stride=2, count_include_pad=False))
                tenTwo_pyr.insert(0, F.avg_pool2d(
                    input=tenTwo_pyr[0], kernel_size=2, stride=2, count_include_pad=False))
            else:
                break

        # Initialize flow at the coarsest level
        tenFlow = tenOne_pyr[0].new_zeros([
            tenOne_pyr[0].shape[0], 2,
            int(math.floor(tenOne_pyr[0].shape[2] / 2.0)),
            int(math.floor(tenOne_pyr[0].shape[3] / 2.0))
        ])

        # Coarse to fine flow estimation
        for intLevel in range(len(tenOne_pyr)):
            tenUpsampled = F.interpolate(
                input=tenFlow, scale_factor=2, mode='bilinear', align_corners=True
            ) * 2.0

            if tenUpsampled.shape[2] != tenOne_pyr[intLevel].shape[2]:
                tenUpsampled = F.pad(
                    input=tenUpsampled, pad=[0, 0, 0, 1], mode='replicate')
            if tenUpsampled.shape[3] != tenOne_pyr[intLevel].shape[3]:
                tenUpsampled = F.pad(
                    input=tenUpsampled, pad=[0, 1, 0, 0], mode='replicate')

            tenFlow = self.netBasic[intLevel](torch.cat([
                tenOne_pyr[intLevel],
                backwarp(tenInput=tenTwo_pyr[intLevel], tenFlow=tenUpsampled),
                tenUpsampled
            ], 1)) + tenUpsampled

        return tenFlow

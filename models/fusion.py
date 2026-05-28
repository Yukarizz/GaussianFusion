"""
Cross-Modal Fusion Module.

Lightweight channel attention (SE-block style) fusion for combining
visible and infrared features.
"""

import torch
import torch.nn as nn

from models import register


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention block."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(x).unsqueeze(-1).unsqueeze(-1)
        return x * w


@register('cross-modal-fusion')
class CrossModalFusion(nn.Module):
    """
    Cross-modal feature fusion with SE channel attention.

    Takes concatenated vis + ir features [B, 2*C, H, W] and
    produces fused features [B, C, H, W].

    Args:
        in_channels: Number of channels per modality (default: 64).
    """

    def __init__(self, in_channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.se = SEBlock(in_channels, reduction=16)

    def forward(self, feat_vis, feat_ir):
        """
        Args:
            feat_vis: Visible features [B, C, H, W].
            feat_ir: Infrared features [B, C, H, W].

        Returns:
            Fused features [B, C, H, W].
        """
        x = torch.cat([feat_vis, feat_ir], dim=1)  # [B, 2C, H, W]
        x = self.relu(self.conv1(x))               # [B, C, H, W]
        x = self.conv2(x)                          # [B, C, H, W]
        x = self.se(x)                             # [B, C, H, W] with channel attention
        return x

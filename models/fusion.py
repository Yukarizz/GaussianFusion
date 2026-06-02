"""
Cross-Modal Fusion Module.

Lightweight Convolutional Block Attention Module (CBAM) fusion for combining
visible and infrared features, ensuring both channel-wise weighting and 
spatial-wise local saliency preservation.
"""

import torch
import torch.nn as nn

from models import register


class ChannelAttention(nn.Module):
    """CBAM: Channel Attention Module"""
    def __init__(self, in_planes, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 共享的 MLP (使用 1x1 卷积代替全连接层，避免 Flatten 操作)
        self.fc1 = nn.Conv2d(in_planes, in_planes // reduction, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_planes // reduction, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 分别经过 AvgPool 和 MaxPool，然后通过共享 MLP
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        
        # 结果相加后激活
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """CBAM: Spatial Attention Module"""
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 沿着通道维度进行 MaxPool 和 AvgPool
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # 将两个空间图拼接 [B, 2, H, W]
        x_cat = torch.cat([avg_out, max_out], dim=1)
        
        # 通过卷积降维为 1 个通道并激活
        out = self.conv1(x_cat)
        return self.sigmoid(out)


class CBAMBlock(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, channels, reduction=16, spatial_kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=spatial_kernel_size)

    def forward(self, x):
        # 顺序执行：先通道注意力，后空间注意力
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out


@register('cross-modal-fusion')
class CrossModalFusion(nn.Module):
    """
    Cross-modal feature fusion with CBAM (Channel + Spatial) attention.

    Takes concatenated vis + ir features [B, 2*C, H, W] and
    produces fused features [B, C, H, W], focusing on both global feature 
    importance and local spatial saliency.

    Args:
        in_channels: Number of channels per modality (default: 64).
    """

    def __init__(self, in_channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        
        # 替换为 CBAM Block
        self.cbam = CBAMBlock(in_channels, reduction=16, spatial_kernel_size=7)

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
        x = self.cbam(x)                           # [B, C, H, W] with CBAM attention
        return x
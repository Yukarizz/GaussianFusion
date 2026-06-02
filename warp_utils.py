"""
Optical flow warping utilities for temporal interpolation.

Provides:
    - flow_warp: warp features/images using optical flow
    - temporal_interpolate_flow: linearly interpolate bidirectional flows to arbitrary time
    - compute_occlusion_mask: forward-backward consistency based occlusion detection
    - temporal_blend: occlusion-aware bidirectional feature blending
"""

import torch
import torch.nn.functional as F


def flow_warp(x, flow, padding_mode='border', align_corners=True):
    """
    Warp an image/feature map using optical flow.

    Args:
        x: Input tensor [B, C, H, W].
        flow: Optical flow [B, 2, H, W] in pixel displacement units.
              flow[:, 0] = horizontal (x) displacement,
              flow[:, 1] = vertical (y) displacement.
        padding_mode: 'zeros', 'border', or 'reflection'.
        align_corners: align_corners for grid_sample.

    Returns:
        Warped tensor [B, C, H, W].
    """
    B, C, H, W = x.shape
    # Create base grid
    grid_y, grid_x = torch.meshgrid(
        torch.arange(0, H, dtype=x.dtype, device=x.device),
        torch.arange(0, W, dtype=x.dtype, device=x.device),
        indexing='ij'
    )
    grid = torch.stack((grid_x, grid_y), dim=0).unsqueeze(0).expand(B, -1, -1, -1)  # [B, 2, H, W]

    # Add flow to get sampling locations
    grid = grid + flow  # [B, 2, H, W]

    # Normalize grid to [-1, 1]
    grid_x = 2.0 * grid[:, 0, :, :] / max(W - 1, 1) - 1.0
    grid_y = 2.0 * grid[:, 1, :, :] / max(H - 1, 1) - 1.0
    grid_norm = torch.stack((grid_x, grid_y), dim=-1)  # [B, H, W, 2]

    return F.grid_sample(
        x, grid_norm, mode='bilinear',
        padding_mode=padding_mode, align_corners=align_corners
    )


def temporal_interpolate_flow(flow_01, flow_10, t):
    """
    Linearly interpolate bidirectional flows to get flow at time t.

    Args:
        flow_01: Flow from frame 0 to frame 1 [B, 2, H, W].
        flow_10: Flow from frame 1 to frame 0 [B, 2, H, W].
        t: Temporal position in (0, 1).

    Returns:
        flow_0t: Flow from frame 0 to time t [B, 2, H, W].
        flow_1t: Flow from frame 1 to time t [B, 2, H, W].
    """
    flow_0t = t * flow_01
    flow_1t = (1 - t) * flow_10
    return flow_0t, flow_1t


def compute_occlusion_mask(flow_01, flow_10, threshold=1.0):
    """
    Compute occlusion mask using forward-backward consistency check.

    Args:
        flow_01: Flow from frame 0 to frame 1 [B, 2, H, W].
        flow_10: Flow from frame 1 to frame 0 [B, 2, H, W].
        threshold: Consistency threshold in pixels.

    Returns:
        occ_mask: [B, 1, H, W]. 1 = non-occluded, 0 = occluded.
    """
    flow_10_warped = flow_warp(flow_10, flow_01)
    consistency = torch.norm(flow_01 + flow_10_warped, dim=1, keepdim=True)
    occ_mask = (consistency < threshold).float()
    return occ_mask


def temporal_blend(feat_0, feat_1, flow_01, flow_10, t, occ_threshold=1.0):
    """
    Occlusion-aware bidirectional temporal blending.

    Warps features from both anchor frames to time t, then blends them
    with occlusion-aware weighting.

    Args:
        feat_0: Features at frame 0 [B, C, H, W].
        feat_1: Features at frame 1 [B, C, H, W].
        flow_01: Flow from frame 0 to frame 1 [B, 2, H, W].
        flow_10: Flow from frame 1 to frame 0 [B, 2, H, W].
        t: Temporal position in (0, 1). Scalar or tensor [B] for per-sample tau.
        occ_threshold: Threshold for occlusion detection.

    Returns:
        Blended features at time t [B, C, H, W].
    """
    # Support per-sample t: reshape to [B, 1, 1, 1] for broadcasting
    if isinstance(t, torch.Tensor) and t.dim() >= 1:
        t_b = t.view(-1, 1, 1, 1).to(feat_0.device)
    else:
        t_b = float(t)

    # Interpolate flows to time t (use scalar for flow interpolation if possible)
    flow_0t = t_b * flow_01
    flow_1t = (1 - t_b) * flow_10

    # Warp features from both directions
    warped_0 = flow_warp(feat_0, flow_0t)   # frame 0 → time t
    warped_1 = flow_warp(feat_1, flow_1t)   # frame 1 → time t

    # Compute occlusion mask (from frame 0's perspective)
    flow_10_warped = flow_warp(flow_10, flow_0t)
    consistency = torch.norm(flow_01 + flow_10_warped, dim=1, keepdim=True)
    occ_0 = (consistency < occ_threshold).float()  # 1 = frame 0 visible

    # Weighting
    w0 = (1 - t_b) * occ_0
    w1 = t_b + (1 - t_b) * (1 - occ_0)  # boost frame 1 where frame 0 is occluded

    # Normalize and blend
    blended = (w0 * warped_0 + w1 * warped_1) / (w0 + w1 + 1e-8)
    return blended

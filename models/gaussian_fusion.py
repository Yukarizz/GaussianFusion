"""
GaussianFusion: Multi-Modal Temporal Fusion via 2D Gaussian Splatting.

Extends ContinuousSR's Gaussian splatting paradigm to multi-modal (visible + infrared)
video fusion at arbitrary spatial resolution and temporal position.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import product

import models
from models import register
from warp_utils import flow_warp, temporal_blend

from gsplat.project_gaussians_2d import project_gaussians_2d
from gsplat.rasterize_sum import rasterize_gaussians_sum


def get_coord(width, height):
    """Generate normalized coordinate grid in [-1, 1]."""
    x_coords = torch.arange(width)
    y_coords = torch.arange(height)
    x_grid, y_grid = torch.meshgrid(x_coords, y_coords, indexing='ij')
    x_grid = 2 * (x_grid / width) - 1
    y_grid = 2 * (y_grid / height) - 1
    coordinates = torch.stack((y_grid, x_grid), dim=-1).reshape(-1, 2)
    return coordinates


@register('gaussian-fusion')
class GaussianFusion(nn.Module):
    """
    Multi-modal temporal fusion model using 2D Gaussian splatting.

    Architecture:
        SpyNet (frozen) → bidirectional optical flow
        Dual-branch EDSR encoder (shared body) → per-modality features
        Temporal blending (occlusion-aware) → feature at time τ
        Cross-modal fusion (SE attention) → fused feature
        Gaussian predictor (color, covariance, offset) → 2D Gaussians
        Rasterization → HR fused output at (τ, scale)

    Args:
        encoder_spec: Spec for the shared EDSR encoder body.
        spynet_pretrained: Pretrained model name for SpyNet.
        n_feats: Number of feature channels (default: 64).
        freeze_spynet: Whether to freeze SpyNet weights.
        occ_threshold: Occlusion detection threshold.
    """

    def __init__(self, encoder_spec, spynet_pretrained='sintel-final',
                 n_feats=64, freeze_spynet=True, occ_threshold=1.0):
        super().__init__()

        self.n_feats = n_feats
        self.occ_threshold = occ_threshold
        self.BLOCK_H, self.BLOCK_W = 16, 16

        # --- SpyNet for optical flow ---
        self.spynet = models.make({
            'name': 'spynet',
            'args': {'pretrained': spynet_pretrained}
        })
        if freeze_spynet:
            for p in self.spynet.parameters():
                p.requires_grad = False

        # --- Dual-branch encoder ---
        # Separate input heads for visible (3ch) and infrared (3ch)
        self.head_vis = nn.Sequential(
            nn.Conv2d(3, n_feats, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.head_ir = nn.Sequential(
            nn.Conv2d(3, n_feats, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # Shared EDSR body (from encoder_spec)
        self.encoder_body = models.make(encoder_spec)

        # --- Temporal fusion ---
        # Merge warped temporal features [feat_τ_from_0, feat_τ_from_1] -> single feat
        # This is handled by temporal_blend (weighted average), no extra conv needed

        # --- Cross-modal fusion ---
        self.fusion = models.make({
            'name': 'cross-modal-fusion',
            'args': {'in_channels': n_feats}
        })

        # --- Gaussian prediction head (reused from ContinuousSR) ---
        self.ps = nn.PixelUnshuffle(2)  # 64ch -> 256ch, spatial /2
        self.conv1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

        # MLP for Gaussian dictionary vector projection
        mlp_vector_spec = {'name': 'mlp', 'args': {
            'in_dim': 3, 'out_dim': 512, 'hidden_list': [256, 512, 512, 512]
        }}
        self.mlp_vector = models.make(mlp_vector_spec)

        # MLP for color prediction
        mlp_color_spec = {'name': 'mlp', 'args': {
            'in_dim': 256, 'out_dim': 3, 'hidden_list': [512, 1024, 256, 128, 64]
        }}
        self.mlp_color = models.make(mlp_color_spec)

        # MLP for offset prediction
        mlp_offset_spec = {'name': 'mlp', 'args': {
            'in_dim': 256, 'out_dim': 2, 'hidden_list': [512, 1024, 256, 128, 64]
        }}
        self.mlp_offset = models.make(mlp_offset_spec)

        # Pre-defined Gaussian covariance dictionary (730 templates)
        cho1 = torch.tensor([0, 0.41, 0.62, 0.98, 1.13, 1.29, 1.64, 1.85, 2.36])
        cho2 = torch.tensor([-0.86, -0.36, -0.16, 0.19, 0.34, 0.49, 0.84, 1.04, 1.54])
        cho3 = torch.tensor([0, 0.33, 0.53, 0.88, 1.03, 1.18, 1.53, 1.73, 2.23])
        gau_dict = torch.tensor(list(product(cho1.tolist(), cho2.tolist(), cho3.tolist())))
        gau_dict = torch.cat((gau_dict, torch.zeros(1, 3)), dim=0)  # [730, 3]
        self.register_buffer('gau_dict', gau_dict)

        self.background = None  # Lazy init on correct device

    def encode(self, vis, ir):
        """
        Encode visible and infrared images through dual-branch encoder.

        Args:
            vis: Visible image [B, 3, H, W].
            ir: Infrared image [B, 3, H, W].

        Returns:
            feat_vis: Visible features [B, n_feats, H, W].
            feat_ir: Infrared features [B, n_feats, H, W].
        """
        # Separate heads
        f_vis = self.head_vis(vis)  # [B, 64, H, W]
        f_ir = self.head_ir(ir)    # [B, 64, H, W]

        # Shared body
        feat_vis = self.encoder_body(f_vis)  # [B, 64, H, W]
        feat_ir = self.encoder_body(f_ir)    # [B, 64, H, W]

        return feat_vis, feat_ir

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def _render_gaussians(self, feat, scale_h, scale_w, lr_h, lr_w):
        """
        Clean Gaussian rendering without PixelUnshuffle complexity.
        Uses feat directly: each spatial position = one Gaussian.
        """
        feat = feat.float()  # gsplat kernels require FP32
        bs, C, fh, fw = feat.shape
        H = round(lr_h * scale_h)  # Target HR height
        W = round(lr_w * scale_w)  # Target HR width
        n_points = fh * fw

        tile_bounds = (
            (W + self.BLOCK_W - 1) // self.BLOCK_W,
            (H + self.BLOCK_H - 1) // self.BLOCK_H,
            1,
        )

        # PixelUnshuffle to get richer per-point features
        feat_ps = self.ps(feat)  # [B, C*4, fh/2, fw/2] = [B, 256, fh/2, fw/2]
        ps_h, ps_w = feat_ps.shape[2], feat_ps.shape[3]
        n_gaussians = ps_h * ps_w

        # Reshape: [B, 256, ps_h, ps_w] -> [B*ps_h*ps_w, 256]
        feat_flat = feat_ps.permute(0, 2, 3, 1).reshape(bs * n_gaussians, 256)

        # Color (sigmoid to ensure [0,1] range)
        color_all = torch.sigmoid(self.mlp_color(feat_flat)).reshape(bs, n_gaussians, 3)

        # Covariance via dictionary
        para_feat = self.leaky_relu(feat_ps)
        para_conv = self.conv1(para_feat)  # [B, 512, ps_h, ps_w]
        para_flat = para_conv.permute(0, 2, 3, 1).reshape(bs * n_gaussians, 512)

        vector = self.mlp_vector(self.gau_dict)  # [730, 512]
        similarity = vector @ para_flat.t()  # [730, B*n_gaussians]
        weights = torch.softmax(similarity, dim=0)  # [730, B*n_gaussians]
        cov_all = (weights.t() @ self.gau_dict).reshape(bs, n_gaussians, 3)

        # Offset
        offset_all = torch.tanh(self.mlp_offset(feat_flat)).reshape(bs, n_gaussians, 2)

        # Rasterize per batch
        if self.background is None or self.background.device != feat.device:
            self.background = torch.zeros(3, device=feat.device)

        # Precompute coords and opacity (shared across batch)
        coords = get_coord(ps_h, ps_w).to(feat.device)  # [n_gaussians, 2] in [-1, 1]
        opacity = torch.ones(n_gaussians, 1, device=feat.device)

        pred = []
        for i in range(bs):
            color_i = color_all[i]    # [n_gaussians, 3]
            cov_i = cov_all[i]        # [n_gaussians, 3]
            offset_i = offset_all[i]  # [n_gaussians, 2]

            # Apply offset
            xyz_x = coords[:, 0:1] + 2 * offset_i[:, 0:1] / fw - 1 / W
            xyz_y = coords[:, 1:2] + 2 * offset_i[:, 1:2] / fh - 1 / H
            xyz = torch.cat((xyz_x, xyz_y), dim=1)

            # Scale covariance by upsampling factors
            weighted_cholesky = cov_i / 4
            weighted_cholesky[:, 0] *= scale_w
            weighted_cholesky[:, 1] *= scale_h
            weighted_cholesky[:, 2] *= scale_h

            # Project and rasterize
            xys, depths, radii, conics, num_tiles_hit = project_gaussians_2d(
                xyz, weighted_cholesky, H, W, tile_bounds
            )
            out_img = rasterize_gaussians_sum(
                xys, depths, radii, conics, num_tiles_hit,
                color_i, opacity, H, W,
                self.BLOCK_H, self.BLOCK_W,
                background=self.background, return_alpha=False
            )
            out_img = out_img.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
            pred.append(out_img)

        return torch.cat(pred, dim=0)  # [B, 3, H, W]

    def forward(self, vis_0, ir_0, vis_N, ir_N, scale, tau):
        """
        Forward pass: produce fused HR frame at time τ between anchor frames.

        Args:
            vis_0: Visible anchor frame 0 [B, 3, H, W].
            ir_0: Infrared anchor frame 0 [B, 3, H, W].
            vis_N: Visible anchor frame N [B, 3, H, W].
            ir_N: Infrared anchor frame N [B, 3, H, W].
            scale: Tuple (scale_h, scale_w) or Tensor.
            tau: Temporal position in (0, 1).

        Returns:
            Fused HR image at time τ [B, 3, H*s_h, W*s_w].
        """
        # Handle scale format
        if isinstance(scale, torch.Tensor):
            if scale.dim() == 2:
                scale_h = float(scale[0, 0])
                scale_w = float(scale[0, 1])
            else:
                scale_h = float(scale[0])
                scale_w = float(scale[0])
        elif isinstance(scale, (tuple, list)):
            scale_h, scale_w = float(scale[0]), float(scale[1])
        else:
            scale_h = scale_w = float(scale)

        if isinstance(tau, torch.Tensor):
            tau = float(tau)

        lr_h, lr_w = vis_0.shape[2], vis_0.shape[3]

        # --- Step 1: Optical flow estimation (on visible frames) ---
        with torch.no_grad() if not self.spynet.training else torch.enable_grad():
            flow_01 = self.spynet(vis_0, vis_N)  # frame 0 → frame N
            flow_10 = self.spynet(vis_N, vis_0)  # frame N → frame 0

        # --- Step 2: Encode both modalities at both time steps ---
        feat_vis_0, feat_ir_0 = self.encode(vis_0, ir_0)
        feat_vis_N, feat_ir_N = self.encode(vis_N, ir_N)

        # --- Step 3: Adapt flow to feature resolution ---
        # Features are same resolution as input (encoder preserves spatial dims)
        # If feature resolution differs from flow resolution, resize flow
        if feat_vis_0.shape[2:] != flow_01.shape[2:]:
            fh, fw = feat_vis_0.shape[2], feat_vis_0.shape[3]
            scale_x = fw / flow_01.shape[3]
            scale_y = fh / flow_01.shape[2]
            flow_01 = F.interpolate(flow_01, size=(fh, fw), mode='bilinear', align_corners=True)
            flow_01[:, 0] *= scale_x
            flow_01[:, 1] *= scale_y
            flow_10 = F.interpolate(flow_10, size=(fh, fw), mode='bilinear', align_corners=True)
            flow_10[:, 0] *= scale_x
            flow_10[:, 1] *= scale_y

        # --- Step 4: Temporal blending to time τ ---
        feat_vis_tau = temporal_blend(
            feat_vis_0, feat_vis_N, flow_01, flow_10, tau,
            occ_threshold=self.occ_threshold
        )
        feat_ir_tau = temporal_blend(
            feat_ir_0, feat_ir_N, flow_01, flow_10, tau,
            occ_threshold=self.occ_threshold
        )

        # --- Step 5: Cross-modal fusion ---
        feat_fused = self.fusion(feat_vis_tau, feat_ir_tau)  # [B, 64, H, W]

        # --- Step 6: Gaussian rendering at target scale ---
        output = self._render_gaussians(feat_fused, scale_h, scale_w, lr_h, lr_w)

        return output

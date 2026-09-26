"""
Temporal Cross-Attention Module.

Replaces fixed linear temporal_blend with a learnable attention mechanism
that uses τ as an explicit condition to interpolate between anchor features.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import register
from warp_utils import flow_warp


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for scalar τ ∈ (0,1)."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """
        Args:
            t: [B] or scalar, temporal position in (0,1).
        Returns:
            [B, dim] positional embedding.
        """
        if not isinstance(t, torch.Tensor):
            t = torch.tensor([t], dtype=torch.float32)
        if t.dim() == 0:
            t = t.unsqueeze(0)

        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
        emb = t.unsqueeze(1) * emb.unsqueeze(0)  # [B, half_dim]
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)  # [B, dim]
        return emb


@register('temporal-cross-attention')
class TemporalCrossAttention(nn.Module):
    """
    Paper-aligned Continuous Gaussian Motion (CGM) feature interpolation.

    1. Scale bidirectional flow according to tau.
    2. Warp endpoint features toward target time tau.
    3. Fuse two warped features with a learnable mask + residual.

    Optical flow is ONLY used to build the intermediate feature F_tau; it never
    directly displaces final Gaussian centers (see _render_motion_gaussians).
    """

    def __init__(self, n_feats=64, tau_dim=64, n_heads=4, use_occlusion_fusion=False,
                 occ_gamma=1.0, occ_gamma_flow_ref=10.0):
        super().__init__()
        self.n_feats = n_feats
        self.last_reg_loss = None
        self._debug_stats = {}
        # Inference-side switch: replace the learned (easily-collapsing) soft mask
        # with a forward-backward-consistency occlusion-aware fusion. This stops
        # the "two half-transparent objects" ghosting at intermediate times.
        self.use_occlusion_fusion = use_occlusion_fusion
        self.occ_gamma = occ_gamma  # steepness of the occlusion soft mask
        # Flow-magnitude reference for adaptive gamma. A fixed gamma over-penalizes
        # large-displacement motion (fb_err grows with flow magnitude), making the
        # mask collapse onto a single endpoint even when both warps are valid.
        self.occ_gamma_flow_ref = occ_gamma_flow_ref

        # Warped endpoint features (2*C) + tau map (1) -> mask (1) + residual (C)
        self.fusion_head = nn.Sequential(
            nn.Conv2d(n_feats * 2 + 1, n_feats, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(n_feats, n_feats + 1, 3, padding=1),
        )

    def forward(self, feat_0, feat_1, flow_01, flow_10, tau, occ_threshold=5.0,
                flow_res=None):
        """
        Args:
            feat_0: Features at frame 0 [B, C, H, W].
            feat_1: Features at frame N [B, C, H, W].
            flow_01: Flow from frame 0 to frame N [B, 2, H, W].
            flow_10: Flow from frame N to frame 0 [B, 2, H, W].
            tau: Temporal position in (0,1). Scalar or Tensor [B].
            occ_threshold: Unused (kept for API compatibility).
            flow_res: Optional learned residual [B, 4, H, W] correcting the
                linear-tau displacement: channels 0:2 add to the tau -> frame 0
                displacement and 2:4 to the tau -> frame N one. None keeps the
                pure constant-velocity behaviour.

        Returns:
            Interpolated features at time tau [B, C, H, W].
        """
        B, C, H, W = feat_0.shape

        # Normalize tau to a per-sample tensor [B]
        if not isinstance(tau, torch.Tensor):
            tau = torch.full((B,), float(tau), device=feat_0.device, dtype=feat_0.dtype)
        elif tau.dim() == 0:
            tau = tau.reshape(1).expand(B)
        tau = tau.to(device=feat_0.device, dtype=feat_0.dtype)

        t = tau.view(B, 1, 1, 1)

        # ======================================================
        # 1. tau-conditioned flow scaling
        # ======================================================
        # flow_warp(x, f) samples x at (p + f(p)), i.e. f must be the displacement
        # from the OUTPUT coordinate back to the INPUT coordinate (backward flow).
        # flow_01 / flow_10 are FORWARD flows (source -> target), so they must be
        # negated before warping an endpoint feature to the intermediate time tau.
        flow_0t = -t * flow_01          # tau -> frame 0
        flow_1t = -(1.0 - t) * flow_10  # tau -> frame N

        # Learned deviation from the constant-velocity assumption. Signed so
        # that flow_res[:, 0:2] is an additive correction to the magnitude of
        # the tau -> frame 0 displacement.
        if flow_res is not None:
            if flow_res.shape[-2:] != flow_01.shape[-2:]:
                flow_res = F.interpolate(flow_res, size=flow_01.shape[-2:],
                                         mode='bilinear', align_corners=True)
            flow_0t = flow_0t - flow_res[:, 0:2]
            flow_1t = flow_1t - flow_res[:, 2:4]

        # ======================================================
        # 2. Backward warping of endpoint features to time tau
        # ======================================================
        warped_0 = flow_warp(feat_0, flow_0t)
        warped_1 = flow_warp(feat_1, flow_1t)

        # ======================================================
        # 3. Feature fusion (mask + residual)
        # ======================================================
        tau_map = t.expand(B, 1, H, W)

        if getattr(self, 'use_occlusion_fusion', False):
            # ---- Occlusion-aware fusion (no learned mask) ----
            # Forward-backward consistency: a pixel at tau is "reliable from
            # endpoint 0" if its forward flow agrees with the backward flow.
            # Where they disagree (occlusion / disocclusion), the other endpoint
            # is trusted instead, avoiding the 0.5-blend ghosting.
            # NOTE: flow_10 must be resampled to tau using the same (negated)
            # backward displacement that built warped_0.
            flow_10_at_0 = flow_warp(flow_10, flow_0t)       # bwd flow resampled to tau
            fb_err = (flow_01 + flow_10_at_0).norm(dim=1, keepdim=True)  # [B,1,H,W]
            # confidence of endpoint-0 content at this tau position
            # Adaptive gamma: normalize fb_err by the local flow magnitude so the
            # confidence does not over-penalize large (but consistent) motion.
            with torch.no_grad():
                flow_ref = (flow_01.norm(dim=1, keepdim=True)
                            + flow_10.norm(dim=1, keepdim=True)).clamp_min(1e-6) * 0.5
                gamma = self.occ_gamma * self.occ_gamma_flow_ref / (
                    self.occ_gamma_flow_ref + flow_ref)
            c0 = torch.exp(-gamma * fb_err).clamp(0, 1)   # high where consistent
            c1 = 1.0 - c0
            denom = c0 + c1 + 1e-8
            w0 = c0 / denom
            w1 = c1 / denom
            feat_tau = w0 * warped_0 + w1 * warped_1

            # Still apply the residual head (trained) for texture refinement.
            fusion_input = torch.cat([warped_0, warped_1, tau_map], dim=1)
            pred = self.fusion_head(fusion_input)
            residual = pred[:, 1:]
            feat_tau = feat_tau + residual
            mask = w0
        else:
            # ---- Original learned-mask path ----
            fusion_input = torch.cat([warped_0, warped_1, tau_map], dim=1)
            pred = self.fusion_head(fusion_input)
            mask = torch.sigmoid(pred[:, 0:1])
            residual = pred[:, 1:]
            feat_tau = mask * warped_0 + (1.0 - mask) * warped_1 + residual

        self.last_reg_loss = None
        with torch.no_grad():
            self._debug_stats = {
                'mask_mean': mask.mean().item(),
                'residual_abs_mean': residual.abs().mean().item(),
                'tau_mean': tau.mean().item(),
            }

        return feat_tau

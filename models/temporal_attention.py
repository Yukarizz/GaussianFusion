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
    Learnable temporal interpolation with τ conditioning.

    Instead of fixed (1-τ)·warp(feat_0) + τ·warp(feat_N), this module:
    1. Warps features using optical flow (like before)
    2. Uses τ-conditioned attention to learn HOW to blend
    3. Adds a residual refinement convolution

    Args:
        n_feats: Feature channels (default: 64).
        tau_dim: Dimension of τ positional embedding (default: 64).
        n_heads: Number of attention heads (default: 4).
    """

    def __init__(self, n_feats=64, tau_dim=64, n_heads=4):
        super().__init__()
        self.n_feats = n_feats
        self.tau_dim = tau_dim
        self.last_reg_loss = None
        self._debug_stats = {}

        # τ positional encoding
        self.tau_embed = SinusoidalPosEmb(tau_dim)
        self.tau_mlp = nn.Sequential(
            nn.Linear(tau_dim, n_feats),
            nn.GELU(),
            nn.Linear(n_feats, n_feats),
        )

        # Attention: query from τ-modulated concat, key/value from warped features
        # Use lightweight spatial attention (1×1 conv to compute weights)
        self.q_proj = nn.Conv2d(n_feats, n_feats, 1)
        self.k0_proj = nn.Conv2d(n_feats, n_feats, 1)
        self.k1_proj = nn.Conv2d(n_feats, n_feats, 1)
        self.v0_proj = nn.Conv2d(n_feats, n_feats, 1)
        self.v1_proj = nn.Conv2d(n_feats, n_feats, 1)

        # Temporal blending weight prediction (per-pixel, per-channel)
        self.blend_conv = nn.Sequential(
            nn.Conv2d(n_feats * 3, n_feats, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(n_feats, 2, 1),  # 2 weights: w0, w1
        )

        # Refinement after blending
        self.refine = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
        )

    def forward(self, feat_0, feat_1, flow_01, flow_10, tau, occ_threshold=5.0):
        """
        Args:
            feat_0: Features at frame 0 [B, C, H, W].
            feat_1: Features at frame N [B, C, H, W].
            flow_01: Flow from frame 0 to frame N [B, 2, H, W].
            flow_10: Flow from frame N to frame 0 [B, 2, H, W].
            tau: Temporal position [B] or scalar.
            occ_threshold: Not used (kept for API compatibility).

        Returns:
            Interpolated features at time τ [B, C, H, W].
        """
        B, C, H, W = feat_0.shape

        # Handle tau format
        if isinstance(tau, torch.Tensor) and tau.dim() >= 1:
            t_b = tau.view(-1, 1, 1, 1).to(feat_0.device)
            tau_flat = tau.to(feat_0.device)
        else:
            t_b = float(tau)
            tau_flat = torch.tensor([tau], device=feat_0.device).expand(B)


        t = t_b
        one_minus_t = 1.0 - t
        temp = -t * one_minus_t

        # 1.1 Calculate basic Super-SloMo approximate flows
        # These flows are mathematically sound in magnitude, but spatially misaligned
        flow_t0_approx = temp * flow_01 + (t * t) * flow_10
        flow_t1_approx = (one_minus_t * one_minus_t) * flow_01 + temp * flow_10

        # 1.2 Warp the base flows to align them to the intermediate grid t
        # We use the approximate intermediate flows to fetch the true vectors from anchor frames
        flow_01_at_t = flow_warp(flow_01, flow_t0_approx)
        flow_10_at_t = flow_warp(flow_10, flow_t1_approx)

        # 1.3 Recalculate accurate intermediate flows using the spatially aligned base flows
        flow_t0_aligned = temp * flow_01_at_t + (t * t) * flow_10_at_t
        flow_t1_aligned = (one_minus_t * one_minus_t) * flow_01_at_t + temp * flow_10_at_t

        # 1.4 Finally, warp the image features using the perfectly aligned flows
        warped_0 = flow_warp(feat_0, flow_t0_aligned)  # [B, C, H, W]
        warped_1 = flow_warp(feat_1, flow_t1_aligned)  # [B, C, H, W]

        # Step 2: τ conditioning
        tau_emb = self.tau_embed(tau_flat)       # [B, tau_dim]
        tau_feat = self.tau_mlp(tau_emb)         # [B, n_feats]
        tau_spatial = tau_feat.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]

        # Step 3: Compute attention-based blending weights
        # Query: τ-modulated average of warped features
        query = (warped_0 + warped_1) / 2.0 + tau_spatial  # [B, C, H, W]
        query = self.q_proj(query)

        k0 = self.k0_proj(warped_0)
        k1 = self.k1_proj(warped_1)

        # Per-pixel channel-wise attention score
        attn_0 = (query * k0).sum(dim=1, keepdim=True) / math.sqrt(C)  # [B, 1, H, W]
        attn_1 = (query * k1).sum(dim=1, keepdim=True) / math.sqrt(C)  # [B, 1, H, W]

        # Also compute explicit blending weights from context
        ctx = torch.cat([warped_0, warped_1, query], dim=1)  # [B, 3C, H, W]
        blend_logits = self.blend_conv(ctx)  # [B, 2, H, W]

        # Combine attention + explicit blend (both contribute)
        w = torch.softmax(
            torch.stack([attn_0.squeeze(1) + blend_logits[:, 0],
                         attn_1.squeeze(1) + blend_logits[:, 1]], dim=1),
            dim=1
        )  # [B, 2, H, W]

        w0 = w[:, 0:1]  # [B, 1, H, W]
        w1 = w[:, 1:2]  # [B, 1, H, W]

        # Differentiable anti-collapse regularization.
        # The target is a soft temporal prior, not a hard constraint:
        # earlier τ should lean to frame 0, later τ should lean to frame N.
        tau_prior = tau_flat.view(B, 1, 1, 1).to(device=w.device, dtype=w.dtype)
        target_w = torch.cat([1.0 - tau_prior, tau_prior], dim=1).expand_as(w)
        prior_loss = F.smooth_l1_loss(w, target_w)

        # Keep attention from becoming one-hot too early, especially near mid-time.
        # Maximum 2-way entropy is log(2); required entropy is relaxed near boundaries.
        w_safe = w.clamp_min(1e-6)
        entropy = -(w_safe * w_safe.log()).sum(dim=1)  # [B, H, W]
        midness = (1.0 - (2.0 * tau_flat - 1.0).abs()).view(B, 1, 1).to(entropy.dtype)
        min_entropy = 0.35 * midness
        entropy_loss = F.relu(min_entropy - entropy).mean()
        self.last_reg_loss = prior_loss + 0.1 * entropy_loss

        with torch.no_grad():
            max_w = w.max(dim=1).values
            self._debug_stats = {
                'w0_mean': w0.mean().item(),
                'w1_mean': w1.mean().item(),
                'w0_std': w0.std().item(),
                'w1_std': w1.std().item(),
                'target_w0_mean': (1.0 - tau_flat).mean().item(),
                'target_w1_mean': tau_flat.mean().item(),
                'entropy_mean': entropy.mean().item(),
                'dominance_frac': (max_w > 0.9).float().mean().item(),
                'prior_loss': prior_loss.detach().item(),
                'entropy_loss': entropy_loss.detach().item(),
            }

        # Step 4: Weighted blend with value projections
        v0 = self.v0_proj(warped_0)
        v1 = self.v1_proj(warped_1)
        blended = w0 * v0 + w1 * v1  # [B, C, H, W]

        # Step 5: Residual refinement
        out = blended + self.refine(blended)

        return out

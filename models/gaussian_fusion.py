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
from warp_utils import flow_warp
from models.temporal_attention import SinusoidalPosEmb

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
        Temporal cross-attention (τ-conditioned) → feature at time τ
        Cross-modal fusion (SE attention) → fused feature
        Gaussian predictor (τ-conditioned FiLM + color, covariance, offset) → 2D Gaussians
        Rasterization → HR fused output at (τ, scale)

    Args:
        encoder_spec: Spec for the shared EDSR encoder body.
        spynet_pretrained: Pretrained model name for SpyNet.
        n_feats: Number of feature channels (default: 64).
        freeze_spynet: Whether to freeze SpyNet weights.
        occ_threshold: Occlusion detection threshold.
    """

    def __init__(self, encoder_spec, spynet_pretrained='sintel-final',
                 n_feats=64, freeze_spynet=True, occ_threshold=1.0,
                 gaussian_render_mode='blend', fb_confidence_scale=None,
                 tau_gaussian_opacity=0.5):
        super().__init__()

        self.n_feats = n_feats
        self.occ_threshold = occ_threshold
        self.gaussian_render_mode = gaussian_render_mode
        self.fb_confidence_scale = fb_confidence_scale or occ_threshold
        self.tau_gaussian_opacity = tau_gaussian_opacity
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

        # --- Temporal cross-attention (learnable, τ-conditioned) ---
        self.temporal_attn = models.make({
            'name': 'temporal-cross-attention',
            'args': {'n_feats': n_feats, 'tau_dim': 64}
        })

        # --- Cross-modal fusion ---
        self.fusion = models.make({
            'name': 'cross-modal-fusion',
            'args': {'in_channels': n_feats}
        })

        # --- Auxiliary temporal reconstruction heads ---
        # These heads force per-modality temporal features to reconstruct the
        # intermediate visible / infrared targets before cross-modal fusion.
        self.aux_vis_decoder = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(n_feats, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        self.aux_ir_decoder = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(n_feats, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

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

        # --- τ conditioning for Gaussian head (FiLM modulation) ---
        self.tau_embed = SinusoidalPosEmb(64)
        self.tau_film = nn.Sequential(
            nn.Linear(64, 256),
            nn.GELU(),
            nn.Linear(256, 256 * 2),  # γ and β for FiLM on 256-ch features
        )
        self.cov_tau_mod = nn.Sequential(
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 6),  # γ and β for [L11, L21, L22]
        )

        # Pre-defined Gaussian covariance dictionary (730 templates)
        cho1 = torch.tensor([0, 0.41, 0.62, 0.98, 1.13, 1.29, 1.64, 1.85, 2.36])
        cho2 = torch.tensor([-0.86, -0.36, -0.16, 0.19, 0.34, 0.49, 0.84, 1.04, 1.54])
        cho3 = torch.tensor([0, 0.33, 0.53, 0.88, 1.03, 1.18, 1.53, 1.73, 2.23])
        gau_dict = torch.tensor(list(product(cho1.tolist(), cho2.tolist(), cho3.tolist())))
        gau_dict = torch.cat((gau_dict, torch.zeros(1, 3)), dim=0)  # [730, 3]
        self.register_buffer('gau_dict', gau_dict)

        self.background = None  # Lazy init on correct device
        self._temporal_reg_loss = None
        self._temporal_stats = {}
        self._aux_outputs = {}

    def _expand_tau(self, tau, bs, device):
        if tau is None:
            return None
        if not isinstance(tau, torch.Tensor):
            return torch.full((bs,), float(tau), device=device)
        if tau.dim() == 0:
            return tau.to(device).reshape(1).expand(bs)
        return tau.to(device).reshape(-1)

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

    def _make_gaussian_features(self, feat, scale_h, scale_w, tau=None):
        """Convert dense feature maps into per-Gaussian latent vectors."""
        feat = feat.float()
        bs, C, fh, fw = feat.shape

        # PixelUnshuffle to get richer per-point features.
        # For scale=1 this keeps the original compact point set. For SR rendering
        # (scale>1), the compact fh/2 × fw/2 grid becomes too sparse in the HR
        # canvas and produces periodic low-coverage gaps. Densify to one Gaussian
        # per LR pixel for scale>1 to avoid grid-like holes between samples.
        feat_ps = self.ps(feat)  # [B, C*4, fh/2, fw/2] = [B, 256, fh/2, fw/2]

        # τ FiLM modulation: condition Gaussian features on temporal position
        tau_t = self._expand_tau(tau, bs, feat.device)
        if tau_t is not None:
            tau_emb = self.tau_embed(tau_t.float())  # [B, 64]
            film_params = self.tau_film(tau_emb)     # [B, 512]
            gamma = film_params[:, :256].unsqueeze(-1).unsqueeze(-1) + 1.0  # [B, 256, 1, 1]
            beta = film_params[:, 256:].unsqueeze(-1).unsqueeze(-1)         # [B, 256, 1, 1]
            feat_ps = gamma * feat_ps + beta

        dense_sr_render = max(scale_h, scale_w) > 1.0
        if dense_sr_render:
            feat_ps = F.interpolate(feat_ps, size=(fh, fw), mode='nearest')

        return feat_ps, dense_sr_render

    def _predict_gaussian_params(self, feat_ps, tau=None):
        """Predict color, covariance, and local offset for each Gaussian."""
        bs = feat_ps.shape[0]
        ps_h, ps_w = feat_ps.shape[2], feat_ps.shape[3]
        n_gaussians = ps_h * ps_w

        # Reshape: [B, 256, ps_h, ps_w] -> [B*ps_h*ps_w, 256]
        feat_flat = feat_ps.permute(0, 2, 3, 1).reshape(bs * n_gaussians, 256)

        # Color (sigmoid with -2 shift: init at sigmoid(-2)≈0.12, below typical targets)
        # Forces gradient to push UP (away from sigmoid saturation at 0)
        color_all = torch.sigmoid(self.mlp_color(feat_flat) - 2.0).reshape(bs, n_gaussians, 3)

        # Covariance via dictionary (same mechanism as ContinuousSR)
        # Detach features to prevent encoder side-effects on covariance
        para_feat = self.leaky_relu(feat_ps.detach())
        para_conv = self.conv1(para_feat)  # [B, 512, ps_h, ps_w]
        para_flat = para_conv.permute(0, 2, 3, 1).reshape(bs * n_gaussians, 512)

        vector = self.mlp_vector(self.gau_dict)  # [730, 512]
        similarity = vector @ para_flat.t()  # [730, B*n_gaussians]
        weights = torch.softmax(similarity, dim=0)  # [730, B*n_gaussians]
        cov_all = (weights.t() @ self.gau_dict).reshape(bs, n_gaussians, 3)
        tau_t = self._expand_tau(tau, bs, feat_ps.device)
        if tau_t is not None:
            tau_emb = self.tau_embed(tau_t.float())
            cov_mod = self.cov_tau_mod(tau_emb)
            cov_gamma = 1.0 + 0.1 * torch.tanh(cov_mod[:, :3]).unsqueeze(1)
            cov_beta = 0.1 * torch.tanh(cov_mod[:, 3:]).unsqueeze(1)
            cov_all = cov_all * cov_gamma + cov_beta

        # Offset
        offset_all = torch.tanh(self.mlp_offset(feat_flat)).reshape(bs, n_gaussians, 2)

        return {
            'color': color_all,
            'cov': cov_all,
            'offset': offset_all,
            'ps_h': ps_h,
            'ps_w': ps_w,
            'n_gaussians': n_gaussians,
        }

    def _base_xyz(self, offset_all, ps_h, ps_w, H, W):
        bs = offset_all.shape[0]
        coords = get_coord(ps_h, ps_w).to(offset_all.device)
        coords = coords.unsqueeze(0).expand(bs, -1, -1)
        xyz_x = coords[:, :, 0:1] + 2 * offset_all[:, :, 0:1] / ps_w - 1 / W
        xyz_y = coords[:, :, 1:2] + 2 * offset_all[:, :, 1:2] / ps_h - 1 / H
        return torch.cat((xyz_x, xyz_y), dim=2)

    def _flow_to_gaussian_grid(self, flow, ps_h, ps_w):
        flow_ps = F.interpolate(flow, size=(ps_h, ps_w), mode='bilinear', align_corners=True)
        return flow_ps.permute(0, 2, 3, 1).reshape(flow.shape[0], ps_h * ps_w, 2)

    def _compute_fb_confidence(self, flow_01, flow_10):
        """Continuous forward-backward confidence maps for both endpoints."""
        flow_10_at_1 = flow_warp(flow_10, flow_01)
        flow_01_at_0 = flow_warp(flow_01, flow_10)
        cons_0 = torch.norm(flow_01 + flow_10_at_1, dim=1, keepdim=True)
        cons_1 = torch.norm(flow_10 + flow_01_at_0, dim=1, keepdim=True)
        sigma = max(float(self.fb_confidence_scale), 1e-6)
        conf_0 = torch.exp(-cons_0 / sigma).clamp(0.0, 1.0)
        conf_1 = torch.exp(-cons_1 / sigma).clamp(0.0, 1.0)
        return conf_0, conf_1, cons_0, cons_1

    def _confidence_to_gaussian_grid(self, confidence, ps_h, ps_w):
        conf_ps = F.interpolate(confidence, size=(ps_h, ps_w), mode='bilinear', align_corners=True)
        return conf_ps.permute(0, 2, 3, 1).reshape(confidence.shape[0], ps_h * ps_w, 1)

    def _apply_flow_motion(self, xyz, flow, tau_weight, ps_h, ps_w):
        flow_flat = self._flow_to_gaussian_grid(flow, ps_h, ps_w)
        if not isinstance(tau_weight, torch.Tensor):
            tau_weight = torch.tensor(tau_weight, device=xyz.device, dtype=xyz.dtype)
        tau_weight = tau_weight.to(device=xyz.device, dtype=xyz.dtype).view(-1, 1, 1)
        disp_x = 2 * tau_weight * flow_flat[:, :, 0:1] / ps_w
        disp_y = 2 * tau_weight * flow_flat[:, :, 1:2] / ps_h
        return xyz + torch.cat((disp_x, disp_y), dim=2)

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def _render_gaussian_params(self, color_all, cov_all, xyz_all, opacity_all,
                                scale_h, scale_w, H, W, dense_sr_render,
                                debug_tag='gaussian'):
        """Rasterize already-positioned Gaussian parameters."""
        bs, n_gaussians = color_all.shape[:2]

        # Rasterize per batch
        if self.background is None or self.background.device != color_all.device:
            self.background = torch.zeros(3, device=color_all.device)

        tile_bounds = (
            (W + self.BLOCK_W - 1) // self.BLOCK_W,
            (H + self.BLOCK_H - 1) // self.BLOCK_H,
            1,
        )

        pred = []
        for i in range(bs):
            color_i = color_all[i]    # [n_gaussians, 3]
            cov_i = cov_all[i]        # [n_gaussians, 3]
            xyz = xyz_all[i]          # [n_gaussians, 2]
            opacity = opacity_all[i]  # [n_gaussians, 1]

            # Scale covariance. In SR mode, a fixed 0.5px footprint is too small
            # for the enlarged canvas and can leave periodic uncovered pixels.
            min_std_w = max(0.5, 0.5 * scale_w) if dense_sr_render else 0.5
            min_std_h = max(0.5, 0.5 * scale_h) if dense_sr_render else 0.5
            L11 = cov_i[:, 0] * scale_w / 2.0 + min_std_w
            L21 = cov_i[:, 1] * scale_h / 2.0
            L22 = cov_i[:, 2] * scale_h / 2.0 + min_std_h
            weighted_cholesky = torch.stack([L11, L21, L22], dim=1)

            # Project and rasterize
            xys, depths, radii, conics, num_tiles_hit = project_gaussians_2d(
                xyz, weighted_cholesky, H, W, tile_bounds
            )
            # Rasterize color (numerator)
            out_rgb = rasterize_gaussians_sum(
                xys, depths, radii, conics, num_tiles_hit,
                color_i, opacity, H, W,
                self.BLOCK_H, self.BLOCK_W,
                background=self.background, return_alpha=False
            )
            # Rasterize weights (denominator) for normalization
            ones_color = torch.ones_like(color_i)
            out_w = rasterize_gaussians_sum(
                xys, depths, radii, conics, num_tiles_hit,
                ones_color, opacity, H, W,
                self.BLOCK_H, self.BLOCK_W,
                background=self.background, return_alpha=False
            )
            # Normalized output: clamp denominator to prevent explosion at
            # low-coverage pixels (borders, offset-shifted regions)
            out_img = out_rgb / out_w.clamp(min=1.0)
            out_img = out_img.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
            pred.append(out_img)

            # Store debug stats (first batch item only)
            if i == 0:
                self._debug_stats = {
                    'render_mode_motion': float(debug_tag == 'motion'),
                    'radii_min': radii.min().item(),
                    'radii_max': radii.max().item(),
                    'radii_zero_frac': (radii == 0).float().mean().item(),
                    'cholesky_L11_min': weighted_cholesky[:, 0].min().item(),
                    'cholesky_L11_max': weighted_cholesky[:, 0].max().item(),
                    'cholesky_L22_min': weighted_cholesky[:, 2].min().item(),
                    'cholesky_L22_max': weighted_cholesky[:, 2].max().item(),
                    'color_mean': color_i.mean().item(),
                    'opacity_mean': opacity.mean().item(),
                    'n_gaussians': n_gaussians,
                    'dense_sr_render': float(dense_sr_render),
                }

        return torch.cat(pred, dim=0)  # [B, 3, H, W]

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def _render_gaussians(self, feat, scale_h, scale_w, lr_h, lr_w, tau=None):
        """
        Original feature-blending Gaussian rendering path with optional τ FiLM.
        """
        H = round(lr_h * scale_h)
        W = round(lr_w * scale_w)
        feat_ps, dense_sr_render = self._make_gaussian_features(feat, scale_h, scale_w, tau=tau)
        params = self._predict_gaussian_params(feat_ps, tau=tau)
        xyz = self._base_xyz(params['offset'], params['ps_h'], params['ps_w'], H, W)
        opacity = torch.ones(
            params['color'].shape[0], params['n_gaussians'], 1,
            device=params['color'].device, dtype=params['color'].dtype
        )
        return self._render_gaussian_params(
            params['color'], params['cov'], xyz, opacity,
            scale_h, scale_w, H, W, dense_sr_render, debug_tag='blend'
        )

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def _render_motion_gaussians(self, feat_fused_0, feat_fused_N, flow_01, flow_10,
                                 scale_h, scale_w, lr_h, lr_w, tau,
                                 feat_fused_tau=None):
        """
        Motion-aware Gaussian rendering.

        Each endpoint predicts its own Gaussian primitives. Their centers are
        explicitly displaced by optical flow to time τ before both primitive
        sets are splatted onto the target canvas.
        """
        bs = feat_fused_0.shape[0]
        device = feat_fused_0.device
        H = round(lr_h * scale_h)
        W = round(lr_w * scale_w)
        tau_t = self._expand_tau(tau, bs, device)
        one_minus_tau = 1.0 - tau_t

        feat_ps_0, dense0 = self._make_gaussian_features(
            feat_fused_0, scale_h, scale_w, tau=tau_t
        )
        feat_ps_N, denseN = self._make_gaussian_features(
            feat_fused_N, scale_h, scale_w, tau=tau_t
        )
        params0 = self._predict_gaussian_params(feat_ps_0, tau=tau_t)
        paramsN = self._predict_gaussian_params(feat_ps_N, tau=tau_t)

        ps_h, ps_w = params0['ps_h'], params0['ps_w']
        xyz0 = self._base_xyz(params0['offset'], ps_h, ps_w, H, W)
        xyzN = self._base_xyz(paramsN['offset'], ps_h, ps_w, H, W)
        xyz0 = self._apply_flow_motion(xyz0, flow_01, tau_t, ps_h, ps_w)
        xyzN = self._apply_flow_motion(xyzN, flow_10, one_minus_tau, ps_h, ps_w)

        conf_0, conf_N, cons_0, cons_N = self._compute_fb_confidence(flow_01, flow_10)
        conf0_grid = self._confidence_to_gaussian_grid(conf_0, ps_h, ps_w)
        confN_grid = self._confidence_to_gaussian_grid(conf_N, ps_h, ps_w)
        opacity0 = one_minus_tau.view(bs, 1, 1).expand(-1, params0['n_gaussians'], 1) * conf0_grid
        opacityN = tau_t.view(bs, 1, 1).expand(-1, paramsN['n_gaussians'], 1) * confN_grid

        colors = [params0['color'], paramsN['color']]
        covs = [params0['cov'], paramsN['cov']]
        xyzs = [xyz0, xyzN]
        opacities = [opacity0, opacityN]
        dense_tau = False

        if feat_fused_tau is not None and self.tau_gaussian_opacity > 0:
            feat_ps_tau, dense_tau = self._make_gaussian_features(
                feat_fused_tau, scale_h, scale_w, tau=tau_t
            )
            params_tau = self._predict_gaussian_params(feat_ps_tau, tau=tau_t)
            xyz_tau = self._base_xyz(params_tau['offset'], params_tau['ps_h'], params_tau['ps_w'], H, W)
            midness = (1.0 - (2.0 * tau_t - 1.0).abs()).clamp(0.0, 1.0)
            opacity_tau = (
                self.tau_gaussian_opacity
                * midness.view(bs, 1, 1)
                * torch.ones(bs, params_tau['n_gaussians'], 1, device=device, dtype=params_tau['color'].dtype)
            )
            colors.append(params_tau['color'])
            covs.append(params_tau['cov'])
            xyzs.append(xyz_tau)
            opacities.append(opacity_tau)

        color = torch.cat(colors, dim=1)
        cov = torch.cat(covs, dim=1)
        xyz = torch.cat(xyzs, dim=1)
        opacity = torch.cat(opacities, dim=1).to(color.dtype)

        with torch.no_grad():
            flow01_grid = self._flow_to_gaussian_grid(flow_01, ps_h, ps_w)
            flow10_grid = self._flow_to_gaussian_grid(flow_10, ps_h, ps_w)
            self._temporal_stats['motion_flow01_abs_mean'] = flow01_grid.abs().mean().item()
            self._temporal_stats['motion_flow10_abs_mean'] = flow10_grid.abs().mean().item()
            self._temporal_stats['motion_tau_mean'] = tau_t.mean().item()
            self._temporal_stats['fb_conf0_mean'] = conf_0.mean().item()
            self._temporal_stats['fb_conf1_mean'] = conf_N.mean().item()
            self._temporal_stats['fb_cons0_mean'] = cons_0.mean().item()
            self._temporal_stats['fb_cons1_mean'] = cons_N.mean().item()
            self._temporal_stats['tau_gaussian_opacity_mean'] = (
                opacities[2].mean().item() if len(opacities) > 2 else 0.0
            )

        return self._render_gaussian_params(
            color, cov, xyz, opacity,
            scale_h, scale_w, H, W, dense0 or denseN or dense_tau, debug_tag='motion'
        )

    def forward(self, vis_0, ir_0, vis_N, ir_N, scale, tau):
        """
        Forward pass: produce fused HR frame at time τ between anchor frames.

        Args:
            vis_0: Visible anchor frame 0 [B, 3, H, W].
            ir_0: Infrared anchor frame 0 [B, 3, H, W].
            vis_N: Visible anchor frame N [B, 3, H, W].
            ir_N: Infrared anchor frame N [B, 3, H, W].
            scale: Tuple (scale_h, scale_w) or Tensor.
            tau: Temporal position in (0, 1). Scalar or Tensor [B].

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

        # Keep tau as tensor for per-sample temporal blending
        if not isinstance(tau, torch.Tensor):
            tau = torch.tensor(tau, dtype=torch.float32, device=vis_0.device)

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

        # --- Step 4: Temporal cross-attention to time τ (learnable) ---
        feat_vis_tau = self.temporal_attn(
            feat_vis_0, feat_vis_N, flow_01, flow_10, tau, self.occ_threshold
        )
        vis_reg_loss = self.temporal_attn.last_reg_loss
        vis_stats = dict(getattr(self.temporal_attn, '_debug_stats', {}))

        feat_ir_tau = self.temporal_attn(
            feat_ir_0, feat_ir_N, flow_01, flow_10, tau, self.occ_threshold
        )
        ir_reg_loss = self.temporal_attn.last_reg_loss
        ir_stats = dict(getattr(self.temporal_attn, '_debug_stats', {}))

        # Auxiliary modality reconstruction at intermediate time τ. These are
        # supervised during training and ignored by the final inference output.
        aux_vis_tau = self.aux_vis_decoder(feat_vis_tau)
        aux_ir_tau = self.aux_ir_decoder(feat_ir_tau)
        self._aux_outputs = {
            'vis_tau': aux_vis_tau,
            'ir_tau': aux_ir_tau,
        }

        if vis_reg_loss is not None and ir_reg_loss is not None:
            self._temporal_reg_loss = 0.5 * (vis_reg_loss + ir_reg_loss)
        else:
            self._temporal_reg_loss = None
        self._temporal_stats = {}
        for k, v in vis_stats.items():
            self._temporal_stats[f'temporal_vis_{k}'] = v
        for k, v in ir_stats.items():
            self._temporal_stats[f'temporal_ir_{k}'] = v
        with torch.no_grad():
            self._temporal_stats['aux_vis_mean'] = aux_vis_tau.mean().item()
            self._temporal_stats['aux_ir_mean'] = aux_ir_tau.mean().item()

        # --- Step 5/6: Cross-modal fusion and Gaussian rendering ---
        if self.gaussian_render_mode == 'motion':
            # Fuse each endpoint first, then move endpoint Gaussians along flow.
            feat_fused_0 = self.fusion(feat_vis_0, feat_ir_0)
            feat_fused_N = self.fusion(feat_vis_N, feat_ir_N)
            feat_fused_tau = self.fusion(feat_vis_tau, feat_ir_tau)
            output = self._render_motion_gaussians(
                feat_fused_0, feat_fused_N, flow_01, flow_10,
                scale_h, scale_w, lr_h, lr_w, tau=tau,
                feat_fused_tau=feat_fused_tau
            )
        else:
            # Backward-compatible path: interpolate features, fuse, then render.
            feat_fused = self.fusion(feat_vis_tau, feat_ir_tau)  # [B, 64, H, W]
            output = self._render_gaussians(feat_fused, scale_h, scale_w, lr_h, lr_w, tau=tau)
        if hasattr(self, '_debug_stats'):
            self._debug_stats.update(self._temporal_stats)

        return output

    def get_temporal_regularization(self):
        """Return the latest differentiable temporal anti-collapse loss."""
        return self._temporal_reg_loss

    def get_aux_outputs(self):
        """Return latest auxiliary visible/infrared reconstructions."""
        return self._aux_outputs

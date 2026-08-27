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
from models.temporal_attention import SinusoidalPosEmb
from warp_utils import flow_warp

from gsplat.project_gaussians_2d import project_gaussians_2d
from gsplat.rasterize import rasterize_gaussians

class Sine(nn.Module):
    """SIREN 核心激活函数"""
    def __init__(self, w0=30.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)

class MotionAwareWindowScorer(nn.Module):
    """
    通过 SIRENs 从端点特征学习运动权重图 V_map (带严格的 SIREN 初始化)
    """
    def __init__(self, in_channels=64, num_windows=10, hidden_dim=64):
        super().__init__()
        
        # 为了方便独立初始化，我们将 Sequential 拆解
        self.conv1 = nn.Conv2d(in_channels * 2, hidden_dim, 1)
        self.sine1 = Sine(w0=30.0)
        
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 1)
        self.sine2 = Sine(w0=1.0)
        
        self.conv3 = nn.Conv2d(hidden_dim, num_windows, 1)
        
        # 执行 SIREN 专属权重初始化
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            # ==========================================
            # 1. 第一层初始化
            # 公式: U(-1 / fan_in, 1 / fan_in)
            # ==========================================
            fan_in_1 = self.conv1.in_channels
            self.conv1.weight.uniform_(-1 / fan_in_1, 1 / fan_in_1)
            if self.conv1.bias is not None:
                self.conv1.bias.uniform_(-1 / fan_in_1, 1 / fan_in_1)

            # ==========================================
            # 2. 隐藏层初始化
            # 公式: U(-sqrt(6 / fan_in) / w0, sqrt(6 / fan_in) / w0)
            # 注意: 这里的 w0 必须是**前一层**激活函数使用的 w0 (即 30.0)
            # ==========================================
            fan_in_2 = self.conv2.in_channels
            bound_2 = math.sqrt(6 / fan_in_2) / 30.0
            self.conv2.weight.uniform_(-bound_2, bound_2)
            if self.conv2.bias is not None:
                self.conv2.bias.uniform_(-bound_2, bound_2)

            # ==========================================
            # 3. 输出层初始化
            # 因为前一层的 sine2 使用的是 w0=1.0，所以这里除以 1.0
            # ==========================================
            fan_in_3 = self.conv3.in_channels
            bound_3 = math.sqrt(6 / fan_in_3) / 1.0
            self.conv3.weight.uniform_(-bound_3, bound_3)
            if self.conv3.bias is not None:
                self.conv3.bias.uniform_(-bound_3, bound_3)

    def forward(self, feat_0, feat_1):
        x = torch.cat([feat_0, feat_1], dim=1) 
        x = self.sine1(self.conv1(x))
        x = self.sine2(self.conv2(x))
        logits = self.conv3(x)
        return torch.softmax(logits, dim=1)

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
                 flow_model='spynet', searaft_pretrained=None,
                 n_feats=64, freeze_spynet=True, occ_threshold=1.0,
                 gaussian_render_mode='blend', fb_confidence_scale=None,
                 fb_confidence_floor=0.2,
                 motion_force_dense=False,
                 motion_dense_mode='none', motion_flow_smooth_kernel=3,
                 motion_pair_merge=True,
                 tau_gaussian_opacity=0.5,
                 full_res_color=False,
                 flow_driven_xyz=False,
                 occ_fusion=False,
                 occ_gamma=1.0,
                 occ_gamma_flow_ref=10.0,
                 motion_window_gain=4.0,
                 motion_window_max=10.0,
                 motion_window_flow_ref=20.0):
        super().__init__()
        self.flow_model = flow_model
        self.occ_fusion_enabled = occ_fusion
        self.occ_gamma = occ_gamma
        self.occ_gamma_flow_ref = occ_gamma_flow_ref
        # Motion-aware window modulation: enlarge the AOW offset window where the
        # optical flow magnitude is large, so the tanh-limited offset can cover the
        # true object displacement (the paper forbids driving xyz with flow directly).
        self.motion_window_gain = motion_window_gain
        self.motion_window_max = motion_window_max
        self.motion_window_flow_ref = motion_window_flow_ref

        self.n_feats = n_feats
        # Inference switch: predict color on the full-res F_tau instead of the
        # PixelUnshuffle(2) grid, which loses small moving-object details.
        # Weights are untrained by default (random init); useful for ablation.
        self.full_res_color = full_res_color
        # Inference switch: drive Gaussian centers by tau-scaled optical flow so
        # moving objects truly translate to their intermediate position.
        # Without this, centers stay on the static grid and the object is
        # "averaged out" instead of moved.
        self.flow_driven_xyz = flow_driven_xyz
        # Runtime flag: set after loading a checkpoint that trained this head,
        # so demo/inference only uses full-res color when the weights exist.
        self.full_res_color_trained = False
        # Compressed Gaussian latent: channels after PixelUnshuffle(2).
        self.gau_feats = 4 * n_feats        # e.g. n_feats=48 -> 192
        self.cov_dim = 2 * self.gau_feats   # covariance embedding dim (384 for 48)
        self.occ_threshold = occ_threshold
        self.gaussian_render_mode = gaussian_render_mode
        self.fb_confidence_scale = fb_confidence_scale or occ_threshold
        self.fb_confidence_floor = fb_confidence_floor
        self.motion_force_dense = motion_force_dense
        self.motion_dense_mode = motion_dense_mode
        self.motion_flow_smooth_kernel = motion_flow_smooth_kernel
        self.motion_pair_merge = motion_pair_merge
        self.tau_gaussian_opacity = tau_gaussian_opacity
        self.BLOCK_H, self.BLOCK_W = 16, 16

        # --- Optical flow model (SpyNet or SEA-RAFT) ---
        # Kept as self.spynet for backward compatibility with forward().
        if flow_model == 'searaft':
            self.spynet = models.make({
                'name': 'searaft',
                'args': {'pretrained': searaft_pretrained, 'freeze': freeze_spynet}
            })
        else:
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
            'args': {'n_feats': n_feats, 'tau_dim': 64,
                     'use_occlusion_fusion': self.occ_fusion_enabled,
                     'occ_gamma': self.occ_gamma,
                     'occ_gamma_flow_ref': occ_gamma_flow_ref}
        })

        # --- Cross-modal fusion ---
        self.fusion = models.make({
            'name': 'cross-modal-fusion',
            'args': {'in_channels': n_feats}
        })

        # --- Auxiliary reconstruction heads: F_tau -> Vi_tau / IR_tau ---
        # Reconstructing the intermediate-time visible & infrared images from the
        # fused feature forces F_tau to retain BOTH modalities' intermediate
        # content (otherwise the temporal/mask fusion can "fade out" one modal).
        self.aux_vis_decoder = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(n_feats, 3, kernel_size=3, padding=1),
        )
        self.aux_ir_decoder = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(n_feats, 3, kernel_size=3, padding=1),
        )

        # --- Gaussian prediction head (reused from ContinuousSR) ---
        self.ps = nn.PixelUnshuffle(2)  # n_feats -> 4*n_feats (gau_feats), spatial /2
        self.conv1 = nn.Conv2d(self.gau_feats, self.cov_dim, kernel_size=3, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

        # MLP for Gaussian dictionary vector projection (Covariance Prior Bank embedding)
        mlp_vector_spec = {'name': 'mlp', 'args': {
            'in_dim': 3, 'out_dim': self.cov_dim,
            'hidden_list': [self.cov_dim // 2, self.cov_dim, self.cov_dim]
        }}
        self.mlp_vector = models.make(mlp_vector_spec)

        # Convolutional heads for color and offset prediction. A real (signed) tau
        # channel is concatenated with feat_ps so the temporal position stays explicit.
        self.conv_color = nn.Sequential(
            nn.Conv2d(self.gau_feats + 1, self.gau_feats, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(self.gau_feats, self.gau_feats // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(self.gau_feats // 2, 3, kernel_size=1),
        )
        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.gau_feats + 1, self.gau_feats, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(self.gau_feats, self.gau_feats // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(self.gau_feats // 2, 2, kernel_size=1),
        )

        # Full-resolution color head. Operates on F_tau at full res (before
        # PixelUnshuffle) to preserve small moving-object details that the coarse
        # ps grid averages away. Matches the aux decoders' structure (full-res
        # 3x3 convs), so it can reach the same sharpness as aux_vis. Trained
        # end-to-end from scratch in the main loop.
        self.conv_color_full = nn.Sequential(
            nn.Conv2d(n_feats + 1, n_feats, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(n_feats, 3, kernel_size=3, padding=1),
        )

        # --- τ conditioning for Gaussian head (FiLM modulation) ---
        self.tau_embed = SinusoidalPosEmb(64)
        self.tau_film = nn.Sequential(
            nn.Linear(64, self.gau_feats),
            nn.GELU(),
            nn.Linear(self.gau_feats, self.gau_feats * 2),  # γ and β for FiLM
        )

        # Pre-defined Gaussian covariance dictionary (730 templates)
        cho1 = torch.tensor([0, 0.41, 0.62, 0.98, 1.13, 1.29, 1.64, 1.85, 2.36])
        cho2 = torch.tensor([-0.86, -0.36, -0.16, 0.19, 0.34, 0.49, 0.84, 1.04, 1.54])
        cho3 = torch.tensor([0, 0.33, 0.53, 0.88, 1.03, 1.18, 1.53, 1.73, 2.23])
        gau_dict = torch.tensor(list(product(cho1.tolist(), cho2.tolist(), cho3.tolist())))
        gau_dict = torch.cat((gau_dict, torch.zeros(1, 3)), dim=0)  # [730, 3]
        self.register_buffer('gau_dict', gau_dict)

        num_prior_kernels = self.gau_dict.shape[0]

        # CRA (Covariance Resampling Alignment) projection.
        # Input : [Sigma_0, Sigma_N, tau] = 3 + 3 + 1 = 7 dims
        # Output: logits over the Covariance Prior Bank (K = 730)
        self.cra_projection = nn.Linear(3 * 2 + 1, num_prior_kernels)
        self.background = None  # Lazy init on correct device
        self._temporal_reg_loss = None
        self._temporal_stats = {}
        self._aux_outputs = {}
        
        # =======================================================
        # 新增：Motion-Aware Adaptive Offset Window (AOW) 依据原文重写
        # =======================================================
        # 1. 预设 10 个离散的窗口大小 (原文：S_win = {1, 2, ..., 10})
        # 注意：这里的 1 到 10 指的是像素级别。为了方便后续与 tanh() 结合，
        # 我们可能需要调整它的尺度，但为了严谨复现，先保留 1-10。
        window_sizes = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        # 必须变成 [1, 10, 1, 1] 方便后面直接做张量广播乘法
        self.register_buffer('window_bank', window_sizes.view(1, -1, 1, 1))
        
        # 2. 注册基于 SIREN 的打分网络
        # 假设你的特征维度 n_feats = 64
        self.window_scorer = MotionAwareWindowScorer(in_channels=self.n_feats, num_windows=10)

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

    def _make_gaussian_features(self, feat, scale_h, scale_w, tau=None, dense_mode='auto'):
        """Convert dense feature maps into per-Gaussian latent vectors."""
        feat = feat.float()
        bs, C, fh, fw = feat.shape

        # PixelUnshuffle to get richer per-point features.
        # For scale=1 this keeps the original compact point set. For SR rendering
        # (scale>1), the compact fh/2 × fw/2 grid becomes too sparse in the HR
        # canvas and produces periodic low-coverage gaps. Densify to one Gaussian
        # per LR pixel for scale>1 to avoid grid-like holes between samples.
        feat_ps = self.ps(feat)  # [B, C*4, fh/2, fw/2] = [B, gau_feats, fh/2, fw/2]

        # τ FiLM modulation: condition Gaussian features on temporal position
        tau_t = self._expand_tau(tau, bs, feat.device)
        if tau_t is not None:
            tau_emb = self.tau_embed(tau_t.float())                 # [B, 64]
            film_params = self.tau_film(tau_emb)                    # [B, 2*gau_feats]
            gamma = film_params[:, :self.gau_feats].unsqueeze(-1).unsqueeze(-1) + 1.0
            beta = film_params[:, self.gau_feats:].unsqueeze(-1).unsqueeze(-1)
            feat_ps = gamma * feat_ps + beta

        dense_sr_render = dense_mode in ('full', 'vertical') or max(scale_h, scale_w) > 1.0
        if dense_mode == 'vertical':
            feat_ps = F.interpolate(feat_ps, size=(fh, feat_ps.shape[-1]), mode='nearest')
        elif dense_sr_render:
            feat_ps = F.interpolate(feat_ps, size=(fh, fw), mode='nearest')

        return feat_ps, dense_sr_render

    def _predict_gaussian_params(self, feat_ps, tau=None):
        """Predict color, covariance, and local offset for each Gaussian."""
        bs = feat_ps.shape[0]
        ps_h, ps_w = feat_ps.shape[2], feat_ps.shape[3]
        n_gaussians = ps_h * ps_w

        # Reshape: [B, gau_feats, ps_h, ps_w] -> [B*ps_h*ps_w, gau_feats]
        feat_flat = feat_ps.permute(0, 2, 3, 1).reshape(bs * n_gaussians, self.gau_feats)

        tau_t = self._expand_tau(tau, bs, feat_ps.device)
        if tau_t is None:
            tau_channel = feat_ps.new_zeros(bs, 1, ps_h, ps_w)
        else:
            tau_channel = tau_t.to(device=feat_ps.device, dtype=feat_ps.dtype).view(bs, 1, 1, 1)
            tau_channel = tau_channel.expand(-1, 1, ps_h, ps_w)
        param_feat = torch.cat([feat_ps, tau_channel], dim=1)

        # Color (sigmoid with -2 shift: init at sigmoid(-2)≈0.12, below typical targets)
        # Forces gradient to push UP (away from sigmoid saturation at 0)
        color_map = self.conv_color(param_feat)
        color_all = torch.sigmoid(
            color_map.permute(0, 2, 3, 1).reshape(bs, n_gaussians, 3) - 2.0
        )

        # Covariance via dictionary (same mechanism as ContinuousSR)
        # Detach features to prevent encoder side-effects on covariance
        para_feat = self.leaky_relu(feat_ps.detach())
        para_conv = self.conv1(para_feat)  # [B, cov_dim, ps_h, ps_w]
        para_flat = para_conv.permute(0, 2, 3, 1).reshape(bs * n_gaussians, self.cov_dim)

        vector = self.mlp_vector(self.gau_dict)  # [730, cov_dim]
        similarity = vector @ para_flat.t()  # [730, B*n_gaussians]
        # 对先验字典的打分权重
        weights = torch.softmax(similarity, dim=0)  # [730, B*n_gaussians]
        # 转置并 reshape 为 [B, n_gaussians, 730] 以便后续 CRAM 拼接
        kernel_weights = weights.t().reshape(bs, n_gaussians, -1)

        cov_all = (weights.t() @ self.gau_dict).reshape(bs, n_gaussians, 3)
        # ==========================================
        # 运动感知自适应偏移 (Motion-Aware Offset)
        # ==========================================
        # 计算基础偏移方向 [-1, 1]
        offset_map = torch.tanh(self.conv_offset(param_feat))
        base_offset_flat = offset_map.permute(0, 2, 3, 1).reshape(bs * n_gaussians, 2) # [B*n_gaussians, 2]
        # 将其还原为空间维度以便与 w_map 对齐
        base_offset = base_offset_flat.view(bs, ps_h, ps_w, 2).permute(0, 3, 1, 2) # [B, 2, ps_h, ps_w]
        # 引入外层计算好的共享窗口 W_map
        if hasattr(self, 'current_w_map') and self.current_w_map is not None:
            w_map = self.current_w_map # [B, 1, H, W]
            
            # 如果 feat_ps 被下采样过（比如 PixelUnshuffle），需要对齐空间分辨率
            if w_map.shape[-2:] != (ps_h, ps_w):
                w_map_aligned = F.interpolate(w_map, size=(ps_h, ps_w), mode='nearest')
            else:
                w_map_aligned = w_map
            
            # 核心公式：Δ𝜇𝑡 = Δ𝜇𝑡 ⊙ 𝑊𝑚𝑎𝑝
            final_offset = base_offset * w_map_aligned
        else:
            # 兼容没有 W_map 的情况（或老版本）
            final_offset = base_offset
        
        offset_all = final_offset.permute(0, 2, 3, 1).reshape(bs, n_gaussians, 2)
        return {
            'color': color_all,
            'kernel_weights': kernel_weights, # 新增：返回分布权重
            'cov': cov_all, # 保留给老式渲染路径使用
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
        """Downsample flow to the Gaussian (ps) grid -> [B, N, 2] pixel units."""
        flow_ps = F.interpolate(flow, size=(ps_h, ps_w), mode='bilinear', align_corners=True)
        return flow_ps.permute(0, 2, 3, 1).reshape(flow.shape[0], ps_h * ps_w, 2)

    def _compute_fb_confidence(self, flow_01, flow_10):
        """Forward-backward consistency confidence for both directions.

        Returns (conf_0, conf_1) in [conf_floor, 1]; high = reliable.
        """
        flow_10_at_1 = flow_warp(flow_10, flow_01)
        flow_01_at_0 = flow_warp(flow_01, flow_10)
        cons_0 = torch.norm(flow_01 + flow_10_at_1, dim=1, keepdim=True)
        cons_1 = torch.norm(flow_10 + flow_01_at_0, dim=1, keepdim=True)
        sigma = max(float(self.fb_confidence_scale), 1e-6)
        conf_floor = float(self.fb_confidence_floor)
        conf_0 = torch.exp(-cons_0 / sigma).clamp(conf_floor, 1.0)
        conf_1 = torch.exp(-cons_1 / sigma).clamp(conf_floor, 1.0)
        return conf_0, conf_1

    def _apply_flow_motion(self, xyz, flow, tau_weight, ps_h, ps_w):
        """Displace Gaussian centers by tau-scaled flow (in normalized coords).

        xyz is in [-1,1] normalized coords; flow is in pixels. We convert the
        displacement to normalized units using the ps grid resolution.
        """
        flow_flat = self._flow_to_gaussian_grid(flow, ps_h, ps_w)   # [B, N, 2] px
        if not isinstance(tau_weight, torch.Tensor):
            tau_weight = torch.tensor(tau_weight, device=xyz.device, dtype=xyz.dtype)
        tau_weight = tau_weight.to(device=xyz.device, dtype=xyz.dtype).view(-1, 1, 1)
        flow_h, flow_w = flow.shape[2], flow.shape[3]
        disp_x = 2 * tau_weight * flow_flat[:, :, 0:1] / flow_w
        disp_y = 2 * tau_weight * flow_flat[:, :, 1:2] / flow_h
        return xyz + torch.cat((disp_x, disp_y), dim=2)

    def _confidence_to_gaussian_grid(self, confidence, ps_h, ps_w):
        conf_ps = F.interpolate(confidence, size=(ps_h, ps_w), mode='bilinear', align_corners=True)
        return conf_ps.permute(0, 2, 3, 1).reshape(confidence.shape[0], ps_h * ps_w, 1)

    def _flow_displacement_to_xyz(self, flow, tau_weight, ps_h, ps_w, H, W):
        """Flow-driven Gaussian displacement -> normalized coords delta."""
        flow_flat = self._flow_to_gaussian_grid(flow, ps_h, ps_w)   # [B,N,2] px
        if not isinstance(tau_weight, torch.Tensor):
            tau_weight = torch.tensor(tau_weight, device=flow.device, dtype=flow.dtype)
        tau_weight = tau_weight.to(device=flow.device, dtype=flow.dtype).view(-1, 1, 1)
        # Convert pixel displacement to normalized [-1,1] delta (grid spacing ~ 2/ps).
        disp_x = 2 * tau_weight * flow_flat[:, :, 0:1] / ps_w
        disp_y = 2 * tau_weight * flow_flat[:, :, 1:2] / ps_h
        return torch.cat((disp_x, disp_y), dim=2)

    def _predict_color_offset(self, feat_ps, tau):
        """
        Predict color / offset directly from the intermediate-time feature F_tau.

        Color : C_rgb,tau = H_c(F_tau, tau)
        Offset: dmu_tau   = tanh(H_mu(F_tau, tau)) ⊙ W_map   (AOW-adjusted)

        Returns dict: color [B, N, 3], offset [B, N, 2], ps_h, ps_w, n_gaussians.
        """
        bs, _, ps_h, ps_w = feat_ps.shape
        n_gaussians = ps_h * ps_w

        tau_t = self._expand_tau(tau, bs, feat_ps.device)
        if tau_t is None:
            tau_t = feat_ps.new_zeros(bs)
        tau_channel = tau_t.to(dtype=feat_ps.dtype).view(bs, 1, 1, 1).expand(-1, 1, ps_h, ps_w)
        param_feat = torch.cat([feat_ps, tau_channel], dim=1)

        # ---- Color ----
        color_map = self.conv_color(param_feat)
        color = torch.sigmoid(
            color_map.permute(0, 2, 3, 1).reshape(bs, n_gaussians, 3) - 2.0
        )

        # ---- Initial offset (tanh in [-1, 1]) ----
        offset_map = torch.tanh(self.conv_offset(param_feat))

        # ---- Adaptive Offset Window (AOW): dmu = dmu_tilde ⊙ W_map ----
        if hasattr(self, 'current_w_map') and self.current_w_map is not None:
            w_map = self.current_w_map  # [B, 1, H, W]
            if w_map.shape[-2:] != (ps_h, ps_w):
                w_map = F.interpolate(w_map, size=(ps_h, ps_w), mode='nearest')
            offset_map = offset_map * w_map

        offset = offset_map.permute(0, 2, 3, 1).reshape(bs, n_gaussians, 2)

        return {
            'color': color,
            'offset': offset,
            'ps_h': ps_h,
            'ps_w': ps_w,
            'n_gaussians': n_gaussians,
        }

    def _predict_endpoint_covariance(self, feat_ps):
        """
        Predict endpoint covariance from an anchor feature (F_0 or F_N) via the
        Covariance Prior Bank (CPB).

        Returns (covariance [B, N, 3], kernel_weights [B, N, K]).
        """
        bs, _, ps_h, ps_w = feat_ps.shape
        n_gaussians = ps_h * ps_w

        para_feat = self.leaky_relu(feat_ps.detach())
        para_conv = self.conv1(para_feat)  # [B, cov_dim, ps_h, ps_w]
        para_flat = para_conv.permute(0, 2, 3, 1).reshape(bs * n_gaussians, self.cov_dim)

        vector = self.mlp_vector(self.gau_dict)  # [K, cov_dim]
        similarity = vector @ para_flat.t()      # [K, B*n_gaussians]
        weights = torch.softmax(similarity, dim=0)  # [K, B*n_gaussians]

        kernel_weights = weights.t().reshape(bs, n_gaussians, -1)  # [B, N, K]
        covariance = (weights.t() @ self.gau_dict).reshape(bs, n_gaussians, 3)  # [B, N, 3]
        return covariance, kernel_weights

    def _cra_resample_covariance(self, cov_0, cov_N, tau):
        """
        CRA (Covariance Resampling Alignment).

        E_tau     = P([Sigma_0, Sigma_N, tau])       # projection -> logits [B, N, K]
        w_tau     = Softmax(E_tau)
        Sigma_tau = sum_k w_tau,k * Sigma_k_prior    # weighted prior recombination

        Returns (cov_tau [B, N, 3], weights [B, N, K]).
        """
        bs, n_gaussians, _ = cov_0.shape

        tau_t = self._expand_tau(tau, bs, cov_0.device)
        if tau_t is None:
            tau_t = cov_0.new_zeros(bs)
        tau_expand = tau_t.to(dtype=cov_0.dtype).view(bs, 1, 1).expand(bs, n_gaussians, 1)

        cra_input = torch.cat([cov_0, cov_N, tau_expand], dim=-1)  # [B, N, 7]

        logits = self.cra_projection(cra_input)                     # [B, N, K]
        weights = torch.softmax(logits, dim=-1)                     # [B, N, K]
        cov_tau = weights @ self.gau_dict.to(dtype=weights.dtype)   # [B, N, 3]

        return cov_tau, weights

    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type='cuda')
    def _render_gaussian_params(self, color_all, cov_all, xyz_all, opacity_all,
                                scale_h, scale_w, H, W, dense_sr_render,
                                debug_tag='gaussian'):
        """Rasterize already-positioned Gaussian parameters."""
        bs, n_gaussians = color_all.shape[:2]

        # Rasterize per batch. A fresh local background avoids caching an
        # inference-mode tensor (which would break training backward later).
        background = torch.zeros(3, device=color_all.device, dtype=color_all.dtype)

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
            min_std_w = max(0.5, 0.5 * scale_w) if dense_sr_render else 0.85
            min_std_h = max(0.5, 0.5 * scale_h) if dense_sr_render else 1.0
            # Inference-only footprint scale (default 1.0 = off). <1 shrinks each
            # Gaussian's footprint to reduce over-blending between adjacent
            # Gaussians; >1 enlarges. For experiments only.
            fp_scale = float(getattr(self, 'footprint_scale', 1.0))
            L11 = (cov_i[:, 0] * scale_w / 2.0 + min_std_w) * fp_scale
            L21 = cov_i[:, 1] * scale_h / 2.0 * fp_scale
            L22 = (cov_i[:, 2] * scale_h / 2.0 + min_std_h) * fp_scale
            weighted_cholesky = torch.stack([L11, L21, L22], dim=1)

            # Project and rasterize
            xys, depths, radii, conics, num_tiles_hit = project_gaussians_2d(
                xyz, weighted_cholesky, H, W, tile_bounds
            )
            # Standard alpha-composited rasterization. (The fork's
            # rasterize_gaussians_sum has a tile-boundary defect: every 16-row
            # tile's bottom rows get zero coverage in dense (1 Gaussian/pixel)
            # mode, producing periodic horizontal black lines. The standard
            # alpha-blended rasterizer has no such artifact.)
            out_img, out_alpha = rasterize_gaussians(
                xys, depths, radii, conics, num_tiles_hit,
                color_i, opacity, H, W,
                self.BLOCK_H, self.BLOCK_W,
                background=background, return_alpha=True
            )
            # out_img: [H, W, 3]; alpha: [H, W] in [0,1]
            out_img = out_img.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
            coverage = out_alpha.unsqueeze(0).unsqueeze(0).clamp(0.0, 1.0)  # [1,1,H,W]
            low_coverage = coverage < 0.25
            if low_coverage.any():
                pooled_num = F.avg_pool2d(out_img * coverage, kernel_size=5, stride=1, padding=2)
                pooled_den = F.avg_pool2d(coverage, kernel_size=5, stride=1, padding=2)
                filled = pooled_num / pooled_den.clamp(min=1e-6)
                out_img = torch.where(low_coverage, filled, out_img)
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
                    'coverage_min': coverage.min().item(),
                    'coverage_low_frac': low_coverage.float().mean().item(),
                    'n_gaussians': n_gaussians,
                    'dense_sr_render': float(dense_sr_render),
                }

        return torch.cat(pred, dim=0)  # [B, 3, H, W]

    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type='cuda')
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

    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type='cuda')
    def _render_motion_gaussians(self, feat_fused_0, feat_fused_N, flow_01, flow_10,
                                scale_h, scale_w, lr_h, lr_w, tau,
                                feat_fused_tau=None):
        """
        Paper-aligned Gaussian parameter generation.

        Branch A (color / offset):
            F_tau -> color_tau / offset_tau;  AOW scales the offset;
            mu_tau = grid + offset_tau.

        Branch B (covariance):
            F_0 -> Sigma_0,  F_N -> Sigma_N  (via Covariance Prior Bank);
            CRA(Sigma_0, Sigma_N, tau) -> Sigma_tau.

        Optical flow was already used to build F_tau; it is NOT applied to xyz here
        (no double motion compensation).
        """
        bs = feat_fused_tau.shape[0]
        device = feat_fused_tau.device
        H = round(lr_h * scale_h)
        W = round(lr_w * scale_w)

        tau_t = self._expand_tau(tau, bs, device)

        if feat_fused_tau is None:
            feat_fused_tau = 0.5 * (feat_fused_0 + feat_fused_N)

        if self.motion_force_dense or max(scale_h, scale_w) > 1.0:
            dense_mode = 'full'
        else:
            dense_mode = self.motion_dense_mode

        # ============================================================
        # Branch A: F_tau -> color_tau / offset_tau -> mu_tau = grid + offset
        # ============================================================
        feat_ps_tau, dense_tau = self._make_gaussian_features(
            feat_fused_tau, scale_h, scale_w,
            tau=tau_t,                 # real temporal hyper-parameter
            dense_mode=dense_mode,
        )
        if getattr(self, 'full_res_color', False) and getattr(self, 'full_res_color_trained', False):
            # Predict color on the full-res F_tau so small moving objects are not
            # averaged away by the PixelUnshuffle grid. Strong full-res head
            # (aux-decoder-like) trained end-to-end in the main loop.
            tau_map = tau_t.to(dtype=feat_fused_tau.dtype).view(bs, 1, 1, 1)
            tau_map = tau_map.expand(-1, 1, feat_fused_tau.shape[2], feat_fused_tau.shape[3])
            param_feat_full = torch.cat([feat_fused_tau, tau_map], dim=1)
            color_tau_full = torch.sigmoid(self.conv_color_full(param_feat_full) - 2.0)
            # offset still comes from the ps grid
            params_tau = self._predict_color_offset(feat_ps_tau, tau_t)
            ps_h, ps_w = params_tau['ps_h'], params_tau['ps_w']
            offset_tau = params_tau['offset']
            n_gaussians = ps_h * ps_w
            # Resample full-res color (H*W) onto the Gaussian grid (ps_h*ps_w).
            # In dense mode the ps grid is already full-res (H*W), so no pooling
            # is needed; only the half-res (PixelUnshuffle) grid needs pooling.
            if (ps_h, ps_w) != color_tau_full.shape[-2:]:
                color_grid = F.avg_pool2d(color_tau_full, kernel_size=2, stride=2)  # [B, 3, ps_h, ps_w]
                if color_grid.shape[-2:] != (ps_h, ps_w):
                    color_grid = F.interpolate(color_grid, size=(ps_h, ps_w), mode='bilinear', align_corners=False)
            else:
                color_grid = color_tau_full
            color_tau = color_grid.permute(0, 2, 3, 1).reshape(bs, n_gaussians, 3)
        else:
            params_tau = self._predict_color_offset(feat_ps_tau, tau_t)
            ps_h, ps_w = params_tau['ps_h'], params_tau['ps_w']
            color_tau = params_tau['color']
            offset_tau = params_tau['offset']
            n_gaussians = params_tau['n_gaussians']

        # mu_tau = grid + Delta mu_tau (AOW-adjusted offset)
        # Optional flow-driven displacement (motion compensation): move Gaussian
        # centers to the intermediate position via tau-scaled optical flow.
        # Use forward-backward consistency to weight which endpoint's motion is
        # more reliable in occluded regions (motion + occlusion direct transfer).
        if getattr(self, 'flow_driven_xyz', False):
            xyz_base = self._base_xyz(offset_tau, ps_h, ps_w, H, W)
            # Displace from frame-0 and frame-N to time tau.
            xyz_from_0 = self._apply_flow_motion(xyz_base, flow_01, tau_t, ps_h, ps_w)
            xyz_from_N = self._apply_flow_motion(xyz_base, flow_10, 1.0 - tau_t, ps_h, ps_w)
            # Forward-backward consistency confidence.
            conf_0, conf_N = self._compute_fb_confidence(flow_01, flow_10)
            conf0_grid = self._confidence_to_gaussian_grid(conf_0, ps_h, ps_w)
            confN_grid = self._confidence_to_gaussian_grid(conf_N, ps_h, ps_w)
            # Blend the two motion-compensated positions by confidence.
            pos_den = (conf0_grid + confN_grid).clamp_min(1e-6)
            pos_w0 = conf0_grid / pos_den
            pos_wN = confN_grid / pos_den
            xyz_tau = pos_w0 * xyz_from_0 + pos_wN * xyz_from_N
            # Also record motion stats for debugging.
            with torch.no_grad():
                self._temporal_stats['fb_conf0_mean'] = conf_0.mean().item()
                self._temporal_stats['fb_confN_mean'] = conf_N.mean().item()
        else:
            xyz_tau = self._base_xyz(offset_tau, ps_h, ps_w, H, W)

        # ============================================================
        # Branch B: endpoint covariance -> CRA -> Sigma_tau
        # ============================================================
        feat_ps_0, dense_0 = self._make_gaussian_features(
            feat_fused_0, scale_h, scale_w, tau=None, dense_mode=dense_mode,
        )
        feat_ps_N, dense_N = self._make_gaussian_features(
            feat_fused_N, scale_h, scale_w, tau=None, dense_mode=dense_mode,
        )
        cov_0, _ = self._predict_endpoint_covariance(feat_ps_0)
        cov_N, _ = self._predict_endpoint_covariance(feat_ps_N)

        cov_tau, cra_weights = self._cra_resample_covariance(cov_0, cov_N, tau_t)

        # ============================================================
        # Opacity (fixed at 1 for the paper-aligned path)
        # ============================================================
        opacity = torch.ones(
            bs, params_tau['n_gaussians'], 1,
            device=device, dtype=color_tau.dtype,
        )

        # ============================================================
        # Debug / monitoring stats
        # ============================================================
        with torch.no_grad():
            self._temporal_stats['tau_mean'] = tau_t.mean().item()
            self._temporal_stats['offset_abs_mean'] = offset_tau.abs().mean().item()
            self._temporal_stats['offset_abs_max'] = offset_tau.abs().max().item()
            if hasattr(self, 'current_w_map') and self.current_w_map is not None:
                self._temporal_stats['aow_mean'] = self.current_w_map.mean().item()
                self._temporal_stats['aow_max'] = self.current_w_map.max().item()
            self._temporal_stats['cov0_mean'] = cov_0.mean().item()
            self._temporal_stats['covN_mean'] = cov_N.mean().item()
            self._temporal_stats['cov_tau_mean'] = cov_tau.mean().item()
            self._temporal_stats['cra_entropy'] = (
                -cra_weights * torch.log(cra_weights.clamp_min(1e-8))
            ).sum(dim=-1).mean().item()

        return self._render_gaussian_params(
            color_tau, cov_tau, xyz_tau, opacity,
            scale_h, scale_w, H, W,
            dense_tau or dense_0 or dense_N, debug_tag='motion',
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

        # --- Step 4: Cross-modal fusion at the two anchor frames ---
        feat_fused_0 = self.fusion(feat_vis_0, feat_ir_0)   # F_0
        feat_fused_N = self.fusion(feat_vis_N, feat_ir_N)   # F_N

        # --- Step 5: AOW (Motion-Aware Adaptive Offset Window) ---
        # SIREN scores endpoint fused features, yielding a per-pixel window size
        # map W_map. Later offset is scaled: dmu_tau = dmu_tilde ⊙ W_map.
        if hasattr(self, 'window_scorer'):
            v_map = self.window_scorer(feat_fused_0, feat_fused_N)  # [B, 10, H, W]
            w_map = torch.sum(v_map * self.window_bank, dim=1, keepdim=True)  # [B, 1, H, W]
            # Keep the raw SIREN window for the AOW regularizer (below). The
            # regularizer should only push the *learned* window toward 1; the
            # deterministic motion modulation (no_grad) must stay untouched.
            self.current_w_map_raw = w_map

            # ---- Motion-aware window modulation ----
            # The tanh-limited offset alone moves a Gaussian at most ~2px, but a
            # moving object may need 10-30px. Since the paper forbids driving xyz
            # with flow directly, we enlarge the offset window where optical flow
            # is large so the offset can span the true displacement.
            with torch.no_grad():
                flow_mag = torch.max(
                    flow_01.norm(dim=1, keepdim=True),
                    flow_10.norm(dim=1, keepdim=True),
                )  # [B,1,H,W]
                if flow_mag.shape[-2:] != w_map.shape[-2:]:
                    flow_mag = F.interpolate(flow_mag, size=w_map.shape[-2:],
                                             mode='bilinear', align_corners=False)
                motion_factor = 1.0 + self.motion_window_gain * (
                    flow_mag / self.motion_window_flow_ref
                )
                motion_factor = motion_factor.clamp(1.0, self.motion_window_max)
                w_map = w_map * motion_factor
            self.current_w_map = w_map

        # --- Step 6: CGM on fused endpoint features -> F_tau ---
        feat_fused_tau = self.temporal_attn(
            feat_fused_0, feat_fused_N, flow_01, flow_10, tau, self.occ_threshold
        )

        self._temporal_reg_loss = None
        self._temporal_stats = {}
        cgm_stats = dict(getattr(self.temporal_attn, '_debug_stats', {}))
        for k, v in cgm_stats.items():
            self._temporal_stats[f'temporal_{k}'] = v
        self._aux_outputs = {}

        # Per-sample tau tensor for aux/color-head conditioning.
        tau_t = self._expand_tau(tau, vis_0.shape[0], vis_0.device)
        if tau_t is None:
            tau_t = vis_0.new_zeros(vis_0.shape[0])
        bs = vis_0.shape[0]

        # --- Step 6b: Auxiliary reconstruction of intermediate-time modalities ---
        # Reconstruct Vi_tau and IR_tau from the fused F_tau feature. Supervision
        # against the real intermediate frames (vis_gt / ir_gt) forces F_tau to
        # keep BOTH modalities' content, preventing modal fade-out in the fusion.
        #
        # Motion-weighted supervision: the reconstruction loss is weighted per
        # pixel by the optical-flow magnitude (normalized). Background regions
        # (tiny flow) barely change and contribute little, so the model focuses
        # on reconstructing the actually-moving regions where temporal fusion
        # matters most.
        if self.training or getattr(self, 'compute_aux', True):
            aux_vis = self.aux_vis_decoder(feat_fused_tau)
            aux_ir = self.aux_ir_decoder(feat_fused_tau)
            self._aux_outputs['vis_tau'] = aux_vis
            self._aux_outputs['ir_tau'] = aux_ir
            # If the full-res color head is active, also expose its raw color
            # map so the training loop can supervise it directly (accelerates
            # convergence of the color head toward aux-level sharpness).
            if getattr(self, 'full_res_color', False) and getattr(self, 'full_res_color_trained', False):
                tau_map = tau_t.to(dtype=feat_fused_tau.dtype).view(bs, 1, 1, 1)
                tau_map = tau_map.expand(-1, 1, feat_fused_tau.shape[2], feat_fused_tau.shape[3])
                self._aux_outputs['full_color_vis'] = torch.sigmoid(
                    self.conv_color_full(torch.cat([feat_fused_tau, tau_map], dim=1)) - 2.0
                )
            # Per-pixel motion weight in [0,1]: normalized average of the two
            # bidirectional flows, upsampled to the aux output resolution.
            with torch.no_grad():
                f_01 = F.interpolate(flow_01, size=feat_fused_tau.shape[-2:],
                                     mode='bilinear', align_corners=True)
                f_10 = F.interpolate(flow_10, size=feat_fused_tau.shape[-2:],
                                     mode='bilinear', align_corners=True)
                mag = (0.5 * (f_01.norm(dim=1, keepdim=True)
                              + f_10.norm(dim=1, keepdim=True)))  # [B,1,H,W]
                w = mag / (mag.flatten(1).max(dim=1).values.view(-1, 1, 1, 1) + 1e-6)
                self._aux_outputs['motion_weight'] = w.clamp(0, 1)

        # --- Step 7: Gaussian rendering ---
        if self.gaussian_render_mode == 'motion':
            output = self._render_motion_gaussians(
                feat_fused_0, feat_fused_N, flow_01, flow_10,
                scale_h, scale_w, lr_h, lr_w, tau=tau,
                feat_fused_tau=feat_fused_tau
            )
        else:
            # Backward-compatible path: render directly from the CGM feature.
            output = self._render_gaussians(feat_fused_tau, scale_h, scale_w, lr_h, lr_w, tau=tau)
        if hasattr(self, '_debug_stats'):
            self._debug_stats.update(self._temporal_stats)

        return output

    def get_temporal_regularization(self):
        """Return the latest differentiable temporal anti-collapse loss."""
        return self._temporal_reg_loss

    def get_aow_regularization(self):
        """
        Mild regularizer pushing the AOW window map towards 1.0, preventing the
        SIREN window scorer from collapsing onto the maximum window (which would
        make offsets reach ±10 px immediately).
        """
        # Regularize the *raw learned* window only; the motion modulation is
        # deterministic and must not be pulled back toward 1 by the loss.
        w_map = getattr(self, 'current_w_map_raw', None)
        if w_map is None:
            return None
        return (w_map - 1.0).abs().mean()

    def get_aux_outputs(self):
        """Return latest auxiliary visible/infrared reconstructions (removed)."""
        return self._aux_outputs

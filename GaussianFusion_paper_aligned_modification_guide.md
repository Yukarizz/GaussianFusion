# GaussianFusion 论文对齐修改指南

> 目标：将当前 GaussianFusion 实现修改为与论文流程更一致的版本：
>
> - **Color / Offset**：由中间时刻 \(\tau\) 的融合特征直接预测；
> - **Offset**：通过 AOW（Adaptive Offset Window）进行自适应尺度调整；
> - **Covariance**：由两个端点的 covariance 经过 CRA（Covariance Resampling Alignment）得到中间时刻 covariance；
> - **Flow**：只负责把端点特征对齐到中间时刻，不再直接对最终 Gaussian 坐标做二次位移；
> - **最终 Gaussian 坐标**：由规则网格 + 中间时刻预测的 offset 直接确定。

---

## 1. 推荐从哪个分支开始修改

建议从：

```bash
feature/regression_color_offset
```

继续修改，而不是从 `main` 或较旧的 `feature/gau_moving` 开始。

建议创建自己的开发分支：

```bash
git fetch origin

git switch feature/regression_color_offset
git pull --ff-only origin feature/regression_color_offset

git switch -c feature/paper_exact_cgm_cra
```

这样可以保留已有的：

- Gaussian 参数预测框架；
- Covariance Prior Bank；
- AOW；
- SIREN window scorer；
- \(\tau\)-condition；
- color / offset head；
- Gaussian splatting 渲染路径。

---

# 2. 最终想实现的整体数据流

目标结构：

```text
Visible_0 + IR_0                 Visible_N + IR_N
       │                                │
       ▼                                ▼
    Encoder                          Encoder
       │                                │
       ▼                                ▼
   Fusion F_0                       Fusion F_N
       │                                │
       ├───────────────┬────────────────┤
       │               │                │
       │               │                │
       ▼               ▼                ▼
      CGM             AOW              CRA
       │               │                │
       ▼               │                │
      F_tau            │         Sigma_0 / Sigma_N
       │               │                │
       ├───────┐       │                │
       ▼       ▼       │                ▼
   color_tau offset_tau │         [Sigma_0,Sigma_N,tau]
               │       │                │
               └── × ──┘                ▼
                    │                Projection
                    ▼                    │
              Delta mu_tau               ▼
                    │               Softmax weights
                    ▼                    │
          mu_tau = grid + offset         ▼
                                     Prior Bank
                                        │
                                        ▼
                                     Sigma_tau
                                        │
                    ┌───────────────────┘
                    ▼
          Gaussian Parameters at tau
       {mu_tau, Sigma_tau, color_tau}
                    │
                    ▼
            Gaussian Splatting
```

---

# 3. 设计原则

## 3.1 Color / Offset

中间时刻融合特征：

\[
F_\tau
\]

直接预测：

\[
C_{rgb,\tau}
\]

和：

\[
\Delta \mu_\tau
\]

最终 Gaussian 中心：

\[
\mu_\tau = p + \Delta \mu_\tau
\]

其中 \(p\) 是规则 Gaussian 网格中心。

---

## 3.2 Offset 的作用

Offset 不是 optical flow。

它代表：

> 当前 \(\tau\) 时刻 Gaussian 中心相对于规则网格 anchor 的二维可学习偏移。

即：

\[
\Delta \mu_\tau =
\begin{bmatrix}
\Delta x_\tau \\
\Delta y_\tau
\end{bmatrix}
\]

最终：

\[
\mu_\tau = \mu_{\text{grid}} + \Delta\mu_\tau
\]

---

## 3.3 Optical Flow 的作用

Flow 只负责：

```text
F_0
  \
   \ backward warp
    \
     F_tau
    /
   / backward warp
  /
F_N
```

也就是用于构造中间时刻特征。

不要再做：

```python
xyz_tau = xyz_tau + tau * flow_01
```

也不要：

```python
xyz_tau = confidence_fuse(xyz_from_0, xyz_from_N)
```

否则相当于：

```text
flow 对齐一次 feature
+
flow 再移动一次 Gaussian
```

存在二次运动补偿。

---

# 4. 第一步：修改 Temporal 模块

建议修改：

```text
models/temporal_attention.py
```

虽然文件名仍然可以保留 `temporal_attention.py`，但内部逻辑改为论文式 CGM：

```text
flow
 ↓
tau condition / linear scale
 ↓
两端 backward warp
 ↓
feature fusion
 ↓
F_tau
```

---

## 4.1 推荐实现

保留原来的 registry 名称，避免改 YAML：

```python
@register('temporal-cross-attention')
class TemporalCrossAttention(nn.Module):
    """
    Paper-aligned Continuous Gaussian Motion feature interpolation.

    1. Scale bidirectional flow according to tau.
    2. Warp endpoint features toward target time tau.
    3. Fuse two warped features.
    """

    def __init__(self, n_feats=64, tau_dim=64, n_heads=4):
        super().__init__()

        self.n_feats = n_feats
        self.last_reg_loss = None
        self._debug_stats = {}

        self.fusion_head = nn.Sequential(
            nn.Conv2d(n_feats * 2 + 1, n_feats, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(n_feats, n_feats, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),

            # 1 channel mask + C channel residual
            nn.Conv2d(n_feats, n_feats + 1, 3, padding=1),
        )

    def forward(
        self,
        feat_0,
        feat_1,
        flow_01,
        flow_10,
        tau,
        occ_threshold=5.0,
    ):
        B, C, H, W = feat_0.shape

        if not isinstance(tau, torch.Tensor):
            tau = torch.full(
                (B,),
                float(tau),
                device=feat_0.device,
                dtype=feat_0.dtype,
            )
        elif tau.dim() == 0:
            tau = tau.reshape(1).expand(B)

        tau = tau.to(
            device=feat_0.device,
            dtype=feat_0.dtype,
        )

        t = tau.view(B, 1, 1, 1)

        # ======================================================
        # 1. tau-conditioned flow scaling
        # ======================================================

        flow_0t = t * flow_01
        flow_1t = (1.0 - t) * flow_10

        # ======================================================
        # 2. Backward Warping
        # ======================================================

        warped_0 = flow_warp(
            feat_0,
            flow_0t,
        )

        warped_1 = flow_warp(
            feat_1,
            flow_1t,
        )

        # ======================================================
        # 3. Feature Fusion
        # ======================================================

        tau_map = t.expand(
            B,
            1,
            H,
            W,
        )

        fusion_input = torch.cat(
            [
                warped_0,
                warped_1,
                tau_map,
            ],
            dim=1,
        )

        pred = self.fusion_head(
            fusion_input
        )

        mask = torch.sigmoid(
            pred[:, 0:1]
        )

        residual = pred[:, 1:]

        feat_tau = (
            mask * warped_0
            + (1.0 - mask) * warped_1
            + residual
        )

        self.last_reg_loss = None

        with torch.no_grad():
            self._debug_stats = {
                'mask_mean': mask.mean().item(),
                'residual_abs_mean': residual.abs().mean().item(),
                'tau_mean': tau.mean().item(),
            }

        return feat_tau
```

---

# 5. 第二步：先做端点多模态融合

原本如果是：

```python
feat_vis_tau = temporal_attn(...)
feat_ir_tau = temporal_attn(...)

feat_fused_tau = fusion(
    feat_vis_tau,
    feat_ir_tau
)
```

建议改为：

```text
vis_0 + ir_0 → F_0

vis_N + ir_N → F_N

F_0 + F_N → CGM → F_tau
```

---

## 5.1 Forward 推荐结构

```python
# ============================================================
# Step 1: Encode
# ============================================================

feat_vis_0, feat_ir_0 = self.encode(
    vis_0,
    ir_0
)

feat_vis_N, feat_ir_N = self.encode(
    vis_N,
    ir_N
)

# ============================================================
# Step 2: Cross-modal fusion at two anchor frames
# ============================================================

feat_fused_0 = self.fusion(
    feat_vis_0,
    feat_ir_0
)

feat_fused_N = self.fusion(
    feat_vis_N,
    feat_ir_N
)

# ============================================================
# Step 3: AOW
# ============================================================

v_map = self.window_scorer(
    feat_fused_0,
    feat_fused_N
)

self.current_w_map = torch.sum(
    v_map * self.window_bank,
    dim=1,
    keepdim=True
)

# ============================================================
# Step 4: CGM
# ============================================================

feat_fused_tau = self.temporal_attn(
    feat_fused_0,
    feat_fused_N,
    flow_01,
    flow_10,
    tau,
    self.occ_threshold
)

# ============================================================
# Step 5: Gaussian parameter generation
# ============================================================

output = self._render_motion_gaussians(
    feat_fused_0,
    feat_fused_N,
    scale_h,
    scale_w,
    lr_h,
    lr_w,
    tau=tau,
    feat_fused_tau=feat_fused_tau,
)
```

---

# 6. 第三步：拆开 Gaussian 参数预测

不要再让：

```python
_predict_gaussian_params()
```

同时承担：

```text
color
offset
covariance
```

因为论文里：

```text
color / offset
```

和：

```text
covariance
```

是两条不同的数据流。

建议拆成：

```python
_predict_color_offset()
```

以及：

```python
_predict_endpoint_covariance()
```

---

# 7. Color / Offset 分支

中间时刻：

\[
F_\tau
\]

预测：

\[
C_{rgb,\tau}
\]

和：

\[
\Delta \mu_\tau
\]

---

## 7.1 推荐实现

```python
def _predict_color_offset(
    self,
    feat_ps,
    tau
):
    bs, _, ps_h, ps_w = feat_ps.shape
    n_gaussians = ps_h * ps_w

    tau_t = self._expand_tau(
        tau,
        bs,
        feat_ps.device
    )

    tau_channel = (
        tau_t
        .to(dtype=feat_ps.dtype)
        .view(bs, 1, 1, 1)
        .expand(
            -1,
            1,
            ps_h,
            ps_w
        )
    )

    param_feat = torch.cat(
        [
            feat_ps,
            tau_channel,
        ],
        dim=1,
    )

    # =========================================================
    # Color
    # =========================================================

    color_map = self.conv_color(
        param_feat
    )

    color = torch.sigmoid(
        color_map
        .permute(0, 2, 3, 1)
        .reshape(
            bs,
            n_gaussians,
            3
        )
        - 2.0
    )

    # =========================================================
    # Initial Offset
    # =========================================================

    offset_map = torch.tanh(
        self.conv_offset(
            param_feat
        )
    )

    # =========================================================
    # Adaptive Offset Window
    # =========================================================

    if (
        hasattr(self, 'current_w_map')
        and self.current_w_map is not None
    ):

        w_map = self.current_w_map

        if w_map.shape[-2:] != (ps_h, ps_w):
            w_map = F.interpolate(
                w_map,
                size=(ps_h, ps_w),
                mode='nearest',
            )

        offset_map = (
            offset_map * w_map
        )

    offset = (
        offset_map
        .permute(0, 2, 3, 1)
        .reshape(
            bs,
            n_gaussians,
            2
        )
    )

    return {
        'color': color,
        'offset': offset,
        'ps_h': ps_h,
        'ps_w': ps_w,
        'n_gaussians': n_gaussians,
    }
```

---

# 8. 第四步：AOW

AOW 不重新预测运动方向。

它负责：

> 根据当前区域的运动强弱，动态调整 Gaussian offset 可以移动的范围。

原始 offset：

\[
\widetilde{\Delta \mu_\tau}
\]

AOW：

\[
W_{\text{map}}
\]

最终：

\[
\Delta\mu_\tau
=
\widetilde{\Delta\mu_\tau}
\odot
W_{\text{map}}
\]

---

## 8.1 Window Bank

已有实现可以继续保留：

```python
window_sizes = torch.tensor(
    [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
    ]
)

self.register_buffer(
    'window_bank',
    window_sizes.view(
        1,
        -1,
        1,
        1
    )
)
```

然后：

```python
v_map = self.window_scorer(
    feat_fused_0,
    feat_fused_N
)

w_map = torch.sum(
    v_map * self.window_bank,
    dim=1,
    keepdim=True
)

self.current_w_map = w_map
```

---

# 9. 第五步：Covariance 分支

Covariance 不应该直接从：

```text
F_tau
```

回归。

而应该：

```text
F_0 → Sigma_0

F_N → Sigma_N

Sigma_0 + Sigma_N + tau
        ↓
       CRA
        ↓
     Sigma_tau
```

---

# 10. Covariance Prior Bank

你的 Gaussian prior bank 可以继续使用：

```python
cho1 = torch.tensor(
    [0, 0.41, 0.62, 0.98, 1.13, 1.29, 1.64, 1.85, 2.36]
)

cho2 = torch.tensor(
    [-0.86, -0.36, -0.16, 0.19, 0.34, 0.49, 0.84, 1.04, 1.54]
)

cho3 = torch.tensor(
    [0, 0.33, 0.53, 0.88, 1.03, 1.18, 1.53, 1.73, 2.23]
)

gau_dict = torch.tensor(
    list(
        product(
            cho1.tolist(),
            cho2.tolist(),
            cho3.tolist()
        )
    )
)

gau_dict = torch.cat(
    (
        gau_dict,
        torch.zeros(1, 3)
    ),
    dim=0
)

self.register_buffer(
    'gau_dict',
    gau_dict
)
```

最终：

```text
730 x 3
```

表示 730 个 covariance prior。

---

# 11. 端点 Covariance 预测

新增：

```python
def _predict_endpoint_covariance(
    self,
    feat_ps
):
    bs, _, ps_h, ps_w = feat_ps.shape

    n_gaussians = (
        ps_h * ps_w
    )

    para_feat = self.leaky_relu(
        feat_ps.detach()
    )

    para_conv = self.conv1(
        para_feat
    )

    para_flat = (
        para_conv
        .permute(0, 2, 3, 1)
        .reshape(
            bs * n_gaussians,
            512
        )
    )

    # =========================================================
    # Covariance Prior Bank embedding
    # =========================================================

    vector = self.mlp_vector(
        self.gau_dict
    )

    # [730, B*N]
    similarity = (
        vector
        @ para_flat.t()
    )

    weights = torch.softmax(
        similarity,
        dim=0
    )

    kernel_weights = (
        weights
        .t()
        .reshape(
            bs,
            n_gaussians,
            -1
        )
    )

    covariance = (
        weights.t()
        @ self.gau_dict
    ).reshape(
        bs,
        n_gaussians,
        3
    )

    return (
        covariance,
        kernel_weights,
    )
```

---

# 12. 第六步：修改 CRA

当前如果使用的是：

```python
[w_0, w_N, tau]
```

作为 CRA 输入，建议改为：

```python
[Sigma_0, Sigma_N, tau]
```

即：

```text
3 + 3 + 1 = 7
```

维输入。

---

## 12.1 修改初始化

删除或替换：

```python
self.cram_projection = nn.Sequential(
    nn.Linear(
        num_prior_kernels * 2 + 1,
        256
    ),
    nn.LeakyReLU(
        0.1,
        inplace=True
    ),
    nn.Linear(
        256,
        num_prior_kernels
    )
)
```

改成：

```python
self.cra_projection = nn.Linear(
    3 * 2 + 1,
    num_prior_kernels
)
```

即：

```python
self.cra_projection = nn.Linear(
    7,
    730
)
```

如果担心容量不足，也可以：

```python
self.cra_projection = nn.Sequential(
    nn.Linear(7, 64),
    nn.LeakyReLU(
        0.1,
        inplace=True
    ),
    nn.Linear(
        64,
        num_prior_kernels
    ),
)
```

如果目标是更贴近论文图中的：

```text
Concat
  ↓
Projection
```

则优先建议：

```python
nn.Linear(7, 730)
```

---

# 13. Low Dim Covariance Sample 到底是什么

这里不要理解成：

```text
random sampling
```

也不是：

```text
直接连续回归 sigma_x / sigma_y / sigma_xy
```

更合理的理解是：

```text
Sigma_0
Sigma_N
tau
 ↓
Projection
 ↓
生成低维 prior weight
 ↓
Softmax
 ↓
Covariance Prior Bank weighted sum
 ↓
Sigma_tau
```

也就是：

\[
E_\tau
=
P(
[
\Sigma_0,
\Sigma_N,
\tau
]
)
\]

然后：

\[
\alpha_\tau
=
Softmax(E_\tau)
\]

最终：

\[
\Sigma_\tau
=
\sum_{k=1}^{K}
\alpha_{\tau,k}
\Sigma_k^{prior}
\]

---

# 14. CRA 实现

新增：

```python
def _cra_resample_covariance(
    self,
    cov_0,
    cov_N,
    tau
):
    bs, n_gaussians, _ = cov_0.shape

    tau_t = self._expand_tau(
        tau,
        bs,
        cov_0.device
    )

    tau_expand = (
        tau_t
        .to(cov_0.dtype)
        .view(bs, 1, 1)
        .expand(
            bs,
            n_gaussians,
            1
        )
    )

    # =========================================================
    # [Sigma_0, Sigma_N, tau]
    # =========================================================

    cra_input = torch.cat(
        [
            cov_0,
            cov_N,
            tau_expand,
        ],
        dim=-1,
    )

    # =========================================================
    # Projection
    # =========================================================

    logits = self.cra_projection(
        cra_input
    )

    # =========================================================
    # Low Dim Covariance Sample
    # =========================================================

    weights = torch.softmax(
        logits,
        dim=-1,
    )

    # =========================================================
    # Sample / recombine covariance prior
    # =========================================================

    cov_tau = (
        weights
        @ self.gau_dict.to(
            dtype=weights.dtype
        )
    )

    return (
        cov_tau,
        weights,
    )
```

---

# 15. 第七步：最终重写 `_render_motion_gaussians`

这是整个修改的核心。

最终逻辑：

```text
F_tau
 ↓
color_tau / offset_tau
 ↓
AOW
 ↓
mu_tau

F_0 → Sigma_0
F_N → Sigma_N
 ↓
CRA
 ↓
Sigma_tau

{mu_tau, Sigma_tau, color_tau}
 ↓
render
```

---

## 15.1 推荐完整实现

```python
@torch.amp.custom_fwd(
    cast_inputs=torch.float32,
    device_type='cuda'
)
def _render_motion_gaussians(
    self,
    feat_fused_0,
    feat_fused_N,
    scale_h,
    scale_w,
    lr_h,
    lr_w,
    tau,
    feat_fused_tau,
):
    """
    Paper-aligned Gaussian parameter generation.

    color / offset:
        predicted directly from F_tau.

    covariance:
        Sigma_0, Sigma_N -> CRA -> Sigma_tau.

    Gaussian center:
        base grid + AOW-adjusted offset.

    Optical flow:
        already used to generate F_tau;
        DO NOT move xyz again here.
    """

    bs = feat_fused_tau.shape[0]
    device = feat_fused_tau.device

    H = round(
        lr_h * scale_h
    )

    W = round(
        lr_w * scale_w
    )

    tau_t = self._expand_tau(
        tau,
        bs,
        device
    )

    if (
        self.motion_force_dense
        or max(scale_h, scale_w) > 1.0
    ):
        dense_mode = 'full'
    else:
        dense_mode = self.motion_dense_mode

    # =========================================================
    # Branch A:
    # F_tau -> color_tau / offset_tau
    # =========================================================

    feat_ps_tau, dense_tau = (
        self._make_gaussian_features(
            feat_fused_tau,
            scale_h,
            scale_w,

            # use real tau
            tau=tau_t,

            dense_mode=dense_mode,
        )
    )

    params_tau = (
        self._predict_color_offset(
            feat_ps_tau,
            tau_t,
        )
    )

    ps_h = params_tau['ps_h']
    ps_w = params_tau['ps_w']

    color_tau = params_tau['color']
    offset_tau = params_tau['offset']

    # =========================================================
    # Gaussian center
    # mu_tau = grid + Delta mu_tau
    # =========================================================

    xyz_tau = self._base_xyz(
        offset_tau,
        ps_h,
        ps_w,
        H,
        W,
    )

    # =========================================================
    # Branch B:
    # F_0 / F_N -> Sigma_0 / Sigma_N
    # =========================================================

    feat_ps_0, dense_0 = (
        self._make_gaussian_features(
            feat_fused_0,
            scale_h,
            scale_w,
            tau=None,
            dense_mode=dense_mode,
        )
    )

    feat_ps_N, dense_N = (
        self._make_gaussian_features(
            feat_fused_N,
            scale_h,
            scale_w,
            tau=None,
            dense_mode=dense_mode,
        )
    )

    cov_0, _ = (
        self._predict_endpoint_covariance(
            feat_ps_0
        )
    )

    cov_N, _ = (
        self._predict_endpoint_covariance(
            feat_ps_N
        )
    )

    # =========================================================
    # CRA:
    # Sigma_0 + Sigma_N + tau -> Sigma_tau
    # =========================================================

    cov_tau, cra_weights = (
        self._cra_resample_covariance(
            cov_0,
            cov_N,
            tau_t,
        )
    )

    # =========================================================
    # Opacity
    # =========================================================

    opacity = torch.ones(
        bs,
        params_tau['n_gaussians'],
        1,
        device=device,
        dtype=color_tau.dtype,
    )

    # =========================================================
    # Render
    # =========================================================

    return self._render_gaussian_params(
        color_tau,
        cov_tau,
        xyz_tau,
        opacity,
        scale_h,
        scale_w,
        H,
        W,
        dense_tau or dense_0 or dense_N,
        debug_tag='motion',
    )
```

---

# 16. 需要删除的旧逻辑

下面这些内容从新的 `_render_motion_gaussians()` 中删除。

---

## 16.1 删除 endpoint Gaussian 参数预测

不要再：

```python
params0 = self._predict_gaussian_params(
    feat_ps_0,
    ...
)

paramsN = self._predict_gaussian_params(
    feat_ps_N,
    ...
)
```

然后融合：

```python
color = ...
offset = ...
```

Color / Offset 只由：

```python
feat_fused_tau
```

预测。

---

## 16.2 删除 Gaussian 坐标的 flow motion

删除：

```python
xyz_from_0 = self._apply_flow_motion(
    ...
)

xyz_from_N = self._apply_flow_motion(
    ...
)
```

以及：

```python
xyz = (
    pos_w0 * xyz_from_0
    + pos_wN * xyz_from_N
)
```

最终只保留：

```python
xyz_tau = self._base_xyz(
    offset_tau,
    ps_h,
    ps_w,
    H,
    W,
)
```

---

## 16.3 删除 confidence-based position merge

下面这些不要再用于 Gaussian 坐标：

```python
conf_0
conf_N
conf0_grid
confN_grid

pos_w0
pos_wN
```

如果将来想保留 confidence，可以用于：

```text
opacity
```

或：

```text
loss weighting
```

但不要再用于：

```text
xyz
```

---

# 17. 为什么不能再把 flow 加到 xyz

现在目标设计是：

```text
F_0 / F_N
    ↓
optical flow warping
    ↓
F_tau
    ↓
offset_tau
    ↓
xyz_tau
```

此时：

```python
offset_tau
```

本身已经是在：

```text
tau 时刻特征空间
```

预测的。

如果再：

```python
xyz_tau += tau * flow_01
```

等价于：

```text
第一次：
flow 把 feature 对齐到 tau

第二次：
flow 又把 Gaussian 坐标移动一次
```

存在 double motion compensation。

---

# 18. `_base_xyz()` 可以继续保留

原函数：

```python
def _base_xyz(
    self,
    offset_all,
    ps_h,
    ps_w,
    H,
    W,
):
    bs = offset_all.shape[0]

    coords = get_coord(
        ps_h,
        ps_w
    ).to(
        offset_all.device
    )

    coords = (
        coords
        .unsqueeze(0)
        .expand(
            bs,
            -1,
            -1
        )
    )

    xyz_x = (
        coords[:, :, 0:1]
        + 2 * offset_all[:, :, 0:1] / ps_w
        - 1 / W
    )

    xyz_y = (
        coords[:, :, 1:2]
        + 2 * offset_all[:, :, 1:2] / ps_h
        - 1 / H
    )

    return torch.cat(
        (
            xyz_x,
            xyz_y,
        ),
        dim=2,
    )
```

对应：

\[
x_\tau
=
x_{grid}
+
\frac{2\Delta x_\tau}{W_g}
\]

\[
y_\tau
=
y_{grid}
+
\frac{2\Delta y_\tau}{H_g}
\]

因此：

```text
grid
+
AOW-adjusted offset
=
Gaussian center at tau
```

---

# 19. `_make_gaussian_features()` 中 tau 怎么传

对于：

```text
F_tau
```

建议：

```python
tau=tau_t
```

而不是：

```python
rel_tau_target = torch.zeros_like(tau_t)
```

然后：

```python
tau=rel_tau_target
```

原因是你的目标明确是：

> 中间时刻 Gaussian 参数受到真实 temporal hyper-parameter \(\tau\) 条件控制。

所以：

```python
feat_ps_tau, dense_tau = self._make_gaussian_features(
    feat_fused_tau,
    scale_h,
    scale_w,
    tau=tau_t,
    dense_mode=dense_mode,
)
```

同时：

```python
params_tau = self._predict_color_offset(
    feat_ps_tau,
    tau_t,
)
```

---

# 20. Endpoint Covariance 是否注入 tau

不建议。

端点 covariance 是：

```text
Sigma_0
Sigma_N
```

它们代表 anchor 自身的 covariance。

因此：

```python
feat_ps_0 = self._make_gaussian_features(
    feat_fused_0,
    ...,
    tau=None
)
```

以及：

```python
feat_ps_N = self._make_gaussian_features(
    feat_fused_N,
    ...,
    tau=None
)
```

真正的 temporal condition 应该出现在：

```text
Sigma_0
Sigma_N
tau
 ↓
CRA
```

而不是先污染：

```text
Sigma_0
Sigma_N
```

本身。

---

# 21. 建议修改后的类结构

最后 `GaussianFusion` 内部推荐形成：

```text
GaussianFusion
│
├── encode()
│
├── temporal_attn / CGM
│
├── fusion
│
├── window_scorer / AOW
│
├── _make_gaussian_features()
│
├── _predict_color_offset()
│
├── _predict_endpoint_covariance()
│
├── _cra_resample_covariance()
│
├── _base_xyz()
│
└── _render_gaussian_params()
```

其中：

```text
_predict_gaussian_params()
```

可以逐渐废弃。

---

# 22. 最终参数来源表

| Gaussian 参数 | 来源 | 是否依赖 \(\tau\) | 是否使用 Flow |
|---|---|---:|---:|
| Color \(C_{rgb,\tau}\) | \(F_\tau\) | 是 | 间接 |
| Offset \(\Delta\mu_\tau\) | \(F_\tau\) | 是 | 间接 |
| AOW \(W_{map}\) | \(F_0,F_N\) | 否 / pair shared | 否 |
| Center \(\mu_\tau\) | grid + offset | 是 | 不直接使用 |
| \(\Sigma_0\) | \(F_0 + CPB\) | 否 | 否 |
| \(\Sigma_N\) | \(F_N + CPB\) | 否 | 否 |
| \(\Sigma_\tau\) | CRA(\(\Sigma_0,\Sigma_N,\tau\)) | 是 | 否 |
| Opacity | 当前可固定为 1 | 否 | 否 |

---

# 23. 最终数学表示

## CGM

\[
F_{0\rightarrow\tau}
=
Warp(
F_0,
\tau M_{0\rightarrow N}
)
\]

\[
F_{N\rightarrow\tau}
=
Warp(
F_N,
(1-\tau)M_{N\rightarrow0}
)
\]

\[
F_\tau
=
M\odot F_{0\rightarrow\tau}
+
(1-M)\odot F_{N\rightarrow\tau}
+
\Delta F
\]

---

## Color

\[
C_{rgb,\tau}
=
H_c(
F_\tau,
\tau
)
\]

---

## Initial Offset

\[
\widetilde{\Delta\mu_\tau}
=
\tanh(
H_\mu(
F_\tau,
\tau
)
)
\]

---

## AOW

\[
V_{map}
=
SIREN(
F_0,
F_N
)
\]

\[
W_{map}
=
\sum_k
V_{map}^{(k)}
s_k
\]

\[
\Delta\mu_\tau
=
\widetilde{\Delta\mu_\tau}
\odot
W_{map}
\]

---

## Gaussian Center

\[
\mu_\tau
=
\mu_{grid}
+
\Delta\mu_\tau
\]

---

## Endpoint Covariance

\[
w_0
=
Softmax(
MLP(CPB)
\cdot
F_0
)
\]

\[
\Sigma_0
=
\sum_k
w_{0,k}
\Sigma_k^{prior}
\]

以及：

\[
w_N
=
Softmax(
MLP(CPB)
\cdot
F_N
)
\]

\[
\Sigma_N
=
\sum_k
w_{N,k}
\Sigma_k^{prior}
\]

---

## CRA

\[
E_\tau
=
Projection(
[
\Sigma_0,
\Sigma_N,
\tau
]
)
\]

\[
w_\tau
=
Softmax(
E_\tau
)
\]

\[
\Sigma_\tau
=
\sum_k
w_{\tau,k}
\Sigma_k^{prior}
\]

---

# 24. 最终渲染参数

最终每个 Gaussian：

\[
G_{i,\tau}
=
\{
\mu_{i,\tau},
\Sigma_{i,\tau},
C_{rgb,i,\tau},
\alpha_i
\}
\]

其中：

\[
\mu_{i,\tau}
=
p_i
+
\Delta\mu_{i,\tau}
\]

最终：

```python
return self._render_gaussian_params(
    color_tau,
    cov_tau,
    xyz_tau,
    opacity,
    scale_h,
    scale_w,
    H,
    W,
    dense_tau or dense_0 or dense_N,
    debug_tag='motion',
)
```

---

# 25. 推荐修改顺序

不要一次性全部改。

建议按照下面的顺序：

```text
Step 1
先改 _render_motion_gaussians
去掉 xyz 上的二次 flow motion
```

↓

```text
Step 2
拆分：
_predict_color_offset
_predict_endpoint_covariance
```

↓

```text
Step 3
让 color / offset 只来自 F_tau
```

↓

```text
Step 4
改 CRA：
Sigma_0 + Sigma_N + tau
→ Sigma_tau
```

↓

```text
Step 5
确认 AOW 只作用于 offset
```

↓

```text
Step 6
最后再替换 temporal_attention
为 paper-aligned CGM
```

这样方便逐步检查：

- loss 是否正常；
- Gaussian coverage 是否异常；
- covariance 是否崩掉；
- offset 是否过大；
- AOW 是否饱和；
- 中间帧是否出现 ghosting。

---

# 26. 推荐加入的 Debug 信息

训练时建议记录：

```python
self._temporal_stats[
    'tau_mean'
] = tau_t.mean().item()

self._temporal_stats[
    'offset_abs_mean'
] = offset_tau.abs().mean().item()

self._temporal_stats[
    'offset_abs_max'
] = offset_tau.abs().max().item()

self._temporal_stats[
    'aow_mean'
] = self.current_w_map.mean().item()

self._temporal_stats[
    'aow_max'
] = self.current_w_map.max().item()

self._temporal_stats[
    'cov0_mean'
] = cov_0.mean().item()

self._temporal_stats[
    'covN_mean'
] = cov_N.mean().item()

self._temporal_stats[
    'cov_tau_mean'
] = cov_tau.mean().item()

self._temporal_stats[
    'cra_entropy'
] = (
        -cra_weights
        * torch.log(
            cra_weights.clamp_min(1e-8)
        )
    ).sum(
        dim=-1
    ).mean().item()
```

---

# 27. 重点检查 AOW 是否失控

因为：

```python
offset_map = tanh(...)
```

范围：

```text
[-1, 1]
```

然后：

```python
window_bank = [1,2,...,10]
```

最终 offset 范围理论上可以达到：

```text
[-10, 10]
```

LR pixels。

所以建议观察：

```text
offset_abs_mean
offset_abs_max
aow_mean
aow_max
```

如果训练初期：

```text
aow_mean ≈ 9~10
```

说明 window scorer 可能直接倾向最大窗口。

可以考虑加入：

```python
window_reg = (
    self.current_w_map - 1.0
).abs().mean()
```

作为轻微正则。

---

# 28. CRA 也建议看 entropy

如果：

```text
CRA weight entropy → 0
```

说明所有位置都 one-hot 选择某一个 covariance prior。

这不一定错误，但如果过早出现，可能说明：

```text
CRA collapse
```

如果：

```text
CRA entropy 一直接近最大值
```

说明：

```text
所有 prior 权重差不多
```

CRA 没有学到有效选择。

---

# 29. 最终应删除/废弃的旧接口

如果新的架构稳定，可以删除：

```python
_apply_flow_motion()
```

至少 motion rendering 不再需要。

下面这几个 motion 相关配置也可能逐渐不再需要：

```python
motion_pair_merge
fb_confidence_scale
fb_confidence_floor
tau_gaussian_opacity
```

其中：

```text
forward-backward confidence
```

如果只为旧的 Gaussian pair merge 服务，也可以一起删除。

---

# 30. 最终一句话总结

修改后的模型应该严格遵循：

```text
Optical Flow
    ↓
构造 F_tau
    ↓
F_tau 直接预测
color_tau + offset_tau
    ↓
AOW 调整 offset 范围
    ↓
mu_tau = grid + offset_tau
```

与此同时：

```text
F_0 / F_N
    ↓
CPB
    ↓
Sigma_0 / Sigma_N
    ↓
Concat + tau
    ↓
CRA Projection
    ↓
Low Dim Covariance Sample
    ↓
Sigma_tau
```

最终：

```text
{
    color_tau,
    mu_tau,
    Sigma_tau
}
    ↓
Gaussian Splatting
```

最关键的三个原则：

1. **Flow 只用于构造中间时刻特征，不再直接移动最终 Gaussian center。**
2. **Color / Offset 只由中间 \(\tau\) 特征预测。**
3. **Covariance 由两个 anchor covariance 通过 CRA + Covariance Prior Bank 重采样得到。**

---

## Reference

- Paper: `https://arxiv.org/pdf/2604.18047`
- Repository: `https://github.com/Yukarizz/GaussianFusion`
- Recommended base branch: `feature/regression_color_offset`

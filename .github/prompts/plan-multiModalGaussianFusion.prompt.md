# Plan: Multi-Modal Gaussian Fusion (任意时空尺度)

**TL;DR**: 将 ContinuousSR 的 2D 高斯溅射范式扩展到多模态（可见光+红外）视频融合，支持任意空间分辨率和任意时间位置输出。SpyNet 提供帧间运动估计；跨模态特征融合后输入高斯预测器，在目标尺度下光栅化。

## 架构总览

```
锚帧 frame[i]:     Vis_i, IR_i ────────┐
                                        ├── [SpyNet] ──→ Flow 双向 (大间隔)
锚帧 frame[i+N]:   Vis_{i+N}, IR_{i+N} ┘
               │
     [双分支 EDSR Encoder] → Feat_vis, Feat_ir (各64ch)
               │
     [双向 Flow Warp → 时间 τ=k/N] (线性插值 + 遮挡感知)
               │
     [Cross-Modal Fusion] (通道拼接 + Conv + SE attention)
               │
     [Gaussian Predictor] → Color, Covariance, Offset
               │
     [2D Gaussian Rasterization @ (scale_h, scale_w)]
               │
         融合 HR 输出 (时刻 τ=k/N, 分辨率 s×)
               │
     [Loss vs GT frame[i+k]] ← vis_gt, ir_gt (真实中间帧监督)
```

---

## Steps

### Phase 1: SpyNet 集成

1. **新建 `models/spynet.py`** — 移植 PyTorch SpyNet（`sniklaus/pytorch-spynet`）。6层空间金字塔，每层5个Conv2d(7×7)。输入8ch（RGB对+上采样flow），输出2ch残差flow。加载预训练权重 `sintel-final`。注册为 `@register('spynet')`。

2. **新建 `utils/warp.py`** — 实现：
   - `flow_warp(feat, flow)` 基于 `F.grid_sample`
   - `temporal_interpolate_flow(flow_01, flow_10, t)` 线性组合双向flow：$F_{0\to\tau} = \tau \cdot F_{0\to1}$，$F_{1\to\tau} = (1-\tau) \cdot F_{1\to0}$
   - `compute_occlusion_mask(flow_01, flow_10)` forward-backward consistency check，判断遮挡区域
   - `temporal_blend(feat_0, feat_1, flow_01, flow_10, t)` 带遮挡感知的双向加权融合：
     ```python
     def temporal_blend(feat_0, feat_1, flow_01, flow_10, t):
         flow_0t = t * flow_01
         flow_t0 = (1 - t) * flow_10
         warped_0 = flow_warp(feat_0, flow_0t)   # t-1 → τ
         warped_1 = flow_warp(feat_1, flow_t0)   # t → τ
         # Occlusion mask via forward-backward consistency
         flow_back = flow_warp(flow_10, t * flow_01)
         occ = (torch.norm(flow_01 + flow_back, dim=1, keepdim=True) < thr).float()
         w0 = (1 - t) * occ
         w1 = t * (1 - occ) + t * occ
         return (w0 * warped_0 + w1 * warped_1) / (w0 + w1 + 1e-8)
     ```

### Phase 2: 双模态编码器

3. **修改编码器** — 两个独立输入头（`conv 3→64` for vis, `conv 3→64` for IR），共享 EDSR ResBlock body。输出各自64维特征。

4. **时序特征对齐** — 用SpyNet光流将t-1帧特征warp到时刻τ，将t帧特征也warp到τ。采用带遮挡感知的双向加权融合（`temporal_blend`）：
   - 线性flow插值：$F_{0\to\tau} = \tau \cdot F_{0\to1}$，$F_{1\to\tau} = (1-\tau) \cdot F_{1\to0}$
   - 遮挡检测：forward-backward consistency check（$\|F_{0\to1} + \text{warp}(F_{1\to0}, F_{0\to\tau})\| < \theta$）
   - 加权融合：非遮挡区域按 $(1-\tau):\tau$ 加权，遮挡区域偏信未被遮挡方向
   - 融合后的时序特征 `[feat_t_warped, feat_{t-1}_warped]` → Conv压缩

   **时间超分机制**（训练时有真实GT监督）：
   - 训练输入：锚帧 frame[i] 和 frame[i+N]，间隔 N ∈ [2, 16]
   - 训练目标：预测中间帧 frame[i+k]，τ = k/N ∈ (0, 1)
   - 测试时：指定任意 τ ∈ (0, 1) 输出对应中间时刻的融合帧
   - 等效时间超分倍率 = N（如 N=10 相当于 10× 时间超分）

### Phase 3: 跨模态融合模块

5. **新建 `models/fusion.py`** — 轻量跨模态注意力：拼接 vis+IR 特征 `[B, 128, H, W]` → Conv → SE Channel Attention → 输出 `[B, 64, H, W]`。

### Phase 4: 高斯预测与光栅化

6. **新建 `models/gaussian_fusion.py`** — 核心模型 `@register('gaussian-fusion')`，结构：
   - `self.spynet`：冻结的SpyNet
   - `self.head_vis / self.head_ir`：双模态输入头
   - `self.encoder_body`：共享EDSR body
   - `self.fusion`：跨模态融合
   - `self.temporal_fusion`：时序特征合并
   - `self.pixel_unshuffle + mlp_color + mlp_offset + mlp_vector + conv1 + gau_dict`：复用ContinuousSR的高斯预测逻辑

7. **Forward pass 伪代码**：
   ```python
   forward(vis_0, ir_0, vis_N, ir_N, scale, tau):
       # vis_0/ir_0 = anchor frame[i]
       # vis_N/ir_N = anchor frame[i+N]
       # tau = k/N ∈ (0, 1), scale = (s_h, s_w)
       flow_fwd = spynet(vis_0, vis_N)     # frame[i] → frame[i+N]
       flow_bwd = spynet(vis_N, vis_0)     # frame[i+N] → frame[i]
       feat_vis_0, feat_ir_0 = encode(vis_0, ir_0)
       feat_vis_N, feat_ir_N = encode(vis_N, ir_N)
       # Warp features to time τ
       feat_vis_τ = temporal_blend(feat_vis_0, feat_vis_N, flow_fwd, flow_bwd, tau)
       feat_ir_τ  = temporal_blend(feat_ir_0, feat_ir_N, flow_fwd, flow_bwd, tau)
       # Fuse modalities
       feat_fused = fusion(feat_vis_τ, feat_ir_τ)
       # Gaussian predict + rasterize at target scale
       return query_output(feat_fused, scale)  # → [B, 3, H*s, W*s]
   ```

### Phase 5: 数据与训练

8. **新建 Dataset `M3SVDTemporalFusion`** — 现有 `M3SVDVideoPaired` 仅取相邻帧(t-1, t)且无scale/τ参数，不满足需求。需要全新的 dataset 类：

   **现有 dataset 的不足**：
   - 固定间隔=1（相邻帧），无法学习大间隔时间插值
   - 无 τ 参数输出
   - 无 scale 参数（不涉及空间超分）
   - 训练集只有 `visible_Enhance` + `infrared_Enhance`（无退化版本）

   **新 dataset 设计** — 注册为 `@register('m3svd-temporal-fusion')`：

   **数据统计**：训练集 ~175 个视频序列，每序列 795~1160+ 帧。

   **采样策略**：大间隔锚帧 + 中间帧作为 GT
   ```
   每个训练样本:
     1. 随机选一个视频 V
     2. 随机选间隔 N ∈ [2, 16] (对应时间超分倍率上限)
     3. 随机选起始帧 i ∈ [1, len(V) - N]
     4. 锚帧对: frame[i], frame[i+N]  ← 模型输入
     5. 随机选中间帧 k ∈ [1, N-1]
     6. GT 帧: frame[i+k]            ← 监督目标
     7. 时间位置: τ = k / N           ← 连续值, ∈ (0, 1)
   ```

   **τ 连续性保证** — N 和 k 都是随机的，τ=k/N 自动覆盖 (0,1) 上的密集值：
   | N | k | τ | 含义 |
   |---|---|---|------|
   | 10 | 3 | 0.30 | 10× 时间超分，预测第3帧 |
   | 7 | 2 | 0.286 | 7× 时间超分，预测第2帧 |
   | 15 | 11 | 0.733 | 15× 时间超分，预测第11帧 |
   | 2 | 1 | 0.50 | 2× 时间超分（最简单） |
   | 16 | 1 | 0.0625 | 极端边缘时刻 |

   **空间维度**：同时随机采样 scale ∈ [1, 4]。

   **训练集可用模态**：仅 `visible_Enhance` + `infrared_Enhance`（都是干净帧）。

   **输出 dict**：
   ```python
   {
       'vis_anchor0': frame[i] vis_Enhance,      # [3, H, W] — 模型输入
       'ir_anchor0':  frame[i] ir_Enhance,       # [3, H, W] — 模型输入
       'vis_anchor1': frame[i+N] vis_Enhance,    # [3, H, W] — 模型输入
       'ir_anchor1':  frame[i+N] ir_Enhance,     # [3, H, W] — 模型输入
       'vis_gt':      frame[i+k] vis_Enhance,    # [3, H, W] — GT
       'ir_gt':       frame[i+k] ir_Enhance,     # [3, H, W] — GT
       'tau':         k / N,                     # float ∈ (0, 1)
       'scale':       (s_h, s_w),                # spatial scale
       'video_id':    str,
       'frame_info':  (i, i+N, i+k, N),          # for debugging
   }
   ```

   **Patch cropping**：在 `__getitem__` 中对所有帧做相同的随机裁剪 (64×64 patch)。

9. **训练配置** — Loss（中间帧 GT 监督 + 融合约束）：
   ```
   # 当 scale=1 时，直接在原始分辨率计算 loss：
   L_temporal = L1(fused, vis_gt) + L1(fused, ir_gt)    # 时间插值精度
   L_fusion   = SSIM(fused, vis_gt) + SSIM(fused, ir_gt) # 结构保持
   L_max      = MSE(fused, max(vis_gt, ir_gt))           # 热辐射保留
   L_total    = λ₁·L_temporal + λ₂·L_fusion + λ₃·L_max

   # 当 scale>1 时，将 fused_HR 双线性下采样回 LR 再与 GT 比较：
   L_total    = λ₁·L1(↓fused, vis_gt) + λ₂·SSIM(↓fused, vis_gt) + ...
   ```
   - Adam lr=1e-4，SpyNet 冻结，batch=4-8，patch=64×64
   - 训练策略：先 scale=1 训练时间插值+融合能力，再加入 scale>1 联合训练

10. **新建 `train.py`** — 标准 PyTorch 训练循环。

### Phase 6: 评估

11. **测试脚本** — 支持融合指标：MI（互信息）、Qabf、VIF、EN（信息熵）、SD（标准差）。

12. **Demo 脚本** — `demo_fusion.py` 接受双模态输入 + scale + t。

---

## Relevant Files

| 文件 | 作用 |
|------|------|
| `models/gaussian.py` | 参考实现：`query_output()` 中的高斯字典构建、协方差加权、光栅化管线 |
| `models/edsr.py` | 编码器骨架，适配双模态输入头 |
| `models/mlp.py` | Color/Offset/Vector MLP，直接复用 |
| `models/models.py` | 模型注册 `register()` / `make()` |
| `datasets/m3svd_video.py` | 已有 M3SVD 数据加载器，扩展训练模式 |
| `utils.py` | 现有工具函数 |

## 需新建的文件

- `models/spynet.py` — SpyNet 光流网络
- `models/fusion.py` — 跨模态融合模块
- `models/gaussian_fusion.py` — 主模型
- `utils/warp.py` — 光流warp工具
- `datasets/m3svd_temporal_fusion.py` — 新 dataset（大间隔锚帧采样）
- `train.py` — 训练脚本
- `configs/train/train-m3svd-fusion.yaml` — 训练配置
- `demo_fusion.py` — 融合演示

---

## Decisions

- **融合监督**: 有GT的半监督策略（中间帧作为参考）：
  - **时间精度**: L1(fused, vis_gt) + L1(fused, ir_gt)，保证时间插值准确
  - **结构保持**: SSIM(fused, vis_gt) + SSIM(fused, ir_gt)
  - **热辐射保留**: MSE(fused, max(vis_gt, ir_gt))，逐像素取最大值作为soft target
  - 总损失: `L = λ₁·L1 + λ₂·SSIM + λ₃·MSE_max`
- **红外输入保持 3 通道**
- **SpyNet仅在可见光帧上估计flow**（红外特征差异大，可见光纹理更适合光流）
- **时间插值采用线性flow近似**
- **高斯字典复用 ContinuousSR 的730模板**

---

## Verification

1. SpyNet在M3SVD连续帧上输出flow可视化，确认运动合理
2. 双分支encoder输出shape正确 `[B, 64, H, W]`
3. Fusion模块 `[B, 128, H, W]` → `[B, 64, H, W]`
4. 端到端forward：dummy数据（2帧×2模态 + scale + t）→ 输出shape正确，无NaN
5. 训练1000步loss下降
6. 测试集生成融合结果，计算MI/Qabf/EN指标

---

## Further Considerations

1. **SpyNet训练策略**: 初始冻结；若flow质量不足，fine-tune最后2层。
2. **时间超分上限**: 线性flow近似在大运动/非匀速场景下退化。后续可扩展为：
   - 二次flow模型（引入加速度项）
   - 轻量 flow refinement 网络（学习残差修正中间时刻flow）
   - 多帧输入（>2帧）提供更丰富的时序约束

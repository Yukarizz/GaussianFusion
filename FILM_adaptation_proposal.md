# GaussianFusion × FILM：大尺度运动改进方案

论文：FILM: Frame Interpolation for Large Motion (ECCV 2022, Google Research)
仓库：google-research/frame-interpolation（已抓取到 `.workbuddy/film_ref/`）
对照代码版本：`feature/regression_color_offset` @ `3cc02ff`

---

## 0. 先纠正一个预期

你提到想用 Gram 损失是因为「大尺度运动下插值效果差」。这里需要分清 FILM 两个贡献的职责：

| FILM 组件 | 真正解决的问题 | 对你症状的效果 |
|---|---|---|
| **多尺度共享 + task-oriented flow** | 大位移本身的运动估计 | ✅ **对症** |
| **Gram (Style) 损失** | 大位移导致的**遮挡空洞修补** + L1 固有的模糊 | ⚠️ 只治画质，不治运动 |

论文正文 Section 3.1 说得很直白：`L1` 出来的帧 "often blurry"；Gram 的作用是让大遮挡区域被「脑补」出清晰纹理。Figure 7 的三档对比是 L1 → +VGG → +Style，肉眼可见的改善是**锐度**，不是**运动对齐**。

所以：**Gram 会让你 30 分的效果在视觉上看起来像 50 分，但运动错位的结构性问题还得靠多尺度。** 两者互补，建议都做，但优先级按治因排序。

---

## 1. 根因诊断

### 根因 1（核心）：offset 是"一步跳跃"回归，没有 coarse-to-fine

`models/gaussian_fusion.py:938-547`

```python
offset_map = torch.tanh(self.conv_offset(param_feat))   # [-1, 1]
offset_map = offset_map * w_map                          # * AOW 窗口
```

`window_bank = [1..10]`（第 314 行），`motion_factor` clamp 到 18，所以单个高斯核理论上能覆盖 **±180px**。看起来够用，但学习上极难：

> 一个 tanh 头要**一次性**回归出 180px 的位移，等价于让网络在一个尺度上同时学会「小运动的精细流淌」和「大运动的整体跳跃」。

FILM 的解法是把这个回归拆成 L 层：粗层只负责 ±几 px 的残差，逐级上采样累加。论文的直觉就是原话：

> "large motion at finer scales should be the same as small motion at coarser scales"

也就是把**一次难回归**变成**多次易回归**。这正是你现在缺的。

### 根因 2：`motion_factor` 是 `no_grad` 的死先验

`models/gaussian_fusion.py:924-936`

```python
with torch.no_grad():
    flow_mag = max(|flow_01|, |flow_10|)
    motion_factor = (1.0 + gain * flow_mag / flow_ref).clamp(1.0, max)
    w_map = w_map * motion_factor
```

这段完全不可学习。它给了一个「运动大 → 窗口放大」的启发式，但网络无法反向传播去纠正这个先验的错误。这也正是你「用预训练光流学残差」想法的落点——只是当前连可学习的残差入口都没有。

### 根因 3：AOW 正则与大运动目标相互打架

`models/gaussian_fusion.py:997-1008`：`get_aow_regularization = |w_map - 1.0|.mean()`，权重 `lambda_aow=0.01`。

它的目的是抑制 SIREN 坍缩到最大窗口。副作用是：**大运动区域恰恰最需要大窗口，而这个正则正把它往 1.0 拉。** 在 `n_max=12` 的训练配置下，这个冲突会被放大。

建议：改成 **单边惩罚 + 运动条件化**——只惩罚「流量小却用了大窗口」，流量大时放行。

### 根因 4：损失全是像素级，天生出模糊

`train.py:170-172`：

```python
total = λ_int * L1(Y) + λ_color * L1(CbCr) + λ_grad * L1(grad)
```

全 L1 线性 supermodel，无一 perceptual 项。这是 Figure 7 最左列的状态。

---

## 2. 方案 A：Gram / Style 损失

### 2.1 实现（PyTorch，可直接落 `losses.py`）

VGG19 层索引（`torchvision.models.vgg19(pretrained=True).features`）：

```python
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights

VGG_LAYERS = {'relu1_2': 3, 'relu2_2': 8, 'relu3_2': 17, 'relu4_2': 26, 'relu5_2': 35}

class GramLoss(nn.Module):
    """FILM Style loss: L2 distance between VGG feature Gram matrices."""
    _LAYERS = ('relu1_2', 'relu2_2', 'relu3_2', 'relu4_2', 'relu5_2')
    # FILM 论文附录 A.1 / vgg19_loss.py 默认值
    _WEIGHTS = (1/2.6, 1/4.8, 1/3.7, 1/5.6, 10/1.5)

    def __init__(self, layers=_LAYERS, weights=_WEIGHTS, resize_input=True):
        super().__init__()
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        idx = [VGG_LAYERS[l] for l in layers]
        self.slices = nn.ModuleList()
        prev, to_freeze = 0, []
        for i in idx:
            self.slices.append(nn.Sequential(*[vgg[j] for j in range(prev, i + 1)]))
            prev = i + 1
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()
        self.register_buffer('weights', torch.tensor(weights), persistent=False)
        self.mean = nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1), requires_grad=False)
        self.std  = nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1), requires_grad=False)

    @staticmethod
    def _gram(x):
        b, c, h, w = x.shape
        f = x.reshape(b, c, h * w)
        return torch.bmm(f, f.transpose(1, 2)) / (h * w)   # [B, C, C]

    def forward(self, pred, ref):
        # pred/ref: RGB in [0,1], VGG expects ImageNet-normalized
        pred = (pred - self.mean.to(pred.dtype)) / self.std.to(pred.dtype)
        ref  = (ref  - self.mean.to(ref.dtype))  / self.std.to(ref.dtype)
        loss = 0.0
        for i, sl in enumerate(self.slices):
            gp = self._gram(sl(pred))
            gr = self._gram(sl(ref))
            loss = loss + F.mse_loss(gp, gr) * self.weights[i]
        return loss
```

### 2.2 ⚠️ 最关键的坑：reference 必须和 L1 的 target 完全一致

你的 GT 不是原始可见光帧，而是**融合目标**（`train.py:146`）：

```python
target_int  = torch.max(vis_y, ir_int)     # Y: 取两者最大
target_cbcr = vis_cbcr                     # CbCr: 完全取自可见光
```

如果 Gram 的 reference 用 `vis_gt` 而 L1 的 target 用 `max(vis_y, ir_y)`，**两个损失会互相拉扯**，红外高亮区域会被 Gram 判为「纹理不匹配」而被抹掉。

必须配套 SSD rate：

```python
# train.py 里构造 reference
target_ycbcr = torch.cat([vis_ycbcr[:, 0:1].max(ir_ycbcr[:, 0:1]), vis_cbcr], dim=1)
target_rgb = kornia.color.ycbcr_to_rgb(target_ycbcr)

loss_gram = criterion_gram(fused, target_rgb.detach())
```

### 2.3 权重调度：不要一上来就加

FILM 附录 A.1（`film_net-Style.gin:57-60`）的**原版 schedule**：

| loss | 0 ~ 1.5M steps | 1.5M ~ 3M steps |
|---|---|---|
| L1 | 1.0 | 1.0 |
| VGG | 1.0 | **0.25** |
| **Gram** | **0.0** | **40.0** |

**前一半训练完全关闭 Gram**。原因是 Gram 是全图二阶统计量，早期网络输出噪声大时梯度方向不稳定。

落到你的 30 epoch：

```yaml
# epoch < 15: 完全关闭
lambda_gram: 0.0
lambda_vgg:  0.0
# epoch >= 15: 开启（在 train.py 里按 epoch 切）
lambda_vgg:  0.25
lambda_gram: 40.0
```

**40.0 不要照抄**。FILM 选这个值的原则是「让各项损失贡献相当」（附录原话）。你的 L1 是 Y 通道 + 梯度，量级必然不同。做法：**第 15 epoch 起，先跑 100 个 batch 打印 `loss_int / loss_gram` 的比值，把 w_gram 调到两者接近 1:1**，再固定。

### 2.4 成本

- 显存：额外一份冻结 VGG19（~550MB fp32），fp16 部署可减半
- 速度：约 **+15~25%** iteration 时间
- 收益预期：PSNR 可能**微降**（-0.2~0.5dB，见 FILM Table 1/2 中 L1 vs LS 那几行），但锐度和遮挡区域主观质量明显更好。这是论文自己承认的 perception-distortion tradeoff。

> 如果你最终要报 PSNR 指标，请把 Gram 版作为**独立实验**，别顶替 L1 版。

---

## 3. 方案 C：多尺度共享 offset head ← 真正治因

### 3.1 先算清楚要几层

FILM 在 `options.py:31-34` 给了硬公式：

```
max_motion_px = 2^(pyramid_levels - 1) × flow_convs[-1]
```

并且 `pyramid_levels=5, flow_convs=4` 的默认值「suitable for up to 64 pixel motions」。
论文模型用 `7 层 × 3 conv = 2^6 × 3 = 192px`。

**⚠️ 尺寸整除约束**（同一段文档）：输入 H/W 必须能被 `2^(L-1)` 整除。

你的 `patch_size: 224`，注意 `224 = 32×7`，**不能被 64 整除**：

| 层数 L | 整除要求 | patch 224 | patch 256 | 可解析上限（conv=3） |
|---|---|---|---|---|
| 6 | 32 | ✅ | ✅ | 2^5×3 = 96px |
| 7 | 64 | ❌ | ✅ | 2^6×3 = 192px |

**建议起步：L=6，patch 224 原样不动**，上限 96px，够覆盖 M3SVD 大多数情况；验证有效后想再扩就必须把 patch 换成 256 或 192。

### 3.2 改法：把单次 offset 拆成逐级累加残差

在 `models/gaussian_fusion.py:938-547` 这一段替换。核心骨架：

```python
class SharedOffsetHead(nn.Module):
    """单一实例，被 level>=specialized_levels 的所有层复用 => 权重共享。"""
    def __init__(self, feat_ch, hidden=64, n_convs=3):
        super().__init__()
        convs = []
        ch = feat_ch + 2 + 1          # 特征 + 上采样的粗 offset(2) + tau(1)
        for i in range(n_convs):
            convs += [nn.Conv2d(ch if i == 0 else hidden, hidden, 3, 1, 1),
                      nn.LeakyReLU(0.2, inplace=True)]
            ch = hidden
        convs.append(nn.Conv2d(hidden, 2, 3, 1, 1))   # 最后一层无激活（FILM 明确说不要用 tanh/sigmoid 限幅）
        self.net = nn.Sequential(*convs)

    def forward(self, feat, coarse_offset, tau):
        x = torch.cat([feat, coarse_offset, tau], dim=1)
        return self.net(x)


class MultiScaleOffset(nn.Module):
    def __init__(self, feat_ch, levels=6, specialized=2, n_convs=3):
        super().__init__()
        self.levels, self.specialized = levels, specialized
        # 最细的 specialized 层各自独立（细层语义差异大，共享会掉点）
        self.heads = nn.ModuleList(
            [SharedOffsetHead(feat_ch, n_convs=n_convs) for _ in range(specialized)])
        # 其余所有粗层共用同一个实例 —— 这就是 FILM 的 shared_predictor
        shared = SharedOffsetHead(feat_ch, n_convs=n_convs)
        for _ in range(specialized, levels):
            self.heads.append(shared)

    def forward(self, feat_pyr, tau):
        """feat_pyr: 细->粗。返回累计后的 offset 金字塔（细->粗）。"""
        L = len(feat_pyr)
        # 从最粗层开始：它是 DC 项，不是残差
        off = torch.zeros_like(feat_pyr[-1][:, :2])
        offs = [None] * L
        for li in range(L - 1, -1, -1):
            feat = feat_pyr[li]
            # 上采样粗 offset，且位移量 ×2（分辨率翻倍，流也要翻倍）
            coarse = F.interpolate(off * 2.0, size=feat.shape[-2:],
                                   mode='bilinear', align_corners=False) \
                     if li != L - 1 else torch.zeros(feat.size(0), 2,
                                                     *feat.shape[-2:], device=feat.device)
            off = self.heads[li](feat, coarse, tau) + coarse
            offs[li] = off
        return offs          # 细->粗
```

对应 FILM 原版 `pyramid_flow_estimator.py:149-163`，注意两个容易写错的细节：

1. **粗 offset 上采后必须 ×2**（第 155 行 `2*v`）——分辨率翻倍，位移量也要翻倍，否则逐级累加会严重欠估计。
2. **最后一层 conv 无激活**（第 81-83 行注释）：FILM 明确测试过 sigmoid 限幅，结果更差。

### 3.3 配套改造

- **AOW 窗口**：保留 `w_map` 作为**细层的额外调制**，但粗层不要乘（粗层天然覆盖范围大）。
- **AOW 正则**：改成单边 + 运动条件化（见根因 3）：
  ```python
  w = w_map_raw
  small_flow = (flow_mag < self.motion_window_flow_ref).float()
  loss_aow = (F.relu(w - 1.0) * small_flow).sum() / (small_flow.sum() + 1e-8)
  ```
- **积分位置**：`_render_motion_gaussians` 里 `xyz = base + offset`，把 offset 换成最细层结果即可，其余层仅作为梯度通路。

### 3.4 预期

FILM Table 3 的消融很值得参考：完整模型（64 filters）**不共享时训练不收敛（标注 N/A）**。说明共享不只是省参数，更是稳定性前提。你这边的收益预期：

- 大位移（>40px 区域）的运动对齐误差显著下降
- 由于改善了「周期性纹理 + 大位移」的经典错配，主观 ghosting 会减少
- 参数量增加很小（共享），但**显存会因存储金字塔特征上升约 20-30%**

---

## 4. 方案 B：预训练光流 + 可学习残差

### 4.1 一个你代码里真实存在的 gap

现在 `flow_model=searaft` 输出的是 **I0→I1 的常规光流**（`models/searaft.py`）。但帧插需要的是 **It→I0 / It→I1 的 task-oriented flow**——注意 FILM 论文第 5 页：

> "in contrast to other methods, we directly predict task oriented flows, Wt→0 and Wt→1, **from the mid-frame to the inputs**"

你现在用 `flow_driven_xyz: false` 规避了直接拿它驱动 xyz（论文的约束），但 CGM 里的 warp（`temporal_attention.py:141-146`）仍然在用这个 I0→I1 流的线性假设。

**这就是残差 head 的价值：让网络学会把「通用光流」纠成「插值任务流」。**

### 4.2 改法

```python
class TaskOrientedFlowResidual(nn.Module):
    """把冻结 SEA-RAFT 的输出纠正为 It->I0 / It->I1 的任务流。"""
    def __init__(self, base_dim, in_ch, hidden=48, n_convs=3):
        super().__init__()
        ch = in_ch + base_dim + 1        # F_tau 特征 + base flow + tau
        layers = []
        for i in range(n_convs):
            layers += [nn.Conv2d(ch if i == 0 else hidden, hidden, 3, 1, 1),
                       nn.LeakyReLU(0.2, inplace=True)]
            ch = hidden
        layers.append(nn.Conv2d(hidden, 2, 1))
        nn.init.zeros_(layers[-1].weight)   # 关键：初始化为 0 => 训练初期等价于原 pipeline
        nn.init.zeros_(layers[-1].bias)
        self.net = nn.Sequential(*layers)

    def forward(self, feat_tau, base_flow, tau):
        return base_flow + self.net(torch.cat([feat_tau, base_flow, tau], dim=1))
```

**最后一层必须 Zero-init**：保证第 0 步时残差为 0，不会破坏你已经收敛的 baseline。这是残差改造能否安全热插拔的关键。

插入点：`_render_motion_gaussians` / `temporal_attn` 调用前，把 `flow_01, flow_10` 过一遍这个 head。

配合多尺度使用效果最好——残差 head 也可以按 `specialized_levels` 共享，那就是 FILM 的完全体了。

### 4.3 训练策略

1. 前 2-3 epoch：SEA-RAFT 全冻（当前 `freeze_spynet: true` 已满足），只训残差 head
2. 之后可选：解冻 SEA-RAFT 最后 2 个 block，`lr × 0.1`，与主网络分开 param group
3. 残差 head 建议**不参与 ** weight decay

---

## 5. Phase 0：先做数据体检（半天，强烈建议先做）

在动手改结构前，先搞清楚你的「大尺度运动」到底有多大。这一步能直接决定 L 该取几层、量程该配多大，**避免盲调**。

```python
# tools/flow_stats.py  (建议放在 tools/ 下)
import torch, numpy as np, tqdm
from models.searaft import build_searaft   # 按你的实际接口调整

flow_net = build_searaft(pretrained=CFG['searaft_pretrained']).cuda().eval()
mags = []
n_list = []
for batch in tqdm.tqdm(loader):
    n, vis_0, vis_N = batch['n'], batch['vis_0'].cuda(), batch['vis_N'].cuda()
    with torch.no_grad():
        f01 = flow_net(vis_0, vis_N)          # 按你的实际调用签名
        m = f01.norm(dim=1)                    # [B,H,W] 位移幅值(px)
    mags.append(m.flatten().cpu())
    n_list.append(n)
mags = torch.cat(mags).numpy()
n = np.array(n_list)

print('=== 全局光流幅值分布 (px) ===')
for p in [50, 75, 90, 95, 99, 99.9]:
    print(f'  P{p:<5}: {np.percentile(mags, p):7.2f}')
print(f'  max  : {mags.max():7.2f}')

print('\n=== 按帧间隔 N 分组 ===')
for ni in sorted(set(n.tolist())):
    sel = mags[np.concatenate([np.full(m.shape[0], k) for k, m in
              enumerate(n_list)]) == ni]   # 简化写法，按实际 batch 结构改写
    if sel.size:
        print(f'  N={ni:2d}: mean={sel.mean():6.2f}  P95={np.percentile(sel,95):7.2f}  n={sel.size}')
```

**怎么用结果**：

- 若 P95 < 40px → `n_max=12` 可能过头了，模型容量被极端样本稀释。FILM Section 5.3 明确发现：**训练集运动范围必须与测试分布匹配**，不是越大越好（更小范围训练的在 Vimeo 上反而更好）。可考虑 `n_max` 降到 8。
- 若 P95 在 60-100px → 直接用 L=6 多尺度（上限 96px）
- 若 P95 > 150px → 必须 L=7，patch 改 256

同时给的建议：**把 `n_max` 暴露为一个可控的分档训练开关**，模仿 FILM 的 bracketed dataset：先在小 motion 档训练到收敛，再逐步放开大 motion 档做课程学习。这比一开始就喂全范围更容易收敛。

---

## 6. 落地路线图

| Phase | 内容 | 工作量 | 风险 | 验证指标 |
|---|---|---|---|---|
| **P0** | 光流幅值体检 + `n_max` 分档 | 0.5 天 | 无 | 得到真实运动分布曲线 |
| **P1** | Gram loss + 两阶段 schedule（reference 用融合 target） | 1-2 天 | 低 | 主观锐度；PSNR 允许微降 |
| **P2** | 多尺度共享 offset head（L=6）+ AOW 正则单边化 | 3-5 天 | **中**（要重构 forward） | 大位移子集 PSNR/SSIM；ghosting 主观 |
| **P3** | Task-oriented 残差光流 head（Zero-init） | 2-3 天 | 低-中 | 同上；观察残差幅值是否学到非零 |

**P1 和 P3 都是热插拔的**（加个 loss 项 / 加个残差分支且零初始化），可以和 P2 并行或先做。
**P2 是唯一侵入式的**，建议单独一个分支，保留 baseline checkpoint 做 A/B。

### 每阶段必须有对照组

改动集中在同一个数据库目录会很乱。建议每次实验改一个 tag：

```bash
python train.py --config configs/train/train-m3svd-fusion.yaml --tag p2_multiscale_L6
```

并在 `train.py` 的日志里记录 `lambda_gram`、当前 epoch 是否进入第二阶段、`n_max` 值——否则三周后你分不清哪个 run 开了什么。

---

## 7. 需要留意的坑（按踩坑概率排序）

1. **Gram 的 reference 用错** → 红外热源被 Gram 当噪声抹掉（见 2.2）
2. **Gram 从头就开** → 训练不稳，FILM 自己前一半训练权重是 0
3. **patch 224 配 L=7 会运行时崩** → 224 不能被 64 整除
4. **粗 offset 上采样忘 ×2** → 逐级累加系统性欠估计，位移越大越糟
5. **AOW 正则仍 Laplace 1.0** → 与新流程目标冲突
6. **残差 head 忘 Zero-init** → 直接毁掉已收敛的 baseline
7. **PSNR 报 crops** → Gram 会牺牲 PSNR 换取感知质量，别拿同一个数同时要求两头

---

## 附：抓取的参考源码位置

`.workbuddy/film_ref/`（未跟踪，不影响仓库）：

```
losses.py                    # Loss 注册表，style/vgg 入口
vgg19_loss.py                # ★ Gram 矩阵 + Style loss 完整实现
feature_extractor.py         # ★ 级联特征金字塔 + SubTreeExtractor 权重共享
pyramid_flow_estimator.py    # ★ 金字塔残差流 + shared_predictor
options.py                   # ★ 含最大可解析运动公式
film_net-Style.gin           # ★ 论文完整超参 + 损失权重 schedule
FILM_paper.txt               # 论文全文 19 页文本
```

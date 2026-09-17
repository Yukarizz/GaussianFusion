"""
Metric implementations for GaussianFusion evaluation.

Reference-based metrics (SSIM, MEF-SSIM, VIF, MI, Qabf) are ported from
VF-Bench (src/util/metric.py) so that numbers stay comparable with the
benchmark. They expect GPU tensors in the [0, 255] range with shape
[B, 3, H, W] and return a per-sample tensor of shape [B].

Flow-based metrics (BiSWE, MS2R) reuse this project's own SEA-RAFT optical
flow wrapper (models/searaft.py) instead of VF-Bench's RAFT, so no external
model copy is required. The flow wrapper takes images in [0, 1].
"""

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers (ported from VF-Bench/src/util/metric.py)
# ---------------------------------------------------------------------------
def _fspecial_gaussian_torch(win_size, sigma):
    """2D Gaussian filter similar to MATLAB fspecial('gaussian')."""
    coords = torch.arange(-win_size // 2 + 1, win_size // 2 + 1, dtype=torch.float64)
    x, y = torch.meshgrid(coords, coords, indexing='ij')
    g = torch.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    g /= g.sum()
    return g.reshape(1, 1, win_size, win_size).contiguous()


def _input_check(batch, ref1=None, ref2=None):
    """Validate/normalize metric inputs. Expects GPU tensors."""
    def _batch_check(batch):
        assert type(batch) is torch.Tensor, "input is not a tensor"
        if len(batch.shape) == 2:
            batch = batch.unsqueeze(0)
        if len(batch.shape) == 3:
            batch = batch.unsqueeze(0)
        assert len(batch.shape) == 4, "dimension number error"
        if batch.shape[1] == 1:
            batch = batch.repeat(1, 3, 1, 1)
        assert batch.shape[1] == 3, "channel number error"
        # Fully-black images break log-based metrics; add mean-1 noise.
        if torch.max(batch) < 1:
            noise = torch.randn_like(batch) * 0.1 + 1
            batch = batch + noise
        return batch

    assert batch.device.type != 'cpu', "input is on CPU"
    batch = _batch_check(batch).to(torch.float64)
    if ref1 is not None:
        ref1 = _batch_check(ref1).to(torch.float64).to(batch.device)
    if ref2 is not None:
        ref2 = _batch_check(ref2).to(torch.float64).to(batch.device)
    return batch, ref1, ref2


def rgb2gray(tensor):
    """RGB [B,3,H,W] -> grayscale [B,1,H,W] (BT.601); pass-through if 1ch."""
    if tensor.shape[1] == 1:
        return tensor
    elif tensor.shape[1] == 3:
        r, g, b = tensor[:, 0:1], tensor[:, 1:2], tensor[:, 2:3]
        gray = torch.round(0.299 * r + 0.587 * g + 0.114 * b)
        return gray.to(torch.float64)
    else:
        raise ValueError(f"channel must be 1 or 3, got {tensor.shape[1]}")


def _ignore_nan(tensor):
    """Sum over channels ignoring NaNs, return [B,1]."""
    not_nan_channel = ~torch.isnan(tensor)
    tensor_without_nan = torch.nan_to_num(tensor, nan=0.0)
    return torch.sum(tensor_without_nan, dim=1).unsqueeze(1) / torch.sum(not_nan_channel, dim=1)


# ---------------------------------------------------------------------------
# Reference-based metrics (ported from VF-Bench, values in [0,255])
# ---------------------------------------------------------------------------
def Metric_SSIM(batch, ref1=None, ref2=None):
    def filter2D(img, win):
        img = F.pad(img, pad=(win.shape[-2] // 2, win.shape[-1] // 2,
                              win.shape[-2] // 2, win.shape[-1] // 2), mode='reflect')
        return F.conv2d(img, win.repeat(img.shape[1], 1, 1, 1), padding=0, groups=img.shape[1])

    def ssim(img, img_ref):
        K1 = (0.01 * 255) ** 2
        K2 = (0.03 * 255) ** 2
        win = _fspecial_gaussian_torch(11, 1.5).to(img.device)

        mu1 = filter2D(img, win)
        mu2 = filter2D(img_ref, win)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = filter2D(img * img, win) - mu1_sq
        sigma2_sq = filter2D(img_ref * img_ref, win) - mu2_sq
        sigma12 = filter2D(img * img_ref, win) - mu1_mu2

        ssim_map = ((2. * mu1_mu2 + K1) * (2. * sigma12 + K2)) / \
            ((mu1_sq + mu2_sq + K1) * (sigma1_sq + sigma2_sq + K2))
        return torch.mean(ssim_map, dim=(2, 3))

    with torch.no_grad():
        batch, ref1, ref2 = _input_check(batch, ref1, ref2)
        output = (ssim(batch, ref1) + ssim(batch, ref2)) / 2
        return torch.mean(output, dim=1)


def Metric_VIF(batch, ref1=None, ref2=None):
    def _vifp_batch(ref, dist):
        ref = ref.to(torch.float64)
        dist = dist.to(torch.float64)
        sigma_nsq = 2.0
        eps = 1e-10
        num = torch.zeros(ref.shape[0], ref.shape[1]).to(dist.device)
        den = torch.zeros(ref.shape[0], ref.shape[1]).to(dist.device)

        for scale in range(1, 5):
            N = 2 ** (4 - scale + 1) + 1
            sd = N / 5.0
            win = _fspecial_gaussian_torch(N, sd).to(dist.device)

            if scale > 1:
                ref = F.conv2d(ref, win.repeat(batch.shape[1], 1, 1, 1), padding=0,
                               groups=batch.shape[1])[:, :, ::2, ::2]
                dist = F.conv2d(dist, win.repeat(batch.shape[1], 1, 1, 1), padding=0,
                                groups=batch.shape[1])[:, :, ::2, ::2]

            mu1 = F.conv2d(ref, win.repeat(batch.shape[1], 1, 1, 1), padding=0, groups=batch.shape[1])
            mu2 = F.conv2d(dist, win.repeat(batch.shape[1], 1, 1, 1), padding=0, groups=batch.shape[1])
            mu1_sq = mu1 * mu1
            mu2_sq = mu2 * mu2
            mu1_mu2 = mu1 * mu2
            sigma1_sq = F.conv2d(ref * ref, win.repeat(batch.shape[1], 1, 1, 1), padding=0,
                                 groups=batch.shape[1]) - mu1_sq
            sigma2_sq = F.conv2d(dist * dist, win.repeat(batch.shape[1], 1, 1, 1), padding=0,
                                 groups=batch.shape[1]) - mu2_sq
            sigma12 = F.conv2d(ref * dist, win.repeat(batch.shape[1], 1, 1, 1), padding=0,
                               groups=batch.shape[1]) - mu1_mu2
            sigma1_sq = torch.clamp(sigma1_sq, min=0)
            sigma2_sq = torch.clamp(sigma2_sq, min=0)
            g = sigma12 / (sigma1_sq + eps)
            sv_sq = sigma2_sq - g * sigma12

            g[sigma1_sq < eps] = 0
            sv_sq[sigma1_sq < eps] = sigma2_sq[sigma1_sq < eps]
            sigma1_sq[sigma1_sq < eps] = 0

            g[sigma2_sq < eps] = 0
            sv_sq[sigma2_sq < eps] = 0

            sv_sq[g < 0] = sigma2_sq[g < 0]
            g[g < 0] = 0
            sv_sq[sv_sq <= eps] = eps

            num += torch.sum(torch.log10(1 + g * g * sigma1_sq / (sv_sq + sigma_nsq)), dim=(2, 3))
            den += torch.sum(torch.log10(1 + sigma1_sq / sigma_nsq), dim=(2, 3))

        vifp_val = num / den
        vifp_val = _ignore_nan(vifp_val)
        return vifp_val

    with torch.no_grad():
        batch, ref1, ref2 = _input_check(batch, ref1, ref2)
        vifp1 = torch.mean(_vifp_batch(ref1, batch), dim=1)
        vifp2 = torch.mean(_vifp_batch(ref2, batch), dim=1)
        return (vifp1 + vifp2) / 2.0


def Metric_MEF_SSIM(batch, ref1=None, ref2=None):
    def mef_ssim(img, img1, img2, K=0.03, window=None):
        imgSeq = torch.cat((img1.unsqueeze(-1), img2.unsqueeze(-1)), dim=-1)  # B,C,H,W,2

        if window is None:
            window = _fspecial_gaussian_torch(11, 1.5).to(img.device)

        wSize = window.shape[2]
        sWindow = torch.ones((wSize, wSize), device=img.device, dtype=torch.float64) / wSize ** 2
        bd = wSize // 2

        imgSeq = torch.cat((imgSeq[:, :, :, :, 0], imgSeq[:, :, :, :, 1]), dim=1)  # B,2*C,H,W
        mu = F.conv2d(imgSeq, sWindow.repeat(imgSeq.shape[1], 1, 1, 1), padding=0, groups=imgSeq.shape[1])
        sigmaSq = F.conv2d(imgSeq ** 2, sWindow.repeat(imgSeq.shape[1], 1, 1, 1), padding=0,
                           groups=imgSeq.shape[1]) - mu ** 2

        ed = torch.sqrt(torch.relu(wSize * wSize * sigmaSq)) + 0.001  # B,2*C,H,W
        del sigmaSq
        temp1, temp2 = torch.chunk(ed.unsqueeze(-1), 2, dim=1)
        ed = torch.cat([temp1, temp2], dim=-1)  # B,C,H,W,2
        del temp1, temp2

        vecs = imgSeq.unfold(2, 2 * bd + 1, 1).unfold(3, 2 * bd + 1, 1)
        vecs = vecs.flatten(start_dim=4)
        denominator = torch.norm(vecs - mu.unsqueeze(-1), p=2, dim=4)
        temp1, temp2 = torch.chunk(denominator, 2, dim=1)
        denominator = temp1 + temp2

        temp1, temp2 = torch.chunk(vecs.unsqueeze(-1), 2, dim=1)
        vecs = torch.cat([temp1, temp2], dim=-1)  # B,C,H,W,win*win,2
        del temp1, temp2
        vecs_sum = torch.sum(vecs, dim=-1)
        del vecs
        numerator = torch.norm(vecs_sum - torch.mean(vecs_sum, dim=-1, keepdim=True), p=2, dim=-1)
        del vecs_sum
        R = (numerator + torch.finfo(torch.float64).eps) / (denominator + torch.finfo(torch.float64).eps)
        del numerator, denominator

        R = torch.clamp(R, min=torch.finfo(torch.float64).eps, max=1 - torch.finfo(torch.float64).eps)
        p = torch.tan(torch.pi / 2 * R)
        p = torch.clamp(p, min=0, max=10)

        wMap = (ed / wSize) ** (p.unsqueeze(-1)) + torch.finfo(torch.float64).eps  # B,C,H,W,2
        wMap /= torch.sum(wMap, dim=-1, keepdim=True)

        maxEd = torch.max(ed, dim=-1)[0]  # B,C,H,W

        C = (K * 255) ** 2
        blocks = imgSeq.unfold(2, 2 * bd + 1, 1).unfold(3, 2 * bd + 1, 1)
        blocks = blocks.flatten(start_dim=4)
        temp1, temp2 = torch.chunk(blocks.unsqueeze(-1), 2, dim=1)
        blocks = torch.cat([temp1, temp2], dim=-1)  # B,C,H,W,win*win,2

        temp1, temp2 = torch.chunk(mu.unsqueeze(-1), 2, dim=1)
        mu = torch.cat([temp1, temp2], dim=-1)  # B,C,H,W,2

        rBlock = wMap.unsqueeze(4) * (blocks - mu.unsqueeze(4)) / ed.unsqueeze(4)
        del ed, wMap, blocks, temp1, temp2, mu

        rBlock = rBlock.sum(-1)
        temp1 = torch.norm(rBlock, p=2, dim=4, keepdim=True)
        rBlock = (temp1 != 0) * rBlock / (temp1 + (temp1 == 0)) * maxEd.unsqueeze(-1)
        del maxEd, temp1
        fBlock = img.unfold(2, 2 * bd + 1, 1).unfold(3, 2 * bd + 1, 1).flatten(start_dim=4)

        window = window.flatten().view(1, 1, 1, 1, -1)

        mu1 = torch.sum(window * rBlock, 4, keepdim=True)
        mu2 = torch.sum(window * fBlock, 4, keepdim=True)
        sigma1Sq = torch.sum(window * (rBlock - mu1) ** 2, dim=-1)
        sigma2Sq = torch.sum(window * (fBlock - mu2) ** 2, dim=-1)
        sigma12 = torch.sum(window * (rBlock - mu1) * (fBlock - mu2), dim=-1)
        del rBlock, fBlock, mu1, mu2
        qMap = (2 * sigma12 + C) / (sigma1Sq + sigma2Sq + C)
        Q = torch.mean(qMap, dim=[2, 3])
        return Q

    with torch.no_grad():
        batch, ref1, ref2 = _input_check(batch, ref1, ref2)
        batch, ref1, ref2 = rgb2gray(batch), rgb2gray(ref1), rgb2gray(ref2)
        return torch.mean(mef_ssim(batch, ref1, ref2), dim=1)


def _torch_contingency_matrix(labels_true, labels_pred):
    unique_true, map_true = torch.unique(labels_true, return_inverse=True)
    unique_pred, map_pred = torch.unique(labels_pred, return_inverse=True)
    contingency = torch.zeros((len(unique_true), len(unique_pred)),
                              dtype=torch.int64, device=labels_true.device)
    contingency.index_put_((map_true, map_pred),
                           torch.ones_like(map_true, dtype=torch.int64), accumulate=True)
    return contingency


def _torch_normalized_mutual_info_score(labels_true, labels_pred, average_method='arithmetic'):
    batch_size = labels_true.shape[0]
    nmi_values = torch.zeros(batch_size, dtype=torch.float64)
    mi_calculate = torch.zeros(labels_true.shape[0], labels_true.shape[1])
    nmi_calculate = torch.zeros(labels_true.shape[0], labels_true.shape[1])
    for b in range(batch_size):
        for c in range(labels_true.shape[1]):
            lbl_true = labels_true[b, c].flatten()
            lbl_pred = labels_pred[b, c].flatten()

            unique_true = torch.unique(lbl_true)
            unique_pred = torch.unique(lbl_pred)
            if (len(unique_true) == len(unique_pred) == 1) or (len(unique_true) == len(unique_pred) == 0):
                nmi_values[b] = 1.0
                continue

            contingency = _torch_contingency_matrix(lbl_true, lbl_pred).to(torch.float64)
            total = contingency.sum()
            if total == 0:
                nmi_values[b] = 1.0
                continue

            pi = contingency.sum(dim=1)
            pj = contingency.sum(dim=0)
            log_contingency = (contingency / total).log()
            log_pi = (pi / total).log().unsqueeze(1)
            log_pj = (pj / total).log().unsqueeze(0)

            mi = (contingency / total) * (log_contingency - log_pi - log_pj)
            mi = mi.nansum()
            mi_calculate[b, c] = mi

            if mi <= 1e-15:
                nmi_values[b] = 0.0
                continue

            h_true = (-(pi[pi > 0] / pi.sum()).log() * (pi[pi > 0] / pi.sum())).sum()
            h_pred = (-(pj[pj > 0] / pj.sum()).log() * (pj[pj > 0] / pj.sum())).sum()

            if average_method == 'arithmetic':
                normalizer = 0.5 * (h_true + h_pred)
            elif average_method == 'geometric':
                normalizer = torch.sqrt(h_true * h_pred)
            else:
                raise ValueError(f"Unsupported average_method: {average_method}")

            nmi_values[b] = mi / normalizer
            nmi_calculate[b, c] = nmi_values[b]
    return nmi_calculate, mi_calculate


def Metric_MI(batch, ref1=None, ref2=None):
    with torch.no_grad():
        batch, ref1, ref2 = _input_check(batch, ref1, ref2)
        _, mi1 = _torch_normalized_mutual_info_score(batch, ref1)
        _, mi2 = _torch_normalized_mutual_info_score(batch, ref2)
    return torch.mean(mi1, dim=1) + torch.mean(mi2, dim=1)


def Metric_Qabf(batch, ref1, ref2):
    def Qabf_getArray(img):
        h1 = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float64,
                          device=img.device).view(1, 1, 3, 3).repeat(img.shape[1], 1, 1, 1)
        h3 = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float64,
                          device=img.device).view(1, 1, 3, 3).repeat(img.shape[1], 1, 1, 1)
        SAx = F.conv2d(img, h3, padding=1, groups=img.shape[1])
        SAy = F.conv2d(img, h1, padding=1, groups=img.shape[1])
        gA = torch.sqrt(SAx.pow(2) + SAy.pow(2))
        aA = torch.zeros_like(img)
        aA[SAx == 0] = torch.pi / 2
        mask = SAx != 0
        aA[mask] = torch.atan(SAy[mask] / SAx[mask])
        return gA, aA

    def Qabf_getQabf(aA, gA, aF, gF):
        Tg, kg, Dg = 0.9994, -15, 0.5
        Ta, ka, Da = 0.9879, -22, 0.8
        GAF = torch.where(gA > gF, gF / gA,
                          torch.where(gA < gF, gA / gF, gF))
        AAF = 1 - (torch.abs(aA - aF) / (torch.pi / 2))
        QgAF = Tg / (1 + torch.exp(kg * (GAF - Dg)))
        QaAF = Ta / (1 + torch.exp(ka * (AAF - Da)))
        return QgAF * QaAF

    def Qabf(img, img1, img2):
        gA, aA = Qabf_getArray(img1)
        gB, aB = Qabf_getArray(img2)
        gF, aF = Qabf_getArray(img)
        QAF = Qabf_getQabf(aA, gA, aF, gF)
        QBF = Qabf_getQabf(aB, gB, aF, gF)
        nume = torch.sum(QAF * gA + QBF * gB, dim=(2, 3))
        deno = torch.sum(gA + gB, dim=(2, 3))
        return nume / deno

    with torch.no_grad():
        batch, ref1, ref2 = _input_check(batch, ref1, ref2)
        results = Qabf(batch, ref1, ref2)
        return torch.mean(results, dim=1)


# ---------------------------------------------------------------------------
# Flow-based metrics (BiSWE / MS2R) using this project's SEA-RAFT flow
# ---------------------------------------------------------------------------
class FlowMetrics:
    """BiSWE and MS2R built on the project's own optical flow model.

    The flow network follows the SeaRAFT wrapper interface:
        flow_net(im1, im2) -> flow [B, 2, H, W], inputs in [0, 1].
    """

    def __init__(self, flow_net, occ_threshold=1.0):
        from warp_utils import flow_warp
        self.flow_net = flow_net
        self.flow_warp = flow_warp
        self.occ_threshold = occ_threshold

    @torch.no_grad()
    def compute_flow(self, img1, img2):
        return self.flow_net(img1, img2)

    @torch.no_grad()
    def occlusion_mask(self, img1, img2, flow_ab):
        """Forward-backward consistency occlusion mask (img1 is current frame)."""
        flow_ba = self.flow_net(img2, img1)
        flow_ba_warped = self.flow_warp(flow_ba, flow_ab)
        fb_diff = flow_ab + flow_ba_warped
        fb_consistency = fb_diff.norm(p=2, dim=1)  # [B, H, W]
        return (fb_consistency < self.occ_threshold).float()

    @torch.no_grad()
    def biswe(self, video_clip, R1_clip, R2_clip, use_occlusion=True):
        """Bi-Directional Self-Warping Error.

        Args:
            video_clip: [B, T=3, C, H, W] fused clip (frame 1 = target).
            R1_clip:    [B, T=3, C, H, W] reference video 1 (e.g. visible).
            R2_clip:    [B, T=3, C, H, W] reference video 2 (e.g. infrared).

        Returns:
            total_error: [B] tensor (lower is better).
        """
        B, _, _, H, W = video_clip.shape
        device = video_clip.device

        cur = video_clip[:, 1]
        prev = video_clip[:, 0]
        nxt = video_clip[:, 2]

        if use_occlusion:
            flow_R1_cur2prev = self.flow_net(R1_clip[:, 1], R1_clip[:, 0])
            flow_R1_cur2next = self.flow_net(R1_clip[:, 1], R1_clip[:, 2])
            flow_R2_cur2prev = self.flow_net(R2_clip[:, 1], R2_clip[:, 0])
            flow_R2_cur2next = self.flow_net(R2_clip[:, 1], R2_clip[:, 2])

            mask_R1_prev = self.occlusion_mask(R1_clip[:, 1], R1_clip[:, 0], flow_R1_cur2prev)
            mask_R1_next = self.occlusion_mask(R1_clip[:, 1], R1_clip[:, 2], flow_R1_cur2next)
            mask_R2_prev = self.occlusion_mask(R2_clip[:, 1], R2_clip[:, 0], flow_R2_cur2prev)
            mask_R2_next = self.occlusion_mask(R2_clip[:, 1], R2_clip[:, 2], flow_R2_cur2next)

            mask_prev = mask_R1_prev * mask_R2_prev
            mask_next = mask_R1_next * mask_R2_next
        else:
            mask_prev = torch.ones((B, H, W), device=device)
            mask_next = torch.ones((B, H, W), device=device)

        flow_cur2prev = self.flow_net(cur, prev)
        flow_cur2next = self.flow_net(cur, nxt)

        recon_prev = self.flow_warp(prev, flow_cur2prev)
        recon_next = self.flow_warp(nxt, flow_cur2next)

        diff_prev = torch.abs(cur - recon_prev).mean(1)
        diff_next = torch.abs(cur - recon_next).mean(1)

        err_prev = (mask_prev * diff_prev).sum(dim=(1, 2)) / (mask_prev.sum(dim=(1, 2)) + 1e-10)
        err_next = (mask_next * diff_next).sum(dim=(1, 2)) / (mask_next.sum(dim=(1, 2)) + 1e-10)

        return err_prev + err_next  # [B]

    @torch.no_grad()
    def ms2r(self, G_clip, R1_clip, R2_clip):
        """Motion Smoothness with Dual Reference videos.

        Args:
            G_clip:  [B, T=3, C, H, W] fused (generated) clip.
            R1_clip: [B, T=3, C, H, W] reference video 1.
            R2_clip: [B, T=3, C, H, W] reference video 2.

        Returns:
            [B] tensor (lower is better).
        """
        B = G_clip.shape[0]
        all_metrics = []

        def differential_flow(G0, G1, G2, R0, R1, R2):
            d_gen = self.compute_flow(G1, G2) - self.compute_flow(G0, G1)
            d_ref = self.compute_flow(R1, R2) - self.compute_flow(R0, R1)
            return d_gen - d_ref

        for k in range(B):
            G0, G1, G2 = G_clip[k]
            R10, R11, R12 = R1_clip[k]
            R20, R21, R22 = R2_clip[k]

            D1 = differential_flow(G0.unsqueeze(0), G1.unsqueeze(0), G2.unsqueeze(0),
                                   R10.unsqueeze(0), R11.unsqueeze(0), R12.unsqueeze(0))
            D2 = differential_flow(G0.unsqueeze(0), G1.unsqueeze(0), G2.unsqueeze(0),
                                   R20.unsqueeze(0), R21.unsqueeze(0), R22.unsqueeze(0))

            metric_val = 0.5 * (torch.mean(torch.abs(D1)) + torch.mean(torch.abs(D2)))
            all_metrics.append(metric_val)

        return torch.stack(all_metrics)  # [B]


def build_flow_metrics(device, searaft_weights=None, occ_threshold=1.0):
    """Construct a FlowMetrics evaluator using the project's SEA-RAFT flow."""
    import os
    from models.searaft import SeaRAFT

    if searaft_weights is None:
        here = os.path.dirname(os.path.abspath(__file__))
        searaft_weights = os.path.join(
            here, 'models', 'weights', 'Tartan-C-T-TSKH-spring540x960-S.pth')

    flow_net = SeaRAFT(pretrained=searaft_weights, freeze=True).to(device).eval()
    return FlowMetrics(flow_net, occ_threshold=occ_threshold)


# Metric registry used by evaluate_metrics.py
REFERENCE_METRICS = {
    'ssim': Metric_SSIM,
    'mef_ssim': Metric_MEF_SSIM,
    'vif': Metric_VIF,
    'mi': Metric_MI,
    'qabf': Metric_Qabf,
}
FLOW_METRICS = ('biswe', 'ms2r')

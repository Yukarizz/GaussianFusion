"""GPU end-to-end forward pass test for GaussianFusion model."""
import torch
import models
from warp_utils import temporal_blend

print("=== GPU Test: GaussianFusion ===")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")
device = torch.device('cuda:0')

# 1. Build model
model = models.make({'name': 'gaussian-fusion', 'args': {
    'encoder_spec': {'name': 'edsr-baseline', 'args': {
        'n_resblocks': 16, 'n_feats': 64, 'res_scale': 1,
        'scale': [1], 'no_upsampling': True, 'n_colors': 64,
        'rgb_range': 1, 'pretrained_path': None}},
    'spynet_pretrained': 'sintel-final', 'n_feats': 64, 'freeze_spynet': True}})
model = model.to(device)
print("[OK] Model on GPU")

# 2. Dummy inputs
B, H, W = 2, 64, 64
vis_0 = torch.randn(B, 3, H, W, device=device)
ir_0 = torch.randn(B, 3, H, W, device=device)
vis_N = torch.randn(B, 3, H, W, device=device)
ir_N = torch.randn(B, 3, H, W, device=device)
scale = torch.tensor([2.0], device=device)
tau = 0.5

# 3. Full forward pass (includes gsplat rasterization)
with torch.no_grad():
    out = model(vis_0, ir_0, vis_N, ir_N, scale, tau)
print(f"[OK] Forward pass output: {out.shape}")
assert out.shape == (B, 3, H * 2, W * 2), f"Expected {(B, 3, H*2, W*2)}, got {out.shape}"
print(f"[OK] Output shape correct: {out.shape}")

# 4. Test different scales
for s in [1.5, 3.0, 4.0]:
    scale_t = torch.tensor([s], device=device)
    out_s = model(vis_0, ir_0, vis_N, ir_N, scale_t, tau)
    exp_h, exp_w = int(H * s), int(W * s)
    print(f"[OK] Scale {s}x -> {out_s.shape} (expected [{B}, 3, {exp_h}, {exp_w}])")

# 5. Test different tau values
for t in [0.0, 0.25, 0.75, 1.0]:
    out_t = model(vis_0, ir_0, vis_N, ir_N, scale, t)
    print(f"[OK] tau={t} -> {out_t.shape}")

# 6. Backward pass (training mode)
model.train()
out = model(vis_0, ir_0, vis_N, ir_N, scale, tau)
loss = out.mean()
loss.backward()
print("[OK] Backward pass completed")

# 7. Memory usage
mem = torch.cuda.max_memory_allocated() / 1024**2
print(f"[OK] Peak GPU memory: {mem:.1f} MB")

print("\n=== ALL GPU TESTS PASSED ===")

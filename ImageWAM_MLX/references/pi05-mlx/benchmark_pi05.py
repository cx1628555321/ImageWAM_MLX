"""
benchmark_pi05.py — Timing comparison: MLX (4-bit quantized) vs PyTorch MPS.

MLX:  load quantized model once (~3s), then run N complete inferences.
      Model stays resident at ~2.4 GB — fits comfortably on 8 GB M1.

PyTorch: chunked loading each run (can't hold float32 model in 8 GB).
         Each run = load-layer-by-layer + MPS compute.
"""

import gc
import math
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

sys.path.insert(0, "/Users/mohan_007/Downloads/dare_to_experiment")

SAFETENSORS = (
    "/Users/mohan_007/Downloads/dare_to_experiment/pi05_cache/"
    "models--lerobot--pi05_base/snapshots/"
    "9e55186ad36e66b95cda57bc47818d9e6237ae30/model.safetensors"
)
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

MLX_RUNS = 5
PT_RUNS  = 3

# ── Fixed inputs ──────────────────────────────────────────────────────────────
np.random.seed(42)
torch.manual_seed(42)
IMAGE_NP  = np.random.uniform(-1, 1, (1, 3, 224, 224)).astype(np.float32)
LANG_IDS  = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=np.int32)
NOISE_NP  = np.random.randn(1, 50, 32).astype(np.float32)
B, CHUNK, ADIM, NUM_STEPS = 1, 50, 32, 10


# ─────────────────────────────────────────────────────────────────────────────
#  PyTorch helpers (same as compare_e2e.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_raw(prefix):
    out = {}
    with safe_open(SAFETENSORS, framework="pt", device="cpu") as f:
        for k in f.keys():
            if k.startswith(prefix):
                out[k[len(prefix):]] = f.get_tensor(k).float()
    return out

def free(*objs):
    for o in objs: del o
    gc.collect()

def sinusoidal_emb_pt(t, dim, min_p=4e-3, max_p=4.0):
    half = dim // 2
    frac = torch.linspace(0., 1., half, dtype=torch.float32, device=t.device)
    period = min_p * (max_p / min_p) ** frac
    arg = (2.0 * math.pi / period).unsqueeze(0) * t.float().unsqueeze(1)
    return torch.cat([arg.sin(), arg.cos()], dim=-1)

def apply_rope(x, offset=0, base=10000.):
    B, H, L, D = x.shape
    half = D // 2
    inv_freq = 1.0 / (base ** (torch.arange(half, dtype=torch.float32, device=x.device) / half))
    pos   = torch.arange(offset, offset + L, dtype=torch.float32, device=x.device)
    freqs = torch.outer(pos, inv_freq).unsqueeze(0).unsqueeze(0)
    cos_ = freqs.cos().to(x.dtype); sin_ = freqs.sin().to(x.dtype)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([cos_, cos_], -1) * x + torch.cat([sin_, sin_], -1) * torch.cat([-x2, x1], -1)

class PTRMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__(); self.eps = eps; self.weight = nn.Parameter(torch.zeros(d))
    def forward(self, x):
        x32 = x.float()
        return ((1 + self.weight) * x32 * torch.rsqrt((x32*x32).mean(-1, keepdim=True) + self.eps)).to(x.dtype)

class PTGemmaAttn(nn.Module):
    def __init__(self, h, nh, nkv, hd):
        super().__init__(); self.nh = nh; self.nkv = nkv; self.hd = hd; self.scale = hd**-0.5
        self.q_proj = nn.Linear(h, nh*hd, bias=False); self.k_proj = nn.Linear(h, nkv*hd, bias=False)
        self.v_proj = nn.Linear(h, nkv*hd, bias=False); self.o_proj = nn.Linear(nh*hd, h, bias=False)
    def forward(self, x, cache=None):
        B, L, _ = x.shape
        q = self.q_proj(x).reshape(B, L, self.nh, self.hd).transpose(1, 2)
        k = self.k_proj(x).reshape(B, L, self.nkv, self.hd).transpose(1, 2)
        v = self.v_proj(x).reshape(B, L, self.nkv, self.hd).transpose(1, 2)
        offset = 0
        if cache:
            kc, vc = cache; offset = kc.shape[2]
            q = apply_rope(q, offset); k = apply_rope(k, offset)
            k = torch.cat([kc, k], 2); v = torch.cat([vc, v], 2)
        else:
            q = apply_rope(q); k = apply_rope(k)
        rep = self.nh // self.nkv
        ke = k.repeat_interleave(rep, 1); ve = v.repeat_interleave(rep, 1)
        a = F.softmax((torch.matmul(q, ke.transpose(-2, -1)) * self.scale).float(), -1).to(q.dtype)
        out = torch.matmul(a, ve).transpose(1, 2).reshape(B, q.shape[2], -1)
        return self.o_proj(out), (k, v)

class PTGemmaMLP(nn.Module):
    def __init__(self, h, f):
        super().__init__()
        self.gate_proj = nn.Linear(h, f, bias=False); self.up_proj = nn.Linear(h, f, bias=False)
        self.down_proj = nn.Linear(f, h, bias=False)
    def forward(self, x):
        return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))

class PTGemmaBlock(nn.Module):
    def __init__(self, h, nh, nkv, hd, f):
        super().__init__()
        self.input_layernorm = PTRMSNorm(h); self.self_attn = PTGemmaAttn(h, nh, nkv, hd)
        self.post_attention_layernorm = PTRMSNorm(h); self.mlp = PTGemmaMLP(h, f)
    def forward(self, x, cache=None):
        r, cache = self.self_attn(self.input_layernorm(x), cache)
        x = x + r
        return x + self.mlp(self.post_attention_layernorm(x)), cache

class PTAdaRMSNorm(nn.Module):
    def __init__(self, d, cd, eps=1e-6):
        super().__init__(); self.eps = eps; self.dense = nn.Linear(cd, d*3, bias=True)
    def _n(self, x):
        x32 = x.float(); return x32 * torch.rsqrt((x32*x32).mean(-1, keepdim=True) + self.eps)
    def forward(self, x, cond):
        mod = self.dense(cond.to(self.dense.weight.dtype)).unsqueeze(-2)
        s, sh, g = mod.chunk(3, -1)
        return (self._n(x)*(1+s.float()) + sh.float()).to(x.dtype), g.to(x.dtype)

class PTAdaRMSFinal(nn.Module):
    def __init__(self, d, cd, eps=1e-6):
        super().__init__(); self.eps = eps; self.dense = nn.Linear(cd, d*3, bias=True)
    def _n(self, x):
        x32 = x.float(); return x32 * torch.rsqrt((x32*x32).mean(-1, keepdim=True) + self.eps)
    def forward(self, x, cond):
        mod = self.dense(cond.to(self.dense.weight.dtype)).unsqueeze(-2)
        s, sh, _ = mod.chunk(3, -1)
        return (self._n(x)*(1+s.float()) + sh.float()).to(x.dtype)

class PTExpertBlock(nn.Module):
    def __init__(self, h, nh, nkv, hd, f):
        super().__init__()
        self.input_layernorm = PTAdaRMSNorm(h, h); self.self_attn = PTGemmaAttn(h, nh, nkv, hd)
        self.post_attention_layernorm = PTAdaRMSNorm(h, h); self.mlp = PTGemmaMLP(h, f)
    def forward(self, x, cond, cache=None):
        n, g = self.input_layernorm(x, cond); a, cache = self.self_attn(n, cache); x = x + a * g
        n, g = self.post_attention_layernorm(x, cond)
        return x + self.mlp(n) * g, cache

class PTSigLIPAttn(nn.Module):
    def __init__(self, h=1152, nh=16):
        super().__init__(); self.nh = nh; self.hd = h//nh; self.scale = self.hd**-0.5
        self.q_proj = nn.Linear(h,h,bias=True); self.k_proj = nn.Linear(h,h,bias=True)
        self.v_proj = nn.Linear(h,h,bias=True); self.out_proj = nn.Linear(h,h,bias=True)
    def forward(self, x):
        B, L, _ = x.shape
        q = self.q_proj(x).reshape(B,L,self.nh,self.hd).transpose(1,2)
        k = self.k_proj(x).reshape(B,L,self.nh,self.hd).transpose(1,2)
        v = self.v_proj(x).reshape(B,L,self.nh,self.hd).transpose(1,2)
        a = F.softmax((torch.matmul(q,k.transpose(-2,-1))*self.scale).float(),-1).to(q.dtype)
        return self.out_proj(torch.matmul(a,v).transpose(1,2).reshape(B,L,-1))

class PTSigLIPLayer(nn.Module):
    def __init__(self, h=1152, nh=16, f=4304):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(h, eps=1e-6); self.self_attn = PTSigLIPAttn(h, nh)
        self.layer_norm2 = nn.LayerNorm(h, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(h,f,bias=True), nn.GELU(approximate="tanh"), nn.Linear(f,h,bias=True))
    def forward(self, x):
        x = x + self.self_attn(self.layer_norm1(x)); return x + self.mlp(self.layer_norm2(x))

class PTSigLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = nn.ModuleDict({
            "patch_embedding": nn.Conv2d(3,1152,14,14,bias=True),
            "position_embedding": nn.Embedding(256,1152),
        })
        self.encoder = nn.ModuleList([PTSigLIPLayer() for _ in range(27)])
        self.post_layernorm = nn.LayerNorm(1152, eps=1e-6)
    def forward(self, x):
        p = self.embeddings["patch_embedding"](x); B,C,H,W = p.shape
        p = p.flatten(2).transpose(1,2)
        p = p + self.embeddings["position_embedding"](torch.arange(H*W, device=x.device))
        for layer in self.encoder: p = layer(p)
        return self.post_layernorm(p)


# ─────────────────────────────────────────────────────────────────────────────
#  PyTorch single run (chunked loading + MPS compute)
# ─────────────────────────────────────────────────────────────────────────────

def pt_single_run():
    t_start = time.time()
    timings = {}

    VIS_PFX = "paligemma_with_expert.paligemma.model.vision_tower.vision_model."
    LM_PFX  = "paligemma_with_expert.paligemma.model.language_model."
    EXP_PFX = "paligemma_with_expert.gemma_expert.model."
    MPROJ   = "paligemma_with_expert.paligemma.model.multi_modal_projector.linear."
    LM_HEAD = "paligemma_with_expert.paligemma.lm_head.weight"

    # SigLIP
    t0 = time.time()
    raw = load_raw(VIS_PFX)
    remapped = {}
    for k, v in raw.items():
        k = k.replace("encoder.layers.", "encoder.").replace(".mlp.fc1.", ".mlp.0.").replace(".mlp.fc2.", ".mlp.2.")
        remapped[k] = v
    siglip = PTSigLIP(); sd = siglip.state_dict()
    siglip.load_state_dict({k: remapped.get(k, sd[k]) for k in sd}, strict=True)
    siglip.eval().to(DEVICE)
    with torch.no_grad():
        img_feat = siglip(torch.from_numpy(IMAGE_NP).to(DEVICE))
    free(siglip, raw, remapped)
    timings["siglip"] = time.time() - t0

    # mm_proj + embed_tokens
    t0 = time.time()
    mproj_raw = load_raw(MPROJ)
    with safe_open(SAFETENSORS, framework="pt", device="cpu") as f:
        embed_w = f.get_tensor(LM_HEAD).float()
    mm_proj = nn.Linear(1152, 2048, bias=True)
    mm_proj.weight = nn.Parameter(mproj_raw["weight"]); mm_proj.bias = nn.Parameter(mproj_raw["bias"])
    mm_proj.eval().to(DEVICE)
    embed_tokens = nn.Embedding(257152, 2048)
    embed_tokens.weight = nn.Parameter(embed_w); embed_tokens.to(DEVICE)
    with torch.no_grad():
        img_emb  = (mm_proj(img_feat.to(DEVICE)) * math.sqrt(2048)).float()
        lang_emb = (embed_tokens(torch.from_numpy(LANG_IDS).long().to(DEVICE)) * math.sqrt(2048)).float()
        prefix   = torch.cat([img_emb, lang_emb], dim=1)
    free(img_feat, img_emb, lang_emb, mm_proj, embed_tokens, mproj_raw, embed_w)
    timings["mproj_embed"] = time.time() - t0

    # VLM Gemma 2B
    t0 = time.time()
    H, NH, NKV, HD, FFN_DIM = 2048, 8, 1, 256, 16384
    kv_cache = []; x = prefix.to(DEVICE)
    with safe_open(SAFETENSORS, framework="pt", device="cpu") as sf:
        for li in range(18):
            pfx = f"{LM_PFX}layers.{li}."
            w = {k[len(pfx):]: sf.get_tensor(k).float() for k in sf.keys() if k.startswith(pfx)}
            block = PTGemmaBlock(H, NH, NKV, HD, FFN_DIM); block.load_state_dict(w, strict=True); block.eval().to(DEVICE)
            with torch.no_grad():
                x, (k_c, v_c) = block(x, cache=None)
            kv_cache.append((k_c.detach(), v_c.detach())); free(block, w)
    norm_w = {}
    with safe_open(SAFETENSORS, framework="pt", device="cpu") as sf:
        norm_w["weight"] = sf.get_tensor(f"{LM_PFX}norm.weight").float()
    vlm_norm = PTRMSNorm(H); vlm_norm.load_state_dict(norm_w, strict=True); vlm_norm.eval().to(DEVICE)
    with torch.no_grad():
        x = vlm_norm(x)
    free(x, vlm_norm, norm_w)
    timings["vlm"] = time.time() - t0

    # Action expert
    t0 = time.time()
    proj_keys = ["action_in_proj.weight","action_in_proj.bias","action_out_proj.weight","action_out_proj.bias",
                 "time_mlp_in.weight","time_mlp_in.bias","time_mlp_out.weight","time_mlp_out.bias"]
    proj_w = {}
    with safe_open(SAFETENSORS, framework="pt", device="cpu") as sf:
        for k in proj_keys: proj_w[k] = sf.get_tensor(k).float()
    action_in  = nn.Linear(ADIM, 1024, bias=True); action_out = nn.Linear(1024, ADIM, bias=True)
    time_in    = nn.Linear(1024, 1024, bias=True);  time_out   = nn.Linear(1024, 1024, bias=True)
    action_in.weight  = nn.Parameter(proj_w["action_in_proj.weight"])
    action_in.bias    = nn.Parameter(proj_w["action_in_proj.bias"])
    action_out.weight = nn.Parameter(proj_w["action_out_proj.weight"])
    action_out.bias   = nn.Parameter(proj_w["action_out_proj.bias"])
    time_in.weight    = nn.Parameter(proj_w["time_mlp_in.weight"])
    time_in.bias      = nn.Parameter(proj_w["time_mlp_in.bias"])
    time_out.weight   = nn.Parameter(proj_w["time_mlp_out.weight"])
    time_out.bias     = nn.Parameter(proj_w["time_mlp_out.bias"])
    action_in.to(DEVICE); action_out.to(DEVICE); time_in.to(DEVICE); time_out.to(DEVICE)

    EH, ENH, ENKV, EHD, EF = 1024, 8, 1, 256, 4096
    expert_blocks = []
    with safe_open(SAFETENSORS, framework="pt", device="cpu") as sf:
        for li in range(18):
            pfx = f"{EXP_PFX}layers.{li}."
            w = {k[len(pfx):]: sf.get_tensor(k).float() for k in sf.keys() if k.startswith(pfx)}
            block = PTExpertBlock(EH, ENH, ENKV, EHD, EF); block.load_state_dict(w, strict=True); block.eval().to(DEVICE)
            expert_blocks.append(block)
    exp_norm_w = {}
    with safe_open(SAFETENSORS, framework="pt", device="cpu") as sf:
        for k in sf.keys():
            if k.startswith(f"{EXP_PFX}norm."): exp_norm_w[k[len(f"{EXP_PFX}norm."):]] = sf.get_tensor(k).float()
    exp_norm = PTAdaRMSFinal(EH, EH); exp_norm.load_state_dict(exp_norm_w, strict=True); exp_norm.eval().to(DEVICE)

    x_t = torch.from_numpy(NOISE_NP).to(DEVICE); dt = -1.0 / NUM_STEPS
    for step in range(NUM_STEPS):
        t_val = torch.tensor([1.0 + step * dt], dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            emb  = sinusoidal_emb_pt(t_val, 1024)
            cond = F.silu(time_in(emb)); cond = F.silu(time_out(cond))
            x_emb = action_in(x_t)
        cache_copy = [(kc.clone(), vc.clone()) for kc, vc in kv_cache]
        with torch.no_grad():
            h = x_emb
            for li, block in enumerate(expert_blocks):
                h, cache_copy[li] = block(h, cond, cache=cache_copy[li])
            h = exp_norm(h, cond)
            v_t = action_out(h[:, -CHUNK:].float())
        x_t = x_t + dt * v_t

    timings["expert"] = time.time() - t0
    timings["total"]  = time.time() - t_start
    return x_t.cpu().numpy(), timings


# ─────────────────────────────────────────────────────────────────────────────
#  MLX quantized — load once, run N times
# ─────────────────────────────────────────────────────────────────────────────

def mlx_benchmark(n_runs):
    import mlx.core as mx
    from mlx_pi05.load import load_model, SAFETENSORS_PATH, QUANTIZED_SAVE_PATH

    print(f"\n[MLX] Loading quantized model ...", flush=True)
    t_load = time.time()
    model = load_model(quantize=True)
    model.eval()
    load_time = time.time() - t_load
    print(f"  Loaded in {load_time:.2f}s")

    image_mlx = mx.array(IMAGE_NP)
    lang_mlx  = mx.array(LANG_IDS)
    noise_mlx = mx.array(NOISE_NP)

    run_times  = []
    prefix_times = []
    denoise_times = []

    for i in range(n_runs):
        t_run = time.time()

        t0 = time.time()
        prefix_kv = model.encode_prefix(image_mlx, lang_mlx)
        mx.eval(prefix_kv)
        prefix_times.append(time.time() - t0)

        t0 = time.time()
        x_t = noise_mlx.astype(mx.float32)
        dt  = -1.0 / NUM_STEPS
        for step in range(NUM_STEPS):
            t_val = mx.full((B,), 1.0 + step * dt, dtype=mx.float32)
            v_t   = model.denoise_step(x_t.astype(mx.float16), t_val, list(prefix_kv))
            x_t   = x_t + dt * v_t
            mx.eval(x_t)
        denoise_times.append(time.time() - t0)

        run_times.append(time.time() - t_run)
        print(f"  run {i+1}/{n_runs}: {run_times[-1]:.2f}s  (prefix={prefix_times[-1]:.2f}s  denoise={denoise_times[-1]:.2f}s)", flush=True)

    actions = np.array(x_t)
    return actions, load_time, run_times, prefix_times, denoise_times


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"PyTorch device : {DEVICE}")
    print(f"PyTorch runs   : {PT_RUNS}")
    print(f"MLX runs       : {MLX_RUNS}")
    print()

    # ── PyTorch ──────────────────────────────────────────────────────────────
    pt_runs_data = []
    for i in range(PT_RUNS):
        print(f"[PyTorch] run {i+1}/{PT_RUNS} ...", flush=True)
        _, t = pt_single_run()
        pt_runs_data.append(t)
        print(f"  siglip={t['siglip']:.1f}s  mproj={t['mproj_embed']:.1f}s  vlm={t['vlm']:.1f}s  expert={t['expert']:.1f}s  total={t['total']:.1f}s")
        gc.collect()

    # ── MLX ──────────────────────────────────────────────────────────────────
    _, mlx_load, mlx_runs, mlx_prefix, mlx_denoise = mlx_benchmark(MLX_RUNS)

    # ── Report ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  BENCHMARK REPORT")
    print("=" * 72)

    print(f"\n  PyTorch MPS float32 — {PT_RUNS} complete pipeline runs (load+infer each time)")
    print(f"  {'Run':<6}  {'SigLIP':>8}  {'mmproj':>8}  {'VLM':>8}  {'Expert':>8}  {'Total':>8}")
    for i, t in enumerate(pt_runs_data):
        print(f"  {i+1:<6}  {t['siglip']:>8.1f}s  {t['mproj_embed']:>7.1f}s  {t['vlm']:>7.1f}s  {t['expert']:>7.1f}s  {t['total']:>7.1f}s")
    pt_totals = [t["total"] for t in pt_runs_data]
    print(f"  {'mean':<6}  {'':>8}  {'':>8}  {'':>8}  {'':>8}  {sum(pt_totals)/len(pt_totals):>7.1f}s")

    print(f"\n  MLX 4-bit quantized — load once, {MLX_RUNS} inference runs")
    print(f"  Model load time: {mlx_load:.2f}s")
    print(f"  {'Run':<6}  {'Prefix(VLM)':>12}  {'Denoise(10s)':>13}  {'Total':>8}")
    for i in range(MLX_RUNS):
        print(f"  {i+1:<6}  {mlx_prefix[i]:>12.2f}s  {mlx_denoise[i]:>12.2f}s  {mlx_runs[i]:>7.2f}s")
    mean_run = sum(mlx_runs) / len(mlx_runs)
    mean_pre = sum(mlx_prefix) / len(mlx_prefix)
    mean_den = sum(mlx_denoise) / len(mlx_denoise)
    print(f"  {'mean':<6}  {mean_pre:>12.2f}s  {mean_den:>12.2f}s  {mean_run:>7.2f}s")

    print("\n  Summary")
    print(f"  {'':30}  {'Mean total':>12}")
    print(f"  {'PyTorch MPS (load+infer/run)':<30}  {sum(pt_totals)/len(pt_totals):>11.1f}s")
    print(f"  {'MLX quant (load amortised)':<30}  {(mlx_load/MLX_RUNS + mean_run):>11.2f}s")
    print(f"  {'MLX quant (model pre-loaded)':<30}  {mean_run:>11.2f}s")
    print("=" * 72)


if __name__ == "__main__":
    main()

"""
compare_e2e.py — Full end-to-end inference comparison: PyTorch MPS vs MLX.

Memory-safe strategy for 8 GB M1:
  Each component (SigLIP, VLM, Expert) is loaded, run, then freed before
  loading the next one.  Peak RAM = ~4 GB (VLM pass) + OS overhead.

Usage:
  # synthetic random inputs (fixed seed, no login needed)
  python3.11 compare_e2e.py

  # real LeRobot image + PaliGemma-tokenized task (requires HF login)
  python3.11 compare_e2e.py --real
  python3.11 compare_e2e.py --real --task "put the block in the bowl"
  python3.11 compare_e2e.py --real --dataset lerobot/pusht_image --img_key observation.image
"""

import argparse
import gc
import math
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"PyTorch device: {DEVICE}")

sys.path.insert(0, "/Users/mohan_007/Downloads/dare_to_experiment")

SAFETENSORS = (
    "/Users/mohan_007/Downloads/dare_to_experiment/pi05_cache/"
    "models--lerobot--pi05_base/snapshots/"
    "9e55186ad36e66b95cda57bc47818d9e6237ae30/model.safetensors"
)

B, CHUNK, ADIM = 1, 50, 32
NUM_STEPS = 10

# ─── Inputs — set by build_inputs() before inference runs ─────────────────────
IMAGE_NP  = None
LANG_IDS  = None
NOISE_NP  = None


def build_inputs_random():
    """Fixed-seed synthetic inputs — reproducible, no login needed."""
    global IMAGE_NP, LANG_IDS, NOISE_NP
    np.random.seed(42)
    torch.manual_seed(42)
    IMAGE_NP = np.random.uniform(-1, 1, (1, 3, 224, 224)).astype(np.float32)
    LANG_IDS = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=np.int32)
    NOISE_NP = np.random.randn(1, 50, 32).astype(np.float32)
    print("Inputs: synthetic (seed=42)")


def build_inputs_real(dataset: str, img_key: str, task: str):
    """Real LeRobot frame + PaliGemma-tokenized task."""
    global IMAGE_NP, LANG_IDS, NOISE_NP
    from PIL import Image
    from datasets import load_dataset
    from transformers import AutoTokenizer

    # ── image ─────────────────────────────────────────────────────────────────
    print(f"  Loading frame from {dataset} ...", flush=True)
    ds    = load_dataset(dataset, split="train", streaming=True)
    frame = next(iter(ds))
    avail = [k for k in frame if "image" in k.lower() or "cam" in k.lower()]
    if img_key not in frame:
        if not avail:
            raise KeyError(f"No image key found. Keys: {list(frame.keys())}")
        img_key = avail[0]
        print(f"  Note: using image key '{img_key}'")
    img   = frame[img_key].convert("RGB").resize((224, 224))
    arr   = np.array(img, dtype=np.float32) / 127.5 - 1.0
    IMAGE_NP = arr.transpose(2, 0, 1)[None]              # [1,3,224,224]

    task = task or frame.get("task") or "transfer the cube to the other side"

    # ── tokenizer ─────────────────────────────────────────────────────────────
    print(f"  Tokenizing: \"{task}\" ...", flush=True)
    try:
        tok = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
    except Exception as e:
        if "gated" in str(e).lower() or "401" in str(e) or "403" in str(e):
            print("\n" + "=" * 60)
            print("  TOKENIZER LOGIN REQUIRED")
            print("  1. Accept license: https://huggingface.co/google/paligemma-3b-pt-224")
            print("  2. Get token:      https://huggingface.co/settings/tokens")
            print("  3. Run:            huggingface-cli login")
            print("=" * 60)
        raise

    ids      = tok.encode(task, add_special_tokens=True)
    LANG_IDS = np.array([ids], dtype=np.int32)

    np.random.seed(0)
    NOISE_NP = np.random.randn(1, CHUNK, ADIM).astype(np.float32)

    print(f"Inputs: real  image={IMAGE_NP.shape}  tokens={LANG_IDS.shape}  task=\"{task}\"")

# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_raw(safetensors_prefix: str, dtype=torch.float32) -> dict[str, torch.Tensor]:
    """Load weights whose original key starts with safetensors_prefix."""
    out = {}
    with safe_open(SAFETENSORS, framework="pt", device="cpu") as f:
        for k in f.keys():
            if k.startswith(safetensors_prefix):
                out[k[len(safetensors_prefix):]] = f.get_tensor(k).to(dtype)
    return out


def free(*objs):
    for o in objs:
        del o
    gc.collect()


def sinusoidal_emb_pt(t: torch.Tensor, dim: int,
                      min_p: float = 4e-3, max_p: float = 4.0) -> torch.Tensor:
    half = dim // 2
    frac = torch.linspace(0., 1., half, dtype=torch.float32, device=t.device)
    period = min_p * (max_p / min_p) ** frac
    scale = 2.0 * math.pi / period
    arg = scale.unsqueeze(0) * t.float().unsqueeze(1)
    return torch.cat([arg.sin(), arg.cos()], dim=-1)


def apply_rope(x: torch.Tensor, offset: int = 0, base: float = 10000.) -> torch.Tensor:
    B, H, L, D = x.shape
    half = D // 2
    inv_freq = 1.0 / (base ** (torch.arange(half, dtype=torch.float32, device=x.device) / half))
    pos = torch.arange(offset, offset + L, dtype=torch.float32, device=x.device)
    freqs = torch.outer(pos, inv_freq).unsqueeze(0).unsqueeze(0)  # [1,1,L,half]
    cos_ = freqs.cos().to(x.dtype)
    sin_ = freqs.sin().to(x.dtype)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([cos_, cos_], dim=-1) * x + torch.cat([sin_, sin_], dim=-1) * torch.cat([-x2, x1], dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
#  PyTorch layers (mirror of mlx_pi05/)
# ─────────────────────────────────────────────────────────────────────────────

class PTRMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(d))
    def forward(self, x):
        orig = x.dtype
        x32 = x.float()
        return ((1 + self.weight) * x32 * torch.rsqrt((x32*x32).mean(-1,keepdim=True) + self.eps)).to(orig)


class PTGemmaAttn(nn.Module):
    def __init__(self, hidden, n_heads, n_kv, head_dim):
        super().__init__()
        self.nh, self.nkv, self.hd = n_heads, n_kv, head_dim
        self.scale = head_dim ** -0.5
        self.q_proj = nn.Linear(hidden, n_heads*head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, n_kv*head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, n_kv*head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads*head_dim, hidden, bias=False)

    def forward(self, x, cache=None):
        B, L, _ = x.shape
        q = self.q_proj(x).reshape(B,L,self.nh,self.hd).transpose(1,2)
        k = self.k_proj(x).reshape(B,L,self.nkv,self.hd).transpose(1,2)
        v = self.v_proj(x).reshape(B,L,self.nkv,self.hd).transpose(1,2)
        offset = 0
        if cache:
            kc, vc = cache
            offset = kc.shape[2]
            q = apply_rope(q, offset)
            k = apply_rope(k, offset)
            k = torch.cat([kc, k], dim=2)
            v = torch.cat([vc, v], dim=2)
        else:
            q = apply_rope(q)
            k = apply_rope(k)
        rep = self.nh // self.nkv
        ke = k.repeat_interleave(rep, 1)
        ve = v.repeat_interleave(rep, 1)
        a = (torch.matmul(q, ke.transpose(-2,-1)) * self.scale)
        a = F.softmax(a.float(), dim=-1).to(q.dtype)
        out = torch.matmul(a, ve).transpose(1,2).reshape(B, q.shape[2], -1)
        return self.o_proj(out), (k, v)


class PTGemmaMLP(nn.Module):
    def __init__(self, h, f):
        super().__init__()
        self.gate_proj = nn.Linear(h, f, bias=False)
        self.up_proj   = nn.Linear(h, f, bias=False)
        self.down_proj = nn.Linear(f, h, bias=False)
    def forward(self, x):
        return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))


class PTGemmaBlock(nn.Module):
    def __init__(self, h, nh, nkv, hd, f):
        super().__init__()
        self.input_layernorm = PTRMSNorm(h)
        self.self_attn = PTGemmaAttn(h, nh, nkv, hd)
        self.post_attention_layernorm = PTRMSNorm(h)
        self.mlp = PTGemmaMLP(h, f)
    def forward(self, x, cache=None):
        r, cache = self.self_attn(self.input_layernorm(x), cache)
        x = x + r
        return x + self.mlp(self.post_attention_layernorm(x)), cache


class PTAdaRMSNorm(nn.Module):
    def __init__(self, d, cd, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.dense = nn.Linear(cd, d*3, bias=True)
    def _n(self, x):
        x32 = x.float()
        return x32 * torch.rsqrt((x32*x32).mean(-1,keepdim=True) + self.eps)
    def forward(self, x, cond):
        n = self._n(x)
        mod = self.dense(cond.to(self.dense.weight.dtype)).unsqueeze(-2)
        s, sh, g = mod.chunk(3, -1)
        return (n*(1+s.float()) + sh.float()).to(x.dtype), g.to(x.dtype)


class PTAdaRMSFinal(nn.Module):
    def __init__(self, d, cd, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.dense = nn.Linear(cd, d*3, bias=True)
    def _n(self, x):
        x32 = x.float()
        return x32 * torch.rsqrt((x32*x32).mean(-1,keepdim=True) + self.eps)
    def forward(self, x, cond):
        n = self._n(x)
        mod = self.dense(cond.to(self.dense.weight.dtype)).unsqueeze(-2)
        s, sh, _ = mod.chunk(3, -1)
        return (n*(1+s.float()) + sh.float()).to(x.dtype)


class PTExpertBlock(nn.Module):
    def __init__(self, h, nh, nkv, hd, f):
        super().__init__()
        self.input_layernorm = PTAdaRMSNorm(h, h)
        self.self_attn = PTGemmaAttn(h, nh, nkv, hd)
        self.post_attention_layernorm = PTAdaRMSNorm(h, h)
        self.mlp = PTGemmaMLP(h, f)
    def forward(self, x, cond, cache=None):
        n, g = self.input_layernorm(x, cond)
        a, cache = self.self_attn(n, cache)
        x = x + a * g
        n, g = self.post_attention_layernorm(x, cond)
        return x + self.mlp(n) * g, cache


# SigLIP
class PTSigLIPAttn(nn.Module):
    def __init__(self, h=1152, nh=16):
        super().__init__()
        self.nh = nh; self.hd = h//nh; self.scale = self.hd**-0.5
        self.q_proj = nn.Linear(h,h,bias=True); self.k_proj = nn.Linear(h,h,bias=True)
        self.v_proj = nn.Linear(h,h,bias=True); self.out_proj = nn.Linear(h,h,bias=True)
    def forward(self, x):
        B,L,_ = x.shape
        q = self.q_proj(x).reshape(B,L,self.nh,self.hd).transpose(1,2)
        k = self.k_proj(x).reshape(B,L,self.nh,self.hd).transpose(1,2)
        v = self.v_proj(x).reshape(B,L,self.nh,self.hd).transpose(1,2)
        a = F.softmax((torch.matmul(q,k.transpose(-2,-1))*self.scale).float(),-1).to(q.dtype)
        return self.out_proj(torch.matmul(a,v).transpose(1,2).reshape(B,L,-1))

class PTSigLIPLayer(nn.Module):
    def __init__(self, h=1152, nh=16, f=4304):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(h, eps=1e-6); self.self_attn = PTSigLIPAttn(h,nh)
        self.layer_norm2 = nn.LayerNorm(h, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(h,f,bias=True), nn.GELU(approximate="tanh"), nn.Linear(f,h,bias=True))
    def forward(self, x):
        x = x + self.self_attn(self.layer_norm1(x))
        return x + self.mlp(self.layer_norm2(x))

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
        p = self.embeddings["patch_embedding"](x)
        B,C,H,W = p.shape
        p = p.flatten(2).transpose(1,2)
        p = p + self.embeddings["position_embedding"](torch.arange(H*W, device=x.device))
        for layer in self.encoder:
            p = layer(p)
        return self.post_layernorm(p)


# ─────────────────────────────────────────────────────────────────────────────
#  PyTorch end-to-end inference (chunked loading, CPU float32)
# ─────────────────────────────────────────────────────────────────────────────

def pt_infer():
    print("\n[PyTorch] Starting chunked CPU float32 inference...")
    VIS_PFX = "paligemma_with_expert.paligemma.model.vision_tower.vision_model."
    LM_PFX  = "paligemma_with_expert.paligemma.model.language_model."
    EXP_PFX = "paligemma_with_expert.gemma_expert.model."
    MPROJ   = "paligemma_with_expert.paligemma.model.multi_modal_projector.linear."
    LM_HEAD = "paligemma_with_expert.paligemma.lm_head.weight"

    # ── Step 1: SigLIP ──────────────────────────────────────────────────────
    t0 = time.time()
    print("  [1/4] SigLIP vision encoder ...", flush=True)
    raw = load_raw(VIS_PFX)
    # Remap: safetensors uses "encoder.layers.N" and "mlp.fc1/fc2"
    #        but PTSigLIP state dict uses "encoder.N" and "mlp.0/2"
    remapped_raw = {}
    for k, v in raw.items():
        k = k.replace("encoder.layers.", "encoder.")
        k = k.replace(".mlp.fc1.", ".mlp.0.")
        k = k.replace(".mlp.fc2.", ".mlp.2.")
        remapped_raw[k] = v
    siglip = PTSigLIP()
    sd = siglip.state_dict()
    new_sd = {}
    for k in sd:
        src = remapped_raw.get(k)
        if src is None:
            print(f"    MISSING: {k}")
            new_sd[k] = sd[k]
        else:
            new_sd[k] = src
    siglip.load_state_dict(new_sd, strict=True)
    siglip.eval().to(DEVICE)
    with torch.no_grad():
        img_feat = siglip(torch.from_numpy(IMAGE_NP).to(DEVICE))  # [1, 256, 1152]
    print(f"    img_feat: {img_feat.shape}  ({time.time()-t0:.1f}s)")
    free(siglip, raw, new_sd)

    # ── Step 2: mm_proj + embed_tokens ──────────────────────────────────────
    t1 = time.time()
    print("  [2/4] mm_proj + token embeddings ...", flush=True)
    mproj_raw = load_raw(MPROJ)
    lmhead_raw = {}
    with safe_open(SAFETENSORS, framework="pt", device="cpu") as f:
        lmhead_raw["embed_tokens.weight"] = f.get_tensor(LM_HEAD).float()

    mm_proj = nn.Linear(1152, 2048, bias=True)
    mm_proj.weight = nn.Parameter(mproj_raw["weight"])
    mm_proj.bias   = nn.Parameter(mproj_raw["bias"])
    mm_proj.eval().to(DEVICE)
    embed_tokens = nn.Embedding(257152, 2048)
    embed_tokens.weight = nn.Parameter(lmhead_raw["embed_tokens.weight"])
    embed_tokens.to(DEVICE)

    with torch.no_grad():
        img_emb  = (mm_proj(img_feat.to(DEVICE)) * math.sqrt(2048)).float()
        lang_emb = (embed_tokens(torch.from_numpy(LANG_IDS).long().to(DEVICE)) * math.sqrt(2048)).float()
        prefix   = torch.cat([img_emb, lang_emb], dim=1)               # [1, 266, 2048]
    S = prefix.shape[1]
    print(f"    prefix: {prefix.shape}  ({time.time()-t1:.1f}s)")
    free(img_feat, img_emb, lang_emb, mm_proj, embed_tokens, mproj_raw, lmhead_raw)

    # ── Step 3: VLM Gemma 2B → KV cache ─────────────────────────────────────
    t2 = time.time()
    print("  [3/4] VLM Gemma 2B (18 layers, ~4 GB) ...", flush=True)
    H, NH, NKV, HD, FFN_DIM = 2048, 8, 1, 256, 16384
    kv_cache = []
    x = prefix.to(DEVICE)

    with safe_open(SAFETENSORS, framework="pt", device="cpu") as sf:
        for li in range(18):
            if li % 6 == 0:
                print(f"    layer {li}/18 ...", flush=True)
            pfx = f"{LM_PFX}layers.{li}."
            w = {k[len(pfx):]: sf.get_tensor(k).float() for k in sf.keys() if k.startswith(pfx)}
            block = PTGemmaBlock(H, NH, NKV, HD, FFN_DIM)
            block.load_state_dict(w, strict=True)
            block.eval().to(DEVICE)
            with torch.no_grad():
                x, (k_c, v_c) = block(x, cache=None)
            kv_cache.append((k_c.detach(), v_c.detach()))
            free(block, w)

    # final norm
    norm_w = {}
    with safe_open(SAFETENSORS, framework="pt", device="cpu") as sf:
        norm_w["weight"] = sf.get_tensor(f"{LM_PFX}norm.weight").float()
    vlm_norm = PTRMSNorm(H)
    vlm_norm.load_state_dict(norm_w, strict=True)
    vlm_norm.eval().to(DEVICE)
    with torch.no_grad():
        x = vlm_norm(x)
    free(x, vlm_norm, norm_w)
    print(f"    KV cache: {len(kv_cache)} layers, K shape {kv_cache[0][0].shape}  ({time.time()-t2:.1f}s)")

    # ── Step 4: Action expert — 10-step denoising ────────────────────────────
    t3 = time.time()
    print("  [4/4] Action expert (18 layers × 10 steps) ...", flush=True)

    # projections
    proj_keys = ["action_in_proj.weight","action_in_proj.bias",
                 "action_out_proj.weight","action_out_proj.bias",
                 "time_mlp_in.weight","time_mlp_in.bias",
                 "time_mlp_out.weight","time_mlp_out.bias"]
    proj_w = {}
    with safe_open(SAFETENSORS, framework="pt", device="cpu") as sf:
        for k in proj_keys:
            proj_w[k] = sf.get_tensor(k).float()

    action_in  = nn.Linear(ADIM, 1024, bias=True)
    action_out = nn.Linear(1024, ADIM, bias=True)
    time_in    = nn.Linear(1024, 1024, bias=True)
    time_out   = nn.Linear(1024, 1024, bias=True)
    action_in.weight  = nn.Parameter(proj_w["action_in_proj.weight"])
    action_in.bias    = nn.Parameter(proj_w["action_in_proj.bias"])
    action_out.weight = nn.Parameter(proj_w["action_out_proj.weight"])
    action_out.bias   = nn.Parameter(proj_w["action_out_proj.bias"])
    time_in.weight    = nn.Parameter(proj_w["time_mlp_in.weight"])
    time_in.bias      = nn.Parameter(proj_w["time_mlp_in.bias"])
    time_out.weight   = nn.Parameter(proj_w["time_mlp_out.weight"])
    time_out.bias     = nn.Parameter(proj_w["time_mlp_out.bias"])
    action_in.to(DEVICE); action_out.to(DEVICE)
    time_in.to(DEVICE);   time_out.to(DEVICE)

    # load expert blocks
    EH, ENH, ENKV, EHD, EF = 1024, 8, 1, 256, 4096
    expert_blocks = []
    with safe_open(SAFETENSORS, framework="pt", device="cpu") as sf:
        for li in range(18):
            pfx = f"{EXP_PFX}layers.{li}."
            w = {k[len(pfx):]: sf.get_tensor(k).float() for k in sf.keys() if k.startswith(pfx)}
            block = PTExpertBlock(EH, ENH, ENKV, EHD, EF)
            block.load_state_dict(w, strict=True)
            block.eval().to(DEVICE)
            expert_blocks.append(block)

    # expert final norm
    exp_norm_w = {}
    with safe_open(SAFETENSORS, framework="pt", device="cpu") as sf:
        for k in sf.keys():
            if k.startswith(f"{EXP_PFX}norm."):
                exp_norm_w[k[len(f"{EXP_PFX}norm."):]] = sf.get_tensor(k).float()
    exp_norm = PTAdaRMSFinal(EH, EH)
    exp_norm.load_state_dict(exp_norm_w, strict=True)
    exp_norm.eval().to(DEVICE)

    # denoising loop
    x_t = torch.from_numpy(NOISE_NP).to(DEVICE)
    dt  = -1.0 / NUM_STEPS

    for step in range(NUM_STEPS):
        time_val = 1.0 + step * dt
        t_tensor = torch.tensor([time_val], dtype=torch.float32, device=DEVICE)

        # timestep conditioning
        with torch.no_grad():
            emb  = sinusoidal_emb_pt(t_tensor, 1024)
            cond = F.silu(time_in(emb))
            cond = F.silu(time_out(cond))              # [1, 1024]

            x_emb = action_in(x_t)                    # [1, 50, 1024]

        # run expert layers with VLM prefix KV appended
        cache_copy = [(kc.clone(), vc.clone()) for kc, vc in kv_cache]
        with torch.no_grad():
            h = x_emb
            for li, block in enumerate(expert_blocks):
                h, cache_copy[li] = block(h, cond, cache=cache_copy[li])
            h = exp_norm(h, cond)                       # [1, 50, 1024]
            v_t = action_out(h[:, -CHUNK:].float())     # [1, 50, 32]
        x_t = x_t + dt * v_t

    pt_actions = x_t.cpu().numpy()
    print(f"    PT actions: {pt_actions.shape}  ({time.time()-t3:.1f}s)")
    return pt_actions


# ─────────────────────────────────────────────────────────────────────────────
#  MLX end-to-end inference (float16, unquantized)
# ─────────────────────────────────────────────────────────────────────────────

def mlx_infer():
    import mlx.core as mx
    from mlx_pi05.load import load_model, SAFETENSORS_PATH

    print("\n[MLX] Loading model (float16, no quantization) ...", flush=True)
    t0 = time.time()
    model = load_model(safetensors_path=SAFETENSORS_PATH, quantize=False, save_after_load=False)
    model.eval()
    print(f"  Loaded in {time.time()-t0:.1f}s")

    print("  Running 10-step flow-matching inference ...", flush=True)
    t1 = time.time()
    image_mlx = mx.array(IMAGE_NP)
    lang_mlx  = mx.array(LANG_IDS)
    noise_mlx = mx.array(NOISE_NP)

    # Run inference with the SAME noise
    prefix_kv = model.encode_prefix(image_mlx, lang_mlx)
    mx.eval(prefix_kv)

    x_t = noise_mlx.astype(mx.float32)
    dt  = -1.0 / NUM_STEPS
    for step in range(NUM_STEPS):
        t_val = mx.full((B,), 1.0 + step * dt, dtype=mx.float32)
        v_t   = model.denoise_step(x_t.astype(mx.float16), t_val, list(prefix_kv))
        x_t   = x_t + dt * v_t
        mx.eval(x_t)

    mlx_actions = np.array(x_t)
    print(f"  Done in {time.time()-t1:.1f}s")
    return mlx_actions


# ─────────────────────────────────────────────────────────────────────────────
#  Compare
# ─────────────────────────────────────────────────────────────────────────────

def compare(pt: np.ndarray, mlx: np.ndarray):
    print("\n" + "=" * 65)
    print(f"  END-TO-END COMPARISON: PyTorch ({DEVICE} float32) vs MLX (float16)")
    print("=" * 65)

    pt_t  = torch.from_numpy(pt.astype(np.float32)).flatten()
    mlx_t = torch.from_numpy(mlx.astype(np.float32)).flatten()

    diff = (pt_t - mlx_t).abs()
    cos  = F.cosine_similarity(pt_t.unsqueeze(0), mlx_t.unsqueeze(0)).item()

    print(f"\n  Actions shape : {pt.shape}")
    print(f"\n  PyTorch  — mean={pt.mean():.5f}  std={pt.std():.5f}  max_abs={np.abs(pt).max():.5f}")
    print(f"  MLX      — mean={mlx.mean():.5f}  std={mlx.std():.5f}  max_abs={np.abs(mlx).max():.5f}")
    print(f"\n  max_abs_diff : {diff.max().item():.6f}")
    print(f"  mean_abs_diff: {diff.mean().item():.6f}")
    print(f"  cosine_sim   : {cos:.8f}")

    # Per-step comparison (first 3 time steps, first 6 dims)
    print("\n  First 3 timesteps × first 6 action dims:")
    print(f"  {'t':<4}  {'PT':>40}  {'MLX':>40}")
    for i in range(3):
        pt_row  = "  ".join(f"{v:+.4f}" for v in pt[0, i, :6])
        mlx_row = "  ".join(f"{v:+.4f}" for v in mlx[0, i, :6])
        print(f"  t={i}  [{pt_row}]  [{mlx_row}]")

    threshold = 0.20  # float32 vs float16 over 10 denoising steps
    passed = diff.max().item() < threshold
    print(f"\n  Threshold {threshold}: {'PASS ✓' if passed else 'FAIL ✗'}")
    print("=" * 65)
    return passed


def main():
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("--real",    action="store_true",
                        help="Use real LeRobot image + PaliGemma tokenizer (requires HF login)")
    parser.add_argument("--dataset", default="lerobot/aloha_sim_transfer_cube_human_image")
    parser.add_argument("--img_key", default="observation.images.top")
    parser.add_argument("--task",    default=None,
                        help="Task string for --real mode (default: from dataset or built-in)")
    args = parser.parse_args()

    if not os.path.isfile(SAFETENSORS):
        print(f"ERROR: safetensors not found at {SAFETENSORS}")
        sys.exit(1)

    # ── build inputs ──────────────────────────────────────────────────────────
    if args.real:
        print("\n[Inputs] Loading real LeRobot frame + tokenizing task ...")
        build_inputs_real(args.dataset, args.img_key, args.task)
    else:
        build_inputs_random()

    t_total = time.time()
    pt_actions  = pt_infer()

    gc.collect()

    mlx_actions = mlx_infer()

    passed = compare(pt_actions, mlx_actions)
    print(f"\n  Total wall time: {time.time()-t_total:.1f}s")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()

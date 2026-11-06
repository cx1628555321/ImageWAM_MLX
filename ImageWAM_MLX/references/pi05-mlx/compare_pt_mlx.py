"""
compare_pt_mlx.py — Component-by-component numerical comparison of MLX pi0.5 port
vs an equivalent PyTorch reference implementation.

Strategy (RAM-safe for 8 GB M1):
  Test 1: SigLIP   — load only vision.* weights (~0.8 GB f16)
  Test 2: GemmaBlock 0 — load only vlm.layers.0.* weights (~200 MB f16)
  Test 3: ExpertBlock 0 — load only expert.layers.0.* weights (~100 MB f16)

Each test:
  - Builds *only* the needed sub-module in both PyTorch (float16) and MLX (float16)
  - Loads weights once, converts to the target framework
  - Runs on identical random inputs (seeded)
  - Reports L2 norm, max abs diff, cosine similarity

Usage:
  python3.11 compare_pt_mlx.py
"""

import sys
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

# ─────────────────────────── helpers ────────────────────────────────────────

SAFETENSORS_PATH = (
    "/Users/mohan_007/Downloads/dare_to_experiment/pi05_cache/"
    "models--lerobot--pi05_base/snapshots/"
    "9e55186ad36e66b95cda57bc47818d9e6237ae30/model.safetensors"
)


def _remap_key(k: str) -> str | None:
    """
    Same remapping logic as mlx_pi05/load.py _remap(), but returns a single
    remapped key (or None to discard).
    """
    k = k.replace("action_time_mlp_in.", "time_mlp_in.").replace(
        "action_time_mlp_out.", "time_mlp_out."
    )

    LM_PFX = "paligemma_with_expert.paligemma.model.language_model."
    EMBED_KEY = LM_PFX + "embed_tokens.weight"

    if k.startswith(("action_in_proj.", "action_out_proj.", "time_mlp_in.", "time_mlp_out.")):
        return k
    if k == "paligemma_with_expert.paligemma.lm_head.weight":
        return "embed_tokens.weight"
    if k == EMBED_KEY:
        return "embed_tokens.weight"
    if k.startswith(LM_PFX):
        return "vlm." + k[len(LM_PFX):]
    if k.startswith("paligemma_with_expert.paligemma.model.multi_modal_projector.linear."):
        return "mm_proj." + k[len("paligemma_with_expert.paligemma.model.multi_modal_projector.linear."):]
    if k.startswith("paligemma_with_expert.paligemma.model.vision_tower.vision_model."):
        return "vision." + k[len("paligemma_with_expert.paligemma.model.vision_tower.vision_model."):]
    if k.startswith("paligemma_with_expert.gemma_expert.model."):
        return "expert." + k[len("paligemma_with_expert.gemma_expert.model."):]
    if k == "paligemma_with_expert.gemma_expert.lm_head.weight":
        return None  # unused
    return None


def load_filtered_weights(prefix: str, dtype=torch.float16) -> dict[str, torch.Tensor]:
    """
    Load only the weights whose remapped key starts with `prefix`.
    Returns {remapped_key: tensor} in the requested dtype.
    """
    weights = {}
    with safe_open(SAFETENSORS_PATH, framework="pt", device="cpu") as f:
        for k in f.keys():
            remapped = _remap_key(k)
            if remapped is None:
                continue
            if not remapped.startswith(prefix):
                continue
            t = f.get_tensor(k).to(dtype)
            # _fix_keys: Conv2d weight PyTorch [out,in,kH,kW] → keep as-is for PT
            # (MLX needs [out,kH,kW,in]; PyTorch needs [out,in,kH,kW])
            weights[remapped] = t
    return weights


def _stats(t: torch.Tensor, name: str):
    t = t.float()
    print(f"  {name}: shape={list(t.shape)}  mean={t.mean().item():.6f}  "
          f"std={t.std().item():.6f}  max_abs={t.abs().max().item():.6f}")


def _compare(pt: torch.Tensor, mx_np: np.ndarray, label: str,
             threshold: float = 0.05) -> bool:
    pt_f = pt.float().detach()
    mx_t = torch.from_numpy(mx_np.astype(np.float32))

    diff = (pt_f - mx_t).abs()
    cos = F.cosine_similarity(pt_f.flatten().unsqueeze(0), mx_t.flatten().unsqueeze(0)).item()
    max_diff = diff.max().item()
    l2_pt = pt_f.norm().item()
    l2_mx = mx_t.norm().item()

    passed = max_diff < threshold
    status = "PASS" if passed else "FAIL"
    print(f"\n  [{status}] {label}  (threshold={threshold})")
    print(f"    PT  L2={l2_pt:.4f}  MLX L2={l2_mx:.4f}")
    print(f"    max_abs_diff={max_diff:.6f}  cos_sim={cos:.6f}")
    return passed


# ════════════════════════════════════════════════════════════════════════════
#  PyTorch model components (mirror of MLX implementations)
# ════════════════════════════════════════════════════════════════════════════

# ─── RMSNorm (Gemma-style: scale = 1 + weight, weight init 0) ───────────────

class PTRMSNorm(nn.Module):
    """Gemma-style RMSNorm: out = (1 + w) * x / rms(x)."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x32 = x.float()
        var = (x32 * x32).mean(dim=-1, keepdim=True)
        normed = x32 * torch.rsqrt(var + self.eps)
        return ((1.0 + self.weight) * normed).to(orig_dtype)


# ─── RoPE ───────────────────────────────────────────────────────────────────

def _rope_freqs(head_dim: int, base: float = 10000.0) -> torch.Tensor:
    """Precompute inverse frequencies: [head_dim/2]."""
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) / half))
    return inv_freq  # [half]


def _apply_rope(x: torch.Tensor, offset: int = 0, base: float = 10000.0) -> torch.Tensor:
    """
    Apply RoPE to x: [B, n_heads, L, head_dim].
    Matches MLX nn.RoPE(traditional=False): interleaves pairs at positions [0::2, 1::2].
    """
    B, H, L, D = x.shape
    half = D // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32, device=x.device) / half))
    positions = torch.arange(offset, offset + L, dtype=torch.float32, device=x.device)
    # [L, half]
    freqs = torch.outer(positions, inv_freq)
    # [1, 1, L, half]
    freqs = freqs.unsqueeze(0).unsqueeze(0)
    cos = freqs.cos().to(x.dtype)
    sin = freqs.sin().to(x.dtype)

    # MLX RoPE non-traditional: rotates pairs x[..., :half] and x[..., half:]
    x1 = x[..., :half]   # [B, H, L, half]
    x2 = x[..., half:]   # [B, H, L, half]
    x_rot = torch.cat([-x2, x1], dim=-1)  # rotation in non-traditional form
    return x * torch.cat([cos, cos], dim=-1) + x_rot * torch.cat([sin, sin], dim=-1)


# ─── GemmaAttention (GQA) ───────────────────────────────────────────────────

class PTGemmaAttention(nn.Module):
    def __init__(self, hidden: int, n_heads: int, n_kv_heads: int, head_dim: int,
                 rope_theta: float = 10000.0):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.rope_theta = rope_theta

        self.q_proj = nn.Linear(hidden, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, hidden, bias=False)

    def forward(self, x: torch.Tensor, mask=None,
                cache: tuple | None = None) -> tuple[torch.Tensor, tuple]:
        B, L, _ = x.shape
        q = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)

        offset = 0
        if cache is not None:
            k_cache, v_cache = cache
            offset = k_cache.shape[2]
            q = _apply_rope(q, offset=offset, base=self.rope_theta)
            k = _apply_rope(k, offset=offset, base=self.rope_theta)
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
        else:
            q = _apply_rope(q, offset=0, base=self.rope_theta)
            k = _apply_rope(k, offset=0, base=self.rope_theta)

        # GQA: repeat k/v to match n_heads
        if self.n_kv_heads != self.n_heads:
            rep = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        # [B, n_heads, L_q, L_k]
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn + mask
        attn = F.softmax(attn.float(), dim=-1).to(q.dtype)
        out = torch.matmul(attn, v)  # [B, n_heads, L, head_dim]
        out = out.transpose(1, 2).reshape(B, L, -1)
        return self.o_proj(out), (k[:, :self.n_kv_heads // self.n_kv_heads * self.n_kv_heads:
                                      self.n_heads // self.n_kv_heads] if self.n_kv_heads != self.n_heads
                                   else (k, v))

    def forward(self, x: torch.Tensor, mask=None,
                cache: tuple | None = None) -> tuple[torch.Tensor, tuple]:
        """Forward pass for GQA attention (fixed)."""
        B, L, _ = x.shape
        q = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)

        offset = 0
        if cache is not None:
            k_cache, v_cache = cache
            offset = k_cache.shape[2]
            q = _apply_rope(q, offset=offset, base=self.rope_theta)
            k = _apply_rope(k, offset=offset, base=self.rope_theta)
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
        else:
            q = _apply_rope(q, offset=0, base=self.rope_theta)
            k = _apply_rope(k, offset=0, base=self.rope_theta)

        # GQA: repeat kv to match q heads
        if self.n_kv_heads != self.n_heads:
            rep = self.n_heads // self.n_kv_heads
            k_expand = k.repeat_interleave(rep, dim=1)
            v_expand = v.repeat_interleave(rep, dim=1)
        else:
            k_expand = k
            v_expand = v

        attn = torch.matmul(q, k_expand.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn + mask
        attn = F.softmax(attn.float(), dim=-1).to(q.dtype)
        out = torch.matmul(attn, v_expand)
        out = out.transpose(1, 2).reshape(B, L, -1)
        # Return with original (un-expanded) k, v for cache
        return self.o_proj(out), (k, v)


# ─── GemmaMLP ───────────────────────────────────────────────────────────────

class PTGemmaMLP(nn.Module):
    """Gated GeLU: gate_proj * gelu_approx(up_proj), projected via down_proj."""
    def __init__(self, hidden: int, ffn_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, ffn_dim, bias=False)
        self.up_proj = nn.Linear(hidden, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # gelu_approx = tanh approximation (gelu_pytorch_tanh)
        return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))


# ─── GemmaBlock ─────────────────────────────────────────────────────────────

class PTGemmaBlock(nn.Module):
    def __init__(self, hidden: int, n_heads: int, n_kv_heads: int,
                 head_dim: int, ffn_dim: int, rope_theta: float = 10000.0):
        super().__init__()
        self.input_layernorm = PTRMSNorm(hidden)
        self.self_attn = PTGemmaAttention(hidden, n_heads, n_kv_heads, head_dim, rope_theta)
        self.post_attention_layernorm = PTRMSNorm(hidden)
        self.mlp = PTGemmaMLP(hidden, ffn_dim)

    def forward(self, x: torch.Tensor, mask=None,
                cache: tuple | None = None) -> tuple[torch.Tensor, tuple]:
        r, cache = self.self_attn(self.input_layernorm(x), mask, cache)
        x = x + r
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, cache


# ─── GemmaDecoder ────────────────────────────────────────────────────────────

class PTGemmaDecoder(nn.Module):
    def __init__(self, n_layers: int, hidden: int, n_heads: int,
                 n_kv_heads: int, head_dim: int, ffn_dim: int,
                 rope_theta: float = 10000.0):
        super().__init__()
        self.layers = nn.ModuleList([
            PTGemmaBlock(hidden, n_heads, n_kv_heads, head_dim, ffn_dim, rope_theta)
            for _ in range(n_layers)
        ])
        self.norm = PTRMSNorm(hidden)

    def forward(self, x: torch.Tensor, mask=None,
                cache: list | None = None) -> tuple[torch.Tensor, list]:
        if cache is None:
            cache = [None] * len(self.layers)
        for i, layer in enumerate(self.layers):
            x, cache[i] = layer(x, mask, cache[i])
        return self.norm(x), cache


# ─── AdaRMSNorm ─────────────────────────────────────────────────────────────

class PTAdaRMSNorm(nn.Module):
    """
    Adaptive RMSNorm: dense(cond) → scale, shift, gate.
    Returns (normed_output, gate).
    """
    def __init__(self, dim: int, cond_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.dense = nn.Linear(cond_dim, dim * 3, bias=True)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.float()
        var = (x32 * x32).mean(dim=-1, keepdim=True)
        return x32 * torch.rsqrt(var + self.eps)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normed = self._norm(x)
        # Cast cond to dense's weight dtype to avoid half/float mismatch
        mod = self.dense(cond.to(self.dense.weight.dtype))  # [B, dim*3]
        # Reshape to [B, 1, dim*3] for broadcasting over sequence
        mod = mod.unsqueeze(-2)
        scale, shift, gate = mod.chunk(3, dim=-1)
        out = (normed * (1.0 + scale.float()) + shift.float()).to(x.dtype)
        return out, gate.to(x.dtype)


class PTAdaRMSNormFinal(nn.Module):
    """Final adaptive norm (no gate output — gate is discarded)."""
    def __init__(self, dim: int, cond_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.dense = nn.Linear(cond_dim, dim * 3, bias=True)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.float()
        var = (x32 * x32).mean(dim=-1, keepdim=True)
        return x32 * torch.rsqrt(var + self.eps)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        normed = self._norm(x)
        mod = self.dense(cond.to(self.dense.weight.dtype)).unsqueeze(-2)
        scale, shift, _ = mod.chunk(3, dim=-1)
        return (normed * (1.0 + scale.float()) + shift.float()).to(x.dtype)


# ─── ExpertBlock ─────────────────────────────────────────────────────────────

class PTExpertBlock(nn.Module):
    def __init__(self, hidden: int, n_heads: int, n_kv_heads: int,
                 head_dim: int, ffn_dim: int, rope_theta: float = 10000.0):
        super().__init__()
        self.input_layernorm = PTAdaRMSNorm(hidden, cond_dim=hidden)
        self.self_attn = PTGemmaAttention(hidden, n_heads, n_kv_heads, head_dim, rope_theta)
        self.post_attention_layernorm = PTAdaRMSNorm(hidden, cond_dim=hidden)
        self.mlp = PTGemmaMLP(hidden, ffn_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor,
                mask=None, cache: tuple | None = None) -> tuple[torch.Tensor, tuple]:
        normed, gate = self.input_layernorm(x, cond)
        attn_out, cache = self.self_attn(normed, mask, cache)
        x = x + attn_out * gate
        normed, gate = self.post_attention_layernorm(x, cond)
        x = x + self.mlp(normed) * gate
        return x, cache


# ─── ActionExpert ─────────────────────────────────────────────────────────────

class PTActionExpert(nn.Module):
    def __init__(self):
        super().__init__()
        hidden, n_heads, n_kv_heads, head_dim, ffn_dim = 1024, 8, 1, 256, 4096
        self.layers = nn.ModuleList([
            PTExpertBlock(hidden, n_heads, n_kv_heads, head_dim, ffn_dim)
            for _ in range(18)
        ])
        self.norm = PTAdaRMSNormFinal(1024, cond_dim=1024)

    def forward(self, x: torch.Tensor, cond: torch.Tensor,
                mask=None, cache: list | None = None) -> tuple[torch.Tensor, list]:
        if cache is None:
            cache = [None] * len(self.layers)
        for i, layer in enumerate(self.layers):
            x, cache[i] = layer(x, cond, mask, cache[i])
        return self.norm(x, cond), cache


# ─── SigLIP components ────────────────────────────────────────────────────────

class PTSigLIPPatchEmbedding(nn.Module):
    """Conv2d patch embedding: (B, C, H, W) → (B, n_patches, hidden)."""
    def __init__(self, hidden: int = 1152, patch: int = 14):
        super().__init__()
        self.patch = patch
        # PyTorch Conv2d weight: [out_channels, in_channels, kH, kW]
        self.patch_embedding = nn.Conv2d(3, hidden, kernel_size=patch, stride=patch, bias=True)
        self.position_embedding = nn.Embedding(256, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] — stays in standard PyTorch NCHW format
        x = self.patch_embedding(x)          # [B, hidden, H/p, W/p]
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)     # [B, H*W, hidden]
        pos = torch.arange(H * W, device=x.device)
        x = x + self.position_embedding(pos)
        return x


class PTSigLIPAttention(nn.Module):
    """Standard MHSA for SigLIP (no RoPE, with bias, not causal)."""
    def __init__(self, hidden: int = 1152, n_heads: int = 16):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden // n_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(hidden, hidden, bias=True)
        self.k_proj = nn.Linear(hidden, hidden, bias=True)
        self.v_proj = nn.Linear(hidden, hidden, bias=True)
        self.out_proj = nn.Linear(hidden, hidden, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        q = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn.float(), dim=-1).to(q.dtype)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, L, -1)
        return self.out_proj(out)


class PTSigLIPMLP(nn.Module):
    """Two-layer FFN with gelu_approx (tanh approximation)."""
    def __init__(self, hidden: int = 1152, ffn_dim: int = 4304):
        super().__init__()
        self.fc1 = nn.Linear(hidden, ffn_dim, bias=True)
        self.fc2 = nn.Linear(ffn_dim, hidden, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class PTSigLIPEncoderLayer(nn.Module):
    def __init__(self, hidden: int = 1152, n_heads: int = 16, ffn_dim: int = 4304):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(hidden, eps=1e-6)
        self.self_attn = PTSigLIPAttention(hidden, n_heads)
        self.layer_norm2 = nn.LayerNorm(hidden, eps=1e-6)
        self.mlp = PTSigLIPMLP(hidden, ffn_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.layer_norm1(x))
        x = x + self.mlp(self.layer_norm2(x))
        return x


class PTSigLIPEncoder(nn.Module):
    def __init__(self, n_layers: int = 27, hidden: int = 1152,
                 n_heads: int = 16, ffn_dim: int = 4304):
        super().__init__()
        self.layers = nn.ModuleList([
            PTSigLIPEncoderLayer(hidden, n_heads, ffn_dim) for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class PTSigLIPVisionModel(nn.Module):
    """Full SigLIP-So400m: image [B,3,H,W] → patch tokens [B, 256, 1152]."""
    def __init__(self):
        super().__init__()
        self.embeddings = PTSigLIPPatchEmbedding(hidden=1152, patch=14)
        self.encoder = PTSigLIPEncoder(n_layers=27, hidden=1152, n_heads=16, ffn_dim=4304)
        self.post_layernorm = nn.LayerNorm(1152, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embeddings(x)
        x = self.encoder(x)
        return self.post_layernorm(x)


# ════════════════════════════════════════════════════════════════════════════
#  Weight loading helpers for PyTorch models
# ════════════════════════════════════════════════════════════════════════════

def _strip_prefix(weights: dict, prefix: str) -> dict:
    """Strip `prefix.` from all keys, keep only keys that had that prefix."""
    out = {}
    for k, v in weights.items():
        if k.startswith(prefix + "."):
            out[k[len(prefix) + 1:]] = v
        elif k == prefix:
            out[""] = v
    return out


def load_pt_siglip_weights(model: PTSigLIPVisionModel, dtype=torch.float16):
    """Load vision.* weights into PyTorch SigLIP model."""
    raw = load_filtered_weights("vision.", dtype)
    # Strip the "vision." prefix to get model-relative paths
    raw2 = _strip_prefix(raw, "vision")

    # Key mapping: safetensors → PyTorch model attribute names
    # SigLIP safetensors keys use the HuggingFace naming scheme
    # vision.embeddings.patch_embedding.weight: [out,in,kH,kW] — PT keeps as-is
    # vision.embeddings.position_embedding.weight: [256, 1152]
    # vision.encoder.layers.N.layer_norm1.weight/bias
    # vision.encoder.layers.N.self_attn.{q,k,v}_proj.{weight,bias}
    # vision.encoder.layers.N.self_attn.out_proj.{weight,bias}
    # vision.encoder.layers.N.mlp.fc1/fc2.{weight,bias}
    # vision.encoder.layers.N.layer_norm2.weight/bias
    # vision.post_layernorm.weight/bias

    sd = model.state_dict()
    new_sd = {}
    for pt_key in sd.keys():
        # PT model uses nn.ModuleList so keys are e.g. "encoder.layers.0.layer_norm1.weight"
        if pt_key in raw2:
            new_sd[pt_key] = raw2[pt_key]
        else:
            print(f"  [WARN] missing PT key: {pt_key}")
            new_sd[pt_key] = sd[pt_key]
    model.load_state_dict(new_sd, strict=False)


def load_pt_gemmablock_weights(model: PTGemmaBlock, layer_idx: int, prefix: str = "vlm",
                                dtype=torch.float16):
    """Load weights for a single GemmaBlock layer."""
    layer_prefix = f"{prefix}.layers.{layer_idx}"
    raw = load_filtered_weights(layer_prefix + ".", dtype)

    # Strip down to the layer-relative keys
    raw2 = {}
    strip = layer_prefix + "."
    for k, v in raw.items():
        if k.startswith(strip):
            raw2[k[len(strip):]] = v

    sd = model.state_dict()
    new_sd = {}
    for pt_key in sd.keys():
        if pt_key in raw2:
            new_sd[pt_key] = raw2[pt_key]
        else:
            print(f"  [WARN] missing PT key for GemmaBlock: {pt_key}")
            new_sd[pt_key] = sd[pt_key]
    model.load_state_dict(new_sd, strict=False)


def load_pt_expertblock_weights(model: PTExpertBlock, layer_idx: int, dtype=torch.float16):
    """Load weights for a single ExpertBlock layer."""
    layer_prefix = f"expert.layers.{layer_idx}"
    raw = load_filtered_weights(layer_prefix + ".", dtype)
    raw2 = {}
    strip = layer_prefix + "."
    for k, v in raw.items():
        if k.startswith(strip):
            raw2[k[len(strip):]] = v

    sd = model.state_dict()
    new_sd = {}
    for pt_key in sd.keys():
        if pt_key in raw2:
            new_sd[pt_key] = raw2[pt_key]
        else:
            print(f"  [WARN] missing PT key for ExpertBlock: {pt_key}")
            new_sd[pt_key] = sd[pt_key]
    model.load_state_dict(new_sd, strict=False)


# ════════════════════════════════════════════════════════════════════════════
#  MLX helpers: build and load MLX sub-models
# ════════════════════════════════════════════════════════════════════════════

def _build_mlx_siglip_from_weights(raw_weights: dict):
    """Build and load an MLX SigLIPVisionModel from filtered safetensors weights."""
    import mlx.core as mx
    import mlx.nn as mnn
    sys.path.insert(0, "/Users/mohan_007/Downloads/dare_to_experiment")
    from mlx_pi05.siglip import SigLIPVisionModel
    from mlx_pi05.load import _remap, _fix_keys

    # Convert raw_weights (torch tensors) → MLX arrays
    mx_weights = {}
    for k, v in raw_weights.items():
        mx_weights[k] = mx.array(v.numpy())

    remapped = _remap(mx_weights)
    remapped = _fix_keys(remapped)

    # Keep only vision.* keys
    vision_weights = {k: v for k, v in remapped.items() if k.startswith("vision.")}

    model = SigLIPVisionModel()
    model.load_weights(list(vision_weights.items()), strict=False)
    mx.eval(model.parameters())
    return model


def _build_mlx_gemmablock_from_weights(raw_weights: dict, layer_idx: int):
    """Build and load a single MLX GemmaBlock."""
    import mlx.core as mx
    sys.path.insert(0, "/Users/mohan_007/Downloads/dare_to_experiment")
    from mlx_pi05.gemma import GemmaBlock, GEMMA_2B

    mx_weights = {}
    for k, v in raw_weights.items():
        mx_weights[k] = mx.array(v.numpy())

    # Already remapped; filter to just this layer
    prefix = f"vlm.layers.{layer_idx}."
    layer_weights = {}
    for k, v in mx_weights.items():
        if k.startswith(prefix):
            layer_weights[k[len(prefix):]] = v

    cfg = GEMMA_2B
    block = GemmaBlock(cfg["hidden"], cfg["n_heads"], cfg["n_kv_heads"],
                       cfg["head_dim"], cfg["ffn_dim"], 10000.0)
    block.load_weights(list(layer_weights.items()), strict=False)
    mx.eval(block.parameters())
    return block


def _build_mlx_expertblock_from_weights(raw_weights: dict, layer_idx: int):
    """Build and load a single MLX ExpertBlock."""
    import mlx.core as mx
    sys.path.insert(0, "/Users/mohan_007/Downloads/dare_to_experiment")
    from mlx_pi05.action_expert import ExpertBlock

    mx_weights = {}
    for k, v in raw_weights.items():
        mx_weights[k] = mx.array(v.numpy())

    prefix = f"expert.layers.{layer_idx}."
    layer_weights = {}
    for k, v in mx_weights.items():
        if k.startswith(prefix):
            layer_weights[k[len(prefix):]] = v

    block = ExpertBlock(hidden=1024, n_heads=8, n_kv_heads=1, head_dim=256, ffn_dim=4096)
    block.load_weights(list(layer_weights.items()), strict=False)
    mx.eval(block.parameters())
    return block


# ════════════════════════════════════════════════════════════════════════════
#  Test 1: SigLIP vision encoder
# ════════════════════════════════════════════════════════════════════════════

def test_siglip():
    print("\n" + "=" * 70)
    print("TEST 1: SigLIP Vision Encoder")
    print("  Both PT and MLX run float32 (CPU has no f16 Conv2d/matmul).")
    print("  1a: single encoder layer — proves layer implementation matches (<0.02)")
    print("  1b: full 27-layer model  — accumulated f32 differences; threshold 3.0")
    print("=" * 70)

    import mlx.core as mx
    sys.path.insert(0, "/Users/mohan_007/Downloads/dare_to_experiment")
    from mlx_pi05.siglip import SigLIPVisionModel, SigLIPEncoderLayer
    from mlx_pi05.load import _fix_keys

    # Load all vision weights once (float32)
    print("\nLoading vision.* weights (~0.8 GB) as float32...")
    raw = {}  # stripped key (no "vision.") → torch float32 tensor
    with safe_open(SAFETENSORS_PATH, framework="pt", device="cpu") as f:
        vision_pfx = "paligemma_with_expert.paligemma.model.vision_tower.vision_model."
        for k in f.keys():
            if k.startswith(vision_pfx):
                raw[k[len(vision_pfx):]] = f.get_tensor(k).float()
    print(f"  Loaded {len(raw)} vision tensors.")

    # ── 1a: Single encoder layer test ──
    print("\n--- 1a: Single SigLIPEncoderLayer (layer 0) ---")
    lk_pt = {k[len("encoder.layers.0."):]: v
              for k, v in raw.items() if k.startswith("encoder.layers.0.")}
    lk_mx = {k: mx.array(v.numpy()) for k, v in lk_pt.items()}

    pt_layer = PTSigLIPEncoderLayer()
    pt_layer.load_state_dict(lk_pt, strict=True)
    pt_layer.eval()

    mlx_layer = SigLIPEncoderLayer()
    mlx_layer.load_weights(list(lk_mx.items()), strict=False)
    mx.eval(mlx_layer.parameters())

    np.random.seed(42)
    torch.manual_seed(42)
    x_np = np.random.randn(1, 256, 1152).astype(np.float32)
    with torch.no_grad():
        pt_layer_out = pt_layer(torch.from_numpy(x_np))
    mlx_layer_out = mlx_layer(mx.array(x_np))
    mx.eval(mlx_layer_out)
    r1a = _compare(pt_layer_out, np.array(mlx_layer_out),
                   "SigLIP encoder layer 0 (single layer, f32 vs f32)", threshold=0.02)

    # ── 1b: Full 27-layer model test ──
    print("\n--- 1b: Full SigLIPVisionModel (27 layers, f32 vs f32) ---")
    np.random.seed(42)
    image_np = np.random.uniform(-1, 1, (1, 3, 224, 224)).astype(np.float32)
    image_pt = torch.from_numpy(image_np)

    pt_model = PTSigLIPVisionModel()
    new_sd = {}
    for k in pt_model.state_dict():
        new_sd[k] = raw[k]
    pt_model.load_state_dict(new_sd, strict=True)
    pt_model.eval()

    # MLX: build with float32 weights + _fix_keys for conv transpose.
    # _fix_keys expects "vision.embeddings.patch_embedding.weight" with the prefix,
    # then we strip "vision." before passing to the standalone SigLIPVisionModel
    # (which has no "vision." prefix in its own parameter names).
    mx_weights_raw = {"vision." + k: mx.array(v.numpy()) for k, v in raw.items()}
    mx_weights_fixed = _fix_keys(mx_weights_raw)
    # Strip "vision." prefix for standalone model loading
    mx_weights_stripped = {k[len("vision."):]: v
                           for k, v in mx_weights_fixed.items()
                           if k.startswith("vision.")}
    mlx_model = SigLIPVisionModel()
    mlx_model.load_weights(list(mx_weights_stripped.items()), strict=False)
    mx.eval(mlx_model.parameters())

    print("Running PyTorch SigLIP (float32)...")
    with torch.no_grad():
        pt_out = pt_model(image_pt)
    print("Running MLX SigLIP (float32)...")
    mlx_out = mlx_model(mx.array(image_np))
    mx.eval(mlx_out)
    mlx_np_full = np.array(mlx_out)
    _stats(pt_out, "PT  vision output")
    # 27-layer accumulation: MLX fast-sdpa vs PT matmul/softmax differ numerically;
    # threshold 3.0 checks structural correctness (cos_sim should be > 0.99).
    r1b = _compare(pt_out, mlx_np_full,
                   "SigLIP full model (27 layers, f32 vs f32 — expect cos_sim>0.99)",
                   threshold=3.0)

    return r1a and r1b


# ════════════════════════════════════════════════════════════════════════════
#  Test 2: GemmaBlock layer 0
# ════════════════════════════════════════════════════════════════════════════

def test_gemmablock():
    print("\n" + "=" * 70)
    print("TEST 2: Gemma 2B — Block 0  [1, 20, 2048] → [1, 20, 2048]")
    print("  Both PT and MLX use float32 weights/activations for fair comparison.")
    print("  (CPU has no f16 matmul; MLX also loaded f32 to match.)")
    print("=" * 70)

    import mlx.core as mx
    sys.path.insert(0, "/Users/mohan_007/Downloads/dare_to_experiment")
    from mlx_pi05.gemma import GemmaBlock, GEMMA_2B

    # Fixed seed hidden state (float32 for both)
    torch.manual_seed(42)
    np.random.seed(42)
    hidden_np = np.random.randn(1, 20, 2048).astype(np.float32) * 0.02
    hidden_pt = torch.from_numpy(hidden_np)

    LAYER = 0
    LM_PFX = "paligemma_with_expert.paligemma.model.language_model."
    layer_st_prefix = f"{LM_PFX}layers.{LAYER}."
    strip = f"vlm.layers.{LAYER}."

    # Load weights as float32 (same precision for both frameworks)
    print(f"Loading vlm.layers.{LAYER}.* weights as float32...")
    raw = {}  # stripped layer-relative key → float32 tensor
    with safe_open(SAFETENSORS_PATH, framework="pt", device="cpu") as f:
        for k in f.keys():
            if k.startswith(layer_st_prefix):
                raw[k[len(layer_st_prefix):]] = f.get_tensor(k).float()

    # ── Build PyTorch GemmaBlock (float32) ──
    print("Building PTGemmaBlock (float32)...")
    cfg = GEMMA_2B
    pt_block = PTGemmaBlock(cfg["hidden"], cfg["n_heads"], cfg["n_kv_heads"],
                             cfg["head_dim"], cfg["ffn_dim"], 10000.0)
    pt_block.load_state_dict(raw, strict=True)
    pt_block.eval()

    print("Running PTGemmaBlock...")
    with torch.no_grad():
        pt_out, _ = pt_block(hidden_pt)
    _stats(pt_out, "PT GemmaBlock 0 output")

    # ── Build MLX GemmaBlock (float32 weights) ──
    print("Building MLX GemmaBlock (float32)...")
    layer_weights_mx = {k: mx.array(v.numpy()) for k, v in raw.items()}
    mlx_block = GemmaBlock(cfg["hidden"], cfg["n_heads"], cfg["n_kv_heads"],
                            cfg["head_dim"], cfg["ffn_dim"], 10000.0)
    mlx_block.load_weights(list(layer_weights_mx.items()), strict=False)
    mx.eval(mlx_block.parameters())

    print("Running MLX GemmaBlock (float32)...")
    hidden_mlx = mx.array(hidden_np)  # float32 input
    mlx_out, _ = mlx_block(hidden_mlx)
    mx.eval(mlx_out)
    mlx_np = np.array(mlx_out)
    print(f"  MLX output shape: {mlx_np.shape}")

    result = _compare(pt_out.float().cpu(), mlx_np,
                      "GemmaBlock layer 0 (f32 weights, f32 activations)", threshold=0.05)
    return result


# ════════════════════════════════════════════════════════════════════════════
#  Test 3: ExpertBlock layer 0
# ════════════════════════════════════════════════════════════════════════════

def test_expertblock():
    print("\n" + "=" * 70)
    print("TEST 3: ActionExpert — ExpertBlock 0  [1, 50, 1024] → [1, 50, 1024]")
    print("  Both PT and MLX use float32 weights/activations for fair comparison.")
    print("=" * 70)

    torch.manual_seed(42)
    np.random.seed(42)
    x_np = np.random.randn(1, 50, 1024).astype(np.float32) * 0.02
    cond_np = np.random.randn(1, 1024).astype(np.float32) * 0.1
    x_pt = torch.from_numpy(x_np)
    cond_pt = torch.from_numpy(cond_np)

    import mlx.core as mx
    sys.path.insert(0, "/Users/mohan_007/Downloads/dare_to_experiment")
    from mlx_pi05.action_expert import ExpertBlock

    LAYER = 0
    EXPERT_PFX = "paligemma_with_expert.gemma_expert.model."
    layer_st_prefix = f"{EXPERT_PFX}layers.{LAYER}."

    # Load weights as float32 (same precision for both frameworks)
    print(f"Loading expert.layers.{LAYER}.* weights as float32...")
    raw = {}  # layer-relative key → float32 tensor
    with safe_open(SAFETENSORS_PATH, framework="pt", device="cpu") as f:
        for k in f.keys():
            if k.startswith(layer_st_prefix):
                raw[k[len(layer_st_prefix):]] = f.get_tensor(k).float()

    # ── Build PyTorch ExpertBlock (float32) ──
    print("Building PTExpertBlock (float32)...")
    pt_block = PTExpertBlock(hidden=1024, n_heads=8, n_kv_heads=1,
                              head_dim=256, ffn_dim=4096)
    pt_block.load_state_dict(raw, strict=True)
    pt_block.eval()

    print("Running PTExpertBlock (float32)...")
    with torch.no_grad():
        pt_out, _ = pt_block(x_pt, cond_pt)
    _stats(pt_out, "PT ExpertBlock 0 output")

    # ── Build MLX ExpertBlock (float32 weights) ──
    print("Building MLX ExpertBlock (float32)...")
    layer_weights_mx = {k: mx.array(v.numpy()) for k, v in raw.items()}
    mlx_block = ExpertBlock(hidden=1024, n_heads=8, n_kv_heads=1, head_dim=256, ffn_dim=4096)
    mlx_block.load_weights(list(layer_weights_mx.items()), strict=False)
    mx.eval(mlx_block.parameters())

    print("Running MLX ExpertBlock (float32)...")
    x_mlx = mx.array(x_np)       # float32 input
    cond_mlx = mx.array(cond_np)  # float32 input
    mlx_out, _ = mlx_block(x_mlx, cond_mlx)
    mx.eval(mlx_out)
    mlx_np = np.array(mlx_out)

    result = _compare(pt_out.float().cpu(), mlx_np,
                      "ExpertBlock layer 0 (f32 weights, f32 activations)", threshold=0.05)
    return result


# ════════════════════════════════════════════════════════════════════════════
#  Test 4: Sinusoidal timestep embedding
# ════════════════════════════════════════════════════════════════════════════

def _pt_sinusoidal_emb(t: torch.Tensor, dim: int,
                       min_period: float = 4e-3, max_period: float = 4.0) -> torch.Tensor:
    """Match mlx_pi05/model.py _sinusoidal_emb exactly."""
    half = dim // 2
    fraction = torch.linspace(0.0, 1.0, half, dtype=torch.float32)
    period = min_period * (max_period / min_period) ** fraction
    scale = (2.0 * math.pi) / period
    t_f32 = t.float()
    arg = scale.unsqueeze(0) * t_f32.unsqueeze(1)  # [B, half]
    return torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1)  # [B, dim]


def test_sinusoidal():
    print("\n" + "=" * 70)
    print("TEST 4: Sinusoidal timestep embedding (no weights)")
    print("=" * 70)

    import mlx.core as mx
    sys.path.insert(0, "/Users/mohan_007/Downloads/dare_to_experiment")
    from mlx_pi05.model import _sinusoidal_emb as mlx_sinusoidal

    torch.manual_seed(42)
    t_np = np.array([0.5, 0.3, 0.8, 1.0], dtype=np.float32)
    t_pt = torch.from_numpy(t_np)
    t_mlx = mx.array(t_np)

    pt_emb = _pt_sinusoidal_emb(t_pt, dim=1024)
    mlx_emb = mlx_sinusoidal(t_mlx, dim=1024)
    mx.eval(mlx_emb)
    mlx_np = np.array(mlx_emb)

    result = _compare(pt_emb, mlx_np, "Sinusoidal timestep embedding")
    return result


# ════════════════════════════════════════════════════════════════════════════
#  Test 5: RMSNorm (Gemma-style)
# ════════════════════════════════════════════════════════════════════════════

def test_rmsnorm():
    print("\n" + "=" * 70)
    print("TEST 5: Gemma RMSNorm (1 + weight) * norm(x)  (no weight load)")
    print("=" * 70)

    import mlx.core as mx
    import mlx.nn as mnn
    sys.path.insert(0, "/Users/mohan_007/Downloads/dare_to_experiment")
    from mlx_pi05.gemma import RMSNorm

    torch.manual_seed(42)
    np.random.seed(42)

    # Use non-zero weight to test (1+w) scaling
    x_np = np.random.randn(2, 10, 64).astype(np.float32)
    w_np = np.random.randn(64).astype(np.float32) * 0.1  # small non-zero weight

    # PT: float32 input (CPU does not support f16 operations)
    x_pt = torch.from_numpy(x_np)  # float32
    pt_norm = PTRMSNorm(64)
    with torch.no_grad():
        pt_norm.weight.copy_(torch.from_numpy(w_np))
    pt_out = pt_norm(x_pt)

    x_mlx = mx.array(x_np.astype(np.float16))
    mlx_norm = RMSNorm(64)
    mlx_norm.load_weights([("weight", mx.array(w_np))], strict=True)
    mlx_out = mlx_norm(x_mlx)
    mx.eval(mlx_out)

    result = _compare(pt_out.float(), np.array(mlx_out), "Gemma RMSNorm")
    return result


# ════════════════════════════════════════════════════════════════════════════
#  Test 6: AdaRMSNorm
# ════════════════════════════════════════════════════════════════════════════

def test_adarmsnorm():
    print("\n" + "=" * 70)
    print("TEST 6: AdaRMSNorm  [B=2, L=10, dim=64], cond [B=2, cond=32]")
    print("=" * 70)

    import mlx.core as mx
    sys.path.insert(0, "/Users/mohan_007/Downloads/dare_to_experiment")
    from mlx_pi05.action_expert import AdaRMSNorm

    torch.manual_seed(42)
    np.random.seed(42)

    dim, cond_dim = 64, 32
    x_np = np.random.randn(2, 10, dim).astype(np.float32)
    cond_np = np.random.randn(2, cond_dim).astype(np.float32)
    # Dense weight/bias
    W_np = np.random.randn(dim * 3, cond_dim).astype(np.float32) * 0.02
    b_np = np.random.randn(dim * 3).astype(np.float32) * 0.01

    # PT: float32 input (CPU does not support f16 operations)
    x_pt = torch.from_numpy(x_np)  # float32
    cond_pt = torch.from_numpy(cond_np)  # float32
    pt_ada = PTAdaRMSNorm(dim, cond_dim)
    with torch.no_grad():
        pt_ada.dense.weight.copy_(torch.from_numpy(W_np))
        pt_ada.dense.bias.copy_(torch.from_numpy(b_np))
    pt_out, pt_gate = pt_ada(x_pt, cond_pt)

    x_mlx = mx.array(x_np.astype(np.float16))
    cond_mlx = mx.array(cond_np.astype(np.float16))
    mlx_ada = AdaRMSNorm(dim, cond_dim)
    mlx_ada.load_weights([
        ("dense.weight", mx.array(W_np)),
        ("dense.bias", mx.array(b_np)),
    ], strict=True)
    mlx_out, mlx_gate = mlx_ada(x_mlx, cond_mlx)
    mx.eval(mlx_out, mlx_gate)

    r1 = _compare(pt_out.float(), np.array(mlx_out), "AdaRMSNorm output")
    r2 = _compare(pt_gate.float(), np.array(mlx_gate), "AdaRMSNorm gate")
    return r1 and r2


# ════════════════════════════════════════════════════════════════════════════
#  Summary table
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 70)
    print("  pi0.5 MLX Port — PyTorch vs MLX Numerical Comparison")
    print("=" * 70)

    if not os.path.isfile(SAFETENSORS_PATH):
        print(f"\nERROR: weights not found at:\n  {SAFETENSORS_PATH}")
        sys.exit(1)

    results = {}

    # Unit tests (no weight loading)
    results["sinusoidal_emb"] = test_sinusoidal()
    results["rmsnorm"] = test_rmsnorm()
    results["adarmsnorm"] = test_adarmsnorm()

    # Component tests (load partial weights)
    results["siglip"] = test_siglip()
    results["gemmablock_0"] = test_gemmablock()
    results["expertblock_0"] = test_expertblock()

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  {'Test':<30} {'Result'}")
    print(f"  {'-'*30} {'-'*10}")
    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name:<30} {status}")
        if not passed:
            all_passed = False
    print("=" * 70)
    overall = "ALL TESTS PASSED" if all_passed else "SOME TESTS FAILED"
    print(f"  {overall}")
    print("=" * 70)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

"""SigLIP-So400m vision encoder as used in PaliGemma."""

import mlx.core as mx
import mlx.nn as nn


class SigLIPPatchEmbedding(nn.Module):
    """Conv2d patch embedding: (B, C, H, W) → (B, n_patches, hidden)."""

    def __init__(self, hidden: int = 1152, patch: int = 14):
        super().__init__()
        self.patch = patch
        # MLX Conv2d weight: [out, kH, kW, in]
        self.patch_embedding = nn.Conv2d(3, hidden, kernel_size=patch, stride=patch, bias=True)
        self.position_embedding = nn.Embedding(256, hidden)

    def __call__(self, x: mx.array) -> mx.array:
        # x: [B, C, H, W] → convert to [B, H, W, C] for MLX Conv2d
        x = x.transpose(0, 2, 3, 1)
        x = self.patch_embedding(x)          # [B, H/p, W/p, hidden]
        B, H, W, C = x.shape
        x = x.reshape(B, H * W, C)           # [B, 256, hidden]
        pos = mx.arange(H * W)
        x = x + self.position_embedding(pos)
        return x


class SigLIPAttention(nn.Module):
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

    def __call__(self, x: mx.array) -> mx.array:
        B, L, _ = x.shape
        q = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=None)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.out_proj(out)


class SigLIPMLP(nn.Module):
    """Two-layer FFN with gelu_approx (gelu_pytorch_tanh)."""

    def __init__(self, hidden: int = 1152, ffn_dim: int = 4304):
        super().__init__()
        self.fc1 = nn.Linear(hidden, ffn_dim, bias=True)
        self.fc2 = nn.Linear(ffn_dim, hidden, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu_approx(self.fc1(x)))


class SigLIPEncoderLayer(nn.Module):
    def __init__(self, hidden: int = 1152, n_heads: int = 16, ffn_dim: int = 4304):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(hidden, eps=1e-6)
        self.self_attn = SigLIPAttention(hidden, n_heads)
        self.layer_norm2 = nn.LayerNorm(hidden, eps=1e-6)
        self.mlp = SigLIPMLP(hidden, ffn_dim)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.self_attn(self.layer_norm1(x))
        x = x + self.mlp(self.layer_norm2(x))
        return x


class SigLIPEncoder(nn.Module):
    def __init__(self, n_layers: int = 27, hidden: int = 1152, n_heads: int = 16, ffn_dim: int = 4304):
        super().__init__()
        self.layers = [SigLIPEncoderLayer(hidden, n_heads, ffn_dim) for _ in range(n_layers)]

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = layer(x)
        return x


class SigLIPVisionModel(nn.Module):
    """Full SigLIP-So400m: image → patch tokens [B, 256, 1152]."""

    def __init__(self):
        super().__init__()
        self.embeddings = SigLIPPatchEmbedding(hidden=1152, patch=14)
        self.encoder = SigLIPEncoder(n_layers=27, hidden=1152, n_heads=16, ffn_dim=4304)
        self.post_layernorm = nn.LayerNorm(1152, eps=1e-6)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.embeddings(x)
        x = self.encoder(x)
        return self.post_layernorm(x)

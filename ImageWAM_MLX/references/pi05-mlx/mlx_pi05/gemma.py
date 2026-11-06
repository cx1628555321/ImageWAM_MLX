"""Gemma 2B decoder for the PaliGemma VLM backbone (prefix encoder)."""

import math
import mlx.core as mx
import mlx.nn as nn


class RMSNorm(nn.Module):
    """Gemma-style RMSNorm: scale = (1 + weight), weight initialised to 0."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.zeros((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        x32 = x.astype(mx.float32)
        var = mx.mean(x32 * x32, axis=-1, keepdims=True)
        normed = x32 * mx.rsqrt(var + self.eps)
        return ((1.0 + self.weight) * normed).astype(x.dtype)


class GemmaAttention(nn.Module):
    """GQA attention: 8 Q heads, 1 KV head, head_dim=256, no bias."""

    def __init__(self, hidden: int, n_heads: int, n_kv_heads: int, head_dim: int, rope_theta: float = 10000.0):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(hidden, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, hidden, bias=False)
        self.rope = nn.RoPE(head_dim, traditional=False, base=rope_theta)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: tuple[mx.array, mx.array] | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array]]:
        B, L, _ = x.shape

        q = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        if cache is not None:
            k_cache, v_cache = cache
            offset = k_cache.shape[2]
            q = self.rope(q, offset=offset)
            k = self.rope(k, offset=offset)
            k = mx.concatenate([k_cache, k], axis=2)
            v = mx.concatenate([v_cache, v], axis=2)
        else:
            q = self.rope(q)
            k = self.rope(k)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out), (k, v)


class GemmaMLP(nn.Module):
    """Gated GeLU MLP: gate_proj * gelu_approx(up_proj)."""

    def __init__(self, hidden: int, ffn_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, ffn_dim, bias=False)
        self.up_proj = nn.Linear(hidden, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, hidden, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.gelu_approx(self.gate_proj(x)) * self.up_proj(x))


class GemmaBlock(nn.Module):
    def __init__(self, hidden: int, n_heads: int, n_kv_heads: int, head_dim: int, ffn_dim: int, rope_theta: float):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden)
        self.self_attn = GemmaAttention(hidden, n_heads, n_kv_heads, head_dim, rope_theta)
        self.post_attention_layernorm = RMSNorm(hidden)
        self.mlp = GemmaMLP(hidden, ffn_dim)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: tuple | None = None,
    ) -> tuple[mx.array, tuple]:
        r, cache = self.self_attn(self.input_layernorm(x), mask, cache)
        x = x + r
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, cache


class GemmaDecoder(nn.Module):
    """Gemma 2B / 300M decoder, no embed_tokens (receives embeddings directly)."""

    def __init__(self, n_layers: int, hidden: int, n_heads: int, n_kv_heads: int, head_dim: int, ffn_dim: int, rope_theta: float = 10000.0):
        super().__init__()
        self.layers = [GemmaBlock(hidden, n_heads, n_kv_heads, head_dim, ffn_dim, rope_theta) for _ in range(n_layers)]
        self.norm = RMSNorm(hidden)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: list | None = None,
    ) -> tuple[mx.array, list]:
        if cache is None:
            cache = [None] * len(self.layers)
        for i, layer in enumerate(self.layers):
            x, cache[i] = layer(x, mask, cache[i])
        return self.norm(x), cache


# Gemma 2B config
GEMMA_2B = dict(n_layers=18, hidden=2048, n_heads=8, n_kv_heads=1, head_dim=256, ffn_dim=16384)
# Gemma 300M config
GEMMA_300M = dict(n_layers=18, hidden=1024, n_heads=8, n_kv_heads=1, head_dim=256, ffn_dim=4096)

"""Gemma 300M action expert with AdaRMS (adaptive RMSNorm via timestep conditioning)."""

import mlx.core as mx
import mlx.nn as nn
from .gemma import GemmaAttention, GemmaMLP


class AdaRMSNorm(nn.Module):
    """
    Adaptive RMSNorm used in the action expert.
    When called with cond, computes scale/shift/gate from a linear projection of cond.
    Returns (normed_output, gate) for use with gated residual.
    """

    def __init__(self, dim: int, cond_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # dense: maps timestep embedding → (scale, shift, gate) each of size dim
        self.dense = nn.Linear(cond_dim, dim * 3, bias=True)

    def _norm(self, x: mx.array) -> mx.array:
        x32 = x.astype(mx.float32)
        var = mx.mean(x32 * x32, axis=-1, keepdims=True)
        return x32 * mx.rsqrt(var + self.eps)

    def __call__(self, x: mx.array, cond: mx.array) -> tuple[mx.array, mx.array]:
        normed = self._norm(x)
        mod = self.dense(cond)          # [B, dim*3]
        mod = mod.reshape(*cond.shape[:-1], 1, -1)  # [B, 1, dim*3] for broadcasting over seq
        scale, shift, gate = mx.split(mod, 3, axis=-1)
        out = (normed * (1.0 + scale.astype(mx.float32)) + shift.astype(mx.float32)).astype(x.dtype)
        return out, gate.astype(x.dtype)


class ExpertBlock(nn.Module):
    """One action-expert transformer block with AdaRMS for both norms."""

    def __init__(self, hidden: int, n_heads: int, n_kv_heads: int, head_dim: int, ffn_dim: int, rope_theta: float = 10000.0):
        super().__init__()
        self.input_layernorm = AdaRMSNorm(hidden, cond_dim=hidden)
        self.self_attn = GemmaAttention(hidden, n_heads, n_kv_heads, head_dim, rope_theta)
        self.post_attention_layernorm = AdaRMSNorm(hidden, cond_dim=hidden)
        self.mlp = GemmaMLP(hidden, ffn_dim)

    def __call__(
        self,
        x: mx.array,
        cond: mx.array,
        mask: mx.array | None = None,
        cache: tuple | None = None,
    ) -> tuple[mx.array, tuple]:
        normed, gate = self.input_layernorm(x, cond)
        attn_out, cache = self.self_attn(normed, mask, cache)
        x = x + attn_out * gate
        normed, gate = self.post_attention_layernorm(x, cond)
        x = x + self.mlp(normed) * gate
        return x, cache


class AdaRMSNormFinal(nn.Module):
    """Final norm in the action expert decoder (also adaptive)."""

    def __init__(self, dim: int, cond_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.dense = nn.Linear(cond_dim, dim * 3, bias=True)

    def _norm(self, x: mx.array) -> mx.array:
        x32 = x.astype(mx.float32)
        var = mx.mean(x32 * x32, axis=-1, keepdims=True)
        return x32 * mx.rsqrt(var + self.eps)

    def __call__(self, x: mx.array, cond: mx.array) -> mx.array:
        normed = self._norm(x)
        mod = self.dense(cond).reshape(*cond.shape[:-1], 1, -1)
        scale, shift, _ = mx.split(mod, 3, axis=-1)
        return (normed * (1.0 + scale.astype(mx.float32)) + shift.astype(mx.float32)).astype(x.dtype)


class ActionExpert(nn.Module):
    """
    Gemma 300M action expert with AdaRMS.
    Receives projected action embeddings (not raw tokens).
    """

    def __init__(self):
        super().__init__()
        hidden, n_heads, n_kv_heads, head_dim, ffn_dim = 1024, 8, 1, 256, 4096
        self.layers = [ExpertBlock(hidden, n_heads, n_kv_heads, head_dim, ffn_dim) for _ in range(18)]
        self.norm = AdaRMSNormFinal(hidden, cond_dim=hidden)

    def __call__(
        self,
        x: mx.array,
        cond: mx.array,
        mask: mx.array | None = None,
        cache: list | None = None,
    ) -> tuple[mx.array, list]:
        if cache is None:
            cache = [None] * len(self.layers)
        for i, layer in enumerate(self.layers):
            x, cache[i] = layer(x, cond, mask, cache[i])
        return self.norm(x, cond), cache

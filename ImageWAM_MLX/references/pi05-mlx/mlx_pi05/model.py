"""Top-level PI05 inference model: prefix encode + flow-matching denoise loop."""

import math
import mlx.core as mx
import mlx.nn as nn

from .gemma import GemmaDecoder, GEMMA_2B
from .siglip import SigLIPVisionModel
from .action_expert import ActionExpert

# Sinusoidal timestep embedding (matches openpi / LeRobot exactly)
def _sinusoidal_emb(t: mx.array, dim: int, min_period: float = 4e-3, max_period: float = 4.0) -> mx.array:
    """t: [B], returns [B, dim]."""
    half = dim // 2
    fraction = mx.linspace(0.0, 1.0, half, dtype=mx.float32)
    period = min_period * (max_period / min_period) ** fraction
    scale = (2.0 * math.pi) / period                           # [half]
    arg = scale[None, :] * t[:, None].astype(mx.float32)       # [B, half]
    return mx.concatenate([mx.sin(arg), mx.cos(arg)], axis=-1)  # [B, dim]


class PI05(nn.Module):
    """
    PI05 inference model.

    Inference flow:
      1. encode_prefix(image, lang_tokens) → kv_cache (list of 18 (K, V) pairs)
      2. For each of num_steps denoising steps:
           v_t = denoise_step(x_t, t, kv_cache)
           x_t = x_t + dt * v_t
      3. Return x_t (action chunk [B, 50, action_dim])
    """

    MAX_ACTION_DIM = 32
    CHUNK_SIZE = 50

    def __init__(self):
        super().__init__()

        # Vision
        self.vision = SigLIPVisionModel()
        self.mm_proj = nn.Linear(1152, 2048, bias=True)       # multi_modal_projector

        # VLM language decoder (Gemma 2B, no embed_tokens — we hold it separately)
        cfg = GEMMA_2B
        self.vlm = GemmaDecoder(**cfg)
        # Token embedding table (weight tied with lm_head in original)
        self.embed_tokens = nn.Embedding(257152, 2048)

        # Action expert
        self.expert = ActionExpert()

        # Timestep MLP: sinusoidal(1024) → silu → 1024 → silu
        self.time_mlp_in = nn.Linear(1024, 1024, bias=True)
        self.time_mlp_out = nn.Linear(1024, 1024, bias=True)

        # Action projections
        self.action_in_proj = nn.Linear(self.MAX_ACTION_DIM, 1024, bias=True)
        self.action_out_proj = nn.Linear(1024, self.MAX_ACTION_DIM, bias=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed_image(self, image: mx.array) -> mx.array:
        """image [B,3,H,W] → [B, 256, 2048]."""
        feats = self.vision(image.astype(mx.float32))   # [B, 256, 1152]
        proj = self.mm_proj(feats)                       # [B, 256, 2048]
        return (proj * math.sqrt(2048)).astype(image.dtype)

    def _time_cond(self, t: mx.array) -> mx.array:
        """t [B] → adarms_cond [B, 1024]."""
        emb = _sinusoidal_emb(t, dim=1024).astype(t.dtype)
        emb = nn.silu(self.time_mlp_in(emb))
        return nn.silu(self.time_mlp_out(emb))

    # ------------------------------------------------------------------
    # Two-phase inference
    # ------------------------------------------------------------------

    def encode_prefix(self, image: mx.array, lang_tokens: mx.array) -> list:
        """
        Run VLM (SigLIP + Gemma 2B) on image + language tokens.
        Returns kv_cache: list of 18 (K, V) tuples, one per Gemma layer.
        All tokens attend to each other (bidirectional, no causal mask).

        image:       [B, 3, 224, 224], float16, normalised to [-1, 1]
        lang_tokens: [B, T] int32
        """
        img_emb = self._embed_image(image)                      # [B, 256, 2048]
        lang_scale = math.sqrt(2048)
        lang_emb = self.embed_tokens(lang_tokens) * lang_scale  # [B, T, 2048]

        prefix = mx.concatenate([img_emb, lang_emb], axis=1)   # [B, 256+T, 2048]
        prefix = prefix.astype(mx.float16)

        # Bidirectional (no causal mask) — prefix tokens attend to each other freely
        _, kv_cache = self.vlm(prefix, mask=None, cache=None)
        return kv_cache

    def denoise_step(self, x_t: mx.array, t: mx.array, prefix_kv: list) -> mx.array:
        """
        One denoising step of the action expert.
        Action tokens attend to prefix (via prefix_kv cache) + all action tokens bidirectionally.

        x_t:       [B, 50, 32]  noisy actions at timestep t
        t:         [B]          current timestep (scalar per batch)
        prefix_kv: list of 18 (K, V) tuples from encode_prefix
        Returns v_t: [B, 50, 32] predicted velocity
        """
        cond = self._time_cond(t)                               # [B, 1024]
        x_emb = self.action_in_proj(x_t.astype(mx.float16))    # [B, 50, 1024]
        x_emb = x_emb.astype(mx.float16)

        # Shallow-copy list so expert can update cache[i] without mutating prefix_kv
        kv_copy = list(prefix_kv)
        x_out, _ = self.expert(x_emb, cond, mask=None, cache=kv_copy)
        x_out = x_out[:, -self.CHUNK_SIZE:].astype(mx.float32)
        return self.action_out_proj(x_out)                       # [B, 50, 32]

    # ------------------------------------------------------------------
    # Full inference
    # ------------------------------------------------------------------

    def sample_actions(
        self,
        image: mx.array,
        lang_tokens: mx.array,
        num_steps: int = 10,
    ) -> mx.array:
        """
        Run flow-matching denoising and return action chunk.

        Returns: [B, 50, 32] — action chunk (padded to max_action_dim=32)
        """
        B = image.shape[0]
        device_dtype = mx.float16

        # Encode prefix once
        prefix_kv = self.encode_prefix(image, lang_tokens)
        mx.eval(prefix_kv)

        # Start from pure noise
        x_t = mx.random.normal(shape=(B, self.CHUNK_SIZE, self.MAX_ACTION_DIM)).astype(mx.float32)

        dt = -1.0 / num_steps
        for step in range(num_steps):
            t = mx.full((B,), 1.0 + step * dt, dtype=mx.float32)
            v_t = self.denoise_step(x_t.astype(device_dtype), t, prefix_kv)
            x_t = x_t + dt * v_t
            mx.eval(x_t)

        return x_t  # [B, 50, 32]

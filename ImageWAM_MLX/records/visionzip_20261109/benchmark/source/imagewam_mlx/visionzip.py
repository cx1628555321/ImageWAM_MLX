"""Opt-in ImageWAM prefix compression. Default inference is unchanged.

Inspired by VisionZip (CVPR 2025): incoming attention ranks dominant tokens,
optionally merge non-dominant features into contextual representatives. This is
an ImageWAM adaptation inside its multimodal prefix, not the original ViT hook.
Never prune text/state, renumber surviving RoPE coordinates, reuse observation
features, change action horizon, or reduce denoising iterations.
"""
from dataclasses import dataclass, asdict, replace
import numpy as np
import mlx.core as mx
from imagewam_mlx.ops import D, attention, flux_rope, norm


@dataclass(frozen=True)
class Config:
    name: str
    keep: int = 252
    cut_after: int = 5  # completed prefix blocks; first five are double-stream
    score: str = "all"  # all queries / text+state queries / uniform control
    wrists_full: bool = False
    contextual: int = 0
    kv_only: bool = False
    align32: bool = False

    def __post_init__(self):
        if not isinstance(self.keep, int) or not 96 <= self.keep <= 288:
            raise ValueError("Visual budget must be between 96 and 288")
        if not isinstance(self.cut_after, int) or not 1 <= self.cut_after <= 24 or self.score not in ("all", "text", "uniform"):
            raise ValueError("Invalid pruning boundary/score")
        if self.contextual not in (0, 12) or self.kv_only and self.contextual:
            raise ValueError("Contextual merge is only implemented for prefix features")
        if self.contextual and (min(self.quotas()) < 12 or self.score == 'uniform'):
            raise ValueError('Invalid contextual budget or uniform/contextual combination')
        if self.align32 and (self.keep != 288 or not self.wrists_full or self.contextual or self.kv_only):
            raise ValueError('Alignment requires dynamic budget, full wrists and no merge/KV-only mode')

    def quotas(self):
        if self.wrists_full:
            return (self.keep - 96, 48, 48)
        wrist = self.keep // 6
        return (self.keep - 2*wrist, wrist, wrist)

    def resolved(self, text_state_count):
        if not self.align32:
            return self
        total = 288 + text_state_count
        # Drop only the rows above the previous 32-row GEMM scheduling boundary.
        return replace(self, keep=288-(total % 32), align32=False)


REGIONS = (
    np.arange(192, dtype=np.int32),
    np.array([r * 16 + c for r in range(12, 18) for c in range(8)], np.int32),
    np.array([r * 16 + c for r in range(12, 18) for c in range(8, 16)], np.int32),
)


def stable_ranks(scores):
    """Descending score, ascending original index for exact ties."""
    n = scores.shape[0]
    idx = mx.arange(n, dtype=mx.int32)
    higher = scores[None, :] > scores[:, None]
    tied_earlier = (scores[None, :] == scores[:, None]) & (idx[None, :] < idx[:, None])
    return mx.sum((higher | tied_earlier).astype(mx.int32), axis=1)


def incoming_scores(q, k, text_count, mode):
    if mode == "uniform":
        return mx.zeros((288,), mx.float32)
    if mode == "text":
        q = q[:, :, :text_count]
    # Keep joint softmax denominator; only rank visual keys after normalization.
    logits = (q.astype(mx.float32) @ k.astype(mx.float32).swapaxes(-1, -2)) * (128 ** -0.5)
    probabilities = mx.softmax(logits, axis=-1)
    return mx.mean(probabilities[..., text_count:], axis=(0, 1, 2))


def select_tokens(features, scores, config):
    """features [1,288,D]; returned positions retain original packed-grid IDs."""
    values, positions = [], []
    for region_index, (group, keep) in enumerate(zip(REGIONS, config.quotas())):
        ids = mx.array(group)
        f = mx.take(features, ids, axis=1)[0]
        if keep == len(group):
            positions.append(ids); values.append(f); continue
        if config.score == "uniform":
            local = mx.array(np.floor((np.arange(keep) + 0.5) * len(group) / keep).astype(np.int32))
            positions.append(mx.take(ids, local)); values.append(mx.take(f, local, axis=0)); continue
        ranks = stable_ranks(mx.take(scores, ids))
        context = (config.contextual * 2 // 3 if region_index == 0 else config.contextual // 6)
        dominant = keep - context
        arange = mx.arange(len(group), dtype=mx.int32)
        chosen = mx.sort(mx.where(ranks < dominant, arange, len(group)))[:dominant]
        positions.append(mx.take(ids, chosen)); values.append(mx.take(f, chosen, axis=0))
        if context:
            remainder = mx.sort(mx.where(ranks >= dominant, arange, len(group)))[:len(group)-dominant]
            anchor_slots = mx.array(np.floor((np.arange(context) + 0.5) * remainder.shape[0] / context).astype(np.int32))
            anchors = mx.take(remainder, anchor_slots)
            remaining_features = mx.take(f, remainder, axis=0).astype(mx.float32)
            key = remaining_features * mx.rsqrt(mx.sum(remaining_features ** 2, axis=-1, keepdims=True) + 1e-12)
            targets = mx.take(key, anchor_slots, axis=0)
            similarity = key @ targets.T
            # Each representative must include itself even if normalized features tie.
            self_anchor = remainder[:, None] == anchors[None, :]
            assignments = mx.argmax(mx.where(self_anchor, 2.0, similarity), axis=-1)
            membership = (mx.arange(context)[:, None] == assignments[None, :]).astype(mx.float32)
            merged = (membership @ remaining_features) / mx.maximum(mx.sum(membership, axis=-1, keepdims=True), 1)
            positions.append(mx.take(ids, anchors)); values.append(merged.astype(features.dtype))
    positions = mx.concatenate(positions)
    order = mx.argsort(positions)
    return mx.take(mx.concatenate(values), order, axis=0)[None], mx.take(positions, order)


def build_prefill(model, config, *, diagnostic=False, capture_cut=False):
    """Return a pure prefill closure; compile separately for each configuration.

    Prefix-pruning keeps the earlier layers' full KV; only subsequent layers are
    sparse. KV-only preserves the full prefix computation, then gathers every
    layer's visual KV. It is a control for speed/precision tradeoffs.
    """
    configured = config
    def prefill(hidden, latent, st):
        config = configured.resolved(hidden.shape[1]+1)
        if latent.shape != (1, 18, 16, 128):
            raise ValueError("This experiment requires the original 288-token mosaic")
        v = model.visual
        ctx = mx.concatenate([hidden, st @ model.proprio.w['weight'].T + model.proprio.w['bias']], axis=1)
        txt = v.linear(ctx, 'txt_in')
        img = v.linear(latent.reshape(1, -1, 128), 'img_in')
        nt = txt.shape[1]
        ids = np.zeros((1, nt, 4), np.float32); ids[0, :, 3] = np.arange(nt)
        ti = flux_rope(mx.array(ids).astype(D))
        ids = np.zeros((1, 18, 16, 4), np.float32); ids[..., 0] = 10
        ids[..., 1] = np.arange(18)[:, None]; ids[..., 2] = np.arange(16)[None, :]
        ii = flux_rope(mx.array(ids.reshape(1, -1, 4)).astype(D))
        pe = tuple(mx.concatenate([a, b], axis=2) for a, b in zip(ti, ii))
        vec = v.embed_time(mx.zeros((1,), dtype=D))
        mi = v.modulation(vec, 'double_stream_modulation_img', True)
        mt = v.modulation(vec, 'double_stream_modulation_txt', True)
        ms = v.modulation(vec, 'single_stream_modulation')
        cache, positions, scores = [], mx.arange(288, dtype=mx.int32), mx.zeros((288,), mx.float32)
        captured = None
        cut = 24 if config.kv_only else config.cut_after
        for i in range(5):
            p = f'double_blocks.{i}'
            tq, tk, tv = v.double_prepare(txt, p, mt, ti, 'txt')
            iq, ik, iv = v.double_prepare(img, p, mi, ii)
            q, k, val = [mx.concatenate(pair, axis=2) for pair in ((tq, iq), (tk, ik), (tv, iv))]
            cache.append((k, val)); out = attention(q, k, val)
            img = v.double_post(img, out[:, nt:], p, mi)
            txt = v.double_post(txt, out[:, :nt], p, mt, 'txt')
            if capture_cut and cut == i + 1:
                captured = (img, q, k)
            if config.keep < 288 and cut == i + 1:
                scores = incoming_scores(q, k, nt, config.score)
                img, positions = select_tokens(img, scores, config)
                ii = tuple(mx.take(a, positions, axis=2) for a in ii)
                pe = tuple(mx.concatenate([a, b], axis=2) for a, b in zip(ti, ii))
        x = mx.concatenate([txt, img], axis=1)
        for i in range(20):
            p = f'single_blocks.{i}'
            if i == 19 and model.compact:
                kv = v.linear((1 + ms[1]) * norm(x) + ms[0], p + '.kv')
                k, val = mx.split(kv.reshape(kv.shape[0], kv.shape[1], 2, 24, 128).transpose(2, 0, 3, 1, 4), 2, axis=0)
                k = v.norm_rotate_key(k[0], v.w[p + '.norm.key_norm.scale'], pe)
                cache.append((k, val[0])); break
            q, k, val, mlp = v.single_prepare(x, p, ms, pe)
            cache.append((k, val))
            if i < 19:
                x = v.single_post(x, attention(q, k, val), mlp, p, ms)
            if capture_cut and cut == i + 6:
                captured = (x[:, nt:], q, k)
            if config.keep < 288 and cut == i + 6:
                scores = incoming_scores(q, k, nt, config.score)
                sparse, positions = select_tokens(x[:, nt:], scores, config)
                if not config.kv_only:
                    x = mx.concatenate([x[:, :nt], sparse], axis=1)
                    full_indices = mx.concatenate([mx.arange(nt, dtype=mx.int32), nt + positions])
                    pe = tuple(mx.take(a, full_indices, axis=2) for a in pe)
        if config.kv_only and config.keep < 288:
            full_indices = mx.concatenate([mx.arange(nt, dtype=mx.int32), nt + positions])
            cache = [(mx.take(k, full_indices, axis=2), mx.take(val, full_indices, axis=2)) for k, val in cache]
        if capture_cut:
            return cache, positions, scores, captured
        if diagnostic:
            return cache, positions, scores
        return cache
    return prefill


def describe(config):
    return {**asdict(config), 'keep': None if config.align32 else config.keep,
            'camera_quotas': None if config.align32 else config.quotas(),
            'budget_policy': 'keep=288-((288+text+state)%32); wrists=48+48' if config.align32 else 'fixed',
            'visual_drop_fraction': None if config.align32 else (288-config.keep)/288,
            'text_state_preserved': True, 'position_policy': 'original_4d_rope_representatives',
            'early_layer_kv': 'pruned_after_full_prefill' if config.kv_only else 'full',
            'contextual_weighting': 'unweighted_feature_mean_no_attention_mass_correction',
            'observation_cache': False}


PRESETS = {
    'merge252': Config('merge252', keep=252, cut_after=5, contextual=12),
    'late_align32': Config('late_align32', keep=288, cut_after=15, wrists_full=True, align32=True),
    'prune252': Config('prune252', keep=252, cut_after=5),
    'prune216': Config('prune216', keep=216, cut_after=5),
}


def install(model, preset='off'):
    """Enable a validated-shape experimental preset, or restore original prefill.

    Reconfiguration replaces the closure; it does not stack wrappers or retain
    image features. Only the speed/all_int8/native-w4 path is supported here.
    """
    if preset != 'off' and preset not in PRESETS:
        raise ValueError(f'Unknown VisionZip preset: {preset}')
    if preset == 'off':
        if hasattr(model, '_visionzip_original_prefill'):
            model.prefill = model._visionzip_original_prefill
        model.visionzip_info = {'enabled': False, 'preset': 'off'}
        return model.visionzip_info
    if (getattr(model, 'inference_profile', None) != 'speed'
            or getattr(model, 'prefix_precision', None) != 'all_int8'
            or getattr(getattr(model, 'action', None), 'kind', None) != 'w4'
            or not getattr(model, 'compact', False)):
        raise ValueError('VisionZip requires compact speed/all_int8/native-w4')
    if not hasattr(model, '_visionzip_original_prefill'):
        model._visionzip_original_prefill = model.prefill
    model.prefill = mx.compile(build_prefill(model, PRESETS[preset]))
    model.visionzip_info = {'enabled': True, 'preset': preset, **describe(PRESETS[preset])}
    return model.visionzip_info

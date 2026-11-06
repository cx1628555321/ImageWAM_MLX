"""
Load pi0.5 weights from safetensors into the MLX PI05 model.

Memory strategy for 8 GB M1:
- Disk weights are float32 (~14.5 GB).
- Load one tensor at a time, cast to float16 (~7.2 GB).
- Immediately apply 4-bit quantization to all Linear layers (~1.8 GB final).
- Peak RAM stays well within 8 GB via MLX lazy evaluation.

Two-tier loading:
  1. First run: hf_hub_download → load_model() → saves quantized npz (~1.8 GB)
  2. Subsequent runs: load_model(quantized_path=...) → loads the small npz directly
"""

import os
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from safetensors import safe_open

HF_REPO = "lerobot/pi05_base"
HF_CACHE_DIR = "/Users/mohan_007/Downloads/dare_to_experiment/pi05_cache"
QUANTIZED_SAVE_PATH = "/Users/mohan_007/Downloads/dare_to_experiment/pi05_cache/pi05_mlx_4bit.npz"
SAFETENSORS_PATH = "/Users/mohan_007/Downloads/dare_to_experiment/pi05_cache/models--lerobot--pi05_base/snapshots/9e55186ad36e66b95cda57bc47818d9e6237ae30/model.safetensors"


def _find_cached_safetensors() -> str | None:
    """Return path to the cached safetensors file, or None if not downloaded yet."""
    # Check the known snapshot path first
    if os.path.isfile(SAFETENSORS_PATH):
        return SAFETENSORS_PATH
    # Fall back to scanning blobs directory
    blobs_dir = f"{HF_CACHE_DIR}/models--lerobot--pi05_base/blobs"
    if os.path.isdir(blobs_dir):
        for fname in os.listdir(blobs_dir):
            fpath = os.path.join(blobs_dir, fname)
            if os.path.isfile(fpath) and os.path.getsize(fpath) > 1e10:
                return fpath
    # Scan snapshots
    snaps_dir = f"{HF_CACHE_DIR}/models--lerobot--pi05_base/snapshots"
    if os.path.isdir(snaps_dir):
        for snap in os.listdir(snaps_dir):
            candidate = os.path.join(snaps_dir, snap, "model.safetensors")
            if os.path.isfile(candidate):
                return candidate
    return None


def _download_weights() -> str:
    from huggingface_hub import hf_hub_download
    print(f"Downloading lerobot/pi05_base model.safetensors (~14.5 GB)...")
    return hf_hub_download(
        repo_id=HF_REPO,
        filename="model.safetensors",
        cache_dir=HF_CACHE_DIR,
    )


def _pt_to_mlx(tensor, dtype=mx.float16) -> mx.array:
    return mx.array(tensor.numpy()).astype(dtype)


def load_weights(path: str) -> dict[str, mx.array]:
    """Load all tensors from safetensors as float16, one at a time."""
    weights: dict[str, mx.array] = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        print(f"  Loading {len(keys)} tensors from safetensors...")
        for i, key in enumerate(keys):
            if i % 100 == 0:
                print(f"  [{i}/{len(keys)}] loading...", flush=True)
            t = f.get_tensor(key)
            weights[key] = _pt_to_mlx(t, dtype=mx.float16)
    return weights


def _remap(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """
    Map safetensors key names → MLX model attribute paths.

    Safetensors prefix                                                → MLX attribute
    ──────────────────────────────────────────────────────────────────────────────────
    action_in_proj.*                                                  → action_in_proj.*
    action_out_proj.*                                                 → action_out_proj.*
    time_mlp_in.*                                                     → time_mlp_in.*
    time_mlp_out.*                                                    → time_mlp_out.*

    paligemma_with_expert.paligemma.lm_head.weight                   → embed_tokens.weight
    paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight
                                                                      → embed_tokens.weight  (skip vlm.)
    paligemma_with_expert.paligemma.model.language_model.*           → vlm.*
    paligemma_with_expert.paligemma.model.multi_modal_projector.linear.* → mm_proj.*
    paligemma_with_expert.paligemma.model.vision_tower.vision_model.* → vision.*

    paligemma_with_expert.gemma_expert.model.*                       → expert.*
    """
    LM_PFX = "paligemma_with_expert.paligemma.model.language_model."
    EMBED_KEY = LM_PFX + "embed_tokens.weight"

    new: dict[str, mx.array] = {}
    for k, v in weights.items():
        # Handle old checkpoint naming (action_time_mlp_* → time_mlp_*)
        k = k.replace("action_time_mlp_in.", "time_mlp_in.").replace("action_time_mlp_out.", "time_mlp_out.")

        if k.startswith("action_in_proj.") or k.startswith("action_out_proj.") \
                or k.startswith("time_mlp_in.") or k.startswith("time_mlp_out."):
            new[k] = v

        elif k == "paligemma_with_expert.paligemma.lm_head.weight":
            # PaliGemma weight-ties lm_head ↔ embed_tokens; use lm_head as source
            new["embed_tokens.weight"] = v

        elif k == EMBED_KEY:
            # Avoid duplicating to vlm.embed_tokens — map here if lm_head wasn't seen yet
            new.setdefault("embed_tokens.weight", v)

        elif k.startswith(LM_PFX):
            suffix = k[len(LM_PFX):]
            new[f"vlm.{suffix}"] = v

        elif k.startswith("paligemma_with_expert.paligemma.model.multi_modal_projector.linear."):
            suffix = k[len("paligemma_with_expert.paligemma.model.multi_modal_projector.linear."):]
            new[f"mm_proj.{suffix}"] = v

        elif k.startswith("paligemma_with_expert.paligemma.model.vision_tower.vision_model."):
            suffix = k[len("paligemma_with_expert.paligemma.model.vision_tower.vision_model."):]
            new[f"vision.{suffix}"] = v

        elif k.startswith("paligemma_with_expert.gemma_expert.model."):
            suffix = k[len("paligemma_with_expert.gemma_expert.model."):]
            new[f"expert.{suffix}"] = v

        elif k == "paligemma_with_expert.gemma_expert.lm_head.weight":
            pass  # unused in inference

    return new


def _fix_keys(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """Fix tensor-shape differences between PyTorch and MLX conventions."""
    out = {}
    for k, v in weights.items():
        if k == "vision.embeddings.patch_embedding.weight" and v.ndim == 4:
            # Conv2d: PyTorch [out, in, kH, kW] → MLX [out, kH, kW, in]
            out[k] = v.transpose(0, 2, 3, 1)
        else:
            out[k] = v
    return out


def _flatten_params(d, prefix=""):
    """Flatten nested parameter dict to {dot.path: array}."""
    flat = {}
    for k, v in d.items():
        full = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(_flatten_params(v, full))
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    flat.update(_flatten_params(item, f"{full}.{i}"))
                elif isinstance(item, mx.array):
                    flat[f"{full}.{i}"] = item
        elif isinstance(v, mx.array):
            flat[full] = v
    return flat


def apply_weights(model, weights: dict[str, mx.array]):
    """Remap and load weights into the MLX model."""
    remapped = _remap(weights)
    remapped = _fix_keys(remapped)
    model_keys = set(_flatten_params(model.parameters()).keys())
    missing = sorted(k for k in model_keys if k not in remapped)
    unexpected = sorted(k for k in remapped if k not in model_keys)
    if missing:
        print(f"  WARNING: {len(missing)} missing keys: {missing[:5]}")
    if unexpected:
        print(f"  WARNING: {len(unexpected)} unexpected keys: {unexpected[:5]}")
    model.load_weights(list(remapped.items()), strict=False)


def quantize_model(model) -> None:
    """
    Quantize Linear layers in VLM + expert to 4-bit (in-place).
    SigLIP layers are excluded because fc2 has input_dim=4304 which is not
    divisible by group_size=64. Keeping SigLIP in float16 adds ~0.8 GB but
    avoids the constraint and keeps vision quality high.
    Total: ~7 GB float16 → ~2.4 GB (0.8 GB SigLIP float16 + 1.6 GB Gemma int4).
    """
    def _should_quantize(layer: nn.Module) -> bool:
        if not isinstance(layer, nn.Linear):
            return False
        out_dim, in_dim = layer.weight.shape
        return in_dim % 64 == 0 and out_dim >= 256

    nn.QuantizedLinear.quantize_module(
        model, group_size=64, bits=4,
        linear_class_predicate=_should_quantize,
    )


def save_quantized(model, path: str = QUANTIZED_SAVE_PATH) -> None:
    """Save quantized weights as npz for fast subsequent loads."""
    mx.eval(model.parameters())
    params = dict(model.parameters())
    flat = {}
    def _flatten(d, prefix=""):
        for k, v in d.items():
            full = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                _flatten(v, full)
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    if isinstance(item, dict):
                        _flatten(item, f"{full}.{i}")
                    elif isinstance(item, mx.array):
                        flat[f"{full}.{i}"] = item
            elif isinstance(v, mx.array):
                flat[full] = v
    _flatten(params)
    mx.savez(path, **flat)
    print(f"  Saved quantized weights → {path}")


def load_model(
    safetensors_path: str | None = None,
    quantized_path: str = QUANTIZED_SAVE_PATH,
    quantize: bool = True,
    save_after_load: bool = True,
) -> "PI05":
    """
    Full pipeline: find/download weights → build model → load → quantize → return.

    On first call: downloads safetensors, loads, quantizes, saves npz.
    On subsequent calls: loads npz directly (fast).
    """
    from .model import PI05

    model = PI05()

    # Fast path: load pre-saved quantized weights
    if quantize and os.path.exists(quantized_path):
        print(f"Loading pre-quantized weights from {quantized_path} ...")
        quantize_model(model)  # replace Linear → QuantizedLinear first
        weights = dict(mx.load(quantized_path))
        model.load_weights(list(weights.items()))
        mx.eval(model.parameters())
        return model

    # Slow path: load from safetensors
    if safetensors_path is None:
        safetensors_path = _find_cached_safetensors()
    if safetensors_path is None:
        safetensors_path = _download_weights()

    print(f"Loading from {safetensors_path} ...")
    weights = load_weights(safetensors_path)
    apply_weights(model, weights)
    del weights  # free float16 copies

    if quantize:
        print("  Quantizing to 4-bit...")
        quantize_model(model)

    mx.eval(model.parameters())

    if quantize and save_after_load:
        try:
            save_quantized(model, quantized_path)
        except Exception as e:
            print(f"  (could not save quantized weights: {e})")

    return model

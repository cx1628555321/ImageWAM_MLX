"""
pi0.5 MLX inference demo.

Usage:
    python3.11 infer_pi05.py

Runs a forward pass with random inputs to verify the model loads and executes.
"""

import math
import sys
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def main():
    print("=== pi0.5 MLX Inference Demo ===")
    print(f"MLX version: {mx.__version__}")

    # ── 1. Load model ────────────────────────────────────────────────
    print("\n[1/3] Loading model (this may take ~30 s on first run) ...")
    t0 = time.time()

    sys.path.insert(0, ".")
    from mlx_pi05.load import load_model
    model = load_model()
    model.eval()

    elapsed = time.time() - t0
    print(f"      Loaded in {elapsed:.1f}s")

    # ── 2. Fake inputs ───────────────────────────────────────────────
    print("\n[2/3] Building dummy inputs ...")
    B = 1

    # Image: [B, 3, 224, 224] normalised to [-1, 1]
    image = mx.random.uniform(low=-1.0, high=1.0, shape=(B, 3, 224, 224)).astype(mx.float16)

    # Language tokens: [B, T], values in [0, vocab_size)
    # Simulate a short prompt: 10 tokens
    lang_tokens = mx.array(np.random.randint(0, 257152, size=(B, 10)), dtype=mx.int32)

    print(f"      image: {image.shape}  lang_tokens: {lang_tokens.shape}")

    # ── 3. Run inference ─────────────────────────────────────────────
    print("\n[3/3] Running inference (10 flow-matching steps) ...")
    t1 = time.time()

    actions = model.sample_actions(image, lang_tokens, num_steps=10)
    mx.eval(actions)

    elapsed = time.time() - t1
    print(f"      Done in {elapsed:.2f}s")
    print(f"      actions shape: {actions.shape}")
    print(f"      actions[:, :3, :6]:\n{np.array(actions[0, :3, :6])}")

    print("\n✓ Inference complete.")


if __name__ == "__main__":
    main()

"""
infer_real.py — Run pi0.5 MLX inference on real LeRobot dataset inputs.

Uses:
  - Image   : first frame of lerobot/aloha_sim_transfer_cube_human_image
  - Language: task string tokenized with PaliGemma tokenizer
  - Model   : MLX 4-bit quantized pi0.5

FIRST-TIME SETUP (tokenizer is gated):
  1. Accept the PaliGemma license at:
       https://huggingface.co/google/paligemma-3b-pt-224
  2. Create a HuggingFace token at:
       https://huggingface.co/settings/tokens
  3. Login:
       huggingface-cli login
  Then re-run this script — the tokenizer (~4 MB) downloads automatically.

Usage:
  python3.11 infer_real.py
  python3.11 infer_real.py --task "put the block in the bowl" --dataset lerobot/pusht_image
"""

import argparse
import sys
import time
import numpy as np
from PIL import Image

sys.path.insert(0, "/Users/mohan_007/Downloads/dare_to_experiment")

# ── Config ───────────────────────────────────────────────────────────────────
DEFAULT_DATASET = "lerobot/aloha_sim_transfer_cube_human_image"
DEFAULT_IMG_KEY = "observation.images.top"
DEFAULT_TASK    = "transfer the cube to the other side"
TOKENIZER_REPO  = "google/paligemma-3b-pt-224"
NUM_STEPS       = 10
CHUNK           = 50
ADIM            = 32


def load_frame(dataset_name: str, img_key: str):
    """Stream one frame from a LeRobot dataset."""
    print(f"  Dataset : {dataset_name}")
    from datasets import load_dataset
    ds = load_dataset(dataset_name, split="train", streaming=True)
    frame = next(iter(ds))
    available = [k for k in frame.keys() if "image" in k.lower() or "cam" in k.lower()]
    if img_key not in frame:
        if available:
            img_key = available[0]
            print(f"  Note: key '{img_key}' not found, using '{img_key}' instead")
        else:
            raise KeyError(f"No image key in dataset. Available: {list(frame.keys())}")
    img = frame[img_key]   # PIL Image
    task_from_data = frame.get("task", None)
    return img, task_from_data, img_key


def preprocess_image(img: Image.Image) -> np.ndarray:
    """Resize to 224×224 and normalize to [-1, 1] (SigLIP convention)."""
    img = img.convert("RGB").resize((224, 224), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 127.5 - 1.0   # [224,224,3] in [-1,1]
    return arr.transpose(2, 0, 1)[None]                     # [1,3,224,224]


def get_tokenizer():
    """Load PaliGemma tokenizer — prompts with login instructions if gated."""
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(TOKENIZER_REPO)
        return tok
    except Exception as e:
        if "gated" in str(e).lower() or "401" in str(e) or "403" in str(e):
            print("\n" + "=" * 60)
            print("  TOKENIZER SETUP REQUIRED")
            print("=" * 60)
            print("  The PaliGemma tokenizer is gated. One-time setup:")
            print()
            print("  1. Accept the license (free, instant):")
            print("       https://huggingface.co/google/paligemma-3b-pt-224")
            print()
            print("  2. Create a HuggingFace token (read access):")
            print("       https://huggingface.co/settings/tokens")
            print()
            print("  3. Login in this terminal:")
            print("       huggingface-cli login")
            print()
            print("  4. Re-run:  python3.11 infer_real.py")
            print("=" * 60 + "\n")
        raise


def tokenize_task(tokenizer, task: str) -> np.ndarray:
    """Tokenize a task string → int32 array [1, seq_len]."""
    ids = tokenizer.encode(task, add_special_tokens=True)
    return np.array([ids], dtype=np.int32)


def run_mlx(image_np: np.ndarray, token_ids: np.ndarray, task: str):
    """Load quantized MLX model and run 10-step denoising."""
    import mlx.core as mx
    from mlx_pi05.load import load_model

    print("\n[MLX] Loading quantized model ...", flush=True)
    t0 = time.time()
    model = load_model(quantize=True)
    model.eval()
    print(f"  Loaded in {time.time()-t0:.2f}s")

    image_mlx = mx.array(image_np)
    lang_mlx  = mx.array(token_ids)
    noise     = mx.array(np.random.randn(1, CHUNK, ADIM).astype(np.float32))

    print(f"  Task  : \"{task}\"")
    print(f"  Tokens: {token_ids.shape[1]} ids  {token_ids[0].tolist()}")
    print(f"  Image : {image_np.shape}  range [{image_np.min():.2f}, {image_np.max():.2f}]")

    print("\n  Running 10-step flow-matching denoising ...", flush=True)
    t1 = time.time()

    prefix_kv = model.encode_prefix(image_mlx, lang_mlx)
    mx.eval(prefix_kv)
    t_prefix = time.time() - t1

    x_t = noise.astype(mx.float32)
    dt  = -1.0 / NUM_STEPS
    t2  = time.time()
    for step in range(NUM_STEPS):
        t_val = mx.full((1,), 1.0 + step * dt, dtype=mx.float32)
        v_t   = model.denoise_step(x_t.astype(mx.float16), t_val, list(prefix_kv))
        x_t   = x_t + dt * v_t
        mx.eval(x_t)
    t_denoise = time.time() - t2

    actions = np.array(x_t)   # [1, 50, 32]
    print(f"  encode_prefix : {t_prefix:.2f}s")
    print(f"  denoise 10 steps: {t_denoise:.2f}s")
    return actions


def display_actions(actions: np.ndarray, task: str):
    """Print a readable summary of the output action chunk."""
    print("\n" + "=" * 65)
    print("  RESULTS")
    print("=" * 65)
    print(f"  Task           : {task}")
    print(f"  Actions shape  : {actions.shape}  (batch, timesteps, dims)")
    print(f"  Mean           : {actions.mean():.5f}")
    print(f"  Std            : {actions.std():.5f}")
    print(f"  Range          : [{actions.min():.4f}, {actions.max():.4f}]")
    print()
    print("  First 5 timesteps (first 8 action dims):")
    print(f"  {'t':<4}  " + "  ".join(f"d{d:<3}" for d in range(8)))
    for i in range(5):
        vals = "  ".join(f"{v:+.3f}" for v in actions[0, i, :8])
        print(f"  t={i}  {vals}")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--img_key", default=DEFAULT_IMG_KEY)
    parser.add_argument("--task",    default=None,
                        help="Task string. Defaults to dataset task or built-in default.")
    parser.add_argument("--save_image", default=None,
                        help="Path to save the input frame (e.g. frame.png)")
    args = parser.parse_args()

    np.random.seed(0)

    # ── 1. Load image ─────────────────────────────────────────────────────────
    print("[1/3] Loading dataset frame ...")
    img, task_from_data, img_key_used = load_frame(args.dataset, args.img_key)
    task = args.task or task_from_data or DEFAULT_TASK
    print(f"  Image key : {img_key_used}  size={img.size}")
    print(f"  Task      : {task}")

    if args.save_image:
        img.save(args.save_image)
        print(f"  Saved frame → {args.save_image}")

    image_np = preprocess_image(img)

    # ── 2. Tokenize task ──────────────────────────────────────────────────────
    print("\n[2/3] Tokenizing task ...")
    tokenizer = get_tokenizer()
    token_ids = tokenize_task(tokenizer, task)
    print(f"  Vocab size : {tokenizer.vocab_size}")
    print(f"  Token ids  : {token_ids[0].tolist()}")

    # ── 3. Run MLX inference ──────────────────────────────────────────────────
    print("\n[3/3] Running MLX inference ...")
    actions = run_mlx(image_np, token_ids, task)

    # ── Report ────────────────────────────────────────────────────────────────
    display_actions(actions, task)


if __name__ == "__main__":
    main()

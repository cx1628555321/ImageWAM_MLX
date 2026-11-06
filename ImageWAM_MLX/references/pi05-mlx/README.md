# pi0.5 MLX Port

[pi0.5](https://huggingface.co/lerobot/pi05_base) (Physical Intelligence's robotics policy) ported to [Apple MLX](https://github.com/ml-explore/mlx) for inference on Apple Silicon.

Runs on an **8 GB M1** at ~2s per action chunk using 4-bit quantization.

## Architecture

- **VLM backbone**: PaliGemma 2B (SigLIP + Gemma 2B, 18 layers)
- **Action expert**: Gemma 300M (18 layers) with AdaRMSNorm timestep conditioning
- **Policy**: Flow-matching (10-step Forward Euler denoising)
- **Output**: action chunk `[B, 50, 32]`

## Setup

```bash
pip install mlx safetensors numpy Pillow huggingface_hub
```

## Quick start

```python
import numpy as np
import mlx.core as mx
from mlx_pi05.load import load_model

# Load 4-bit quantized weights from HuggingFace (~2.6 GB, one-time download)
model = load_model(quantize=True)
model.eval()

image  = mx.array(np.zeros((1, 3, 224, 224), dtype=np.float32))  # replace with real image
tokens = mx.array(np.array([[1, 2, 3, 4, 5]], dtype=np.int32))   # replace with tokenized task

actions = model.sample_actions(image, tokens)  # [1, 50, 32]
print(actions.shape)
```

## Scripts

| Script | Description |
|--------|-------------|
| `infer_pi05.py` | Quick demo with random inputs |
| `infer_real.py` | Inference with real LeRobot dataset frame + PaliGemma tokenizer |
| `compare_e2e.py` | End-to-end numerical comparison: PyTorch MPS vs MLX |
| `compare_pt_mlx.py` | Component-level comparison (RMSNorm, attention, SigLIP, etc.) |
| `benchmark_pi05.py` | Timing benchmark: PyTorch MPS vs MLX 4-bit |

## Benchmark (M1 8 GB)

| | Mean total |
|--|--|
| PyTorch MPS float32 (chunked load+infer) | 75.1s |
| MLX 4-bit quantized (load amortised) | 3.3s |
| MLX 4-bit quantized (model pre-loaded) | **2.1s** |

## Weights

Pre-quantized weights (skip the 14.5 GB download):
- HuggingFace: [mohan007/pi05-mlx-4bit](https://huggingface.co/mohan007/pi05-mlx-4bit)

Original weights: [lerobot/pi05_base](https://huggingface.co/lerobot/pi05_base)

## Verification

PyTorch MPS float32 vs MLX float16 end-to-end:
```
max_abs_diff : 0.001310
cosine_sim   : 0.99999809   ✓
```

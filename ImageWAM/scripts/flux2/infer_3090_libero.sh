#!/usr/bin/env bash
# Run the native LIBERO evaluation on one GPU, one task, one episode by default.
# The repository .venv is pre-provisioned; this script never installs packages.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# These paths also control preflight checks and tee; keep one source of truth.
for argument in "$@"; do
  override="${argument#+}"
  override="${override#+}"
  case "${override}" in
    ckpt=*|EVALUATION.output_dir=*|hydra.run.dir=*)
      printf 'Set CKPT_PATH or OUTPUT_DIR as an environment variable; do not override this path through Hydra: %s\n' "${argument}" >&2
      exit 2
      ;;
  esac
done

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
FLUX2_SRC="${FLUX2_SRC:-${REPO_ROOT}/third_party/flux2}"
LIBERO_DIR="${LIBERO_DIR:-${REPO_ROOT}/third_party/LIBERO}"
CKPT_PATH="${CKPT_PATH:-${REPO_ROOT}/../flux2_klein_4b/model.pt}"
DATASET_STATS_PATH="${DATASET_STATS_PATH:-${REPO_ROOT}/../flux2_klein_4b/dataset_stats.json}"
FLUX2_MODEL_PATH="${FLUX2_MODEL_PATH:-${REPO_ROOT}/checkpoints/3090/flux2_imagewam_visual.safetensors}"
FLUX2_AE_MODEL_PATH="${FLUX2_AE_MODEL_PATH:-${REPO_ROOT}/checkpoints/3090/ae.safetensors}"
FLUX2_QWEN3_MODEL_SPEC="${FLUX2_QWEN3_MODEL_SPEC:-${REPO_ROOT}/checkpoints/3090/Qwen3-4B}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${REPO_ROOT}/.libero}"
export PYTHONPATH="${REPO_ROOT}/src:${FLUX2_SRC}/src:${LIBERO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export HF_HOME="${REPO_ROOT}/.cache/huggingface"
export TORCH_HOME="${REPO_ROOT}/.cache/torch"
export XDG_CACHE_HOME="${REPO_ROOT}/.cache"
export MPLCONFIGDIR="${REPO_ROOT}/.cache/matplotlib"
export NUMBA_CACHE_DIR="${REPO_ROOT}/.cache/numba"
mkdir -p "${HF_HOME}" "${TORCH_HOME}" "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}"

if [ ! -x "${PYTHON_BIN}" ]; then
  printf 'Missing repository Python: %s\n' "${PYTHON_BIN}" >&2
  exit 2
fi
for required_file in "${CKPT_PATH}" "${DATASET_STATS_PATH}" "${FLUX2_MODEL_PATH}" "${FLUX2_AE_MODEL_PATH}"; do
  if [ ! -s "${required_file}" ]; then
    printf 'Missing or empty required file: %s\n' "${required_file}" >&2
    exit 2
  fi
done
for required_dir in "${FLUX2_SRC}/src/flux2" "${LIBERO_DIR}/libero/libero" "${FLUX2_QWEN3_MODEL_SPEC}"; do
  if [ ! -d "${required_dir}" ]; then
    printf 'Missing required directory: %s\n' "${required_dir}" >&2
    exit 2
  fi
done

"${PYTHON_BIN}" "${SCRIPT_DIR}/configure_3090_libero.py" \
  --libero-dir "${LIBERO_DIR}" --config-dir "${LIBERO_CONFIG_PATH}" \
  --patch-init-state-load

# Match eval_libero_single._resolve_ckpt_tag to keep its output path stable.
CKPT_TAG="$("${PYTHON_BIN}" - "${CKPT_PATH}" <<'PY'
from pathlib import Path
import sys
path = Path(sys.argv[1]).resolve()
parts = path.parts
if "runs" in parts:
    index = parts.index("runs")
    if index + 2 >= len(parts):
        raise ValueError(f"Unexpected checkpoint path under runs: {path}")
    print(f"{parts[index + 1]}_{parts[index + 2]}")
else:
    print(path.stem)
PY
)"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/evaluate_results/3090/${CKPT_TAG}/${RUN_ID}}"
# Apply the native tagging rule to custom OUTPUT_DIR values as well, so the
# console log, Hydra log, resolved config, results, and videos stay together.
OUTPUT_DIR="$("${PYTHON_BIN}" - "${OUTPUT_DIR}" "${CKPT_TAG}" "${REPO_ROOT}" <<'PY'
from pathlib import Path
import sys
raw = Path(sys.argv[1]).expanduser().resolve()
tag = sys.argv[2]
eval_root = (Path(sys.argv[3]) / "evaluate_results").resolve()
try:
    relative = raw.relative_to(eval_root)
except ValueError:
    result = raw if raw.parent.name == tag else raw.parent / tag / raw.name
else:
    if len(relative.parts) >= 2 and relative.parts[1] == tag:
        result = raw
    else:
        family = relative.parts[0] if relative.parts else "libero"
        result = eval_root / family / tag / raw.name
print(result)
PY
)"
mkdir -p "${OUTPUT_DIR}"

ARGS=(
  --config-name sim_libero_omnigen2
  task=libero_flux2_klein_4b_base_imagewam
  "ckpt=${CKPT_PATH}"
  "model.flux2_src_path=${FLUX2_SRC}"
  "model.flux2_model_path=${FLUX2_MODEL_PATH}"
  "model.ae_model_path=${FLUX2_AE_MODEL_PATH}"
  "model.qwen3_model_spec=${FLUX2_QWEN3_MODEL_SPEC}"
  model.variant=klein-base-4b
  model.load_text_encoder=true
  model.action_dit_pretrained_path=null
  model.proprio_dim=8
  model.pack_proprio_after_text=true
  "model.qwen_context_len=${QWEN_CONTEXT_LEN:-512}"
  data.train.dataset_dirs=[]
  data.train.qwen_text_cache_dir=null
  "data.train.qwen_context_len=${QWEN_CONTEXT_LEN:-512}"
  data.train.qwen_text_cache_format=qwen3_flux2
  mixed_precision=bf16
  wandb.enabled=false
  gpu_id=0
  "seed=${SEED:-42}"
  "EVALUATION.dataset_stats_path=${DATASET_STATS_PATH}"
  "EVALUATION.task_suite_name=${TASK_SUITE:-libero_spatial}"
  "EVALUATION.task_id=${TASK_ID:-0}"
  "EVALUATION.num_trials=${NUM_TRIALS:-1}"
  EVALUATION.env_num=1
  EVALUATION.device=cuda:0
  "EVALUATION.num_inference_steps=${INFERENCE_STEPS:-10}"
  "EVALUATION.action_horizon=${ACTION_HORIZON:-16}"
  "EVALUATION.replan_steps=${REPLAN_STEPS:-12}"
  EVALUATION.visualize_future_video=false
  "EVALUATION.output_dir=${OUTPUT_DIR}"
  "hydra.run.dir=${OUTPUT_DIR}"
)

printf 'Native LIBERO inference output: %s\n' "${OUTPUT_DIR}"
printf 'Console log: %s\n' "${OUTPUT_DIR}/run.log"
printf 'GPU visibility: %s; MuJoCo: %s\n' "${CUDA_VISIBLE_DEVICES}" "${MUJOCO_GL}"
"${PYTHON_BIN}" -u experiments/libero/eval_libero_single.py "${ARGS[@]}" "$@" 2>&1 | tee "${OUTPUT_DIR}/run.log"

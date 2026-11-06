#!/usr/bin/env bash
# One model load, adaptive warmup, uninstrumented timings, then module timings.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
for argument in "$@"; do
  case "${argument}" in
    --output-dir|--output-dir=*)
      echo 'Use the OUTPUT_DIR environment variable so the console log and reports share one directory.' >&2
      exit 2
      ;;
  esac
done
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
FLUX2_SRC="${FLUX2_SRC:-${REPO_ROOT}/third_party/flux2}"
LIBERO_DIR="${LIBERO_DIR:-${REPO_ROOT}/third_party/LIBERO}"
export PYTHONPATH="${REPO_ROOT}/src:${FLUX2_SRC}/src:${LIBERO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MUJOCO_GL="${MUJOCO_GL:-egl}" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${REPO_ROOT}/.libero}"
export HF_HOME="${REPO_ROOT}/.cache/huggingface" TORCH_HOME="${REPO_ROOT}/.cache/torch"
export XDG_CACHE_HOME="${REPO_ROOT}/.cache" MPLCONFIGDIR="${REPO_ROOT}/.cache/matplotlib" NUMBA_CACHE_DIR="${REPO_ROOT}/.cache/numba"
mkdir -p "${HF_HOME}" "${TORCH_HOME}" "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/configure_3090_libero.py" --libero-dir "${LIBERO_DIR}" --config-dir "${LIBERO_CONFIG_PATH}" --patch-init-state-load
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/evaluate_results/3090_profile/${RUN_ID}}"
if [ -e "${OUTPUT_DIR}" ]; then
  printf 'Output already exists; choose a new RUN_ID or OUTPUT_DIR: %s\n' "${OUTPUT_DIR}" >&2
  exit 2
fi
mkdir -p "${OUTPUT_DIR}"
ARGS=(
  --output-dir "${OUTPUT_DIR}"
  --checkpoint "${CKPT_PATH:-${REPO_ROOT}/../flux2_klein_4b/model.pt}"
  --dataset-stats "${DATASET_STATS_PATH:-${REPO_ROOT}/../flux2_klein_4b/dataset_stats.json}"
  --flux2-src "${FLUX2_SRC}"
  --visual-weights "${FLUX2_MODEL_PATH:-${REPO_ROOT}/checkpoints/3090/flux2_imagewam_visual.safetensors}"
  --ae-weights "${FLUX2_AE_MODEL_PATH:-${REPO_ROOT}/checkpoints/3090/ae.safetensors}"
  --qwen-path "${FLUX2_QWEN3_MODEL_SPEC:-${REPO_ROOT}/checkpoints/3090/Qwen3-4B}"
  --libero-dir "${LIBERO_DIR}"
  --task-suite "${TASK_SUITE:-libero_spatial}" --task-id "${TASK_ID:-0}"
  --seed "${SEED:-42}" --steps "${INFERENCE_STEPS:-10}"
  --horizon "${ACTION_HORIZON:-16}" --replan-steps "${REPLAN_STEPS:-12}"
  --context-length "${QWEN_CONTEXT_LEN:-512}"
  --iterations "${BENCH_ITERS:-20}" --profile-iterations "${PROFILE_ITERS:-10}"
  --warmup-min "${WARMUP_MIN:-6}" --warmup-max "${WARMUP_MAX:-20}"
  --stability-window "${STABILITY_WINDOW:-3}" --stability-tolerance "${STABILITY_TOLERANCE:-0.10}"
)
printf 'Performance output: %s\n' "${OUTPUT_DIR}"
"${PYTHON_BIN}" -u "${SCRIPT_DIR}/benchmark_3090_inference.py" "${ARGS[@]}" "$@" 2>&1 | tee "${OUTPUT_DIR}/run.log"

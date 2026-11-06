#!/usr/bin/env bash
# Install an inference overlay while preserving the existing CUDA/PyTorch stack.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BASE_PYTHON="${BASE_PYTHON:-/home/ubuntu/miniconda3/envs/py312/bin/python}"
UV_BIN="${UV_BIN:-${HOME}/.local/bin/uv}"
if [ ! -x "${UV_BIN}" ]; then
  UV_BIN="$(command -v uv || true)"
fi
if [ ! -x "${BASE_PYTHON}" ] || [ ! -x "${UV_BIN}" ]; then
  echo "Set BASE_PYTHON to the existing PyTorch interpreter and UV_BIN to an installed uv executable." >&2
  exit 2
fi

cd "${REPO_ROOT}"
RUN_DIR="${REPO_ROOT}/artifacts/3090_setup/run_$(date -u +%Y%m%dT%H%M%SZ)_$$"
mkdir -p "${RUN_DIR}" "${REPO_ROOT}/.cache/uv"
exec > >(tee -a "${RUN_DIR}/installation.log") 2>&1

export UV_CACHE_DIR="${REPO_ROOT}/.cache/uv"
export UV_NO_CONFIG=1
export UV_PYTHON_DOWNLOADS=never
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${REPO_ROOT}/src"
VENV_PYTHON="${REPO_ROOT}/.venv/bin/python"
GUARD="${SCRIPT_DIR}/guard_3090_environment.py"
LOCK_FILE="${RUN_DIR}/requirements.lock"
EXCLUDES_FILE="${RUN_DIR}/inherited-gpu-packages.txt"

echo "Installing the project-local FLUX.2/LIBERO inference overlay. Logs: ${RUN_DIR}"
"${UV_BIN}" --version
UV_COMPILE_HELP="$("${UV_BIN}" pip compile --help)"
case "${UV_COMPILE_HELP}" in
  *--excludes*) ;;
  *)
    echo "This setup requires uv pip compile --excludes; the existing uv executable does not support it." >&2
    exit 2
    ;;
esac
"${BASE_PYTHON}" "${GUARD}" snapshot "${RUN_DIR}/base_before.json" --repo "${REPO_ROOT}"

# Capture and compare the stack even if dependency resolution or installation fails.
finish() {
  local result=$?
  trap - EXIT
  set +e
  "${BASE_PYTHON}" "${GUARD}" snapshot "${RUN_DIR}/base_after.json" --repo "${REPO_ROOT}"
  if [ "$?" -eq 0 ]; then
    "${BASE_PYTHON}" "${GUARD}" compare "${RUN_DIR}/base_before.json" "${RUN_DIR}/base_after.json"
    if [ "$?" -ne 0 ]; then result=1; fi
  else
    result=1
  fi
  if [ -x "${VENV_PYTHON}" ]; then
    "${VENV_PYTHON}" "${GUARD}" snapshot "${RUN_DIR}/venv_after.json" --repo "${REPO_ROOT}"
    if [ "$?" -eq 0 ]; then
      "${BASE_PYTHON}" "${GUARD}" compare "${RUN_DIR}/base_before.json" "${RUN_DIR}/venv_after.json" --overlay
      if [ "$?" -ne 0 ]; then result=1; fi
    else
      result=1
    fi
  fi
  echo "Setup exit status: ${result}; installation log: ${RUN_DIR}/installation.log"
  exit "${result}"
}
trap finish EXIT

if [ ! -e "${REPO_ROOT}/.venv" ] && [ ! -L "${REPO_ROOT}/.venv" ]; then
  "${UV_BIN}" venv --python "${BASE_PYTHON}" --system-site-packages "${REPO_ROOT}/.venv"
fi
if [ ! -x "${VENV_PYTHON}" ]; then
  echo "Existing .venv has no executable Python; refusing to replace it." >&2
  exit 2
fi
"${VENV_PYTHON}" "${GUARD}" snapshot "${RUN_DIR}/venv_before.json" --repo "${REPO_ROOT}"
"${BASE_PYTHON}" "${GUARD}" compare "${RUN_DIR}/base_before.json" "${RUN_DIR}/venv_before.json" --overlay
"${BASE_PYTHON}" "${GUARD}" exclusions "${RUN_DIR}/base_before.json" "${EXCLUDES_FILE}"

if [ -f "${REPO_ROOT}/requirements/3090-inference.lock" ] && [ "${UPDATE_LOCK:-0}" != 1 ]; then
  cp "${REPO_ROOT}/requirements/3090-inference.lock" "${LOCK_FILE}"
else
  "${UV_BIN}" pip compile --quiet \
    --python "${VENV_PYTHON}" \
    --excludes "${EXCLUDES_FILE}" \
    --output-file "${LOCK_FILE}" \
    "${REPO_ROOT}/requirements/3090-inference.in"
fi
"${BASE_PYTHON}" "${GUARD}" check-lock "${LOCK_FILE}"

# --no-deps is intentional: only the reviewed, GPU-free compiled list is installed.
"${UV_BIN}" pip install --python "${VENV_PYTHON}" --no-deps --dry-run -r "${LOCK_FILE}"
"${BASE_PYTHON}" "${GUARD}" check-lock "${LOCK_FILE}"
"${UV_BIN}" pip install --python "${VENV_PYTHON}" --no-deps -r "${LOCK_FILE}"

"${VENV_PYTHON}" - <<'PY'
import accelerate
import av
import datasets
import einops
import hydra
import mujoco
import numpy
import omegaconf
import safetensors
import torch
import torchvision
import transformers
from imagewam.models.backbones.imagewam import ImageWAM
from imagewam.runtime import create_imagewam_flux2_klein
from imagewam.datasets.lerobot.processors.imagewam_processor import ImageWAMProcessor
from imagewam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT

print("Core inference and LIBERO data imports passed.")
print(f"torch={torch.__version__}, torchvision={torchvision.__version__}, CUDA={torch.version.cuda}")
print(f"transformers={transformers.__version__}, numpy={numpy.__version__}, mujoco={mujoco.__version__}")
print("Native simulator assets, external FLUX.2 source, model files, and a model forward require separate verification.")
PY

echo "Run inference with .venv/bin/python and PYTHONPATH=${REPO_ROOT}/src."
echo "The original pyproject.toml and uv.lock are preserved; do not uv sync this overlay."

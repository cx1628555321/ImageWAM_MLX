#!/usr/bin/env bash
# Each mode runs in its own process: one model load, validation, warmup and repeats.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
for argument in "$@"; do
  case "${argument}" in
    --execution-mode|--execution-mode=*|--output-dir|--output-dir=*)
      echo 'Use EXECUTION_MODES / COMPARISON_DIR to choose modes and output.' >&2
      exit 2
      ;;
  esac
done
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export TORCHINDUCTOR_CACHE_DIR="${REPO_ROOT}/.cache/torchinductor"
export TRITON_CACHE_DIR="${REPO_ROOT}/.cache/triton"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
export MAX_JOBS="${MAX_JOBS:-1}"
COMPARISON_ID="${COMPARISON_ID:-modes_$(date +%Y%m%d_%H%M%S)_$$}"
COMPARISON_DIR="${COMPARISON_DIR:-${REPO_ROOT}/evaluate_results/3090_modes/${COMPARISON_ID}}"
if [ -e "${COMPARISON_DIR}" ]; then
  printf 'Comparison directory already exists: %s\n' "${COMPARISON_DIR}" >&2
  exit 2
fi
mkdir -p "${COMPARISON_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"
printf 'Comparison output: %s\n' "${COMPARISON_DIR}"
any_failed=0
for mode in ${EXECUTION_MODES:-eager inference_mode compile torchscript cudagraph}; do
  case "${mode}" in eager|inference_mode|compile|torchscript|cudagraph) ;; *) echo "Unsupported execution mode: ${mode}" >&2; exit 2 ;; esac
  if OUTPUT_DIR="${COMPARISON_DIR}/${mode}" timeout --signal=TERM --kill-after=30s "${MODE_TIMEOUT_SECONDS:-1800}s" \
    bash "${SCRIPT_DIR}/profile_3090_inference.sh" \
      --execution-mode "${mode}" --validate-execution \
      --execution-boundaries "${EXECUTION_BOUNDARIES:-qwen,ae,visual,action}" "$@"; then
    result=0
  else
    result=$?
    any_failed=1
  fi
  printf '%s\t%s\n' "${mode}" "${result}" >> "${COMPARISON_DIR}/exit_codes.tsv"
done
"${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}" "${SCRIPT_DIR}/summarize_3090_modes.py" "${COMPARISON_DIR}"
exit "${any_failed}"

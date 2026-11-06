#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
COMPARISON_ID="${COMPARISON_ID:-loaders_$(date +%Y%m%d_%H%M%S)_$$}"
RESULT_ROOT="${REPO_ROOT}/evaluate_results/3090_loaders/${COMPARISON_ID}"
if [ -e "${RESULT_ROOT}" ]; then
  echo "Refusing to overwrite ${RESULT_ROOT}" >&2
  exit 2
fi
mkdir -p "${RESULT_ROOT}"
OUTPUT_DIR="${RESULT_ROOT}/native" bash "${SCRIPT_DIR}/profile_3090_inference.sh" \
  --execution-mode eager --loader native --audit-loading --validate-execution
OUTPUT_DIR="${RESULT_ROOT}/action_only" bash "${SCRIPT_DIR}/profile_3090_inference.sh" \
  --execution-mode eager --loader action_only --audit-loading --validate-execution \
  --reference-dir "${RESULT_ROOT}/native"
"${REPO_ROOT}/.venv/bin/python" "${SCRIPT_DIR}/summarize_3090_loaders.py" "${RESULT_ROOT}"
echo "Loader comparison complete: ${RESULT_ROOT}"

#!/usr/bin/env bash
# Normal LIBERO rollout, using the separately implemented one-pass loader.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CHECKPOINT="${CKPT_PATH:-${REPO_ROOT}/../flux2_klein_4b/model.pt}"
exec bash "${SCRIPT_DIR}/infer_3090_libero.sh" \
  model._target_=imagewam.optimized_action_loader.create_imagewam_flux2_action_only \
  "+model.checkpoint_path=${CHECKPOINT}" "$@"

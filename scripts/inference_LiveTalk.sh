#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BITS="${1:-8}"
if [[ $# -gt 0 ]]; then shift; fi
if [[ "${1:-}" == "--" ]]; then shift; fi

exec python "${ROOT_DIR}/scripts/run_livetalk_varq.py" \
  --bits "${BITS}" \
  "$@"

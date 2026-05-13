#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SF_DIR="${ROOT_DIR}/third_party/Self-Forcing"
CONFIG="${1:-${ROOT_DIR}/configs/self_forcing/varq/base/SF-VARQ-8.json}"

if [[ $# -gt 0 ]]; then shift; fi
if [[ "${1:-}" == "--" ]]; then shift; fi

if [[ ! -d "${SF_DIR}" ]]; then
  echo "Missing third_party/Self-Forcing. Clone https://github.com/guandeh17/Self-Forcing to ${SF_DIR}." >&2
  exit 1
fi

if [[ $# -eq 0 ]]; then
  echo "Usage: bash scripts/inference_SelfForcing.sh [config] -- <upstream self-forcing inference command>" >&2
  echo "Example: bash scripts/inference_SelfForcing.sh configs/self_forcing/varq/base/SF-VARQ-4.json -- python sample.py ..." >&2
  exit 2
fi

export PYTHONPATH="${ROOT_DIR}:${SF_DIR}:${PYTHONPATH:-}"
export SELF_FORCING_VARQ_CONFIG="${CONFIG}"
export VARQ_CONFIG_FILE="${CONFIG}"
export VARQ_BACKEND="self_forcing"

cd "${SF_DIR}"
"$@"

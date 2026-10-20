#!/usr/bin/env bash
set -euo pipefail
root=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
cd "$root"
if [[ ! -x .venv/bin/python ]]; then
  echo 'First create the Linux .venv and install vLLM/test dependencies; see AZURE-AGENT.md.' >&2
  exit 2
fi
exec .venv/bin/python validation/oot_norm/run.py "$@"
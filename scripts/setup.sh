#!/usr/bin/env bash
# Install locally and verify the no-key workflow. Never runs paid inference.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python3}"
if [[ ! -d .venv ]]; then "$PYTHON" -m venv .venv; fi
if [[ -x .venv/bin/python ]]; then
  VENV_PY=.venv/bin/python
elif [[ -f .venv/Scripts/python.exe ]]; then
  VENV_PY=.venv/Scripts/python.exe
else
  printf '%s\n' 'No usable project venv interpreter was found.' >&2
  exit 2
fi
EXTRAS='.[dev]'
if [[ "${1:-}" == '--full' ]]; then EXTRAS='.[all,dev]'; fi
"$VENV_PY" -m pip install -e "$EXTRAS"
"$VENV_PY" -m typewright doctor
if [[ "${1:-}" == '--full' ]]; then
  "$VENV_PY" -m typewright doctor --check-optional
fi
"$VENV_PY" -m pytest -q
"$VENV_PY" -m typewright demo --out "runs/setup-demo-$(date +%Y%m%d-%H%M%S)-$$"
printf '%s\n' 'Local setup and synthetic demo complete. No live inference was run.'

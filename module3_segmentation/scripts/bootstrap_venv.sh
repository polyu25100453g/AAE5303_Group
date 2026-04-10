#!/usr/bin/env bash
# Create .venv under module3_segmentation/ and install requirements (not in git).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [ ! -x ".venv/bin/python" ]; then
  echo "Creating .venv in $ROOT ..."
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
echo ""
echo "OK. Before every session run:"
echo "  cd \"$ROOT\""
echo "  source .venv/bin/activate"

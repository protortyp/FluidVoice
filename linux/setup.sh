#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# System python provides PyGObject (gi); the venv reuses it.
uv venv --system-site-packages --python /usr/bin/python3 .venv
uv pip install --python .venv/bin/python -r requirements.txt

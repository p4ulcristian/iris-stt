#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

SOPS_FILE="$PWD/secrets/main.sops.yaml"
SOPS_AGE_KEY_FILE="${SOPS_AGE_KEY_FILE:-$HOME/.config/sops/age/keys.txt}" \
  sops --decrypt --output-type dotenv "$SOPS_FILE" > .env

set -a
source .env
set +a

# main.sops.yaml is shared with iris-os which uses 8080; force our port.
export PORT=4260
# STT (parakeet-tdt-0.6b-v3) and TTS (chatterbox-turbo) both run in-process;
# there are no external model services. IRIS_VOICE overrides the voice clip.

# The venv must run on a uv-managed CPython (see [tool.uv] in pyproject.toml):
# the Nix python's private glibc cannot load the system NVIDIA driver, which
# surfaced as "CUDA error: unknown error" while loading the STT model.
# No LD_PRELOAD / LD_LIBRARY_PATH shims are needed with the managed python.
exec uv run --managed-python --python 3.11 server.py

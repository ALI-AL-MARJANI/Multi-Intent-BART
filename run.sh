#!/usr/bin/env bash
# Wrapper that sets required environment variables before running any GEMIS script.
# TensorFlow is installed but broken on this system — USE_TF=0 prevents transformers
# from attempting to import it (which causes a fatal mutex crash on macOS).

set -euo pipefail
export USE_TF=0
export TOKENIZERS_PARALLELISM=false
exec python3 "$@"

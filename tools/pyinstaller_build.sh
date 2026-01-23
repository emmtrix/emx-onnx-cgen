#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cli_path="${repo_root}/src/emx_onnx_cgen/cli.py"

command=(
  python -m PyInstaller
  --clean
  --noconfirm
  --onedir
  --name emx-onnx-cgen
  --paths src
  "${cli_path}"
)

command+=("$@")

"${command[@]}"

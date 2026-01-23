#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
entrypoint_path="${repo_root}/tools/pyinstaller_entrypoint.py"
lowering_modules="$(
  REPO_ROOT="${repo_root}" python - <<'PY'
import os
import sys

repo_root = os.environ["REPO_ROOT"]
sys.path.insert(0, os.path.join(repo_root, "src"))

from emx_onnx_cgen import lowering  # noqa: E402

for module_name in lowering._LOWERING_MODULES:
    print(f"emx_onnx_cgen.lowering.{module_name}")
PY
)"

command=(
  python -m PyInstaller
  --clean
  --noconfirm
  --onedir
  --name emx-onnx-cgen
  --paths src
  "${entrypoint_path}"
)

while IFS= read -r module_name; do
  command+=(--hidden-import "${module_name}")
done <<<"${lowering_modules}"

command+=("$@")

"${command[@]}"

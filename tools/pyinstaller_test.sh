#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! python -m PyInstaller --version >/dev/null 2>&1; then
  echo "PyInstaller is not available. Install it in the active environment to run this test." >&2
  exit 1
fi

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

dist_dir="${tmp_dir}/dist"
build_dir="${tmp_dir}/build"
spec_dir="${tmp_dir}/spec"

python "${repo_root}/tools/pyinstaller_build.py" \
  --distpath "${dist_dir}" \
  --workpath "${build_dir}" \
  --specpath "${spec_dir}"

"${dist_dir}/emx-onnx-cgen/emx-onnx-cgen" --help >/dev/null
compile_output="${tmp_dir}/single_relu.c"
"${dist_dir}/emx-onnx-cgen/emx-onnx-cgen" compile \
  "${repo_root}/onnx-org/examples/resources/single_relu.onnx" \
  "${compile_output}"
test -s "${compile_output}"

echo "PyInstaller build succeeded."

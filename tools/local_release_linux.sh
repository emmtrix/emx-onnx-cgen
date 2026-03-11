#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image="quay.io/pypa/manylinux_2_28_x86_64"
python_bin="/usr/bin/python3.11"
output_archive="${OUTPUT_ARCHIVE:-emx-onnx-cgen-linux-amd64.tar.gz}"

if ! command -v docker >/dev/null 2>&1; then
  echo "error: docker is required for local Linux release builds." >&2
  exit 1
fi

# Keep these package/install commands aligned with .github/workflows/linux-release.yml.
docker run --rm \
  -v "${repo_root}:/workspace" \
  -w /workspace \
  -e OUTPUT_ARCHIVE="${output_archive}" \
  "${image}" \
  /bin/bash -lc '
    set -euo pipefail

    if command -v dnf >/dev/null 2>&1; then
      dnf install -y python3.11 python3.11-devel gcc gcc-c++ make
    elif command -v yum >/dev/null 2>&1; then
      yum install -y python3.11 python3.11-devel gcc gcc-c++ make
    elif command -v apt-get >/dev/null 2>&1; then
      apt-get update
      apt-get install -y --no-install-recommends python3.11 python3.11-dev build-essential
      rm -rf /var/lib/apt/lists/*
    else
      echo "No supported package manager found (dnf/yum/apt-get)." >&2
      exit 1
    fi

    "${python_bin}" --version
    "${python_bin}" -m ensurepip --upgrade || true
    "${python_bin}" -m pip install --upgrade pip
    "${python_bin}" -m pip install -r requirements-ci.txt
    "${python_bin}" -m pip install pyinstaller

    if [[ -f tools/build_linux_pyinstaller_release.sh ]]; then
      bash tools/build_linux_pyinstaller_release.sh
      if [[ "${output_archive}" != "emx-onnx-cgen-linux-amd64.tar.gz" ]]; then
        mv -f emx-onnx-cgen-linux-amd64.tar.gz "${output_archive}"
      fi
    else
      "${python_bin}" tools/pyinstaller_build.py
      tar -czf "${output_archive}" -C dist emx-onnx-cgen
    fi
  '

echo "Linux release bundle created: ${repo_root}/${output_archive}"

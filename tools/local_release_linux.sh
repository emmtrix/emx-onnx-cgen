#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image="quay.io/pypa/manylinux_2_28_x86_64"
python_bin="/opt/release-venv/bin/python"
output_archive="${OUTPUT_ARCHIVE:-emx-onnx-cgen-linux-amd64.tar.gz}"
output_sbom="${output_archive%.tar.gz}.cdx.json"
# Version recorded in the SBOM. Defaults to setuptools_scm, which cannot read the
# repository inside the container when it is a git worktree.
release_version="${RELEASE_VERSION:-}"

if ! command -v docker >/dev/null 2>&1; then
  echo "error: docker is required for local Linux release builds." >&2
  exit 1
fi

# Keep these package/install commands aligned with .github/workflows/linux-release.yml.
docker run --rm \
  -v "${repo_root}:/workspace" \
  -w /workspace \
  -e PYTHON_BIN="${python_bin}" \
  -e OUTPUT_ARCHIVE="${output_archive}" \
  -e OUTPUT_SBOM="${output_sbom}" \
  -e RELEASE_VERSION="${release_version}" \
  "${image}" \
  /bin/bash -lc '
    set -euo pipefail

    dnf install -y gcc gcc-c++ make
    python="$(/opt/python/cp312-cp312/bin/python3 tools/fetch_release_python.py \
      --platform linux-amd64 --dest /opt/release-python)"
    "${python}" -m venv /opt/release-venv

    "${PYTHON_BIN}" --version
    "${PYTHON_BIN}" -m pip install --upgrade pip
    "${PYTHON_BIN}" -m pip install -r requirements-ci.txt
    "${PYTHON_BIN}" -m pip install pyinstaller

    if [[ -f tools/build_linux_pyinstaller_release.sh ]]; then
      bash tools/build_linux_pyinstaller_release.sh
      if [[ "${OUTPUT_ARCHIVE}" != "emx-onnx-cgen-linux-amd64.tar.gz" ]]; then
        mv -f emx-onnx-cgen-linux-amd64.tar.gz "${OUTPUT_ARCHIVE}"
      fi
    else
      "${PYTHON_BIN}" tools/pyinstaller_build.py
      tar -czf "${OUTPUT_ARCHIVE}" -C dist emx-onnx-cgen
    fi

    if [[ -z "${RELEASE_VERSION}" ]]; then
      git config --global --add safe.directory /workspace
      RELEASE_VERSION="$("${PYTHON_BIN}" -m setuptools_scm)"
    fi
    "${PYTHON_BIN}" tools/generate_sbom.py bundle \
      --workpath build/emx-onnx-cgen \
      --dist dist/emx-onnx-cgen \
      --version "${RELEASE_VERSION}" \
      --platform linux-amd64 \
      --artifact "${OUTPUT_ARCHIVE}" \
      --output "${OUTPUT_SBOM}" \
      --python-build tools/release_python.json
  '

echo "Linux release bundle created: ${repo_root}/${output_archive}"
echo "SBOM created: ${repo_root}/${output_sbom}"

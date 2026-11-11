# Development

This document describes how to set up a local development environment for **emx-onnx-cgen** and how to run the most common developer workflows.

For end-user usage and CLI flags, also see `README.md`.

---

## Prerequisites

- **Python >= 3.10** (project requirement).
- A C compiler for end-to-end verification (one of `cc`, `gcc`, `clang`, or provide via `--cc` / `CC`).
- `git` (needed for submodules; tests can auto-init them if available).

---

## Repository overview

The project is a Python package with a CLI:

- Package sources live under `src/` (import root is `src/emx_onnx_cgen`).
- Templates are shipped as package data (Jinja2 templates).

The repository also contains submodules used for tests and reference material (ONNX Runtime-derived artifacts and the emx-pytorch2c reference implementation). ONNX operator specs come from the installed onnx package, see `tools/onnx_op_spec.py`.

---

## Setup (recommended)

### 1) Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell:
# .\.venv\Scripts\Activate.ps1
````

### 2) Install dependencies

For development and CI parity, install the same set used by CI:

```bash
python -m pip install -U pip
python -m pip install -r requirements-ci.txt
```

### 3) Install the package in editable mode

```bash
python -m pip install -e .
```

This should make the CLI available:

```bash
emx-onnx-cgen --help
```

---

## Local release builds (matching GitHub release workflows)

To reproduce the release artifacts locally with the same settings as the
GitHub workflows:

- Linux (uses the same Docker image as `.github/workflows/linux-release.yml`):

  ```bash
  ./tools/local_release_linux.sh
  ```

- Windows (matches the install/build steps from
  `.github/workflows/windows-release.yml`):

  ```powershell
  powershell -ExecutionPolicy Bypass -File .\tools\local_release_windows.ps1
  ```

  If the exact pinned `onnxruntime` version from `requirements-ci.txt` is not
  available on your package index, the script falls back to a compatible
  version (`onnxruntime>=<pinned>,<2`). To enforce strict pin-only behavior, use:

  ```powershell
  powershell -ExecutionPolicy Bypass -File .\tools\local_release_windows.ps1 -DisableOnnxruntimeFallback
  ```

Both scripts produce the same archive names used by the release workflows:

- `emx-onnx-cgen-linux-amd64.tar.gz`
- `emx-onnx-cgen-windows-amd64.zip`

Required at runtime (both `compile` and `verify`):

- `onnx`
- `numpy`
- `jinja2`

Optional for verification and tests:

- `onnxruntime`
- A C compiler (`cc`, `gcc`, `clang` or via `--cc`)
- `pytest` for ONNX backend compliance smoke tests

---

## Common workflows

## Run the CLI locally (from sources)

When iterating on the CLI entrypoint and you want to ensure you are running the in-tree code:

```bash
PYTHONPATH=src python -m emx_onnx_cgen --help
```

---

## Compile an ONNX model

Generate a `.c` file from an ONNX model:

```bash
emx-onnx-cgen compile path/to/model.onnx build/model.c
```

Notes:

* The compiler aims for deterministic, auditable C output (no hidden runtime, static memory layout).
* There are CLI options for testbench emission and splitting large data/weights.

---

## Verify a model end-to-end

Verification compares the generated code against a runtime backend (ONNX reference runtime or ONNX Runtime depending on CLI args).

```bash
emx-onnx-cgen verify path/to/model.onnx
```

If a test or expectation uses `--sanitize` but your local compiler setup cannot
link sanitizers, you can override the flag via `EMX_ENABLE_SANITIZE=0`:

```bash
# Linux/macOS
EMX_ENABLE_SANITIZE=0 emx-onnx-cgen verify --sanitize path/to/model.onnx

# Windows PowerShell
$env:EMX_ENABLE_SANITIZE = "0"
emx-onnx-cgen verify --sanitize path/to/model.onnx
```

Verification conceptually:

1. Compile with an emitted testbench (JSON I/O)
2. Build + run the generated C testbench using the selected compiler
3. Run the runtime backend on the same inputs
4. Compare outputs (floating outputs by ULP threshold; non-floats must match exactly)

---

## Testing

### Quick targeted tests

Prefer running a focused subset while developing:

```bash
pytest -q tests/test_some_module.py
```

If you need to exercise full verification (instead of the early-exit checksum
path) in the official/local ONNX file tests, you can set:

```bash
export DISABLE_CHECKSUM=1
```

When `UPDATE_REFS=1` is set, expected checksums may still be passed if present.
Set `UPDATE_REFS=2` to skip checksum usage for tests that are already expected
to fail (non-OK expectations), or `UPDATE_REFS=3` to skip checksums entirely.

### Full suite

```bash
pytest -q --maxfail=10
```

Notes:

* The suite can take several minutes; `pytest` defaults to `-n auto` via xdist.
* Tests include “golden” expectations of generated artifacts and expected error JSON files.

---

## Updating golden references

If you intentionally changed codegen output or expected-error baselines, regenerate references by running:

```bash
UPDATE_REFS=1 pytest -q --maxfail=10
```

In CI (PRs), the test workflow runs with `UPDATE_REFS=1` and may auto-commit updated references to the PR branch (for branches in the same repo).

---

## Test data (onnx-backend-test-data / emx-ort-test-artifacts-org)

Some tests require data that is not checked into this repository:

* Official ONNX backend test data in `onnx-backend-test-data/` (generated and gitignored, see its
  `README.md`). pytest assembles it on startup from the installed onnx package
  (`onnx-backend-test-data/generate.py`): the `node/` tests are generated from
  `onnx.backend.test.case.node` (onnx 1.23 no longer ships them), the other kinds are
  copied from the wheel. The installed onnx must match the `onnx==` pin in `requirements-ci.txt`;
  otherwise pytest warns and skips the official tests. A `.onnx-version` stamp triggers a rebuild
  when the pin changes. Run `python onnx-backend-test-data/generate.py` to build it without pytest.
* Additional ONNX Runtime-derived artifact models under `emx-ort-test-artifacts-org/artifacts/onnxruntime`
  (submodule).

CI initializes submodules recursively. Locally, you can do the same:

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

### Auto-init behavior

The test suite initializes the `emx-ort-test-artifacts-org` submodule automatically if `git` is
available. When the data is missing, the relevant tests are skipped.

---

## Formatting and linting

The CI requirements include:

* `ruff` (linting)
* `black` (formatting)

Typical local usage:

```bash
ruff check .
black .
```

To detect potentially unused code in `src/`:

```bash
vulture
```

If the repository later adds explicit tool configuration in `pyproject.toml` (or dedicated config files), follow those settings.

---

## Release / packaging (FYI)

The repository contains GitHub workflows that build release bundles using **PyInstaller** on:

* Linux (container-based build)
* Windows

These workflows set up the release Python, install `requirements-ci.txt` and `pyinstaller` into it, run `tools/pyinstaller_build.py`, and package the resulting bundle as a release asset.
Because the build environment also contains the test tooling, `tools/pyinstaller_build.py`
excludes modules that only optional code paths import (`EXCLUDED_MODULES`, e.g. `pytest`
via `sympy.testing`), so they do not end up in the bundle or its SBOM.

### Release Python

Both bundles are built with the same CPython build from
[python-build-standalone](https://github.com/astral-sh/python-build-standalone),
so that they ship the same interpreter version and the same versions of the
libraries compiled into it (OpenSSL, SQLite, xz, …). Expat and zlib still differ:
python-build-standalone uses CPython's bundled Expat and zlib-ng on Windows, and
its own builds on Linux. `tools/release_python.json` pins the release, the
CPython version and the SHA-256 of each platform's archive. The `install_only_stripped`
variant is used because the unstripped `libpython` on Linux is about four times larger.
`tools/fetch_release_python.py` downloads the archive, checks the hash and unpacks it.
The workflows then create a virtual environment from it.

Choosing the version: use the newest CPython minor version that still gets bugfix
releases. Only those get binary installers on python.org, and python-build-standalone
follows the same schedule. Every dependency in `requirements-ci.txt` also needs wheels
for it. Move on to the next minor version before the current one becomes security-only.

To update it:

1. Pick a python-build-standalone release and take the SHA-256 of
   `cpython-<version>+<release>-<target>-install_only_stripped.tar.gz` for both
   targets from its `SHA256SUMS`.
2. Update `libraries`. These are the versions of libraries that CPython cannot
   report at runtime, and the SBOM takes them from this table:
   * Linux: the `version` entries of `pythonbuild/downloads.json` at the release tag.
   * Windows: `bzip2` and `xz` come from `downloads.json` as well. `libffi` comes from
     the `cpython-source-deps` commit that `cpython-windows/build.py` checks out
     (the `AC_INIT` line of its `configure.ac`).
3. Build both bundles (`tools/local_release_*.sh/.ps1`).

### Software bill of materials (SBOM)

`tools/generate_sbom.py` writes a CycloneDX 1.6 JSON SBOM for each release
asset. The release workflows attach it to the GitHub release next to the asset
(`emx-onnx-cgen-<platform>.cdx.json`, and `emx-onnx-cgen-wheel.cdx.json` for
the PyPI wheel).

* **Bundles** (`bundle` mode, run right after PyInstaller with the same
  interpreter). The script reads PyInstaller's build records
  (`build/emx-onnx-cgen/*.toc`) and attributes every bundled file to its
  origin. It tries these sources in order:
  1. emx-onnx-cgen itself (`src/`), or a source vendored into it
     (`third_party/stb_image.h`).
  2. An installed Python distribution, found through its `RECORD`.
  3. PyInstaller's bootloader and loader.
  4. The Microsoft C/C++ runtime DLLs.
  5. On Linux, the rpm package that owns the file in the manylinux container.
  6. The CPython installation, together with the libraries compiled into the
     bundled interpreter and extension modules. Their versions come from the
     running interpreter where CPython reports them (OpenSSL, SQLite, Expat,
     zlib/zlib-ng, mpdecimal, zstd). Otherwise they come from the `libraries`
     table in `tools/release_python.json`, which the release workflows pass with
     `--python-build`; this covers bzip2, xz, libffi, uuid, ncurses and libedit.

  A file with no known origin is still listed as a `file` component with its
  hashes, and the script prints a warning.
* **Wheel** (`wheel` mode). The script runs in a fresh virtual environment into
  which the published wheel was installed. The SBOM lists the runtime
  dependency closure in the versions pip resolved there.

Known limitations:

* Native libraries that a Python wheel vendors are reported as part of that
  wheel, not as separate components. Examples are OpenBLAS inside `numpy` and
  libjpeg/libwebp inside `pillow`.
* Python package licenses are taken as declared in the package metadata.

To create an SBOM for a local bundle (`tools/local_release_*.sh/.ps1` do this
automatically):

```bash
python tools/pyinstaller_build.py
tar -czf emx-onnx-cgen-linux-amd64.tar.gz -C dist emx-onnx-cgen
python tools/generate_sbom.py bundle --workpath build/emx-onnx-cgen \
  --version "$(python -m setuptools_scm)" --platform linux-amd64 \
  --artifact emx-onnx-cgen-linux-amd64.tar.gz --output emx-onnx-cgen-linux-amd64.cdx.json
```

---

## Contribution guidelines (summary)

* Keep changes deterministic: stable naming/order and reproducible output.
* Prefer small, focused PRs.
* For changes that affect generated C output, update references and run verification-oriented tests.
* Avoid introducing new dependencies unless truly necessary.

See `AGENTS.md` for additional conventions, including operator-addition checklist and error-handling guidelines.

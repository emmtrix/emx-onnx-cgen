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

The repository also contains submodules used for tests and reference material (e.g. official ONNX backend test data and local operator tests).

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

## Common workflows

## Run the CLI locally (from sources)

When iterating on the CLI entrypoint and you want to ensure you are running the in-tree code:

```bash
PYTHONPATH=src python -m emx_onnx_cgen.cli --help
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
pytest -q tests/test_some_module.py -n auto
```

If you need to exercise full verification (instead of the early-exit checksum
path) in the official/local ONNX file tests, you can set:

```bash
export DISABLE_CHECKSUM=1
```

When `UPDATE_REFS=1` is set, expected checksums may still be passed if present.

### Full suite

```bash
pytest -n auto -q --maxfail=10
```

Notes:

* The suite can take several minutes; use `-n auto` (xdist) for speed.
* Tests include “golden” expectations of generated artifacts and expected error JSON files.

---

## Updating golden references

If you intentionally changed codegen output or expected-error baselines, regenerate references by running:

```bash
UPDATE_REFS=1 pytest -n auto -q --maxfail=10
```

In CI (PRs), the test workflow runs with `UPDATE_REFS=1` and may auto-commit updated references to the PR branch (for branches in the same repo).

---

## Submodules (onnx-org / onnx2c-org)

Some tests require data from submodules:

* Official ONNX backend test data (may be stored via Git LFS in the upstream project).
* Local operator test data.

CI initializes submodules recursively. Locally, you can do the same:

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

### Auto-init behavior

The test suite may attempt to initialize certain submodules automatically if:

* `git` is available, and
* auto-init is enabled via environment.

If you want to skip auto-init behavior for the official ONNX data, set:

```bash
export ONNX_ORG_AUTO_INIT=0
```

(When disabled and data is missing, the relevant tests will be skipped.)

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

If the repository later adds explicit tool configuration in `pyproject.toml` (or dedicated config files), follow those settings.

---

## Release / packaging (FYI)

The repository contains GitHub workflows that build release bundles using **PyInstaller** on:

* Linux (container-based build)
* Windows

These workflows install `requirements-ci.txt`, install `pyinstaller`, run `tools/pyinstaller_build.py`, and package the resulting bundle as a release asset.

---

## Contribution guidelines (summary)

* Keep changes deterministic: stable naming/order and reproducible output.
* Prefer small, focused PRs.
* For changes that affect generated C output, update references and run verification-oriented tests.
* Avoid introducing new dependencies unless truly necessary.

See `AGENTS.md` for additional conventions, including operator-addition checklist and error-handling guidelines.

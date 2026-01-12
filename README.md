# emmtrix ONNX to C compiler

`emmtrix ONNX to C compiler` compiles ONNX models into portable, deterministic C code. The
generated C is designed to be readable, stable across runs, and easy to embed in
larger projects without heavy runtime dependencies.

## Goals

- Correctness-first compilation with outputs comparable to ONNX Runtime.
- Deterministic and reproducible C code generation.
- Clean, pass-based compiler architecture (import → normalize → optimize → lower → emit).
- Minimal C runtime with explicit, predictable data movement.

## Non-goals

- Aggressive performance optimizations in generated C.
- Implicit runtime dependencies or dynamic loading.
- Training/backpropagation support.

## Features

- CLI for ONNX-to-C compilation and verification.
- Deterministic codegen with explicit tensor shapes and loop nests.
- Minimal C runtime templates in `templates/`.
- ONNX Runtime comparison for end-to-end validation.
- Official ONNX operator coverage tracking.

## Requirements

- Python 3.9+
- `onnx` for compilation
- Optional for verification:
  - `onnxruntime`
  - `numpy`
  - A C compiler (uses `cc`, `gcc`, or `clang`, or `CC`/`--cc`)

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-ci.txt
pip install -e .
```

## Quickstart

Compile an ONNX model into a C source file:

```bash
python -m onnx2c compile path/to/model.onnx build/model.c
```

Verify an ONNX model end-to-end against ONNX Runtime:

```bash
python -m onnx2c verify path/to/model.onnx
```

## CLI Reference

`onnx2c` provides two subcommands: `compile` and `verify`.

### `compile`

```bash
python -m onnx2c compile <model.onnx> <output.c> [options]
```

Options:

- `--template-dir`: Directory containing the C templates (default: `templates`).
- `--model-name`: Override the generated model name (default: output file stem).
- `--emit-testbench`: Emit a JSON-producing `main()` testbench for validation.

### `verify`

```bash
python -m onnx2c verify <model.onnx> [options]
```

Options:

- `--template-dir`: Directory containing the C templates (default: `templates`).
- `--model-name`: Override the generated model name (default: model file stem).
- `--cc`: Explicit C compiler command for building the testbench binary.

## Output

The compiler emits a single C source file that includes:

- A generated entry point that mirrors the ONNX graph inputs/outputs.
- Tensor buffers for constants and temporaries.
- A lightweight runtime implemented via templates in `templates/`.

## Official ONNX test coverage

See [`OFFICIAL_ONNX_FILE_SUPPORT.md`](OFFICIAL_ONNX_FILE_SUPPORT.md) for the generated support matrix.

## Project layout

- `src/onnx2c`: Compiler package.
- `templates/`: C templates and runtime snippets.
- `tests/`: Pytest-based tests and ONNX fixtures.
- `onnx2c-org/`, `emx-pytorch2c-org/`: Reference implementations for semantics.

## Tests

```bash
UPDATE_REFS=1 pytest -n auto -q
```

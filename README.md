# emmtrix ONNX to C Compiler (emx-onnx2c)

[![PyPI - Version](https://img.shields.io/pypi/v/emx-onnx2c.svg)](https://pypi.org/project/emx-onnx2c)

`emx-onnx2c` compiles ONNX models to portable, deterministic C code for deeply embedded systems. The generated code is designed to run without dynamic memory allocation, operating systems, or external runtimes, making it suitable for safety-critical and resource-constrained targets.

Key characteristics:

- **No dynamic memory allocation** (`malloc`, `free`, heap usage)
- **Static, compile-time known memory layout** for parameters, activations, and temporaries
- **Deterministic control flow** (explicit loops, no hidden dispatch or callbacks)
- **No OS or libc dependencies** beyond basic C
- **Single-threaded execution model**
- **Bitwise-stable code generation** for reproducible builds
- **Readable, auditable C code** suitable for certification and code reviews
- Designed for **bare-metal and RTOS-based systems**

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
- Support for a wide range of ONNX operators (see `OFFICIAL_ONNX_FILE_SUPPORT.md`).
- Supported data types:
  - `float`, `double`, `float16`
  - `int8_t`, `uint8_t`, `int16_t`, `uint16_t`, `int32_t`, `uint32_t`, `int64_t`, `uint64_t`
  - `bool`
- Supporting dynamic dimensions by utilizing C99 variable-length arrays (VLAs).

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
- `--emit-data-file`: Emit constant data arrays into a companion `_data` C file.
- `--no-restrict-arrays`: Disable `restrict` qualifiers on generated array parameters.

### `verify`

```bash
python -m onnx2c verify <model.onnx> [options]
```

Options:

- `--template-dir`: Directory containing the C templates (default: `templates`).
- `--model-name`: Override the generated model name (default: model file stem).
- `--cc`: Explicit C compiler command for building the testbench binary.

## Output

By default, the compiler emits a single C source file that includes:

- A generated entry point that mirrors the ONNX graph inputs/outputs.
- Tensor buffers for constants and temporaries.
- A lightweight runtime implemented via templates in `templates/`.

When `--emit-data-file` is enabled, the main C source declares constant arrays
as `extern`, and a second file named like the output with a `_data` suffix
contains the constant definitions.

## Official ONNX test coverage

See [`OFFICIAL_ONNX_FILE_SUPPORT.md`](OFFICIAL_ONNX_FILE_SUPPORT.md) for the generated support matrix.

## Maintained by

This project is maintained by [emmtrix](https://www.emmtrix.com).

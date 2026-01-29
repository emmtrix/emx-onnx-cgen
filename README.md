# emmtrix ONNX-to-C Code Generator (emx-onnx-cgen)

[![PyPI - Version](https://img.shields.io/pypi/v/emx-onnx-cgen.svg)](https://pypi.org/project/emx-onnx-cgen)

`emx-onnx-cgen` compiles ONNX models to portable, deterministic C code for deeply embedded systems. The generated code is designed to run without dynamic memory allocation, operating-system services, or external runtimes, making it suitable for safety-critical and resource-constrained targets.

Key characteristics:

- **No dynamic memory allocation** (`malloc`, `free`, heap usage)
- **Static, compile-time known memory layout** for parameters, activations, and temporaries
- **Deterministic control flow** (explicit loops, no hidden dispatch or callbacks)
- **No OS dependencies**, using only standard C headers (for example, `stdint.h` and `stddef.h`)
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
- Minimal C runtime templates in `src/emx_onnx_cgen/templates/`.
- ONNX Runtime comparison for end-to-end validation.
- Official ONNX operator coverage tracking.
- Support for a wide range of ONNX operators (see `OFFICIAL_ONNX_FILE_SUPPORT.md`).
- Supported data types:
  - `float`, `double`, `float16`
  - `int8_t`, `uint8_t`, `int16_t`, `uint16_t`, `int32_t`, `uint32_t`, `int64_t`, `uint64_t`
  - `bool`
- Optional support for dynamic dimensions using C99 variable-length arrays (VLAs), when the target compiler supports them.

## Installation

Install the package directly from PyPI (recommended):

```bash
pip install emx-onnx-cgen
```

Optional for verification and tests:

- `onnxruntime`
- `numpy`
- A C compiler (`cc`, `gcc`, `clang` or via `--cc`)

## Quickstart

Compile an ONNX model into a C source file:

```bash
emx-onnx-cgen compile path/to/model.onnx build/model.c
```

Verify an ONNX model end-to-end against ONNX Runtime (default):

```bash
emx-onnx-cgen verify path/to/model.onnx
```

## CLI Reference

`emx-onnx-cgen` provides two subcommands: `compile` and `verify`.

### `compile`

```bash
emx-onnx-cgen compile <model.onnx> <output.c> [options]
```

Options:

- `--model-base-dir`, `-B`: Base directory for resolving the model path (example: `emx-onnx-cgen compile --model-base-dir /data model.onnx out.c`).
- `--color`: Colorize CLI output (`auto`, `always`, `never`; default: `auto`).
- `--model-name`: Override the generated model name (default: output file stem).
- `--emit-testbench`: Emit a JSON-producing `main()` testbench for validation.
- `--emit-data-file`: Emit constant data arrays into a companion `_data` C file.
- `--large-weight-threshold`: Store weights in a binary file once the cumulative byte size exceeds this threshold (default: `102400`; set to `0` to disable).
- `--large-temp-threshold`: Mark temporary buffers larger than this threshold as static (default: `1024`).
- `--no-restrict-arrays`: Disable `restrict` qualifiers on generated array parameters.

### `verify`

```bash
emx-onnx-cgen verify <model.onnx> [options]
```

Options:

- `--model-base-dir`, `-B`: Base directory for resolving the model and test data paths (example: `emx-onnx-cgen verify --model-base-dir /data model.onnx --test-data-dir inputs`).
- `--color`: Colorize CLI output (`auto`, `always`, `never`; default: `auto`).
- `--model-name`: Override the generated model name (default: model file stem).
- `--cc`: Explicit C compiler command for building the testbench binary.
- `--large-weight-threshold`: Store weights in a binary file once the cumulative byte size exceeds this threshold (default: `102400`).
- `--large-temp-threshold`: Mark temporary buffers larger than this threshold as static (default: `1024`).
- `--max-ulp`: Maximum allowed ULP distance for floating outputs (default: `100`).
- `--atol-eps`: Absolute tolerance as a multiple of machine epsilon for floating outputs (default: `1.0`).
- `--runtime`: Runtime backend for verification (`onnxruntime` or `onnx-reference`, default: `onnxruntime`).
- `--temp-dir-root`: Root directory in which to create a temporary verification directory (default: system temp dir).
- `--temp-dir`: Exact directory to use for temporary verification files (default: create a temporary directory).
- `--keep-temp-dir`: Keep the temporary verification directory instead of deleting it.

How verification works:

1. **Compile with a testbench**: the compiler is invoked with `--emit-testbench`,
   generating a C program that runs the model and prints inputs/outputs as JSON.
2. **Build and execute**: the testbench is compiled with the selected C compiler
   (`--cc`, `CC`, or a detected `cc/gcc/clang`) and executed in a temporary
   directory.
3. **Run runtime backend**: the JSON inputs from the testbench are fed to the
   selected runtime (`onnxruntime` or `onnx-reference`) using the same model.
   The compiler no longer ships a Python runtime evaluator.
4. **Compare outputs**: floating outputs are compared by maximum ULP distance.
   Floating-point verification first ignores very small differences up to
   **--atol-eps × [machine epsilon](https://en.wikipedia.org/wiki/Machine_epsilon) of
   the evaluated floating-point type**, treating such values as equal. For
   values with a larger absolute difference, the ULP distance is computed, and
   the maximum ULP distance is reported; non-floating outputs must match
   exactly.
   Missing outputs or mismatches are treated as failures.
5. **ORT unsupported models**: when using `onnxruntime`, if ORT reports
   `NOT_IMPLEMENTED`, verification is skipped with a warning (exit code 0).

## Output

By default, the compiler emits a single C source file that includes:

- A generated entry point that mirrors the ONNX graph inputs/outputs.
- Tensor buffers for constants and temporaries.

When `--emit-data-file` is enabled, the main C source declares constant arrays
as `extern`, and a second file named like the output with a `_data` suffix
contains the constant definitions.

When `--large-weight-threshold` is set and a weight exceeds the threshold, the
compiler emits a `<model>.bin` file with weights packed contiguously and
generates a `<model>_load` helper that loads weights from the binary file at
runtime.

## Official ONNX test coverage

See [`OFFICIAL_ONNX_FILE_SUPPORT.md`](OFFICIAL_ONNX_FILE_SUPPORT.md) for the generated support matrix.
See [`SUPPORT_OPS.md`](SUPPORT_OPS.md) for operator-level support derived from the expectation JSON files.

## Maintained by

This project is maintained by [emmtrix](https://www.emmtrix.com).

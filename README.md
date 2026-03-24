<p align="center"><img width="50%" src="docs/assets/emx-onnx-cgen-logo.svg" /></p>

<div align="center">
  
[![PyPI - Version](https://img.shields.io/pypi/v/emx-onnx-cgen.svg)](https://pypi.org/project/emx-onnx-cgen)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://pypi.org/project/emx-onnx-cgen)
[![CI](https://github.com/emmtrix/emx-onnx-cgen/actions/workflows/tests.yml/badge.svg)](https://github.com/emmtrix/emx-onnx-cgen/actions/workflows/tests.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Vulture](https://img.shields.io/badge/dead%20code-vulture-blue?logo=python&logoColor=white)](https://github.com/jendrikseipp/vulture)

</div>

**emmtrix ONNX-to-C Code Generator (emx-onnx-cgen)** compiles ONNX models to portable, deterministic C code for deeply embedded systems. The generated code is designed to run without dynamic memory allocation, operating-system services, or external runtimes, making it suitable for safety-critical and resource-constrained targets.

Key characteristics:

- **No dynamic memory allocation** (`malloc`, `free`, heap usage)
- **Static, compile-time known memory layout** for parameters, activations, and temporaries
- **Deterministic control flow** (explicit loops, no hidden dispatch or callbacks)
- **No OS dependencies**, using only standard C headers (for example, `stdint.h` and `stddef.h`)
- **Single-threaded execution model**
- **Bitwise-stable code generation** for reproducible builds
- **Readable, auditable C code** suitable for certification and code reviews
- **Generated C output format spec:** [`docs/output-format.md`](docs/output-format.md)
- Designed for **bare-metal and RTOS-based systems**

For PyTorch models, see the related project [`emx-pytorch-cgen`](https://github.com/emmtrix/emx-pytorch-cgen).

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
- Support for a wide range of ONNX operators (see [`SUPPORT_OPS.md`](SUPPORT_OPS.md)).
- Supported data types:
  - `bfloat16`, `float16`, `float`, `double`
  - `float8e4m3fn`, `float8e4m3fnuz`, `float8e5m2`, `float8e5m2fnuz`, `float8e8m0` (stored as `uint8_t` with manual conversion to/from `float`)
  - `float4e2m1` (stored as `uint8_t` with manual conversion to/from `float`)
  - `int2`, `uint2`, `int4`, `uint4` (using C23 `_BitInt` types)
  - `int8`, `uint8`, `int16`, `uint16`, `int32`, `uint32`, `int64`, `uint64`
  - `bool`
  - `string` (fixed-size `'\0'`-terminated C strings; see [`docs/output-format.md`](docs/output-format.md))
  - `sequence(<tensor type>)` (fixed-capacity tensor sequences with presence/length metadata; see [`docs/output-format.md`](docs/output-format.md))
  - `optional(<tensor type>)` (optional tensors represented via an extra `_Bool <name>_present` flag; see [`docs/output-format.md`](docs/output-format.md))
  - Not supported: `complex64/complex128`, and ONNX `map/sparse_tensor/opaque` value types.
- Optional support for dynamic dimensions using C99 variable-length arrays (VLAs), when the target compiler supports them.

## Usage Scenarios

### 1. Fully Embedded, Standalone C Firmware

The generated C code can be embedded directly into a bare-metal C firmware or application where **all model weights and parameters are compiled into the C source**.

Typical characteristics:

* No file system or OS required.
* All weights stored as `static const` arrays in flash/ROM.
* Deterministic memory usage with no runtime allocation.
* Suitable for:
  * Microcontrollers
  * Safety-critical firmware
  * Systems with strict certification requirements

This scenario is enabled via --large-weight-threshold 0, forcing all weights to be embedded directly into the generated C code.

### 2. Embedded or Host C/C++ Application with External Weights

The generated C code can be embedded into C or C++ applications where **large model weights are stored externally and loaded from a binary file at runtime**.

Typical characteristics:

* Code and control logic compiled into the application.
* Large constant tensors packed into a separate `.bin` file.
* Explicit, generated loader functions handle weight initialization.
* Suitable for:
  * Embedded Linux or RTOS systems
  * Applications with limited flash but available external storage
  * Larger models where code size must be minimized

This scenario is enabled automatically once the cumulative weight size exceeds `--large-weight-threshold` (default: 102400 bytes).

### 3. Target-Optimized Code Generation via emmtrix Source-to-Source Tooling

In both of the above scenarios, the generated C code can serve as **input to emmtrix source-to-source compilation and optimization tools**, enabling target-specific optimizations while preserving functional correctness.

Examples of applied transformations include:

* Kernel fusion and loop restructuring
* Memory layout optimization and buffer reuse
* Reduction of internal temporary memory
* Utilization of SIMD / vector instruction sets
* Offloading of large weights to external memory
* Dynamic loading of weights or activations via DMA

This workflow allows a clear separation between:

* **Correctness-first, deterministic ONNX lowering**, and
* **Target-specific performance and memory optimization**,

while keeping the generated C code readable, auditable, and traceable.

The generated C code is intentionally structured to make such transformations explicit and analyzable, rather than relying on opaque backend-specific code generation.

## Installation

Install the package directly from PyPI (recommended):

```bash
pip install emx-onnx-cgen
```

Minimum Python version: **3.10**.

## Development

For local setup, testing, and contributor workflows, see [`docs/development.md`](docs/development.md).

## Quickstart

Compile an ONNX model into a C source file:

```bash
emx-onnx-cgen compile path/to/model.onnx build/model.c
```

Verify an ONNX model end-to-end against ONNX Runtime (default):

```bash
emx-onnx-cgen verify path/to/model.onnx
```

Models that require extra representative inputs to resolve dynamic shapes are not
supported for code generation. Export them with static shapes instead.

`--test-data-dir` is verification input/output data only. It does not change the
generated C code.

Use `emx-onnx-cgen` as an importable ONNX backend:

```python
import onnx
from onnx.backend import prepare

import emx_onnx_cgen.onnx_backend as emx_backend

model = onnx.load("path/to/model.onnx")
rep = prepare(model, backend=emx_backend)
outputs = rep.run(inputs)
```

The backend module is `emx_onnx_cgen.onnx_backend`. It compiles the ONNX model
to C on demand, builds a temporary executable, and runs that executable through
the standard ONNX backend interface.

You can also call it directly without `onnx.backend.prepare`:

```python
import onnx

from emx_onnx_cgen.onnx_backend import run_model

model = onnx.load("path/to/model.onnx")
outputs = run_model(model, inputs)
```

## CLI Reference

`emx-onnx-cgen` provides two subcommands: `compile` and `verify`.

### Common options

These options are accepted by both `compile` and `verify`:

- `--model-base-dir`: Base directory for resolving the model path (and related paths).
- `--color`: Colorize CLI output (`auto`, `always`, `never`; default: `auto`).
- `--verbose` / `-v`: Enable verbose logging (includes codegen timing).
- `--truncate-weights-after`: Truncate inline weight initializers after `N` values and insert `...` placeholders.
- `--large-weight-threshold`: Store weights in a binary file once the cumulative byte size exceeds this threshold (default: `102400`; set to `0` to disable).
- `--large-temp-threshold`: Mark local arrays larger than this threshold as static (default: `1024`). This applies to generated model temporaries and to generated testbench input/output buffers.
- `--restrict-arrays` / `--no-restrict-arrays`: Enable or disable `restrict` qualifiers on generated array parameters.
- `--fp32-accumulation-strategy`: Accumulation strategy for float32 inputs (`simple` uses float32, `fp64` uses double; default: `simple`).
- `--fp16-accumulation-strategy`: Accumulation strategy for float16 inputs (`simple` uses float16, `fp32` uses float; default: `fp32`).
- `--replicate-ort-bugs`: Compatibility switch for verification/debugging. Enables emulation of known behavior differences of the ONNX Runtime version pinned in `requirements-ci.txt`.
- `--sequence-element-shape`: Declare rank and per-axis maxima for sequence inputs with variable element shapes.

### `compile`

```bash
emx-onnx-cgen compile <model.onnx> [output.c] [options]
```

Options:

- `--model-name`: Override the generated model name (default: output file stem).
- `--emit-testbench`: Emit a JSON-producing `main()` testbench for validation.
- `--testbench-output-format`: Choose the generated testbench output format (`json`, `txt`, `txt-emmtrix`, or `txt-emmtrix:<float>`).
- `--testbench-file`: Emit the testbench into a separate C file at the given path (implies `--emit-testbench`). If not set, the testbench is embedded in the main output C file (legacy behavior).
- `--emit-data-file`: Emit constant data arrays into a companion `_data` C file.

### `verify`

```bash
emx-onnx-cgen verify <model.onnx> [options]
```

Options:

- `--cc`: Explicit C compiler command for building the testbench binary.
- `--sanitize`: Enable sanitizer instrumentation when compiling the verification binary (`-fsanitize=address,undefined`). If `EMX_ENABLE_SANITIZE` is set, it overrides this flag.
- `--per-node-accuracy`: Also compare intermediate tensor outputs and print max error per node.
- `--test-data-dir`: Seed verification inputs from `input_*.pb` files instead of generating random testbench inputs.
- `--test-data-inputs-only`: Read only `input_*.pb` from `--test-data-dir` and still compare outputs against the selected runtime.
- `--max-ulp`: Maximum allowed ULP distance for floating outputs (default: `100`).
- `--atol-eps`: Absolute tolerance as a multiple of machine epsilon for floating outputs (default: `1.0`).
- `--runtime`: Runtime backend for verification (`onnxruntime` or `onnx-reference`, default: `onnxruntime`).
- `--expected-checksum`: Exit early with `CHECKSUM` when the generated C checksum matches the expected SHA-256.
- `--replicate-ort-bugs`: Verification-only compatibility mode to reproduce known behavior differences of the ONNX Runtime version pinned in `requirements-ci.txt`.
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

## Official ONNX test coverage

- [`ONNX_SUPPORT.md`](ONNX_SUPPORT.md): overview of ONNX models and their current verification status.
- [`ONNX_ERRORS.md`](ONNX_ERRORS.md): summary of the most common verification outcomes and failure reasons.
- [`SUPPORT_OPS.md`](SUPPORT_OPS.md): list of ONNX operators and whether they are currently supported.

## Related Projects

- **emx-pytorch-cgen**  
  A PyTorch-to-C compiler following the same design principles as emx-onnx-cgen, but operating directly on PyTorch models instead of ONNX graphs.  
  https://github.com/emmtrix/emx-pytorch-cgen
- **onnx2c**  
  An ONNX-to-C code generator with a different design focus and code generation approach.  
  https://github.com/kraiskil/onnx2c

## Supporting Projects

- **emx-regex-cgen**  
  A regex-to-C code generator used to implement the ONNX `RegexFullMatch` operator in emx-onnx-cgen.  
  https://github.com/emmtrix/emx-regex-cgen
  
## Maintained by

This project is maintained by [emmtrix Technologies GmbH](https://www.emmtrix.com).

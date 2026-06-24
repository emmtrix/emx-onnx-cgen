# Source input

This file collects the currently known source material for the ONNX Community
Day presentation.

## User-provided event context

- The presentation is for an **ONNX Community Day** / ONNX community event.
- Target duration: **20 minutes**.
- The beginning should cover the **history** of why the project was started.

## Submitted abstract

> This talk presents lessons learned from developing emx-onnx-cgen, an
> ahead-of-time (AOT) compiler that translates ONNX models into portable C code
> for embedded and resource-constrained systems. A primary engineering goal of
> the project was achieving 100% test coverage across the compiler pipeline — a
> target that significantly influenced the architecture, testing strategy, and
> implementation details of the system.
>
> While ONNX is widely used as an interchange format between machine learning
> frameworks, practical experience shows that it is not well suited for
> deterministic AOT compilation without additional assumptions and constraints.
> Several aspects of the standard are effectively unbounded, including dynamic
> tensor dimensions, sequence lengths, and string sizes, making static memory
> planning and compile-time optimization difficult or impossible in the general
> case.
>
> The talk discusses challenges encountered during parsing, type inference,
> operator lowering, and code generation. In particular, ONNX type inference is
> incomplete for some operators, and container types such as sequence may contain
> arbitrarily mixed element types, complicating static analysis and code
> generation. Another recurring issue is the lack of clearly specified numerical
> accuracy requirements in both the ONNX standard and the official ONNX test
> cases, leading to ambiguities when validating generated code against reference
> implementations.

## User-provided shape/type inference point

- Another ONNX issue is shape and type inference.
- There is no sufficiently reliable, standalone library that computes shapes and
  types for ONNX models with dynamic dimensions in a way that is reusable by
  static code generators.
- As a result, code generators, MLIR importers, verifiers, and runtimes may all
  duplicate inference logic.
- A useful community direction would be a common, extensible ONNX shape/type
  inference tool that preserves symbolic dimensions, dynamic dimensions,
  constraints, and unresolved facts explicitly.
- The tool must be extensible for additional operator domains, for example
  Microsoft/ORT contrib operators, not only ONNX core operators.
- This likely needs a registry/plugin mechanism for operator-specific inference
  functions.
- Ideally, the results should be written back into ONNX models, for example via
  enriched `value_info`, metadata, or annotations.
- This would make the inference algorithm independent of individual code
  generators or MLIR-based flows.
- Code generators could consume a model with persisted inference results instead
  of reimplementing the algorithm.
- Current/vision work: we are working on a DSL for specifying shape/type
  inference rules.
- The goal is that the DSL can be used to compute shapes and types from both
  C++ and Python.
- The same rules should support numeric evaluation for concrete dimensions and
  symbolic evaluation, for example via SymPy, for symbolic dimensions.
- The DSL should make custom operators easier to support because their
  shape/type rules could be provided alongside the operator instead of being
  hard-coded into every backend.
- As a longer-term vision, such DSL rules could even be stored inside ONNX
  models for custom operators.
- Present this as vision/research direction, not as completed functionality.

## User-provided project history

1. The project started after using `onnx2c`.
   - `onnx2c` demonstrated that ONNX-to-C is useful.
   - The quality of the generated code was difficult.
   - The project needed more control over generated C code.

2. A main issue with `onnx2c` was support for dynamic dimensions.
   - There was an idea to experiment with **VLA parameters** in C.
   - This was one motivation for building a new implementation.

3. After initial success, the goal expanded.
   - The project shifted from an experiment/prototype toward broad operator and
     model coverage.
   - The goal became achieving as much ONNX coverage as possible.

## Project facts from repository material

From `README.md`:

- `emx-onnx-cgen` compiles ONNX models to portable, deterministic C code.
- Target use case: deeply embedded, safety-critical, and resource-constrained
  systems.
- Generated code is designed to avoid:
  - dynamic memory allocation
  - operating-system dependencies
  - external runtimes
  - hidden dispatch/callbacks
- Generated code aims to be:
  - deterministic
  - auditable
  - readable
  - suitable for reproducible builds
- The compiler targets ONNX opset 26 based on ONNX 1.21.0.
- It supports nearly all Microsoft ONNX operators based on ONNX Runtime 1.26.0.
- Reported coverage highlights are tracked in generated support documents:
  - operator coverage in `SUPPORT_OPS.md`
  - official ONNX backend model coverage in `ONNX_SUPPORT.md`
  - ONNX Runtime artifact coverage in `ONNX_SUPPORT.md`

From `README.md`, `docs/development.md`, and `ONNX_SUPPORT.md` on
`emx-ort-test-artifacts`:

- `emx-ort-test-artifacts` is a supporting emmtrix repository containing
  exported ONNX test artifacts (`*.onnx` / `*.pb`) produced by the ONNX Runtime
  test infrastructure.
- The principle is to generate ONNX-backend-test-like artifacts from ONNX
  Runtime tests: `model.onnx` plus protobuf input/output test data in
  `test_data_set_*` directories.
- This makes ORT test cases usable by other ONNX backends without depending on
  ORT's internal C++ test harness.
- ORT tests may also carry accuracy/validation constraints.
- In `emx-ort-test-artifacts`, those constraints are currently stored in
  per-test `validation.json` files.
- Observed `validation.json` fields include:
  - `expects_failure`
  - `expected_failure_substring`
  - per-output `relative_error`
  - per-output `absolute_error`
  - per-output `sort_output`
- For broader ONNX backend compatibility, these validation constraints should
  be converted to the ONNX backend-test `data.json` format.
- This would make each exported test more self-contained: model, protobuf test
  data, and comparison/tolerance metadata in the expected ONNX backend-test
  convention.
- In this repository it is present as `emx-ort-test-artifacts-org/`.
- The test data lives under
  `emx-ort-test-artifacts-org/artifacts/onnxruntime`.
- The project treats it as an additional verification/test source, not as code
  to modify from this repository.
- `ONNX_SUPPORT.md` reports ONNX Runtime artifact coverage as a generated,
  corpus-level status metric.
- These artifacts are useful because they apply the official ONNX backend test
  style to a broader ONNX Runtime-derived corpus, including contrib/operator
  scenarios beyond the compact official ONNX backend node tests.
- The expectation JSON files record exact reproduction commands using
  `verify --model-base-dir ... model.onnx --test-data-dir test_data_set_0`.
- Talk framing: official ONNX backend tests are the standards-facing baseline;
  `emx-ort-test-artifacts` extends the same artifact idea to ORT tests and is
  the broader reality check against ORT-derived test corpora and edge cases.
- Community framing: the same artifact-generation principle could be useful for
  other ONNX backends that want to run ORT-derived tests outside the ORT test
  harness.
- User note: this idea was mentioned in ORT forums, but the response was close
  to zero. Mention carefully as an underused opportunity rather than as a
  complaint.

From ONNX Backend Scoreboard:

- Source: <https://onnx.ai/backend-scoreboard/>
- The page states that the score is based on the ONNX backend unit tests.
- In the checked current stable-build table, `emx-onnx-cgen` is listed directly
  after `ONNX Reference`.
- This can be mentioned as external validation that `emx-onnx-cgen` is not only
  a local experiment, but a visible ONNX backend.
- Talk framing: mention the scoreboard placement briefly as credibility, but do
  not make it the main claim. The scoreboard measures backend unit-test
  execution, not generated-code auditability, deterministic AOT constraints,
  static memory planning, or safety-oriented code quality.

From `SUPPORT_OPS.md`:

- `SUPPORT_OPS.md` tracks generated operator support status.
- Remaining unsupported entries are documented in the generated report.

From `docs/output-format.md`:

- Generated artifacts can include:
  - `<out>.c`
  - optional `<out>_data.c`
  - optional testbench C file
  - optional `<model_name>.bin` for external weights
- Public generated C API is intentionally small:
  - `_Bool <model_name>_load(const char *path);`
  - `void <model_name>(...inputs..., ...outputs...);`
- Tensor inputs/outputs become C array parameters.
- Dynamic dimensions can be represented using C99 VLAs.
- ONNX dynamic or symbolic dimensions do not provide maximum sizes.
- For a backend without dynamic allocation, VLA parameters can represent runtime
  extents in the function ABI, but they do not by themselves provide a static
  memory bound for temporaries or buffers.
- Sequence inputs with dynamic element dimensions require explicit
  `--sequence-element-shape` declarations.
- Sequence length has the same static-storage problem as string length:
  ONNX does not provide a maximum length, but a backend without dynamic memory
  needs an explicit bound.
- For strings the missing bound is maximum characters per string element; for
  sequences the missing bound is maximum number of sequence elements.
- ONNX sequences are represented in generated model entrypoints as
  fixed-capacity arrays plus sequence-count metadata.
- For sequence tensor IO, generated C uses:
  - input: `const T name[EMX_SEQUENCE_MAX_LEN][elem_shape...]` plus
    `idx_t name__count`
  - output: `T name[EMX_SEQUENCE_MAX_LEN][elem_shape...]` plus
    `idx_t *name__count`
- `EMX_SEQUENCE_MAX_LEN` defaults to `32` in generated code and is guarded by
  `#ifndef`, so it can be overridden by the build.
- Ragged sequence inputs are not accepted implicitly.
- For ragged sequence inputs, `--sequence-element-shape` declares rank and
  per-axis maxima, for example `sequence=[<=8]` or `boxes=[<=100,4]`.
- With ragged sequence hints, generated C uses max-sized element storage and
  extra per-item dimension arrays such as
  `idx_t name__dim_<axis>[EMX_SEQUENCE_MAX_LEN]`.
- Only the first `name__count` entries of per-item dimension arrays are
  meaningful.
- Testbench JSON output includes `item_shapes` for sequence outputs so verify
  can compare true per-item shapes instead of padded max-sized storage.
- Strings use fixed-size `'\0'`-terminated C strings with
  `EMX_STRING_MAX_LEN`.
- ONNX string tensors do not carry a maximum string length in the type.
- This is a problem for backends without dynamic memory allocation, because
  storage must be reserved statically or bounded through an explicit compiler
  policy.
- `emx-onnx-cgen` uses fixed-size C string slots:
  `char[EMX_STRING_MAX_LEN]` per string element.
- Generated C defines `EMX_STRING_MAX_LEN` as `256` by default under
  `#ifndef`, so the macro can be overridden by the build.
- In the testbench serialization path, strings longer than the fixed slot are
  truncated to `EMX_STRING_MAX_LEN - 1` bytes and null-padded.
- Integer and common floating-point mappings are mostly straightforward.
- `float16` uses `_Float16` where supported by the compiler/target.
- `bfloat16` uses `__bf16`, which is compiler/target-specific in practice.
- 2-bit and 4-bit integer types use C23 `_BitInt(N)` /
  `unsigned _BitInt(N)`.
- `_BitInt(N)` represents the value range, but does not guarantee compact packed
  tensor storage; memory can be wasted compared with manual bit packing.
- FP8/FP4 types are not C standard arithmetic types and are emulated using
  integer storage plus conversion helpers.
- Generated code uses explicit loop nests and deterministic ordering.

## emmtrix embedded AI compiler context

Public emmtrix sources:

- <https://www.emmtrix.com/wiki/emmtrix_ONNX-to-C_Code_Generator>
- <https://www.emmtrix.com/news/onnx-to-generic-c-for-embedded-ai.html>
- <https://www.emmtrix.com/tools/emmtrix-code-vectorizer>
- <https://www.emmtrix.com/tools/emmtrix-parallel-studio>
- <https://www.emmtrix.com/>

Findings from emmtrix website material:

- `emx-onnx-cgen` is described as an open-source AI frontend compiler.
- It translates ONNX models into deterministic, analyzable C code.
- The generated C is intended for auto-vectorization and embedded target
  optimization.
- The stated flow is:

```text
ONNX -> Clean C -> Vectorizer -> Target Architecture
```

- In the emmtrix toolchain, `emx-onnx-cgen` is the frontend that imports,
  normalizes, lowers, and emits vectorization-friendly C.
- The C output then feeds tools such as the emmtrix Vectorizer and backend
  toolchains for embedded architectures.
- Publicly stated design goals include deterministic code generation, fully
  static memory layout, explicit loops/control flow, predictable memory access
  patterns, readable/auditable C, compatibility with auto-vectorization tools,
  and suitability for embedded and safety-critical systems.
- Publicly stated code-generation principles include simple canonical loop
  forms, linear array accesses, no hidden pointer aliasing, no dynamic dispatch,
  no recursion, no dynamic memory allocation, and explicit tensor dimensions and
  strides.
- The release announcement frames generated generic C as a core building block
  of the embedded AI flow and emphasizes no dynamic memory allocation, no OS
  dependencies, no external runtimes, static control flow/memory layout, and
  correctness verification against ONNX Runtime.
- emmtrix Parallel Studio is described as a C source-to-source compiler flow
  that takes sequential C and generates optimized source code for a selected
  target architecture.
- The Parallel Studio material explicitly mentions use in functional safety
  contexts such as ISO 26262 and DO-178C when combined with the qualification
  kit.
- This supports the talk framing that C is not just an output language here. It
  is the handoff format into existing embedded, source-to-source optimization,
  and safety-critical tooling.

Internal/project framing for the talk:

- We use a disciplined subset of C plus conventions as an intermediate
  representation.
- `emx-onnx-cgen` is therefore a frontend in the emmtrix/eAI compiler flow.
- This is similar in spirit to MLIR-style staged lowering, but the exposed
  artifact is standard C source code rather than an internal compiler IR.
- This makes the generated artifact understandable, auditable, and compatible
  with heterogeneous C/C++ compilers, including compilers common in
  safety-critical embedded environments.
- Before starting the project, we looked for existing ways to generate clean C
  from ONNX or ML compiler stacks.
- Apart from `onnx2c`, there were few obvious solutions matching the desired
  standalone, deterministic, auditable C output.
- Apache TVM and IREE are relevant compiler stacks, but in our evaluation they
  did not provide the straightforward ONNX-to-generic-C flow needed at the time.

From `docs/development.md` and `README.md`:

- Verification compares generated C output against ONNX Runtime or ONNX
  Reference.
- Verification flow:
  1. generate C with testbench
  2. compile and execute the C testbench
  3. run the selected reference runtime on the same inputs
  4. compare outputs
- Floating-point outputs are compared using ULP-based thresholds after an
  absolute epsilon check.
- Non-floating outputs must match exactly.

## Accuracy and tolerance research notes

Checked sources:

- ONNX backend test loader:
  <https://github.com/onnx/onnx/blob/main/onnx/backend/test/loader/__init__.py>
- ONNX backend test runner:
  <https://github.com/onnx/onnx/blob/main/onnx/backend/test/runner/__init__.py>
- ONNX Runtime model tests:
  <https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/test/providers/cpu/model_tests.cc>
- ONNX Runtime test case config parsing:
  <https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/test/onnx/TestCase.cc>
- PyTorch `torch.testing.assert_close` documentation:
  <https://docs.pytorch.org/docs/stable/testing.html>
- TensorFlow Datasets testing API, including dtype-aware close checks:
  <https://www.tensorflow.org/datasets/api_docs/python/tfds/testing/TestCase>
- NumPy `assert_allclose` documentation:
  <https://numpy.org/doc/2.2/reference/generated/numpy.testing.assert_allclose.html>

Findings:

- ONNX backend tests use default tolerances `rtol=1e-3` and `atol=1e-7`.
- ONNX backend tests can override those values from test-case `data.json`.
- In the current checked ONNX tree, `data.json` occurs only for `real` model
  tests, not for `node` operator tests.
- Therefore `data.json` does not currently provide operator-specific accuracy
  tolerances for ONNX node/operator backend tests.
- ONNX backend output comparison uses `numpy.testing.assert_allclose`.
- NumPy-style allclose compares against `atol + rtol * abs(expected)`.
- ONNX Runtime model tests use default `per_sample_tolerance` and
  `relative_per_sample_tolerance`, both initialized to `1e-3` in the checked
  test runner path.
- ONNX Runtime test cases can override those tolerances through `config.txt`
  using `per_sample_tolerance` and `relative_per_sample_tolerance`.
- PyTorch `torch.testing.assert_close` uses dtype-dependent defaults, for
  example different tolerances for `float16`, `bfloat16`, `float32`, and
  `float64`.
- TensorFlow testing utilities include dtype-aware close checks such as
  `assertAllCloseAccordingToType`.

Additional framework/tooling comparison:

| Framework / tool | Mechanism | Default style | Notable point |
| --- | --- | --- | --- |
| NumPy | `numpy.testing.assert_allclose` | `rtol=1e-7`, `atol=0` | Uses `atol + rtol * abs(expected)`; `numpy.allclose` has different defaults (`rtol=1e-5`, `atol=1e-8`). |
| PyTorch | `torch.testing.assert_close` | dtype-dependent | Uses `atol + rtol * abs(expected)` and selects defaults by dtype, e.g. `float16`, `bfloat16`, `float32`, `float64`. |
| TensorFlow | `tf.test.TestCase.assertAllClose` and `assertAllCloseAccordingToType` | generic plus dtype-aware helper | `assertAllCloseAccordingToType` has separate settings for float32, float16 and bfloat16. |
| JAX | `jax.numpy.allclose` | NumPy-like | Uses `rtol=1e-5`, `atol=1e-8` and the standard `atol + rtol * abs(b)` condition. |
| MXNet | `mxnet.test_utils.assert_almost_equal` | configurable relative/absolute thresholds | Treats arrays as equal if either relative or absolute check passes. |
| NVIDIA Polygraphy | comparator CLI options | user-configurable, per-output | Supports per-output `--rtol` and `--atol`; docs note FP32 defaults can be too strict for FP16 or INT8. |

Project-specific notes:

- `emx-onnx-cgen` does not use a classic fixed `rtol`/`atol` comparison for
  verification.
- It first ignores absolute differences up to `atol_eps * eps(dtype)`.
- It then computes ULP distance for remaining floating-point differences.
- The CLI exposes this through `--atol-eps` and `--max-ulp`.
- Some official ONNX test expectations use
  `--fp32-accumulation-strategy fp64`, indicating that higher-precision
  accumulation was needed for selected test cases.

## Existing constraints and project principles

From project conventions:

- Correctness first.
- Deterministic output.
- Stable node ordering and stable symbol names.
- No hidden state in passes where possible.
- Explicit errors for unsupported operators or ambiguous shapes.
- Verification-only inputs must not change generated code implicitly.
- Models that require representative inputs to resolve dynamic shapes should be
  exported with static shapes or supplied with explicit compiler options.

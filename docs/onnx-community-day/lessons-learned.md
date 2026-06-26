# Lessons learned

This file expands the technical lessons that should inform the presentation.

## 1. Existing ONNX-to-C tooling proved the need, but not the final approach

The project history starts with `onnx2c`.

What worked:

- It demonstrated that compiling ONNX graphs to C is useful.
- It gave a concrete reference point for what generated C could look like.
- It helped identify practical requirements for embedded deployments.

What did not work well enough:

- Generated-code quality was difficult for the desired use case.
- There was not enough control over the generated code structure.
- Dynamic-dimension support was a major pain point.

Implication:

For embedded systems, generated code is not just an implementation detail. It is
an artifact that may need review, analysis, certification, debugging, and further
source-to-source optimization.

## 2. Dynamic dimensions need a C representation, but representation is not semantics

The VLA-parameter idea was one motivation for starting a new implementation.

Example:

```c
void model(int N, int C,
           const float x[restrict N][C],
           float y[restrict N][C]);
```

Why this is useful:

- C can express runtime tensor extents in the function signature.
- The generated ABI stays explicit.
- The rank remains statically known.
- Loop bounds can use explicit extent parameters.
- N-dimensional C array types carry more semantic information than flat 1D
  buffers.

Why N-dimensional arrays matter:

- Both N-dimensional arrays and flat 1D arrays can address the same contiguous
  tensor memory.
- But in C, the declared array type matters.
- Accessing beyond the bounds of a declared array object is undefined behavior,
  even if the next logical tensor row is contiguous in memory.
- With `float x[N][C]`, the rank and inner extent are visible in the type.
- With `float *x`, the tensor structure exists only as an offset-calculation
  convention.
- Keeping tensor structure in the type improves readability, reviewability, and
  downstream analysis.

Flat generated code:

```c
void add_flat(int N, int C,
              const float *restrict a,
              const float *restrict b,
              float *restrict out) {
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            int idx = n * C + c;
            out[idx] = a[idx] + b[idx];
        }
    }
}
```

N-dimensional VLA generated code:

```c
void add_vla(int N, int C,
             const float a[restrict N][C],
             const float b[restrict N][C],
             float out[restrict N][C]) {
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            out[n][c] = a[n][c] + b[n][c];
        }
    }
}
```

Limits:

- The target compiler must support VLAs.
- Memory may still be unbounded unless extents are constrained.
- It does not solve dynamic rank.
- It does not solve unbounded sequence length.
- It does not solve unbounded string size.

Takeaway:

VLAs are a useful lowering strategy for some dynamic shapes, but an AOT compiler
still needs explicit bounds and constraints.

## 3. Broad coverage changes the compiler architecture

After initial success, the project goal expanded to broad ONNX support.

This changed the system:

- The compiler needed a pass-based architecture.
- Import, normalization, lowering, and code generation needed clearer boundaries.
- Unsupported cases needed explicit, testable diagnostics.
- Coverage reports became part of the development workflow.
- Golden/reference updates became necessary for deterministic code generation.

Design consequence:

A high-coverage ONNX compiler cannot be a set of ad hoc special cases. It needs
registries, structured lowering, stable IR objects, and a verification strategy.

## 4. 100% test coverage is an architectural constraint

The project goal of 100% test coverage across the compiler pipeline influenced
implementation details.

It encouraged:

- small functions and classes
- clear pass contracts
- explicit context objects instead of global state
- deterministic output for stable golden tests
- predictable error handling
- separable verification logic

It discouraged:

- hidden global state
- oversized "god" modules
- implicit shape assumptions
- broad mutation during passes
- codegen behavior that depends on verification-only inputs

Talk framing:

The important point is not only "we wanted high coverage". The point is that the
coverage target forced better compiler architecture.

## 4a. ORT-derived artifacts made coverage more realistic

The test strategy did not stop at local unit tests or the official ONNX backend
tests. `emx-ort-test-artifacts` provides an additional corpus by exporting ONNX
Runtime tests into an ONNX-backend-test-like format: a model directory with
`model.onnx` plus protobuf input/output test data.

The principle is:

```text
ORT test case -> exported ONNX model + test_data_set_* -> backend-runnable test artifact
```

That makes ORT test cases usable outside the ORT C++ test harness.

The artifact should also carry the comparison contract. ORT tests can have
per-output validation expectations, including relative and absolute error
thresholds. In the current artifact repository, these are stored in
`validation.json` files with fields such as:

- `relative_error`
- `absolute_error`
- `sort_output`
- `expects_failure`
- `expected_failure_substring`

For ONNX-backend-test compatibility, these constraints should be translated to
the ONNX backend `data.json` format. That would make the generated artifacts
look more like native ONNX backend tests: model, protobuf test data, and
comparison tolerances in the expected place.

Why this matters:

- Official ONNX backend tests are an essential standards-facing baseline.
- ORT-derived artifacts apply a similar model-plus-test-data pattern to a
  broader runtime-oriented test corpus.
- They include model directories, test data sets, contrib/operator scenarios,
  and edge cases that are closer to how a production runtime validates itself.
- They force the compiler and verification pipeline to handle real model/test
  directory structure, not only hand-written micrographs.
- They can preserve ORT's intended numerical tolerances instead of forcing all
  backends into one global comparison policy.
- They make unsupported cases reproducible through recorded command lines.

Project use:

- The corpus is available as the `emx-ort-test-artifacts-org/` submodule.
- Test data lives under `emx-ort-test-artifacts-org/artifacts/onnxruntime`.
- The generated `ONNX_SUPPORT.md` report tracks this corpus separately from
  official ONNX backend coverage.
- Expected-error JSON files record exact `verify` commands for individual
  failing or unsupported artifacts.

Talk framing:

- Do not make exact coverage percentages the central claim.
- The important point is the engineering loop: broad corpus, deterministic
  verification, reproducible failures, and visible progress.
- `emx-ort-test-artifacts` helped turn "operator support" into a more realistic
  backend validation story.
- This could also be useful beyond `emx-onnx-cgen`: other ONNX backends could
  use ORT-exported, ONNX-backend-test-like artifacts to compare against a
  broader ORT-derived behavior corpus.
- The idea was mentioned in ORT-related forums, but the response was minimal.
  Present this neutrally as an underused community opportunity, not as a
  complaint.

## 5. ONNX is excellent interchange, but not naturally bounded AOT compiler IR

ONNX allows flexibility that is valuable for interchange and runtime execution.
For deterministic AOT compilation, that flexibility becomes a problem unless it
is constrained.

Problem areas:

- unknown tensor dimensions
- symbolic dimensions without concrete maxima
- dynamic output sizes
- unbounded sequence length
- unbounded string length
- container values that do not map cleanly to a static C ABI

Compiler response:

- require static shapes where necessary
- support VLAs where the shape is representable and acceptable
- require explicit sequence element shape declarations for ragged inputs
- use fixed-size string storage
- fail clearly when required information is missing

Community implication:

An AOT-friendly ONNX profile could be useful. It might specify what model
properties must be bounded or statically known for deterministic code generation.

Dynamic-dimension-specific issue:

- ONNX can mark dimensions as symbolic or unknown.
- That gives the compiler a runtime extent, but not a maximum extent.
- VLA parameters can represent known-rank tensors with runtime dimensions in the
  C ABI.
- VLAs do not solve memory planning by themselves: temporaries and caller buffers
  may still grow without a compile-time bound.
- A backend without dynamic memory must either require maxima, require
  caller-provided storage with capacities, accept unbounded stack usage, or
  reject the model.

Sequence-specific issue:

- ONNX sequences are container values, not just tensors with one more axis.
- A runtime can represent a sequence as a growable object or a list of tensor
  objects.
- This is the same basic problem as ONNX strings: the logical value has a
  variable, unbounded length, but static C storage needs an explicit maximum.
- For strings, the missing bound is maximum characters per string element.
- For sequences, the missing bound is maximum number of sequence elements.
- A static C ABI needs a bounded representation: maximum sequence length,
  element dtype, element rank, element shape, and sometimes per-item dynamic
  dimensions.
- Without a maximum length, the compiler cannot reserve fixed storage.
- Without element-shape information, the compiler cannot choose the C array
  type for the sequence payload.
- Ragged sequences are especially awkward: the data buffer needs max-sized
  storage, and the real per-item dimensions need separate metadata.
- Nested containers, optionals, or mixed practical representations make the
  analysis even less natural for generated C.

`emx-onnx-cgen` policy:

- support `sequence(tensor(...))` at the model IO ABI as fixed-capacity tensor
  arrays plus count metadata
- represent inputs as
  `const T name[EMX_SEQUENCE_MAX_LEN][elem_shape...]` plus `idx_t name__count`
- represent outputs as
  `T name[EMX_SEQUENCE_MAX_LEN][elem_shape...]` plus `idx_t *name__count`
- emit `#define EMX_SEQUENCE_MAX_LEN 32` by default, guarded with `#ifndef`
- allow users/toolchains to override the macro at compile time
- require explicit `--sequence-element-shape` hints for sequence inputs with
  unknown or dynamic element dimensions
- for ragged sequence inputs, allocate max-sized element storage and pass
  per-item dimension arrays such as `name__dim_<axis>[EMX_SEQUENCE_MAX_LEN]`
- keep only the first `name__count` entries meaningful

Talk framing:

- A sequence length is like a dynamic dimension, but one level higher.
- It is also directly analogous to string length: both are unbounded lengths in
  the ONNX value model that must become bounded storage in C.
- A tensor dimension asks: "how large is this axis?"
- A sequence asks: "how many tensor objects exist, and what is each tensor's
  shape?"
- That object-level variability is comfortable for runtimes, but not for static
  C without an explicit capacity contract.

String-specific issue:

- ONNX string tensors do not declare a maximum string length.
- A backend with dynamic allocation can represent variable-length strings at
  runtime.
- A backend without dynamic memory must reserve storage at compile time, for
  example `char x[...][EMX_STRING_MAX_LEN]`.
- Without a maximum, the compiler must either require a user-provided bound,
  pick a global policy, risk truncation/wasted memory, or reject the model.

`emx-onnx-cgen` policy:

- represent each string element as a fixed-size, null-terminated C char array
  `char[EMX_STRING_MAX_LEN]`
- emit `#define EMX_STRING_MAX_LEN 256` by default, guarded with `#ifndef`
- allow users/toolchains to override the macro at compile time
- keep memory layout static and deterministic
- accept the trade-off that arbitrary ONNX string length is not represented
  without an explicit bound
- in testbench serialization, truncate strings that exceed the fixed slot size
  so that the result remains null-terminated

## 6. ONNX type and shape inference cannot be the only analysis layer

Practical issue:

ONNX type and shape inference is incomplete for some operators or insufficient
for generated C decisions.

Examples of analysis difficulty:

- output rank or dimensions may depend on input values
- container element shapes may be missing
- subgraph operators may require pattern recognition
- dynamic-output operators may need capacity decisions

Compiler response:

- normalize attributes and types during import
- add compiler-side validation/inference where needed
- make missing information an explicit error
- avoid using verification data to fill codegen gaps implicitly

Community-facing gap:

- There is no broadly reliable, standalone library that computes shapes and
  types for ONNX models with dynamic dimensions well enough for static compiler
  use cases.
- Every serious backend or compiler is tempted to reimplement parts of this
  logic: ONNX importers, MLIR frontends, code generators, verifiers, and
  optimization tools.
- This duplicates effort and can produce different answers for the same model.
- A useful community artifact would be a shared and extensible shape/type
  inference library that understands symbolic and dynamic dimensions, preserves
  constraints, and reports unresolved information explicitly.
- Extensibility matters: the library should support registered inference rules
  for external operator domains, for example Microsoft/ORT contrib operators,
  not only the ONNX core operator set.
- Ideally, the inferred results and constraints could be written back into the
  ONNX model, for example as enriched `value_info` / metadata / annotations.
- Then code generators would not need to own the inference algorithm. They
  could consume a model with persisted analysis results.

Vision:

- We are exploring a DSL for specifying shape/type inference rules.
- The rule should be written once and be usable from both C++ and Python.
- The same rule should support numeric shape evaluation for concrete dimensions
  and symbolic shape evaluation for symbolic dimensions, for example via SymPy.
- A DSL makes custom operators more tractable: a custom operator could ship its
  own shape/type rule rather than requiring every backend to hard-code it.
- In the long term, such DSL rules could even be stored inside ONNX models as
  metadata for custom operators.
- This should be presented as a vision and research direction, not as a
  finished capability.

Talk framing:

- Shape/type inference should be separable from code generation.
- Shape/type inference should also be separable from a fixed operator universe;
  external domains need a plugin or registry mechanism.
- Persisting inference results in ONNX would make the analysis reusable across
  backends, MLIR import paths, validators, and source-to-source compilers.
- This is similar to the `emx-ort-test-artifacts` idea: make useful compiler
  knowledge portable as model artifacts rather than trapping it inside one
  backend implementation.

## 7. ONNX dtype mapping is uneven in C

Straightforward cases:

- standard integer tensors map to fixed-width C integer types
- `float` and `double` map naturally to C floating types
- `bool` maps naturally to C boolean storage

Less straightforward cases:

- `float16` can use `_Float16` where the C compiler and target support it
- `bfloat16` typically relies on compiler/target extensions such as `__bf16`
- 2-bit and 4-bit integer types can use C23 `_BitInt(N)`
- `_BitInt(N)` gives a type-level value range, but arrays may still waste memory
  compared with manually packed sub-byte storage
- FP8/FP4 types are not standard C arithmetic types
- `emx-onnx-cgen` represents FP8/FP4 values as small integer storage, usually
  `uint8_t`, plus explicit conversion helpers

Talk framing:

- common numeric types are boring in a good way
- reduced-precision and sub-byte types expose the gap between ONNX's type system
  and portable C
- FP8 support is currently more useful for type coverage and completeness than
  for efficient portable C execution

## 8. Numerical accuracy requirements need clearer contracts

When validating generated C against reference implementations, numerical
differences are not always straightforward.

Sources of ambiguity:

- accumulation precision
- rounding behavior
- math-library differences
- approximations
- implementation-defined edge cases
- differences between ONNX Runtime and ONNX Reference behavior
- official tests that do not define a complete tolerance policy

Current project response:

- compare floating outputs using ULP-based thresholds after an absolute epsilon
  check
- require exact matches for non-floating outputs
- expose compatibility switches where known runtime behavior needs to be
  replicated for debugging

Community implication:

Backend authors would benefit from clearer numerical accuracy expectations in
operator specifications and official tests.

Observed tolerance models:

- ONNX backend tests default to `rtol=1e-3` and `atol=1e-7`, with optional
  per-test `data.json` overrides.
- In the checked ONNX tree, `data.json` is used for `real` model tests, not for
  individual `node` operator tests.
- Therefore ONNX backend tests do not currently provide operator-specific
  tolerances through `data.json`.
- ONNX backend tests use a NumPy-style absolute-plus-relative check:
  `abs(actual - expected) <= atol + rtol * abs(expected)`.
- ONNX Runtime model tests use per-sample absolute and relative tolerances and
  can override them via test-case `config.txt`.
- PyTorch's `torch.testing.assert_close` uses dtype-dependent default
  tolerances, for example different defaults for `float16`, `bfloat16`,
  `float32`, and `float64`.
- TensorFlow testing also has dtype-aware close checks, such as
  `assertAllCloseAccordingToType`.
- JAX follows NumPy-style `allclose` defaults (`rtol=1e-5`, `atol=1e-8`) and
  the same absolute-plus-relative formula.
- MXNet's `assert_almost_equal` combines relative and absolute thresholds, but
  considers arrays equal if either check passes.
- NVIDIA Polygraphy exposes `--rtol` and `--atol` as comparator options and
  supports per-output tolerances; its docs explicitly warn that FP32-oriented
  defaults may be too strict for lower precision such as FP16 or INT8.

`emx-onnx-cgen` verification takes a different approach:

- ignore tiny absolute differences up to `atol_eps * eps(dtype)`
- compute ULP distance for remaining floating differences
- report the maximum ULP distance
- compare non-floating outputs exactly

Why this matters:

- A single fixed `rtol`/`atol` pair is not equally meaningful for all floating
  dtypes.
- `float16`, `float32`, and `float64` have very different spacing between
  representable values.
- ULP-based measurement is closer to the question "how many representable
  floating-point values apart are these results?"

Practical consequence in this project:

- Some ONNX reference outputs appear to be computed with very high precision.
- Combined with strict ULP-based verification, this required 64-bit accumulation
  for selected official tests.
- The compiler exposes this as `--fp32-accumulation-strategy fp64`.
- Several expectation files for official ONNX tests use that option.

## 8. Generated-code quality is a backend feature

For embedded and resource-constrained systems, generated C has different
requirements than a hidden runtime implementation.

Desired generated-code properties:

- readable
- deterministic
- auditable
- stable across runs
- explicit memory layout
- no dynamic allocation
- minimal runtime dependencies
- simple public API

This was one reason to move beyond `onnx2c` for this project.

External validation:

- The public ONNX Backend Scoreboard lists `emx-onnx-cgen` directly after
  `ONNX Reference` in the stable-build table.
- The scoreboard is based on ONNX backend unit tests, so it is a useful
  compatibility signal.
- It does not capture all AOT-compiler concerns: deterministic generated C,
  static memory planning, readable source, auditability, or safety-oriented
  build integration.
- Talk framing: use the scoreboard placement as credibility, then pivot to the
  deeper lesson that "can execute the unit tests" is only one dimension of
  backend quality.

## 9. C is the handoff IR in the emmtrix embedded AI flow

`emx-onnx-cgen` should be presented as more than a standalone ONNX-to-C
translator. In the emmtrix toolchain, it acts as an AI frontend compiler:

```text
ONNX model -> emx-onnx-cgen -> clean C -> emmtrix Vectorizer / backend tools
```

The generated C is deliberately used as an intermediate representation for
later source-to-source analysis and optimization. Internally, this means the
project is not merely "printing C"; it is producing a disciplined subset of C
plus conventions:

- explicit loop nests
- explicit tensor dimensions and strides
- predictable memory accesses
- stable symbol names and deterministic ordering
- no hidden dynamic dispatch
- no heap allocation
- minimal runtime assumptions

This is close in spirit to MLIR-style compilation: lower a high-level model into
a representation that later compiler passes can analyze and transform. The
important difference is the exposed handoff artifact. MLIR/LLVM-centered flows
often keep the key IR inside the compiler stack and then lower toward LLVM or a
runtime. The emmtrix flow intentionally exposes C source code as the handoff
format.

Why this matters:

- C is standardized and broadly understood.
- The result can be reviewed by engineers and static analysis tools.
- Existing embedded build systems can compile it.
- Heterogeneous C/C++ compilers can be used, including compilers from
  safety-critical environments.
- The generated artifact fits workflows where traceability, qualification, and
  source-level audits matter.

This also reframes generated-code quality:

- Poor generated C is not only ugly output.
- It weakens downstream analysis, vectorization, review, and safety arguments.
- Clean generated C is part of the compiler contract.

Relationship to other stacks:

- `onnx2c` was the most direct existing reference point for ONNX-to-C.
- Apache TVM and IREE are relevant ML compiler projects, but in our evaluation
  they did not provide the straightforward standalone ONNX-to-auditable-generic-C
  path we needed at the time.
- MLIR is the right conceptual comparison for staged lowering, but the practical
  trade-off is different: emmtrix uses C as the visible bridge into embedded and
  safety-critical toolchains.

## 10. Verification must preserve compile/verify parity

Important invariant:

Verification must not give the compiler extra information that normal
compilation does not receive.

Bad pattern:

- `verify` reads representative `input_*.pb` files
- those values or shapes make code generation succeed
- `compile` without those inputs would fail or generate different code

Project principle:

If a model requires representative inputs to resolve dynamic shapes, it should
fail clearly or require an explicit compiler option that is also available to
`compile`.

Why it matters:

- reproducibility
- deterministic code generation
- honest compiler contract
- meaningful test results

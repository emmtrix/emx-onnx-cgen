# Proposed 20 minute outline

This is a content outline, not yet a final slide deck.

## Slide 1: Title

**Lessons learned from building an AOT ONNX-to-C compiler**

Subtitle:

`emx-onnx-cgen` for embedded and resource-constrained systems

Speaker notes:

- Frame the talk as lessons learned.
- The focus is not just what the compiler supports, but what the project taught
  us about ONNX as an AOT compiler input.

## Slide 2: Why we started

Key points:

- We first used `onnx2c`.
- It demonstrated that ONNX-to-C is useful.
- Generated-code quality and control were difficult.
- Embedded use cases require readable, deterministic, reviewable C.

Speaker notes:

- Start with the practical history.
- Make clear that this was not initially a "build a complete ONNX compiler"
  project.
- The first driver was control over generated C.

## Slide 3: The dynamic-dimension trigger

Key points:

- A main issue with `onnx2c` was dynamic-dimension support.
- Idea: represent some dynamic dimensions with C99 VLA parameters.
- Example:

```c
void model(int N, int C,
           const float x[restrict N][C],
           float y[restrict N][C]);
```

Speaker notes:

- Explain that VLA parameters are a representation technique for known rank
  with runtime extents.
- They do not solve arbitrary unbounded ONNX semantics.
- This experiment helped justify building a new implementation.

## Slide 4: From experiment to coverage project

Key points:

- Initial experiments were successful.
- The goal expanded toward broad ONNX operator/model coverage.
- The project developed into an AOT compiler with verification and coverage
  reporting.

Speaker notes:

- This is the turning point in the story.
- Once coverage became a goal, the project needed architecture and testing
  discipline.

## Slide 5: What emx-onnx-cgen is - goals and generated C

Key points:

- Ahead-of-time ONNX-to-C compiler.
- Produces portable, deterministic C.
- Target: embedded and resource-constrained systems.
- Avoids dynamic memory allocation and external runtimes.
- Small generated API:
  - `<model_name>_load(...)`
  - `<model_name>(...)`

Goals (from the project README):

- Correctness-first: outputs comparable to ONNX Runtime.
- Static, compile-time memory layout - no `malloc`/`free`, no heap.
- No OS, no external runtime, no hidden dispatch or callbacks.
- Readable, auditable C suitable for review and certification.
- Pass-based pipeline: import -> normalize -> optimize -> lower -> emit.

Generated C output example (excerpt from the in-repo golden reference
`tests/golden/mul_add_relu_model.c`, a `Mul -> Add -> Relu` graph):

```c
/* tensors map to typed N-D C arrays; each ONNX node is one loop nest */
static inline float ref_scalar_f32_mul(float a, float b) { return a * b; }

EMX_NODE_FN void node0_mul(const float input0[2][3],
                           const float input1[2][3], float output[2][3]) {
    for (idx_t i0 = 0; i0 < 2; ++i0)
        for (idx_t i1 = 0; i1 < 3; ++i1)
            output[i0][i1] = ref_scalar_f32_mul(input0[i0][i1], input1[i0][i1]);
}
/* node1_add, node2_relu: same shape, one op each */

_Bool model_load(const char *path) { (void)path; return 1; }

void model(const float a[restrict 2][3], const float b[restrict 2][3],
           const float c[restrict 2][3], float out[restrict 2][3]) {
    float tmp0_mul_out[2][3];   /* static, stack temporaries */
    float tmp1_add_out[2][3];
    node0_mul(a, b, tmp0_mul_out);
    node1_add(tmp0_mul_out, c, tmp1_add_out);
    node2_relu(tmp1_add_out, out);   /* no malloc, no runtime, no OS */
}
```

Speaker notes:

- Keep this concise.
- The audience needs enough project context before the lessons learned.
- The example is a faithful excerpt of a real golden reference, not a mock-up:
  typed N-D arrays, one explicit loop nest per node, stack temporaries, and a
  two-function public API (`model_load` + `model`).

## Slide 6: Architecture shaped by testing

Key points:

- Primary engineering goal: 100% test coverage across the compiler pipeline.
- This shaped:
  - pass-based architecture
  - explicit IR boundaries
  - deterministic codegen
  - precise diagnostics
  - reference-based verification

Speaker notes:

- Coverage is not just a metric.
- It forced the system to be modular and deterministic.

## Slide 7: Verification loop

Key points:

- Generate C with a testbench.
- Compile and run generated C.
- Run ONNX Runtime or ONNX Reference on the same inputs.
- Compare outputs numerically.
- Record coverage and expected failures.

Important invariant:

- Verification-only inputs must not implicitly change generated code.

Speaker notes:

- Explain why compile/verify parity matters.
- If verification uses extra shape information, it is testing a different
  compiler path.

## Slide 8: Lesson 1 - ONNX is effectively unbounded

Key points:

- Dynamic tensor dimensions can be unknown.
- Sequence lengths are not naturally bounded.
- String tensor element sizes are not bounded.
- General static memory planning is impossible without additional assumptions.

Speaker notes:

- This is the central AOT-compilation lesson.
- ONNX is intentionally flexible as interchange format.
- Embedded C needs concrete bounds.

## Slide 9: Lesson 2 - Type inference is not enough

Key points:

- ONNX type/shape inference is incomplete for some operators.
- Some output information depends on attributes, values, or subgraph patterns.
- Container types such as sequence complicate static analysis and ABI design.

Speaker notes:

- Avoid framing this as a flaw only.
- The point is mismatch: flexible model format versus static compiler input.

## Slide 10: Lesson 3 - Numerical accuracy is underspecified

Key points:

- Official tests often provide examples, not full accuracy contracts.
- Reference implementations can differ in edge cases or floating-point behavior.
- Generated C needs clear decisions for:
  - accumulation precision
  - rounding
  - approximations
  - tolerance thresholds

Speaker notes:

- This is likely highly relevant to backend authors.
- Validation is an engineering policy unless the standard defines the contract.

## Slide 11: Current status

Key points:

- ONNX opset 26 target, based on ONNX 1.21.0.
- Broad Microsoft ONNX Runtime operator support.
- Generated reports track:
  - operator support
  - official ONNX backend model coverage
  - ONNX Runtime artifact coverage via `emx-ort-test-artifacts`
- Public ONNX Backend Scoreboard lists `emx-onnx-cgen` directly after
  `ONNX Reference` in the stable-build table.
- Remaining unsupported cases are documented as expected errors.

Speaker notes:

- Phrase coverage carefully as report-based and corpus-based.
- Do not make exact percentages the point.
- Mention the scoreboard as credibility, but clarify that it is based on ONNX
  backend unit tests.
- The main point is that coverage became systematic, reproducible, and visible.

## Slide 12: Takeaways for the ONNX community

Key points:

- AOT-friendly ONNX usage needs explicit bounds.
- Compiler-oriented infrastructure should include reliable, reusable shape/type
  inference for dynamic models.
- That inference infrastructure should be extensible for external domains such
  as Microsoft/ORT contrib operators.
- Inferred shape/type facts should ideally be persistable in ONNX models.
- Vision: a small DSL could describe shape/type rules once and drive both C++
  and Python inference, including symbolic evaluation.
- Numerical accuracy requirements should be more explicit.
- Generated-code quality, determinism, and auditability are backend features.

Speaker notes:

- End constructively.
- Ask what an AOT-friendly ONNX profile should define.

## Slide 13: Discussion

Questions:

- What should an AOT-friendly ONNX profile specify?
- Should ONNX have a stronger shared shape/type inference artifact for dynamic
  models?
- How should external operator domains register shape/type inference rules?
- Could custom operators carry portable shape/type rules inside ONNX?
- Which operators need clearer numerical contracts?
- How should backend scoreboards represent deterministic code generation and
  generated-code quality?

# Additional slides in the master deck (v3)

These slides exist in the extended master deck beyond the 13-slide 20-minute
structure above. They can be swapped in depending on the audience and time.

## How we use emx-onnx-cgen at emmtrix: C as IR

- C is the intermediate representation, not just the output (instead of MLIR).
- Flow: ONNX -> emx-onnx-cgen -> C (our IR) -> emmtrix optimizer -> target.
- Why C, not MLIR: standardized, reviewable, toolchain-agnostic, no IR lock-in.
- Then the generated C is optimized source-to-source: node fusion, memory
  reduction, vectorization (SIMD), memory-layout optimization, buffer reuse,
  weight offloading / DMA.
- Example: emmtrix Edge AI Compiler output for RISC-V with RVV (project issue
  #723) — a `Gemm -> Relu` model becomes vector FMA accumulation with the Relu
  fused into the vector store; the loop nest (over N, K) is preserved, 16 is the
  vector length, not the problem size.

## Explicit typed arrays - even for dynamic models

- We always emit explicit, typed N-dimensional C arrays; rank and per-axis
  extents live in the C type, not in hand-written index math.
- `x[n][c]` mirrors the tensor; out-of-bounds is UB on the declared object, so it
  is analyzable; cleaner for review and vectorization.
- This holds even for dynamic models: C99 VLAs (`x[N][C]`) keep the array
  explicit with runtime extents, for parameters and local temporaries.
- Keeping arrays explicit for dynamic shapes (onnx2c's weak spot) is why the
  project started.

## Data types: complete coverage

- Every ONNX value type maps to an explicit C representation.
- For ~100% coverage you need them all - the awkward ones (sub-byte, FP8/FP4,
  strings, containers) cannot be skipped.
- Table: integers & bool, floating-point, sub-byte integers (C23 `_BitInt`),
  low-precision floats (uint8 storage + converters), strings, sequences
  (`x[EMX_SEQUENCE_MAX_LEN][...]` + `__count`), optional (value + `_present`).
- Not supported: `complex64/128`, `map`, `sparse_tensor`, `opaque`.

## Sequences: the element type is underspecified

- `sequence(T)` gives the element dtype, but the element tensor's concrete shape
  (sometimes its rank) can be unspecified, and may vary per item (ragged).
- A runtime resolves it at execution; static C must know it up front.
- emx-onnx-cgen requires an explicit extra specification on the CLI:
  `--sequence-element-shape name=[<=max, ...]` -> element rank + per-axis maxima;
  capacity from `EMX_SEQUENCE_MAX_LEN`; variable axes get `name__dim_<axis>`.
- Missing / insufficient spec -> fail clearly, never a silent guess.

## Proposal: size bounds in the ONNX type system

- Framing: NOT an emx-specific feature — a general extension of the ONNX type
  system that every backend and tool benefits from.
- Gap is general: ONNX types carry no max size for strings / sequences / dynamic
  dims, so each static backend invents its own (emx's global macro is just one
  symptom -> waste or truncation).
- Proposal: make a maximum size an optional, standard part of the type — per-type
  default + per-tensor override, carried in type info / `value_info`,
  backwards-compatible.
- Type-level sketch (not emx metadata): `tensor(string)[max_len=64]`,
  `sequence(tensor(float)[<=100, 4])[max_len=20]` — the element bound carries the
  tensor rank and per-dimension maxima, positionally (axis 0 <= 100, axis 1 = 4),
  plus item count.
- Then any backend sizes each buffer exactly; same model -> same bounds.

## Lesson 4: Operator importance is unknown

- ONNX lists hundreds of operators flat, with no signal which are common,
  critical, or niche; implementation priority was guesswork.
- Example: `ImageDecode` is pure preprocessing, not core inference, yet looks no
  different from essential ops.
- Idea 1: operator categories (core math, NN layers, preprocessing/IO, control
  flow, quantization, classic ML, contrib) so backends prioritize by category.
- Idea 2: an operator atlas of real-world usage frequency, e.g. by indexing the
  ~40k ONNX models on Hugging Face.

## Condensed single-topic alternatives

- One-slide versions of Lessons 1-3 from the 20-minute deck, for when time is
  short (use these or the expanded multi-slide versions).

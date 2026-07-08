# Talk brief

## Submitted title

**Lessons learned from building an AOT ONNX-to-C compiler**

## One-sentence summary

`emx-onnx-cgen` started as a way to gain more control over generated C code than
existing ONNX-to-C tooling provided, and became a broad-coverage AOT compiler
that exposes where ONNX needs additional assumptions for deterministic embedded
code generation.

## Target duration

20 minutes.

Suggested pacing:

- 2-3 minutes: project history and motivation
- 3 minutes: what the compiler does
- 4 minutes: architecture and test coverage
- 7 minutes: ONNX lessons learned
- 2 minutes: current status
- 1-2 minutes: community takeaways and Q&A setup

## Desired tone

Technical, candid, and community-facing.

Avoid making the talk a marketing pitch. The most valuable material is the
engineering experience: where ONNX worked well, where it became difficult, and
which constraints matter for deterministic AOT compilation.

## Core narrative

1. We first used `onnx2c`.
2. It showed the value of ONNX-to-C, but generated-code quality and control were
   not sufficient for our needs.
3. Dynamic dimensions were a central pain point.
4. We wanted to try a C representation based on VLA parameters.
5. Early results worked.
6. The project goal then expanded from proof-of-concept to broad ONNX coverage.
7. The 100% test coverage goal shaped the architecture.
8. The broader the coverage became, the more ONNX's open-ended nature became
   visible.
9. The main lessons are useful for ONNX backend authors, exporter authors, and
   test/spec maintainers.

## Key claims

- ONNX is excellent as an interchange format.
- ONNX is not always a convenient deterministic AOT compiler IR without
  additional constraints.
- AOT compilation needs boundedness:
  - tensor dimensions must be static or explicitly bounded
  - sequence lengths must have capacities
  - string sizes must have maximum lengths
- The ONNX type system should gain optional max-size bounds (per type and per
  tensor) — a general standard extension every backend reads, not an
  emx-specific annotation, replacing per-backend global caps.
- The compiler always emits explicit, typed N-dimensional C arrays — even for
  dynamic models (via C99 VLAs) — instead of flat pointers with manual indexing.
- Complete data-type support (including sub-byte, FP8/FP4, strings, sequences,
  optional) is a prerequisite for high coverage, not an optional extra.
- ONNX type and shape inference is not complete enough to be the only compiler
  analysis layer.
- Numerical accuracy validation needs clearer contracts than many official tests
  currently express.
- Operator importance is unsignalled: categories and a real-world usage atlas
  (e.g. indexing the ~40k ONNX models on Hugging Face) would guide backend
  priorities.
- C is the handoff IR (instead of MLIR); performance comes from later
  source-to-source passes (fusion, vectorization, memory-layout, DMA), e.g. RVV
  for RISC-V.
- Generated-code quality is a backend feature for embedded and safety-adjacent
  targets.

## Possible closing question

What would an AOT-friendly ONNX profile need to specify so that embedded
backends can compile models deterministically without relying on hidden
assumptions?

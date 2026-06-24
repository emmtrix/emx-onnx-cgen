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

## Slide 5: What emx-onnx-cgen does

Key points:

- Ahead-of-time ONNX-to-C compiler.
- Produces portable, deterministic C.
- Target: embedded and resource-constrained systems.
- Avoids dynamic memory allocation and external runtimes.
- Small generated API:
  - `<model_name>_load(...)`
  - `<model_name>(...)`

Speaker notes:

- Keep this concise.
- The audience needs enough project context before the lessons learned.

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

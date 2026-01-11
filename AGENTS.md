# AGENTS.md

This repository builds an **ONNX → C compiler** implemented in Python.

This document describes the working conventions for humans and automated agents (e.g., code assistants) contributing to the project.

## Project Goals

- Compile ONNX models into **portable C code** with minimal runtime dependencies.
- Produce **correct-by-comparison** outputs against ONNX Runtime (ORT).
- Keep the compiler architecture **pass-based** (import → normalize → optimize → lower → emit).
- Support reproducible builds and deterministic code generation.
- Maintain high code quality: prioritize clean OO design, avoid code smells, and continuously refactor toward simpler abstractions.

## Related Repositories / References

The directories **`emx-pytorch2c-org/`** and **`onnx2c-org/`** serve as **reference implementations / knowledge bases** for ideas, operator coverage, and practical edge cases:

- **`onnx2c-org/`** is a **C++ implementation** with similar goals (ONNX → C compilation, deterministic codegen, minimal runtime).
  *Note:* It may generate **ugly C code**. Use it for semantic reference and operator handling, not as a style template.
- **`emx-pytorch2c-org/`** is a **Python-based PyTorch → C compiler**.
  *Note:* It may contain **poor design choices / architecture smells**. Use it for concepts, codegen patterns, and testing approaches, but do **not** replicate the design blindly.

When making architectural decisions, prefer clean and maintainable design even if it diverges from these repositories.

## Repository Structure (expected)

> If folders do not exist yet, create them following these conventions.

- `src/`  
  Python package for the compiler.
- `src/onnx2c/`  
  Main package (or rename to your chosen package name).
- `src/onnx2c/ir/`  
  Internal IR definitions (graph, tensors, types, attributes).
- `src/onnx2c/passes/`  
  Normalization + optimization passes.
- `src/onnx2c/lowering/`  
  Lowering from IR to C-level IR / operator kernels.
- `src/onnx2c/codegen/`  
  C emission (templates + emit logic).
- `src/onnx2c/runtime/`  
  Minimal C runtime support (allocator, tensor struct, operator kernels).
- `templates/`  
  Jinja2 templates for C sources/headers.
- `tests/`  
  Pytest-based tests.
- `tests/models/`  
  Small ONNX models + known-good test artifacts.
- `tools/`  
  Developer scripts (benchmarking, model zoo fetch, etc.).
- `docs/`  
  Design notes, decisions, and architecture docs.
- `examples/`  
  Example models and compilation pipelines.

## Key Invariants

- **Correctness first:** any optimization must preserve semantics.
- **Deterministic output:** code generation must be stable across runs:
  - stable node ordering
  - stable symbol names
  - stable formatting
- **No hidden state:** passes should be pure functions over IR where possible.
- **Explicit errors:** unsupported ops or ambiguous shapes must raise actionable errors.

## How to Run

### Unit tests

```bash
pytest -n auto -q
```

When reporting executed tests, include the test duration in your feedback.

### Golden reference updates

Golden tests compare generated code against reference files. To refresh references
after intentional changes, set `UPDATE_REFS=1` when running tests:

```bash
UPDATE_REFS=1 pytest -n auto -q
```

### Test execution policy

Prefer running targeted single tests while working (e.g., a specific test module or
test case) to keep iteration fast. At the end of every change, run the full test
suite with reference updates enabled:

```bash
UPDATE_REFS=1 pytest -n auto -q
```

## Compiler Pipeline (conceptual)

1. **Load ONNX**

   * Validate model
   * Resolve initializers
   * Run ONNX shape inference (best-effort)

2. **Import to IR**

   * Build internal graph representation
   * Normalize attributes and types

3. **Normalize passes**

   * Constant folding
   * Canonicalize patterns
   * Eliminate no-ops / identities
   * Ensure all required shapes/types exist

4. **Optimize passes**

   * Fuse patterns (e.g., Conv+Relu)
   * Layout transforms (optional)
   * Dead code elimination

5. **Lowering**

   * Map IR nodes to C-level kernels or intrinsic ops
   * Produce explicit tensor views / memory model
   * Decide storage (stack/static/global)

6. **Code generation**

   * Emit `model.c/.h`
   * Emit kernels (or link runtime kernels)
   * Emit weights

7. **Verification**

   * Run ORT and generated code, compare numerically
   * Log max abs/rel error

## Coding Standards

### Python style

* Format: `black`
* Lint: `ruff`
* Prefer type hints (`typing`, `dataclasses`)
* Avoid global mutable state
* Keep functions small and testable

### Design & Architecture (OO)

* Prefer clear, maintainable OO design over cleverness.
* Keep responsibilities small:

  * one module = one concern
  * one class = one responsibility
  * one function = one purpose
* Use composition over inheritance unless inheritance is clearly justified.
* Prefer immutable data structures for IR objects where practical.
* Keep pass interfaces consistent:

  * every pass has a clear input/output contract
  * avoid side effects and implicit global state
* Separate concerns strictly:

  * ONNX parsing/import
  * IR + transformations
  * lowering decisions (memory model, kernels)
  * code generation (pure emission)
  * runtime kernels (C)
* Prefer explicit "context" objects over global configuration:

  * e.g. `CompilerOptions`, `Target`, `DiagnosticContext`

### Code Smells to Avoid

* Long methods/functions (split when complexity grows).
* "God" classes or modules that combine unrelated responsibilities.
* Hidden coupling via globals, singletons, or implicit module-level state.
* Deep inheritance hierarchies.
* Feature envy: methods that operate mostly on other objects’ data.
* Duplicated logic (introduce helpers or shared utilities).
* Unclear naming: prefer descriptive names over abbreviations.
* Tight coupling between passes and codegen or runtime internals.

### Error handling

* Prefer explicit custom exceptions:

  * `UnsupportedOpError`
  * `ShapeInferenceError`
  * `LoweringError`
  * `CodegenError`
* Error messages must include:

  * op type
  * node name (if any)
  * input/output shapes (if known)
  * hint to fix (e.g., “run onnxsim” or “export with static shapes”)

### Logging

* Use Python `logging` (no prints in library code).
* Default level: INFO.
* Provide debug dumps behind flags.

## Testing Guidelines

### Requirements for new features

* Every new operator/kernel requires:

  * at least one unit test
  * at least one ORT comparison test
  * edge cases (broadcasting, padding, axis, negative indices, etc.)

### Numeric tolerances

* Use tolerances appropriate for `float32`:

  * default: `atol=1e-5`, `rtol=1e-4`
* For ops with higher numeric drift (e.g., `Softmax`):

  * loosen tolerances slightly, document why

### Test model sources

* Keep test ONNX models small.
* Prefer:

  * hand-constructed graphs
  * ONNX Model Zoo models only for integration tests (optional)

## Performance Guidelines

* Avoid Python-level tensor computations in hot paths except for:

  * constant folding
  * verification
* In C runtime:

  * prefer contiguous memory layouts
  * avoid dynamic allocations unless explicitly enabled
* Any performance optimization must include a benchmark or profiling note.

## Documentation Expectations

* Any major design change needs a short note in `docs/`:

  * motivation
  * alternatives considered
  * decision
  * consequences
* Public-facing usage should be documented in `README.md`.

## Contribution Workflow

* Keep PRs focused and small.
* For any change affecting output C code:

  * include golden snapshot updates (if applicable)
  * run ORT comparison tests
* Prefer descriptive commit messages:

  * `passes: fold constants for Add/Mul`
  * `codegen: stable symbol naming`
  * `runtime: add Conv2D kernel (NHWC)`

## Agent Instructions (for automated assistants)

When acting as an agent in this repo:

1. **Do not introduce new dependencies** without a clear need.
2. **Do not change code generation output formatting** unless necessary.
3. For new ops:

   * implement kernel
   * update lowering mapping
   * add unit test + ORT comparison test
4. Maintain determinism:

   * stable ordering, stable naming
5. If unsure about semantics, prefer correctness and add a TODO + test.
6. Always propose refactorings when they improve:

   * readability
   * testability
   * separation of concerns
   * determinism
7. Detect and call out common code smells (see above) and suggest concrete improvements.
8. Prefer incremental refactorings:

   * keep PRs small and mechanical
   * preserve behavior and determinism
   * add tests before/with refactoring when needed
9. When adding new functionality, keep the design extensible:

   * avoid hardcoding operator names and special cases
   * prefer registries / dispatch tables for op lowering and kernels
10. If the architecture starts drifting, propose a short design note in `docs/` with:

* problem
* options
* recommendation

## Maintaining this document (AGENTS.md)

This file is part of the project’s contract.

Update `AGENTS.md` when:
- the repository structure changes (new top-level folders, renamed packages)
- the compiler pipeline changes (new major phases or invariants)
- coding standards or tooling decisions change (`ruff`, `black`, test strategy, etc.)
- new supported targets/constraints are added (dynamic shapes, quantization, new runtimes)
- recurring confusion appears in PR reviews (add clarifying rules)
- an agent repeatedly makes the same mistake (add an explicit guardrail)

Rules for updating:
- Keep changes small and specific.
- Include the rationale in the commit/PR description.
- Prefer adding explicit examples over vague guidelines.
- Do not add rules that are not enforced or not useful.
- If a new rule conflicts with existing rules, resolve the conflict in `AGENTS.md` as part of the same change.

Agents:
- If you encounter ambiguity not covered by this document, propose an update to `AGENTS.md`.
- If your change introduces new conventions, update this file in the same PR.

## Security / Safety

* Do not execute untrusted ONNX models from unknown sources.
* Avoid shelling out to external tools in the compilation path.
* Generated C code should not include:

  * filesystem I/O
  * network I/O
  * dynamic loading

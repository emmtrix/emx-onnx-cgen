# The emmtrix Embedded AI Compiler Workflow

The emmtrix embedded AI compiler workflow treats generated C code as more than
the final output of a model converter. In this flow, C is the central handoff
representation between the AI model frontend and the downstream embedded
optimization toolchain.

At a high level, the workflow can be summarized as:

```text
ONNX -> Clean C -> Vectorizer -> Target Architecture
```

or, in a slightly more detailed form:

```text
AI model -> frontend compiler -> analyzable C -> emmtrix optimizer/backend tools -> target code
```

The important design decision is that the compiler does not lower an AI model
directly into an opaque runtime format, target binary, or compiler-internal IR
that disappears behind the toolchain boundary. Instead, the frontend produces
standard C source code that is deliberately structured so that it can be read,
reviewed, compiled, analyzed, and transformed by later tools.

## C as the Handoff IR

In the emmtrix flow, C acts as the exposed intermediate representation. This is
similar in spirit to staged compiler flows such as MLIR-based pipelines: a
high-level model is lowered into a representation that later passes can analyze
and optimize. The difference is that the handoff artifact is not an internal
compiler dialect. It is C source code.

This choice is practical for embedded and safety-adjacent development. C is a
standardized language, broadly supported by embedded compilers, static analysis
tools, build systems, code review processes, and qualification workflows. A C
handoff also avoids locking the flow to a single compiler runtime or backend
infrastructure.

The generated C is therefore expected to be understandable and analyzable. It is
not merely "printed C" for execution. It is a disciplined subset of C plus
conventions that downstream tools can rely on.

Typical properties of this generated C include:

- explicit loop nests
- explicit tensor dimensions and strides
- predictable memory access patterns
- stable symbol names
- deterministic ordering
- simple canonical loop forms
- linear array accesses where possible
- no hidden pointer aliasing assumptions
- no dynamic dispatch
- no recursion
- no heap allocation
- minimal runtime assumptions

These properties make the C suitable as an input for source-to-source
optimization. They also make it easier to inspect, debug, verify, and integrate
into existing embedded software projects.

## Role of the AI Frontend Compiler

The frontend compiler is responsible for translating the model representation
into clean, deterministic C. In the ONNX-based flow, this means importing the
model, normalizing it, lowering operators into explicit C-level computations,
and emitting code that follows the conventions needed by the rest of the
toolchain.

The frontend's job is primarily to preserve model semantics and expose them in a
form that later tools can optimize. That means correctness and analyzability are
more important at this stage than aggressive target-specific performance
tuning.

The generated C should make the computation visible. Operators become explicit
code, tensor shapes become explicit array dimensions or parameters, and control
flow and memory behavior are represented in a way that ordinary C tooling can
understand. This is especially important for embedded targets where dynamic
allocation, runtime dispatch, and opaque framework dependencies are often
undesirable.

## Downstream Source-to-Source Optimization

Performance is added by downstream emmtrix tools that operate on the generated
C. The clean C produced by the frontend becomes the input to source-to-source
analysis and optimization passes.

Typical downstream transformations include:

- node or kernel fusion
- reduction of intermediate memory
- buffer reuse
- vectorization for SIMD or vector instruction sets
- target-specific memory-layout optimization
- offloading or DMA streaming of large weights
- emission of target-oriented C or intrinsic-based code

This separation of responsibilities is important. The frontend produces a
correct and analyzable C representation. The optimizer then uses that structure
to generate code that is better suited to a specific target architecture.

For example, a simple model containing a matrix multiplication followed by a
ReLU can first be emitted as scalar C loop nests. A downstream optimizer can
then recognize the structure, vectorize the matrix multiplication for a target
such as RISC-V with RVV, and fuse the ReLU into the vector store. In that kind
of transformation, the original loop structure remains recognizable, while the
implementation becomes target-optimized.

This only works well if the frontend emits clean and predictable C in the first
place. Poor generated C is not just hard to read; it weakens later analysis,
vectorization, memory optimization, verification, and review.

Because this downstream work is organized as C-to-C transformation steps, it can
also be treated as a sequence of explicit, bounded toolchain stages. Internally,
the emmtrix flow contains many such C transformation steps rather than one
monolithic optimizer; the current pipeline contains more than 20 individual
C-to-C transformation steps. Each step can have a defined input language subset,
output language subset, transformation contract, test strategy, and
qualification argument. This makes it possible to qualify individual C-to-C
steps instead of treating the whole AI compiler flow as one opaque tool.

That property is especially important for safety-oriented development. A
qualified or qualifiable pipeline can be assembled from smaller stages: a model
frontend that emits deterministic C, one or more qualified C-to-C
transformation steps, and a qualified target C compiler. The handoff between stages
remains source-level C, so each stage can be inspected, tested, traced, and
reasoned about using established embedded software practices.

## Why Clean C Matters

Generated-code quality is a core part of the emmtrix workflow. The generated C
is not a disposable artifact. It is an interface between tools and also a source
artifact that engineers may need to inspect.

Clean C has several advantages in this setting:

- It can be reviewed by engineers.
- It can be checked by static analysis tools.
- It can be compiled by existing embedded C/C++ compilers.
- It can be integrated into existing build systems.
- It supports traceability from model operations to generated implementation.
- It is compatible with source-level audit and qualification workflows.
- It gives downstream optimizers a stable structure to analyze.

This is particularly relevant for embedded and safety-critical environments,
where generated code may need to fit established development, review, testing,
and certification processes. In such contexts, a transparent C artifact can be
easier to justify than an opaque runtime or binary-only compilation path.

The same argument applies to tool qualification. If the AI compiler flow is
structured as a sequence of source-level C transformation steps, then each
C-to-C step can be qualified with its own scope and evidence. Those steps can
then be combined with already qualified C compilers used in safety-critical
embedded projects. The result is not just optimized code, but a pipeline whose
individual parts are amenable to qualification, traceability, and certification
arguments.

## Embedded and Safety-Oriented Constraints

The workflow is shaped by constraints common in embedded AI deployments. These
systems often need predictable memory usage, reproducible builds, and a small
runtime footprint. They may also have limited operating-system support or no
operating system at all.

For that reason, the generated code is designed to avoid hidden dynamic
behavior. In particular, the public project material emphasizes:

- no dynamic memory allocation
- no operating-system dependency
- no external AI runtime dependency
- static or explicitly bounded memory layout
- deterministic control flow
- explicit data movement
- readable and auditable source code
- correctness verification against a reference implementation

These constraints are not just implementation details. They define what kind of
compiler output is useful for the rest of the emmtrix toolchain.

## Relationship to Other Compiler Stacks

The emmtrix approach is related to broader compiler ideas, but it makes a
different trade-off than many machine-learning compiler stacks.

MLIR-centered flows often keep the key intermediate representation inside the
compiler infrastructure and lower progressively toward LLVM, a runtime, or a
target backend. That approach is powerful for compiler-internal transformations.

The emmtrix flow instead exposes C as the practical bridge between AI model
lowering and embedded optimization. This makes the handoff language less rich
than a dedicated compiler IR, but much easier to integrate with existing C
toolchains, embedded build systems, static analyzers, and safety-oriented
engineering workflows.

The choice is therefore not that C is a perfect compiler IR. It is that C is the
right visible handoff format for a workflow centered on embedded deployment,
reviewability, portability, and source-to-source optimization.

## Summary

The emmtrix embedded AI compiler workflow can be understood as a C-centered
compiler pipeline:

1. Import an AI model.
2. Lower it into deterministic, analyzable C.
3. Treat that C as the handoff IR.
4. Run downstream source-to-source optimization.
5. Generate target-oriented code for embedded architectures.

The central idea is that clean generated C is both an executable artifact and a
compiler representation. It provides the bridge between model semantics,
embedded toolchains, source-level analysis, and target-specific optimization.

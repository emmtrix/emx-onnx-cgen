# Story document: Lessons learned from building emx-onnx-cgen

This document is the long-form narrative basis for the ONNX Community Day talk.
It is intentionally more detailed than a slide deck. The second step will be to
distill this story into slides.

## Submitted title

**Lessons learned from building an AOT ONNX-to-C compiler**

## One-paragraph thesis

`emx-onnx-cgen` started from a practical embedded-code-generation problem:
existing ONNX-to-C tooling showed that the approach was useful, but did not give
enough control over generated-code quality and dynamic dimensions. What began as
an experiment around a more controlled C backend, including C99 VLA parameters
for some dynamic shapes, grew into a broad-coverage ahead-of-time compiler with
a strong testing discipline. That journey exposed a central tension: ONNX is an
excellent interchange format, but deterministic AOT compilation to static,
portable C needs additional constraints around bounded shapes, containers,
strings, type inference, and numerical accuracy.

## Intended audience

The audience is the ONNX Community Day audience: people who care about ONNX as a
standard, as an ecosystem, and as a target for runtimes and backends. The talk
should assume basic ONNX familiarity, but not detailed knowledge of
`emx-onnx-cgen`.

The most valuable contribution is not "we built another backend". The valuable
contribution is the compiler-builder perspective: what becomes hard when ONNX is
used not as a runtime interchange format, but as the input to deterministic,
statically analyzable C code generation.

## Narrative arc

The story has five acts:

1. **Origin:** why we started after using `onnx2c`.
2. **Experiment:** dynamic dimensions and the VLA-parameter idea.
3. **Expansion:** early success led to the ambition of broad ONNX coverage.
4. **Engineering response:** architecture and testing had to become systematic.
5. **Lessons for ONNX:** AOT compilation exposes gaps around bounds, inference,
   containers, and numerical accuracy.

The deck should follow this arc. It should not start with coverage numbers or a
feature matrix. The human and technical story is stronger if it starts with the
problem that forced the project into existence.

## Act 1: Why we started

The project did not start from the abstract goal of implementing a complete ONNX
backend. It started from practical experience with existing tooling.

We first used `onnx2c`. That was useful because it demonstrated that ONNX-to-C
compilation can be a practical path for embedded deployment. It also provided a
reference point: ONNX graphs can be lowered into C code, and that code can be
compiled without bringing a full runtime into the target system.

But for our use case, the generated-code quality was difficult. In embedded and
resource-constrained environments, generated code is not just an opaque artifact.
It is part of the deliverable. Engineers may need to inspect it, review it,
analyze memory use, integrate it into existing firmware, feed it into
source-to-source tooling, or reason about it in a safety-critical process.

That made control over the generated code important:

- stable symbol names
- deterministic layout
- readable loops
- explicit memory use
- small generated APIs
- predictable formatting
- no hidden runtime assumptions

The existing approach did not give us enough control over those properties.
That was the first motivation for starting `emx-onnx-cgen`.

## Act 2: Dynamic dimensions and the VLA idea

A second major motivation was dynamic-dimension support.

Dynamic dimensions are common in ONNX models. From a runtime perspective, they
are natural: a runtime can allocate buffers dynamically or dispatch kernels with
runtime shape metadata. From a static C-code-generation perspective, they are
harder. The compiler must decide what the generated function signature looks
like, where memory lives, which extents are known, and which extents are runtime
parameters.

One concrete idea was to use C99 variable-length array parameters for certain
dynamic dimensions. For example:

```c
void model(int N, int C,
           const float x[restrict N][C],
           float y[restrict N][C]);
```

This representation has attractive properties:

- the rank remains statically visible in the C type
- runtime extents are explicit function parameters
- loops can use normal C bounds
- the generated ABI remains small and readable
- the model entry point still mirrors ONNX graph inputs and outputs
- the C type carries more semantic information than a flat pointer or 1D buffer

This was not intended to solve all dynamic ONNX models. It is a representation
strategy for a useful subset: known rank, runtime extents, and target compilers
that support VLAs. But it was enough of an idea to justify experimentation.

An important background detail is why the compiler prefers N-dimensional C
arrays over flattening every tensor into a 1D buffer. Both representations can
address the same memory locations mathematically. A tensor element
`x[i][j][k]` can always be rewritten as an offset into `x_flat[...]`.

But in C, those two representations do not carry the same semantics.
N-dimensional array parameters preserve the tensor rank and per-axis extents in
the C type. A declaration such as:

```c
const float x[restrict N][C]
```

does more than pass a pointer. It describes an array-of-arrays layout where the
compiler and the reader know that the second dimension has extent `C`. This
matters because the C standard treats accesses beyond array bounds as undefined
behavior. Even if the underlying memory is contiguous, writing "through" the end
of one declared subarray into the next is not a harmless indexing trick from the
language's point of view.

With a flat 1D representation, the generated code must encode all tensor
structure manually in offset calculations. The compiler sees mostly a pointer
and a length convention. With N-dimensional arrays, some of the tensor structure
is present in the C type itself:

- rank is visible
- per-axis extents are visible
- indexing syntax mirrors tensor indexing
- bounds become part of the declared object model
- incorrect cross-boundary indexing is easier to spot in review

This does not eliminate the need for correct generated indexing, but it gives
the generated C more semantic content. For a project whose generated code is
meant to be readable, auditable, and useful for downstream tooling, that extra
semantic information is valuable.

A small example makes the difference visible. A flat representation might look
like this:

```c
void add_flat(int N, int C,
              const float *restrict a,
              const float *restrict b,
              float *restrict out) {
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            const int idx = n * C + c;
            out[idx] = a[idx] + b[idx];
        }
    }
}
```

The equivalent N-dimensional VLA representation keeps the tensor shape in the
signature and in the indexing expression:

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

Both functions can generate similar machine code after optimization. The
difference is at the C source level: the second version exposes the tensor rank
and extents directly to readers, static analysis, and downstream transformation
tools.

The important lesson is that representation is not the same as semantics. VLAs
can represent some dynamic tensor extents in C, but they do not solve:

- dynamic rank
- unbounded sequence length
- unbounded string size
- dynamic output sizes without capacities
- target compilers that do not support VLAs
- static memory budgets when runtime extents are unconstrained

So the VLA experiment was productive, but it also foreshadowed a larger theme:
AOT compilation needs bounds. ONNX often leaves those bounds open.

## Act 3: From prototype to coverage ambition

After initial successes, the project changed character.

At first, the goal was to gain control over generated C and test ideas such as
VLA-based dynamic dimensions. Once those ideas worked on initial models, the
question became: how far can this go?

That shifted the project from an experiment into a broad-coverage compiler.
Coverage became a central goal. Supporting a narrow set of operators is one
kind of engineering problem. Supporting a large part of ONNX is a different
problem.

Broad coverage means the compiler must handle more than common neural-network
building blocks. It has to face:

- tensor arithmetic
- broadcasting
- reshaping
- convolution and pooling
- recurrent networks
- attention-related operators
- quantization operators
- string operators
- sequence operators
- control flow
- dynamic-output operators
- classic ML operators
- Microsoft-contrib operators

As coverage grows, ad hoc implementation becomes less sustainable. The compiler
needs structure: a real import layer, a real internal representation, lowering
rules, reusable scalar helpers, deterministic codegen, verification tooling, and
clear unsupported-case diagnostics.

This is where the project became more than "generate some C". It became a
compiler pipeline.

## Act 4: Testing as an architectural constraint

A primary engineering goal was achieving 100% test coverage across the compiler
pipeline. This was not just a quality metric at the end. It affected the design
from the inside.

To make the compiler testable, the architecture had to become modular:

- parsing and ONNX import need clear responsibilities
- type and shape handling must be inspectable
- normalization should be separate from lowering
- lowering should validate operator-specific constraints
- code generation should be deterministic
- verification should be reproducible
- errors should be precise enough to assert in tests

The test-coverage goal pushed against code smells:

- hidden global state
- large mixed-responsibility modules
- implicit shape assumptions
- side effects in passes
- special cases embedded directly in templates
- verification paths that behave differently from compile paths

In other words, testing pressure became architecture pressure.

This is an important part of the story because it explains why the project is
pass-based and why determinism matters. Stable generated C is not just nice for
users. It is also what makes golden tests meaningful. If symbol names, node
ordering, formatting, or helper emission change nondeterministically, the test
suite stops being a reliable signal.

## Act 5: The verification loop

The compiler's verification loop is central to the project:

1. Generate C code, optionally with a testbench.
2. Compile and run the generated C.
3. Run the same ONNX model through ONNX Runtime or ONNX Reference.
4. Compare outputs.
5. Record successes, expected failures, and error categories.

This makes verification a functional part of compiler development. It is not
only "does the C compile?" It is "does the generated C behave like the ONNX
model?"

One important invariant emerged: verification-only data must not change code
generation implicitly.

It is tempting to use representative inputs from test data to resolve dynamic
shapes. But if `verify` uses those inputs to make code generation succeed, while
`compile` would fail or produce different code, then verification is no longer
testing the same compiler contract.

The project therefore treats verification I/O as verification I/O. It must not
secretly provide shape information to code generation unless the same
information is explicitly available through a compiler option that `compile`
can also use.

This matters for:

- reproducibility
- deterministic output
- honest error reporting
- meaningful coverage numbers
- user trust

The verification corpus also grew beyond small local tests and the official
ONNX backend tests. One important additional source is
`emx-ort-test-artifacts`.

The principle behind that repository is important for the talk: it turns ONNX
Runtime tests into ONNX-backend-test-like artifacts. Instead of depending on
ORT's internal C++ test harness, the relevant case is exported as a directory
with an ONNX model and protobuf input/output data. That gives another backend a
portable test case it can run with its own compiler and verification pipeline.

There is one additional detail that matters for numerical validation. ORT tests
also carry validation expectations such as per-output relative and absolute
error thresholds. In `emx-ort-test-artifacts`, these are currently stored in a
`validation.json` file next to the exported model and test data. A typical entry
contains the output name, `relative_error`, `absolute_error`, and whether output
sorting is expected before comparison.

For wider ONNX backend compatibility, this should eventually be converted into
the ONNX backend-test `data.json` convention. Conceptually, the artifact should
not only export:

```text
model.onnx
test_data_set_*
```

but also the comparison contract:

```text
validation.json today -> data.json-style tolerance metadata in the future
```

That makes the artifact more self-contained: it contains the model, the sample
inputs and expected outputs, and the intended accuracy constraints.

In this repository, that corpus is available as the
`emx-ort-test-artifacts-org/` submodule. The relevant models live under:

```text
emx-ort-test-artifacts-org/artifacts/onnxruntime
```

This corpus plays a different role from the official ONNX backend tests. The
official tests are the standards-facing baseline. They are essential, but they
are often compact operator tests. `emx-ort-test-artifacts` applies the same
basic model-plus-test-data idea to ORT's broader test world. The resulting
artifacts are closer to the cases that a major production runtime uses to
validate behavior across contrib operators, model patterns, data files, and
edge cases.

For `emx-onnx-cgen`, this became a useful second pressure test:

- Can the compiler handle more than isolated operator examples?
- Do generated C entrypoints work with real test data directories?
- Do Microsoft/ORT contrib operators expose assumptions missing from the core
  ONNX tests?
- Are failures reproducible as exact command lines?
- Can coverage be tracked and discussed as a corpus-level metric?

Each expected error entry records the command line needed to reproduce the
case, typically using:

```text
verify --model-base-dir ... model.onnx --test-data-dir test_data_set_0
```

This is important for the talk because it shows that coverage did not only mean
"we support many operator names". It meant building a reproducible verification
system around multiple corpora: local tests, official ONNX backend tests, and
ONNX-backend-test-like artifacts generated from ORT tests.

There is also a community angle here. This kind of artifact generation is not
specific to `emx-onnx-cgen`. Any backend that wants to compare itself against
ORT behavior could benefit from ORT tests exported into a portable
model-plus-test-data format. In principle, this could become a useful shared
resource for backend authors, because it lowers the barrier to running a broader
runtime-derived test corpus outside the ORT codebase.

That idea was mentioned in ORT-related forums, but the response was essentially
zero. The talk can mention this carefully: not as criticism, but as an example
of a useful testing idea that may still be underexploited by the wider ONNX/ORT
community.

## Act 6: Why C is the intermediate representation in the emmtrix flow

`emx-onnx-cgen` is not an isolated code generator. In the emmtrix toolchain, it
acts as an AI frontend compiler: ONNX is imported, normalized, lowered, and
emitted as deterministic, analyzable C. That C then becomes the input to
downstream emmtrix optimization tools such as the Vectorizer and target-specific
backend toolchains.

Public emmtrix material describes the flow as:

```text
ONNX -> Clean C -> Vectorizer -> Target Architecture
```

Internally, this is an important design choice. The compiler does not lower ONNX
directly to a target binary or to an opaque runtime representation. Instead, it
uses a disciplined subset of C plus conventions as the intermediate
representation for further analysis and transformation.

This is similar in spirit to MLIR: a frontend lowers a high-level model into a
more compiler-friendly representation, and later passes progressively optimize
and lower it. But the emmtrix choice of representation is different. Instead of
making the central handoff a compiler-internal IR, the handoff is C source code.

That has several consequences:

- The intermediate artifact is readable by humans.
- It is based on a standardized language rather than a project-specific binary
  or runtime format.
- It can be compiled by heterogeneous C/C++ compiler toolchains.
- It can fit into existing embedded build and review workflows.
- It is more approachable for safety-critical environments where C source,
  static analysis, tool qualification, and traceability already play a central
  role.

This does not mean C is a perfect IR. It is less expressive than MLIR for
compiler-internal transformations, and care is needed to keep generated C
canonical enough for analysis. But for the emmtrix ecosystem, the benefits are
practical: C is the common language understood by embedded compilers, safety
toolchains, static analyzers, and downstream source-to-source optimizers.

The generated C therefore follows conventions that make it useful as an IR:

- explicit loops
- explicit tensor dimensions
- predictable memory access patterns
- no hidden dynamic dispatch
- no heap allocation
- stable symbol names and ordering
- simple canonical forms that downstream tools can analyze

This also explains why generated-code quality was such a strong driver from the
beginning. If C is the intermediate representation for the rest of the toolchain,
then messy C is not just aesthetically bad. It weakens later analysis,
vectorization, verification, and review.

### Relation to other compiler stacks

Before starting `emx-onnx-cgen`, we looked for existing ways to generate clean C
from ONNX or machine-learning compiler stacks. Apart from `onnx2c`, there were
not many obvious solutions that matched the embedded C requirements.

Apache TVM is an open machine-learning compiler framework and has codegen
extension mechanisms such as BYOC. In principle, it can be part of a path toward
C/C++ integration, but we could not get a clean ONNX-to-generic-C flow working
for our needs at the time.

IREE is also an important ML compiler project and exposes C APIs for compiler
and runtime integration. However, its standard model is still centered around
the IREE compiler/runtime stack rather than emitting the kind of standalone,
auditable C source that we wanted as an intermediate artifact.

MLIR is open-source compiler infrastructure and is very relevant as a comparison
point. The similarity is the idea of staged lowering through compiler-friendly
representations. The difference is the artifact we choose to expose: the emmtrix
flow intentionally uses C as the handoff format, because C is directly usable in
embedded and safety-critical toolchains.

## Act 7: ONNX as interchange format versus AOT compiler IR

The broader the project became, the clearer the central mismatch became.

ONNX is very strong as an interchange format. It lets different frameworks,
runtimes, and tools communicate. That requires flexibility. Models may have
symbolic shapes, dynamic outputs, optional values, sequences, strings, subgraphs,
and runtime-dependent behavior.

But deterministic AOT C compilation wants different properties:

- tensor ranks known at compile time
- dimensions known or explicitly bounded
- sequence capacities known
- string lengths bounded
- output sizes predictable or capacity-constrained
- types fully resolved
- memory layout statically plannable
- numerical behavior sufficiently specified for validation

This does not mean ONNX is bad. It means ONNX is not, by default, a fully
constrained static compiler IR.

That distinction is one of the most important messages for the ONNX community.
The question is not whether ONNX can be used for AOT compilation. It can. The
question is what assumptions must be added to make it deterministic and
portable.

## Lesson: Boundedness matters

Several parts of ONNX are effectively unbounded from an AOT compiler
perspective.

### Dynamic tensor dimensions

If a dimension is unknown or symbolic, generated C can sometimes represent it as
a runtime extent. But memory planning still needs a policy. Is the array on the
stack? Static storage? Provided by the caller? Is there a maximum?

Without answers, the compiler cannot always generate safe, predictable C.

This is the same class of problem as strings, but for tensor extents. ONNX can
represent a dimension as symbolic or unknown, but that does not imply a maximum
size. For a runtime backend, this can be acceptable: allocate the needed buffer
when the actual shape is known. For a backend without dynamic memory, the
compiler needs a bound before it can reserve storage.

For example, a model input might conceptually be:

```text
tensor(float)[N, C]
```

If `N` is dynamic, a generated C signature can expose it as a runtime extent:

```c
void model(int N, int C,
           const float x[restrict N][C],
           float y[restrict N][C]);
```

That solves the ABI representation for a known-rank tensor, but it does not
answer how large `N` may become. If the generated model needs intermediate
buffers depending on `N`, a no-heap backend must still decide where those
buffers live and how much memory to reserve. Without a maximum, every option has
a cost:

- place VLAs on the stack and risk unbounded stack usage
- require caller-provided buffers with explicit capacities
- require a user-provided maximum dimension
- reject the model as not statically plannable
- introduce dynamic allocation, which violates the embedded target constraint

So dynamic dimensions are representable in C for some cases, but not fully
bounded. For deterministic AOT compilation, "dynamic" still needs a contract:
which dimensions are runtime values, and what are their maxima?

### Sequence length

Sequences are especially difficult for static C. A runtime can grow or manage a
sequence dynamically. A C ABI for embedded systems typically needs a capacity:

- maximum number of elements
- element shape
- per-item dynamic dimensions, if any
- count metadata

Without capacity, the compiler cannot allocate fixed storage.

This is the same class of problem as ONNX string tensors. A string tensor needs
a maximum number of characters per string element before it can be represented
as fixed C storage. A sequence needs a maximum number of contained elements
before it can be represented as fixed C storage. In both cases, ONNX describes a
variable-length value, while an embedded AOT compiler needs a bounded storage
contract.

This is subtly different from a dynamic tensor dimension. A dynamic tensor still
has one tensor object with a known rank. A sequence is a container of tensor
objects. The compiler needs to know not only the shape of an item, but also how
many items may exist and whether the items all share the same concrete shape.

In ONNX runtime-oriented implementations, this is a natural abstraction: a
sequence can be represented as a list-like object containing tensor objects.
For static C, that abstraction has to be lowered into arrays and metadata.

`emx-onnx-cgen` uses a fixed-capacity ABI for tensor sequences. Conceptually,
a sequence input becomes:

```c
const float x[EMX_SEQUENCE_MAX_LEN][H][W],
idx_t x__count
```

and a sequence output becomes:

```c
float y[EMX_SEQUENCE_MAX_LEN][H][W],
idx_t *y__count
```

The generated code defines a default capacity:

```c
#ifndef EMX_SEQUENCE_MAX_LEN
#define EMX_SEQUENCE_MAX_LEN 32
#endif
```

This makes memory layout deterministic and keeps the ABI simple. The downside
is the same as for strings: arbitrary ONNX sequence length is replaced by an
explicit bounded C policy.

Ragged sequences add another layer. If each sequence item may have a different
length or shape, max-sized element storage is not enough. The generated ABI also
needs per-item shape metadata:

```c
const float boxes[EMX_SEQUENCE_MAX_LEN][MAX_BOXES][4],
idx_t boxes__count,
const idx_t boxes__dim_0[EMX_SEQUENCE_MAX_LEN]
```

This is why `emx-onnx-cgen` requires explicit
`--sequence-element-shape` declarations for sequence inputs with unknown or
dynamic element dimensions, for example:

```bash
emx-onnx-cgen verify model.onnx \
  --sequence-element-shape boxes=[<=100,4]
```

The important lesson is that a sequence is not free in AOT C. It needs a
capacity, element storage, count metadata, and sometimes per-item dimension
metadata. ONNX gives the container abstraction, but a static backend needs the
storage contract.

### String size

String tensors also need bounds. C storage needs a maximum string length, such
as `EMX_STRING_MAX_LEN`. ONNX string tensors do not inherently provide the
fixed-size storage model that C requires.

This is a particularly concrete problem for backends without dynamic memory.
The ONNX tensor type can say "this is a tensor of strings", but it does not
state the maximum length of each string element. A runtime with heap allocation
can often store each string dynamically. A static C backend has to decide at
compile time how much storage to reserve:

```c
char tokens[batch][items][EMX_STRING_MAX_LEN];
```

Without a declared maximum, every choice is a policy decision:

- choose a fixed global maximum and risk truncation or wasted memory
- require a user-provided maximum
- reject the model as not statically compilable
- introduce dynamic allocation, which may be unacceptable for the target

For embedded AOT compilation, "string" is therefore not a fully specified
storage type unless a maximum element length is part of the model contract.

`emx-onnx-cgen` handles this by making the bound explicit in the generated C
ABI. String tensors are represented as fixed-size, null-terminated character
arrays:

```c
char x[...][EMX_STRING_MAX_LEN];
```

The generated C emits a default macro:

```c
#ifndef EMX_STRING_MAX_LEN
#define EMX_STRING_MAX_LEN 256
#endif
```

This gives the generated code a deterministic storage layout without heap
allocation. The macro can be overridden at compile time if a target or model
needs a different maximum. The trade-off is explicit: string support becomes a
bounded C storage policy, not a faithful representation of arbitrary
variable-length ONNX strings.

In the generated testbench path, strings are serialized into these fixed-size
slots. Strings that do not fit into the slot are truncated to leave room for the
terminating `'\0'`. That behavior is practical for fixed-storage C, but it also
illustrates the underlying specification gap: without a model-level maximum,
there is no universally correct bound for a static backend to choose.

### Dynamic output sizes

Operators such as `NonZero`, `Unique`, and `Compress` can produce output sizes
that depend on data values. Static C generation needs either a conservative
capacity, a shape guarantee, or a different ABI strategy.

The project response is to support useful bounded cases and fail clearly when
the information is insufficient.

## Lesson: Type and shape inference are not enough

ONNX provides shape and type inference, but practical compiler work shows that
it cannot be the only analysis layer.

Problems include:

- incomplete inference for some operators
- output shapes depending on input values
- attributes that require operator-specific interpretation
- subgraphs that need pattern recognition
- containers whose element information is insufficient for static ABI
  generation

For a compiler, missing information cannot be hand-waved. It must be resolved,
bounded, or rejected.

This pushed `emx-onnx-cgen` toward its own validation and inference logic around
the ONNX import. The compiler must understand enough operator semantics to know
whether it can generate correct C.

The larger community issue is that this logic should not live separately in
every backend. What is missing is a reliable, standalone, extensible shape and
type inference library for ONNX that handles dynamic dimensions, symbolic
dimensions, containers, and partially known shapes in a way that is useful for
compilers. Extensible is important: the library must not stop at the ONNX core
operator set. Real models and test corpora also contain external domains such
as Microsoft/ORT contrib operators.

Today, a code generator, an MLIR importer, a verifier, and a runtime-oriented
tool can all end up answering similar questions:

- What is the rank of this value?
- Which dimensions are static?
- Which dimensions are symbolic but related?
- Which dimensions are genuinely unknown?
- Which output shapes depend on input values?
- Which element type does a sequence or optional contain?
- Which facts are known, and which facts are only assumptions?

If every tool answers these questions itself, the ecosystem gets duplicated
logic and subtly different behavior. That is bad for backend authors and also
bad for model producers, because "valid enough for one tool" may not mean
"valid enough for another".

A better direction would be to make shape/type analysis a reusable artifact:

1. Run a common ONNX shape/type inference tool.
2. Compute all known shapes, dtypes, symbolic constraints, and unresolved
   facts.
3. Allow additional operator domains to register inference functions, for
   example ORT/Microsoft contrib operators.
4. Store those results back into the ONNX model, for example via enriched
   `value_info`, metadata, or dedicated annotations.
5. Let code generators, MLIR importers, validators, and verification tools
   consume the same persisted analysis results.

One possible vision is a small DSL for shape and type inference rules. Instead
of implementing every operator's inference logic separately in C++, Python,
MLIR import code, and backend-specific code generators, the rule would be
specified once in a domain-specific language.

The DSL could then have multiple consumers:

- generate or drive C++ inference code for runtime/compiler integrations
- generate or drive Python inference code for tooling and tests
- evaluate shape expressions numerically when dimensions are concrete
- evaluate shape expressions symbolically, for example through SymPy, when
  dimensions are symbolic
- expose unresolved constraints explicitly instead of silently dropping them

For custom operators, the same idea becomes even more interesting. A custom
operator domain could ship its own shape/type inference rule in this DSL, or
even store the rule inside the ONNX model as operator metadata. A backend would
not need to know that operator in advance just to propagate shapes and types. It
could load the rule, evaluate it, and persist the inferred facts.

This is still a vision, but it points at the kind of infrastructure that would
make ONNX easier to use as input for compilers: inference rules become portable
artifacts, not hand-coded fragments scattered across every backend.

This would make the inference algorithm independent from any individual code
generator. `emx-onnx-cgen` would not need to invent its own answer for every
missing shape fact, and neither would another backend. The model itself would
carry more of the compiler-relevant information.

This is a good ONNX Community Day discussion point: ONNX already standardizes
the graph format. Should the ecosystem also standardize a stronger way to
persist inferred shape/type facts, especially for dynamic models?

## Lesson: Containers complicate static C generation

Container types such as sequences are useful in ONNX, but they are not a natural
fit for a static C ABI.

A tensor maps reasonably well to a C array when rank and element type are known.
A sequence requires more design decisions:

- What is the maximum length?
- Are all elements the same shape?
- If element shapes vary, where are per-item dimensions stored?
- Are elements optional?
- Can nested containers occur?

The general ONNX type system is more flexible than the subset that a static
embedded C compiler can comfortably expose.

This leads to a broader point: AOT-friendly ONNX usage may need a constrained
profile or explicit annotations.

## Lesson: ONNX dtypes do not all map equally well to C

Another practical lesson is that ONNX's type system is broader than the set of
types that map naturally to portable C.

For many numeric types, the mapping is straightforward. Standard integer types
and common floating-point types can be represented with normal C types:

- `int8`, `int16`, `int32`, `int64`
- `uint8`, `uint16`, `uint32`, `uint64`
- `float`
- `double`
- `bool`

Those types mostly require a careful, deterministic mapping to fixed-width C
types. The more interesting cases are newer or smaller floating-point and
integer formats.

### FP16 and BF16

For `float16`, generated C can use `_Float16` where the target compiler supports
it. In practice this depends on compiler and target support, especially with GCC
and Clang. For `bfloat16`, the situation is even more target-specific; compilers
such as GCC and Clang expose support through types such as `__bf16` on supported
targets.

This is already a portability trade-off. The generated C can be explicit, but
the final behavior depends on the C compiler and target ABI.

### 2-bit and 4-bit integers

ONNX includes sub-byte integer types such as 2-bit and 4-bit integers. C does
not have traditional fixed-width `int2_t` or `int4_t` types. `emx-onnx-cgen`
therefore uses C23 `_BitInt(N)` types for these cases, for example:

```c
_BitInt(4)
unsigned _BitInt(4)
```

That gives the generated C a type-level representation of the requested value
range. But it does not automatically mean the data is packed efficiently in
memory. Depending on compiler layout and ABI rules, arrays of `_BitInt(2)` or
`_BitInt(4)` may still waste storage compared with a manually packed
representation.

So `_BitInt` is semantically useful, but it is not a complete answer for
memory-efficient sub-byte tensor storage.

### FP8 and FP4

The float8 and float4 ONNX types are even less natural for portable C. They are
not standard C arithmetic types. `emx-onnx-cgen` therefore represents them as
small integer storage, typically `uint8_t`, plus explicit conversion helpers to
and from `float`.

This makes it possible to parse, store, move, and compare those values in
generated code, but it is not the same as having native FP8 arithmetic in C. At
the moment this support is useful for coverage and demonstrating that the
compiler can carry the types through the pipeline. From a practical embedded C
performance perspective, it is less compelling unless a target-specific backend
or later source-to-source pass maps those emulated operations to real hardware
support.

The honest framing for the talk is:

- integer and common floating-point types are straightforward
- `float16` and `bfloat16` depend on compiler/target support
- 2-bit and 4-bit integers can use C23 `_BitInt`, but storage efficiency is not
  guaranteed
- FP8/FP4 are currently emulated, which is useful for coverage and completeness
  but not necessarily for efficient generated C

## Lesson: Numerical accuracy needs clearer contracts

Validation against a reference implementation is essential, but it exposes
another ambiguity: numerical accuracy.

For integer and boolean outputs, exact matching is usually straightforward.
For floating-point outputs, several questions arise:

- Which tolerance is acceptable?
- Should comparison be absolute, relative, ULP-based, or operator-specific?
- Which accumulation precision is expected?
- How should rounding be handled?
- What about math-library differences?
- What about known differences between ONNX Runtime and ONNX Reference?

Official tests often show expected behavior on selected examples, but they do
not always define a complete numerical contract. For generated C, the compiler
must still choose a policy.

`emx-onnx-cgen` uses ULP-based comparison for floating outputs after a small
absolute epsilon check, and exact comparison for non-floating outputs. But the
larger lesson is that backend authors would benefit from more explicit
operator-level numerical accuracy guidance.

There are several different tolerance models in the surrounding ecosystem:

- The official ONNX backend test loader has default tolerances of `rtol=1e-3`
  and `atol=1e-7`, with optional `data.json` overrides per test case.
- In the checked ONNX repository state, those `data.json` overrides are present
  for `real` model tests such as AlexNet, DenseNet, ResNet, VGG, and similar
  models, but not for individual `node` operator tests.
- That means the ONNX backend test data does not provide operator-specific
  accuracy tolerances through `data.json`.
- The ONNX backend runner compares floating outputs with
  `numpy.testing.assert_allclose(outputs, expected, rtol=rtol, atol=atol)`.
  That is a conventional absolute-plus-relative check:
  `abs(actual - expected) <= atol + rtol * abs(expected)`.
- ONNX Runtime's ONNX model tests use default per-sample absolute and relative
  tolerances and can override them from a test-case `config.txt` using keys such
  as `per_sample_tolerance` and `relative_per_sample_tolerance`.
- PyTorch's `torch.testing.assert_close` uses dtype-dependent default
  tolerances. For example, PyTorch documents different defaults for `float16`,
  `bfloat16`, `float32`, and `float64`.
- TensorFlow testing utilities have both a generic `assertAllClose` and
  dtype-aware variants such as `assertAllCloseAccordingToType`, again showing
  that one global tolerance is rarely ideal across dtypes.
- JAX follows the NumPy-style `allclose` model with defaults such as
  `rtol=1e-5` and `atol=1e-8`.
- MXNet's `assert_almost_equal` also combines relative and absolute thresholds,
  but its documented behavior accepts equality if either the relative or
  absolute check passes.
- NVIDIA Polygraphy exposes tolerances as comparator configuration, including
  per-output `--rtol` and `--atol`, and explicitly notes that default tolerances
  that work for FP32 may be too strict for lower precision such as FP16 or INT8.

This matters for `emx-onnx-cgen` because the compiler tries to verify generated
C across a wide range of operators and dtypes. A single fixed absolute and
relative tolerance is not always meaningful. For example, a tolerance chosen for
`float32` may be too strict or too loose for `float16`, depending on the scale
of the values and the operation.

The project therefore took a different verification path:

- First ignore very small absolute differences up to
  `atol_eps * eps(dtype)`.
- Then compute the ULP distance for remaining floating-point differences.
- Report the maximum ULP difference.
- Keep exact comparison for non-floating outputs.

This makes the tolerance scale with the evaluated floating-point type. The
default project CLI exposes this as `--atol-eps` and `--max-ulp`.

There was also a practical consequence. Some ONNX reference data appears to have
been produced with very high numerical accuracy. That can be a reasonable choice
for reference data, but it does not necessarily match the precision that a
simple generated C implementation would naturally use. Combined with the
project's strict ULP-based measurement, this forced `emx-onnx-cgen` to introduce
64-bit accumulation for selected test cases. The compiler exposes this through
`--fp32-accumulation-strategy fp64`, and several official-test expectation
entries use that option.

The conclusion for the talk should be careful:

- It may be intentional that the ONNX specification does not over-constrain
  numerical accuracy, because implementations need flexibility across hardware,
  kernels, and runtimes.
- But the official tests and operator specs often do not state enough about the
  expected numerical contract for a backend author to know what is required.
- In particular, the backend-test `data.json` mechanism should not be read as an
  operator-level accuracy specification: in the checked tree it is used for a
  small set of real-model tests, while node/operator tests rely on the default
  tolerances unless the runner is configured externally.
- For AOT code generation, this ambiguity becomes visible because every
  accumulation strategy, rounding choice, and approximation is part of the
  generated artifact.

This is a good community discussion point: should ONNX define more explicit
accuracy guidance for selected operators or dtypes, or should this remain a
backend/test-suite policy?

## Lesson: Generated-code quality is a backend feature

In many runtime-oriented systems, generated or lowered code is hidden. In this
project, generated C is the product.

Therefore generated-code quality matters:

- stable ordering
- stable naming
- simple control flow
- explicit memory layout
- no dynamic allocation
- small public API
- minimal runtime dependencies
- readable helper functions
- predictable formatting

This connects back to the original `onnx2c` experience. The project started
because generating C was not enough. The generated C needed to be suitable for
embedded engineering workflows.

## Current status to mention

The talk can include a short status section, but it should not dominate the
story.

Current repository facts:

- targets ONNX opset 26 based on ONNX 1.21.0
- supports nearly all Microsoft ONNX operators based on ONNX Runtime 1.26.0
- generated support reports track operator coverage, official ONNX backend
  model coverage, and ONNX Runtime artifact coverage
- remaining unsupported cases are documented as reproducible expected errors
- the public ONNX Backend Scoreboard lists `emx-onnx-cgen` directly after
  `ONNX Reference` in the stable-build table

These reports should be presented carefully as test-corpus/report-based
coverage, not as a universal proof that every possible model is supported. The
exact numbers are less important than the engineering practice: broad corpora,
reproducible commands, expected-error tracking, and visible progress over time.

The ONNX Backend Scoreboard is useful external context, but it also illustrates
a limitation of scoreboards. The page states that the score is based on ONNX
backend unit tests. That is a valuable signal, and being listed near the top
helps show that `emx-onnx-cgen` is a serious backend. But the scoreboard does
not measure the properties that matter most for this talk: deterministic AOT
compilation, static memory planning, generated C readability, auditability, or
fitness for embedded and safety-oriented workflows.

## Community-facing takeaways

The talk should end with constructive takeaways for ONNX.

Possible asks:

1. **AOT-friendly profiles**

   Define what must be bounded or statically known for deterministic AOT
   compilation.

2. **Better shape/type inference expectations**

   Official tests could more explicitly cover inference results, not only
   runtime output values.

3. **Container constraints**

   Sequences, strings, optional values, and dynamic outputs need clearer
   representation strategies for static backends.

4. **Numerical accuracy contracts**

   Operator specs and tests could provide clearer guidance on expected
   tolerances, accumulation precision, and edge cases.

5. **Generated-code quality as a backend concern**

   Backend scoreboards could eventually distinguish between "can execute" and
   "can produce deterministic, auditable, statically plannable code".

## Possible ending

The final message could be:

> ONNX gave us a common model language. Building `emx-onnx-cgen` showed us what
> additional information an embedded AOT backend needs to turn that language
> into deterministic C. The next step is not to make ONNX less flexible, but to
> make the constraints for static compilation more explicit.

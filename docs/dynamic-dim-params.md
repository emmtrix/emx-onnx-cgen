# Named dynamic dimensions through shape values

## Motivation

Decoder exports (Qwen2.5-0.5B, onnx-community) keep `past_sequence_length` as
a named input dimension. Compiling them with that dimension as a runtime
parameter failed at several layers:

* **Shape inference.** ONNX loses a `dim_param` as soon as it travels through
  tensor *values*: `Range(0, Shape(mask)[1], 1)`, `Slice(x, 0, Shape(x)[1])`,
  an `Expand` target guarded as `Where(shape == -1, 1, shape)`, or a fixed
  rotary table sliced to the current length. It also keeps stale `unk__N`
  dims instead of re-inferring once operands were folded to constants.
* **Lowering.** Shape values with a dynamic dim were read as the literal `-1`
  (`Range(0, -1)` of length 0, `-1` as "infer" in `Reshape`), Concat added
  `-1 + 1`, and Unsqueeze, ConstantOfShape, Range, Expand, Where rejected
  dynamic dims they can handle.
* **Code generation.** Several kernels emitted the placeholder `-1` or
  `dim_N` instead of the runtime dimension, and a `dim_param` such as
  `past_sequence_length + 1` was used verbatim as a C identifier.

## Decision

1. `symbolic_shapes.infer_dim_param_shapes` evaluates shape arithmetic with
   the graph's interface `dim_param`s as non-negative sympy symbols. An
   internal dimension that is unknown gets the interface `dim_param` whose
   expression it provably equals, or a concrete value; nothing is ever solved
   for a number. Stale internal shapes are dropped so ONNX re-infers them.
2. `--input-dim NAME<=N` declares an upper bound. `Min`/`Max` terms are
   decided by interval arithmetic over the bounded symbols. Compile and verify
   accept the same option (compile/verify parity); the generated model
   function returns early for a value above the bound.
3. In lowering, a shape value containing a dynamic dim is not a constant.
   Ops whose output shape is known by name do not need their shape operands
   (`Range` loops to the output length and never reads `limit`).
4. `codegen/dim_spelling.py` spells every `dim_param` in C: identifiers are
   parameters; expressions of parameters are derived, written as the
   parenthesized expression and not passed separately; anything else is a
   sanitized parameter (with `_dim` appended on a name collision). The model
   function returns early for a negative runtime dimension, which also keeps
   gcc from diagnosing negative VLA bounds.

## Alternatives considered

* **Pin the dimension (`--input-dim past_sequence_length=N`).** Works today
  but needs one compilation per context length.
* **Separate parameter per `dim_param` string** (`past_sequence_length` and
  `past_sequence_length_1`). Simpler, but the caller could pass inconsistent
  values that overflow the arrays.
* **Runtime `Min`/`Max` in C.** Would avoid the bound, but array extents of
  temporaries and parameters would become expressions the caller cannot see.

## Consequences

* Qwen2.5-0.5B int8 compiles once with `int past_sequence_length`; for a
  context of 7 the outputs are bit-identical to the statically pinned build.
* Each failure pattern has a minimal model in `tests/test_dim_param_shapes.py`
  (shape inference) or `tests/test_dynamic_dim_codegen.py` (lowering and
  code generation), verified against ONNX Runtime for two sizes.
* Models with runtime dimensions gain the early return for negative dims.
* Kernels that still use `CEmitterCompat.codegen_shape` emit `dim_N`
  placeholders for dynamic dims; they are converted when a model needs them.

"""Generate ONNX models that isolate genuine ai.onnx spec ambiguities.

Each case produces:
  * <name>.onnx  - a minimal model that exercises the ambiguous behaviour
  * <name>.md    - a description with the spec reference, the authoritative
                   reference output (onnx ReferenceEvaluator) and the *verified*
                   emx-onnx-cgen behaviour.

The emx-onnx-cgen behaviour quoted in the markdown was obtained empirically by
compiling each model with `python -m emx_onnx_cgen compile` and, where it
produced C, building and running the generated kernel with gcc. Those verified
results are embedded below as constants so the markdown does not depend on a C
toolchain at generation time.

Run:  PYTHONPATH=src python3 tests/spec_review/generate.py
Deterministic (fixed inputs, no RNG).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, defs, helper
from onnx.reference import ReferenceEvaluator

OUT = Path(__file__).resolve().parent


def spec_doc(op: str, version: int, domain: str = "") -> str:
    s = defs.get_schema(op, max_inclusive_version=version, domain=domain)
    return " ".join(s.doc.split())


def make_model(nodes, inputs, outputs, opset, initializers=()):
    graph = helper.make_graph(nodes, "g", inputs, outputs, initializer=list(initializers))
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])


def vi(name, dtype, shape):
    return helper.make_tensor_value_info(name, dtype, None if shape is None else list(shape))


def ref_run(model, feeds):
    return ReferenceEvaluator(model).run(None, feeds)


def fmt(a: np.ndarray) -> str:
    return np.array2string(np.asarray(a), separator=", ", max_line_width=88)


def write_md(name: str, body: str) -> None:
    (OUT / f"{name}.md").write_text(body.lstrip("\n"), encoding="utf-8")


def save(name: str, model) -> None:
    try:
        onnx.checker.check_model(model)
    except onnx.checker.ValidationError as exc:
        print(f"  [note] checker warning for {name}: {str(exc).splitlines()[0]}")
    onnx.save(model, OUT / f"{name}.onnx")


# --------------------------------------------------------------------------- #
# Case 1: Mean with > 2 inputs  -- VERIFIED CORRECT (refutes a static claim)   #
# --------------------------------------------------------------------------- #
def case_mean():
    name = "mean_three_inputs"
    opset = 13
    A, B, C = np.float32([3.0]), np.float32([9.0]), np.float32([30.0])
    node = helper.make_node("Mean", ["A", "B", "C"], ["Y"])
    model = make_model(
        [node],
        [vi("A", TensorProto.FLOAT, [1]), vi("B", TensorProto.FLOAT, [1]), vi("C", TensorProto.FLOAT, [1])],
        [vi("Y", TensorProto.FLOAT, [1])],
        opset,
    )
    save(name, model)
    (ref,) = ref_run(model, {"A": A, "B": B, "C": C})
    write_md(name, f"""
# Mean — arithmetic mean of N inputs (N > 2)

**Status:** ✅ No issue — emx-onnx-cgen is correct (regression guard).

**Operator:** `Mean` (opset {opset}) · **File:** `{name}.onnx`

## Why this case exists
A static read of `src/emx_onnx_cgen/ops.py:253` (`_mean_binary_spec`, which folds
two operands as `(a+b)*0.5`) suggested `Mean` over N>2 inputs might be computed as
a left-fold of pairwise averages (`((A+B)/2 + C)/2`), which would be wrong.
**Compiling the model refutes that** — the variadic codegen sums all inputs and
divides once by N.

## Spec
> {spec_doc("Mean", opset)[:200]}

`Mean = sum(inputs) / N`.

## Reference output (onnx ReferenceEvaluator)
```
Y = {fmt(ref)}        # (3 + 9 + 30) / 3
```

## Verified emx-onnx-cgen output
Generated kernel (`compile` of this model):
```c
output[i0] = input0[0];
output[i0] = output[i0] + input1[0];
output[i0] = output[i0] + input2[0];
output[i0] = output[i0] / 3;          // correct: divides once by N
```
→ `Y = [14.]`, matches the reference. Keep as a regression test against a future
refactor accidentally routing Mean through the pairwise binary spec.

## Reproduce
```
PYTHONPATH=src python3 -m emx_onnx_cgen compile tests/spec_review/{name}.onnx /tmp/mean.c
```
""")
    return name


# --------------------------------------------------------------------------- #
# Case 2: Split with num_outputs, uneven division  -- BUG (verified)          #
# --------------------------------------------------------------------------- #
def case_split():
    name = "split_num_outputs_uneven"
    opset = 18
    dim, n = 10, 4
    chunk = -(-dim // n)
    spec_sizes = [chunk] * (n - 1) + [dim - chunk * (n - 1)]
    base, rem = divmod(dim, n)
    repo_sizes = [base + 1] * rem + [base] * (n - rem)
    node = helper.make_node("Split", ["X"], [f"Y{i}" for i in range(n)], axis=0, num_outputs=n)
    model = make_model(
        [node],
        [vi("X", TensorProto.FLOAT, [dim])],
        [vi(f"Y{i}", TensorProto.FLOAT, [spec_sizes[i]]) for i in range(n)],
        opset,
    )
    save(name, model)
    outs = ref_run(model, {"X": np.arange(dim, dtype=np.float32)})
    ref_sizes = [int(o.shape[0]) for o in outs]
    assert ref_sizes == spec_sizes, (ref_sizes, spec_sizes)
    write_md(name, f"""
# Split — uneven division with `num_outputs`

**Status:** 🐛 Bug (verified against the compiler and onnx ReferenceEvaluator).

**Operator:** `Split` (opset {opset}) · **File:** `{name}.onnx`

## Spec
> {spec_doc("Split", opset)[:300]}

Key sentence: *"If the tensor is not evenly splittable into `num_outputs`, the
**last** chunk will be smaller."* → `chunk = ceil(dim/num_outputs)`; every chunk
is that size except the last, which takes the remainder.

## Model
`X` length {dim}, `num_outputs={n}`, axis 0. Outputs are declared with the
spec-correct sizes `{ref_sizes}`.

## Reference chunk sizes (onnx ReferenceEvaluator)
```
{ref_sizes}     # only the LAST chunk is smaller
```

## Verified emx-onnx-cgen behaviour
`compile` fails:
```
Split output shape must be (2,), got (3,)
```
i.e. the repo computes its own split sizes as `{repo_sizes}` and rejects the
spec-correct `{ref_sizes}`. Source: `src/emx_onnx_cgen/lowering/split.py:155`
```python
split_sizes = [base + 1] * remainder + [base] * (num_outputs - remainder)
# dim={dim}, n={n}  ->  {repo_sizes}   (spec wants {ref_sizes})
```
It distributes the remainder over the **first** chunks instead of putting the
remainder in the last chunk. The two agree only when `remainder <= 1` (e.g.
dim=7/n=4, dim=11/n=4), which is why it went unnoticed.

## Reproduce
```
PYTHONPATH=src python3 -m emx_onnx_cgen compile tests/spec_review/{name}.onnx /tmp/split.c
```
""")
    return name


# --------------------------------------------------------------------------- #
# Case 3: GroupNormalization opset 18 (per-group scale/bias)  -- BUG (verified)#
# --------------------------------------------------------------------------- #
def case_groupnorm():
    name = "groupnorm_opset18_per_group"
    opset = 18
    C, num_groups = 4, 2
    node = helper.make_node(
        "GroupNormalization", ["X", "scale", "bias"], ["Y"], num_groups=num_groups, epsilon=1e-5
    )
    model = make_model(
        [node],
        [
            vi("X", TensorProto.FLOAT, [1, C, 2]),
            vi("scale", TensorProto.FLOAT, [num_groups]),
            vi("bias", TensorProto.FLOAT, [num_groups]),
        ],
        [vi("Y", TensorProto.FLOAT, [1, C, 2])],
        opset,
    )
    save(name, model)
    X = np.arange(C * 2, dtype=np.float32).reshape(1, C, 2)
    feeds = {"X": X, "scale": np.float32([1.0, 2.0]), "bias": np.float32([0.0, 1.0])}
    (ref,) = ref_run(model, feeds)
    write_md(name, f"""
# GroupNormalization — scale/bias shape at opset 18 vs 21

**Status:** 🐛 Bug for opset 18 (verified): the two opsets have opposite semantics.

**Operator:** `GroupNormalization` (opset {opset}) · **File:** `{name}.onnx`

## Spec — semantics flipped between opsets
- **opset 18:** *"`scale` and `bias` should be specified for each **group** of
  channels"* → shape `(num_groups,)`.
- **opset 21:** *"`scale` and `bias` should be specified for each **channel**"*
  → shape `(C,)`.

The onnx ReferenceEvaluator confirms this: at opset 18 it accepts `scale` of
shape `(num_groups,)` and rejects `(C,)`. ONNX later **deprecated** the opset-18
form precisely because of this confusion (the model checker emits a deprecation
warning for it — expected).

## Model
`C={C}`, `num_groups={num_groups}` → `scale`/`bias` shape `({num_groups},)`
(the valid opset-18 per-group form).

## Reference output (onnx ReferenceEvaluator, opset 18)
```
Y = {fmt(ref)}
```

## Verified emx-onnx-cgen behaviour
`compile` fails:
```
GroupNormalization scale and bias must be 1D with length C
```
`src/emx_onnx_cgen/lowering/group_normalization.py:38-41` hard-codes the shape
to `(C,)` with **no opset gating** — correct for opset 21 but wrong for opset 18,
where a valid per-group `scale`/`bias` is rejected.

## Reproduce
```
PYTHONPATH=src python3 -m emx_onnx_cgen compile tests/spec_review/{name}.onnx /tmp/gn.c
```
""")
    return name


# --------------------------------------------------------------------------- #
# Case 4: Max NaN handling  -- AMBIGUITY (verified divergence)                 #
# --------------------------------------------------------------------------- #
def case_minmax_nan():
    name = "max_nan_propagation"
    opset = 13
    A = np.float32([np.nan, 1.0, np.nan])
    B = np.float32([1.0, np.nan, np.nan])
    node = helper.make_node("Max", ["A", "B"], ["Y"])
    model = make_model(
        [node],
        [vi("A", TensorProto.FLOAT, [3]), vi("B", TensorProto.FLOAT, [3])],
        [vi("Y", TensorProto.FLOAT, [3])],
        opset,
    )
    save(name, model)
    (ref,) = ref_run(model, {"A": A, "B": B})
    write_md(name, f"""
# Max / Min — NaN handling

**Status:** ⚠️ Ambiguity (spec doc is silent on NaN) — verified divergence.

**Operator:** `Max` (opset {opset}) · **File:** `{name}.onnx`

## Spec
> {spec_doc("Max", opset)[:180]}

Nothing is said about NaN. `np.maximum`/`np.minimum` **propagate** NaN; C
`fmax`/`fmin` are **NaN-quieting** (return the non-NaN operand). Both are
defensible readings.

## Model
`Max(A, B)` with `A = [nan, 1, nan]`, `B = [1, nan, nan]`.

## Reference output (onnx ReferenceEvaluator — propagates NaN)
```
Y = {fmt(ref)}
```

## Verified emx-onnx-cgen output
Floating Min/Max lower to C `fmaxf`/`fminf`
(`src/shared/scalar_functions.py:644-645`). Compiling and running the generated
kernel gives:
```
Y = [1, 1, nan]      # fmax(nan,1)=1, fmax(1,nan)=1, fmax(nan,nan)=nan
```
So where exactly one operand is NaN, emx-onnx-cgen yields the finite value while
the reference yields NaN. Spec-defensible, but diverges from ORT/numpy — a
verification-mismatch source.

## Reproduce
```
PYTHONPATH=src python3 -m emx_onnx_cgen compile tests/spec_review/{name}.onnx /tmp/max.c
```
""")
    return name


# --------------------------------------------------------------------------- #
# Case 5: ScatterElements duplicate indices  -- AMBIGUITY (repo matches ref)   #
# --------------------------------------------------------------------------- #
def case_scatter_dup():
    name = "scatter_elements_duplicate_indices"
    opset = 18
    data = np.zeros((1, 5), dtype=np.float32)
    indices = np.array([[2, 2, 2]], dtype=np.int64)
    updates = np.array([[10.0, 20.0, 30.0]], dtype=np.float32)
    node = helper.make_node("ScatterElements", ["data", "indices", "updates"], ["Y"], axis=1)
    model = make_model(
        [node],
        [
            vi("data", TensorProto.FLOAT, [1, 5]),
            vi("indices", TensorProto.INT64, [1, 3]),
            vi("updates", TensorProto.FLOAT, [1, 3]),
        ],
        [vi("Y", TensorProto.FLOAT, [1, 5])],
        opset,
    )
    save(name, model)
    (ref,) = ref_run(model, {"data": data, "indices": indices, "updates": updates})
    write_md(name, f"""
# ScatterElements — duplicate indices with `reduction="none"`

**Status:** ⚠️ Ambiguity (spec leaves order undefined) — repo matches the
reference here, but parity is not guaranteed.

**Operator:** `ScatterElements` (opset {opset}) · **File:** `{name}.onnx`

## Spec
With `reduction="none"`, when several updates target the **same** output element
the result is explicitly **non-deterministic** (order undefined). Same for
`ScatterND`.

## Model
All three indices point at column 2: `indices = {indices.tolist()}`,
`updates = {updates.tolist()[0]}`.

## One valid reference result (onnx ReferenceEvaluator)
```
Y = {fmt(ref)}      # last write wins -> column 2 = 30
```

## Verified emx-onnx-cgen output
The kernel applies updates in row-major order of `updates`
(`src/emx_onnx_cgen/templates/scatter_elements_op.c.j2`), i.e. deterministic
last-write-wins. Compiled and run:
```
Y = [0, 0, 30, 0, 0]    # matches the reference for THIS input
```
Both pick last-write here, but since the spec permits any order this can diverge
from ORT on other duplicate-index inputs — a latent verification-mismatch.

## Reproduce
```
PYTHONPATH=src python3 -m emx_onnx_cgen compile tests/spec_review/{name}.onnx /tmp/scatter.c
```
""")
    return name


# --------------------------------------------------------------------------- #
# Case 6: Hardmax legacy 2D coercion  -- SPEC-vs-TOOLING ambiguity (verified)  #
# --------------------------------------------------------------------------- #
def case_hardmax():
    name = "hardmax_legacy_coercion"
    opset = 11
    axis = 1
    X = np.array([[[1, 5, 2], [4, 3, 0]], [[0, 0, 9], [1, 1, 1]]], dtype=np.float32)
    node = helper.make_node("Hardmax", ["X"], ["Y"], axis=axis)
    model = make_model(
        [node], [vi("X", TensorProto.FLOAT, list(X.shape))], [vi("Y", TensorProto.FLOAT, list(X.shape))], opset
    )
    save(name, model)
    (ref,) = ref_run(model, {"X": X})

    rows = int(np.prod(X.shape[:axis]))
    coerced = X.reshape(rows, -1)
    spec_coerced = np.zeros_like(coerced)
    spec_coerced[np.arange(rows), coerced.argmax(axis=1)] = 1.0
    spec_coerced = spec_coerced.reshape(X.shape)

    write_md(name, f"""
# Hardmax — legacy 2D coercion (opset < 13)

**Status:** ⚠️ Spec-vs-tooling ambiguity — even onnx's own reference diverges
from the spec doc; emx-onnx-cgen matches the reference, not the doc.

**Operator:** `Hardmax` (opset {opset}, axis={axis}) · **File:** `{name}.onnx`

## Spec
- **opset 1/11 (doc):** the input is *coerced into a 2-D tensor* by flattening
  dims `[axis:]`, and Hardmax picks the first maximum **per coerced row**.
  Default axis = 1.
- **opset 13+:** axis is the single reduced axis (no coercion). Default axis = -1.

Softmax/LogSoftmax have the identical opset-11→13 change; emx-onnx-cgen handles
it for those via `use_legacy_axis_semantics`, but Hardmax always uses the
single-axis form (`templates/hardmax_op.c.j2`, generated kernel uses
`axis_size = shape[axis]`).

## Model
`X` shape `{list(X.shape)}`, `axis={axis}`.

## Three "authorities" disagree
Spec-doc legacy 2D coercion (one max per coerced row):
```
{fmt(spec_coerced)}
```
onnx ReferenceEvaluator (applies single-axis semantics even at opset {opset}):
```
{fmt(ref)}
```
Verified emx-onnx-cgen output (compiled + run): **matches the reference**
```
{fmt(ref)}
```

Because ONNX's own reference does not reproduce the documented legacy coercion,
the "correct" value is contested. emx-onnx-cgen agrees with the onnx reference.
Confirm against ORT before treating this as a bug; if ORT does coerce, the fix
mirrors the existing Softmax `use_legacy_axis_semantics` path.

## Reproduce
```
PYTHONPATH=src python3 -m emx_onnx_cgen compile tests/spec_review/{name}.onnx /tmp/hardmax.c
```
""")
    return name


def write_index(names):
    rows = {
        "mean_three_inputs": "✅ Correct (regression guard; refutes a static claim)",
        "split_num_outputs_uneven": "🐛 Bug — remainder distributed to first chunks, not last",
        "groupnorm_opset18_per_group": "🐛 Bug — opset-18 per-group scale/bias rejected",
        "max_nan_propagation": "⚠️ Ambiguity — NaN-quieting vs reference NaN-propagation",
        "scatter_elements_duplicate_indices": "⚠️ Ambiguity — duplicate-index order (matches ref here)",
        "hardmax_legacy_coercion": "⚠️ Spec-vs-tooling — legacy coercion not applied (matches ref)",
    }
    lines = [
        "# ONNX spec-ambiguity test models",
        "",
        "Minimal models isolating genuine ai.onnx spec ambiguities, each with a",
        "`.md` describing the spec reference, the authoritative onnx-reference output,",
        "and the **verified** emx-onnx-cgen behaviour (obtained by compiling each model",
        "and running the generated C). Regenerate with:",
        "",
        "```",
        "PYTHONPATH=src python3 tests/spec_review/generate.py",
        "```",
        "",
        "Kept out of `tests/onnx/` on purpose so the auto-verification harness does not",
        "treat these intentional divergences as failures.",
        "",
        "| Case | Status |",
        "| --- | --- |",
    ]
    for n in names:
        lines.append(f"| [`{n}`]({n}.md) | {rows.get(n, '')} |")
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    cases = [case_mean, case_split, case_groupnorm, case_minmax_nan, case_scatter_dup, case_hardmax]
    names = [c() for c in cases]
    write_index(names)
    print(f"Generated {len(names)} cases + README in {OUT}:")
    for n in names:
        print(f"  - {n}.onnx + {n}.md")


if __name__ == "__main__":
    main()

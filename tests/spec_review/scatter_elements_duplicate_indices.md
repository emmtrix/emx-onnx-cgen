# ScatterElements — duplicate indices with `reduction="none"`

**Status:** ⚠️ Ambiguity (spec leaves order undefined) — repo matches the
reference here, but parity is not guaranteed.

**Operator:** `ScatterElements` (opset 18) · **File:** `scatter_elements_duplicate_indices.onnx`

## Spec
With `reduction="none"`, when several updates target the **same** output element
the result is explicitly **non-deterministic** (order undefined). Same for
`ScatterND`.

## Model
All three indices point at column 2: `indices = [[2, 2, 2]]`,
`updates = [10.0, 20.0, 30.0]`.

## One valid reference result (onnx ReferenceEvaluator)
```
Y = [[ 0.,  0., 30.,  0.,  0.]]      # last write wins -> column 2 = 30
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
PYTHONPATH=src python3 -m emx_onnx_cgen compile tests/spec_review/scatter_elements_duplicate_indices.onnx /tmp/scatter.c
```

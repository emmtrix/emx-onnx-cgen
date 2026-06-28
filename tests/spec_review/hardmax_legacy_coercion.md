# Hardmax — legacy 2D coercion (opset < 13)

**Status:** ⚠️ Spec-vs-tooling ambiguity — even onnx's own reference diverges
from the spec doc; emx-onnx-cgen matches the reference, not the doc.

**Operator:** `Hardmax` (opset 11, axis=1) · **File:** `hardmax_legacy_coercion.onnx`

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
`X` shape `[2, 2, 3]`, `axis=1`.

## Three "authorities" disagree
Spec-doc legacy 2D coercion (one max per coerced row):
```
[[[0., 1., 0.],
  [0., 0., 0.]],

 [[0., 0., 1.],
  [0., 0., 0.]]]
```
onnx ReferenceEvaluator (applies single-axis semantics even at opset 11):
```
[[[0., 1., 1.],
  [1., 0., 0.]],

 [[0., 0., 1.],
  [1., 1., 0.]]]
```
Verified emx-onnx-cgen output (compiled + run): **matches the reference**
```
[[[0., 1., 1.],
  [1., 0., 0.]],

 [[0., 0., 1.],
  [1., 1., 0.]]]
```

Because ONNX's own reference does not reproduce the documented legacy coercion,
the "correct" value is contested. emx-onnx-cgen agrees with the onnx reference.
Confirm against ORT before treating this as a bug; if ORT does coerce, the fix
mirrors the existing Softmax `use_legacy_axis_semantics` path.

## Reproduce
```
PYTHONPATH=src python3 -m emx_onnx_cgen compile tests/spec_review/hardmax_legacy_coercion.onnx /tmp/hardmax.c
```

# Split — uneven division with `num_outputs`

**Status:** 🐛 Bug (verified against the compiler and onnx ReferenceEvaluator).

**Operator:** `Split` (opset 18) · **File:** `split_num_outputs_uneven.onnx`

## Spec
> Split a tensor into a list of tensors, along the specified 'axis'. Either input 'split' or the attribute 'num_outputs' should be specified, but not both. If the attribute 'num_outputs' is specified, then the tensor is split into equal sized parts. If the tensor is not evenly splittable into `num_out

Key sentence: *"If the tensor is not evenly splittable into `num_outputs`, the
**last** chunk will be smaller."* → `chunk = ceil(dim/num_outputs)`; every chunk
is that size except the last, which takes the remainder.

## Model
`X` length 10, `num_outputs=4`, axis 0. Outputs are declared with the
spec-correct sizes `[3, 3, 3, 1]`.

## Reference chunk sizes (onnx ReferenceEvaluator)
```
[3, 3, 3, 1]     # only the LAST chunk is smaller
```

## Verified emx-onnx-cgen behaviour
`compile` fails:
```
Split output shape must be (2,), got (3,)
```
i.e. the repo computes its own split sizes as `[3, 3, 2, 2]` and rejects the
spec-correct `[3, 3, 3, 1]`. Source: `src/emx_onnx_cgen/lowering/split.py:155`
```python
split_sizes = [base + 1] * remainder + [base] * (num_outputs - remainder)
# dim=10, n=4  ->  [3, 3, 2, 2]   (spec wants [3, 3, 3, 1])
```
It distributes the remainder over the **first** chunks instead of putting the
remainder in the last chunk. The two agree only when `remainder <= 1` (e.g.
dim=7/n=4, dim=11/n=4), which is why it went unnoticed.

## Reproduce
```
PYTHONPATH=src python3 -m emx_onnx_cgen compile tests/spec_review/split_num_outputs_uneven.onnx /tmp/split.c
```

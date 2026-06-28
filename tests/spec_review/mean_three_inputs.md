# Mean — arithmetic mean of N inputs (N > 2)

**Status:** ✅ No issue — emx-onnx-cgen is correct (regression guard).

**Operator:** `Mean` (opset 13) · **File:** `mean_three_inputs.onnx`

## Why this case exists
A static read of `src/emx_onnx_cgen/ops.py:253` (`_mean_binary_spec`, which folds
two operands as `(a+b)*0.5`) suggested `Mean` over N>2 inputs might be computed as
a left-fold of pairwise averages (`((A+B)/2 + C)/2`), which would be wrong.
**Compiling the model refutes that** — the variadic codegen sums all inputs and
divides once by N.

## Spec
> Element-wise mean of each of the input tensors (with Numpy-style broadcasting support). All inputs and outputs must have the same data type. This operator supports **multidirectional (i.e., Numpy-styl

`Mean = sum(inputs) / N`.

## Reference output (onnx ReferenceEvaluator)
```
Y = [14.]        # (3 + 9 + 30) / 3
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
PYTHONPATH=src python3 -m emx_onnx_cgen compile tests/spec_review/mean_three_inputs.onnx /tmp/mean.c
```

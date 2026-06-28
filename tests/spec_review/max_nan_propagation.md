# Max / Min — NaN handling

**Status:** ⚠️ Ambiguity (spec doc is silent on NaN) — verified divergence.

**Operator:** `Max` (opset 13) · **File:** `max_nan_propagation.onnx`

## Spec
> Element-wise max of each of the input tensors (with Numpy-style broadcasting support). All inputs and outputs must have the same data type. This operator supports **multidirectiona

Nothing is said about NaN. `np.maximum`/`np.minimum` **propagate** NaN; C
`fmax`/`fmin` are **NaN-quieting** (return the non-NaN operand). Both are
defensible readings.

## Model
`Max(A, B)` with `A = [nan, 1, nan]`, `B = [1, nan, nan]`.

## Reference output (onnx ReferenceEvaluator — propagates NaN)
```
Y = [nan, nan, nan]
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
PYTHONPATH=src python3 -m emx_onnx_cgen compile tests/spec_review/max_nan_propagation.onnx /tmp/max.c
```

# GroupNormalization — scale/bias shape at opset 18 vs 21

**Status:** 🐛 Bug for opset 18 (verified): the two opsets have opposite semantics.

**Operator:** `GroupNormalization` (opset 18) · **File:** `groupnorm_opset18_per_group.onnx`

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
`C=4`, `num_groups=2` → `scale`/`bias` shape `(2,)`
(the valid opset-18 per-group form).

## Reference output (onnx ReferenceEvaluator, opset 18)
```
Y = [[[-1.3416355 , -0.44721183],
  [ 0.44721183,  1.3416355 ],
  [-1.6832709 ,  0.10557634],
  [ 1.8944237 ,  3.683271  ]]]
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
PYTHONPATH=src python3 -m emx_onnx_cgen compile tests/spec_review/groupnorm_opset18_per_group.onnx /tmp/gn.c
```

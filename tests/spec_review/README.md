# ONNX spec-ambiguity test models

Minimal models isolating genuine ai.onnx spec ambiguities, each with a
`.md` describing the spec reference, the authoritative onnx-reference output,
and the **verified** emx-onnx-cgen behaviour (obtained by compiling each model
and running the generated C). Regenerate with:

```
PYTHONPATH=src python3 tests/spec_review/generate.py
```

Kept out of `tests/onnx/` on purpose so the auto-verification harness does not
treat these intentional divergences as failures.

| Case | Status |
| --- | --- |
| [`mean_three_inputs`](mean_three_inputs.md) | ✅ Correct (regression guard; refutes a static claim) |
| [`split_num_outputs_uneven`](split_num_outputs_uneven.md) | 🐛 Bug — remainder distributed to first chunks, not last |
| [`groupnorm_opset18_per_group`](groupnorm_opset18_per_group.md) | 🐛 Bug — opset-18 per-group scale/bias rejected |
| [`max_nan_propagation`](max_nan_propagation.md) | ⚠️ Ambiguity — NaN-quieting vs reference NaN-propagation |
| [`scatter_elements_duplicate_indices`](scatter_elements_duplicate_indices.md) | ⚠️ Ambiguity — duplicate-index order (matches ref here) |
| [`hardmax_legacy_coercion`](hardmax_legacy_coercion.md) | ⚠️ Spec-vs-tooling — legacy coercion not applied (matches ref) |

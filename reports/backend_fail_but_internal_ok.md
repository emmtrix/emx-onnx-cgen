# Backend Fail But Internal OK

This list contains ONNX backend tests that failed in the full backend run even though the corresponding internal ONNX file test was already marked `OK ...` in `tests/expected_errors`.

## Open Tests

| Status | Test | Reason |
| --- | --- | --- |
| [ ] | `src/emx_onnx_cgen/onnx_backend/test_suite.py::OnnxBackendNodeModelTest::test_affine_grid_2d_align_corners_expanded_cpu` | [R1](#r1-affine_grid-2d-output-corruption) |
| [ ] | `src/emx_onnx_cgen/onnx_backend/test_suite.py::OnnxBackendNodeModelTest::test_affine_grid_2d_expanded_cpu` | [R1](#r1-affine_grid-2d-output-corruption) |
| [ ] | `src/emx_onnx_cgen/onnx_backend/test_suite.py::OnnxBackendNodeModelTest::test_affine_grid_3d_align_corners_expanded_cpu` | [R2](#r2-affine_grid-3d-lowering-failure) |
| [ ] | `src/emx_onnx_cgen/onnx_backend/test_suite.py::OnnxBackendNodeModelTest::test_affine_grid_3d_expanded_cpu` | [R2](#r2-affine_grid-3d-lowering-failure) |
| [ ] | `src/emx_onnx_cgen/onnx_backend/test_suite.py::OnnxBackendNodeModelTest::test_attention_3d_causal_expanded_cpu` | [R3](#r3-attention-operator-resultprecision-mismatch) |
| [ ] | `src/emx_onnx_cgen/onnx_backend/test_suite.py::OnnxBackendNodeModelTest::test_attention_3d_diff_heads_sizes_causal_expanded_cpu` | [R3](#r3-attention-operator-resultprecision-mismatch) |
| [ ] | `src/emx_onnx_cgen/onnx_backend/test_suite.py::OnnxBackendNodeModelTest::test_attention_3d_gqa_causal_expanded_cpu` | [R3](#r3-attention-operator-resultprecision-mismatch) |
| [ ] | `src/emx_onnx_cgen/onnx_backend/test_suite.py::OnnxBackendNodeModelTest::test_attention_4d_attn_mask_3d_causal_expanded_cpu` | [R3](#r3-attention-operator-resultprecision-mismatch) |
| [ ] | `src/emx_onnx_cgen/onnx_backend/test_suite.py::OnnxBackendNodeModelTest::test_attention_4d_attn_mask_4d_causal_expanded_cpu` | [R3](#r3-attention-operator-resultprecision-mismatch) |
| [ ] | `src/emx_onnx_cgen/onnx_backend/test_suite.py::OnnxBackendNodeModelTest::test_attention_4d_causal_expanded_cpu` | [R3](#r3-attention-operator-resultprecision-mismatch) |
| [ ] | `src/emx_onnx_cgen/onnx_backend/test_suite.py::OnnxBackendNodeModelTest::test_attention_4d_diff_heads_sizes_causal_expanded_cpu` | [R3](#r3-attention-operator-resultprecision-mismatch) |
| [ ] | `src/emx_onnx_cgen/onnx_backend/test_suite.py::OnnxBackendNodeModelTest::test_attention_4d_fp16_cpu` | [R3](#r3-attention-operator-resultprecision-mismatch) |
| [ ] | `src/emx_onnx_cgen/onnx_backend/test_suite.py::OnnxBackendNodeModelTest::test_attention_4d_gqa_causal_expanded_cpu` | [R3](#r3-attention-operator-resultprecision-mismatch) |
| [ ] | `src/emx_onnx_cgen/onnx_backend/test_suite.py::OnnxBackendNodeModelTest::test_attention_4d_gqa_with_past_and_present_fp16_cpu` | [R3](#r3-attention-operator-resultprecision-mismatch) |
| [ ] | `src/emx_onnx_cgen/onnx_backend/test_suite.py::OnnxBackendNodeModelTest::test_attention_4d_gqa_with_past_and_present_fp16_expanded_cpu` | [R3](#r3-attention-operator-resultprecision-mismatch) |
| [ ] | `src/emx_onnx_cgen/onnx_backend/test_suite.py::OnnxBackendNodeModelTest::test_attention_4d_with_past_and_present_qk_matmul_bias_3d_mask_causal_expanded_cpu` | [R3](#r3-attention-operator-resultprecision-mismatch) |
| [ ] | `src/emx_onnx_cgen/onnx_backend/test_suite.py::OnnxBackendNodeModelTest::test_attention_4d_with_past_and_present_qk_matmul_bias_4d_mask_causal_expanded_cpu` | [R3](#r3-attention-operator-resultprecision-mismatch) |
| [ ] | `src/emx_onnx_cgen/onnx_backend/test_suite.py::OnnxBackendNodeModelTest::test_bernoulli_cpu` | [R4](#r4-bernoulli-nondeterministic-output) |
| [ ] | `src/emx_onnx_cgen/onnx_backend/test_suite.py::OnnxBackendNodeModelTest::test_bernoulli_double_cpu` | [R4](#r4-bernoulli-nondeterministic-output) |
| [ ] | `src/emx_onnx_cgen/onnx_backend/test_suite.py::OnnxBackendNodeModelTest::test_bernoulli_double_expanded_cpu` | [R4](#r4-bernoulli-nondeterministic-output) |
| [ ] | `src/emx_onnx_cgen/onnx_backend/test_suite.py::OnnxBackendNodeModelTest::test_bernoulli_expanded_cpu` | [R4](#r4-bernoulli-nondeterministic-output) |
| [ ] | `src/emx_onnx_cgen/onnx_backend/test_suite.py::OnnxBackendNodeModelTest::test_bernoulli_seed_cpu` | [R4](#r4-bernoulli-nondeterministic-output) |
| [ ] | `src/emx_onnx_cgen/onnx_backend/test_suite.py::OnnxBackendNodeModelTest::test_bernoulli_seed_expanded_cpu` | [R4](#r4-bernoulli-nondeterministic-output) |
| [ ] | `src/emx_onnx_cgen/onnx_backend/test_suite.py::OnnxBackendNodeModelTest::test_blackmanwindow_expanded_cpu` | [R5](#r5-window-ops-lowering-failure) |
| [ ] | `src/emx_onnx_cgen/onnx_backend/test_suite.py::OnnxBackendNodeModelTest::test_blackmanwindow_symmetric_expanded_cpu` | [R5](#r5-window-ops-lowering-failure) |
| [ ] | `src/emx_onnx_cgen/onnx_backend/test_suite.py::OnnxBackendNodeModelTest::test_center_crop_pad_crop_and_pad_expanded_cpu` | [R6](#r6-centercroppad-kernel-result-wrong) |
| [ ] | `src/emx_onnx_cgen/onnx_backend/test_suite.py::OnnxBackendNodeModelTest::test_center_crop_pad_crop_axes_chw_expanded_cpu` | [R6](#r6-centercroppad-kernel-result-wrong) |
| [ ] | `src/emx_onnx_cgen/onnx_backend/test_suite.py::OnnxBackendNodeModelTest::test_center_crop_pad_crop_axes_hwc_expanded_cpu` | [R6](#r6-centercroppad-kernel-result-wrong) |
| [ ] | `src/emx_onnx_cgen/onnx_backend/test_suite.py::OnnxBackendNodeModelTest::test_center_crop_pad_crop_expanded_cpu` | [R6](#r6-centercroppad-kernel-result-wrong) |
| [ ] | `src/emx_onnx_cgen/onnx_backend/test_suite.py::OnnxBackendNodeModelTest::test_center_crop_pad_crop_negative_axes_hwc_expanded_cpu` | [R6](#r6-centercroppad-kernel-result-wrong) |
| [ ] | `src/emx_onnx_cgen/onnx_backend/test_suite.py::OnnxBackendNodeModelTest::test_center_crop_pad_pad_expanded_cpu` | [R6](#r6-centercroppad-kernel-result-wrong) |
| [ ] | `src/emx_onnx_cgen/onnx_backend/test_suite.py::OnnxBackendNodeModelTest::test_dft_inverse_cpu` | [R7](#r7-dft-inverse-numeric-tolerance) |
| [ ] | `src/emx_onnx_cgen/onnx_backend/test_suite.py::OnnxBackendNodeModelTest::test_dft_inverse_opset19_cpu` | [R7](#r7-dft-inverse-numeric-tolerance) |
| [ ] | `src/emx_onnx_cgen/onnx_backend/test_suite.py::OnnxBackendNodeModelTest::test_group_normalization_epsilon_expanded_cpu` | [R8](#r8-groupnormalization-produces-nans) |
| [ ] | `src/emx_onnx_cgen/onnx_backend/test_suite.py::OnnxBackendNodeModelTest::test_group_normalization_example_expanded_cpu` | [R8](#r8-groupnormalization-produces-nans) |
| [ ] | `src/emx_onnx_cgen/onnx_backend/test_suite.py::OnnxBackendNodeModelTest::test_hammingwindow_expanded_cpu` | [R5](#r5-window-ops-lowering-failure) |
| [ ] | `src/emx_onnx_cgen/onnx_backend/test_suite.py::OnnxBackendNodeModelTest::test_hammingwindow_symmetric_expanded_cpu` | [R5](#r5-window-ops-lowering-failure) |
| [ ] | `src/emx_onnx_cgen/onnx_backend/test_suite.py::OnnxBackendNodeModelTest::test_hannwindow_expanded_cpu` | [R5](#r5-window-ops-lowering-failure) |
| [ ] | `src/emx_onnx_cgen/onnx_backend/test_suite.py::OnnxBackendNodeModelTest::test_hannwindow_symmetric_expanded_cpu` | [R5](#r5-window-ops-lowering-failure) |
| [ ] | `src/emx_onnx_cgen/onnx_backend/test_suite.py::OnnxBackendNodeModelTest::test_loop13_seq_cpu` | [R12](#r12-sequence-element-shape-missing-in-model) |
| [ ] | `src/emx_onnx_cgen/onnx_backend/test_suite.py::OnnxBackendNodeModelTest::test_loop16_seq_none_cpu` | [R12](#r12-sequence-element-shape-missing-in-model) |
| [ ] | `src/emx_onnx_cgen/onnx_backend/test_suite.py::OnnxBackendNodeModelTest::test_reduce_l2_default_axes_keepdims_example_expanded_cpu` | [R9](#r9-reducel2-lowering-bugs) |
| [ ] | `src/emx_onnx_cgen/onnx_backend/test_suite.py::OnnxBackendNodeModelTest::test_reduce_l2_default_axes_keepdims_random_expanded_cpu` | [R9](#r9-reducel2-lowering-bugs) |
| [ ] | `src/emx_onnx_cgen/onnx_backend/test_suite.py::OnnxBackendNodeModelTest::test_reduce_l2_do_not_keepdims_example_expanded_cpu` | [R9](#r9-reducel2-lowering-bugs) |
| [ ] | `src/emx_onnx_cgen/onnx_backend/test_suite.py::OnnxBackendNodeModelTest::test_reduce_l2_do_not_keepdims_random_expanded_cpu` | [R9](#r9-reducel2-lowering-bugs) |
| [ ] | `src/emx_onnx_cgen/onnx_backend/test_suite.py::OnnxBackendNodeModelTest::test_reduce_l2_empty_set_expanded_cpu` | [R9](#r9-reducel2-lowering-bugs) |
| [ ] | `src/emx_onnx_cgen/onnx_backend/test_suite.py::OnnxBackendNodeModelTest::test_reduce_l2_keep_dims_example_expanded_cpu` | [R9](#r9-reducel2-lowering-bugs) |
| [ ] | `src/emx_onnx_cgen/onnx_backend/test_suite.py::OnnxBackendNodeModelTest::test_reduce_l2_keep_dims_random_expanded_cpu` | [R9](#r9-reducel2-lowering-bugs) |
| [ ] | `src/emx_onnx_cgen/onnx_backend/test_suite.py::OnnxBackendNodeModelTest::test_reduce_l2_negative_axes_keep_dims_example_expanded_cpu` | [R9](#r9-reducel2-lowering-bugs) |
| [ ] | `src/emx_onnx_cgen/onnx_backend/test_suite.py::OnnxBackendNodeModelTest::test_reduce_l2_negative_axes_keep_dims_random_expanded_cpu` | [R9](#r9-reducel2-lowering-bugs) |
| [ ] | `src/emx_onnx_cgen/onnx_backend/test_suite.py::OnnxBackendNodeModelTest::test_reduce_log_sum_asc_axes_expanded_cpu` | [R10](#r10-reducelogsum-and-reducelogsumexp-shape-lowering-bugs) |
| [ ] | `src/emx_onnx_cgen/onnx_backend/test_suite.py::OnnxBackendNodeModelTest::test_reduce_log_sum_desc_axes_expanded_cpu` | [R10](#r10-reducelogsum-and-reducelogsumexp-shape-lowering-bugs) |
| [ ] | `src/emx_onnx_cgen/onnx_backend/test_suite.py::OnnxBackendNodeModelTest::test_reduce_log_sum_empty_set_expanded_cpu` | [R10](#r10-reducelogsum-and-reducelogsumexp-shape-lowering-bugs) |
| [ ] | `src/emx_onnx_cgen/onnx_backend/test_suite.py::OnnxBackendNodeModelTest::test_reduce_log_sum_exp_default_axes_keepdims_example_expanded_cpu` | [R10](#r10-reducelogsum-and-reducelogsumexp-shape-lowering-bugs) |
| [ ] | `src/emx_onnx_cgen/onnx_backend/test_suite.py::OnnxBackendNodeModelTest::test_reduce_log_sum_exp_default_axes_keepdims_random_expanded_cpu` | [R10](#r10-reducelogsum-and-reducelogsumexp-shape-lowering-bugs) |
| [ ] | `src/emx_onnx_cgen/onnx_backend/test_suite.py::OnnxBackendNodeModelTest::test_reduce_log_sum_exp_do_not_keepdims_example_expanded_cpu` | [R10](#r10-reducelogsum-and-reducelogsumexp-shape-lowering-bugs) |
| [ ] | `src/emx_onnx_cgen/onnx_backend/test_suite.py::OnnxBackendNodeModelTest::test_reduce_log_sum_exp_do_not_keepdims_random_expanded_cpu` | [R10](#r10-reducelogsum-and-reducelogsumexp-shape-lowering-bugs) |
| [ ] | `src/emx_onnx_cgen/onnx_backend/test_suite.py::OnnxBackendNodeModelTest::test_reduce_log_sum_exp_empty_set_expanded_cpu` | [R10](#r10-reducelogsum-and-reducelogsumexp-shape-lowering-bugs) |
| [ ] | `src/emx_onnx_cgen/onnx_backend/test_suite.py::OnnxBackendNodeModelTest::test_reduce_log_sum_exp_keepdims_example_expanded_cpu` | [R10](#r10-reducelogsum-and-reducelogsumexp-shape-lowering-bugs) |
| [ ] | `src/emx_onnx_cgen/onnx_backend/test_suite.py::OnnxBackendNodeModelTest::test_reduce_log_sum_exp_keepdims_random_expanded_cpu` | [R10](#r10-reducelogsum-and-reducelogsumexp-shape-lowering-bugs) |
| [ ] | `src/emx_onnx_cgen/onnx_backend/test_suite.py::OnnxBackendNodeModelTest::test_reduce_log_sum_exp_negative_axes_keepdims_example_expanded_cpu` | [R10](#r10-reducelogsum-and-reducelogsumexp-shape-lowering-bugs) |
| [ ] | `src/emx_onnx_cgen/onnx_backend/test_suite.py::OnnxBackendNodeModelTest::test_reduce_log_sum_exp_negative_axes_keepdims_random_expanded_cpu` | [R10](#r10-reducelogsum-and-reducelogsumexp-shape-lowering-bugs) |
| [ ] | `src/emx_onnx_cgen/onnx_backend/test_suite.py::OnnxBackendNodeModelTest::test_reduce_log_sum_negative_axes_expanded_cpu` | [R10](#r10-reducelogsum-and-reducelogsumexp-shape-lowering-bugs) |
| [ ] | `src/emx_onnx_cgen/onnx_backend/test_suite.py::OnnxBackendNodeModelTest::test_sequence_insert_at_back_cpu` | [R13](#r13-jagged-sequence-elements-vs-fixed-lowered-shape) |
| [ ] | `src/emx_onnx_cgen/onnx_backend/test_suite.py::OnnxBackendNodeModelTest::test_sequence_insert_at_front_cpu` | [R13](#r13-jagged-sequence-elements-vs-fixed-lowered-shape) |
| [ ] | `src/emx_onnx_cgen/onnx_backend/test_suite.py::OnnxBackendNodeModelTest::test_sequence_map_add_1_sequence_1_tensor_cpu` | [R14](#r14-prepare-only-compilation-collapses-dynamic-dims) |
| [ ] | `src/emx_onnx_cgen/onnx_backend/test_suite.py::OnnxBackendNodeModelTest::test_sequence_map_add_1_sequence_1_tensor_expanded_cpu` | [R14](#r14-prepare-only-compilation-collapses-dynamic-dims) |
| [ ] | `src/emx_onnx_cgen/onnx_backend/test_suite.py::OnnxBackendNodeModelTest::test_sequence_map_add_2_sequences_cpu` | [R14](#r14-prepare-only-compilation-collapses-dynamic-dims) |
| [ ] | `src/emx_onnx_cgen/onnx_backend/test_suite.py::OnnxBackendNodeModelTest::test_sequence_map_add_2_sequences_expanded_cpu` | [R14](#r14-prepare-only-compilation-collapses-dynamic-dims) |
| [ ] | `src/emx_onnx_cgen/onnx_backend/test_suite.py::OnnxBackendNodeModelTest::test_sequence_map_extract_shapes_cpu` | [R14](#r14-prepare-only-compilation-collapses-dynamic-dims) |
| [ ] | `src/emx_onnx_cgen/onnx_backend/test_suite.py::OnnxBackendNodeModelTest::test_sequence_map_extract_shapes_expanded_cpu` | [R14](#r14-prepare-only-compilation-collapses-dynamic-dims) |
| [ ] | `src/emx_onnx_cgen/onnx_backend/test_suite.py::OnnxBackendNodeModelTest::test_sequence_map_identity_1_sequence_1_tensor_cpu` | [R14](#r14-prepare-only-compilation-collapses-dynamic-dims) |
| [ ] | `src/emx_onnx_cgen/onnx_backend/test_suite.py::OnnxBackendNodeModelTest::test_sequence_map_identity_1_sequence_1_tensor_expanded_cpu` | [R14](#r14-prepare-only-compilation-collapses-dynamic-dims) |
| [ ] | `src/emx_onnx_cgen/onnx_backend/test_suite.py::OnnxBackendNodeModelTest::test_sequence_map_identity_1_sequence_cpu` | [R14](#r14-prepare-only-compilation-collapses-dynamic-dims) |
| [ ] | `src/emx_onnx_cgen/onnx_backend/test_suite.py::OnnxBackendNodeModelTest::test_sequence_map_identity_1_sequence_expanded_cpu` | [R14](#r14-prepare-only-compilation-collapses-dynamic-dims) |
| [ ] | `src/emx_onnx_cgen/onnx_backend/test_suite.py::OnnxBackendNodeModelTest::test_sequence_map_identity_2_sequences_cpu` | [R14](#r14-prepare-only-compilation-collapses-dynamic-dims) |
| [ ] | `src/emx_onnx_cgen/onnx_backend/test_suite.py::OnnxBackendNodeModelTest::test_sequence_map_identity_2_sequences_expanded_cpu` | [R14](#r14-prepare-only-compilation-collapses-dynamic-dims) |
| [ ] | `src/emx_onnx_cgen/onnx_backend/test_suite.py::OnnxBackendNodeModelTest::test_split_to_sequence_1_cpu` | [R14](#r14-prepare-only-compilation-collapses-dynamic-dims) |
| [ ] | `src/emx_onnx_cgen/onnx_backend/test_suite.py::OnnxBackendNodeModelTest::test_split_to_sequence_2_cpu` | [R14](#r14-prepare-only-compilation-collapses-dynamic-dims) |
| [ ] | `src/emx_onnx_cgen/onnx_backend/test_suite.py::OnnxBackendSimpleModelTest::test_sequence_model4_cpu` | [R14](#r14-prepare-only-compilation-collapses-dynamic-dims) |
| [ ] | `src/emx_onnx_cgen/onnx_backend/test_suite.py::OnnxBackendSimpleModelTest::test_sequence_model8_cpu` | [R15](#r15-empty-dynamic-input-vs-prepare-only-fixed-shape) |
| [ ] | `src/emx_onnx_cgen/onnx_backend/test_suite.py::OnnxBackendNodeModelTest::test_identity_sequence_cpu` | [R12](#r12-sequence-element-shape-missing-in-model) |

## Reasons

### R1 `affine_grid` 2D output corruption

The models compile and run, and backend input/output serialization looks correct, but the compiled result quickly diverges into clearly wrong or uninitialized values. This points to operator lowering or generated kernel behavior, not to the ONNX backend adapter.

### R2 `affine_grid` 3D lowering failure

These models already fail during normal lowering with a reshape element-count mismatch (`Reshape input/output element counts must match`). This is outside the backend adapter.

### R3 `attention` operator result/precision mismatch

The `attention_*` cases reach execution, but the remaining failures look like operator-result or numeric-precision problems rather than harness I/O. `test_attention_4d_fp16_cpu`, for example, only misses backend tolerance by a small FP16 margin (`max abs diff 0.0004883`, `max rel diff 0.001202`).

### R4 `Bernoulli` nondeterministic output

The internal file-based verification is already only `OK (non-deterministic output)` for this operator family, while the ONNX backend suite expects concrete random results. This is not a pure adapter bug.

### R5 Window ops lowering failure

`BlackmanWindow`, `HammingWindow`, and `HannWindow` currently fail during lowering with `UnsupportedOpError: Reshape expects int64 or int32 shape input, got float`.

### R6 `CenterCropPad` kernel result wrong

The models compile and run, but the generated output is largely wrong or zeroed. This points to compiler or kernel logic, not backend transport.

### R7 `DFT` inverse numeric tolerance

The inverse DFT cases only differ by a very small floating-point margin (`~1.9e-7` while the backend test uses `atol=1e-7`). This looks like a numeric issue, not harness I/O.

### R8 `GroupNormalization` produces NaNs

The generated execution returns NaNs where the reference output is finite. This is a compiler or kernel issue.

### R9 `ReduceL2` lowering bugs

These failures happen in lowering, typically with `CastLike input and output shapes must match` or `ReduceSum output shape rank must match input rank`.

### R10 `ReduceLogSum` and `ReduceLogSumExp` shape/lowering bugs

This cluster is mixed: some cases already fail in lowering, while others run but return the wrong scalar/vector shape. The problem is in reduce lowering or generated operator behavior, not in backend input/output wiring.

### R12 Sequence element shape missing in model

The imported ONNX type only exposes `sequence(tensor(float))` without a usable element shape, so the prepare-only backend/testbench path collapses sequence elements to scalar buffers. This affects `identity_sequence`-style and loop-with-sequence cases.

### R13 Jagged sequence elements vs fixed lowered shape

These inputs contain tensors of different lengths, for example `[4]`, `[3]`, `[2]`. The backend adapter now serializes such sequences item-wise instead of crashing, but the prepare-only lowered model still fixes the element shape to `(3,)`, so values like `[1, 2, 3, 4]` are truncated to `[1, 2, 3]`.

### R14 Prepare-only compilation collapses dynamic dims

These models internally rely on compile-time concretization of dynamic tensor or sequence element sizes, but the ONNX backend must compile in `prepare()` without concrete runtime data. The resulting lowered shapes collapse to placeholders such as `(1,)`, `(3, 6)`, or `(2, 1, 4)`, which then truncates or pads runtime results.

### R15 Empty dynamic input vs prepare-only fixed shape

`test_sequence_model8_cpu` feeds an empty tensor where the prepare-only compile path lowered the dynamic input to a non-empty fixed shape. The testbench therefore consumes the wrong byte count for the first input and then fails on the next one with `Failed to read input Splits`.

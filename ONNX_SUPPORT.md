<!-- AUTO-GENERATED FILE. DO NOT EDIT. -->
<!-- Regenerate with: UPDATE_REFS=1 pytest -q tests/test_official_onnx_files_docs.py::test_official_onnx_file_support_doc -->

# ONNX test coverage

Overview:

| Test suite | Coverage | Version |
| --- | --- | --- |
| [Official ONNX test coverage](#official-onnx-test-coverage) | 1796 / 1802, 99.7% | 1.20.1 |
| [ONNX Runtime artifact coverage](#onnx-runtime-artifact-coverage) | 3944 / 4182, 94.3% | n/a |
| [Local ONNX test coverage](#local-onnx-test-coverage) | 7 / 7, 100.0% | n/a |

See [`ONNX_ERRORS.md`](ONNX_ERRORS.md) for the error histogram.

Floating-point verification first ignores very small differences up to **1.0 × [machine epsilon](https://en.wikipedia.org/wiki/Machine_epsilon) of the evaluated floating-point type**, treating such values as equal. For values with a larger absolute difference, the ULP distance is computed, and the maximum ULP distance is reported.

The `Verification` column uses `Input/Reference` notation (for example `Random/ORT`, `Random/ONNXRef`, `Data/Data`): `Input` can be `Random` (generated from model input metadata) or `Data` (loaded from ONNX test data files), and `Reference` can be `ORT` (computed with ONNX Runtime), `ONNXRef` (computed with the ONNX reference evaluator), or `Data` (expected outputs loaded from ONNX test data files).

## Official ONNX test coverage

Test directory: `onnx-org/onnx/backend/test/data`

Coverage 1796 / 1802 ONNX files (99.7%).

| File | Opset | Verification | Supported | Error |
| --- | --- | --- | --- | --- |
| light/light_bvlc_alexnet.onnx | 9 | Random/ORT | ✅ | OK (max ULP 0) |
| light/light_densenet121.onnx | 9 | Random/ORT | ✅ | OK (max ULP 71) |
| light/light_inception_v1.onnx | 9 | Random/ORT | ✅ | OK (max ULP 0) |
| light/light_inception_v2.onnx | 9 | Random/ORT | ✅ | OK (max ULP 0) |
| light/light_resnet50.onnx | 9 | Random/ORT | ✅ | OK (max ULP 0) |
| light/light_shufflenet.onnx | 9 | Random/ORT | ✅ | OK (max ULP 0) |
| light/light_squeezenet.onnx | 9 | Random/ORT | ✅ | OK (max ULP 0) |
| light/light_vgg19.onnx | 9 | Random/ORT | ✅ | OK (max ULP 0) |
| light/light_zfnet512.onnx | 9 | Random/ORT | ✅ | OK (max ULP 0) |
| node/test_abs/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_acos/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_acos_example/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_acosh/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_acosh_example/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_adagrad/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| node/test_adagrad_multiple/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| node/test_adam/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| node/test_adam_multiple/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 62311) |
| node/test_add/model.onnx | 14 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_add_bcast/model.onnx | 14 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_add_int16/model.onnx | 14 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_add_int8/model.onnx | 14 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_add_uint16/model.onnx | 14 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_add_uint32/model.onnx | 14 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_add_uint64/model.onnx | 14 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_add_uint8/model.onnx | 14 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_affine_grid_2d/model.onnx | 20 | Data/Data | ✅ | OK (max ULP 1) |
| node/test_affine_grid_2d_align_corners/model.onnx | 20 | Data/Data | ✅ | OK (max ULP 4) |
| node/test_affine_grid_2d_align_corners_expanded/model.onnx | 20 | Data/Data | ✅ | OK (max ULP 4) |
| node/test_affine_grid_2d_expanded/model.onnx | 20 | Data/Data | ✅ | OK (max ULP 5) |
| node/test_affine_grid_3d/model.onnx (--fp32-accumulation-strategy fp64) | 20 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_affine_grid_3d_align_corners/model.onnx | 20 | Data/Data | ✅ | OK (max ULP 23) |
| node/test_affine_grid_3d_align_corners_expanded/model.onnx | 20 | Data/Data | ✅ | OK (max ULP 23) |
| node/test_affine_grid_3d_expanded/model.onnx (--fp32-accumulation-strategy fp64) | 20 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_ai_onnx_ml_array_feature_extractor/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| node/test_ai_onnx_ml_binarizer/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| node/test_ai_onnx_ml_label_encoder_string_int/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_ai_onnx_ml_label_encoder_string_int_no_default/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_ai_onnx_ml_label_encoder_tensor_mapping/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_ai_onnx_ml_label_encoder_tensor_value_only_mapping/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_ai_onnx_ml_tree_ensemble_set_membership/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| node/test_ai_onnx_ml_tree_ensemble_single_tree/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| node/test_and2d/model.onnx | 7 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_and3d/model.onnx | 7 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_and4d/model.onnx | 7 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_and_bcast3v1d/model.onnx | 7 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_and_bcast3v2d/model.onnx | 7 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_and_bcast4v2d/model.onnx | 7 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_and_bcast4v3d/model.onnx | 7 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_and_bcast4v4d/model.onnx | 7 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_argmax_default_axis_example/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_argmax_default_axis_example_select_last_index/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_argmax_default_axis_random/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_argmax_default_axis_random_select_last_index/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_argmax_keepdims_example/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_argmax_keepdims_example_select_last_index/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_argmax_keepdims_random/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_argmax_keepdims_random_select_last_index/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_argmax_negative_axis_keepdims_example/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_argmax_negative_axis_keepdims_example_select_last_index/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_argmax_negative_axis_keepdims_random/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_argmax_negative_axis_keepdims_random_select_last_index/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_argmax_no_keepdims_example/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_argmax_no_keepdims_example_select_last_index/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_argmax_no_keepdims_random/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_argmax_no_keepdims_random_select_last_index/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_argmin_default_axis_example/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_argmin_default_axis_example_select_last_index/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_argmin_default_axis_random/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_argmin_default_axis_random_select_last_index/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_argmin_keepdims_example/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_argmin_keepdims_example_select_last_index/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_argmin_keepdims_random/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_argmin_keepdims_random_select_last_index/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_argmin_negative_axis_keepdims_example/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_argmin_negative_axis_keepdims_example_select_last_index/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_argmin_negative_axis_keepdims_random/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_argmin_negative_axis_keepdims_random_select_last_index/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_argmin_no_keepdims_example/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_argmin_no_keepdims_example_select_last_index/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_argmin_no_keepdims_random/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_argmin_no_keepdims_random_select_last_index/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_asin/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_asin_example/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_asinh/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_asinh_example/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_atan/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_atan_example/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_atanh/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_atanh_example/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_3d/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_3d_attn_mask/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_3d_attn_mask_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_3d_causal/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_3d_causal_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_3d_diff_heads_sizes/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_3d_diff_heads_sizes_attn_mask/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_3d_diff_heads_sizes_attn_mask_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_3d_diff_heads_sizes_causal/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_3d_diff_heads_sizes_causal_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_3d_diff_heads_sizes_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_3d_diff_heads_sizes_scaled/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_3d_diff_heads_sizes_scaled_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_3d_diff_heads_sizes_softcap/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_3d_diff_heads_sizes_softcap_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_3d_diff_heads_with_past_and_present/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_3d_diff_heads_with_past_and_present_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_3d_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_3d_gqa/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_3d_gqa_attn_mask/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_3d_gqa_attn_mask_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_3d_gqa_causal/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_3d_gqa_causal_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_3d_gqa_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_3d_gqa_scaled/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 3) |
| node/test_attention_3d_gqa_scaled_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_3d_gqa_softcap/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_3d_gqa_softcap_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 3) |
| node/test_attention_3d_gqa_with_past_and_present/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 3) |
| node/test_attention_3d_gqa_with_past_and_present_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 3) |
| node/test_attention_3d_scaled/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 3) |
| node/test_attention_3d_scaled_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_3d_softcap/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_3d_softcap_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_3d_transpose_verification/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_3d_transpose_verification_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_3d_with_past_and_present/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 3) |
| node/test_attention_3d_with_past_and_present_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_3d_with_past_and_present_qk_matmul/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 3) |
| node/test_attention_3d_with_past_and_present_qk_matmul_bias/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 3) |
| node/test_attention_3d_with_past_and_present_qk_matmul_bias_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_attention_3d_with_past_and_present_qk_matmul_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 3) |
| node/test_attention_3d_with_past_and_present_qk_matmul_softcap/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 3) |
| node/test_attention_3d_with_past_and_present_qk_matmul_softcap_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 3) |
| node/test_attention_3d_with_past_and_present_qk_matmul_softmax/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 3) |
| node/test_attention_3d_with_past_and_present_qk_matmul_softmax_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_4d/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_4d_attn_mask/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_4d_attn_mask_3d/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_4d_attn_mask_3d_causal/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_4d_attn_mask_3d_causal_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_4d_attn_mask_3d_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_4d_attn_mask_4d/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_4d_attn_mask_4d_causal/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_4d_attn_mask_4d_causal_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_4d_attn_mask_4d_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_4d_attn_mask_bool/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_4d_attn_mask_bool_4d/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_4d_attn_mask_bool_4d_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_4d_attn_mask_bool_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_4d_attn_mask_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_4d_causal/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_4d_causal_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_4d_diff_heads_mask4d_padded_kv/model.onnx | 24 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_4d_diff_heads_mask4d_padded_kv_expanded/model.onnx | 24 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_4d_diff_heads_sizes/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_4d_diff_heads_sizes_attn_mask/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_4d_diff_heads_sizes_attn_mask_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_4d_diff_heads_sizes_causal/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_4d_diff_heads_sizes_causal_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_4d_diff_heads_sizes_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_4d_diff_heads_sizes_scaled/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_4d_diff_heads_sizes_scaled_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_4d_diff_heads_sizes_softcap/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_4d_diff_heads_sizes_softcap_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_4d_diff_heads_with_past_and_present/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 3) |
| node/test_attention_4d_diff_heads_with_past_and_present_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_4d_diff_heads_with_past_and_present_mask3d/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_4d_diff_heads_with_past_and_present_mask3d_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 3) |
| node/test_attention_4d_diff_heads_with_past_and_present_mask4d/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 3) |
| node/test_attention_4d_diff_heads_with_past_and_present_mask4d_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 3) |
| node/test_attention_4d_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_4d_fp16/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_4d_fp16_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_4d_gqa/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_4d_gqa_attn_mask/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_4d_gqa_attn_mask_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_4d_gqa_causal/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_4d_gqa_causal_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_4d_gqa_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_4d_gqa_scaled/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_4d_gqa_scaled_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_4d_gqa_softcap/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_4d_gqa_softcap_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_4d_gqa_with_past_and_present/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_4d_gqa_with_past_and_present_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 3) |
| node/test_attention_4d_gqa_with_past_and_present_fp16/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 3) |
| node/test_attention_4d_gqa_with_past_and_present_fp16_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_4d_scaled/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_4d_scaled_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_4d_softcap/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_4d_softcap_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_4d_with_past_and_present/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_4d_with_past_and_present_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 3) |
| node/test_attention_4d_with_past_and_present_qk_matmul/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_4d_with_past_and_present_qk_matmul_bias/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_attention_4d_with_past_and_present_qk_matmul_bias_3d_mask/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_attention_4d_with_past_and_present_qk_matmul_bias_3d_mask_causal/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_attention_4d_with_past_and_present_qk_matmul_bias_3d_mask_causal_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 3) |
| node/test_attention_4d_with_past_and_present_qk_matmul_bias_3d_mask_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 3) |
| node/test_attention_4d_with_past_and_present_qk_matmul_bias_4d_mask/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_attention_4d_with_past_and_present_qk_matmul_bias_4d_mask_causal/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_attention_4d_with_past_and_present_qk_matmul_bias_4d_mask_causal_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_attention_4d_with_past_and_present_qk_matmul_bias_4d_mask_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_attention_4d_with_past_and_present_qk_matmul_bias_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 3) |
| node/test_attention_4d_with_past_and_present_qk_matmul_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 3) |
| node/test_attention_4d_with_qk_matmul/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_4d_with_qk_matmul_bias/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_attention_4d_with_qk_matmul_bias_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 1) |
| node/test_attention_4d_with_qk_matmul_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_attention_4d_with_qk_matmul_softcap/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_attention_4d_with_qk_matmul_softcap_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 3) |
| node/test_attention_4d_with_qk_matmul_softmax/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_attention_4d_with_qk_matmul_softmax_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_averagepool_1d_default/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_averagepool_2d_ceil/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_averagepool_2d_ceil_last_window_starts_on_pad/model.onnx (--runtime onnx-reference --test-data-inputs-only) | 22 | Data/ONNXRef | ✅ | OK (max ULP 0) |
| node/test_averagepool_2d_default/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_averagepool_2d_dilations/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_averagepool_2d_pads/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 3) |
| node/test_averagepool_2d_pads_count_include_pad/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_averagepool_2d_precomputed_pads/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_averagepool_2d_precomputed_pads_count_include_pad/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_averagepool_2d_precomputed_same_upper/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_averagepool_2d_precomputed_strides/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_averagepool_2d_same_lower/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_averagepool_2d_same_upper/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_averagepool_2d_strides/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_averagepool_3d_default/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_averagepool_3d_dilations_large_count_include_pad_is_0_ceil_mode_is_False/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_averagepool_3d_dilations_large_count_include_pad_is_0_ceil_mode_is_True/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_averagepool_3d_dilations_large_count_include_pad_is_1_ceil_mode_is_False/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_averagepool_3d_dilations_large_count_include_pad_is_1_ceil_mode_is_True/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_averagepool_3d_dilations_small/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_basic_conv_with_padding/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_basic_conv_without_padding/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_basic_deform_conv_with_padding/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_basic_deform_conv_without_padding/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_batchnorm_epsilon/model.onnx | 15 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_batchnorm_epsilon_training_mode/model.onnx | 15 | Data/Data | ✅ | OK (max ULP 1) |
| node/test_batchnorm_example/model.onnx | 15 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_batchnorm_example_training_mode/model.onnx | 15 | Data/Data | ✅ | OK (max ULP 6) |
| node/test_bernoulli/model.onnx | 22 | Data/Data | ✅ | OK (non-deterministic output) |
| node/test_bernoulli_double/model.onnx | 22 | Data/Data | ✅ | OK (non-deterministic output) |
| node/test_bernoulli_double_expanded/model.onnx | 22 | Data/Data | ✅ | OK (non-deterministic output) |
| node/test_bernoulli_expanded/model.onnx | 22 | Data/Data | ✅ | OK (non-deterministic output) |
| node/test_bernoulli_seed/model.onnx | 22 | Data/Data | ✅ | OK (non-deterministic output) |
| node/test_bernoulli_seed_expanded/model.onnx | 22 | Data/Data | ✅ | OK (non-deterministic output) |
| node/test_bitshift_left_uint16/model.onnx | 11 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_bitshift_left_uint32/model.onnx | 11 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_bitshift_left_uint64/model.onnx | 11 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_bitshift_left_uint8/model.onnx | 11 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_bitshift_right_uint16/model.onnx | 11 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_bitshift_right_uint32/model.onnx | 11 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_bitshift_right_uint64/model.onnx | 11 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_bitshift_right_uint8/model.onnx | 11 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_bitwise_and_i16_3d/model.onnx | 18 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_bitwise_and_i32_2d/model.onnx | 18 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_bitwise_and_ui64_bcast_3v1d/model.onnx | 18 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_bitwise_and_ui8_bcast_4v3d/model.onnx | 18 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_bitwise_not_2d/model.onnx | 18 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_bitwise_not_3d/model.onnx | 18 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_bitwise_not_4d/model.onnx | 18 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_bitwise_or_i16_4d/model.onnx | 18 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_bitwise_or_i32_2d/model.onnx | 18 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_bitwise_or_ui64_bcast_3v1d/model.onnx | 18 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_bitwise_or_ui8_bcast_4v3d/model.onnx | 18 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_bitwise_xor_i16_3d/model.onnx | 18 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_bitwise_xor_i32_2d/model.onnx | 18 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_bitwise_xor_ui64_bcast_3v1d/model.onnx | 18 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_bitwise_xor_ui8_bcast_4v3d/model.onnx | 18 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_blackmanwindow/model.onnx | 17 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_blackmanwindow_expanded/model.onnx | 17 | Data/Data | ✅ | OK (max ULP 4) |
| node/test_blackmanwindow_symmetric/model.onnx | 17 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_blackmanwindow_symmetric_expanded/model.onnx | 17 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_cast_BFLOAT16_to_FLOAT/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_cast_DOUBLE_to_FLOAT/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_cast_DOUBLE_to_FLOAT16/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_cast_FLOAT16_to_DOUBLE/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_cast_FLOAT16_to_FLOAT/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_cast_FLOAT16_to_FLOAT4E2M1/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_cast_FLOAT16_to_FLOAT8E4M3FN/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_cast_FLOAT16_to_FLOAT8E4M3FNUZ/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_cast_FLOAT16_to_FLOAT8E5M2/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_cast_FLOAT16_to_FLOAT8E5M2FNUZ/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_cast_FLOAT16_to_INT2/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_cast_FLOAT16_to_INT4/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_cast_FLOAT16_to_UINT2/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_cast_FLOAT16_to_UINT4/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_cast_FLOAT4E2M1_to_FLOAT/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_cast_FLOAT4E2M1_to_FLOAT16/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_cast_FLOAT8E4M3FNUZ_to_FLOAT/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_cast_FLOAT8E4M3FNUZ_to_FLOAT16/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_cast_FLOAT8E4M3FN_to_FLOAT/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_cast_FLOAT8E4M3FN_to_FLOAT16/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_cast_FLOAT8E5M2FNUZ_to_FLOAT/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_cast_FLOAT8E5M2FNUZ_to_FLOAT16/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_cast_FLOAT8E5M2_to_FLOAT/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_cast_FLOAT8E5M2_to_FLOAT16/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_cast_FLOAT_to_BFLOAT16/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_cast_FLOAT_to_DOUBLE/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_cast_FLOAT_to_FLOAT16/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_cast_FLOAT_to_FLOAT4E2M1/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_cast_FLOAT_to_FLOAT8E4M3FN/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_cast_FLOAT_to_FLOAT8E4M3FNUZ/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_cast_FLOAT_to_FLOAT8E5M2/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_cast_FLOAT_to_FLOAT8E5M2FNUZ/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_cast_FLOAT_to_INT2/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_cast_FLOAT_to_INT4/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_cast_FLOAT_to_UINT2/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_cast_FLOAT_to_UINT4/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_cast_INT2_to_FLOAT/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_cast_INT2_to_FLOAT16/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_cast_INT2_to_INT8/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_cast_INT4_to_FLOAT/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_cast_INT4_to_FLOAT16/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_cast_INT4_to_INT8/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_cast_UINT2_to_FLOAT/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_cast_UINT2_to_FLOAT16/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_cast_UINT2_to_UINT8/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_cast_UINT4_to_FLOAT/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_cast_UINT4_to_FLOAT16/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_cast_UINT4_to_UINT8/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_cast_e8m0_FLOAT16_to_FLOAT8E8M0/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_cast_e8m0_FLOAT8E8M0_to_FLOAT/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_cast_e8m0_FLOAT8E8M0_to_FLOAT16/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_cast_e8m0_FLOAT_to_FLOAT8E8M0/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_cast_no_saturate_FLOAT16_to_FLOAT8E4M3FN/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_cast_no_saturate_FLOAT16_to_FLOAT8E4M3FNUZ/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_cast_no_saturate_FLOAT16_to_FLOAT8E5M2/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_cast_no_saturate_FLOAT16_to_FLOAT8E5M2FNUZ/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_cast_no_saturate_FLOAT_to_FLOAT8E4M3FN/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_cast_no_saturate_FLOAT_to_FLOAT8E4M3FNUZ/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_cast_no_saturate_FLOAT_to_FLOAT8E5M2/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_cast_no_saturate_FLOAT_to_FLOAT8E5M2FNUZ/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_castlike_BFLOAT16_to_FLOAT/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_castlike_BFLOAT16_to_FLOAT_expanded/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_castlike_DOUBLE_to_FLOAT/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_castlike_DOUBLE_to_FLOAT16/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_castlike_DOUBLE_to_FLOAT16_expanded/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_castlike_DOUBLE_to_FLOAT_expanded/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_castlike_FLOAT16_to_DOUBLE/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_castlike_FLOAT16_to_DOUBLE_expanded/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_castlike_FLOAT16_to_FLOAT/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_castlike_FLOAT16_to_FLOAT4E2M1/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_castlike_FLOAT16_to_FLOAT4E2M1_expanded/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_castlike_FLOAT16_to_FLOAT8E4M3FN/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_castlike_FLOAT16_to_FLOAT8E4M3FNUZ/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_castlike_FLOAT16_to_FLOAT8E4M3FNUZ_expanded/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_castlike_FLOAT16_to_FLOAT8E4M3FN_expanded/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_castlike_FLOAT16_to_FLOAT8E5M2/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_castlike_FLOAT16_to_FLOAT8E5M2FNUZ/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_castlike_FLOAT16_to_FLOAT8E5M2FNUZ_expanded/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_castlike_FLOAT16_to_FLOAT8E5M2_expanded/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_castlike_FLOAT16_to_FLOAT_expanded/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_castlike_FLOAT16_to_INT2/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_castlike_FLOAT16_to_INT2_expanded/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_castlike_FLOAT16_to_INT4/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_castlike_FLOAT16_to_INT4_expanded/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_castlike_FLOAT16_to_UINT2/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_castlike_FLOAT16_to_UINT2_expanded/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_castlike_FLOAT16_to_UINT4/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_castlike_FLOAT16_to_UINT4_expanded/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_castlike_FLOAT4E2M1_to_FLOAT/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_castlike_FLOAT4E2M1_to_FLOAT16/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_castlike_FLOAT4E2M1_to_FLOAT16_expanded/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_castlike_FLOAT4E2M1_to_FLOAT_expanded/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_castlike_FLOAT8E4M3FNUZ_to_FLOAT/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_castlike_FLOAT8E4M3FNUZ_to_FLOAT16/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_castlike_FLOAT8E4M3FNUZ_to_FLOAT16_expanded/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_castlike_FLOAT8E4M3FNUZ_to_FLOAT_expanded/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_castlike_FLOAT8E4M3FN_to_FLOAT/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_castlike_FLOAT8E4M3FN_to_FLOAT16/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_castlike_FLOAT8E4M3FN_to_FLOAT16_expanded/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_castlike_FLOAT8E4M3FN_to_FLOAT_expanded/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_castlike_FLOAT8E5M2FNUZ_to_FLOAT/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_castlike_FLOAT8E5M2FNUZ_to_FLOAT16/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_castlike_FLOAT8E5M2FNUZ_to_FLOAT16_expanded/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_castlike_FLOAT8E5M2FNUZ_to_FLOAT_expanded/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_castlike_FLOAT8E5M2_to_FLOAT/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_castlike_FLOAT8E5M2_to_FLOAT16/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_castlike_FLOAT8E5M2_to_FLOAT16_expanded/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_castlike_FLOAT8E5M2_to_FLOAT_expanded/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_castlike_FLOAT_to_BFLOAT16/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_castlike_FLOAT_to_BFLOAT16_expanded/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_castlike_FLOAT_to_DOUBLE/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_castlike_FLOAT_to_DOUBLE_expanded/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_castlike_FLOAT_to_FLOAT16/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_castlike_FLOAT_to_FLOAT16_expanded/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_castlike_FLOAT_to_FLOAT4E2M1/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_castlike_FLOAT_to_FLOAT4E2M1_expanded/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_castlike_FLOAT_to_FLOAT8E4M3FN/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_castlike_FLOAT_to_FLOAT8E4M3FNUZ/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_castlike_FLOAT_to_FLOAT8E4M3FNUZ_expanded/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_castlike_FLOAT_to_FLOAT8E4M3FN_expanded/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_castlike_FLOAT_to_FLOAT8E5M2/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_castlike_FLOAT_to_FLOAT8E5M2FNUZ/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_castlike_FLOAT_to_FLOAT8E5M2FNUZ_expanded/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_castlike_FLOAT_to_FLOAT8E5M2_expanded/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_castlike_FLOAT_to_INT2/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_castlike_FLOAT_to_INT2_expanded/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_castlike_FLOAT_to_INT4/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_castlike_FLOAT_to_INT4_expanded/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_castlike_FLOAT_to_UINT2/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_castlike_FLOAT_to_UINT2_expanded/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_castlike_FLOAT_to_UINT4/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_castlike_FLOAT_to_UINT4_expanded/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_castlike_INT2_to_FLOAT/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_castlike_INT2_to_FLOAT16/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_castlike_INT2_to_FLOAT16_expanded/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_castlike_INT2_to_FLOAT_expanded/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_castlike_INT2_to_INT8/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_castlike_INT2_to_INT8_expanded/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_castlike_INT4_to_FLOAT/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_castlike_INT4_to_FLOAT16/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_castlike_INT4_to_FLOAT16_expanded/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_castlike_INT4_to_FLOAT_expanded/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_castlike_INT4_to_INT8/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_castlike_INT4_to_INT8_expanded/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_castlike_UINT2_to_FLOAT/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_castlike_UINT2_to_FLOAT16/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_castlike_UINT2_to_FLOAT16_expanded/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_castlike_UINT2_to_FLOAT_expanded/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_castlike_UINT2_to_UINT8/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_castlike_UINT2_to_UINT8_expanded/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_castlike_UINT4_to_FLOAT/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_castlike_UINT4_to_FLOAT16/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_castlike_UINT4_to_FLOAT16_expanded/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_castlike_UINT4_to_FLOAT_expanded/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_castlike_UINT4_to_UINT8/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_castlike_UINT4_to_UINT8_expanded/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_castlike_no_saturate_FLOAT16_to_FLOAT8E4M3FN/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_castlike_no_saturate_FLOAT16_to_FLOAT8E4M3FNUZ/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_castlike_no_saturate_FLOAT16_to_FLOAT8E4M3FNUZ_expanded/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_castlike_no_saturate_FLOAT16_to_FLOAT8E4M3FN_expanded/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_castlike_no_saturate_FLOAT16_to_FLOAT8E5M2/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_castlike_no_saturate_FLOAT16_to_FLOAT8E5M2FNUZ/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_castlike_no_saturate_FLOAT16_to_FLOAT8E5M2FNUZ_expanded/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_castlike_no_saturate_FLOAT16_to_FLOAT8E5M2_expanded/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_castlike_no_saturate_FLOAT_to_FLOAT8E4M3FN/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_castlike_no_saturate_FLOAT_to_FLOAT8E4M3FNUZ/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_castlike_no_saturate_FLOAT_to_FLOAT8E4M3FNUZ_expanded/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_castlike_no_saturate_FLOAT_to_FLOAT8E4M3FN_expanded/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_castlike_no_saturate_FLOAT_to_FLOAT8E5M2/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_castlike_no_saturate_FLOAT_to_FLOAT8E5M2FNUZ/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_castlike_no_saturate_FLOAT_to_FLOAT8E5M2FNUZ_expanded/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_castlike_no_saturate_FLOAT_to_FLOAT8E5M2_expanded/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_ceil/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_ceil_example/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_celu/model.onnx | 12 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_celu_expanded/model.onnx | 12 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_center_crop_pad_crop/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_center_crop_pad_crop_and_pad/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_center_crop_pad_crop_and_pad_expanded/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_center_crop_pad_crop_axes_chw/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_center_crop_pad_crop_axes_chw_expanded/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_center_crop_pad_crop_axes_hwc/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_center_crop_pad_crop_axes_hwc_expanded/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_center_crop_pad_crop_expanded/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_center_crop_pad_crop_negative_axes_hwc/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_center_crop_pad_crop_negative_axes_hwc_expanded/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_center_crop_pad_pad/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_center_crop_pad_pad_expanded/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_clip/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_clip_default_inbounds/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_clip_default_inbounds_expanded/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_clip_default_int8_inbounds/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_clip_default_int8_inbounds_expanded/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_clip_default_int8_max/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_clip_default_int8_max_expanded/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_clip_default_int8_min/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_clip_default_int8_min_expanded/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_clip_default_max/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_clip_default_max_expanded/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_clip_default_min/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_clip_default_min_expanded/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_clip_example/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_clip_example_expanded/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_clip_expanded/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_clip_inbounds/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_clip_inbounds_expanded/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_clip_min_greater_than_max/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_clip_min_greater_than_max_expanded/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_clip_outbounds/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_clip_outbounds_expanded/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_clip_splitbounds/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_clip_splitbounds_expanded/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_col2im/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_col2im_5d/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_col2im_dilations/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_col2im_pads/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_col2im_strides/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_compress_0/model.onnx | 11 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_compress_1/model.onnx | 11 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_compress_default_axis/model.onnx | 11 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_compress_negative_axis/model.onnx | 11 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_concat_1d_axis_0/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_concat_1d_axis_negative_1/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_concat_2d_axis_0/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_concat_2d_axis_1/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_concat_2d_axis_negative_1/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_concat_2d_axis_negative_2/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_concat_3d_axis_0/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_concat_3d_axis_1/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_concat_3d_axis_2/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_concat_3d_axis_negative_1/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_concat_3d_axis_negative_2/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_concat_3d_axis_negative_3/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_constant/model.onnx (--runtime onnx-reference) | 25 | Random/ONNXRef | ✅ | OK (max ULP 0) |
| node/test_constant_pad/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_constant_pad_axes/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_constant_pad_negative_axes/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_constantofshape_float_ones/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_constantofshape_int_shape_zero/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_constantofshape_int_zeros/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_conv_with_autopad_same/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_conv_with_strides_and_asymmetric_padding/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_conv_with_strides_no_padding/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_conv_with_strides_padding/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_convinteger_with_padding/model.onnx | 10 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_convinteger_without_padding/model.onnx | 10 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_convtranspose/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_convtranspose_1d/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_convtranspose_3d/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_convtranspose_autopad_same/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_convtranspose_dilations/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_convtranspose_group_2/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_convtranspose_group_2_image_3/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_convtranspose_kernel_shape/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_convtranspose_output_shape/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_convtranspose_pad/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_convtranspose_pads/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_cos/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_cos_example/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_cosh/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_cosh_example/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_cumsum_1d/model.onnx | 14 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_cumsum_1d_exclusive/model.onnx | 14 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_cumsum_1d_int32_exclusive/model.onnx | 14 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_cumsum_1d_reverse/model.onnx | 14 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_cumsum_1d_reverse_exclusive/model.onnx | 14 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_cumsum_2d_axis_0/model.onnx | 14 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_cumsum_2d_axis_1/model.onnx | 14 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_cumsum_2d_int32/model.onnx | 14 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_cumsum_2d_negative_axis/model.onnx | 14 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_deform_conv_with_mask_bias/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_deform_conv_with_multiple_offset_groups/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_depthtospace_crd_mode_example/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_depthtospace_example/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_dequantizelinear/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_dequantizelinear_axis/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_dequantizelinear_blocked/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_dequantizelinear_e4m3fn/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_dequantizelinear_e4m3fn_float16/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_dequantizelinear_e4m3fn_zero_point/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_dequantizelinear_e5m2/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_dequantizelinear_float4e2m1/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_dequantizelinear_int16/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_dequantizelinear_int2/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_dequantizelinear_int4/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_dequantizelinear_uint16/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_dequantizelinear_uint2/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_dequantizelinear_uint4/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_det_2d/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_det_nd/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_dft/model.onnx | 20 | Data/Data | ✅ | OK (max ULP 4) |
| node/test_dft_axis/model.onnx | 20 | Data/Data | ✅ | OK (max ULP 72) |
| node/test_dft_axis_opset19/model.onnx | 19 | Data/Data | ✅ | OK (max ULP 72) |
| node/test_dft_inverse/model.onnx (--atol-eps 2) | 20 | Data/Data | ✅ | OK (max ULP 8) |
| node/test_dft_inverse_opset19/model.onnx (--atol-eps 2) | 19 | Data/Data | ✅ | OK (max ULP 8) |
| node/test_dft_opset19/model.onnx | 19 | Data/Data | ✅ | OK (max ULP 4) |
| node/test_div/model.onnx | 14 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_div_bcast/model.onnx | 14 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_div_example/model.onnx | 14 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_div_int16/model.onnx | 14 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_div_int8/model.onnx | 14 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_div_uint16/model.onnx | 14 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_div_uint32/model.onnx | 14 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_div_uint64/model.onnx | 14 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_div_uint8/model.onnx | 14 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_dropout_default/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_dropout_default_mask/model.onnx | 22 | Data/Data | ✅ | OK (max abs diff 0, max ULP 0) |
| node/test_dropout_default_mask_ratio/model.onnx | 22 | Data/Data | ✅ | OK (max abs diff 0, max ULP 0) |
| node/test_dropout_default_old/model.onnx | 11 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_dropout_default_ratio/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_dropout_random_old/model.onnx | 11 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_dynamicquantizelinear/model.onnx | 11 | Data/Data | ✅ | OK (max abs diff 0, max ULP 0) |
| node/test_dynamicquantizelinear_expanded/model.onnx | 11 | Data/Data | ✅ | OK (max abs diff 0, max ULP 0) |
| node/test_dynamicquantizelinear_max_adjusted/model.onnx | 11 | Data/Data | ✅ | OK (max abs diff 0, max ULP 0) |
| node/test_dynamicquantizelinear_max_adjusted_expanded/model.onnx | 11 | Data/Data | ✅ | OK (max abs diff 0, max ULP 0) |
| node/test_dynamicquantizelinear_min_adjusted/model.onnx | 11 | Data/Data | ✅ | OK (max abs diff 0, max ULP 0) |
| node/test_dynamicquantizelinear_min_adjusted_expanded/model.onnx | 11 | Data/Data | ✅ | OK (max abs diff 0, max ULP 0) |
| node/test_edge_pad/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_einsum_batch_diagonal/model.onnx | 12 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_einsum_batch_matmul/model.onnx | 12 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_einsum_inner_prod/model.onnx | 12 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_einsum_scalar/model.onnx | 12 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_einsum_sum/model.onnx | 12 | Data/Data | ✅ | OK (max ULP 1) |
| node/test_einsum_transpose/model.onnx | 12 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_elu/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_elu_default/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_elu_default_expanded_ver18/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_elu_example/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_elu_example_expanded_ver18/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_elu_expanded_ver18/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_equal/model.onnx | 19 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_equal_bcast/model.onnx | 19 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_equal_int16/model.onnx | 19 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_equal_int8/model.onnx | 19 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_equal_string/model.onnx | 19 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_equal_string_broadcast/model.onnx | 19 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_equal_uint16/model.onnx | 19 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_equal_uint32/model.onnx | 19 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_equal_uint64/model.onnx | 19 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_equal_uint8/model.onnx | 19 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_erf/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_exp/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 1) |
| node/test_exp_example/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 1) |
| node/test_expand_dim_changed/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_expand_dim_unchanged/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_eyelike_populate_off_main_diagonal/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_eyelike_with_dtype/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_eyelike_without_dtype/model.onnx | 22 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_flatten_axis0/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_flatten_axis1/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_flatten_axis2/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_flatten_axis3/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_flatten_default_axis/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_flatten_negative_axis1/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_flatten_negative_axis2/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_flatten_negative_axis3/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_flatten_negative_axis4/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_floor/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_floor_example/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_gather_0/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_gather_1/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_gather_2d_indices/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_gather_elements_0/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_gather_elements_1/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_gather_elements_negative_indices/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_gather_negative_indices/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_gathernd_example_float32/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_gathernd_example_int32/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_gathernd_example_int32_batch_dim1/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_gelu_default_1/model.onnx | 20 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_gelu_default_1_expanded/model.onnx | 20 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_gelu_default_2/model.onnx | 20 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_gelu_default_2_expanded/model.onnx | 20 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_gelu_tanh_1/model.onnx | 20 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_gelu_tanh_1_expanded/model.onnx | 20 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_gelu_tanh_2/model.onnx | 20 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_gelu_tanh_2_expanded/model.onnx | 20 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_gemm_all_attributes/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_gemm_alpha/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_gemm_beta/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 1) |
| node/test_gemm_default_matrix_bias/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 1) |
| node/test_gemm_default_no_bias/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_gemm_default_scalar_bias/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_gemm_default_single_elem_vector_bias/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 1) |
| node/test_gemm_default_vector_bias/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 1) |
| node/test_gemm_default_zero_bias/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_gemm_transposeA/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 1) |
| node/test_gemm_transposeB/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 1) |
| node/test_globalaveragepool/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_globalaveragepool_precomputed/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_globalmaxpool/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_globalmaxpool_precomputed/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_greater/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_greater_bcast/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_greater_equal/model.onnx | 16 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_greater_equal_bcast/model.onnx | 16 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_greater_equal_bcast_expanded/model.onnx | 16 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_greater_equal_expanded/model.onnx | 16 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_greater_equal_int16/model.onnx | 16 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_greater_equal_int16_expanded/model.onnx | 16 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_greater_equal_int8/model.onnx | 16 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_greater_equal_int8_expanded/model.onnx | 16 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_greater_equal_uint16/model.onnx | 16 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_greater_equal_uint16_expanded/model.onnx | 16 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_greater_equal_uint32/model.onnx | 16 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_greater_equal_uint32_expanded/model.onnx | 16 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_greater_equal_uint64/model.onnx | 16 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_greater_equal_uint64_expanded/model.onnx | 16 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_greater_equal_uint8/model.onnx | 16 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_greater_equal_uint8_expanded/model.onnx | 16 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_greater_int16/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_greater_int8/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_greater_uint16/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_greater_uint32/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_greater_uint64/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_greater_uint8/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_gridsample/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_gridsample_aligncorners_true/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_gridsample_bicubic/model.onnx (--runtime onnx-reference --test-data-inputs-only) | 22 | Data/ONNXRef | ✅ | OK (max ULP 13) |
| node/test_gridsample_bicubic_align_corners_0_additional_1/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 13) |
| node/test_gridsample_bicubic_align_corners_1_additional_1/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 34) |
| node/test_gridsample_bilinear/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_gridsample_bilinear_align_corners_0_additional_1/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_gridsample_bilinear_align_corners_1_additional_1/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_gridsample_border_padding/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_gridsample_nearest/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_gridsample_nearest_align_corners_0_additional_1/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_gridsample_nearest_align_corners_1_additional_1/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_gridsample_reflection_padding/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_gridsample_volumetric_bilinear_align_corners_0/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 1) |
| node/test_gridsample_volumetric_bilinear_align_corners_1/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_gridsample_volumetric_nearest_align_corners_0/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_gridsample_volumetric_nearest_align_corners_1/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_gridsample_zeros_padding/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_group_normalization_epsilon/model.onnx | 21 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_group_normalization_epsilon_expanded/model.onnx | 21 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_group_normalization_example/model.onnx | 21 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_group_normalization_example_expanded/model.onnx | 21 | Data/Data | ✅ | OK (max ULP 1) |
| node/test_gru_batchwise/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_gru_defaults/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_gru_seq_length/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_gru_with_initial_bias/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_hammingwindow/model.onnx | 17 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_hammingwindow_expanded/model.onnx | 17 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_hammingwindow_symmetric/model.onnx | 17 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_hammingwindow_symmetric_expanded/model.onnx | 17 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_hannwindow/model.onnx | 17 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_hannwindow_expanded/model.onnx | 17 | Data/Data | ✅ | OK (max ULP 3) |
| node/test_hannwindow_symmetric/model.onnx | 17 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_hannwindow_symmetric_expanded/model.onnx | 17 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_hardmax_axis_0/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_hardmax_axis_1/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_hardmax_axis_2/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_hardmax_default_axis/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_hardmax_example/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_hardmax_negative_axis/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_hardmax_one_hot/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_hardsigmoid/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_hardsigmoid_default/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_hardsigmoid_default_expanded_ver18/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_hardsigmoid_example/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_hardsigmoid_example_expanded_ver18/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_hardsigmoid_expanded_ver18/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_hardswish/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_hardswish_expanded/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_identity/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_identity_opt/model.onnx | 16 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_identity_sequence/model.onnx (--sequence-element-shape x=[1,1,2,2]) | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_if/model.onnx | 11 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_if_opt/model.onnx | 16 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_if_seq/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_image_decoder_decode_bmp_rgb/model.onnx | 20 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_image_decoder_decode_jpeg2k_rgb/model.onnx | 20 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_image_decoder_decode_jpeg_bgr/model.onnx | 20 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_image_decoder_decode_jpeg_grayscale/model.onnx | 20 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_image_decoder_decode_jpeg_rgb/model.onnx | 20 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_image_decoder_decode_png_rgb/model.onnx | 20 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_image_decoder_decode_pnm_rgb/model.onnx | 20 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_image_decoder_decode_tiff_rgb/model.onnx | 20 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_image_decoder_decode_webp_rgb/model.onnx | 20 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_instancenorm_epsilon/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_instancenorm_example/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_isinf/model.onnx | 20 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_isinf_float16/model.onnx | 20 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_isinf_negative/model.onnx | 20 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_isinf_positive/model.onnx | 20 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_isnan/model.onnx | 20 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_isnan_float16/model.onnx | 20 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_l1normalization_axis_0/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_l1normalization_axis_1/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_l1normalization_axis_last/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_l2normalization_axis_0/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_l2normalization_axis_1/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_layer_normalization_2d_axis0/model.onnx | 17 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_layer_normalization_2d_axis0_expanded/model.onnx | 17 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_layer_normalization_2d_axis0_expanded_ver18/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_layer_normalization_2d_axis1/model.onnx | 17 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_layer_normalization_2d_axis1_expanded/model.onnx | 17 | Data/Data | ✅ | OK (max ULP 24) |
| node/test_layer_normalization_2d_axis1_expanded_ver18/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 24) |
| node/test_layer_normalization_2d_axis_negative_1/model.onnx | 17 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_layer_normalization_2d_axis_negative_1_expanded/model.onnx | 17 | Data/Data | ✅ | OK (max ULP 80) |
| node/test_layer_normalization_2d_axis_negative_1_expanded_ver18/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 80) |
| node/test_layer_normalization_2d_axis_negative_2/model.onnx | 17 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_layer_normalization_2d_axis_negative_2_expanded/model.onnx | 17 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_layer_normalization_2d_axis_negative_2_expanded_ver18/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_layer_normalization_3d_axis0_epsilon/model.onnx | 17 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_layer_normalization_3d_axis0_epsilon_expanded/model.onnx | 17 | Data/Data | ✅ | OK (max ULP 8) |
| node/test_layer_normalization_3d_axis0_epsilon_expanded_ver18/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 8) |
| node/test_layer_normalization_3d_axis1_epsilon/model.onnx | 17 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_layer_normalization_3d_axis1_epsilon_expanded/model.onnx | 17 | Data/Data | ✅ | OK (max ULP 16) |
| node/test_layer_normalization_3d_axis1_epsilon_expanded_ver18/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 16) |
| node/test_layer_normalization_3d_axis2_epsilon/model.onnx | 17 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_layer_normalization_3d_axis2_epsilon_expanded/model.onnx | 17 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_layer_normalization_3d_axis2_epsilon_expanded_ver18/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_layer_normalization_3d_axis_negative_1_epsilon/model.onnx | 17 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_layer_normalization_3d_axis_negative_1_epsilon_expanded/model.onnx | 17 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_layer_normalization_3d_axis_negative_1_epsilon_expanded_ver18/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_layer_normalization_3d_axis_negative_2_epsilon/model.onnx | 17 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_layer_normalization_3d_axis_negative_2_epsilon_expanded/model.onnx | 17 | Data/Data | ✅ | OK (max ULP 4) |
| node/test_layer_normalization_3d_axis_negative_2_epsilon_expanded_ver18/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 4) |
| node/test_layer_normalization_3d_axis_negative_3_epsilon/model.onnx | 17 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_layer_normalization_3d_axis_negative_3_epsilon_expanded/model.onnx | 17 | Data/Data | ✅ | OK (max ULP 9) |
| node/test_layer_normalization_3d_axis_negative_3_epsilon_expanded_ver18/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 9) |
| node/test_layer_normalization_4d_axis0/model.onnx | 17 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_layer_normalization_4d_axis0_expanded/model.onnx | 17 | Data/Data | ✅ | OK (max ULP 6) |
| node/test_layer_normalization_4d_axis0_expanded_ver18/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 6) |
| node/test_layer_normalization_4d_axis1/model.onnx | 17 | Data/Data | ✅ | OK (max ULP 1) |
| node/test_layer_normalization_4d_axis1_expanded/model.onnx | 17 | Data/Data | ✅ | OK (max ULP 12) |
| node/test_layer_normalization_4d_axis1_expanded_ver18/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 12) |
| node/test_layer_normalization_4d_axis2/model.onnx | 17 | Data/Data | ✅ | OK (max ULP 3) |
| node/test_layer_normalization_4d_axis2_expanded/model.onnx | 17 | Data/Data | ✅ | OK (max ULP 8) |
| node/test_layer_normalization_4d_axis2_expanded_ver18/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 8) |
| node/test_layer_normalization_4d_axis3/model.onnx | 17 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_layer_normalization_4d_axis3_expanded/model.onnx | 17 | Data/Data | ✅ | OK (max ULP 12) |
| node/test_layer_normalization_4d_axis3_expanded_ver18/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 12) |
| node/test_layer_normalization_4d_axis_negative_1/model.onnx | 17 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_layer_normalization_4d_axis_negative_1_expanded/model.onnx | 17 | Data/Data | ✅ | OK (max ULP 24) |
| node/test_layer_normalization_4d_axis_negative_1_expanded_ver18/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 24) |
| node/test_layer_normalization_4d_axis_negative_2/model.onnx | 17 | Data/Data | ✅ | OK (max ULP 12) |
| node/test_layer_normalization_4d_axis_negative_2_expanded/model.onnx | 17 | Data/Data | ✅ | OK (max ULP 24) |
| node/test_layer_normalization_4d_axis_negative_2_expanded_ver18/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 24) |
| node/test_layer_normalization_4d_axis_negative_3/model.onnx | 17 | Data/Data | ✅ | OK (max ULP 4) |
| node/test_layer_normalization_4d_axis_negative_3_expanded/model.onnx | 17 | Data/Data | ✅ | OK (max ULP 48) |
| node/test_layer_normalization_4d_axis_negative_3_expanded_ver18/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 48) |
| node/test_layer_normalization_4d_axis_negative_4/model.onnx | 17 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_layer_normalization_4d_axis_negative_4_expanded/model.onnx | 17 | Data/Data | ✅ | OK (max ULP 6) |
| node/test_layer_normalization_4d_axis_negative_4_expanded_ver18/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 6) |
| node/test_layer_normalization_default_axis/model.onnx | 17 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_layer_normalization_default_axis_expanded/model.onnx | 17 | Data/Data | ✅ | OK (max ULP 6) |
| node/test_layer_normalization_default_axis_expanded_ver18/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 6) |
| node/test_leakyrelu/model.onnx | 16 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_leakyrelu_default/model.onnx | 16 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_leakyrelu_default_expanded/model.onnx | 16 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_leakyrelu_example/model.onnx | 16 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_leakyrelu_example_expanded/model.onnx | 16 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_leakyrelu_expanded/model.onnx | 16 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_less/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_less_bcast/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_less_equal/model.onnx | 16 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_less_equal_bcast/model.onnx | 16 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_less_equal_bcast_expanded/model.onnx | 16 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_less_equal_expanded/model.onnx | 16 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_less_equal_int16/model.onnx | 16 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_less_equal_int16_expanded/model.onnx | 16 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_less_equal_int8/model.onnx | 16 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_less_equal_int8_expanded/model.onnx | 16 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_less_equal_uint16/model.onnx | 16 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_less_equal_uint16_expanded/model.onnx | 16 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_less_equal_uint32/model.onnx | 16 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_less_equal_uint32_expanded/model.onnx | 16 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_less_equal_uint64/model.onnx | 16 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_less_equal_uint64_expanded/model.onnx | 16 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_less_equal_uint8/model.onnx | 16 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_less_equal_uint8_expanded/model.onnx | 16 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_less_int16/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_less_int8/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_less_uint16/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_less_uint32/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_less_uint64/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_less_uint8/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_log/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_log_example/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_logsoftmax_axis_0/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_logsoftmax_axis_0_expanded/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_logsoftmax_axis_0_expanded_ver18/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_logsoftmax_axis_1/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 3) |
| node/test_logsoftmax_axis_1_expanded/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 3) |
| node/test_logsoftmax_axis_1_expanded_ver18/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 3) |
| node/test_logsoftmax_axis_2/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 1) |
| node/test_logsoftmax_axis_2_expanded/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 1) |
| node/test_logsoftmax_axis_2_expanded_ver18/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 1) |
| node/test_logsoftmax_default_axis/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 1) |
| node/test_logsoftmax_default_axis_expanded/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 1) |
| node/test_logsoftmax_default_axis_expanded_ver18/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 1) |
| node/test_logsoftmax_example_1/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 1) |
| node/test_logsoftmax_example_1_expanded/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 1) |
| node/test_logsoftmax_example_1_expanded_ver18/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 1) |
| node/test_logsoftmax_large_number/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_logsoftmax_large_number_expanded/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_logsoftmax_large_number_expanded_ver18/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_logsoftmax_negative_axis/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 1) |
| node/test_logsoftmax_negative_axis_expanded/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 1) |
| node/test_logsoftmax_negative_axis_expanded_ver18/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 1) |
| node/test_loop11/model.onnx | 11 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_loop13_seq/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_loop16_seq_none/model.onnx | 16 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_lpnormalization_default/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_lppool_1d_default/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_lppool_2d_default/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_lppool_2d_dilations/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_lppool_2d_pads/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 1) |
| node/test_lppool_2d_same_lower/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_lppool_2d_same_upper/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_lppool_2d_strides/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_lppool_3d_default/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_lrn/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_lrn_default/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_lstm_batchwise/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_lstm_defaults/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_lstm_with_initial_bias/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_lstm_with_peepholes/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_matmul_1d_1d/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_matmul_1d_3d/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_matmul_2d/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 1) |
| node/test_matmul_3d/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_matmul_4d/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 1) |
| node/test_matmul_4d_1d/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_matmul_bcast/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_matmulinteger/model.onnx | 10 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_max_example/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_max_float16/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_max_float32/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_max_float64/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_max_int16/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_max_int32/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_max_int64/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_max_int8/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_max_one_input/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_max_two_inputs/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_max_uint16/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_max_uint32/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_max_uint64/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_max_uint8/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_maxpool_1d_default/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_maxpool_2d_ceil/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_maxpool_2d_ceil_output_size_reduce_by_one/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_maxpool_2d_default/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_maxpool_2d_dilations/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_maxpool_2d_pads/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_maxpool_2d_precomputed_pads/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_maxpool_2d_precomputed_same_upper/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_maxpool_2d_precomputed_strides/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_maxpool_2d_same_lower/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_maxpool_2d_same_upper/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_maxpool_2d_strides/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_maxpool_2d_uint8/model.onnx | 22 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_maxpool_3d_default/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_maxpool_3d_dilations/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_maxpool_3d_dilations_use_ref_impl/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_maxpool_3d_dilations_use_ref_impl_large/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_maxpool_with_argmax_2d_precomputed_pads/model.onnx | 22 | Data/Data | ✅ | OK (max abs diff 0, max ULP 0) |
| node/test_maxpool_with_argmax_2d_precomputed_strides/model.onnx | 22 | Data/Data | ✅ | OK (max abs diff 0, max ULP 0) |
| node/test_maxunpool_export_with_output_shape/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_maxunpool_export_without_output_shape/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_mean_example/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_mean_one_input/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_mean_two_inputs/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_melweightmatrix/model.onnx | 17 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_min_example/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_min_float16/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_min_float32/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_min_float64/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_min_int16/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_min_int32/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_min_int64/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_min_int8/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_min_one_input/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_min_two_inputs/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_min_uint16/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_min_uint32/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_min_uint64/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_min_uint8/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_mish/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 1) |
| node/test_mish_expanded/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 1) |
| node/test_mod_broadcast/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_mod_int64_fmod/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_mod_mixed_sign_float16/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_mod_mixed_sign_float32/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_mod_mixed_sign_float64/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_mod_mixed_sign_int16/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_mod_mixed_sign_int32/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_mod_mixed_sign_int64/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_mod_mixed_sign_int8/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_mod_uint16/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_mod_uint32/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_mod_uint64/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_mod_uint8/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_momentum/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| node/test_momentum_multiple/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| node/test_mul/model.onnx | 14 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_mul_bcast/model.onnx | 14 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_mul_example/model.onnx | 14 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_mul_int16/model.onnx | 14 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_mul_int8/model.onnx | 14 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_mul_uint16/model.onnx | 14 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_mul_uint32/model.onnx | 14 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_mul_uint64/model.onnx | 14 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_mul_uint8/model.onnx | 14 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_mvn/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 3) |
| node/test_mvn_expanded/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_mvn_expanded_ver18/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_neg/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_neg_example/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_nesterov_momentum/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| node/test_nllloss_NC/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_nllloss_NC_expanded/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_nllloss_NCd1/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_nllloss_NCd1_expanded/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_nllloss_NCd1_ii/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_nllloss_NCd1_ii_expanded/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_nllloss_NCd1_mean_weight_negative_ii/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_nllloss_NCd1_mean_weight_negative_ii_expanded/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_nllloss_NCd1_weight/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_nllloss_NCd1_weight_expanded/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_nllloss_NCd1_weight_ii/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_nllloss_NCd1_weight_ii_expanded/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_nllloss_NCd1d2/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_nllloss_NCd1d2_expanded/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_nllloss_NCd1d2_no_weight_reduction_mean_ii/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_nllloss_NCd1d2_no_weight_reduction_mean_ii_expanded/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_nllloss_NCd1d2_reduction_mean/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_nllloss_NCd1d2_reduction_mean_expanded/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_nllloss_NCd1d2_reduction_sum/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 3) |
| node/test_nllloss_NCd1d2_reduction_sum_expanded/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 3) |
| node/test_nllloss_NCd1d2_with_weight/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_nllloss_NCd1d2_with_weight_expanded/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_nllloss_NCd1d2_with_weight_reduction_mean/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_nllloss_NCd1d2_with_weight_reduction_mean_expanded/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_nllloss_NCd1d2_with_weight_reduction_sum/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 1) |
| node/test_nllloss_NCd1d2_with_weight_reduction_sum_expanded/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 1) |
| node/test_nllloss_NCd1d2_with_weight_reduction_sum_ii/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 3) |
| node/test_nllloss_NCd1d2_with_weight_reduction_sum_ii_expanded/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 3) |
| node/test_nllloss_NCd1d2d3_none_no_weight_negative_ii/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_nllloss_NCd1d2d3_none_no_weight_negative_ii_expanded/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_nllloss_NCd1d2d3_sum_weight_high_ii/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_nllloss_NCd1d2d3_sum_weight_high_ii_expanded/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_nllloss_NCd1d2d3d4d5_mean_weight/model.onnx (--fp32-accumulation-strategy fp64) | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_nllloss_NCd1d2d3d4d5_mean_weight_expanded/model.onnx (--fp32-accumulation-strategy fp64) | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_nllloss_NCd1d2d3d4d5_none_no_weight/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_nllloss_NCd1d2d3d4d5_none_no_weight_expanded/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_nonmaxsuppression_center_point_box_format/model.onnx | 11 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_nonmaxsuppression_flipped_coordinates/model.onnx | 11 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_nonmaxsuppression_identical_boxes/model.onnx | 11 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_nonmaxsuppression_limit_output_size/model.onnx | 11 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_nonmaxsuppression_single_box/model.onnx | 11 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_nonmaxsuppression_suppress_by_IOU/model.onnx | 11 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_nonmaxsuppression_suppress_by_IOU_and_scores/model.onnx | 11 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_nonmaxsuppression_two_batches/model.onnx | 11 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_nonmaxsuppression_two_classes/model.onnx | 11 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_nonzero_example/model.onnx | 13 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_not_2d/model.onnx | 1 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_not_3d/model.onnx | 1 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_not_4d/model.onnx | 1 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_onehot_negative_indices/model.onnx | 11 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_onehot_with_axis/model.onnx | 11 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_onehot_with_negative_axis/model.onnx | 11 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_onehot_without_axis/model.onnx | 11 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_optional_get_element_optional_sequence/model.onnx | 18 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_optional_get_element_optional_tensor/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_optional_get_element_sequence/model.onnx | 18 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_optional_get_element_tensor/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_optional_has_element_empty_no_input_name_optional_input/model.onnx | 18 | Random/ORT | ✅ | OK (max abs diff 0) |
| node/test_optional_has_element_empty_no_input_name_tensor_input/model.onnx | 18 | Random/ORT | ✅ | OK (max abs diff 0) |
| node/test_optional_has_element_empty_no_input_optional_input/model.onnx | 18 | Random/ORT | ✅ | OK (max abs diff 0) |
| node/test_optional_has_element_empty_no_input_tensor_input/model.onnx | 18 | Random/ORT | ✅ | OK (max abs diff 0) |
| node/test_optional_has_element_empty_optional_input/model.onnx | 18 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_optional_has_element_optional_input/model.onnx | 18 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_optional_has_element_tensor_input/model.onnx | 18 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_or2d/model.onnx | 7 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_or3d/model.onnx | 7 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_or4d/model.onnx | 7 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_or_bcast3v1d/model.onnx | 7 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_or_bcast3v2d/model.onnx | 7 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_or_bcast4v2d/model.onnx | 7 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_or_bcast4v3d/model.onnx | 7 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_or_bcast4v4d/model.onnx | 7 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_pow/model.onnx | 15 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_pow_bcast_array/model.onnx | 15 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_pow_bcast_scalar/model.onnx | 15 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_pow_example/model.onnx | 15 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_pow_types_float32_int32/model.onnx | 15 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_pow_types_float32_int64/model.onnx | 15 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_pow_types_float32_uint32/model.onnx | 15 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_pow_types_float32_uint64/model.onnx | 15 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_pow_types_int32_float32/model.onnx | 15 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_pow_types_int32_int32/model.onnx | 15 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_pow_types_int64_float32/model.onnx | 15 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_pow_types_int64_int64/model.onnx | 15 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_prelu_broadcast/model.onnx | 16 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_prelu_broadcast_expanded/model.onnx | 16 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_prelu_example/model.onnx | 16 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_prelu_example_expanded/model.onnx | 16 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_qlinearconv/model.onnx | 10 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_qlinearmatmul_2D_int8_float16/model.onnx | 21 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_qlinearmatmul_2D_int8_float32/model.onnx | 21 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_qlinearmatmul_2D_uint8_float16/model.onnx | 21 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_qlinearmatmul_2D_uint8_float32/model.onnx | 21 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_qlinearmatmul_3D_int8_float16/model.onnx | 21 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_qlinearmatmul_3D_int8_float32/model.onnx | 21 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_qlinearmatmul_3D_uint8_float16/model.onnx | 21 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_qlinearmatmul_3D_uint8_float32/model.onnx | 21 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_quantizelinear/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_quantizelinear_axis/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_quantizelinear_blocked_asymmetric/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_quantizelinear_blocked_symmetric/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_quantizelinear_e4m3fn/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_quantizelinear_e5m2/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_quantizelinear_float4e2m1/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_quantizelinear_int16/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_quantizelinear_int2/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_quantizelinear_int4/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_quantizelinear_uint16/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_quantizelinear_uint2/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_quantizelinear_uint4/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_range_float_type_positive_delta/model.onnx | 11 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_range_float_type_positive_delta_expanded/model.onnx | 11 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_range_int32_type_negative_delta/model.onnx | 11 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_range_int32_type_negative_delta_expanded/model.onnx | 11 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_reciprocal/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reciprocal_example/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_l1_default_axes_keepdims_example/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_l1_default_axes_keepdims_example_expanded/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_l1_default_axes_keepdims_random/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 1) |
| node/test_reduce_l1_default_axes_keepdims_random_expanded/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 1) |
| node/test_reduce_l1_do_not_keepdims_example/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_l1_do_not_keepdims_example_expanded/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_l1_do_not_keepdims_random/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_l1_do_not_keepdims_random_expanded/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_l1_empty_set/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_l1_empty_set_expanded/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_l1_keep_dims_example/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_l1_keep_dims_example_expanded/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_l1_keep_dims_random/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_l1_keep_dims_random_expanded/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_l1_negative_axes_keep_dims_example/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_l1_negative_axes_keep_dims_example_expanded/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_l1_negative_axes_keep_dims_random/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_l1_negative_axes_keep_dims_random_expanded/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_l2_default_axes_keepdims_example/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_l2_default_axes_keepdims_example_expanded/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_l2_default_axes_keepdims_random/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_l2_default_axes_keepdims_random_expanded/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_l2_do_not_keepdims_example/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_l2_do_not_keepdims_example_expanded/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_l2_do_not_keepdims_random/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_l2_do_not_keepdims_random_expanded/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_l2_empty_set/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_l2_empty_set_expanded/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_l2_keep_dims_example/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_l2_keep_dims_example_expanded/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_l2_keep_dims_random/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_l2_keep_dims_random_expanded/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_l2_negative_axes_keep_dims_example/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_l2_negative_axes_keep_dims_example_expanded/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_l2_negative_axes_keep_dims_random/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_l2_negative_axes_keep_dims_random_expanded/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_log_sum_asc_axes/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_log_sum_asc_axes_expanded/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_log_sum_default/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 1) |
| node/test_reduce_log_sum_default_expanded/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 1) |
| node/test_reduce_log_sum_desc_axes/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_log_sum_desc_axes_expanded/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_log_sum_empty_set/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_log_sum_empty_set_expanded/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_log_sum_exp_default_axes_keepdims_example/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_log_sum_exp_default_axes_keepdims_example_expanded/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_log_sum_exp_default_axes_keepdims_random/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_log_sum_exp_default_axes_keepdims_random_expanded/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_log_sum_exp_do_not_keepdims_example/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_log_sum_exp_do_not_keepdims_example_expanded/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_log_sum_exp_do_not_keepdims_random/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_log_sum_exp_do_not_keepdims_random_expanded/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_log_sum_exp_empty_set/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_log_sum_exp_empty_set_expanded/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_log_sum_exp_keepdims_example/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_log_sum_exp_keepdims_example_expanded/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_log_sum_exp_keepdims_random/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_log_sum_exp_keepdims_random_expanded/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_log_sum_exp_negative_axes_keepdims_example/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_log_sum_exp_negative_axes_keepdims_example_expanded/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_log_sum_exp_negative_axes_keepdims_random/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_log_sum_exp_negative_axes_keepdims_random_expanded/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_log_sum_negative_axes/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_log_sum_negative_axes_expanded/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_max_bool_inputs/model.onnx | 20 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_reduce_max_default_axes_keepdim_example/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_max_default_axes_keepdims_random/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_max_do_not_keepdims_example/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_max_do_not_keepdims_random/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_max_empty_set/model.onnx | 20 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_max_keepdims_example/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_max_keepdims_random/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_max_negative_axes_keepdims_example/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_max_negative_axes_keepdims_random/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_mean_default_axes_keepdims_example/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_mean_default_axes_keepdims_random/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_reduce_mean_do_not_keepdims_example/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_mean_do_not_keepdims_random/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_mean_keepdims_example/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_mean_keepdims_random/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_mean_negative_axes_keepdims_example/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_mean_negative_axes_keepdims_random/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_min_bool_inputs/model.onnx | 20 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_reduce_min_default_axes_keepdims_example/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_min_default_axes_keepdims_random/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_min_do_not_keepdims_example/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_min_do_not_keepdims_random/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_min_empty_set/model.onnx | 20 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_min_keepdims_example/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_min_keepdims_random/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_min_negative_axes_keepdims_example/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_min_negative_axes_keepdims_random/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_prod_default_axes_keepdims_example/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_prod_default_axes_keepdims_random/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_prod_do_not_keepdims_example/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_prod_do_not_keepdims_random/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_prod_empty_set/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_prod_keepdims_example/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_prod_keepdims_random/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_prod_negative_axes_keepdims_example/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_prod_negative_axes_keepdims_random/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_sum_default_axes_keepdims_example/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_sum_default_axes_keepdims_random/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_reduce_sum_do_not_keepdims_example/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_sum_do_not_keepdims_random/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_sum_empty_axes_input_noop/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_sum_empty_axes_input_noop_example/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_sum_empty_set/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_sum_empty_set_non_reduced_axis_zero/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_sum_keepdims_example/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_sum_keepdims_random/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_sum_negative_axes_keepdims_example/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_sum_negative_axes_keepdims_random/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_sum_square_default_axes_keepdims_example/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_sum_square_default_axes_keepdims_example_expanded/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_sum_square_default_axes_keepdims_random/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_sum_square_default_axes_keepdims_random_expanded/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_sum_square_do_not_keepdims_example/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_sum_square_do_not_keepdims_example_expanded/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_sum_square_do_not_keepdims_random/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_sum_square_do_not_keepdims_random_expanded/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_sum_square_empty_set/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_sum_square_empty_set_expanded/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_sum_square_keepdims_example/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_sum_square_keepdims_example_expanded/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_sum_square_keepdims_random/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_sum_square_keepdims_random_expanded/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_sum_square_negative_axes_keepdims_example/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_sum_square_negative_axes_keepdims_example_expanded/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_sum_square_negative_axes_keepdims_random/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reduce_sum_square_negative_axes_keepdims_random_expanded/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reflect_pad/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_regex_full_match_basic/model.onnx | 20 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_regex_full_match_email_domain/model.onnx | 20 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_regex_full_match_empty/model.onnx | 20 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_relu/model.onnx | 14 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_relu_expanded_ver18/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reshape_allowzero_reordered/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reshape_extended_dims/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reshape_negative_dim/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reshape_negative_extended_dims/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reshape_one_dim/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reshape_reduced_dims/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reshape_reordered_all_dims/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reshape_reordered_last_dims/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reshape_zero_and_negative_dim/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reshape_zero_dim/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_resize_downsample_scales_cubic/model.onnx | 19 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_resize_downsample_scales_cubic_A_n0p5_exclude_outside/model.onnx | 19 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_resize_downsample_scales_cubic_align_corners/model.onnx | 19 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_resize_downsample_scales_cubic_antialias/model.onnx | 19 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_resize_downsample_scales_linear/model.onnx | 19 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_resize_downsample_scales_linear_align_corners/model.onnx | 19 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_resize_downsample_scales_linear_antialias/model.onnx | 19 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_resize_downsample_scales_linear_half_pixel_symmetric/model.onnx | 19 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_resize_downsample_scales_nearest/model.onnx | 19 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_resize_downsample_sizes_cubic/model.onnx | 19 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_resize_downsample_sizes_cubic_antialias/model.onnx | 19 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_resize_downsample_sizes_linear_antialias/model.onnx | 19 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_resize_downsample_sizes_linear_pytorch_half_pixel/model.onnx | 19 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_resize_downsample_sizes_nearest/model.onnx | 19 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_resize_downsample_sizes_nearest_not_larger/model.onnx | 19 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_resize_downsample_sizes_nearest_not_smaller/model.onnx | 19 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_resize_tf_crop_and_resize/model.onnx | 19 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_resize_tf_crop_and_resize_axes_2_3/model.onnx | 19 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_resize_tf_crop_and_resize_axes_3_2/model.onnx | 19 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_resize_tf_crop_and_resize_extrapolation_value/model.onnx | 19 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_resize_upsample_scales_cubic/model.onnx | 19 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_resize_upsample_scales_cubic_A_n0p5_exclude_outside/model.onnx | 19 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_resize_upsample_scales_cubic_align_corners/model.onnx | 19 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_resize_upsample_scales_cubic_asymmetric/model.onnx | 19 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_resize_upsample_scales_linear/model.onnx | 19 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_resize_upsample_scales_linear_align_corners/model.onnx | 19 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_resize_upsample_scales_linear_half_pixel_symmetric/model.onnx | 19 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_resize_upsample_scales_nearest/model.onnx | 19 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_resize_upsample_scales_nearest_axes_2_3/model.onnx | 19 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_resize_upsample_scales_nearest_axes_3_2/model.onnx | 19 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_resize_upsample_sizes_cubic/model.onnx | 19 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_resize_upsample_sizes_nearest/model.onnx | 19 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_resize_upsample_sizes_nearest_axes_2_3/model.onnx | 19 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_resize_upsample_sizes_nearest_axes_3_2/model.onnx | 19 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_resize_upsample_sizes_nearest_ceil_half_pixel/model.onnx | 19 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_resize_upsample_sizes_nearest_floor_align_corners/model.onnx | 19 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_resize_upsample_sizes_nearest_not_larger/model.onnx | 19 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_resize_upsample_sizes_nearest_not_smaller/model.onnx | 19 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_resize_upsample_sizes_nearest_round_prefer_ceil_asymmetric/model.onnx | 19 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reversesequence_batch/model.onnx | 10 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_reversesequence_time/model.onnx | 10 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_rms_normalization_2d_axis0/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_rms_normalization_2d_axis0_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_rms_normalization_2d_axis1/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_rms_normalization_2d_axis1_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_rms_normalization_2d_axis_negative_1/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_rms_normalization_2d_axis_negative_1_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_rms_normalization_2d_axis_negative_2/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_rms_normalization_2d_axis_negative_2_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_rms_normalization_3d_axis0_epsilon/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_rms_normalization_3d_axis0_epsilon_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_rms_normalization_3d_axis1_epsilon/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_rms_normalization_3d_axis1_epsilon_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_rms_normalization_3d_axis2_epsilon/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_rms_normalization_3d_axis2_epsilon_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_rms_normalization_3d_axis_negative_1_epsilon/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_rms_normalization_3d_axis_negative_1_epsilon_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_rms_normalization_3d_axis_negative_2_epsilon/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_rms_normalization_3d_axis_negative_2_epsilon_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_rms_normalization_3d_axis_negative_3_epsilon/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_rms_normalization_3d_axis_negative_3_epsilon_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_rms_normalization_4d_axis0/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 1) |
| node/test_rms_normalization_4d_axis0_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 1) |
| node/test_rms_normalization_4d_axis1/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_rms_normalization_4d_axis1_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_rms_normalization_4d_axis2/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_rms_normalization_4d_axis2_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_rms_normalization_4d_axis3/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 1) |
| node/test_rms_normalization_4d_axis3_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 1) |
| node/test_rms_normalization_4d_axis_negative_1/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_rms_normalization_4d_axis_negative_1_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_rms_normalization_4d_axis_negative_2/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 1) |
| node/test_rms_normalization_4d_axis_negative_2_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 1) |
| node/test_rms_normalization_4d_axis_negative_3/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_rms_normalization_4d_axis_negative_3_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_rms_normalization_4d_axis_negative_4/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_rms_normalization_4d_axis_negative_4_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_rms_normalization_default_axis/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_rms_normalization_default_axis_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_rnn_seq_length/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_roialign_aligned_false/model.onnx (--runtime onnx-reference --test-data-inputs-only) | 22 | Data/ONNXRef | ✅ | OK (max ULP 10) |
| node/test_roialign_aligned_true/model.onnx (--runtime onnx-reference --test-data-inputs-only) | 22 | Data/ONNXRef | ✅ | OK (max ULP 11) |
| node/test_roialign_mode_max/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 36) |
| node/test_rotary_embedding/model.onnx | 23 | Data/Data | ❌ | RotaryEmbedding inputs must share the same dtype |
| node/test_rotary_embedding_3d_input/model.onnx | 23 | Data/Data | ❌ | RotaryEmbedding inputs must share the same dtype |
| node/test_rotary_embedding_3d_input_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_rotary_embedding_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_rotary_embedding_interleaved/model.onnx | 23 | Data/Data | ❌ | RotaryEmbedding inputs must share the same dtype |
| node/test_rotary_embedding_interleaved_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_rotary_embedding_no_position_ids/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_rotary_embedding_no_position_ids_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_rotary_embedding_no_position_ids_interleaved/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_rotary_embedding_no_position_ids_interleaved_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_rotary_embedding_no_position_ids_rotary_dim/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_rotary_embedding_no_position_ids_rotary_dim_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_rotary_embedding_with_interleaved_rotary_dim/model.onnx | 23 | Data/Data | ❌ | RotaryEmbedding inputs must share the same dtype |
| node/test_rotary_embedding_with_interleaved_rotary_dim_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_rotary_embedding_with_rotary_dim/model.onnx | 23 | Data/Data | ❌ | RotaryEmbedding inputs must share the same dtype |
| node/test_rotary_embedding_with_rotary_dim_expanded/model.onnx | 23 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_round/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_scan9_sum/model.onnx | 9 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_scan_sum/model.onnx | 8 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_scatter_elements_with_axis/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_scatter_elements_with_duplicate_indices/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_scatter_elements_with_negative_indices/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_scatter_elements_with_reduction_max/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_scatter_elements_with_reduction_min/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_scatter_elements_without_axis/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_scatter_with_axis/model.onnx | 10 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_scatter_without_axis/model.onnx | 10 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_scatternd/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_scatternd_add/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_scatternd_max/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_scatternd_min/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_scatternd_multiply/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_sce_NCd1_mean_weight_negative_ii/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_sce_NCd1_mean_weight_negative_ii_expanded/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_sce_NCd1_mean_weight_negative_ii_log_prob/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_sce_NCd1_mean_weight_negative_ii_log_prob_expanded/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_sce_NCd1d2d3_none_no_weight_negative_ii/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_sce_NCd1d2d3_none_no_weight_negative_ii_expanded/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_sce_NCd1d2d3_none_no_weight_negative_ii_log_prob/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_sce_NCd1d2d3_none_no_weight_negative_ii_log_prob_expanded/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_sce_NCd1d2d3_sum_weight_high_ii/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_sce_NCd1d2d3_sum_weight_high_ii_expanded/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_sce_NCd1d2d3_sum_weight_high_ii_log_prob/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_sce_NCd1d2d3_sum_weight_high_ii_log_prob_expanded/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_sce_NCd1d2d3d4d5_mean_weight/model.onnx (--fp32-accumulation-strategy fp64) | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_sce_NCd1d2d3d4d5_mean_weight_expanded/model.onnx (--fp32-accumulation-strategy fp64) | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_sce_NCd1d2d3d4d5_mean_weight_log_prob/model.onnx (--fp32-accumulation-strategy fp64) | 13 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_sce_NCd1d2d3d4d5_mean_weight_log_prob_expanded/model.onnx (--fp32-accumulation-strategy fp64) | 13 | Data/Data | ✅ | OK (max ULP 3) |
| node/test_sce_NCd1d2d3d4d5_none_no_weight/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 3) |
| node/test_sce_NCd1d2d3d4d5_none_no_weight_expanded/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 3) |
| node/test_sce_NCd1d2d3d4d5_none_no_weight_log_prob/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 3) |
| node/test_sce_NCd1d2d3d4d5_none_no_weight_log_prob_expanded/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 3) |
| node/test_sce_mean/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_sce_mean_3d/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_sce_mean_3d_expanded/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_sce_mean_3d_log_prob/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_sce_mean_3d_log_prob_expanded/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_sce_mean_expanded/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_sce_mean_log_prob/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_sce_mean_log_prob_expanded/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_sce_mean_no_weight_ii/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_sce_mean_no_weight_ii_3d/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_sce_mean_no_weight_ii_3d_expanded/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_sce_mean_no_weight_ii_3d_log_prob/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_sce_mean_no_weight_ii_3d_log_prob_expanded/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_sce_mean_no_weight_ii_4d/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_sce_mean_no_weight_ii_4d_expanded/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_sce_mean_no_weight_ii_4d_log_prob/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_sce_mean_no_weight_ii_4d_log_prob_expanded/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_sce_mean_no_weight_ii_expanded/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_sce_mean_no_weight_ii_log_prob/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_sce_mean_no_weight_ii_log_prob_expanded/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_sce_mean_weight/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_sce_mean_weight_expanded/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_sce_mean_weight_ii/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_sce_mean_weight_ii_3d/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_sce_mean_weight_ii_3d_expanded/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_sce_mean_weight_ii_3d_log_prob/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_sce_mean_weight_ii_3d_log_prob_expanded/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_sce_mean_weight_ii_4d/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_sce_mean_weight_ii_4d_expanded/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_sce_mean_weight_ii_4d_log_prob/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_sce_mean_weight_ii_4d_log_prob_expanded/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_sce_mean_weight_ii_expanded/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_sce_mean_weight_ii_log_prob/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_sce_mean_weight_ii_log_prob_expanded/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_sce_mean_weight_log_prob/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_sce_mean_weight_log_prob_expanded/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_sce_none/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_sce_none_expanded/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_sce_none_log_prob/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_sce_none_log_prob_expanded/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_sce_none_weights/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_sce_none_weights_expanded/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_sce_none_weights_log_prob/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_sce_none_weights_log_prob_expanded/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_sce_sum/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 1) |
| node/test_sce_sum_expanded/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 1) |
| node/test_sce_sum_log_prob/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_sce_sum_log_prob_expanded/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_selu/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 24) |
| node/test_selu_default/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 5) |
| node/test_selu_default_expanded_ver18/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 43) |
| node/test_selu_example/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_selu_example_expanded_ver18/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 2) |
| node/test_selu_expanded_ver18/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 24) |
| node/test_sequence_insert_at_back/model.onnx (--sequence-element-shape sequence=[<=4]) | 11 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_sequence_insert_at_front/model.onnx (--sequence-element-shape sequence=[<=4]) | 11 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_sequence_map_add_1_sequence_1_tensor/model.onnx (--sequence-element-shape x0=[10]) | 17 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_sequence_map_add_1_sequence_1_tensor_expanded/model.onnx (--sequence-element-shape x0=[10]) | 17 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_sequence_map_add_2_sequences/model.onnx (--sequence-element-shape x0=[<=6] --sequence-element-shape x1=[<=6]) | 17 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_sequence_map_add_2_sequences_expanded/model.onnx (--sequence-element-shape x0=[<=6] --sequence-element-shape x1=[<=6]) | 17 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_sequence_map_extract_shapes/model.onnx (--sequence-element-shape in_seq=[<=40,<=30,3]) | 17 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_sequence_map_extract_shapes_expanded/model.onnx (--sequence-element-shape in_seq=[<=40,<=30,3]) | 17 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_sequence_map_identity_1_sequence/model.onnx (--sequence-element-shape x=[10]) | 17 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_sequence_map_identity_1_sequence_1_tensor/model.onnx (--sequence-element-shape x0=[<=9]) | 17 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_sequence_map_identity_1_sequence_1_tensor_expanded/model.onnx (--sequence-element-shape x0=[<=9]) | 17 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_sequence_map_identity_1_sequence_expanded/model.onnx (--sequence-element-shape x=[10]) | 17 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_sequence_map_identity_2_sequences/model.onnx (--sequence-element-shape x0=[<=9] --sequence-element-shape x1=[<=8]) | 17 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_sequence_map_identity_2_sequences_expanded/model.onnx (--sequence-element-shape x0=[<=9] --sequence-element-shape x1=[<=8]) | 17 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_shape/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_shape_clip_end/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_shape_clip_start/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_shape_end_1/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_shape_end_negative_1/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_shape_example/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_shape_start_1/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_shape_start_1_end_2/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_shape_start_1_end_negative_1/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_shape_start_greater_than_end/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_shape_start_negative_1/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_shrink_hard/model.onnx | 9 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_shrink_hard_expanded_ver18/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_shrink_soft/model.onnx | 9 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_shrink_soft_expanded_ver18/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_sigmoid/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_sigmoid_example/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_sign/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_simple_rnn_batchwise/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_simple_rnn_defaults/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_simple_rnn_with_initial_bias/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_sin/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_sin_example/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_sinh/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_sinh_example/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_size/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_size_example/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_slice/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_slice_default_axes/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_slice_default_steps/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_slice_end_out_of_bounds/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_slice_neg/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_slice_neg_steps/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_slice_negative_axes/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_slice_start_out_of_bounds/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_softmax_axis_0/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_softmax_axis_0_expanded/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_softmax_axis_0_expanded_ver18/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_softmax_axis_1/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_softmax_axis_1_expanded/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_softmax_axis_1_expanded_ver18/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_softmax_axis_2/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_softmax_axis_2_expanded/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_softmax_axis_2_expanded_ver18/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_softmax_default_axis/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_softmax_default_axis_expanded/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_softmax_default_axis_expanded_ver18/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_softmax_example/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_softmax_example_expanded/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_softmax_example_expanded_ver18/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_softmax_large_number/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_softmax_large_number_expanded/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_softmax_large_number_expanded_ver18/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_softmax_negative_axis/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_softmax_negative_axis_expanded/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_softmax_negative_axis_expanded_ver18/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_softplus/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_softplus_example/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_softplus_example_expanded_ver18/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_softplus_expanded_ver18/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 1) |
| node/test_softsign/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_softsign_example/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_softsign_example_expanded_ver18/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_softsign_expanded_ver18/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_spacetodepth/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_spacetodepth_example/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_split_1d_uneven_split_opset18/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_split_2d_uneven_split_opset18/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_split_equal_parts_1d_opset13/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_split_equal_parts_1d_opset18/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_split_equal_parts_2d/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_split_equal_parts_2d_opset13/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_split_equal_parts_default_axis_opset13/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_split_equal_parts_default_axis_opset18/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_split_to_sequence_1/model.onnx | 24 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_split_to_sequence_2/model.onnx | 24 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_split_to_sequence_nokeepdims/model.onnx | 24 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_split_variable_parts_1d_opset13/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_split_variable_parts_1d_opset18/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_split_variable_parts_2d_opset13/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_split_variable_parts_2d_opset18/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_split_variable_parts_default_axis_opset13/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_split_variable_parts_default_axis_opset18/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_split_zero_size_splits_opset13/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_split_zero_size_splits_opset18/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_sqrt/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_sqrt_example/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_squeeze/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_squeeze_negative_axes/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_stft/model.onnx | 17 | Data/Data | ✅ | OK (max ULP 15) |
| node/test_stft_with_window/model.onnx | 17 | Data/Data | ✅ | OK (max ULP 99) |
| node/test_string_concat/model.onnx | 20 | Data/Data | ✅ | OK (no numeric comparisons) |
| node/test_string_concat_broadcasting/model.onnx | 20 | Data/Data | ✅ | OK (no numeric comparisons) |
| node/test_string_concat_empty_string/model.onnx | 20 | Data/Data | ✅ | OK (no numeric comparisons) |
| node/test_string_concat_utf8/model.onnx | 20 | Data/Data | ✅ | OK (no numeric comparisons) |
| node/test_string_concat_zero_dimensional/model.onnx | 20 | Data/Data | ✅ | OK (no numeric comparisons) |
| node/test_string_split_basic/model.onnx | 20 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_string_split_consecutive_delimiters/model.onnx | 20 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_string_split_empty_string_delimiter/model.onnx | 20 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_string_split_empty_tensor/model.onnx | 20 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_string_split_maxsplit/model.onnx | 20 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_string_split_no_delimiter/model.onnx | 20 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_strnormalizer_export_monday_casesensintive_lower/model.onnx | 10 | Data/Data | ✅ | OK (no numeric comparisons) |
| node/test_strnormalizer_export_monday_casesensintive_nochangecase/model.onnx | 10 | Data/Data | ✅ | OK (no numeric comparisons) |
| node/test_strnormalizer_export_monday_casesensintive_upper/model.onnx | 10 | Data/Data | ✅ | OK (no numeric comparisons) |
| node/test_strnormalizer_export_monday_empty_output/model.onnx | 10 | Data/Data | ✅ | OK (no numeric comparisons) |
| node/test_strnormalizer_export_monday_insensintive_upper_twodim/model.onnx | 10 | Data/Data | ✅ | OK (no numeric comparisons) |
| node/test_strnormalizer_nostopwords_nochangecase/model.onnx | 10 | Data/Data | ✅ | OK (no numeric comparisons) |
| node/test_sub/model.onnx | 14 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_sub_bcast/model.onnx | 14 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_sub_example/model.onnx | 14 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_sub_int16/model.onnx | 14 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_sub_int8/model.onnx | 14 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_sub_uint16/model.onnx | 14 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_sub_uint32/model.onnx | 14 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_sub_uint64/model.onnx | 14 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_sub_uint8/model.onnx | 14 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_sum_example/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_sum_one_input/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_sum_two_inputs/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_swish/model.onnx | 24 | Data/Data | ✅ | OK (max ULP 1) |
| node/test_swish_expanded/model.onnx | 24 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_tan/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_tan_example/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_tanh/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_tanh_example/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_tensorscatter/model.onnx | 24 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_tensorscatter_3d/model.onnx | 24 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_tensorscatter_circular/model.onnx | 24 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_tfidfvectorizer_tf_batch_onlybigrams_skip0/model.onnx | 9 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_tfidfvectorizer_tf_batch_onlybigrams_skip5/model.onnx | 9 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_tfidfvectorizer_tf_batch_uniandbigrams_skip5/model.onnx | 9 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_tfidfvectorizer_tf_only_bigrams_skip0/model.onnx | 9 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_tfidfvectorizer_tf_onlybigrams_levelempty/model.onnx | 9 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_tfidfvectorizer_tf_onlybigrams_skip5/model.onnx | 9 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_tfidfvectorizer_tf_uniandbigrams_skip5/model.onnx | 9 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_thresholdedrelu/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_thresholdedrelu_default/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_thresholdedrelu_default_expanded_ver18/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_thresholdedrelu_example/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_thresholdedrelu_example_expanded_ver18/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_thresholdedrelu_expanded_ver18/model.onnx | 18 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_tile/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_tile_precomputed/model.onnx | 13 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_top_k/model.onnx | 24 | Data/Data | ✅ | OK (max abs diff 0, max ULP 0) |
| node/test_top_k_negative_axis/model.onnx | 24 | Data/Data | ✅ | OK (max abs diff 0, max ULP 0) |
| node/test_top_k_same_values/model.onnx | 24 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_top_k_same_values_2d/model.onnx | 24 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_top_k_same_values_largest/model.onnx | 24 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_top_k_smallest/model.onnx | 24 | Data/Data | ✅ | OK (max abs diff 0, max ULP 0) |
| node/test_top_k_uint64/model.onnx | 24 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_training_dropout/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_training_dropout_default/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_training_dropout_default_mask/model.onnx | 22 | Data/Data | ✅ | OK (max abs diff 0, max ULP 0) |
| node/test_training_dropout_mask/model.onnx | 22 | Data/Data | ✅ | OK (max abs diff 0, max ULP 0) |
| node/test_training_dropout_zero_ratio/model.onnx | 22 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_training_dropout_zero_ratio_mask/model.onnx | 22 | Data/Data | ✅ | OK (max abs diff 0, max ULP 0) |
| node/test_transpose_all_permutations_0/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_transpose_all_permutations_1/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_transpose_all_permutations_2/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_transpose_all_permutations_3/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_transpose_all_permutations_4/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_transpose_all_permutations_5/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_transpose_default/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_tril/model.onnx | 14 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_tril_neg/model.onnx | 14 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_tril_one_row_neg/model.onnx | 14 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_tril_out_neg/model.onnx | 14 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_tril_out_pos/model.onnx | 14 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_tril_pos/model.onnx | 14 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_tril_square/model.onnx | 14 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_tril_square_neg/model.onnx | 14 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_tril_zero/model.onnx | 14 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_triu/model.onnx | 14 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_triu_neg/model.onnx | 14 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_triu_one_row/model.onnx | 14 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_triu_out_neg_out/model.onnx | 14 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_triu_out_pos/model.onnx | 14 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_triu_pos/model.onnx | 14 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_triu_square/model.onnx | 14 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_triu_square_neg/model.onnx | 14 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_triu_zero/model.onnx | 14 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_unique_length_1/model.onnx | 11 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_unique_not_sorted_without_axis/model.onnx | 11 | Data/Data | ✅ | OK (max abs diff 0, max ULP 0) |
| node/test_unique_sorted_with_axis/model.onnx | 11 | Data/Data | ✅ | OK (max abs diff 0, max ULP 0) |
| node/test_unique_sorted_with_axis_3d/model.onnx | 11 | Data/Data | ✅ | OK (max abs diff 0, max ULP 0) |
| node/test_unique_sorted_with_negative_axis/model.onnx | 11 | Data/Data | ✅ | OK (max abs diff 0, max ULP 0) |
| node/test_unique_sorted_without_axis/model.onnx | 11 | Data/Data | ✅ | OK (max abs diff 0, max ULP 0) |
| node/test_unsqueeze_axis_0/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_unsqueeze_axis_1/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_unsqueeze_axis_2/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_unsqueeze_negative_axes/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_unsqueeze_three_axes/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_unsqueeze_two_axes/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_unsqueeze_unsorted_axes/model.onnx | 25 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_upsample_nearest/model.onnx | 9 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_where_example/model.onnx | 16 | Data/Data | ✅ | OK (max ULP 0) |
| node/test_where_long_example/model.onnx | 16 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_wrap_pad/model.onnx | 25 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_xor2d/model.onnx | 7 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_xor3d/model.onnx | 7 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_xor4d/model.onnx | 7 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_xor_bcast3v1d/model.onnx | 7 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_xor_bcast3v2d/model.onnx | 7 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_xor_bcast4v2d/model.onnx | 7 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_xor_bcast4v3d/model.onnx | 7 | Data/Data | ✅ | OK (max abs diff 0) |
| node/test_xor_bcast4v4d/model.onnx | 7 | Data/Data | ✅ | OK (max abs diff 0) |
| pytorch-converted/test_AvgPool1d/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-converted/test_AvgPool1d_stride/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-converted/test_AvgPool2d/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-converted/test_AvgPool2d_stride/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-converted/test_AvgPool3d/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-converted/test_AvgPool3d_stride/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-converted/test_AvgPool3d_stride1_pad0_gpu_input/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-converted/test_BatchNorm1d_3d_input_eval/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-converted/test_BatchNorm2d_eval/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-converted/test_BatchNorm2d_momentum_eval/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-converted/test_BatchNorm3d_eval/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-converted/test_BatchNorm3d_momentum_eval/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 2) |
| pytorch-converted/test_ConstantPad2d/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-converted/test_Conv1d/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 4) |
| pytorch-converted/test_Conv1d_dilated/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-converted/test_Conv1d_groups/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-converted/test_Conv1d_pad1/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-converted/test_Conv1d_pad1size1/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-converted/test_Conv1d_pad2/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 3) |
| pytorch-converted/test_Conv1d_pad2size1/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-converted/test_Conv1d_stride/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-converted/test_Conv2d/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-converted/test_Conv2d_depthwise/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-converted/test_Conv2d_depthwise_padded/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-converted/test_Conv2d_depthwise_strided/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-converted/test_Conv2d_depthwise_with_multiplier/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-converted/test_Conv2d_dilated/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 1) |
| pytorch-converted/test_Conv2d_groups/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-converted/test_Conv2d_groups_thnn/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 3) |
| pytorch-converted/test_Conv2d_no_bias/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-converted/test_Conv2d_padding/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-converted/test_Conv2d_strided/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 9) |
| pytorch-converted/test_Conv3d/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 4) |
| pytorch-converted/test_Conv3d_dilated/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 3) |
| pytorch-converted/test_Conv3d_dilated_strided/model.onnx (--fp32-accumulation-strategy fp64) | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-converted/test_Conv3d_groups/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 9) |
| pytorch-converted/test_Conv3d_no_bias/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-converted/test_Conv3d_stride/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 5) |
| pytorch-converted/test_Conv3d_stride_padding/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 7) |
| pytorch-converted/test_ConvTranspose2d/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-converted/test_ConvTranspose2d_no_bias/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-converted/test_ELU/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-converted/test_Embedding/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-converted/test_Embedding_sparse/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-converted/test_GLU/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-converted/test_GLU_dim/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-converted/test_LeakyReLU/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-converted/test_LeakyReLU_with_negval/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-converted/test_Linear/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-converted/test_Linear_no_bias/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-converted/test_LogSoftmax/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 2) |
| pytorch-converted/test_MaxPool1d/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-converted/test_MaxPool1d_stride/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-converted/test_MaxPool1d_stride_padding_dilation/model.onnx | 12 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-converted/test_MaxPool2d/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-converted/test_MaxPool2d_stride_padding_dilation/model.onnx | 12 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-converted/test_MaxPool3d/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-converted/test_MaxPool3d_stride/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-converted/test_MaxPool3d_stride_padding/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-converted/test_PReLU_1d/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-converted/test_PReLU_1d_multiparam/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-converted/test_PReLU_2d/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-converted/test_PReLU_2d_multiparam/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-converted/test_PReLU_3d/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-converted/test_PReLU_3d_multiparam/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-converted/test_PixelShuffle/model.onnx | 9 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-converted/test_PoissonNLLLLoss_no_reduce/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 1) |
| pytorch-converted/test_ReLU/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-converted/test_ReflectionPad2d/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-converted/test_ReplicationPad2d/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-converted/test_SELU/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-converted/test_Sigmoid/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-converted/test_Softmax/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-converted/test_Softmin/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-converted/test_Softplus/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-converted/test_Softsign/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-converted/test_Tanh/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-converted/test_ZeroPad2d/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-converted/test_log_softmax_dim3/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 1) |
| pytorch-converted/test_log_softmax_lastdim/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 1) |
| pytorch-converted/test_softmax_functional_dim3/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-converted/test_softmax_lastdim/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-operator/test_operator_add_broadcast/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-operator/test_operator_add_size1_broadcast/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-operator/test_operator_add_size1_right_broadcast/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-operator/test_operator_add_size1_singleton_broadcast/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-operator/test_operator_addconstant/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-operator/test_operator_addmm/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-operator/test_operator_basic/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-operator/test_operator_chunk/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-operator/test_operator_clip/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-operator/test_operator_concat2/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-operator/test_operator_conv/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-operator/test_operator_convtranspose/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-operator/test_operator_exp/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-operator/test_operator_flatten/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-operator/test_operator_index/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-operator/test_operator_max/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-operator/test_operator_maxpool/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-operator/test_operator_min/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-operator/test_operator_mm/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-operator/test_operator_non_float_params/model.onnx | 6 | Data/Data | ✅ | OK (max abs diff 0) |
| pytorch-operator/test_operator_pad/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-operator/test_operator_params/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-operator/test_operator_permute2/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-operator/test_operator_pow/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-operator/test_operator_reduced_mean/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-operator/test_operator_reduced_mean_keepdim/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-operator/test_operator_reduced_sum/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-operator/test_operator_reduced_sum_keepdim/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-operator/test_operator_repeat/model.onnx | 9 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-operator/test_operator_repeat_dim_overflow/model.onnx | 9 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-operator/test_operator_selu/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-operator/test_operator_sqrt/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-operator/test_operator_symbolic_override/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 9) |
| pytorch-operator/test_operator_symbolic_override_nested/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| pytorch-operator/test_operator_view/model.onnx | 6 | Data/Data | ✅ | OK (max ULP 0) |
| simple/test_expand_shape_model1/model.onnx | 9 | Data/Data | ✅ | OK (max ULP 0) |
| simple/test_expand_shape_model2/model.onnx | 9 | Data/Data | ✅ | OK (max ULP 0) |
| simple/test_expand_shape_model3/model.onnx | 9 | Data/Data | ✅ | OK (max ULP 0) |
| simple/test_expand_shape_model4/model.onnx | 9 | Data/Data | ✅ | OK (max ULP 0) |
| simple/test_gradient_of_add/model.onnx | 12 | Data/Data | ✅ | OK (max ULP 0) |
| simple/test_gradient_of_add_and_mul/model.onnx | 12 | Data/Data | ✅ | OK (max ULP 0) |
| simple/test_sequence_model1/model.onnx | 12 | Data/Data | ✅ | OK (max ULP 0) |
| simple/test_sequence_model2/model.onnx | 12 | Data/Data | ✅ | OK (max ULP 0) |
| simple/test_sequence_model3/model.onnx | 12 | Data/Data | ✅ | OK (max ULP 0) |
| simple/test_sequence_model4/model.onnx | 12 | Data/Data | ✅ | OK (max ULP 0) |
| simple/test_sequence_model5/model.onnx | 12 | Data/Data | ✅ | OK (max ULP 0) |
| simple/test_sequence_model6/model.onnx | 12 | Data/Data | ✅ | OK (max abs diff 0) |
| simple/test_sequence_model7/model.onnx | 12 | Data/Data | ✅ | OK (max ULP 0) |
| simple/test_sequence_model8/model.onnx | 12 | Data/Data | ✅ | OK (max abs diff 0) |
| simple/test_shrink/model.onnx | 10 | Data/Data | ✅ | OK (max ULP 0) |
| simple/test_sign_model/model.onnx | 9 | Data/Data | ✅ | OK (max ULP 0) |
| simple/test_single_relu_model/model.onnx | 9 | Data/Data | ✅ | OK (max ULP 0) |
| simple/test_strnorm_model_monday_casesensintive_lower/model.onnx | 10 | Data/Data | ✅ | OK (no numeric comparisons) |
| simple/test_strnorm_model_monday_casesensintive_nochangecase/model.onnx | 10 | Data/Data | ✅ | OK (no numeric comparisons) |
| simple/test_strnorm_model_monday_casesensintive_upper/model.onnx | 10 | Data/Data | ✅ | OK (no numeric comparisons) |
| simple/test_strnorm_model_monday_empty_output/model.onnx | 10 | Data/Data | ✅ | OK (no numeric comparisons) |
| simple/test_strnorm_model_monday_insensintive_upper_twodim/model.onnx | 10 | Data/Data | ✅ | OK (no numeric comparisons) |
| simple/test_strnorm_model_nostopwords_nochangecase/model.onnx | 10 | Data/Data | ✅ | OK (no numeric comparisons) |

## ONNX Runtime artifact coverage

Test directory: `emx-ort-test-artifacts-org/artifacts/onnxruntime`

Coverage 3944 / 4182 ONNX files (94.3%).

| File | Opset | Verification | Supported | Error |
| --- | --- | --- | --- | --- |
| test/contrib_ops/attention_lstm_op_test/BidirectionLstmWithBahdanauAM2BatchShortenSeqLen_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.AttnLSTM |
| test/contrib_ops/attention_lstm_op_test/BidirectionLstmWithBahdanauAMShortenSeqLength_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.AttnLSTM |
| test/contrib_ops/attention_lstm_op_test/ForwardLstmWithBahdanauAMShortenSeqLength_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.AttnLSTM |
| test/contrib_ops/attention_lstm_op_test/ForwardLstmWithBahdanauAMZeroAttention_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.AttnLSTM |
| test/contrib_ops/attention_lstm_op_test/ForwardLstmWithBahdanauAM_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.AttnLSTM |
| test/contrib_ops/attention_lstm_op_test/ReverseLstmWithBahdanauAMShortenSeqLength_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.AttnLSTM |
| test/contrib_ops/attention_op_test/Attention3DMask_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 2) |
| test/contrib_ops/attention_op_test/Attention3DMask_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 2) |
| test/contrib_ops/attention_op_test/AttentionBatch1AttentionBias_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/attention_op_test/AttentionBatch1AttentionBias_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/attention_op_test/AttentionBatch1WithQKVAttr1_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/attention_op_test/AttentionBatch1WithQKVAttr1_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/attention_op_test/AttentionBatch1WithQKVAttr2_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/attention_op_test/AttentionBatch1WithQKVAttr2_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/attention_op_test/AttentionBatch1_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 5) |
| test/contrib_ops/attention_op_test/AttentionBatch1_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 5) |
| test/contrib_ops/attention_op_test/AttentionBatch2AttentionBias_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/attention_op_test/AttentionBatch2AttentionBias_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/attention_op_test/AttentionBatch2AttentionMask_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 2) |
| test/contrib_ops/attention_op_test/AttentionBatch2AttentionMask_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 2) |
| test/contrib_ops/attention_op_test/AttentionBatch2LeftPaddingMaskIndex2_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 2) |
| test/contrib_ops/attention_op_test/AttentionBatch2LeftPaddingMaskIndex2_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 2) |
| test/contrib_ops/attention_op_test/AttentionBatch2MaskIndex2_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 5) |
| test/contrib_ops/attention_op_test/AttentionBatch2MaskIndex2_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 5) |
| test/contrib_ops/attention_op_test/AttentionBatch2_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 5) |
| test/contrib_ops/attention_op_test/AttentionBatch2_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 5) |
| test/contrib_ops/attention_op_test/AttentionDummyMask2D_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 5) |
| test/contrib_ops/attention_op_test/AttentionDummyMask2D_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 5) |
| test/contrib_ops/attention_op_test/AttentionEmptyPastState_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/attention_op_test/AttentionEmptyPastState_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/attention_op_test/AttentionLeftPaddingMaskIndex2_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/attention_op_test/AttentionLeftPaddingMaskIndex2_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/attention_op_test/AttentionMask1DEndNoWord_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 2) |
| test/contrib_ops/attention_op_test/AttentionMask1DEndNoWord_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 2) |
| test/contrib_ops/attention_op_test/AttentionMask1DNoWord_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 2) |
| test/contrib_ops/attention_op_test/AttentionMask1DNoWord_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 2) |
| test/contrib_ops/attention_op_test/AttentionMask2DNoWord_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 2) |
| test/contrib_ops/attention_op_test/AttentionMask2DNoWord_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 2) |
| test/contrib_ops/attention_op_test/AttentionMask3DNoWord_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 2) |
| test/contrib_ops/attention_op_test/AttentionMask3DNoWord_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 2) |
| test/contrib_ops/attention_op_test/AttentionMaskExceedSequence_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 5) |
| test/contrib_ops/attention_op_test/AttentionMaskExceedSequence_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 5) |
| test/contrib_ops/attention_op_test/AttentionMaskIndexOutOfRange_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 5) |
| test/contrib_ops/attention_op_test/AttentionMaskIndexOutOfRange_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 5) |
| test/contrib_ops/attention_op_test/AttentionMaskPartialSequence_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/attention_op_test/AttentionMaskPartialSequence_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/attention_op_test/AttentionNoMaskIndex_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 5) |
| test/contrib_ops/attention_op_test/AttentionNoMaskIndex_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 5) |
| test/contrib_ops/attention_op_test/AttentionPastStateBatch1_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/attention_op_test/AttentionPastStateBatch1_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/attention_op_test/AttentionPastStateBatch2WithPadding_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/attention_op_test/AttentionPastStateBatch2WithPadding_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/attention_op_test/AttentionPastStateBatch2_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/attention_op_test/AttentionPastStateBatch2_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/attention_op_test/AttentionPastState_dynamic_run0/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 58877) |
| test/contrib_ops/attention_op_test/AttentionPrunedModel_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 2) |
| test/contrib_ops/attention_op_test/AttentionPrunedModel_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 2) |
| test/contrib_ops/attention_op_test/AttentionRightPaddingMaskIndex2_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/attention_op_test/AttentionRightPaddingMaskIndex2_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/attention_op_test/AttentionUnidirectional3DMask_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 2) |
| test/contrib_ops/attention_op_test/AttentionUnidirectional3DMask_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 2) |
| test/contrib_ops/attention_op_test/AttentionUnidirectionalAttentionMask_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 2) |
| test/contrib_ops/attention_op_test/AttentionUnidirectionalAttentionMask_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 2) |
| test/contrib_ops/attention_op_test/AttentionUnidirectional_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/attention_op_test/AttentionUnidirectional_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/attention_op_test/AttentionWithNormFactor_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 2) |
| test/contrib_ops/attention_op_test/AttentionWithNormFactor_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 2) |
| test/contrib_ops/attention_op_test/Attention_Mask1D_Fp32_B2_S64_run0/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 739520) |
| test/contrib_ops/attention_op_test/Attention_Mask2D_Fp32_B2_S32_run0/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 980293) |
| test/contrib_ops/attention_op_test/SharedPrepackedWeights_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 2) |
| test/contrib_ops/attention_op_test/SharedPrepackedWeights_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 2) |
| test/contrib_ops/bifurcation_detector_op_test/Test1_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.BifurcationDetector |
| test/contrib_ops/bifurcation_detector_op_test/Test2_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.BifurcationDetector |
| test/contrib_ops/cdist_op_test/DoubleEuclidean_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.CDist |
| test/contrib_ops/cdist_op_test/Euclidean_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.CDist |
| test/contrib_ops/cdist_op_test/Sqeuclidean_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.CDist |
| test/contrib_ops/conv_transpose_with_dynamic_pads_test/ConvTransposeWithDynamicPads_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.ConvTransposeWithDynamicPads |
| test/contrib_ops/crop_and_resize_op_test/CropAndResize_1122_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.CropAndResize |
| test/contrib_ops/crop_and_resize_op_test/CropAndResize_1122_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.CropAndResize |
| test/contrib_ops/crop_and_resize_op_test/CropAndResize_1122_run2/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.CropAndResize |
| test/contrib_ops/crop_and_resize_op_test/CropAndResize_1133_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.CropAndResize |
| test/contrib_ops/crop_and_resize_op_test/CropAndResize_1133_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.CropAndResize |
| test/contrib_ops/crop_and_resize_op_test/CropAndResize_1133_run2/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.CropAndResize |
| test/contrib_ops/crop_and_resize_op_test/CropAndResize_1222_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.CropAndResize |
| test/contrib_ops/crop_and_resize_op_test/CropAndResize_1222_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.CropAndResize |
| test/contrib_ops/crop_and_resize_op_test/CropAndResize_2122_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.CropAndResize |
| test/contrib_ops/crop_and_resize_op_test/CropAndResize_2122_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.CropAndResize |
| test/contrib_ops/crop_op_test/Crop_Border_run0/model.onnx | 1 | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/crop_op_test/Crop_Scale_run0/model.onnx | 1 | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/decoder_masked_multihead_attention_op_test/cpu_cross_attn_fp32_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.DecoderMaskedMultiHeadAttention |
| test/contrib_ops/decoder_masked_multihead_attention_op_test/cpu_self_attn_fp32_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.DecoderMaskedMultiHeadAttention |
| test/contrib_ops/dynamic_quantize_matmul_test/Int8_run0/model.onnx (--max-ulp 5000) |  | Data/Data | ✅ | OK (max ULP 165) |
| test/contrib_ops/dynamic_quantize_matmul_test/Int8_run1/model.onnx (--max-ulp 5000) |  | Data/Data | ✅ | OK (max ULP 272) |
| test/contrib_ops/dynamic_quantize_matmul_test/Int8_run10/model.onnx (--max-ulp 5000) |  | Data/Data | ✅ | OK (max ULP 1345) |
| test/contrib_ops/dynamic_quantize_matmul_test/Int8_run11/model.onnx (--max-ulp 5000) |  | Data/Data | ✅ | OK (max ULP 1345) |
| test/contrib_ops/dynamic_quantize_matmul_test/Int8_run12/model.onnx (--max-ulp 5000) |  | Data/Data | ✅ | OK (max ULP 3526) |
| test/contrib_ops/dynamic_quantize_matmul_test/Int8_run13/model.onnx (--max-ulp 5000) |  | Data/Data | ✅ | OK (max ULP 882) |
| test/contrib_ops/dynamic_quantize_matmul_test/Int8_run14/model.onnx (--max-ulp 5000) |  | Data/Data | ✅ | OK (max ULP 456) |
| test/contrib_ops/dynamic_quantize_matmul_test/Int8_run15/model.onnx (--max-ulp 5000) |  | Data/Data | ✅ | OK (max ULP 2064) |
| test/contrib_ops/dynamic_quantize_matmul_test/Int8_run2/model.onnx (--max-ulp 5000) |  | Data/Data | ✅ | OK (max ULP 1345) |
| test/contrib_ops/dynamic_quantize_matmul_test/Int8_run3/model.onnx (--max-ulp 5000) |  | Data/Data | ✅ | OK (max ULP 1345) |
| test/contrib_ops/dynamic_quantize_matmul_test/Int8_run4/model.onnx (--max-ulp 5000) |  | Data/Data | ✅ | OK (max ULP 3526) |
| test/contrib_ops/dynamic_quantize_matmul_test/Int8_run5/model.onnx (--max-ulp 5000) |  | Data/Data | ✅ | OK (max ULP 882) |
| test/contrib_ops/dynamic_quantize_matmul_test/Int8_run6/model.onnx (--max-ulp 5000) |  | Data/Data | ✅ | OK (max ULP 456) |
| test/contrib_ops/dynamic_quantize_matmul_test/Int8_run7/model.onnx (--max-ulp 5000) |  | Data/Data | ✅ | OK (max ULP 2064) |
| test/contrib_ops/dynamic_quantize_matmul_test/Int8_run8/model.onnx (--max-ulp 5000) |  | Data/Data | ✅ | OK (max ULP 165) |
| test/contrib_ops/dynamic_quantize_matmul_test/Int8_run9/model.onnx (--max-ulp 5000) |  | Data/Data | ✅ | OK (max ULP 272) |
| test/contrib_ops/dynamic_quantize_matmul_test/UInt8_run0/model.onnx (--max-ulp 5000) |  | Data/Data | ✅ | OK (max ULP 1475) |
| test/contrib_ops/dynamic_quantize_matmul_test/UInt8_run1/model.onnx (--max-ulp 5000) |  | Data/Data | ✅ | OK (max ULP 236) |
| test/contrib_ops/dynamic_quantize_matmul_test/UInt8_run10/model.onnx (--max-ulp 5000) |  | Data/Data | ✅ | OK (max ULP 265) |
| test/contrib_ops/dynamic_quantize_matmul_test/UInt8_run11/model.onnx (--max-ulp 5000) |  | Data/Data | ✅ | OK (max ULP 616) |
| test/contrib_ops/dynamic_quantize_matmul_test/UInt8_run12/model.onnx (--max-ulp 5000) |  | Data/Data | ✅ | OK (max ULP 4098) |
| test/contrib_ops/dynamic_quantize_matmul_test/UInt8_run13/model.onnx (--max-ulp 5000) |  | Data/Data | ✅ | OK (max ULP 1024) |
| test/contrib_ops/dynamic_quantize_matmul_test/UInt8_run14/model.onnx (--max-ulp 5000) |  | Data/Data | ✅ | OK (max ULP 149) |
| test/contrib_ops/dynamic_quantize_matmul_test/UInt8_run15/model.onnx (--max-ulp 5000) |  | Data/Data | ✅ | OK (max ULP 298) |
| test/contrib_ops/dynamic_quantize_matmul_test/UInt8_run2/model.onnx (--max-ulp 5000) |  | Data/Data | ✅ | OK (max ULP 265) |
| test/contrib_ops/dynamic_quantize_matmul_test/UInt8_run3/model.onnx (--max-ulp 5000) |  | Data/Data | ✅ | OK (max ULP 616) |
| test/contrib_ops/dynamic_quantize_matmul_test/UInt8_run4/model.onnx (--max-ulp 5000) |  | Data/Data | ✅ | OK (max ULP 4098) |
| test/contrib_ops/dynamic_quantize_matmul_test/UInt8_run5/model.onnx (--max-ulp 5000) |  | Data/Data | ✅ | OK (max ULP 1024) |
| test/contrib_ops/dynamic_quantize_matmul_test/UInt8_run6/model.onnx (--max-ulp 5000) |  | Data/Data | ✅ | OK (max ULP 149) |
| test/contrib_ops/dynamic_quantize_matmul_test/UInt8_run7/model.onnx (--max-ulp 5000) |  | Data/Data | ✅ | OK (max ULP 298) |
| test/contrib_ops/dynamic_quantize_matmul_test/UInt8_run8/model.onnx (--max-ulp 5000) |  | Data/Data | ✅ | OK (max ULP 1475) |
| test/contrib_ops/dynamic_quantize_matmul_test/UInt8_run9/model.onnx (--max-ulp 5000) |  | Data/Data | ✅ | OK (max ULP 236) |
| test/contrib_ops/dynamic_quantize_matmul_test/UInt8_test_with_empty_input_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/dynamic_quantize_matmul_test/WithConstantBInputs_run0/model.onnx (--max-ulp 5000) |  | Data/Data | ✅ | OK (max ULP 165) |
| test/contrib_ops/dynamic_quantize_matmul_test/WithConstantBInputs_run1/model.onnx (--max-ulp 5000) |  | Data/Data | ✅ | OK (max ULP 1475) |
| test/contrib_ops/dynamic_quantize_matmul_test/WithConstantBInputs_run2/model.onnx (--max-ulp 5000) |  | Data/Data | ✅ | OK (max ULP 165) |
| test/contrib_ops/dynamic_quantize_matmul_test/WithConstantBInputs_run3/model.onnx (--max-ulp 5000) |  | Data/Data | ✅ | OK (max ULP 1475) |
| test/contrib_ops/dynamic_quantize_matmul_test/WithConstantBInputs_run4/model.onnx (--max-ulp 5000) |  | Data/Data | ✅ | OK (max ULP 1345) |
| test/contrib_ops/dynamic_quantize_matmul_test/WithConstantBInputs_run5/model.onnx (--max-ulp 5000) |  | Data/Data | ✅ | OK (max ULP 265) |
| test/contrib_ops/dynamic_slice_op_test/dynamic_slice_ends_out_of_bounds_run0/model.onnx | 1 | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/dynamic_slice_op_test/dynamic_slice_full_axes_run0/model.onnx | 1 | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/dynamic_slice_op_test/dynamic_slice_full_axes_run1/model.onnx | 1 | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/dynamic_slice_op_test/dynamic_slice_varied_types_run0/model.onnx | 1 | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/dynamic_slice_op_test/dynamic_slice_varied_types_run1/model.onnx | 1 | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/dynamic_slice_op_test/dynamic_slice_varied_types_run2/model.onnx | 1 | Data/Data | ✅ | OK (no numeric comparisons) |
| test/contrib_ops/dynamic_slice_op_test/dynamic_slice_varied_types_run3/model.onnx | 1 | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/dynamic_slice_op_test/dynamic_slice_varied_types_run4/model.onnx | 1 | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/dynamic_slice_op_test/dynamic_slice_with_axes_run0/model.onnx | 1 | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/dynamic_slice_op_test/dynamic_slice_with_axes_run1/model.onnx | 1 | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/dynamic_slice_op_test/dynamic_slice_with_negative_axes_run0/model.onnx | 1 | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/dynamic_slice_op_test/dynamic_slice_with_negative_axes_run1/model.onnx | 1 | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/dynamic_time_warping_op_test/simple_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.DynamicTimeWarping |
| test/contrib_ops/element_wise_ops_test/AffineDefaultAttributes_run0/model.onnx | 7 | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/element_wise_ops_test/Affine_run0/model.onnx | 7 | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/element_wise_ops_test/Float_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/element_wise_ops_test/Float_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/element_wise_ops_test/Float_run2/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/element_wise_ops_test/Float_run3/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/element_wise_ops_test/Float_run4/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/element_wise_ops_test/Float_run5/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/element_wise_ops_test/Scale_Default_run0/model.onnx | 7 | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/element_wise_ops_test/Scale_run0/model.onnx | 7 | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/embed_layer_norm_op_test/EmbedLayerNormBatch1_EmbeddingSum_NoMaskIndex_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.EmbedLayerNormalization |
| test/contrib_ops/embed_layer_norm_op_test/EmbedLayerNormBatch1_EmbeddingSum_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.EmbedLayerNormalization |
| test/contrib_ops/embed_layer_norm_op_test/EmbedLayerNormBatch1_PositionIdsDiffOrder_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.EmbedLayerNormalization |
| test/contrib_ops/embed_layer_norm_op_test/EmbedLayerNormBatch1_PositionIds_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.EmbedLayerNormalization |
| test/contrib_ops/embed_layer_norm_op_test/EmbedLayerNormBatch1_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.EmbedLayerNormalization |
| test/contrib_ops/embed_layer_norm_op_test/EmbedLayerNormBatch2_NoMask_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.EmbedLayerNormalization |
| test/contrib_ops/embed_layer_norm_op_test/EmbedLayerNormBatch2_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.EmbedLayerNormalization |
| test/contrib_ops/embed_layer_norm_op_test/EmbedLayerNormBatch3_PositionIds_BroadCast_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.EmbedLayerNormalization |
| test/contrib_ops/embed_layer_norm_op_test/EmbedLayerNormBatch_Distill_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.EmbedLayerNormalization |
| test/contrib_ops/embed_layer_norm_op_test/EmbedLayerNormLargeBatchSmallHiddenSize_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.EmbedLayerNormalization |
| test/contrib_ops/expand_dims_test/Basic_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/expand_dims_test/Basic_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/expand_dims_test/Basic_run2/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/expand_dims_test/MaxAxis_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/expand_dims_test/MaxAxis_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/expand_dims_test/MinAxis_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/expand_dims_test/MinAxis_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/expand_dims_test/NegativeAxisOutOfRange_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/expand_dims_test/NegativeAxisOutOfRange_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/expand_dims_test/PositiveAxisOutOfRange_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/expand_dims_test/PositiveAxisOutOfRange_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fastgelu_op_test/FastGeluWithBiasFloat32_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fastgelu_op_test/FastGeluWithNullInput_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fastgelu_op_test/FastGeluWithoutBiasFloat32_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_conv_test/Conv2D_Bias_Relu_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.FusedConv |
| test/contrib_ops/fused_conv_test/Conv2D_HardSigmoid_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.FusedConv |
| test/contrib_ops/fused_conv_test/Conv2D_Relu_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.FusedConv |
| test/contrib_ops/fused_conv_test/Cpu_Conv2D_Bias_Z_Relu_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.FusedConv |
| test/contrib_ops/fused_matmul_op_test/FloatTypeNoTranspose_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeNoTranspose_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeNoTranspose_run10/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeNoTranspose_run11/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeNoTranspose_run2/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeNoTranspose_run3/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeNoTranspose_run4/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeNoTranspose_run5/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeNoTranspose_run6/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeNoTranspose_run7/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeNoTranspose_run8/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeNoTranspose_run9/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run10/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run11/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run12/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run13/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run14/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run15/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run16/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run17/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run18/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run19/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run2/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run20/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run21/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run22/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run23/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run24/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run25/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run26/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run27/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run28/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run29/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run3/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run30/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run31/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run32/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run33/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run34/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run35/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run36/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run37/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run38/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run39/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run4/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run40/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run41/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run42/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run43/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run44/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run45/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run46/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run47/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run48/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run49/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run5/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run50/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run51/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run52/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run53/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run54/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run55/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run56/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run57/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run58/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run59/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run6/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run60/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run61/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run62/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run63/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run64/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run65/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run66/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run67/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run68/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run69/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run7/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run70/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run71/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run8/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run9/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeAB_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeAB_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeAB_run10/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeAB_run11/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeAB_run12/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeAB_run13/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeAB_run14/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeAB_run15/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeAB_run16/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeAB_run17/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeAB_run18/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeAB_run19/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeAB_run2/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeAB_run20/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeAB_run21/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeAB_run22/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeAB_run23/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeAB_run3/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeAB_run4/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeAB_run5/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeAB_run6/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeAB_run7/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeAB_run8/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeAB_run9/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeA_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeA_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeA_run10/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeA_run11/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeA_run2/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeA_run3/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeA_run4/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeA_run5/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeA_run6/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeA_run7/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeA_run8/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeA_run9/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeB_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeB_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeB_run10/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeB_run11/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeB_run12/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeB_run13/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeB_run14/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeB_run15/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeB_run16/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeB_run17/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeB_run18/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeB_run19/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeB_run2/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeB_run20/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeB_run21/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeB_run22/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeB_run23/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeB_run3/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeB_run4/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeB_run5/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeB_run6/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeB_run7/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeB_run8/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeB_run9/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run0/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 6684672) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run10/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 14680064) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run11/model.onnx |  | Data/Data | ❌ | FusedMatMul batch dimensions are not broadcastable: (1, 3) vs (3, 2) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run12/model.onnx |  | Data/Data | ❌ | FusedMatMul inner dimensions must match after transposition, got effective shapes (3, 2, 2) and (1, 3, 4) (original: (3, 2, 2) and (1, 3, 4)) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run13/model.onnx |  | Data/Data | ❌ | FusedMatMul inner dimensions must match after transposition, got effective shapes (3, 2, 1) and (1, 3, 4) (original: (3, 1, 2) and (1, 3, 4)) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run14/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 16777216) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run15/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 22020096) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run16/model.onnx |  | Data/Data | ❌ | FusedMatMul inner dimensions must match after transposition, got effective shapes (2, 2, 3) and (3, 1, 4) (original: (2, 3, 2) and (3, 1, 4)) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run17/model.onnx |  | Data/Data | ❌ | FusedMatMul inner dimensions must match after transposition, got effective shapes (1, 2, 3) and (3, 1, 4) (original: (1, 3, 2) and (3, 1, 4)) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run18/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 8388608) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run19/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 8388608) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run2/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 6815744) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run20/model.onnx |  | Data/Data | ❌ | FusedMatMul inner dimensions must match after transposition, got effective shapes (3, 2, 2) and (3, 1, 4) (original: (3, 2, 2) and (3, 1, 4)) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run21/model.onnx |  | Data/Data | ❌ | Output shape mismatch for Y (actual_shape=(3, 2, 4), actual_size=24, expected_shape=(1, 2, 4), expected_size=8, output=Y) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run22/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 25165824) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run23/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 29360128) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run24/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 6684672) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run25/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run26/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 6815744) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run27/model.onnx |  | Data/Data | ❌ | FusedMatMul batch dimensions are not broadcastable: (1, 3) vs (3, 2) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run28/model.onnx |  | Data/Data | ❌ | FusedMatMul batch dimensions are not broadcastable: (2,) vs (4,) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run29/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 12058624) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run3/model.onnx |  | Data/Data | ❌ | FusedMatMul batch dimensions are not broadcastable: (1, 3) vs (3, 2) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run30/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 11272192) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run31/model.onnx |  | Data/Data | ❌ | Output shape mismatch for Y (actual_shape=(3, 2, 3, 2), actual_size=36, expected_shape=(3, 2, 3, 1), expected_size=18, output=Y) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run32/model.onnx |  | Data/Data | ❌ | FusedMatMul batch dimensions are not broadcastable: (2,) vs (4,) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run33/model.onnx |  | Data/Data | ❌ | FusedMatMul batch dimensions are not broadcastable: (2,) vs (4,) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run34/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 17301504) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run35/model.onnx |  | Data/Data | ❌ | Output shape mismatch for Y (actual_shape=(3, 3, 2, 2), actual_size=36, expected_shape=(3, 2, 3, 1), expected_size=18, output=Y) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run36/model.onnx |  | Data/Data | ❌ | FusedMatMul inner dimensions must match after transposition, got effective shapes (3, 2, 2) and (1, 3, 4) (original: (3, 2, 2) and (1, 4, 3)) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run37/model.onnx |  | Data/Data | ❌ | FusedMatMul inner dimensions must match after transposition, got effective shapes (3, 2, 1) and (1, 3, 4) (original: (3, 1, 2) and (1, 4, 3)) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run38/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 16777216) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run39/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 22020096) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run4/model.onnx |  | Data/Data | ❌ | FusedMatMul inner dimensions must match after transposition, got effective shapes (2, 2, 3) and (3, 1, 4) (original: (2, 2, 3) and (3, 1, 4)) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run40/model.onnx |  | Data/Data | ❌ | FusedMatMul batch dimensions are not broadcastable: (2,) vs (4,) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run41/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 12058624) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run42/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 11272192) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run43/model.onnx |  | Data/Data | ❌ | Output shape mismatch for Y (actual_shape=(3, 2, 3, 2), actual_size=36, expected_shape=(3, 2, 3, 1), expected_size=18, output=Y) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run44/model.onnx |  | Data/Data | ❌ | FusedMatMul inner dimensions must match after transposition, got effective shapes (3, 2, 2) and (4, 3, 1) (original: (3, 2, 2) and (4, 1, 3)) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run45/model.onnx |  | Data/Data | ❌ | FusedMatMul inner dimensions must match after transposition, got effective shapes (3, 2, 1) and (4, 3, 1) (original: (3, 1, 2) and (4, 1, 3)) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run46/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 25165824) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run47/model.onnx |  | Data/Data | ❌ | Output shape mismatch for Y (actual_shape=(3, 2, 3, 2), actual_size=36, expected_shape=(3, 2, 3, 1), expected_size=18, output=Y) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run5/model.onnx |  | Data/Data | ❌ | FusedMatMul inner dimensions must match after transposition, got effective shapes (1, 2, 3) and (3, 1, 4) (original: (1, 2, 3) and (3, 1, 4)) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run6/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 8388608) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run7/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 8388608) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run8/model.onnx |  | Data/Data | ❌ | FusedMatMul inner dimensions must match after transposition, got effective shapes (2, 2, 3) and (3, 1, 4) (original: (2, 2, 3) and (3, 1, 4)) |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run9/model.onnx |  | Data/Data | ❌ | FusedMatMul inner dimensions must match after transposition, got effective shapes (2, 1, 3) and (3, 1, 4) (original: (2, 1, 3) and (3, 1, 4)) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_4Bits_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_4Bits_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_4Bits_run10/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_4Bits_run11/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_4Bits_run12/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_4Bits_run13/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_4Bits_run14/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_4Bits_run15/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_4Bits_run2/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_4Bits_run3/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_4Bits_run4/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_4Bits_run5/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_4Bits_run6/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_4Bits_run7/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_4Bits_run8/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_4Bits_run9/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_8Bits_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_8Bits_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_8Bits_run2/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_8Bits_run3/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_8Bits_run4/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_8Bits_run5/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_8Bits_run6/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_8Bits_run7/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_run10/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_run11/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_run12/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_run13/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_run14/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_run15/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_run2/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_run3/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_run4/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_run5/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_run6/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_run7/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_run8/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_run9/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_4Bits_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_4Bits_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_4Bits_run2/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_4Bits_run3/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_4Bits_run4/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_4Bits_run5/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_4Bits_run6/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_4Bits_run7/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_8Bits_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_8Bits_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_8Bits_run2/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_8Bits_run3/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_8Bits_run4/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_8Bits_run5/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_8Bits_run6/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_8Bits_run7/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run10/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run11/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run12/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run13/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run14/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run15/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run16/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run17/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run18/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run19/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run2/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run20/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run21/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run22/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run23/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run24/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run25/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run26/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run27/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run28/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run29/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run3/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run30/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run31/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run4/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run5/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run6/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run7/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run8/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run9/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run10/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run11/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run12/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run13/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run14/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run15/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run16/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run17/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run18/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run19/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run2/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run20/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run21/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run22/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run23/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run24/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run25/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run26/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run27/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run28/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run29/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run3/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run30/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run31/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run4/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run5/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run6/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run7/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run8/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run9/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run10/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run11/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run12/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run13/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run14/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run15/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run16/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run17/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run18/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run19/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run2/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run20/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run21/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run22/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run23/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run24/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run25/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run26/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run27/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run28/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run29/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run3/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run30/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run31/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run4/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run5/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run6/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run7/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run8/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run9/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/InvalidBlockSize_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/InvalidBlockSize_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/InvalidBlockSize_run2/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/InvalidBlockSize_run3/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/InvalidBlockSize_run4/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/InvalidBlockSize_run5/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gather_block_quantized_op_test/InvalidGatherAxis_run0/model.onnx |  | Data/Data | ❌ | GatherBlockQuantized axis 3 is out of range for rank 3 |
| test/contrib_ops/gather_block_quantized_op_test/InvalidGatherAxis_run1/model.onnx |  | Data/Data | ❌ | GatherBlockQuantized axis 3 is out of range for rank 3 |
| test/contrib_ops/gather_block_quantized_op_test/InvalidGatherAxis_run2/model.onnx |  | Data/Data | ❌ | GatherBlockQuantized axis -1 is out of range for rank 3 |
| test/contrib_ops/gather_block_quantized_op_test/InvalidGatherAxis_run3/model.onnx |  | Data/Data | ❌ | GatherBlockQuantized axis -1 is out of range for rank 3 |
| test/contrib_ops/gather_block_quantized_op_test/InvalidGatherAxis_run4/model.onnx |  | Data/Data | ❌ | GatherBlockQuantized axis -1 is out of range for rank 3 |
| test/contrib_ops/gather_block_quantized_op_test/InvalidGatherAxis_run5/model.onnx |  | Data/Data | ❌ | GatherBlockQuantized axis -1 is out of range for rank 3 |
| test/contrib_ops/gather_block_quantized_op_test/InvalidIndices_run0/model.onnx |  | Data/Data | ❌ | Testbench execution failed: exit code 1 |
| test/contrib_ops/gather_block_quantized_op_test/InvalidIndices_run1/model.onnx |  | Data/Data | ❌ | Testbench execution failed: exit code 1 |
| test/contrib_ops/gather_block_quantized_op_test/InvalidIndices_run2/model.onnx |  | Data/Data | ❌ | Testbench execution failed: exit code 1 |
| test/contrib_ops/gather_block_quantized_op_test/InvalidIndices_run3/model.onnx |  | Data/Data | ❌ | Testbench execution failed: exit code 1 |
| test/contrib_ops/gather_block_quantized_op_test/InvalidIndices_run4/model.onnx |  | Data/Data | ❌ | Testbench execution failed: exit code 1 |
| test/contrib_ops/gather_block_quantized_op_test/InvalidIndices_run5/model.onnx |  | Data/Data | ❌ | Testbench execution failed: exit code 1 |
| test/contrib_ops/gather_block_quantized_op_test/InvalidQuantizeAxis_run0/model.onnx |  | Data/Data | ❌ | GatherBlockQuantized axis 3 is out of range for rank 3 |
| test/contrib_ops/gather_block_quantized_op_test/InvalidQuantizeAxis_run1/model.onnx |  | Data/Data | ❌ | GatherBlockQuantized axis 3 is out of range for rank 3 |
| test/contrib_ops/gather_block_quantized_op_test/InvalidQuantizeAxis_run2/model.onnx |  | Data/Data | ❌ | GatherBlockQuantized axis -1 is out of range for rank 3 |
| test/contrib_ops/gather_block_quantized_op_test/InvalidQuantizeAxis_run3/model.onnx |  | Data/Data | ❌ | GatherBlockQuantized axis -1 is out of range for rank 3 |
| test/contrib_ops/gather_block_quantized_op_test/InvalidQuantizeAxis_run4/model.onnx |  | Data/Data | ❌ | GatherBlockQuantized axis -1 is out of range for rank 3 |
| test/contrib_ops/gather_block_quantized_op_test/InvalidQuantizeAxis_run5/model.onnx |  | Data/Data | ❌ | GatherBlockQuantized axis -1 is out of range for rank 3 |
| test/contrib_ops/gather_block_quantized_op_test/NotSupportedBits_run0/model.onnx |  | Data/Data | ❌ | GatherBlockQuantized supports bits in [4, 8], got 1 |
| test/contrib_ops/gather_block_quantized_op_test/NotSupportedBits_run1/model.onnx |  | Data/Data | ❌ | GatherBlockQuantized supports bits in [4, 8], got 1 |
| test/contrib_ops/gather_block_quantized_op_test/NotSupportedBits_run10/model.onnx |  | Data/Data | ❌ | GatherBlockQuantized supports bits in [4, 8], got 7 |
| test/contrib_ops/gather_block_quantized_op_test/NotSupportedBits_run11/model.onnx |  | Data/Data | ❌ | GatherBlockQuantized supports bits in [4, 8], got 7 |
| test/contrib_ops/gather_block_quantized_op_test/NotSupportedBits_run2/model.onnx |  | Data/Data | ❌ | GatherBlockQuantized supports bits in [4, 8], got 2 |
| test/contrib_ops/gather_block_quantized_op_test/NotSupportedBits_run3/model.onnx |  | Data/Data | ❌ | GatherBlockQuantized supports bits in [4, 8], got 2 |
| test/contrib_ops/gather_block_quantized_op_test/NotSupportedBits_run4/model.onnx |  | Data/Data | ❌ | GatherBlockQuantized supports bits in [4, 8], got 3 |
| test/contrib_ops/gather_block_quantized_op_test/NotSupportedBits_run5/model.onnx |  | Data/Data | ❌ | GatherBlockQuantized supports bits in [4, 8], got 3 |
| test/contrib_ops/gather_block_quantized_op_test/NotSupportedBits_run6/model.onnx |  | Data/Data | ❌ | GatherBlockQuantized supports bits in [4, 8], got 5 |
| test/contrib_ops/gather_block_quantized_op_test/NotSupportedBits_run7/model.onnx |  | Data/Data | ❌ | GatherBlockQuantized supports bits in [4, 8], got 5 |
| test/contrib_ops/gather_block_quantized_op_test/NotSupportedBits_run8/model.onnx |  | Data/Data | ❌ | GatherBlockQuantized supports bits in [4, 8], got 6 |
| test/contrib_ops/gather_block_quantized_op_test/NotSupportedBits_run9/model.onnx |  | Data/Data | ❌ | GatherBlockQuantized supports bits in [4, 8], got 6 |
| test/contrib_ops/gather_block_quantized_op_test/ShapeMismatch_run0/model.onnx |  | Data/Data | ❌ | GatherBlockQuantized scales shape (2, 2, 1) does not match expected (2, 3, 1) (data_shape=(2, 3, 4), quantize_axis=2, block_size=16, packed=False) |
| test/contrib_ops/gather_block_quantized_op_test/ShapeMismatch_run1/model.onnx |  | Data/Data | ❌ | GatherBlockQuantized scales shape (2, 2, 1) does not match expected (2, 3, 1) (data_shape=(2, 3, 4), quantize_axis=2, block_size=16, packed=False) |
| test/contrib_ops/gather_block_quantized_op_test/ShapeMismatch_run2/model.onnx |  | Data/Data | ❌ | GatherBlockQuantized scales shape (2, 2, 1) does not match expected (2, 3, 1) (data_shape=(2, 3, 4), quantize_axis=2, block_size=16, packed=False) |
| test/contrib_ops/gather_block_quantized_op_test/ShapeMismatch_run3/model.onnx |  | Data/Data | ❌ | GatherBlockQuantized scales shape (2, 2, 1) does not match expected (2, 3, 1) (data_shape=(2, 3, 4), quantize_axis=2, block_size=16, packed=False) |
| test/contrib_ops/gather_block_quantized_op_test/ShapeMismatch_run4/model.onnx |  | Data/Data | ❌ | GatherBlockQuantized scales shape (2, 2, 1) does not match expected (2, 3, 1) (data_shape=(2, 3, 2), quantize_axis=2, block_size=16, packed=True) |
| test/contrib_ops/gather_block_quantized_op_test/ShapeMismatch_run5/model.onnx |  | Data/Data | ❌ | GatherBlockQuantized scales shape (2, 2, 1) does not match expected (2, 3, 1) (data_shape=(2, 3, 2), quantize_axis=2, block_size=16, packed=True) |
| test/contrib_ops/gather_block_quantized_op_test/UnsupportedTypes_run0/model.onnx |  | Data/Data | ❌ | Output shape mismatch for output (actual_shape=(1, 3, 8), actual_size=24, expected_shape=(1, 3, 4), expected_size=12, output=output) |
| test/contrib_ops/gather_block_quantized_op_test/UnsupportedTypes_run1/model.onnx |  | Data/Data | ❌ | Output shape mismatch for output (actual_shape=(1, 3, 8), actual_size=24, expected_shape=(1, 3, 4), expected_size=12, output=output) |
| test/contrib_ops/gather_block_quantized_op_test/UnsupportedTypes_run10/model.onnx |  | Data/Data | ❌ | GatherBlockQuantized data dtype must be integer, got int64 |
| test/contrib_ops/gather_block_quantized_op_test/UnsupportedTypes_run11/model.onnx |  | Data/Data | ❌ | GatherBlockQuantized data dtype must be integer, got int64 |
| test/contrib_ops/gather_block_quantized_op_test/UnsupportedTypes_run12/model.onnx |  | Data/Data | ❌ | GatherBlockQuantized data dtype must be integer, got uint64 |
| test/contrib_ops/gather_block_quantized_op_test/UnsupportedTypes_run13/model.onnx |  | Data/Data | ❌ | GatherBlockQuantized data dtype must be integer, got uint64 |
| test/contrib_ops/gather_block_quantized_op_test/UnsupportedTypes_run14/model.onnx |  | Data/Data | ❌ | GatherBlockQuantized indices must be int32 or int64 |
| test/contrib_ops/gather_block_quantized_op_test/UnsupportedTypes_run15/model.onnx |  | Data/Data | ❌ | GatherBlockQuantized indices must be int32 or int64 |
| test/contrib_ops/gather_block_quantized_op_test/UnsupportedTypes_run16/model.onnx |  | Data/Data | ❌ | GatherBlockQuantized indices must be int32 or int64 |
| test/contrib_ops/gather_block_quantized_op_test/UnsupportedTypes_run17/model.onnx |  | Data/Data | ❌ | GatherBlockQuantized indices must be int32 or int64 |
| test/contrib_ops/gather_block_quantized_op_test/UnsupportedTypes_run18/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/gather_block_quantized_op_test/UnsupportedTypes_run19/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/gather_block_quantized_op_test/UnsupportedTypes_run2/model.onnx |  | Data/Data | ❌ | Testbench execution failed: exit code 1 |
| test/contrib_ops/gather_block_quantized_op_test/UnsupportedTypes_run20/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/gather_block_quantized_op_test/UnsupportedTypes_run21/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/gather_block_quantized_op_test/UnsupportedTypes_run22/model.onnx |  | Data/Data | ❌ | GatherBlockQuantized indices must be int32 or int64 |
| test/contrib_ops/gather_block_quantized_op_test/UnsupportedTypes_run23/model.onnx |  | Data/Data | ❌ | GatherBlockQuantized indices must be int32 or int64 |
| test/contrib_ops/gather_block_quantized_op_test/UnsupportedTypes_run3/model.onnx |  | Data/Data | ❌ | Testbench execution failed: exit code 1 |
| test/contrib_ops/gather_block_quantized_op_test/UnsupportedTypes_run4/model.onnx |  | Data/Data | ❌ | Testbench execution failed: exit code 1 |
| test/contrib_ops/gather_block_quantized_op_test/UnsupportedTypes_run5/model.onnx |  | Data/Data | ❌ | Testbench execution failed: exit code 1 |
| test/contrib_ops/gather_block_quantized_op_test/UnsupportedTypes_run6/model.onnx |  | Data/Data | ❌ | GatherBlockQuantized scales shape (2, 3, 1) does not match expected (2, 3, 2) (data_shape=(2, 3, 4), quantize_axis=2, block_size=16, packed=True) |
| test/contrib_ops/gather_block_quantized_op_test/UnsupportedTypes_run7/model.onnx |  | Data/Data | ❌ | GatherBlockQuantized scales shape (2, 3, 1) does not match expected (2, 3, 2) (data_shape=(2, 3, 4), quantize_axis=2, block_size=16, packed=True) |
| test/contrib_ops/gather_block_quantized_op_test/UnsupportedTypes_run8/model.onnx |  | Data/Data | ❌ | GatherBlockQuantized scales shape (2, 3, 1) does not match expected (2, 3, 2) (data_shape=(2, 3, 4), quantize_axis=2, block_size=16, packed=True) |
| test/contrib_ops/gather_block_quantized_op_test/UnsupportedTypes_run9/model.onnx |  | Data/Data | ❌ | GatherBlockQuantized scales shape (2, 3, 1) does not match expected (2, 3, 2) (data_shape=(2, 3, 4), quantize_axis=2, block_size=16, packed=True) |
| test/contrib_ops/gather_block_quantized_op_test/UnsupportedUInt8DataType_run0/model.onnx |  | Data/Data | ❌ | Output shape mismatch for output (actual_shape=(2, 1, 4), actual_size=8, expected_shape=(1, 3, 4), expected_size=12, output=output) |
| test/contrib_ops/gather_block_quantized_op_test/UnsupportedUInt8DataType_run1/model.onnx |  | Data/Data | ❌ | Output shape mismatch for output (actual_shape=(2, 1, 4), actual_size=8, expected_shape=(1, 3, 4), expected_size=12, output=output) |
| test/contrib_ops/gather_block_quantized_op_test/UnsupportedUInt8DataType_run2/model.onnx |  | Data/Data | ❌ | GatherBlockQuantized indices must be int32 or int64 |
| test/contrib_ops/gather_block_quantized_op_test/UnsupportedUInt8DataType_run3/model.onnx |  | Data/Data | ❌ | GatherBlockQuantized indices must be int32 or int64 |
| test/contrib_ops/gridsample_test/gridsample_aligncorners_true_run0/model.onnx |  | Data/Data | ❌ | GridSample mode 'bilinear' is not supported |
| test/contrib_ops/gridsample_test/gridsample_default_run0/model.onnx |  | Data/Data | ❌ | GridSample mode 'bilinear' is not supported |
| test/contrib_ops/gridsample_test/gridsample_mode_bicubic_run0/model.onnx |  | Data/Data | ❌ | GridSample mode 'bicubic' is not supported |
| test/contrib_ops/gridsample_test/gridsample_mode_bilinear_run0/model.onnx |  | Data/Data | ❌ | GridSample mode 'bilinear' is not supported |
| test/contrib_ops/gridsample_test/gridsample_mode_nearest_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/gridsample_test/gridsample_paddingmode_border_run0/model.onnx |  | Data/Data | ❌ | GridSample mode 'bilinear' is not supported |
| test/contrib_ops/gridsample_test/gridsample_paddingmode_reflection_run0/model.onnx |  | Data/Data | ❌ | GridSample mode 'bilinear' is not supported |
| test/contrib_ops/gridsample_test/gridsample_paddingmode_zeros_run0/model.onnx |  | Data/Data | ❌ | GridSample mode 'bilinear' is not supported |
| test/contrib_ops/inverse_test/four_by_four_batches_float_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/inverse_test/four_by_four_float_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/inverse_test/two_by_two_double_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/inverse_test/two_by_two_float16_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/inverse_test/two_by_two_float_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/layer_norm_op_test/BERTLayerNorm_NoBias_run0/model.onnx | 17 | Data/Data | ❌ | Out of tolerance (max ULP 2272) |
| test/contrib_ops/layer_norm_op_test/BERTLayerNorm_run0/model.onnx | 17 | Data/Data | ❌ | Out of tolerance (max ULP 5632) |
| test/contrib_ops/layer_norm_op_test/LayerNorm17_double_run0/model.onnx | 17 | Data/Data | ❌ | Out of tolerance (max ULP 160715044774) |
| test/contrib_ops/layer_norm_op_test/LayerNorm17_opset_run0/model.onnx | 17 | Data/Data | ❌ | Out of tolerance (max ULP 299) |
| test/contrib_ops/layer_norm_op_test/LayerNorm17_opset_run1/model.onnx | 17 | Data/Data | ❌ | Out of tolerance (max ULP 299) |
| test/contrib_ops/layer_norm_op_test/LayerNorm_NormSize1_NoBias_run0/model.onnx | 17 | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/layer_norm_op_test/LayerNorm_NormSize1_Valid_run0/model.onnx | 7 | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/layer_norm_op_test/LayerNorm_NormSize1_WithBiasScale_run0/model.onnx | 17 | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/layer_norm_op_test/LayerNorm_Scale_Bias_4D_OuterInnerBroadcast_Axis3_run0/model.onnx | 17 | Data/Data | ❌ | Out of tolerance (max ULP 1613) |
| test/contrib_ops/layer_norm_op_test/LayerNorm_Scale_Bias_4D_OuterInnerBroadcast_run0/model.onnx | 17 | Data/Data | ❌ | Out of tolerance (max ULP 1613) |
| test/contrib_ops/layer_norm_op_test/LayerNorm_Scale_Bias_Axis2_run0/model.onnx | 17 | Data/Data | ❌ | Out of tolerance (max ULP 336) |
| test/contrib_ops/layer_norm_op_test/LayerNorm_Scale_Bias_Broadcast_Dim0_Fp16_run0/model.onnx | 7 | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/layer_norm_op_test/LayerNorm_Scale_Bias_Broadcast_Dim0_run0/model.onnx | 7 | Data/Data | ❌ | Out of tolerance (max ULP 286) |
| test/contrib_ops/layer_norm_op_test/LayerNorm_Scale_Bias_Broadcast_Dim1_Fp16_run0/model.onnx | 7 | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/layer_norm_op_test/LayerNorm_Scale_Bias_Broadcast_Dim1_run0/model.onnx | 7 | Data/Data | ❌ | Out of tolerance (max ULP 286) |
| test/contrib_ops/layer_norm_op_test/LayerNorm_Scale_Bias_Broadcast_Fp16_run0/model.onnx | 7 | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/layer_norm_op_test/LayerNorm_Scale_Bias_Broadcast_Fp16_run1/model.onnx | 7 | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/layer_norm_op_test/LayerNorm_Scale_Bias_Float16InputScaleBiasOutput_Initializers_run0/model.onnx | 7 | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/layer_norm_op_test/LayerNorm_Scale_Bias_Float16InputScaleBiasOutput_run0/model.onnx | 7 | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/layer_norm_op_test/LayerNorm_Scale_Bias_Float16InputScaleBiasOutput_run1/model.onnx | 7 | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/layer_norm_op_test/LayerNorm_Scale_Bias_NoBroadcast_Fp16_run0/model.onnx | 7 | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/layer_norm_op_test/LayerNorm_Scale_Bias_NoBroadcast_run0/model.onnx | 7 | Data/Data | ✅ | OK (max ULP 30) |
| test/contrib_ops/layer_norm_op_test/LayerNorm_Scale_Bias_PerLastDim_run0/model.onnx | 17 | Data/Data | ❌ | Out of tolerance (max ULP 336) |
| test/contrib_ops/layer_norm_op_test/LayerNorm_Scale_Bias_Scalar_Axis2_run0/model.onnx | 17 | Data/Data | ❌ | Out of tolerance (max ULP 252) |
| test/contrib_ops/layer_norm_op_test/LayerNorm_Scale_Bias_Scalar_run0/model.onnx | 17 | Data/Data | ❌ | Out of tolerance (max ULP 252) |
| test/contrib_ops/layer_norm_op_test/LayerNorm_Scale_Bias_run0/model.onnx | 7 | Data/Data | ❌ | Out of tolerance (max ULP 10874) |
| test/contrib_ops/layer_norm_op_test/LayerNorm_Scale_Broadcast_Inner_Mixed_run0/model.onnx | 17 | Data/Data | ✅ | OK (max ULP 21) |
| test/contrib_ops/layer_norm_op_test/LayerNorm_Scale_Float16InputScaleOutput_Initializers_run0/model.onnx | 7 | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/layer_norm_op_test/LayerNorm_Scale_Float16InputScaleOutput_run0/model.onnx | 7 | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/layer_norm_op_test/LayerNorm_Scale_Scalar_NoBias_Axis2_run0/model.onnx | 17 | Data/Data | ❌ | Out of tolerance (max ULP 252) |
| test/contrib_ops/layer_norm_op_test/LayerNorm_Scale_Scalar_NoBias_run0/model.onnx | 17 | Data/Data | ❌ | Out of tolerance (max ULP 252) |
| test/contrib_ops/layer_norm_op_test/LayerNorm_Scale_run0/model.onnx | 7 | Data/Data | ✅ | OK (max ULP 3) |
| test/contrib_ops/layer_norm_op_test/LayerNorm_ValidScaleBias_Broadcast_run0/model.onnx | 7 | Data/Data | ✅ | OK (max ULP 80) |
| test/contrib_ops/layer_norm_op_test/LayerNorm_run0/model.onnx | 7 | Data/Data | ❌ | Out of tolerance (max ULP 299) |
| test/contrib_ops/math/matmul_sparse_test/TestCoo_run0/model.onnx |  | Data/Data | ❌ | Unsupported value type 'sparse_tensor_type' for 'A'. Hint: export the model with tensor inputs/outputs. |
| test/contrib_ops/math/matmul_sparse_test/TestCoo_run1/model.onnx |  | Data/Data | ❌ | Unsupported value type 'sparse_tensor_type' for 'A'. Hint: export the model with tensor inputs/outputs. |
| test/contrib_ops/math/matmul_sparse_test/TestCoo_run2/model.onnx |  | Data/Data | ❌ | Unsupported value type 'sparse_tensor_type' for 'A'. Hint: export the model with tensor inputs/outputs. |
| test/contrib_ops/math/matmul_sparse_test/TestCoo_run3/model.onnx |  | Data/Data | ❌ | Unsupported value type 'sparse_tensor_type' for 'A'. Hint: export the model with tensor inputs/outputs. |
| test/contrib_ops/math/matmul_sparse_test/TestCsr_run0/model.onnx |  | Data/Data | ❌ | Unsupported value type 'sparse_tensor_type' for 'A'. Hint: export the model with tensor inputs/outputs. |
| test/contrib_ops/math/matmul_sparse_test/TestCsr_run1/model.onnx |  | Data/Data | ❌ | Unsupported value type 'sparse_tensor_type' for 'A'. Hint: export the model with tensor inputs/outputs. |
| test/contrib_ops/math/matmul_sparse_test/TestCsr_run2/model.onnx |  | Data/Data | ❌ | Unsupported value type 'sparse_tensor_type' for 'A'. Hint: export the model with tensor inputs/outputs. |
| test/contrib_ops/math/matmul_sparse_test/TestCsr_run3/model.onnx |  | Data/Data | ❌ | Unsupported value type 'sparse_tensor_type' for 'A'. Hint: export the model with tensor inputs/outputs. |
| test/contrib_ops/matmul_2bits_test/Float32_2Bits_Asymmetric_128x128_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2Bits_Asymmetric_128x256_BlkLen128_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2Bits_Asymmetric_256x256_BlkLen64_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2Bits_Asymmetric_256x256_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2Bits_Asymmetric_Batch32_256x256_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2Bits_Symmetric_128x128_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2Bits_Symmetric_128x256_BlkLen128_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2Bits_Symmetric_256x256_BlkLen64_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2Bits_Symmetric_256x256_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2Bits_Symmetric_Batch32_128x128_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy0_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy0_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy0_run10/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy0_run11/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy0_run12/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy0_run13/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy0_run14/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy0_run15/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy0_run16/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy0_run17/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy0_run18/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy0_run19/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy0_run2/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy0_run20/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy0_run21/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy0_run22/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy0_run23/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy0_run24/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy0_run25/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy0_run26/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy0_run27/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy0_run28/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy0_run29/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy0_run3/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy0_run30/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy0_run31/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy0_run32/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy0_run33/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy0_run34/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy0_run35/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy0_run36/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy0_run37/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy0_run38/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy0_run39/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy0_run4/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy0_run40/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy0_run41/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy0_run42/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy0_run43/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy0_run44/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy0_run45/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy0_run46/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy0_run47/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy0_run48/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy0_run49/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy0_run5/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy0_run50/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy0_run51/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy0_run52/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy0_run53/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy0_run54/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy0_run55/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy0_run56/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy0_run57/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy0_run58/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy0_run59/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy0_run6/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy0_run60/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy0_run61/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy0_run62/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy0_run63/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy0_run64/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy0_run65/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy0_run66/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy0_run67/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy0_run68/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy0_run69/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy0_run7/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy0_run70/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy0_run71/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy0_run72/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy0_run73/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy0_run74/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy0_run75/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy0_run76/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy0_run77/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy0_run78/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy0_run79/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy0_run8/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy0_run9/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy4_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy4_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy4_run10/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy4_run11/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy4_run12/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy4_run13/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy4_run14/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy4_run15/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy4_run16/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy4_run17/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy4_run18/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy4_run19/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy4_run2/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy4_run20/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy4_run21/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy4_run22/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy4_run23/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy4_run24/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy4_run25/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy4_run26/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy4_run27/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy4_run28/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy4_run29/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy4_run3/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy4_run30/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy4_run31/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy4_run32/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy4_run33/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy4_run34/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy4_run35/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy4_run36/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy4_run37/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy4_run38/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy4_run39/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy4_run4/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy4_run40/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy4_run41/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy4_run42/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy4_run43/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy4_run44/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy4_run45/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy4_run46/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy4_run47/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy4_run48/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy4_run49/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy4_run5/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy4_run50/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy4_run51/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy4_run52/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy4_run53/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy4_run54/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy4_run55/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy4_run56/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy4_run57/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy4_run58/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy4_run59/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy4_run6/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy4_run60/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy4_run61/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy4_run62/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy4_run63/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy4_run64/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy4_run65/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy4_run66/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy4_run67/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy4_run68/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy4_run69/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy4_run7/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy4_run70/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy4_run71/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy4_run72/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy4_run73/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy4_run74/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy4_run75/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy4_run76/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy4_run77/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy4_run78/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy4_run79/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy4_run8/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_2bits_test/Float32_2b_Accuracy4_run9/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy0_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy0_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy0_run10/model.onnx |  | Data/Data | ✅ | OK (max ULP 22) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy0_run11/model.onnx (--max-ulp 12000) |  | Data/Data | ✅ | OK (max ULP 135) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy0_run12/model.onnx |  | Data/Data | ✅ | OK (max ULP 22) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy0_run13/model.onnx |  | Data/Data | ✅ | OK (max ULP 29) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy0_run14/model.onnx (--max-ulp 12000) |  | Data/Data | ✅ | OK (max ULP 135) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy0_run15/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy0_run16/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy0_run17/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy0_run18/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy0_run19/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy0_run2/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy0_run20/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy0_run21/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy0_run22/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy0_run23/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy0_run24/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy0_run25/model.onnx (--max-ulp 12000) |  | Data/Data | ✅ | OK (max ULP 861) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy0_run26/model.onnx (--max-ulp 12000) |  | Data/Data | ✅ | OK (max ULP 7454) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy0_run27/model.onnx (--max-ulp 12000) |  | Data/Data | ✅ | OK (max ULP 861) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy0_run28/model.onnx (--max-ulp 12000) |  | Data/Data | ✅ | OK (max ULP 571) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy0_run29/model.onnx (--max-ulp 12000) |  | Data/Data | ✅ | OK (max ULP 7454) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy0_run3/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy0_run30/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy0_run31/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy0_run32/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy0_run33/model.onnx |  | Data/Data | ✅ | OK (max ULP 5) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy0_run34/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy0_run35/model.onnx (--max-ulp 12000) |  | Data/Data | ✅ | OK (max ULP 1730) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy0_run36/model.onnx (--max-ulp 12000) |  | Data/Data | ✅ | OK (max ULP 9141) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy0_run37/model.onnx (--max-ulp 12000) |  | Data/Data | ✅ | OK (max ULP 1730) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy0_run38/model.onnx (--max-ulp 12000) |  | Data/Data | ✅ | OK (max ULP 940) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy0_run39/model.onnx (--max-ulp 12000) |  | Data/Data | ✅ | OK (max ULP 9141) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy0_run4/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy0_run5/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy0_run6/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy0_run7/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy0_run8/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy0_run9/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run10/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run100/model.onnx (--max-ulp 12000) |  | Data/Data | ✅ | OK (max ULP 861) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run101/model.onnx (--max-ulp 12000) |  | Data/Data | ✅ | OK (max ULP 7454) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run102/model.onnx (--max-ulp 12000) |  | Data/Data | ✅ | OK (max ULP 861) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run103/model.onnx (--max-ulp 12000) |  | Data/Data | ✅ | OK (max ULP 571) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run104/model.onnx (--max-ulp 12000) |  | Data/Data | ✅ | OK (max ULP 7454) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run105/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run106/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run107/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run108/model.onnx |  | Data/Data | ✅ | OK (max ULP 5) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run109/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run11/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run110/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run111/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run112/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run113/model.onnx |  | Data/Data | ✅ | OK (max ULP 10) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run114/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run115/model.onnx (--max-ulp 12000) |  | Data/Data | ✅ | OK (max ULP 1730) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run116/model.onnx (--max-ulp 12000) |  | Data/Data | ✅ | OK (max ULP 9141) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run117/model.onnx (--max-ulp 12000) |  | Data/Data | ✅ | OK (max ULP 1730) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run118/model.onnx (--max-ulp 12000) |  | Data/Data | ✅ | OK (max ULP 940) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run119/model.onnx (--max-ulp 12000) |  | Data/Data | ✅ | OK (max ULP 9141) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run12/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run13/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run14/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run15/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run16/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run17/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run18/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run19/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run2/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run20/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run21/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run22/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run23/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run24/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run25/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run26/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run27/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run28/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run29/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run3/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run30/model.onnx |  | Data/Data | ✅ | OK (max ULP 21) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run31/model.onnx |  | Data/Data | ✅ | OK (max ULP 40) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run32/model.onnx |  | Data/Data | ✅ | OK (max ULP 21) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run33/model.onnx |  | Data/Data | ✅ | OK (max ULP 10) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run34/model.onnx |  | Data/Data | ✅ | OK (max ULP 40) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run35/model.onnx |  | Data/Data | ✅ | OK (max ULP 39) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run36/model.onnx |  | Data/Data | ✅ | OK (max ULP 36) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run37/model.onnx |  | Data/Data | ✅ | OK (max ULP 39) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run38/model.onnx |  | Data/Data | ✅ | OK (max ULP 31) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run39/model.onnx |  | Data/Data | ✅ | OK (max ULP 36) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run4/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run40/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run41/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run42/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run43/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run44/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run45/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run46/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run47/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run48/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run49/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run5/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run50/model.onnx |  | Data/Data | ✅ | OK (max ULP 22) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run51/model.onnx (--max-ulp 12000) |  | Data/Data | ✅ | OK (max ULP 135) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run52/model.onnx |  | Data/Data | ✅ | OK (max ULP 22) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run53/model.onnx |  | Data/Data | ✅ | OK (max ULP 29) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run54/model.onnx (--max-ulp 12000) |  | Data/Data | ✅ | OK (max ULP 135) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run55/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run56/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run57/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run58/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run59/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run6/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run60/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run61/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run62/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run63/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run64/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run65/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run66/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run67/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run68/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run69/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run7/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run70/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run71/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run72/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run73/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run74/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run75/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run76/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run77/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run78/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run79/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run8/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run80/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run81/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run82/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run83/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run84/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run85/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run86/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run87/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run88/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run89/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run9/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run90/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run91/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run92/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run93/model.onnx |  | Data/Data | ✅ | OK (max ULP 2) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run94/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run95/model.onnx (--max-ulp 12000) |  | Data/Data | ✅ | OK (max ULP 6080) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run96/model.onnx (--max-ulp 12000) |  | Data/Data | ✅ | OK (max ULP 9218) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run97/model.onnx (--max-ulp 12000) |  | Data/Data | ✅ | OK (max ULP 6080) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run98/model.onnx (--max-ulp 12000) |  | Data/Data | ✅ | OK (max ULP 2330) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run99/model.onnx (--max-ulp 12000) |  | Data/Data | ✅ | OK (max ULP 9218) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run10/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run100/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run101/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run102/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run103/model.onnx |  | Data/Data | ✅ | OK (max ULP 2) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run104/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run105/model.onnx (--max-ulp 12000) |  | Data/Data | ✅ | OK (max ULP 6080) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run106/model.onnx (--max-ulp 12000) |  | Data/Data | ✅ | OK (max ULP 9218) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run107/model.onnx (--max-ulp 12000) |  | Data/Data | ✅ | OK (max ULP 6080) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run108/model.onnx (--max-ulp 12000) |  | Data/Data | ✅ | OK (max ULP 2330) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run109/model.onnx (--max-ulp 12000) |  | Data/Data | ✅ | OK (max ULP 9218) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run11/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run110/model.onnx (--max-ulp 12000) |  | Data/Data | ✅ | OK (max ULP 861) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run111/model.onnx (--max-ulp 12000) |  | Data/Data | ✅ | OK (max ULP 7454) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run112/model.onnx (--max-ulp 12000) |  | Data/Data | ✅ | OK (max ULP 861) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run113/model.onnx (--max-ulp 12000) |  | Data/Data | ✅ | OK (max ULP 571) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run114/model.onnx (--max-ulp 12000) |  | Data/Data | ✅ | OK (max ULP 7454) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run115/model.onnx |  | Data/Data | ✅ | OK (max ULP 10) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run116/model.onnx |  | Data/Data | ✅ | OK (max ULP 5) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run117/model.onnx |  | Data/Data | ✅ | OK (max ULP 10) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run118/model.onnx (--max-ulp 12000) |  | Data/Data | ✅ | OK (max ULP 152) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run119/model.onnx |  | Data/Data | ✅ | OK (max ULP 5) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run12/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run120/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run121/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run122/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run123/model.onnx |  | Data/Data | ✅ | OK (max ULP 5) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run124/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run125/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run126/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run127/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run128/model.onnx |  | Data/Data | ✅ | OK (max ULP 10) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run129/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run13/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run130/model.onnx (--max-ulp 12000) |  | Data/Data | ✅ | OK (max ULP 1730) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run131/model.onnx (--max-ulp 12000) |  | Data/Data | ✅ | OK (max ULP 9141) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run132/model.onnx (--max-ulp 12000) |  | Data/Data | ✅ | OK (max ULP 1730) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run133/model.onnx (--max-ulp 12000) |  | Data/Data | ✅ | OK (max ULP 940) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run134/model.onnx (--max-ulp 12000) |  | Data/Data | ✅ | OK (max ULP 9141) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run135/model.onnx (--max-ulp 12000) |  | Data/Data | ✅ | OK (max ULP 10976) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run136/model.onnx (--max-ulp 12000) |  | Data/Data | ✅ | OK (max ULP 11529) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run137/model.onnx (--max-ulp 12000) |  | Data/Data | ✅ | OK (max ULP 10976) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run138/model.onnx (--max-ulp 12000) |  | Data/Data | ✅ | OK (max ULP 11961) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run139/model.onnx (--max-ulp 12000) |  | Data/Data | ✅ | OK (max ULP 11529) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run14/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run15/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run16/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run17/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run18/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run19/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run2/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run20/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run21/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run22/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run23/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run24/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run25/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run26/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run27/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run28/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run29/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run3/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run30/model.onnx |  | Data/Data | ✅ | OK (max ULP 21) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run31/model.onnx |  | Data/Data | ✅ | OK (max ULP 40) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run32/model.onnx |  | Data/Data | ✅ | OK (max ULP 21) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run33/model.onnx |  | Data/Data | ✅ | OK (max ULP 10) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run34/model.onnx |  | Data/Data | ✅ | OK (max ULP 40) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run35/model.onnx |  | Data/Data | ✅ | OK (max ULP 39) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run36/model.onnx |  | Data/Data | ✅ | OK (max ULP 36) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run37/model.onnx |  | Data/Data | ✅ | OK (max ULP 39) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run38/model.onnx |  | Data/Data | ✅ | OK (max ULP 31) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run39/model.onnx |  | Data/Data | ✅ | OK (max ULP 36) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run4/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run40/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run41/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run42/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run43/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run44/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run45/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run46/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run47/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run48/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run49/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run5/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run50/model.onnx |  | Data/Data | ✅ | OK (max ULP 22) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run51/model.onnx (--max-ulp 12000) |  | Data/Data | ✅ | OK (max ULP 135) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run52/model.onnx |  | Data/Data | ✅ | OK (max ULP 22) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run53/model.onnx |  | Data/Data | ✅ | OK (max ULP 29) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run54/model.onnx (--max-ulp 12000) |  | Data/Data | ✅ | OK (max ULP 135) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run55/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run56/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run57/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run58/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run59/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run6/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run60/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run61/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run62/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run63/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run64/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run65/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run66/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run67/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run68/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run69/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run7/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run70/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run71/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run72/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run73/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run74/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run75/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run76/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run77/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run78/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run79/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run8/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run80/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run81/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run82/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run83/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run84/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run85/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run86/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run87/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run88/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run89/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run9/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run90/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run91/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run92/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run93/model.onnx |  | Data/Data | ✅ | OK (max ULP 2) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run94/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run95/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run96/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run97/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run98/model.onnx |  | Data/Data | ✅ | OK (max ULP 3) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run99/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run10/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run100/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run101/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run102/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run103/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run104/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run105/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run106/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run107/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run108/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run109/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run11/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run110/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run111/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run112/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run113/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run114/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run115/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run116/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run117/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run118/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run119/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run12/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run13/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run14/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run15/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run16/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run17/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run18/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run19/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run2/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run20/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run21/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run22/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run23/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run24/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run25/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run26/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run27/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run28/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run29/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run3/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run30/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run31/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run32/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run33/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run34/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run35/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run36/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run37/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run38/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run39/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run4/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run40/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run41/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run42/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run43/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run44/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run45/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run46/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run47/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run48/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run49/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run5/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run50/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run51/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run52/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run53/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run54/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run55/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run56/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run57/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run58/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run59/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run6/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run60/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run61/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run62/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run63/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run64/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run65/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run66/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run67/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run68/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run69/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run7/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run70/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run71/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run72/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run73/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run74/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run75/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run76/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run77/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run78/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run79/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run8/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run80/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run81/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run82/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run83/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run84/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run85/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run86/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run87/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run88/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run89/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run9/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run90/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run91/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run92/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run93/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run94/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run95/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run96/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run97/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run98/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run99/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy1_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy1_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy1_run10/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy1_run11/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy1_run12/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy1_run13/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy1_run14/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy1_run15/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy1_run16/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy1_run17/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy1_run18/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy1_run19/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy1_run2/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy1_run20/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy1_run21/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy1_run22/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy1_run23/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy1_run24/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy1_run25/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy1_run26/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy1_run27/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy1_run28/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy1_run29/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy1_run3/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy1_run30/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy1_run31/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy1_run32/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy1_run33/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy1_run34/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy1_run35/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy1_run36/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy1_run37/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy1_run38/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy1_run39/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy1_run4/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy1_run5/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy1_run6/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy1_run7/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy1_run8/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy1_run9/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_Batch_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_Batch_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_Batch_run2/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_Batch_run3/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_Batch_run4/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_Batch_run5/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_Batch_run6/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_Batch_run7/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_Batch_run8/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run10/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run100/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run101/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run102/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run103/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run104/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run105/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run106/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run107/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run108/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run109/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run11/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run110/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run111/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run112/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run113/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run114/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run115/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run116/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run117/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run118/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run119/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run12/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run120/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run121/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run122/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run123/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run124/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run13/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run14/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run15/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run16/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run17/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run18/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run19/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run2/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run20/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run21/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run22/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run23/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run24/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run25/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run26/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run27/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run28/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run29/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run3/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run30/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run31/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run32/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run33/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run34/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run35/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run36/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run37/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run38/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run39/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run4/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run40/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run41/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run42/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run43/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run44/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run45/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run46/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run47/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run48/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run49/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run5/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run50/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run51/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run52/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run53/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run54/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run55/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run56/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run57/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run58/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run59/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run6/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run60/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run61/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run62/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run63/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run64/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run65/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run66/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run67/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run68/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run69/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run7/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run70/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run71/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run72/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run73/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run74/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run75/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run76/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run77/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run78/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run79/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run8/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run80/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run81/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run82/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run83/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run84/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run85/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run86/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run87/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run88/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run89/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run9/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run90/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run91/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run92/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run93/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run94/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run95/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run96/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run97/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run98/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run99/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/LegacyShape_4b_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/LegacyShape_4b_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/LegacyShape_4b_run2/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/LegacyShape_4b_run3/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/LegacyShape_4b_run4/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/LegacyShape_4b_run5/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/LegacyShape_4b_run6/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/LegacyShape_4b_run7/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_4bits_test/LegacyShape_4b_run8/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/matmul_4bits_test/LegacyShape_4b_run9/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run10/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run100/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run101/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run102/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run103/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run104/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run105/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run106/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run107/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run108/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run109/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run11/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run110/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run111/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run112/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run113/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run114/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run115/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run116/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run117/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run118/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run119/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run12/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run120/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run121/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run122/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run123/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run124/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run125/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run126/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run127/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run128/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run129/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run13/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run130/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run131/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run14/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run15/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run16/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run17/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run18/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run19/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run2/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run20/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run21/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run22/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run23/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run24/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run25/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run26/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run27/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run28/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run29/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run3/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run30/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run31/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run32/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run33/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run34/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run35/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run36/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run37/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run38/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run39/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run4/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run40/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run41/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run42/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run43/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run44/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run45/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run46/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run47/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run48/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run49/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run5/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run50/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run51/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run52/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run53/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run54/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run55/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run56/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run57/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run58/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run59/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run6/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run60/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run61/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run62/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run63/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run64/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run65/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run66/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run67/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run68/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run69/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run7/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run70/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run71/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run72/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run73/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run74/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run75/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run76/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run77/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run78/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run79/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run8/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run80/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run81/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run82/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run83/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run84/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run85/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run86/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run87/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run88/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run89/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run9/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run90/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run91/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run92/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run93/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run94/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run95/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run96/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run97/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run98/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel1_run99/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run10/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run100/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run101/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run102/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run103/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run104/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run105/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run106/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run107/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run108/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run109/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run11/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run110/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run111/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run112/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run113/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run114/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run115/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run116/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run117/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run118/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run119/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run12/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run120/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run121/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run122/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run123/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run124/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run125/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run126/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run127/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run128/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run129/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run13/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run130/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run131/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run132/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run133/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run134/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run135/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run14/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run15/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run16/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run17/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run18/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run19/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run2/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run20/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run21/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run22/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run23/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run24/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run25/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run26/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run27/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run28/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run29/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run3/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run30/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run31/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run32/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run33/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run34/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run35/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run36/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run37/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run38/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run39/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run4/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run40/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run41/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run42/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run43/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run44/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run45/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run46/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run47/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run48/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run49/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run5/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run50/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run51/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run52/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run53/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run54/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run55/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run56/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run57/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run58/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run59/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run6/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run60/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run61/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run62/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run63/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run64/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run65/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run66/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run67/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run68/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run69/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run7/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run70/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run71/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run72/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run73/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run74/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run75/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run76/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run77/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run78/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run79/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run8/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run80/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run81/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run82/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run83/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run84/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run85/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run86/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run87/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run88/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run89/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run9/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run90/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run91/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run92/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run93/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run94/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run95/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run96/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run97/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run98/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_8bits_test/Float32_8b_AccuracyLevel4_run99/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run10/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run100/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run101/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run102/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run103/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run104/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run105/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run106/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run107/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run108/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run109/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run11/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run110/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run111/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run112/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run113/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run114/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run115/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run116/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run117/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run118/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run119/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run12/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run120/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run121/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run122/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run123/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run124/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run125/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run126/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run127/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run128/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run129/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run13/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run130/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run131/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run132/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run133/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run134/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run135/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run136/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run137/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run138/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run139/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run14/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run140/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run141/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run142/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run143/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run144/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run145/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run146/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run147/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run148/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run149/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run15/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run150/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run151/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run152/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run153/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run154/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run155/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run156/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run157/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run158/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run159/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run16/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run160/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run161/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run162/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run163/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run164/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run165/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run166/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run167/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run168/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run169/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run17/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run170/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run171/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run172/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run173/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run174/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run175/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run176/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run177/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run178/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run179/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run18/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run180/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run181/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run182/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run183/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run184/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run185/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run186/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run187/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run188/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run189/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run19/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run190/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run191/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run192/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run193/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run194/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run195/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run196/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run197/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run198/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run199/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run2/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run20/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run200/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run201/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run202/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run203/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run204/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run205/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run206/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run207/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run208/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run209/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run21/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run210/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run211/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run212/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run213/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run214/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run215/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run216/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run217/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run218/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run219/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run22/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run220/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run221/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run222/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run223/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run224/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run225/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run226/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run227/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run228/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run229/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run23/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run230/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run231/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run232/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run233/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run234/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run235/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run236/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run237/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run238/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run239/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run24/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run240/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run241/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run242/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run243/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run244/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run245/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run246/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run247/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run248/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run249/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run25/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run250/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run251/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run252/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run253/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run254/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run255/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run256/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run257/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run258/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run259/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run26/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run260/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run261/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run262/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run263/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run264/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run265/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run266/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run267/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run268/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run269/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run27/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run270/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run271/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run272/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run273/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run274/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run275/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run276/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run277/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run278/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run279/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run28/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run280/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run281/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run282/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run283/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run284/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run285/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run286/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run287/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run288/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run289/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run29/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run290/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run291/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run292/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run293/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run294/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run295/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run296/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run297/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run298/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run299/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run3/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run30/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run300/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run301/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run302/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run303/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run304/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run305/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run306/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run307/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run308/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run309/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run31/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run310/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run311/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run312/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run313/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run314/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run315/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run316/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run317/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run318/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run319/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run32/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run320/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run321/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run322/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run323/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run324/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run325/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run326/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run327/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run328/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run329/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run33/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run330/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run331/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run332/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run333/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run334/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run335/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run336/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run337/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run338/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run339/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run34/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run340/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run341/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run342/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run343/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run344/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run345/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run346/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run347/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run348/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run349/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run35/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run350/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run351/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run352/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run353/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run354/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run355/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run356/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run357/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run358/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run359/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run36/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run360/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run361/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run362/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run363/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run364/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run365/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run366/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run367/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run368/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run369/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run37/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run370/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run371/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run372/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run373/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run374/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run375/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run376/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run377/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run378/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run379/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run38/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run380/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run381/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run382/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run383/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run384/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run385/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run386/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run387/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run388/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run389/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run39/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run390/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run391/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run392/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run393/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run394/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run395/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run396/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run397/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run398/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run399/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run4/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run40/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run400/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run401/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run402/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run403/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run404/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run405/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run406/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run407/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run408/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run409/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run41/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run410/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run411/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run412/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run413/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run414/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run415/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run416/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run417/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run418/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run419/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run42/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run420/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run421/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run422/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run423/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run424/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run425/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run426/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run427/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run428/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run429/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run43/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run430/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run431/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run432/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run433/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run434/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run435/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run436/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run437/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run438/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run439/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run44/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run440/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run441/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run442/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run443/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run444/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run445/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run446/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run447/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run448/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run449/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run45/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run450/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run451/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run452/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run453/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run454/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run455/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run456/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run457/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run458/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run459/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run46/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run460/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run461/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run462/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run463/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run464/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run465/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run466/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run467/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run468/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run469/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run47/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run470/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run471/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run472/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run473/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run474/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run475/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run476/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run477/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run478/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run479/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run48/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run480/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run481/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run482/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run483/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run484/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run485/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run486/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run487/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run488/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run489/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run49/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run490/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run491/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run492/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run493/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run494/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run495/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run496/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run497/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run498/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run499/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run5/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run50/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run500/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run501/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run502/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run503/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run504/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run505/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run506/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run507/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run508/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run509/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run51/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run510/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run511/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run512/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run513/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run514/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run515/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run516/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run517/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run518/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run519/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run52/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run520/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run521/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run522/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run523/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run524/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run525/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run526/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run527/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run528/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run529/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run53/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run530/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run531/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run532/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run533/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run534/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run535/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run536/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run537/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run538/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run539/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run54/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run540/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run541/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run542/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run543/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run544/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run545/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run546/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run547/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run548/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run549/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run55/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run550/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run551/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run552/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run553/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run554/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run555/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run556/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run557/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run558/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run559/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run56/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run560/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run561/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run562/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run563/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run564/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run565/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run566/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run567/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run568/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run569/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run57/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run570/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run571/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run572/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run573/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run574/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run575/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run576/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run577/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run578/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run579/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run58/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run580/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run581/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run582/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run583/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run584/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run585/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run586/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run587/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run588/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run589/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run59/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run590/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run591/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run592/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run593/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run594/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run595/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run596/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run597/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run598/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run599/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run6/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run60/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run600/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run601/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run602/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run603/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run604/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run605/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run606/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run607/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run608/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run609/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run61/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run610/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run611/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run612/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run613/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run614/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run615/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run616/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run617/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run618/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run619/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run62/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run620/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run621/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run622/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run623/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run624/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run625/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run626/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run627/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run628/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run629/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run63/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run630/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run631/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run632/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run633/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run634/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run635/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run636/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run637/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run638/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run639/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run64/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run640/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run641/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run642/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run643/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run644/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run645/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run646/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run647/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run648/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run649/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run65/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run650/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run651/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run652/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run653/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run654/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run655/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run656/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run657/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run658/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run659/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run66/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run660/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run661/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run662/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run663/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run664/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run665/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run666/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run667/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run668/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run669/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run67/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run670/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run671/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run672/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run673/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run674/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run675/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run676/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run677/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run678/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run679/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run68/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run680/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run681/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run682/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run683/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run684/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run685/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run686/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run687/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run688/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run689/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run69/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run690/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run691/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run692/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run693/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run694/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run695/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run696/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run697/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run698/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run699/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run7/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run70/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run700/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run701/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run702/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run703/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run704/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run705/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run706/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run707/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run708/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run709/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run71/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run710/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run711/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run712/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run713/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run714/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run715/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run716/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run717/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run718/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run719/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run72/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run720/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run721/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run722/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run723/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run724/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run725/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run726/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run727/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run728/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run729/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run73/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run730/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run731/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run732/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run733/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run734/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run735/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run736/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run737/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run738/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run739/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run74/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run740/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run741/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run742/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run743/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run744/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run745/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run746/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run747/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run748/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run749/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run75/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run750/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run751/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run752/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run753/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run754/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run755/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run756/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run757/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run758/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run759/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run76/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run760/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run761/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run762/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run763/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run764/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run765/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run766/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run767/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run77/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run78/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run79/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run8/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run80/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run81/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run82/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run83/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run84/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run85/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run86/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run87/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run88/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run89/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run9/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run90/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run91/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run92/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run93/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run94/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run95/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run96/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run97/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run98/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_bnb4_test/Float32_run99/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_integer16_test/MatMulInteger16_1_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MatMulInteger16 |
| test/contrib_ops/matmul_integer16_test/MatMulInteger16_2_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MatMulInteger16 |
| test/contrib_ops/matmul_integer16_test/MatMulInteger16_3_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MatMulInteger16 |
| test/contrib_ops/matmul_integer16_test/MatMulInteger16_Empty_input_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MatMulInteger16 |
| test/contrib_ops/matmul_integer_to_float_test/HasZeroPoint_HasBias_test_S8S8_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_integer_to_float_test/HasZeroPoint_HasBias_test_S8S8_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_integer_to_float_test/HasZeroPoint_HasBias_test_S8S8_run2/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_integer_to_float_test/HasZeroPoint_HasBias_test_S8S8_run3/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_integer_to_float_test/HasZeroPoint_HasBias_test_U8S8_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_integer_to_float_test/HasZeroPoint_HasBias_test_U8S8_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_integer_to_float_test/HasZeroPoint_HasBias_test_U8S8_run2/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_integer_to_float_test/HasZeroPoint_HasBias_test_U8S8_run3/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_integer_to_float_test/HasZeroPoint_HasBias_test_U8X8_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_integer_to_float_test/HasZeroPoint_HasBias_test_U8X8_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_integer_to_float_test/HasZeroPoint_HasBias_test_U8X8_run2/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_integer_to_float_test/HasZeroPoint_HasBias_test_U8X8_run3/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_integer_to_float_test/HasZeroPoint_NoBias_test_S8S8_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_integer_to_float_test/HasZeroPoint_NoBias_test_S8S8_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_integer_to_float_test/HasZeroPoint_NoBias_test_S8S8_run2/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_integer_to_float_test/HasZeroPoint_NoBias_test_S8S8_run3/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_integer_to_float_test/HasZeroPoint_NoBias_test_U8S8_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_integer_to_float_test/HasZeroPoint_NoBias_test_U8S8_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_integer_to_float_test/HasZeroPoint_NoBias_test_U8S8_run2/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_integer_to_float_test/HasZeroPoint_NoBias_test_U8S8_run3/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_integer_to_float_test/HasZeroPoint_NoBias_test_U8U8_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_integer_to_float_test/HasZeroPoint_NoBias_test_U8U8_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_integer_to_float_test/HasZeroPoint_NoBias_test_U8U8_run2/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_integer_to_float_test/HasZeroPoint_NoBias_test_U8U8_run3/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_integer_to_float_test/NoZeroPoint_HasBias_test_S8S8_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_integer_to_float_test/NoZeroPoint_HasBias_test_S8S8_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_integer_to_float_test/NoZeroPoint_HasBias_test_S8S8_run2/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_integer_to_float_test/NoZeroPoint_HasBias_test_S8S8_run3/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_integer_to_float_test/NoZeroPoint_HasBias_test_U8S8_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_integer_to_float_test/NoZeroPoint_HasBias_test_U8S8_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_integer_to_float_test/NoZeroPoint_HasBias_test_U8S8_run2/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_integer_to_float_test/NoZeroPoint_HasBias_test_U8S8_run3/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_integer_to_float_test/NoZeroPoint_HasBias_test_U8U8_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_integer_to_float_test/NoZeroPoint_HasBias_test_U8U8_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_integer_to_float_test/NoZeroPoint_HasBias_test_U8U8_run2/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_integer_to_float_test/NoZeroPoint_HasBias_test_U8U8_run3/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_integer_to_float_test/NoZeroPoint_NoBias_test_S8S8_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_integer_to_float_test/NoZeroPoint_NoBias_test_S8S8_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_integer_to_float_test/NoZeroPoint_NoBias_test_S8S8_run2/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_integer_to_float_test/NoZeroPoint_NoBias_test_S8S8_run3/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_integer_to_float_test/NoZeroPoint_NoBias_test_U8S8_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_integer_to_float_test/NoZeroPoint_NoBias_test_U8S8_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_integer_to_float_test/NoZeroPoint_NoBias_test_U8S8_run2/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_integer_to_float_test/NoZeroPoint_NoBias_test_U8S8_run3/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_integer_to_float_test/NoZeroPoint_NoBias_test_U8U8_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_integer_to_float_test/NoZeroPoint_NoBias_test_U8U8_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_integer_to_float_test/NoZeroPoint_NoBias_test_U8U8_run2/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/matmul_integer_to_float_test/NoZeroPoint_NoBias_test_U8U8_run3/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/maxpool_mask_test/MaxPoolWithMask_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MaxpoolWithMask |
| test/contrib_ops/moe_test/MoECpuTest_BasicSwiGLU_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MoE |
| test/contrib_ops/multihead_attention_op_test/CrossAttentionWithPast_run0/model.onnx (--test-data-inputs-only) |  | Data/ORT | ✅ | OK (max ULP 0) |
| test/contrib_ops/multihead_attention_op_test/CrossAttentionWithPast_run1/model.onnx (--test-data-inputs-only) |  | Data/ORT | ✅ | OK (max ULP 0) |
| test/contrib_ops/multihead_attention_op_test/CrossAttention_Batch1_HeadSize16_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 2) |
| test/contrib_ops/multihead_attention_op_test/CrossAttention_Batch1_HeadSize16_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 2) |
| test/contrib_ops/multihead_attention_op_test/CrossAttention_Batch1_HeadSize16_run2/model.onnx |  | Data/Data | ✅ | OK (max ULP 4) |
| test/contrib_ops/multihead_attention_op_test/CrossAttention_Batch1_HeadSize16_run3/model.onnx |  | Data/Data | ✅ | OK (max ULP 4) |
| test/contrib_ops/multihead_attention_op_test/CrossAttention_Batch1_HeadSize8_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/multihead_attention_op_test/CrossAttention_Batch1_HeadSize8_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/multihead_attention_op_test/CrossAttention_Batch2_HeadSize16_8_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 20) |
| test/contrib_ops/multihead_attention_op_test/CrossAttention_Batch2_HeadSize16_8_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 20) |
| test/contrib_ops/multihead_attention_op_test/CrossAttention_Batch2_HeadSize16_8_run2/model.onnx |  | Data/Data | ✅ | OK (max ULP 7) |
| test/contrib_ops/multihead_attention_op_test/CrossAttention_Batch2_HeadSize16_8_run3/model.onnx |  | Data/Data | ✅ | OK (max ULP 7) |
| test/contrib_ops/multihead_attention_op_test/CrossAttention_Batch2_HeadSize40_run0/model.onnx (--test-data-inputs-only --max-ulp 20000) |  | Data/ORT | ✅ | OK (max ULP 16414) |
| test/contrib_ops/multihead_attention_op_test/CrossAttention_Batch2_HeadSize40_run1/model.onnx (--test-data-inputs-only --max-ulp 20000) |  | Data/ORT | ✅ | OK (max ULP 16414) |
| test/contrib_ops/multihead_attention_op_test/CrossAttention_Batch2_HeadSize40_run2/model.onnx (--test-data-inputs-only --max-ulp 500) |  | Data/ORT | ✅ | OK (max ULP 388) |
| test/contrib_ops/multihead_attention_op_test/CrossAttention_Batch2_HeadSize40_run3/model.onnx (--test-data-inputs-only --max-ulp 500) |  | Data/ORT | ✅ | OK (max ULP 388) |
| test/contrib_ops/multihead_attention_op_test/CrossAttention_DiffSequenceLengths_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/multihead_attention_op_test/CrossAttention_DiffSequenceLengths_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/multihead_attention_op_test/CrossAttention_DiffSequenceLengths_run2/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/multihead_attention_op_test/CrossAttention_DiffSequenceLengths_run3/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/multihead_attention_op_test/CrossAttention_DiffSequenceLengths_run4/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/multihead_attention_op_test/CrossAttention_DiffSequenceLengths_run5/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/multihead_attention_op_test/SelfAttention_WithPastAndPresent_NoMask_NoRelPosBias_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/multihead_attention_op_test/SelfAttention_WithPastAndPresent_NoMask_NoRelPosBias_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/multihead_attention_op_test/SelfAttention_WithPastAndPresent_NoMask_NoRelPosBias_run2/model.onnx |  | Data/Data | ✅ | OK (max ULP 24) |
| test/contrib_ops/multihead_attention_op_test/SelfAttention_WithPastAndPresent_NoMask_NoRelPosBias_run3/model.onnx |  | Data/Data | ✅ | OK (max ULP 24) |
| test/contrib_ops/multihead_attention_op_test/SelfAttention_WithPastAndPresent_NoMask_NoRelPosBias_run4/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/multihead_attention_op_test/SelfAttention_WithPastAndPresent_NoMask_NoRelPosBias_run5/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/murmur_hash3_test/DefaultSeed_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/murmur_hash3_test/MoreDataFloat_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/murmur_hash3_test/MoreDataInt_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/murmur_hash3_test/MultipleStringsKeyUIntResult_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/murmur_hash3_test/NonZeroSeedUIntResult_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/murmur_hash3_test/NonZeroSeed_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/murmur_hash3_test/StringKeyIntResult_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/murmur_hash3_test/StringKeyIntWithSeed42_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/murmur_hash3_test/StringKeyUIntResult_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/murmur_hash3_test/StringKeyUIntWithSeed42_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/murmur_hash3_test/UnsupportedInputType_run0/model.onnx |  | Data/Data | ❌ | MurmurHash3 does not support input dtype int8; supported: int32, int64, float, double, string |
| test/contrib_ops/murmur_hash3_test/ZeroSeedDoubleResult_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/murmur_hash3_test/ZeroSeedFloatResult_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/murmur_hash3_test/ZeroSeedUIntResult2_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/murmur_hash3_test/ZeroSeedUIntResult3_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/murmur_hash3_test/ZeroSeedUIntResult_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/murmur_hash3_test/ZeroSeed_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/ngram_repeat_block_op_test/NGramSize_3_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.NGramRepeatBlock |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run1/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run10/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run11/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run12/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run13/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run14/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run15/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run16/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run17/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run18/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run19/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run2/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run20/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run21/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run22/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run23/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run24/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run25/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run26/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run27/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run28/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run29/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run3/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run30/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run31/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run32/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run33/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run34/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run35/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run36/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run37/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run38/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run39/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run4/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run40/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run41/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run42/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run43/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run44/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run45/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run46/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run47/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run48/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run49/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run5/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run50/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run51/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run52/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run53/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run54/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run55/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run56/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run57/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run58/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run59/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run6/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run60/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run61/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run62/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run63/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run64/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run65/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run66/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run67/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run68/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run69/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run7/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run70/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run71/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run72/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run73/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run74/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run75/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run76/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run77/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run78/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run79/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run8/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run80/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run81/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run82/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run83/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run84/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run85/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run86/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run87/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run88/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run89/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run9/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run90/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run91/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run92/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run1/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run10/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run11/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run12/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run13/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run14/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run15/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run16/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run17/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run18/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run19/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run2/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run20/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run21/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run22/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run23/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run24/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run25/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run26/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run27/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run28/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run29/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run3/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run30/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run31/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run32/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run33/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run34/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run35/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run36/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run37/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run38/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run39/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run4/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run40/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run41/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run42/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run43/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run44/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run45/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run46/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run47/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run48/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run49/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run5/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run50/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run51/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run52/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run53/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run54/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run55/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run56/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run57/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run58/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run59/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run6/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run60/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run61/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run62/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run63/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run64/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run65/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run66/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run67/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run68/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run69/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run7/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run70/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run71/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run72/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run73/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run74/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run75/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run76/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run77/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run78/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run79/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run8/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run80/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run81/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run82/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run83/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run84/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run85/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run86/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run87/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run88/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run89/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run9/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run90/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run91/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run92/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run1/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run10/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run11/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run12/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run13/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run14/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run15/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run16/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run17/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run18/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run19/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run2/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run20/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run21/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run22/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run23/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run24/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run25/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run26/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run27/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run28/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run29/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run3/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run30/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run31/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run32/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run33/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run34/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run35/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run36/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run37/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run38/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run39/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run4/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run40/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run41/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run42/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run43/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run44/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run45/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run46/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run47/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run48/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run49/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run5/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run50/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run51/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run52/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run53/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run54/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run55/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run56/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run57/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run58/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run59/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run6/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run60/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run61/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run62/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run63/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run64/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run65/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run66/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run67/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run68/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run69/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run7/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run70/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run71/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run72/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run73/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run74/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run75/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run76/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run77/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run78/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run79/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run8/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run80/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run81/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run82/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run83/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run84/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run85/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run86/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run87/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run88/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run89/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run9/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run90/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run91/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run92/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run1/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run10/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run11/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run12/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run13/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run14/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run15/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run16/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run17/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run18/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run19/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run2/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run20/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run21/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run22/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run23/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run24/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run25/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run26/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run27/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run28/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run29/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run3/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run30/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run31/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run32/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run33/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run34/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run35/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run36/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run37/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run38/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run39/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run4/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run40/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run41/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run42/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run43/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run44/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run45/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run46/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run47/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run48/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run49/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run5/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run50/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run51/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run52/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run53/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run54/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run55/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run56/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run57/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run58/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run59/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run6/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run60/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run61/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run62/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run63/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run64/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run65/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run66/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run67/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run68/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run69/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run7/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run70/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run71/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run72/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run73/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run74/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run75/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run76/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run77/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run78/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run79/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run8/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run80/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run81/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run82/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run83/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run84/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run85/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run86/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run87/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run88/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run89/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run9/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run90/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run91/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run92/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run1/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run10/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run11/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run12/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run13/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run14/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run15/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run16/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run17/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run18/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run19/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run2/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run20/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run21/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run22/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run23/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run24/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run25/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run26/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run27/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run28/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run29/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run3/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run30/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run31/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run32/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run33/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run34/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run35/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run36/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run37/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run38/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run39/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run4/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run40/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run41/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run42/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run43/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run44/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run45/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run46/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run47/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run48/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run49/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run5/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run50/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run51/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run52/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run53/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run54/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run55/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run56/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run57/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run58/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run59/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run6/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run60/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run61/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run62/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run63/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run64/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run65/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run66/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run67/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run68/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run69/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run7/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run70/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run71/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run72/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run73/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run74/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run75/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run76/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run77/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run78/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run79/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run8/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run80/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run81/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run82/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run83/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run84/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run85/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run86/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run87/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run88/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run89/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run9/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run90/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run91/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run92/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run1/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run10/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run11/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run12/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run13/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run14/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run15/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run16/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run17/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run18/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run19/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run2/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run20/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run21/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run22/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run23/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run24/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run25/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run26/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run27/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run28/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run29/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run3/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run30/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run31/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run32/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run33/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run34/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run35/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run36/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run37/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run38/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run39/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run4/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run40/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run41/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run42/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run43/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run44/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run45/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run46/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run47/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run48/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run49/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run5/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run50/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run51/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run52/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run53/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run54/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run55/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run56/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run57/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run58/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run59/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run6/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run60/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run61/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run62/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run63/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run64/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run65/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run66/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run67/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run68/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run69/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run7/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run70/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run71/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run72/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run73/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run74/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run75/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run76/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run77/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run78/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run79/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run8/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run80/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run81/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run82/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run83/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run84/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run85/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run86/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run87/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run88/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run89/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run9/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run90/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run91/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run92/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPoolDilations_S8_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPoolDilations_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPoolStrides_S8_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPoolStrides_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qembed_layer_norm_op_test/EmbedLayerNormBatch1_Float16_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QEmbedLayerNormalization |
| test/contrib_ops/qembed_layer_norm_op_test/EmbedLayerNormBatch1_Float16_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QEmbedLayerNormalization |
| test/contrib_ops/qembed_layer_norm_op_test/EmbedLayerNormBatch1_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QEmbedLayerNormalization |
| test/contrib_ops/qembed_layer_norm_op_test/EmbedLayerNormBatch1_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QEmbedLayerNormalization |
| test/contrib_ops/qembed_layer_norm_op_test/EmbedLayerNormBatch2_NoMask_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QEmbedLayerNormalization |
| test/contrib_ops/qembed_layer_norm_op_test/EmbedLayerNormBatch2_NoMask_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QEmbedLayerNormalization |
| test/contrib_ops/qembed_layer_norm_op_test/EmbedLayerNormBatch2_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QEmbedLayerNormalization |
| test/contrib_ops/qembed_layer_norm_op_test/EmbedLayerNormBatch2_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QEmbedLayerNormalization |
| test/contrib_ops/qembed_layer_norm_op_test/EmbedLayerNormBatch_Distill_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QEmbedLayerNormalization |
| test/contrib_ops/qembed_layer_norm_op_test/EmbedLayerNormBatch_Distill_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QEmbedLayerNormalization |
| test/contrib_ops/qembed_layer_norm_op_test/EmbedLayerNormLargeBatchSmallHiddenSize_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QEmbedLayerNormalization |
| test/contrib_ops/qembed_layer_norm_op_test/EmbedLayerNormLargeBatchSmallHiddenSize_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QEmbedLayerNormalization |
| test/contrib_ops/qlinear_binary_op_test/AddS8ScalarVectorBroadcast_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_binary_op_test/AddS8ScalarVectorBroadcast_run1/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_binary_op_test/AddS8ScalarVectorBroadcast_run2/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_binary_op_test/AddS8ScalarVectorFull_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_binary_op_test/AddS8ScalarVectorFull_run1/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_binary_op_test/AddS8ScalarVectorFull_run2/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_binary_op_test/AddS8VectorScalarBroadcast_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_binary_op_test/AddS8VectorScalarBroadcast_run1/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_binary_op_test/AddS8VectorScalarBroadcast_run2/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_binary_op_test/AddS8VectorScalarFull_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_binary_op_test/AddS8VectorScalarFull_run1/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_binary_op_test/AddS8VectorScalarFull_run2/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_binary_op_test/AddS8VectorVectorBroadcast_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_binary_op_test/AddS8VectorVectorBroadcast_run1/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_binary_op_test/AddS8VectorVectorBroadcast_run2/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_binary_op_test/AddS8VectorVectorFull_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_binary_op_test/AddS8VectorVectorFull_run1/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_binary_op_test/AddS8VectorVectorFull_run2/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_binary_op_test/AddU8ScalarVectorBroadcast_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_binary_op_test/AddU8ScalarVectorBroadcast_run1/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_binary_op_test/AddU8ScalarVectorBroadcast_run2/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_binary_op_test/AddU8ScalarVectorFull_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_binary_op_test/AddU8ScalarVectorFull_run1/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_binary_op_test/AddU8ScalarVectorFull_run2/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_binary_op_test/AddU8VectorScalarBroadcast_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_binary_op_test/AddU8VectorScalarBroadcast_run1/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_binary_op_test/AddU8VectorScalarBroadcast_run2/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_binary_op_test/AddU8VectorScalarFull_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_binary_op_test/AddU8VectorScalarFull_run1/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_binary_op_test/AddU8VectorScalarFull_run2/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_binary_op_test/AddU8VectorVectorBroadcast_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_binary_op_test/AddU8VectorVectorBroadcast_run1/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_binary_op_test/AddU8VectorVectorBroadcast_run2/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_binary_op_test/AddU8VectorVectorFull_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_binary_op_test/AddU8VectorVectorFull_run1/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_binary_op_test/AddU8VectorVectorFull_run2/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_binary_op_test/MulS8ScalarVectorBroadcast_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_binary_op_test/MulS8ScalarVectorBroadcast_run1/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_binary_op_test/MulS8ScalarVectorBroadcast_run2/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_binary_op_test/MulS8ScalarVectorFull_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_binary_op_test/MulS8ScalarVectorFull_run1/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_binary_op_test/MulS8ScalarVectorFull_run2/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_binary_op_test/MulS8VectorScalarBroadcast_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_binary_op_test/MulS8VectorScalarBroadcast_run1/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_binary_op_test/MulS8VectorScalarBroadcast_run2/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_binary_op_test/MulS8VectorScalarFull_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_binary_op_test/MulS8VectorScalarFull_run1/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_binary_op_test/MulS8VectorScalarFull_run2/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_binary_op_test/MulS8VectorVectorBroadcast_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_binary_op_test/MulS8VectorVectorBroadcast_run1/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_binary_op_test/MulS8VectorVectorBroadcast_run2/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_binary_op_test/MulS8VectorVectorFull_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_binary_op_test/MulS8VectorVectorFull_run1/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_binary_op_test/MulS8VectorVectorFull_run2/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_binary_op_test/MulU8ScalarVectorBroadcast_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_binary_op_test/MulU8ScalarVectorBroadcast_run1/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_binary_op_test/MulU8ScalarVectorBroadcast_run2/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_binary_op_test/MulU8ScalarVectorFull_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_binary_op_test/MulU8ScalarVectorFull_run1/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_binary_op_test/MulU8ScalarVectorFull_run2/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_binary_op_test/MulU8VectorScalarBroadcast_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_binary_op_test/MulU8VectorScalarBroadcast_run1/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_binary_op_test/MulU8VectorScalarBroadcast_run2/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_binary_op_test/MulU8VectorScalarFull_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_binary_op_test/MulU8VectorScalarFull_run1/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_binary_op_test/MulU8VectorScalarFull_run2/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_binary_op_test/MulU8VectorVectorBroadcast_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_binary_op_test/MulU8VectorVectorBroadcast_run1/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_binary_op_test/MulU8VectorVectorBroadcast_run2/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_binary_op_test/MulU8VectorVectorFull_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_binary_op_test/MulU8VectorVectorFull_run1/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_binary_op_test/MulU8VectorVectorFull_run2/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_concat_test/ExpectFail_MixedConstDynamic_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_concat_test/ExpectFail_MixedConstDynamic_run1/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_concat_test/ExpectFail_MixedConstDynamic_run2/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_concat_test/ExpectFail_MixedConstDynamic_run3/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_concat_test/ExpectFail_MixedConstDynamic_run4/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_concat_test/ExpectFail_MixedConstDynamic_run5/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_concat_test/ExpectFail_MixedConstDynamic_run6/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_concat_test/ExpectFail_MixedConstDynamic_run7/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_concat_test/ExpectFail_WrongScaleType_0_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_concat_test/ExpectFail_WrongScaleType_0_run1/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_concat_test/ExpectFail_WrongScaleType_1_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_concat_test/ExpectFail_WrongScaleType_1_run1/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_concat_test/ExpectFail_WrongTensorType_0_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_concat_test/ExpectFail_WrongTensorType_0_run1/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_concat_test/ExpectFail_WrongTensorType_1_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_concat_test/ExpectFail_WrongTensorType_1_run1/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_concat_test/ExpectFail_WrongZeroPointType_0_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_concat_test/ExpectFail_WrongZeroPointType_0_run1/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_concat_test/ExpectFail_WrongZeroPointType_1_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_concat_test/ExpectFail_WrongZeroPointType_1_run1/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_concat_test/Input3_ConstConstConst_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_concat_test/Input3_ConstConstConst_run1/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_concat_test/Input3_DynamicDynamicDynamic_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_concat_test/Input3_DynamicDynamicDynamic_run1/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_concat_test/Input3_MixedConstDynamic_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_concat_test/Input3_MixedConstDynamic_run1/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_concat_test/Input3_MixedConstDynamic_run2/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_concat_test/Input3_MixedConstDynamic_run3/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_concat_test/Input3_MixedConstDynamic_run4/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_concat_test/Input3_MixedConstDynamic_run5/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_concat_test/Input3_MixedConstDynamic_run6/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_concat_test/Input3_MixedConstDynamic_run7/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_concat_test/InputOne_Const_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_concat_test/InputOne_Const_run1/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_concat_test/InputOne_Dynamic_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_concat_test/InputOne_Dynamic_run1/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_global_average_pool_test/Nchw_1x32x32x1_S8_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_global_average_pool_test/Nchw_1x32x32x1_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_global_average_pool_test/Nchw_1x7x7x255_S8_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_global_average_pool_test/Nchw_1x7x7x255_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_global_average_pool_test/Nchw_1x7x7x256_S8_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_global_average_pool_test/Nchw_1x7x7x256_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_global_average_pool_test/Nchw_1x8x8x255_S8_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_global_average_pool_test/Nchw_1x8x8x255_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_global_average_pool_test/Nchw_1x8x8x256_S8_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_global_average_pool_test/Nchw_1x8x8x256_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_global_average_pool_test/Nchw_3x7x7x255_S8_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_global_average_pool_test/Nchw_3x7x7x255_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_global_average_pool_test/Nchw_3x7x7x256_S8_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_global_average_pool_test/Nchw_3x7x7x256_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_global_average_pool_test/Nchw_3x8x8x255_S8_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_global_average_pool_test/Nchw_3x8x8x255_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_global_average_pool_test/Nchw_3x8x8x256_S8_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_global_average_pool_test/Nchw_3x8x8x256_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_global_average_pool_test/Nhwc_1x1x32x32_S8_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_global_average_pool_test/Nhwc_1x1x32x32_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_global_average_pool_test/Nhwc_1x255x7x7_S8_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_global_average_pool_test/Nhwc_1x255x7x7_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_global_average_pool_test/Nhwc_1x255x8x8_S8_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_global_average_pool_test/Nhwc_1x255x8x8_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_global_average_pool_test/Nhwc_1x256x7x7_S8_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_global_average_pool_test/Nhwc_1x256x7x7_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_global_average_pool_test/Nhwc_1x256x8x8_S8_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_global_average_pool_test/Nhwc_1x256x8x8_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_global_average_pool_test/Nhwc_3x255x7x7_S8_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_global_average_pool_test/Nhwc_3x255x7x7_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_global_average_pool_test/Nhwc_3x255x8x8_S8_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_global_average_pool_test/Nhwc_3x255x8x8_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_global_average_pool_test/Nhwc_3x256x7x7_S8_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_global_average_pool_test/Nhwc_3x256x7x7_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_global_average_pool_test/Nhwc_3x256x8x8_S8_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_global_average_pool_test/Nhwc_3x256x8x8_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_lookup_table_test/QLinearLeakyRelu_Int8_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_lookup_table_test/QLinearLeakyRelu_UInt8_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_lookup_table_test/QLinearSigmoid_Int8_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_lookup_table_test/QLinearSigmoid_UInt8_0_Y_ZP_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_lookup_table_test/QLinearSigmoid_UInt8_0_Y_ZP_run1/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_lookup_table_test/QLinearSigmoid_UInt8_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_lookup_table_test/QLinearSoftmax_Int8_v12_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_lookup_table_test/QLinearSoftmax_Int8_v12_run1/model.onnx |  | Data/Data | ❌ | QLinearSoftmax axis -2 is out of bounds for shape () |
| test/contrib_ops/qlinear_lookup_table_test/QLinearSoftmax_Int8_v13_run0/model.onnx |  | Data/Data | ❌ | Out of tolerance (max abs diff 1) |
| test/contrib_ops/qlinear_lookup_table_test/QLinearSoftmax_Int8_v13_run1/model.onnx |  | Data/Data | ❌ | QLinearSoftmax axis -2 is out of bounds for shape () |
| test/contrib_ops/qlinear_lookup_table_test/QLinearSoftmax_UInt8_v12_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_lookup_table_test/QLinearSoftmax_UInt8_v12_run1/model.onnx |  | Data/Data | ❌ | QLinearSoftmax axis -2 is out of bounds for shape () |
| test/contrib_ops/qlinear_lookup_table_test/QLinearSoftmax_UInt8_v13_run0/model.onnx |  | Data/Data | ❌ | Out of tolerance (max abs diff 1) |
| test/contrib_ops/qlinear_lookup_table_test/QLinearSoftmax_UInt8_v13_run1/model.onnx |  | Data/Data | ❌ | QLinearSoftmax axis -2 is out of bounds for shape () |
| test/contrib_ops/qlinear_pool_test/AveragePool1D_ExcludePadPixel_S8_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_pool_test/AveragePool1D_ExcludePadPixel_S8_run1/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_pool_test/AveragePool1D_ExcludePadPixel_S8_run2/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_pool_test/AveragePool1D_ExcludePadPixel_nhwc_S8_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_pool_test/AveragePool1D_ExcludePadPixel_nhwc_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_pool_test/AveragePool1D_ExcludePadPixel_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_pool_test/AveragePool1D_ExcludePadPixel_run1/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_pool_test/AveragePool1D_ExcludePadPixel_run2/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_pool_test/AveragePool1D_IncludePadPixel_S8_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_pool_test/AveragePool1D_IncludePadPixel_S8_run1/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_pool_test/AveragePool1D_IncludePadPixel_S8_run2/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_pool_test/AveragePool1D_IncludePadPixel_nhwc_S8_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_pool_test/AveragePool1D_IncludePadPixel_nhwc_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_pool_test/AveragePool1D_IncludePadPixel_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_pool_test/AveragePool1D_IncludePadPixel_run1/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_pool_test/AveragePool1D_IncludePadPixel_run2/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_pool_test/AveragePool2D_BigImage_S8_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_pool_test/AveragePool2D_BigImage_S8_run1/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_pool_test/AveragePool2D_BigImage_S8_run2/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_pool_test/AveragePool2D_BigImage_nhwc_S8_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_pool_test/AveragePool2D_BigImage_nhwc_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_pool_test/AveragePool2D_BigImage_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_pool_test/AveragePool2D_BigImage_run1/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_pool_test/AveragePool2D_BigImage_run2/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_pool_test/AveragePool2D_ExcludePadPixel_S8_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_pool_test/AveragePool2D_ExcludePadPixel_S8_run1/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_pool_test/AveragePool2D_ExcludePadPixel_S8_run2/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_pool_test/AveragePool2D_ExcludePadPixel_nhwc_S8_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_pool_test/AveragePool2D_ExcludePadPixel_nhwc_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_pool_test/AveragePool2D_ExcludePadPixel_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_pool_test/AveragePool2D_ExcludePadPixel_run1/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_pool_test/AveragePool2D_ExcludePadPixel_run2/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_pool_test/AveragePool2D_Global_S8_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_pool_test/AveragePool2D_Global_S8_run1/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_pool_test/AveragePool2D_Global_S8_run2/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_pool_test/AveragePool2D_Global_nhwc_S8_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_pool_test/AveragePool2D_Global_nhwc_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_pool_test/AveragePool2D_Global_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_pool_test/AveragePool2D_Global_run1/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_pool_test/AveragePool2D_Global_run2/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_pool_test/AveragePool2D_IncludePadPixel_S8_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_pool_test/AveragePool2D_IncludePadPixel_S8_run1/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_pool_test/AveragePool2D_IncludePadPixel_S8_run2/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_pool_test/AveragePool2D_IncludePadPixel_nhwc_S8_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_pool_test/AveragePool2D_IncludePadPixel_nhwc_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_pool_test/AveragePool2D_IncludePadPixel_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_pool_test/AveragePool2D_IncludePadPixel_run1/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_pool_test/AveragePool2D_IncludePadPixel_run2/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_pool_test/AveragePool2D_MultiChannel_S8_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_pool_test/AveragePool2D_MultiChannel_S8_run1/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_pool_test/AveragePool2D_MultiChannel_S8_run2/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_pool_test/AveragePool2D_MultiChannel_nhwc_S8_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_pool_test/AveragePool2D_MultiChannel_nhwc_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_pool_test/AveragePool2D_MultiChannel_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_pool_test/AveragePool2D_MultiChannel_run1/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_pool_test/AveragePool2D_MultiChannel_run2/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_pool_test/AveragePool3D_ExcludePadPixel_S8_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_pool_test/AveragePool3D_ExcludePadPixel_S8_run1/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_pool_test/AveragePool3D_ExcludePadPixel_S8_run2/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_pool_test/AveragePool3D_ExcludePadPixel_nhwc_S8_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_pool_test/AveragePool3D_ExcludePadPixel_nhwc_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_pool_test/AveragePool3D_ExcludePadPixel_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_pool_test/AveragePool3D_ExcludePadPixel_run1/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_pool_test/AveragePool3D_ExcludePadPixel_run2/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_pool_test/AveragePool3D_IncludePadPixel_S8_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_pool_test/AveragePool3D_IncludePadPixel_S8_run1/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_pool_test/AveragePool3D_IncludePadPixel_S8_run2/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_pool_test/AveragePool3D_IncludePadPixel_nhwc_S8_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_pool_test/AveragePool3D_IncludePadPixel_nhwc_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_pool_test/AveragePool3D_IncludePadPixel_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_pool_test/AveragePool3D_IncludePadPixel_run1/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_pool_test/AveragePool3D_IncludePadPixel_run2/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/qlinear_where_test/QLinearWhereMatrixAll_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearWhere |
| test/contrib_ops/qlinear_where_test/QLinearWhereMatrixAll_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearWhere |
| test/contrib_ops/qlinear_where_test/QLinearWhereScalarAll_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearWhere |
| test/contrib_ops/qlinear_where_test/QLinearWhereScalarAll_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearWhere |
| test/contrib_ops/qlinear_where_test/QLinearWhereScalarX_VectorY_MatrixCondition_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearWhere |
| test/contrib_ops/qlinear_where_test/QLinearWhereScalarX_VectorY_MatrixCondition_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearWhere |
| test/contrib_ops/qlinear_where_test/QLinearWhereVectorAll_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearWhere |
| test/contrib_ops/qlinear_where_test/QLinearWhereVectorAll_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearWhere |
| test/contrib_ops/qlinear_where_test/QLinearWhereVectorX_VectorY_MatrixCondition_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearWhere |
| test/contrib_ops/qlinear_where_test/QLinearWhereVectorX_VectorY_MatrixCondition_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearWhere |
| test/contrib_ops/quant_gemm_test/GEMM_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run1/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run10/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run100/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run101/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run102/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run103/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run104/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run105/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run106/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run107/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run108/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run109/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run11/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run110/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run111/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run112/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run113/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run114/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run115/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run116/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run117/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run118/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run119/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run12/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run120/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run121/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run122/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run123/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run124/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run125/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run126/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run127/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run128/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/quant_gemm_test/GEMM_run129/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run13/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run130/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run131/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run132/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run133/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run134/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run135/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run136/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run137/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run138/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run139/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run14/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run140/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run141/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run142/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run143/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run144/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run145/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run146/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run147/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run148/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run149/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run15/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run150/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run151/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run152/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/quant_gemm_test/GEMM_run153/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run154/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run155/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run156/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run157/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run158/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run159/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run16/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run160/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run161/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run162/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run163/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run164/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run165/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run166/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run167/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run168/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run169/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run17/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run170/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run171/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run172/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run173/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run174/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run175/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run176/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/quant_gemm_test/GEMM_run177/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run178/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run179/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run18/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run180/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run181/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run182/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/quant_gemm_test/GEMM_run183/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run184/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run185/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run186/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run187/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run188/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run189/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run19/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run190/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run191/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run2/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run20/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run21/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run22/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run23/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run24/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run25/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run26/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run27/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run28/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run29/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run3/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run30/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run31/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run32/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run33/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run34/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run35/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run36/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run37/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run38/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run39/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run4/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run40/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run41/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run42/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run43/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run44/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run45/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run46/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run47/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run48/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run49/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run5/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run50/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run51/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run52/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run53/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run54/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run55/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run56/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/quant_gemm_test/GEMM_run57/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run58/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run59/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run6/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run60/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run61/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run62/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run63/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run64/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run65/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run66/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run67/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run68/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run69/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run7/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run70/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run71/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run72/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run73/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run74/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run75/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run76/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run77/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run78/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run79/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run8/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/quant_gemm_test/GEMM_run80/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run81/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run82/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run83/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run84/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run85/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run86/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run87/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run88/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run89/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run9/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run90/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run91/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run92/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/quant_gemm_test/GEMM_run93/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run94/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run95/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run96/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run97/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run98/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMM_run99/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run1/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run10/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run100/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run101/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run102/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run103/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run104/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run105/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run106/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run107/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run108/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run109/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run11/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run110/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run111/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run112/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run113/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run114/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run115/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run116/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run117/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run118/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run119/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run12/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run120/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run121/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run122/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run123/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run124/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run125/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run126/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run127/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run128/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run129/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run13/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run130/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run131/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run132/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run133/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run134/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run135/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run136/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run137/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run138/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run139/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run14/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run140/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run141/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run142/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run143/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run144/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run145/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run146/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run147/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run148/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run149/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run15/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run150/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run151/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run152/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run153/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run154/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run155/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run156/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run157/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run158/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run159/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run16/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run160/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run161/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run162/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run163/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run164/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/quant_gemm_test/GEMV_run165/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run166/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run167/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run168/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run169/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run17/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run170/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run171/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run172/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run173/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run174/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run175/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run176/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run177/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run178/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run179/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run18/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run180/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run181/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run182/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run183/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run184/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run185/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run186/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run187/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run188/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run189/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run19/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run190/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run191/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run192/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run193/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run194/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/quant_gemm_test/GEMV_run195/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run196/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run197/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run198/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run199/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run2/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run20/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run200/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/quant_gemm_test/GEMV_run201/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run202/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run203/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run204/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run205/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run206/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/quant_gemm_test/GEMV_run207/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run208/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run209/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run21/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run210/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run211/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run212/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/quant_gemm_test/GEMV_run213/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run214/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run215/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run216/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run217/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run218/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run219/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run22/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run220/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run221/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run222/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run223/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run224/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/quant_gemm_test/GEMV_run225/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run226/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run227/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run228/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run229/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run23/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run230/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/quant_gemm_test/GEMV_run231/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run232/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run233/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run234/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run235/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run236/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/quant_gemm_test/GEMV_run237/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run238/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run239/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run24/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run240/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/quant_gemm_test/GEMV_run241/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run242/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/quant_gemm_test/GEMV_run243/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run244/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/quant_gemm_test/GEMV_run245/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run246/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/quant_gemm_test/GEMV_run247/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run248/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/quant_gemm_test/GEMV_run249/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run25/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run250/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/quant_gemm_test/GEMV_run251/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run252/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/quant_gemm_test/GEMV_run253/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run254/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/quant_gemm_test/GEMV_run255/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run256/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/quant_gemm_test/GEMV_run257/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run258/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/quant_gemm_test/GEMV_run259/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run26/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run260/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/quant_gemm_test/GEMV_run261/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run262/model.onnx |  | Data/Data | ✅ | OK (max ULP 2) |
| test/contrib_ops/quant_gemm_test/GEMV_run263/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run264/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/quant_gemm_test/GEMV_run265/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run266/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/quant_gemm_test/GEMV_run267/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run268/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/quant_gemm_test/GEMV_run269/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run27/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run270/model.onnx |  | Data/Data | ✅ | OK (max ULP 2) |
| test/contrib_ops/quant_gemm_test/GEMV_run271/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run272/model.onnx |  | Data/Data | ✅ | OK (max ULP 2) |
| test/contrib_ops/quant_gemm_test/GEMV_run273/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run274/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/quant_gemm_test/GEMV_run275/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run276/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/quant_gemm_test/GEMV_run277/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run278/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/quant_gemm_test/GEMV_run279/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run28/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run280/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/quant_gemm_test/GEMV_run281/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run282/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/quant_gemm_test/GEMV_run283/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run284/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/quant_gemm_test/GEMV_run285/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run286/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/quant_gemm_test/GEMV_run287/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run29/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run3/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run30/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run31/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run32/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run33/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run34/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run35/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run36/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run37/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run38/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run39/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run4/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run40/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run41/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run42/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run43/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run44/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run45/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run46/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run47/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run48/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run49/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run5/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run50/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run51/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run52/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run53/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run54/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run55/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run56/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run57/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run58/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run59/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run6/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run60/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run61/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run62/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run63/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run64/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run65/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run66/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run67/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run68/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run69/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run7/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run70/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run71/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run72/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run73/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run74/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run75/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run76/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run77/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run78/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run79/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run8/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run80/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run81/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run82/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run83/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run84/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run85/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run86/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run87/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run88/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run89/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run9/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run90/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run91/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run92/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run93/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run94/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run95/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run96/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run97/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run98/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/GEMV_run99/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run1/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run10/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run100/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run101/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run102/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run103/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run104/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run105/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run106/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run107/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run108/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run109/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run11/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run110/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run111/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run112/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run113/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run114/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run115/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run116/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run117/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run118/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run119/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run12/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run120/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run121/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run122/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/quant_gemm_test/Scalar_run123/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run124/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run125/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run126/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run127/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run128/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run129/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run13/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run130/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run131/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run132/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run133/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run134/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run135/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run136/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run137/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run138/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run139/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run14/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run140/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run141/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run142/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run143/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run15/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run16/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run17/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run18/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run19/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run2/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run20/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run21/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run22/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run23/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run24/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run25/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run26/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run27/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run28/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run29/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run3/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run30/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run31/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run32/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run33/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run34/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run35/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run36/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run37/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run38/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run39/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run4/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run40/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run41/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run42/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run43/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run44/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run45/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run46/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run47/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run48/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run49/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run5/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run50/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run51/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run52/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run53/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run54/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run55/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run56/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run57/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run58/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run59/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run6/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run60/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run61/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run62/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run63/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run64/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run65/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run66/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run67/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run68/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run69/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run7/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run70/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run71/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run72/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run73/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run74/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run75/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run76/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run77/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run78/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run79/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run8/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run80/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/quant_gemm_test/Scalar_run81/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run82/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run83/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run84/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run85/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run86/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run87/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run88/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run89/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run9/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run90/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run91/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run92/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run93/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run94/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run95/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run96/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run97/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run98/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quant_gemm_test/Scalar_run99/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quantize_attention_op_test/QAttentionBatch1_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 3) |
| test/contrib_ops/quantize_attention_op_test/QAttentionBatch1_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 3) |
| test/contrib_ops/quantize_attention_op_test/QAttentionBatch2_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 3) |
| test/contrib_ops/quantize_attention_op_test/QAttentionBatch2_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 3) |
| test/contrib_ops/quantize_attention_op_test/QAttentionMaskExceedSequence_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 3) |
| test/contrib_ops/quantize_attention_op_test/QAttentionMaskExceedSequence_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 3) |
| test/contrib_ops/quantize_attention_op_test/QAttentionMaskPartialSequence_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 3) |
| test/contrib_ops/quantize_attention_op_test/QAttentionMaskPartialSequence_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 3) |
| test/contrib_ops/quantize_attention_op_test/QAttentionNoMaskIndex_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 3) |
| test/contrib_ops/quantize_attention_op_test/QAttentionNoMaskIndex_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 3) |
| test/contrib_ops/quantize_attention_op_test/QAttentionPastState_u8s8_run0/model.onnx (--max-ulp 3000) |  | Data/Data | ✅ | OK (max ULP 633) |
| test/contrib_ops/quantize_attention_op_test/QAttentionPastState_u8s8_run1/model.onnx (--max-ulp 3000) |  | Data/Data | ✅ | OK (max ULP 633) |
| test/contrib_ops/quantize_attention_op_test/QAttentionPastState_u8s8_run2/model.onnx (--max-ulp 3000) |  | Data/Data | ✅ | OK (max ULP 2569) |
| test/contrib_ops/quantize_attention_op_test/QAttentionPastState_u8s8_run3/model.onnx (--max-ulp 3000) |  | Data/Data | ✅ | OK (max ULP 2569) |
| test/contrib_ops/quantize_attention_op_test/QAttentionPastState_u8u8_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 67) |
| test/contrib_ops/quantize_attention_op_test/QAttentionPastState_u8u8_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 67) |
| test/contrib_ops/quantize_attention_op_test/QAttentionPastState_u8u8_run2/model.onnx (--max-ulp 3000) |  | Data/Data | ✅ | OK (max ULP 1220) |
| test/contrib_ops/quantize_attention_op_test/QAttentionPastState_u8u8_run3/model.onnx (--max-ulp 3000) |  | Data/Data | ✅ | OK (max ULP 1220) |
| test/contrib_ops/quantize_attention_op_test/QAttentionPrunedModel_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 4) |
| test/contrib_ops/quantize_attention_op_test/QAttentionPrunedModel_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 4) |
| test/contrib_ops/quantize_attention_op_test/QAttentionUnidirectional_U8S8_run0/model.onnx (--max-ulp 3000) |  | Data/Data | ✅ | OK (max ULP 1522) |
| test/contrib_ops/quantize_attention_op_test/QAttentionUnidirectional_U8U8_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quantize_attention_op_test/SharedPrepackedWeights_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 3) |
| test/contrib_ops/quantize_attention_op_test/SharedPrepackedWeights_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 3) |
| test/contrib_ops/quantize_lstm_op_test/LargeSize_run0/model.onnx (--atol-eps 6) |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quantize_lstm_op_test/LargeSize_run1/model.onnx (--atol-eps 3) |  | Data/Data | ✅ | OK (max ULP 16) |
| test/contrib_ops/quantize_lstm_op_test/LargeSize_run10/model.onnx (--atol-eps 6) |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quantize_lstm_op_test/LargeSize_run11/model.onnx (--atol-eps 3) |  | Data/Data | ✅ | OK (max ULP 49) |
| test/contrib_ops/quantize_lstm_op_test/LargeSize_run12/model.onnx (--atol-eps 4) |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quantize_lstm_op_test/LargeSize_run13/model.onnx (--atol-eps 2) |  | Data/Data | ✅ | OK (max ULP 90) |
| test/contrib_ops/quantize_lstm_op_test/LargeSize_run14/model.onnx (--atol-eps 4) |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quantize_lstm_op_test/LargeSize_run15/model.onnx (--atol-eps 2) |  | Data/Data | ✅ | OK (max ULP 90) |
| test/contrib_ops/quantize_lstm_op_test/LargeSize_run16/model.onnx (--atol-eps 3) |  | Data/Data | ✅ | OK (max ULP 65) |
| test/contrib_ops/quantize_lstm_op_test/LargeSize_run17/model.onnx (--atol-eps 4) |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quantize_lstm_op_test/LargeSize_run18/model.onnx (--atol-eps 3) |  | Data/Data | ✅ | OK (max ULP 65) |
| test/contrib_ops/quantize_lstm_op_test/LargeSize_run19/model.onnx (--atol-eps 4) |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quantize_lstm_op_test/LargeSize_run2/model.onnx (--atol-eps 6) |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quantize_lstm_op_test/LargeSize_run20/model.onnx (--atol-eps 2) |  | Data/Data | ✅ | OK (max ULP 38) |
| test/contrib_ops/quantize_lstm_op_test/LargeSize_run21/model.onnx (--atol-eps 3) |  | Data/Data | ✅ | OK (max ULP 50) |
| test/contrib_ops/quantize_lstm_op_test/LargeSize_run22/model.onnx (--atol-eps 2) |  | Data/Data | ✅ | OK (max ULP 38) |
| test/contrib_ops/quantize_lstm_op_test/LargeSize_run23/model.onnx (--atol-eps 3) |  | Data/Data | ✅ | OK (max ULP 50) |
| test/contrib_ops/quantize_lstm_op_test/LargeSize_run3/model.onnx (--atol-eps 3) |  | Data/Data | ✅ | OK (max ULP 16) |
| test/contrib_ops/quantize_lstm_op_test/LargeSize_run4/model.onnx (--atol-eps 2) |  | Data/Data | ✅ | OK (max ULP 14) |
| test/contrib_ops/quantize_lstm_op_test/LargeSize_run5/model.onnx (--atol-eps 3) |  | Data/Data | ✅ | OK (max ULP 49) |
| test/contrib_ops/quantize_lstm_op_test/LargeSize_run6/model.onnx (--atol-eps 2) |  | Data/Data | ✅ | OK (max ULP 14) |
| test/contrib_ops/quantize_lstm_op_test/LargeSize_run7/model.onnx (--atol-eps 3) |  | Data/Data | ✅ | OK (max ULP 49) |
| test/contrib_ops/quantize_lstm_op_test/LargeSize_run8/model.onnx (--atol-eps 6) |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quantize_lstm_op_test/LargeSize_run9/model.onnx (--atol-eps 3) |  | Data/Data | ✅ | OK (max ULP 49) |
| test/contrib_ops/quantize_lstm_op_test/SharedPrepackedWeights_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quantize_lstm_op_test/SharedPrepackedWeights_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quantize_lstm_op_test/SmallSize_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quantize_lstm_op_test/SmallSize_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quantize_lstm_op_test/SmallSize_run10/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quantize_lstm_op_test/SmallSize_run11/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quantize_lstm_op_test/SmallSize_run12/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quantize_lstm_op_test/SmallSize_run13/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quantize_lstm_op_test/SmallSize_run14/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quantize_lstm_op_test/SmallSize_run15/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quantize_lstm_op_test/SmallSize_run16/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quantize_lstm_op_test/SmallSize_run17/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quantize_lstm_op_test/SmallSize_run18/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quantize_lstm_op_test/SmallSize_run19/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quantize_lstm_op_test/SmallSize_run2/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quantize_lstm_op_test/SmallSize_run20/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quantize_lstm_op_test/SmallSize_run21/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quantize_lstm_op_test/SmallSize_run22/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quantize_lstm_op_test/SmallSize_run23/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quantize_lstm_op_test/SmallSize_run3/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quantize_lstm_op_test/SmallSize_run4/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quantize_lstm_op_test/SmallSize_run5/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quantize_lstm_op_test/SmallSize_run6/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quantize_lstm_op_test/SmallSize_run7/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quantize_lstm_op_test/SmallSize_run8/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quantize_lstm_op_test/SmallSize_run9/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quantize_ops_test/DequantizeLinearOpTest_BroadcastTensorOfOne_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quantize_ops_test/DequantizeLinear_0_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quantize_ops_test/DequantizeLinear_1_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quantize_ops_test/DequantizeLinear_2_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quantize_ops_test/DequantizeLinear_3_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quantize_ops_test/DequantizeLinear_per_tensor_float_int16_cpu_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quantize_ops_test/DequantizeLinear_per_tensor_float_int32_cpu_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quantize_ops_test/DequantizeLinear_per_tensor_float_int8_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quantize_ops_test/DequantizeLinear_per_tensor_float_uint16_cpu_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quantize_ops_test/DequantizeLinear_per_tensor_float_uint8_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quantize_ops_test/DequantizeLinear_per_tensor_float_uint8_use_initializer_except_x_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/quantize_ops_test/QuantizeLinear_per_channel_negative_axis_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quantize_ops_test/QuantizeLinear_per_channel_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quantize_ops_test/QuantizeLinear_per_tensor_float_int16_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quantize_ops_test/QuantizeLinear_per_tensor_float_int8_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quantize_ops_test/QuantizeLinear_per_tensor_float_uint16_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/quantize_ops_test/QuantizeLinear_per_tensor_float_uint8_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/rotary_embedding_op_test/RotaryEmbedding_CustomRotaryDim_SmallData_Phi_Packed_Batching_run0/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 6754) |
| test/contrib_ops/rotary_embedding_op_test/RotaryEmbedding_CustomRotaryDim_SmallData_Phi_Packed_Batching_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/rotary_embedding_op_test/RotaryEmbedding_CustomRotaryDim_SmallData_Phi_run0/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 6754) |
| test/contrib_ops/rotary_embedding_op_test/RotaryEmbedding_CustomRotaryDim_SmallData_Phi_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/rotary_embedding_op_test/RotaryEmbedding_Interleaved_LargeData_LlamaMSFT_run0/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 177823) |
| test/contrib_ops/rotary_embedding_op_test/RotaryEmbedding_Interleaved_LargeData_LlamaMSFT_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/rotary_embedding_op_test/RotaryEmbedding_Interleaved_SmallData_LlamaMSFT_run0/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 6391) |
| test/contrib_ops/rotary_embedding_op_test/RotaryEmbedding_Interleaved_SmallData_LlamaMSFT_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/rotary_embedding_op_test/RotaryEmbedding_NotInterleaved_LargeData_LlamaMSFT_run0/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 52786) |
| test/contrib_ops/rotary_embedding_op_test/RotaryEmbedding_NotInterleaved_LargeData_LlamaMSFT_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/rotary_embedding_op_test/RotaryEmbedding_NotInterleaved_SmallData_LlamaMSFT_run0/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 26739) |
| test/contrib_ops/rotary_embedding_op_test/RotaryEmbedding_NotInterleaved_SmallData_LlamaMSFT_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/sample_op_test/SampleOpFloat_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.SampleOp |
| test/contrib_ops/skiplayernorm_op_test/SkipLayerNormBatch1_NoBeta_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/skiplayernorm_op_test/SkipLayerNormBatch1_NoBeta_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 1) |
| test/contrib_ops/skiplayernorm_op_test/SkipLayerNormBatch1_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/skiplayernorm_op_test/SkipLayerNormBatch1_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/skiplayernorm_op_test/SkipLayerNormBatch2_Bias_ProducingOptionalOutput_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 24) |
| test/contrib_ops/skiplayernorm_op_test/SkipLayerNormBatch2_Bias_ProducingOptionalOutput_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 24) |
| test/contrib_ops/skiplayernorm_op_test/SkipLayerNormBatch2_Bias_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 24) |
| test/contrib_ops/skiplayernorm_op_test/SkipLayerNormBatch2_Bias_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 24) |
| test/contrib_ops/skiplayernorm_op_test/SkipLayerNormBatch2_Skip_Broadcast_Batch_Size_1_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/skiplayernorm_op_test/SkipLayerNormBatch2_Skip_Broadcast_No_Batch_Size_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/skiplayernorm_op_test/SkipLayerNormBatch2_TokenCount_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/skiplayernorm_op_test/SkipLayerNormBatch2_TokenCount_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/skiplayernorm_op_test/SkipLayerNormBatch2_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/skiplayernorm_op_test/SkipLayerNormBatch2_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/skiplayernorm_op_test/SkipLayerNormNullInput_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/skiplayernorm_op_test/SkipLayerNormNullInput_run1/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/skiplayernorm_op_test/SkipLayerNormPrePack_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/tensor_op_test/CropBorderAndScale_run0/model.onnx | 7 | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/tensor_op_test/CropBorderOnly_run0/model.onnx | 7 | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/tensor_op_test/ImageScalerTest_run0/model.onnx | 7 | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/tensor_op_test/LastDim_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.UnfoldTensor |
| test/contrib_ops/tensor_op_test/MeanVarianceNormalizationCPUTest_Version1_TO_8_run0/model.onnx | 7 | Data/Data | ❌ | Out of tolerance (max ULP 7137119) |
| test/contrib_ops/tensor_op_test/MeanVarianceNormalizationCPUTest_Version1_TO_8_run1/model.onnx | 7 | Data/Data | ❌ | Out of tolerance (max ULP 21399263) |
| test/contrib_ops/tensor_op_test/MeanVarianceNormalizationCPUTest_Version1_TO_8_run2/model.onnx | 7 | Data/Data | ❌ | Out of tolerance (max ULP 15806857) |
| test/contrib_ops/tensor_op_test/MeanVarianceNormalizationCPUTest_Version1_TO_8_run3/model.onnx | 7 | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/tensor_op_test/NormalDim_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.UnfoldTensor |
| test/contrib_ops/tokenizer_test/TokenizerCharLevel_CyrillicCharsWithMarkersC_run0/model.onnx |  | Data/Data | ✅ | OK (no numeric comparisons) |
| test/contrib_ops/tokenizer_test/TokenizerCharLevel_EmptyOutputC_run0/model.onnx |  | Data/Data | ✅ | OK (no numeric comparisons) |
| test/contrib_ops/tokenizer_test/TokenizerCharLevel_EmptyOutputNC_run0/model.onnx |  | Data/Data | ✅ | OK (no numeric comparisons) |
| test/contrib_ops/tokenizer_test/TokenizerCharLevel_LatinCharsNoMarkersC_run0/model.onnx |  | Data/Data | ✅ | OK (no numeric comparisons) |
| test/contrib_ops/tokenizer_test/TokenizerCharLevel_LatinCharsNoMarkersNC_run0/model.onnx |  | Data/Data | ✅ | OK (no numeric comparisons) |
| test/contrib_ops/tokenizer_test/TokenizerCharLevel_LatinCharsWithMarkersC_run0/model.onnx |  | Data/Data | ✅ | OK (no numeric comparisons) |
| test/contrib_ops/tokenizer_test/TokenizerCharLevel_LatinCharsWithMarkersNC_run0/model.onnx |  | Data/Data | ✅ | OK (no numeric comparisons) |
| test/contrib_ops/tokenizer_test/TokenizerCharLevel_MixedCharsWithMarkersC_run0/model.onnx |  | Data/Data | ✅ | OK (no numeric comparisons) |
| test/contrib_ops/tokenizer_test/TokenizerExpression_Grouping_run0/model.onnx |  | Data/Data | ✅ | OK (no numeric comparisons) |
| test/contrib_ops/tokenizer_test/TokenizerExpression_RegChar_run0/model.onnx |  | Data/Data | ✅ | OK (no numeric comparisons) |
| test/contrib_ops/tokenizer_test/TokenizerExpression_RegDot_run0/model.onnx |  | Data/Data | ✅ | OK (no numeric comparisons) |
| test/contrib_ops/tokenizer_test/TokenizerExpression_RegEx_run0/model.onnx |  | Data/Data | ✅ | OK (no numeric comparisons) |
| test/contrib_ops/tokenizer_test/TokenizerExpression_RegRep_run0/model.onnx |  | Data/Data | ✅ | OK (no numeric comparisons) |
| test/contrib_ops/tokenizer_test/TokenizerWithSeparators_MixCharCommonPrefixC_run0/model.onnx |  | Data/Data | ✅ | OK (no numeric comparisons) |
| test/contrib_ops/tokenizer_test/TokenizerWithSeparators_MixCharsNoMarkersSeparatorsOverlapLongFirstC_run0/model.onnx |  | Data/Data | ✅ | OK (no numeric comparisons) |
| test/contrib_ops/tokenizer_test/TokenizerWithSeparators_MixCharsNoMarkersSeparatorsOverlapLongFirstRepeatedShortC_run0/model.onnx |  | Data/Data | ✅ | OK (no numeric comparisons) |
| test/contrib_ops/tokenizer_test/TokenizerWithSeparators_MixCharsNoMarkersSeparatorsOverlapShortFirstC_run0/model.onnx |  | Data/Data | ✅ | OK (no numeric comparisons) |
| test/contrib_ops/tokenizer_test/TokenizerWithSeparators_MixCharsNoMarkersSeparatorsOverlapingMatchC_run0/model.onnx |  | Data/Data | ✅ | OK (no numeric comparisons) |
| test/contrib_ops/tokenizer_test/TokenizerWithSeparators_MixCharsWithMarkersC_run0/model.onnx |  | Data/Data | ✅ | OK (no numeric comparisons) |
| test/contrib_ops/tokenizer_test/TokenizerWithSeparators_MixCharsWithMarkersCompleteMatchEmptyOutputC_run0/model.onnx |  | Data/Data | ✅ | OK (no numeric comparisons) |
| test/contrib_ops/tokenizer_test/TokenizerWithSeparators_MixCharsWithMarkersEmptyInputEmptyOutputC_run0/model.onnx |  | Data/Data | ✅ | OK (no numeric comparisons) |
| test/contrib_ops/tokenizer_test/TokenizerWithSeparators_MixCharsWithMarkersEmptyInputEmptyOutputNC_run0/model.onnx |  | Data/Data | ✅ | OK (no numeric comparisons) |
| test/contrib_ops/tokenizer_test/TokenizerWithSeparators_MixCharsWithMarkersEndMatchAtLeast4CharsC_run0/model.onnx |  | Data/Data | ✅ | OK (no numeric comparisons) |
| test/contrib_ops/tokenizer_test/TokenizerWithSeparators_MixCharsWithMarkersEndMatchC_run0/model.onnx |  | Data/Data | ✅ | OK (no numeric comparisons) |
| test/contrib_ops/tokenizer_test/TokenizerWithSeparators_MixCharsWithMarkersStartMatchC_run0/model.onnx |  | Data/Data | ✅ | OK (no numeric comparisons) |
| test/contrib_ops/tokenizer_test/Tokenizer_EmptyInput_run0/model.onnx |  | Data/Data | ✅ | OK (no numeric comparisons) |
| test/contrib_ops/tokenizer_test/Tokenizer_EmptyInput_run1/model.onnx |  | Data/Data | ✅ | OK (no numeric comparisons) |
| test/contrib_ops/tokenizer_test/Tokenizer_EmptyInput_run2/model.onnx |  | Data/Data | ✅ | OK (no numeric comparisons) |
| test/contrib_ops/trilu_test/neg_k_float_lower_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/trilu_test/neg_k_float_upper_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/trilu_test/small_k_float_lower_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/trilu_test/small_k_float_upper_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/trilu_test/three_dim_float_lower_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/trilu_test/three_dim_float_upper_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/trilu_test/two_by_two_double_lower_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/trilu_test/two_by_two_double_upper_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/trilu_test/two_by_two_float_lower_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/trilu_test/two_by_two_float_upper_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/trilu_test/two_by_two_long_lower_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/trilu_test/two_by_two_long_upper_run0/model.onnx |  | Data/Data | ✅ | OK (max abs diff 0) |
| test/contrib_ops/trilu_test/zero_dim_2_lower_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/trilu_test/zero_dim_2_upper_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/trilu_test/zero_dim_lower_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/trilu_test/zero_dim_upper_run0/model.onnx |  | Data/Data | ✅ | OK (max ULP 0) |
| test/contrib_ops/unique_op_test/Unique_AllUniqueElements_run0/model.onnx |  | Data/Data | ❌ | Unique must have 1 input and 4 outputs |
| test/contrib_ops/unique_op_test/Unique_Complicated_Example_run0/model.onnx |  | Data/Data | ❌ | Unique must have 1 input and 4 outputs |
| test/contrib_ops/unique_op_test/Unique_Example_SingleElement_run0/model.onnx |  | Data/Data | ❌ | Unique must have 1 input and 4 outputs |
| test/contrib_ops/unique_op_test/Unique_Spec_Example_run0/model.onnx |  | Data/Data | ❌ | Unique must have 1 input and 4 outputs |
| test/contrib_ops/word_conv_embedding_test/WordConvEmbedding_char_embedding_shape_conv_shape_not_match_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.WordConvEmbedding |
| test/contrib_ops/word_conv_embedding_test/WordConvEmbedding_char_embedding_size_mismatch_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.WordConvEmbedding |
| test/contrib_ops/word_conv_embedding_test/WordConvEmbedding_conv_window_size_mismatch_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.WordConvEmbedding |
| test/contrib_ops/word_conv_embedding_test/WordConvEmbedding_embedding_size_mismatch_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.WordConvEmbedding |
| test/contrib_ops/word_conv_embedding_test/WordConvEmbedding_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.WordConvEmbedding |
| test/contrib_ops/word_conv_embedding_test/WordConvEmbedding_valid_attribute_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.WordConvEmbedding |

## Local ONNX test coverage

Test directory: `tests/onnx`

Coverage 7 / 7 ONNX files (100.0%).

| File | Opset | Verification | Supported | Error |
| --- | --- | --- | --- | --- |
| micro_kws_m_qdq.onnx | 15 | Random/ORT | ✅ | OK (max ULP 0) |
| micro_kws_m_qoperator_add_shape.onnx (--replicate-ort-bugs) | 15 | Random/ORT | ✅ | OK (max ULP 0) |
| micro_kws_m_qoperator_avg_pool.onnx (--replicate-ort-bugs) | 15 | Random/ORT | ✅ | OK (max ULP 0) |
| micro_kws_m_qoperator_softmax.onnx (--replicate-ort-bugs) | 15 | Random/ORT | ✅ | OK (max ULP 0) |
| micro_kws_m_static_fp32.onnx | 15 | Random/ORT | ✅ | OK (max ULP 6) |
| micro_kws_m_static_qdq.onnx | 15 | Random/ORT | ✅ | OK (max ULP 0) |
| micro_kws_m_static_qoperator.onnx (--replicate-ort-bugs) | 15 | Random/ORT | ✅ | OK (max ULP 0) |
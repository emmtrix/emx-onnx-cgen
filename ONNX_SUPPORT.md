# ONNX test coverage

Overview:

| Test suite | Coverage | Version |
| --- | --- | --- |
| [Official ONNX test coverage](#official-onnx-test-coverage) | 1482 / 1802, 82.2% | 1.20.1 |
| [ONNX2C test coverage](#onnx2c-test-coverage) | 118 / 125, 94.4% | n/a |
| [Local ONNX test coverage](#local-onnx-test-coverage) | 1 / 3, 33.3% | n/a |

See [`ONNX_ERRORS_HISTOGRAM.md`](ONNX_ERRORS_HISTOGRAM.md) for the error histogram.

Floating-point verification first ignores very small differences up to **1.0 × [machine epsilon](https://en.wikipedia.org/wiki/Machine_epsilon) of the evaluated floating-point type**, treating such values as equal. For values with a larger absolute difference, the ULP distance is computed, and the maximum ULP distance is reported.

## Official ONNX test coverage

Test directory: `onnx-org/onnx/backend/test/data`

Coverage 1482 / 1802 ONNX files (82.2%).

| File | Opset | Supported | Error |
| --- | --- | --- | --- |
| light/light_bvlc_alexnet.onnx | 9 | ✅ | OK (max ULP 0) |
| light/light_densenet121.onnx | 9 | ✅ | OK (max ULP 73) |
| light/light_inception_v1.onnx | 9 | ✅ | OK (max ULP 0) |
| light/light_inception_v2.onnx | 9 | ✅ | OK (max ULP 0) |
| light/light_resnet50.onnx | 9 | ✅ | OK (max ULP 0) |
| light/light_shufflenet.onnx | 9 | ✅ | OK (max ULP 0) |
| light/light_squeezenet.onnx | 9 | ✅ | OK (max ULP 0) |
| light/light_vgg19.onnx | 9 | ✅ | OK (max ULP 0) |
| light/light_zfnet512.onnx | 9 | ✅ | OK (max ULP 0) |
| node/test_abs/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_acos/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_acos_example/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_acosh/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_acosh_example/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_adagrad/model.onnx |  | ✅ | OK (max ULP 0) |
| node/test_adagrad_multiple/model.onnx |  | ✅ | OK (max ULP 0) |
| node/test_adam/model.onnx |  | ❌ | Unsupported op Adam |
| node/test_adam_multiple/model.onnx |  | ❌ | Unsupported op Adam |
| node/test_add/model.onnx | 14 | ✅ | OK (max ULP 0) |
| node/test_add_bcast/model.onnx | 14 | ✅ | OK (max ULP 0) |
| node/test_add_int16/model.onnx | 14 | ✅ | OK (max ULP 0) |
| node/test_add_int8/model.onnx | 14 | ✅ | OK (max ULP 0) |
| node/test_add_uint16/model.onnx | 14 | ✅ | OK (max ULP 0) |
| node/test_add_uint32/model.onnx | 14 | ✅ | OK (max ULP 0) |
| node/test_add_uint64/model.onnx | 14 | ✅ | OK (max ULP 0) |
| node/test_add_uint8/model.onnx | 14 | ✅ | OK (max ULP 0) |
| node/test_affine_grid_2d/model.onnx | 20 | ❌ | Unsupported op AffineGrid |
| node/test_affine_grid_2d_align_corners/model.onnx | 20 | ❌ | Unsupported op AffineGrid |
| node/test_affine_grid_2d_align_corners_expanded/model.onnx | 20 | ❌ | Where inputs must be broadcastable, got ((), (1,), (0,)) |
| node/test_affine_grid_2d_expanded/model.onnx | 20 | ❌ | Where inputs must be broadcastable, got ((), (1,), (0,)) |
| node/test_affine_grid_3d/model.onnx | 20 | ❌ | Unsupported op AffineGrid |
| node/test_affine_grid_3d_align_corners/model.onnx | 20 | ❌ | Unsupported op AffineGrid |
| node/test_affine_grid_3d_align_corners_expanded/model.onnx | 20 | ❌ | Split output shape must be (1,), got (2,) |
| node/test_affine_grid_3d_expanded/model.onnx | 20 | ❌ | Split output shape must be (1,), got (2,) |
| node/test_ai_onnx_ml_array_feature_extractor/model.onnx |  | ❌ | Unsupported op ArrayFeatureExtractor |
| node/test_ai_onnx_ml_binarizer/model.onnx |  | ❌ | Unsupported op Binarizer |
| node/test_ai_onnx_ml_label_encoder_string_int/model.onnx |  | ❌ | Unsupported op LabelEncoder |
| node/test_ai_onnx_ml_label_encoder_string_int_no_default/model.onnx |  | ❌ | Unsupported op LabelEncoder |
| node/test_ai_onnx_ml_label_encoder_tensor_mapping/model.onnx |  | ❌ | Unsupported op LabelEncoder |
| node/test_ai_onnx_ml_label_encoder_tensor_value_only_mapping/model.onnx |  | ❌ | Unsupported op LabelEncoder |
| node/test_ai_onnx_ml_tree_ensemble_set_membership/model.onnx |  | ❌ | Unsupported op TreeEnsemble |
| node/test_ai_onnx_ml_tree_ensemble_single_tree/model.onnx |  | ❌ | Unsupported op TreeEnsemble |
| node/test_and2d/model.onnx | 7 | ✅ | OK (max ULP 0) |
| node/test_and3d/model.onnx | 7 | ✅ | OK (max ULP 0) |
| node/test_and4d/model.onnx | 7 | ✅ | OK (max ULP 0) |
| node/test_and_bcast3v1d/model.onnx | 7 | ✅ | OK (max ULP 0) |
| node/test_and_bcast3v2d/model.onnx | 7 | ✅ | OK (max ULP 0) |
| node/test_and_bcast4v2d/model.onnx | 7 | ✅ | OK (max ULP 0) |
| node/test_and_bcast4v3d/model.onnx | 7 | ✅ | OK (max ULP 0) |
| node/test_and_bcast4v4d/model.onnx | 7 | ✅ | OK (max ULP 0) |
| node/test_argmax_default_axis_example/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_argmax_default_axis_example_select_last_index/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_argmax_default_axis_random/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_argmax_default_axis_random_select_last_index/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_argmax_keepdims_example/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_argmax_keepdims_example_select_last_index/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_argmax_keepdims_random/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_argmax_keepdims_random_select_last_index/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_argmax_negative_axis_keepdims_example/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_argmax_negative_axis_keepdims_example_select_last_index/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_argmax_negative_axis_keepdims_random/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_argmax_negative_axis_keepdims_random_select_last_index/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_argmax_no_keepdims_example/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_argmax_no_keepdims_example_select_last_index/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_argmax_no_keepdims_random/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_argmax_no_keepdims_random_select_last_index/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_argmin_default_axis_example/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_argmin_default_axis_example_select_last_index/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_argmin_default_axis_random/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_argmin_default_axis_random_select_last_index/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_argmin_keepdims_example/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_argmin_keepdims_example_select_last_index/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_argmin_keepdims_random/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_argmin_keepdims_random_select_last_index/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_argmin_negative_axis_keepdims_example/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_argmin_negative_axis_keepdims_example_select_last_index/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_argmin_negative_axis_keepdims_random/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_argmin_negative_axis_keepdims_random_select_last_index/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_argmin_no_keepdims_example/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_argmin_no_keepdims_example_select_last_index/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_argmin_no_keepdims_random/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_argmin_no_keepdims_random_select_last_index/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_asin/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_asin_example/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_asinh/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_asinh_example/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_atan/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_atan_example/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_atanh/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_atanh_example/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_attention_3d/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_3d_attn_mask/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_3d_attn_mask_expanded/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_3d_causal/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_3d_causal_expanded/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_3d_diff_heads_sizes/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_3d_diff_heads_sizes_attn_mask/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_3d_diff_heads_sizes_attn_mask_expanded/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_3d_diff_heads_sizes_causal/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_3d_diff_heads_sizes_causal_expanded/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_3d_diff_heads_sizes_expanded/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_3d_diff_heads_sizes_scaled/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_3d_diff_heads_sizes_scaled_expanded/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_3d_diff_heads_sizes_softcap/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_3d_diff_heads_sizes_softcap_expanded/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_3d_diff_heads_with_past_and_present/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_3d_diff_heads_with_past_and_present_expanded/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_3d_expanded/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_3d_gqa/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_3d_gqa_attn_mask/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_3d_gqa_attn_mask_expanded/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_3d_gqa_causal/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_3d_gqa_causal_expanded/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_3d_gqa_expanded/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_3d_gqa_scaled/model.onnx | 23 | ✅ | OK (max ULP 3) |
| node/test_attention_3d_gqa_scaled_expanded/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_3d_gqa_softcap/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_3d_gqa_softcap_expanded/model.onnx | 23 | ✅ | OK (max ULP 3) |
| node/test_attention_3d_gqa_with_past_and_present/model.onnx | 23 | ✅ | OK (max ULP 3) |
| node/test_attention_3d_gqa_with_past_and_present_expanded/model.onnx | 23 | ✅ | OK (max ULP 3) |
| node/test_attention_3d_scaled/model.onnx | 23 | ✅ | OK (max ULP 3) |
| node/test_attention_3d_scaled_expanded/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_3d_softcap/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_3d_softcap_expanded/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_3d_transpose_verification/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_3d_transpose_verification_expanded/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_3d_with_past_and_present/model.onnx | 23 | ✅ | OK (max ULP 3) |
| node/test_attention_3d_with_past_and_present_expanded/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_3d_with_past_and_present_qk_matmul/model.onnx | 23 | ✅ | OK (max ULP 3) |
| node/test_attention_3d_with_past_and_present_qk_matmul_bias/model.onnx | 23 | ✅ | OK (max ULP 3) |
| node/test_attention_3d_with_past_and_present_qk_matmul_bias_expanded/model.onnx | 23 | ✅ | OK (max ULP 2) |
| node/test_attention_3d_with_past_and_present_qk_matmul_expanded/model.onnx | 23 | ✅ | OK (max ULP 3) |
| node/test_attention_3d_with_past_and_present_qk_matmul_softcap/model.onnx | 23 | ✅ | OK (max ULP 3) |
| node/test_attention_3d_with_past_and_present_qk_matmul_softcap_expanded/model.onnx | 23 | ✅ | OK (max ULP 3) |
| node/test_attention_3d_with_past_and_present_qk_matmul_softmax/model.onnx | 23 | ✅ | OK (max ULP 3) |
| node/test_attention_3d_with_past_and_present_qk_matmul_softmax_expanded/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_4d/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_4d_attn_mask/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_4d_attn_mask_3d/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_4d_attn_mask_3d_causal/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_4d_attn_mask_3d_causal_expanded/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_4d_attn_mask_3d_expanded/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_4d_attn_mask_4d/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_4d_attn_mask_4d_causal/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_4d_attn_mask_4d_causal_expanded/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_4d_attn_mask_4d_expanded/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_4d_attn_mask_bool/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_4d_attn_mask_bool_4d/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_4d_attn_mask_bool_4d_expanded/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_4d_attn_mask_bool_expanded/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_4d_attn_mask_expanded/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_4d_causal/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_4d_causal_expanded/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_4d_diff_heads_mask4d_padded_kv/model.onnx | 24 | ✅ | OK (max ULP 0) |
| node/test_attention_4d_diff_heads_mask4d_padded_kv_expanded/model.onnx | 24 | ❌ | Pad value input must be a scalar |
| node/test_attention_4d_diff_heads_sizes/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_4d_diff_heads_sizes_attn_mask/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_4d_diff_heads_sizes_attn_mask_expanded/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_4d_diff_heads_sizes_causal/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_4d_diff_heads_sizes_causal_expanded/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_4d_diff_heads_sizes_expanded/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_4d_diff_heads_sizes_scaled/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_4d_diff_heads_sizes_scaled_expanded/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_4d_diff_heads_sizes_softcap/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_4d_diff_heads_sizes_softcap_expanded/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_4d_diff_heads_with_past_and_present/model.onnx | 23 | ✅ | OK (max ULP 3) |
| node/test_attention_4d_diff_heads_with_past_and_present_expanded/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_4d_diff_heads_with_past_and_present_mask3d/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_4d_diff_heads_with_past_and_present_mask3d_expanded/model.onnx | 23 | ✅ | OK (max ULP 3) |
| node/test_attention_4d_diff_heads_with_past_and_present_mask4d/model.onnx | 23 | ✅ | OK (max ULP 3) |
| node/test_attention_4d_diff_heads_with_past_and_present_mask4d_expanded/model.onnx | 23 | ✅ | OK (max ULP 3) |
| node/test_attention_4d_expanded/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_4d_fp16/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_4d_fp16_expanded/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_4d_gqa/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_4d_gqa_attn_mask/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_4d_gqa_attn_mask_expanded/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_4d_gqa_causal/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_4d_gqa_causal_expanded/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_4d_gqa_expanded/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_4d_gqa_scaled/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_4d_gqa_scaled_expanded/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_4d_gqa_softcap/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_4d_gqa_softcap_expanded/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_4d_gqa_with_past_and_present/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_4d_gqa_with_past_and_present_expanded/model.onnx | 23 | ✅ | OK (max ULP 3) |
| node/test_attention_4d_gqa_with_past_and_present_fp16/model.onnx | 23 | ✅ | OK (max ULP 3) |
| node/test_attention_4d_gqa_with_past_and_present_fp16_expanded/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_4d_scaled/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_4d_scaled_expanded/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_4d_softcap/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_4d_softcap_expanded/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_4d_with_past_and_present/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_4d_with_past_and_present_expanded/model.onnx | 23 | ✅ | OK (max ULP 3) |
| node/test_attention_4d_with_past_and_present_qk_matmul/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_4d_with_past_and_present_qk_matmul_bias/model.onnx | 23 | ✅ | OK (max ULP 2) |
| node/test_attention_4d_with_past_and_present_qk_matmul_bias_3d_mask/model.onnx | 23 | ✅ | OK (max ULP 2) |
| node/test_attention_4d_with_past_and_present_qk_matmul_bias_3d_mask_causal/model.onnx | 23 | ✅ | OK (max ULP 2) |
| node/test_attention_4d_with_past_and_present_qk_matmul_bias_3d_mask_causal_expanded/model.onnx | 23 | ✅ | OK (max ULP 3) |
| node/test_attention_4d_with_past_and_present_qk_matmul_bias_3d_mask_expanded/model.onnx | 23 | ✅ | OK (max ULP 3) |
| node/test_attention_4d_with_past_and_present_qk_matmul_bias_4d_mask/model.onnx | 23 | ✅ | OK (max ULP 2) |
| node/test_attention_4d_with_past_and_present_qk_matmul_bias_4d_mask_causal/model.onnx | 23 | ✅ | OK (max ULP 2) |
| node/test_attention_4d_with_past_and_present_qk_matmul_bias_4d_mask_causal_expanded/model.onnx | 23 | ✅ | OK (max ULP 2) |
| node/test_attention_4d_with_past_and_present_qk_matmul_bias_4d_mask_expanded/model.onnx | 23 | ✅ | OK (max ULP 2) |
| node/test_attention_4d_with_past_and_present_qk_matmul_bias_expanded/model.onnx | 23 | ✅ | OK (max ULP 3) |
| node/test_attention_4d_with_past_and_present_qk_matmul_expanded/model.onnx | 23 | ✅ | OK (max ULP 3) |
| node/test_attention_4d_with_qk_matmul/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_4d_with_qk_matmul_bias/model.onnx | 23 | ✅ | OK (max ULP 2) |
| node/test_attention_4d_with_qk_matmul_bias_expanded/model.onnx | 23 | ✅ | OK (max ULP 1) |
| node/test_attention_4d_with_qk_matmul_expanded/model.onnx | 23 | ✅ | OK (max ULP 2) |
| node/test_attention_4d_with_qk_matmul_softcap/model.onnx | 23 | ✅ | OK (max ULP 2) |
| node/test_attention_4d_with_qk_matmul_softcap_expanded/model.onnx | 23 | ✅ | OK (max ULP 3) |
| node/test_attention_4d_with_qk_matmul_softmax/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_attention_4d_with_qk_matmul_softmax_expanded/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_averagepool_1d_default/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_averagepool_2d_ceil/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_averagepool_2d_ceil_last_window_starts_on_pad/model.onnx | 22 | ❌ | Out of tolerance (max ULP 2983) |
| node/test_averagepool_2d_default/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_averagepool_2d_dilations/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_averagepool_2d_pads/model.onnx | 22 | ✅ | OK (max ULP 3) |
| node/test_averagepool_2d_pads_count_include_pad/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_averagepool_2d_precomputed_pads/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_averagepool_2d_precomputed_pads_count_include_pad/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_averagepool_2d_precomputed_same_upper/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_averagepool_2d_precomputed_strides/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_averagepool_2d_same_lower/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_averagepool_2d_same_upper/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_averagepool_2d_strides/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_averagepool_3d_default/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_averagepool_3d_dilations_large_count_include_pad_is_0_ceil_mode_is_False/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_averagepool_3d_dilations_large_count_include_pad_is_0_ceil_mode_is_True/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_averagepool_3d_dilations_large_count_include_pad_is_1_ceil_mode_is_False/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_averagepool_3d_dilations_large_count_include_pad_is_1_ceil_mode_is_True/model.onnx | 22 | ❌ | Out of tolerance (max ULP 8165636) |
| node/test_averagepool_3d_dilations_small/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_basic_conv_with_padding/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_basic_conv_without_padding/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_basic_deform_conv_with_padding/model.onnx | 22 | ❌ | Unsupported op DeformConv |
| node/test_basic_deform_conv_without_padding/model.onnx | 22 | ❌ | Unsupported op DeformConv |
| node/test_batchnorm_epsilon/model.onnx | 15 | ✅ | OK (max ULP 2) |
| node/test_batchnorm_epsilon_training_mode/model.onnx | 15 | ✅ | OK (max ULP 1) |
| node/test_batchnorm_example/model.onnx | 15 | ✅ | OK (max ULP 2) |
| node/test_batchnorm_example_training_mode/model.onnx | 15 | ✅ | OK (max ULP 6) |
| node/test_bernoulli/model.onnx | 22 | ✅ | OK (non-deterministic output) |
| node/test_bernoulli_double/model.onnx | 22 | ✅ | OK (non-deterministic output) |
| node/test_bernoulli_double_expanded/model.onnx | 22 | ❌ | Unsupported op RandomUniformLike |
| node/test_bernoulli_expanded/model.onnx | 22 | ❌ | Unsupported op RandomUniformLike |
| node/test_bernoulli_seed/model.onnx | 22 | ✅ | OK (non-deterministic output) |
| node/test_bernoulli_seed_expanded/model.onnx | 22 | ❌ | Unsupported op RandomUniformLike |
| node/test_bitshift_left_uint16/model.onnx | 11 | ✅ | OK (max ULP 0) |
| node/test_bitshift_left_uint32/model.onnx | 11 | ✅ | OK (max ULP 0) |
| node/test_bitshift_left_uint64/model.onnx | 11 | ✅ | OK (max ULP 0) |
| node/test_bitshift_left_uint8/model.onnx | 11 | ✅ | OK (max ULP 0) |
| node/test_bitshift_right_uint16/model.onnx | 11 | ✅ | OK (max ULP 0) |
| node/test_bitshift_right_uint32/model.onnx | 11 | ✅ | OK (max ULP 0) |
| node/test_bitshift_right_uint64/model.onnx | 11 | ✅ | OK (max ULP 0) |
| node/test_bitshift_right_uint8/model.onnx | 11 | ✅ | OK (max ULP 0) |
| node/test_bitwise_and_i16_3d/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_bitwise_and_i32_2d/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_bitwise_and_ui64_bcast_3v1d/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_bitwise_and_ui8_bcast_4v3d/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_bitwise_not_2d/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_bitwise_not_3d/model.onnx | 18 | ❌ | Unsupported op BitwiseNot |
| node/test_bitwise_not_4d/model.onnx | 18 | ❌ | Unsupported op BitwiseNot |
| node/test_bitwise_or_i16_4d/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_bitwise_or_i32_2d/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_bitwise_or_ui64_bcast_3v1d/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_bitwise_or_ui8_bcast_4v3d/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_bitwise_xor_i16_3d/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_bitwise_xor_i32_2d/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_bitwise_xor_ui64_bcast_3v1d/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_bitwise_xor_ui8_bcast_4v3d/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_blackmanwindow/model.onnx | 17 | ❌ | Unsupported op BlackmanWindow |
| node/test_blackmanwindow_expanded/model.onnx | 17 | ✅ | OK (max ULP 4) |
| node/test_blackmanwindow_symmetric/model.onnx | 17 | ❌ | Unsupported op BlackmanWindow |
| node/test_blackmanwindow_symmetric_expanded/model.onnx | 17 | ✅ | OK (max ULP 0) |
| node/test_cast_BFLOAT16_to_FLOAT/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_cast_DOUBLE_to_FLOAT/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_cast_DOUBLE_to_FLOAT16/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_cast_FLOAT16_to_DOUBLE/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_cast_FLOAT16_to_FLOAT/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_cast_FLOAT16_to_FLOAT4E2M1/model.onnx | 25 | ❌ | Unsupported elem_type 23 (FLOAT4E2M1) for tensor 'output'. |
| node/test_cast_FLOAT16_to_FLOAT8E4M3FN/model.onnx | 25 | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'output'. |
| node/test_cast_FLOAT16_to_FLOAT8E4M3FNUZ/model.onnx | 25 | ❌ | Unsupported elem_type 18 (FLOAT8E4M3FNUZ) for tensor 'output'. |
| node/test_cast_FLOAT16_to_FLOAT8E5M2/model.onnx | 25 | ❌ | Unsupported elem_type 19 (FLOAT8E5M2) for tensor 'output'. |
| node/test_cast_FLOAT16_to_FLOAT8E5M2FNUZ/model.onnx | 25 | ❌ | Unsupported elem_type 20 (FLOAT8E5M2FNUZ) for tensor 'output'. |
| node/test_cast_FLOAT16_to_INT2/model.onnx | 25 | ❌ | Unsupported elem_type 26 (INT2) for tensor 'output'. |
| node/test_cast_FLOAT16_to_INT4/model.onnx | 25 | ❌ | Unsupported elem_type 22 (INT4) for tensor 'output'. |
| node/test_cast_FLOAT16_to_UINT2/model.onnx | 25 | ❌ | Unsupported elem_type 25 (UINT2) for tensor 'output'. |
| node/test_cast_FLOAT16_to_UINT4/model.onnx | 25 | ❌ | Unsupported elem_type 21 (UINT4) for tensor 'output'. |
| node/test_cast_FLOAT4E2M1_to_FLOAT/model.onnx | 25 | ❌ | Unsupported elem_type 23 (FLOAT4E2M1) for tensor 'input'. |
| node/test_cast_FLOAT4E2M1_to_FLOAT16/model.onnx | 25 | ❌ | Unsupported elem_type 23 (FLOAT4E2M1) for tensor 'input'. |
| node/test_cast_FLOAT8E4M3FNUZ_to_FLOAT/model.onnx | 25 | ❌ | Unsupported elem_type 18 (FLOAT8E4M3FNUZ) for tensor 'input'. |
| node/test_cast_FLOAT8E4M3FNUZ_to_FLOAT16/model.onnx | 25 | ❌ | Unsupported elem_type 18 (FLOAT8E4M3FNUZ) for tensor 'input'. |
| node/test_cast_FLOAT8E4M3FN_to_FLOAT/model.onnx | 25 | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'input'. |
| node/test_cast_FLOAT8E4M3FN_to_FLOAT16/model.onnx | 25 | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'input'. |
| node/test_cast_FLOAT8E5M2FNUZ_to_FLOAT/model.onnx | 25 | ❌ | Unsupported elem_type 20 (FLOAT8E5M2FNUZ) for tensor 'input'. |
| node/test_cast_FLOAT8E5M2FNUZ_to_FLOAT16/model.onnx | 25 | ❌ | Unsupported elem_type 20 (FLOAT8E5M2FNUZ) for tensor 'input'. |
| node/test_cast_FLOAT8E5M2_to_FLOAT/model.onnx | 25 | ❌ | Unsupported elem_type 19 (FLOAT8E5M2) for tensor 'input'. |
| node/test_cast_FLOAT8E5M2_to_FLOAT16/model.onnx | 25 | ❌ | Unsupported elem_type 19 (FLOAT8E5M2) for tensor 'input'. |
| node/test_cast_FLOAT_to_BFLOAT16/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_cast_FLOAT_to_DOUBLE/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_cast_FLOAT_to_FLOAT16/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_cast_FLOAT_to_FLOAT4E2M1/model.onnx | 25 | ❌ | Unsupported elem_type 23 (FLOAT4E2M1) for tensor 'output'. |
| node/test_cast_FLOAT_to_FLOAT8E4M3FN/model.onnx | 25 | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'output'. |
| node/test_cast_FLOAT_to_FLOAT8E4M3FNUZ/model.onnx | 25 | ❌ | Unsupported elem_type 18 (FLOAT8E4M3FNUZ) for tensor 'output'. |
| node/test_cast_FLOAT_to_FLOAT8E5M2/model.onnx | 25 | ❌ | Unsupported elem_type 19 (FLOAT8E5M2) for tensor 'output'. |
| node/test_cast_FLOAT_to_FLOAT8E5M2FNUZ/model.onnx | 25 | ❌ | Unsupported elem_type 20 (FLOAT8E5M2FNUZ) for tensor 'output'. |
| node/test_cast_FLOAT_to_INT2/model.onnx | 25 | ❌ | Unsupported elem_type 26 (INT2) for tensor 'output'. |
| node/test_cast_FLOAT_to_INT4/model.onnx | 25 | ❌ | Unsupported elem_type 22 (INT4) for tensor 'output'. |
| node/test_cast_FLOAT_to_UINT2/model.onnx | 25 | ❌ | Unsupported elem_type 25 (UINT2) for tensor 'output'. |
| node/test_cast_FLOAT_to_UINT4/model.onnx | 25 | ❌ | Unsupported elem_type 21 (UINT4) for tensor 'output'. |
| node/test_cast_INT2_to_FLOAT/model.onnx | 25 | ❌ | Unsupported elem_type 26 (INT2) for tensor 'input'. |
| node/test_cast_INT2_to_FLOAT16/model.onnx | 25 | ❌ | Unsupported elem_type 26 (INT2) for tensor 'input'. |
| node/test_cast_INT2_to_INT8/model.onnx | 25 | ❌ | Unsupported elem_type 26 (INT2) for tensor 'input'. |
| node/test_cast_INT4_to_FLOAT/model.onnx | 25 | ❌ | Unsupported elem_type 22 (INT4) for tensor 'input'. |
| node/test_cast_INT4_to_FLOAT16/model.onnx | 25 | ❌ | Unsupported elem_type 22 (INT4) for tensor 'input'. |
| node/test_cast_INT4_to_INT8/model.onnx | 25 | ❌ | Unsupported elem_type 22 (INT4) for tensor 'input'. |
| node/test_cast_UINT2_to_FLOAT/model.onnx | 25 | ❌ | Unsupported elem_type 25 (UINT2) for tensor 'input'. |
| node/test_cast_UINT2_to_FLOAT16/model.onnx | 25 | ❌ | Unsupported elem_type 25 (UINT2) for tensor 'input'. |
| node/test_cast_UINT2_to_UINT8/model.onnx | 25 | ❌ | Unsupported elem_type 25 (UINT2) for tensor 'input'. |
| node/test_cast_UINT4_to_FLOAT/model.onnx | 25 | ❌ | Unsupported elem_type 21 (UINT4) for tensor 'input'. |
| node/test_cast_UINT4_to_FLOAT16/model.onnx | 25 | ❌ | Unsupported elem_type 21 (UINT4) for tensor 'input'. |
| node/test_cast_UINT4_to_UINT8/model.onnx | 25 | ❌ | Unsupported elem_type 21 (UINT4) for tensor 'input'. |
| node/test_cast_e8m0_FLOAT16_to_FLOAT8E8M0/model.onnx | 25 | ❌ | Unsupported elem_type 24 (FLOAT8E8M0) for tensor 'output'. |
| node/test_cast_e8m0_FLOAT8E8M0_to_FLOAT/model.onnx | 25 | ❌ | Unsupported elem_type 24 (FLOAT8E8M0) for tensor 'input'. |
| node/test_cast_e8m0_FLOAT8E8M0_to_FLOAT16/model.onnx | 25 | ❌ | Unsupported elem_type 24 (FLOAT8E8M0) for tensor 'input'. |
| node/test_cast_e8m0_FLOAT_to_FLOAT8E8M0/model.onnx | 25 | ❌ | Unsupported elem_type 24 (FLOAT8E8M0) for tensor 'output'. |
| node/test_cast_no_saturate_FLOAT16_to_FLOAT8E4M3FN/model.onnx | 25 | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'output'. |
| node/test_cast_no_saturate_FLOAT16_to_FLOAT8E4M3FNUZ/model.onnx | 25 | ❌ | Unsupported elem_type 18 (FLOAT8E4M3FNUZ) for tensor 'output'. |
| node/test_cast_no_saturate_FLOAT16_to_FLOAT8E5M2/model.onnx | 25 | ❌ | Unsupported elem_type 19 (FLOAT8E5M2) for tensor 'output'. |
| node/test_cast_no_saturate_FLOAT16_to_FLOAT8E5M2FNUZ/model.onnx | 25 | ❌ | Unsupported elem_type 20 (FLOAT8E5M2FNUZ) for tensor 'output'. |
| node/test_cast_no_saturate_FLOAT_to_FLOAT8E4M3FN/model.onnx | 25 | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'output'. |
| node/test_cast_no_saturate_FLOAT_to_FLOAT8E4M3FNUZ/model.onnx | 25 | ❌ | Unsupported elem_type 18 (FLOAT8E4M3FNUZ) for tensor 'output'. |
| node/test_cast_no_saturate_FLOAT_to_FLOAT8E5M2/model.onnx | 25 | ❌ | Unsupported elem_type 19 (FLOAT8E5M2) for tensor 'output'. |
| node/test_cast_no_saturate_FLOAT_to_FLOAT8E5M2FNUZ/model.onnx | 25 | ❌ | Unsupported elem_type 20 (FLOAT8E5M2FNUZ) for tensor 'output'. |
| node/test_castlike_BFLOAT16_to_FLOAT/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_castlike_BFLOAT16_to_FLOAT_expanded/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_castlike_DOUBLE_to_FLOAT/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_castlike_DOUBLE_to_FLOAT16/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_castlike_DOUBLE_to_FLOAT16_expanded/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_castlike_DOUBLE_to_FLOAT_expanded/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_castlike_FLOAT16_to_DOUBLE/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_castlike_FLOAT16_to_DOUBLE_expanded/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_castlike_FLOAT16_to_FLOAT/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_castlike_FLOAT16_to_FLOAT4E2M1/model.onnx | 25 | ❌ | Unsupported elem_type 23 (FLOAT4E2M1) for tensor 'like'. |
| node/test_castlike_FLOAT16_to_FLOAT4E2M1_expanded/model.onnx | 25 | ❌ | Unsupported elem_type 23 (FLOAT4E2M1) for tensor 'like'. |
| node/test_castlike_FLOAT16_to_FLOAT8E4M3FN/model.onnx | 25 | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'like'. |
| node/test_castlike_FLOAT16_to_FLOAT8E4M3FNUZ/model.onnx | 25 | ❌ | Unsupported elem_type 18 (FLOAT8E4M3FNUZ) for tensor 'like'. |
| node/test_castlike_FLOAT16_to_FLOAT8E4M3FNUZ_expanded/model.onnx | 25 | ❌ | Unsupported elem_type 18 (FLOAT8E4M3FNUZ) for tensor 'like'. |
| node/test_castlike_FLOAT16_to_FLOAT8E4M3FN_expanded/model.onnx | 25 | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'like'. |
| node/test_castlike_FLOAT16_to_FLOAT8E5M2/model.onnx | 25 | ❌ | Unsupported elem_type 19 (FLOAT8E5M2) for tensor 'like'. |
| node/test_castlike_FLOAT16_to_FLOAT8E5M2FNUZ/model.onnx | 25 | ❌ | Unsupported elem_type 20 (FLOAT8E5M2FNUZ) for tensor 'like'. |
| node/test_castlike_FLOAT16_to_FLOAT8E5M2FNUZ_expanded/model.onnx | 25 | ❌ | Unsupported elem_type 20 (FLOAT8E5M2FNUZ) for tensor 'like'. |
| node/test_castlike_FLOAT16_to_FLOAT8E5M2_expanded/model.onnx | 25 | ❌ | Unsupported elem_type 19 (FLOAT8E5M2) for tensor 'like'. |
| node/test_castlike_FLOAT16_to_FLOAT_expanded/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_castlike_FLOAT16_to_INT2/model.onnx | 25 | ❌ | Unsupported elem_type 26 (INT2) for tensor 'like'. |
| node/test_castlike_FLOAT16_to_INT2_expanded/model.onnx | 25 | ❌ | Unsupported elem_type 26 (INT2) for tensor 'like'. |
| node/test_castlike_FLOAT16_to_INT4/model.onnx | 25 | ❌ | Unsupported elem_type 22 (INT4) for tensor 'like'. |
| node/test_castlike_FLOAT16_to_INT4_expanded/model.onnx | 25 | ❌ | Unsupported elem_type 22 (INT4) for tensor 'like'. |
| node/test_castlike_FLOAT16_to_UINT2/model.onnx | 25 | ❌ | Unsupported elem_type 25 (UINT2) for tensor 'like'. |
| node/test_castlike_FLOAT16_to_UINT2_expanded/model.onnx | 25 | ❌ | Unsupported elem_type 25 (UINT2) for tensor 'like'. |
| node/test_castlike_FLOAT16_to_UINT4/model.onnx | 25 | ❌ | Unsupported elem_type 21 (UINT4) for tensor 'like'. |
| node/test_castlike_FLOAT16_to_UINT4_expanded/model.onnx | 25 | ❌ | Unsupported elem_type 21 (UINT4) for tensor 'like'. |
| node/test_castlike_FLOAT4E2M1_to_FLOAT/model.onnx | 25 | ❌ | Unsupported elem_type 23 (FLOAT4E2M1) for tensor 'input'. |
| node/test_castlike_FLOAT4E2M1_to_FLOAT16/model.onnx | 25 | ❌ | Unsupported elem_type 23 (FLOAT4E2M1) for tensor 'input'. |
| node/test_castlike_FLOAT4E2M1_to_FLOAT16_expanded/model.onnx | 25 | ❌ | Unsupported elem_type 23 (FLOAT4E2M1) for tensor 'input'. |
| node/test_castlike_FLOAT4E2M1_to_FLOAT_expanded/model.onnx | 25 | ❌ | Unsupported elem_type 23 (FLOAT4E2M1) for tensor 'input'. |
| node/test_castlike_FLOAT8E4M3FNUZ_to_FLOAT/model.onnx | 25 | ❌ | Unsupported elem_type 18 (FLOAT8E4M3FNUZ) for tensor 'input'. |
| node/test_castlike_FLOAT8E4M3FNUZ_to_FLOAT16/model.onnx | 25 | ❌ | Unsupported elem_type 18 (FLOAT8E4M3FNUZ) for tensor 'input'. |
| node/test_castlike_FLOAT8E4M3FNUZ_to_FLOAT16_expanded/model.onnx | 25 | ❌ | Unsupported elem_type 18 (FLOAT8E4M3FNUZ) for tensor 'input'. |
| node/test_castlike_FLOAT8E4M3FNUZ_to_FLOAT_expanded/model.onnx | 25 | ❌ | Unsupported elem_type 18 (FLOAT8E4M3FNUZ) for tensor 'input'. |
| node/test_castlike_FLOAT8E4M3FN_to_FLOAT/model.onnx | 25 | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'input'. |
| node/test_castlike_FLOAT8E4M3FN_to_FLOAT16/model.onnx | 25 | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'input'. |
| node/test_castlike_FLOAT8E4M3FN_to_FLOAT16_expanded/model.onnx | 25 | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'input'. |
| node/test_castlike_FLOAT8E4M3FN_to_FLOAT_expanded/model.onnx | 25 | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'input'. |
| node/test_castlike_FLOAT8E5M2FNUZ_to_FLOAT/model.onnx | 25 | ❌ | Unsupported elem_type 20 (FLOAT8E5M2FNUZ) for tensor 'input'. |
| node/test_castlike_FLOAT8E5M2FNUZ_to_FLOAT16/model.onnx | 25 | ❌ | Unsupported elem_type 20 (FLOAT8E5M2FNUZ) for tensor 'input'. |
| node/test_castlike_FLOAT8E5M2FNUZ_to_FLOAT16_expanded/model.onnx | 25 | ❌ | Unsupported elem_type 20 (FLOAT8E5M2FNUZ) for tensor 'input'. |
| node/test_castlike_FLOAT8E5M2FNUZ_to_FLOAT_expanded/model.onnx | 25 | ❌ | Unsupported elem_type 20 (FLOAT8E5M2FNUZ) for tensor 'input'. |
| node/test_castlike_FLOAT8E5M2_to_FLOAT/model.onnx | 25 | ❌ | Unsupported elem_type 19 (FLOAT8E5M2) for tensor 'input'. |
| node/test_castlike_FLOAT8E5M2_to_FLOAT16/model.onnx | 25 | ❌ | Unsupported elem_type 19 (FLOAT8E5M2) for tensor 'input'. |
| node/test_castlike_FLOAT8E5M2_to_FLOAT16_expanded/model.onnx | 25 | ❌ | Unsupported elem_type 19 (FLOAT8E5M2) for tensor 'input'. |
| node/test_castlike_FLOAT8E5M2_to_FLOAT_expanded/model.onnx | 25 | ❌ | Unsupported elem_type 19 (FLOAT8E5M2) for tensor 'input'. |
| node/test_castlike_FLOAT_to_BFLOAT16/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_castlike_FLOAT_to_BFLOAT16_expanded/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_castlike_FLOAT_to_DOUBLE/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_castlike_FLOAT_to_DOUBLE_expanded/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_castlike_FLOAT_to_FLOAT16/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_castlike_FLOAT_to_FLOAT16_expanded/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_castlike_FLOAT_to_FLOAT4E2M1/model.onnx | 25 | ❌ | Unsupported elem_type 23 (FLOAT4E2M1) for tensor 'like'. |
| node/test_castlike_FLOAT_to_FLOAT4E2M1_expanded/model.onnx | 25 | ❌ | Unsupported elem_type 23 (FLOAT4E2M1) for tensor 'like'. |
| node/test_castlike_FLOAT_to_FLOAT8E4M3FN/model.onnx | 25 | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'like'. |
| node/test_castlike_FLOAT_to_FLOAT8E4M3FNUZ/model.onnx | 25 | ❌ | Unsupported elem_type 18 (FLOAT8E4M3FNUZ) for tensor 'like'. |
| node/test_castlike_FLOAT_to_FLOAT8E4M3FNUZ_expanded/model.onnx | 25 | ❌ | Unsupported elem_type 18 (FLOAT8E4M3FNUZ) for tensor 'like'. |
| node/test_castlike_FLOAT_to_FLOAT8E4M3FN_expanded/model.onnx | 25 | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'like'. |
| node/test_castlike_FLOAT_to_FLOAT8E5M2/model.onnx | 25 | ❌ | Unsupported elem_type 19 (FLOAT8E5M2) for tensor 'like'. |
| node/test_castlike_FLOAT_to_FLOAT8E5M2FNUZ/model.onnx | 25 | ❌ | Unsupported elem_type 20 (FLOAT8E5M2FNUZ) for tensor 'like'. |
| node/test_castlike_FLOAT_to_FLOAT8E5M2FNUZ_expanded/model.onnx | 25 | ❌ | Unsupported elem_type 20 (FLOAT8E5M2FNUZ) for tensor 'like'. |
| node/test_castlike_FLOAT_to_FLOAT8E5M2_expanded/model.onnx | 25 | ❌ | Unsupported elem_type 19 (FLOAT8E5M2) for tensor 'like'. |
| node/test_castlike_FLOAT_to_INT2/model.onnx | 25 | ❌ | Unsupported elem_type 26 (INT2) for tensor 'like'. |
| node/test_castlike_FLOAT_to_INT2_expanded/model.onnx | 25 | ❌ | Unsupported elem_type 26 (INT2) for tensor 'like'. |
| node/test_castlike_FLOAT_to_INT4/model.onnx | 25 | ❌ | Unsupported elem_type 22 (INT4) for tensor 'like'. |
| node/test_castlike_FLOAT_to_INT4_expanded/model.onnx | 25 | ❌ | Unsupported elem_type 22 (INT4) for tensor 'like'. |
| node/test_castlike_FLOAT_to_UINT2/model.onnx | 25 | ❌ | Unsupported elem_type 25 (UINT2) for tensor 'like'. |
| node/test_castlike_FLOAT_to_UINT2_expanded/model.onnx | 25 | ❌ | Unsupported elem_type 25 (UINT2) for tensor 'like'. |
| node/test_castlike_FLOAT_to_UINT4/model.onnx | 25 | ❌ | Unsupported elem_type 21 (UINT4) for tensor 'like'. |
| node/test_castlike_FLOAT_to_UINT4_expanded/model.onnx | 25 | ❌ | Unsupported elem_type 21 (UINT4) for tensor 'like'. |
| node/test_castlike_INT2_to_FLOAT/model.onnx | 25 | ❌ | Unsupported elem_type 26 (INT2) for tensor 'input'. |
| node/test_castlike_INT2_to_FLOAT16/model.onnx | 25 | ❌ | Unsupported elem_type 26 (INT2) for tensor 'input'. |
| node/test_castlike_INT2_to_FLOAT16_expanded/model.onnx | 25 | ❌ | Unsupported elem_type 26 (INT2) for tensor 'input'. |
| node/test_castlike_INT2_to_FLOAT_expanded/model.onnx | 25 | ❌ | Unsupported elem_type 26 (INT2) for tensor 'input'. |
| node/test_castlike_INT2_to_INT8/model.onnx | 25 | ❌ | Unsupported elem_type 26 (INT2) for tensor 'input'. |
| node/test_castlike_INT2_to_INT8_expanded/model.onnx | 25 | ❌ | Unsupported elem_type 26 (INT2) for tensor 'input'. |
| node/test_castlike_INT4_to_FLOAT/model.onnx | 25 | ❌ | Unsupported elem_type 22 (INT4) for tensor 'input'. |
| node/test_castlike_INT4_to_FLOAT16/model.onnx | 25 | ❌ | Unsupported elem_type 22 (INT4) for tensor 'input'. |
| node/test_castlike_INT4_to_FLOAT16_expanded/model.onnx | 25 | ❌ | Unsupported elem_type 22 (INT4) for tensor 'input'. |
| node/test_castlike_INT4_to_FLOAT_expanded/model.onnx | 25 | ❌ | Unsupported elem_type 22 (INT4) for tensor 'input'. |
| node/test_castlike_INT4_to_INT8/model.onnx | 25 | ❌ | Unsupported elem_type 22 (INT4) for tensor 'input'. |
| node/test_castlike_INT4_to_INT8_expanded/model.onnx | 25 | ❌ | Unsupported elem_type 22 (INT4) for tensor 'input'. |
| node/test_castlike_UINT2_to_FLOAT/model.onnx | 25 | ❌ | Unsupported elem_type 25 (UINT2) for tensor 'input'. |
| node/test_castlike_UINT2_to_FLOAT16/model.onnx | 25 | ❌ | Unsupported elem_type 25 (UINT2) for tensor 'input'. |
| node/test_castlike_UINT2_to_FLOAT16_expanded/model.onnx | 25 | ❌ | Unsupported elem_type 25 (UINT2) for tensor 'input'. |
| node/test_castlike_UINT2_to_FLOAT_expanded/model.onnx | 25 | ❌ | Unsupported elem_type 25 (UINT2) for tensor 'input'. |
| node/test_castlike_UINT2_to_UINT8/model.onnx | 25 | ❌ | Unsupported elem_type 25 (UINT2) for tensor 'input'. |
| node/test_castlike_UINT2_to_UINT8_expanded/model.onnx | 25 | ❌ | Unsupported elem_type 25 (UINT2) for tensor 'input'. |
| node/test_castlike_UINT4_to_FLOAT/model.onnx | 25 | ❌ | Unsupported elem_type 21 (UINT4) for tensor 'input'. |
| node/test_castlike_UINT4_to_FLOAT16/model.onnx | 25 | ❌ | Unsupported elem_type 21 (UINT4) for tensor 'input'. |
| node/test_castlike_UINT4_to_FLOAT16_expanded/model.onnx | 25 | ❌ | Unsupported elem_type 21 (UINT4) for tensor 'input'. |
| node/test_castlike_UINT4_to_FLOAT_expanded/model.onnx | 25 | ❌ | Unsupported elem_type 21 (UINT4) for tensor 'input'. |
| node/test_castlike_UINT4_to_UINT8/model.onnx | 25 | ❌ | Unsupported elem_type 21 (UINT4) for tensor 'input'. |
| node/test_castlike_UINT4_to_UINT8_expanded/model.onnx | 25 | ❌ | Unsupported elem_type 21 (UINT4) for tensor 'input'. |
| node/test_castlike_no_saturate_FLOAT16_to_FLOAT8E4M3FN/model.onnx | 25 | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'like'. |
| node/test_castlike_no_saturate_FLOAT16_to_FLOAT8E4M3FNUZ/model.onnx | 25 | ❌ | Unsupported elem_type 18 (FLOAT8E4M3FNUZ) for tensor 'like'. |
| node/test_castlike_no_saturate_FLOAT16_to_FLOAT8E4M3FNUZ_expanded/model.onnx | 25 | ❌ | Unsupported elem_type 18 (FLOAT8E4M3FNUZ) for tensor 'like'. |
| node/test_castlike_no_saturate_FLOAT16_to_FLOAT8E4M3FN_expanded/model.onnx | 25 | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'like'. |
| node/test_castlike_no_saturate_FLOAT16_to_FLOAT8E5M2/model.onnx | 25 | ❌ | Unsupported elem_type 19 (FLOAT8E5M2) for tensor 'like'. |
| node/test_castlike_no_saturate_FLOAT16_to_FLOAT8E5M2FNUZ/model.onnx | 25 | ❌ | Unsupported elem_type 20 (FLOAT8E5M2FNUZ) for tensor 'like'. |
| node/test_castlike_no_saturate_FLOAT16_to_FLOAT8E5M2FNUZ_expanded/model.onnx | 25 | ❌ | Unsupported elem_type 20 (FLOAT8E5M2FNUZ) for tensor 'like'. |
| node/test_castlike_no_saturate_FLOAT16_to_FLOAT8E5M2_expanded/model.onnx | 25 | ❌ | Unsupported elem_type 19 (FLOAT8E5M2) for tensor 'like'. |
| node/test_castlike_no_saturate_FLOAT_to_FLOAT8E4M3FN/model.onnx | 25 | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'like'. |
| node/test_castlike_no_saturate_FLOAT_to_FLOAT8E4M3FNUZ/model.onnx | 25 | ❌ | Unsupported elem_type 18 (FLOAT8E4M3FNUZ) for tensor 'like'. |
| node/test_castlike_no_saturate_FLOAT_to_FLOAT8E4M3FNUZ_expanded/model.onnx | 25 | ❌ | Unsupported elem_type 18 (FLOAT8E4M3FNUZ) for tensor 'like'. |
| node/test_castlike_no_saturate_FLOAT_to_FLOAT8E4M3FN_expanded/model.onnx | 25 | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'like'. |
| node/test_castlike_no_saturate_FLOAT_to_FLOAT8E5M2/model.onnx | 25 | ❌ | Unsupported elem_type 19 (FLOAT8E5M2) for tensor 'like'. |
| node/test_castlike_no_saturate_FLOAT_to_FLOAT8E5M2FNUZ/model.onnx | 25 | ❌ | Unsupported elem_type 20 (FLOAT8E5M2FNUZ) for tensor 'like'. |
| node/test_castlike_no_saturate_FLOAT_to_FLOAT8E5M2FNUZ_expanded/model.onnx | 25 | ❌ | Unsupported elem_type 20 (FLOAT8E5M2FNUZ) for tensor 'like'. |
| node/test_castlike_no_saturate_FLOAT_to_FLOAT8E5M2_expanded/model.onnx | 25 | ❌ | Unsupported elem_type 19 (FLOAT8E5M2) for tensor 'like'. |
| node/test_ceil/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_ceil_example/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_celu/model.onnx | 12 | ✅ | OK (max ULP 0) |
| node/test_celu_expanded/model.onnx | 12 | ✅ | OK (max ULP 0) |
| node/test_center_crop_pad_crop/model.onnx | 18 | ❌ | Unsupported op CenterCropPad |
| node/test_center_crop_pad_crop_and_pad/model.onnx | 18 | ❌ | Unsupported op CenterCropPad |
| node/test_center_crop_pad_crop_and_pad_expanded/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_center_crop_pad_crop_axes_chw/model.onnx | 18 | ❌ | Unsupported op CenterCropPad |
| node/test_center_crop_pad_crop_axes_chw_expanded/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_center_crop_pad_crop_axes_hwc/model.onnx | 18 | ❌ | Unsupported op CenterCropPad |
| node/test_center_crop_pad_crop_axes_hwc_expanded/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_center_crop_pad_crop_expanded/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_center_crop_pad_crop_negative_axes_hwc/model.onnx | 18 | ❌ | Unsupported op CenterCropPad |
| node/test_center_crop_pad_crop_negative_axes_hwc_expanded/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_center_crop_pad_pad/model.onnx | 18 | ❌ | Unsupported op CenterCropPad |
| node/test_center_crop_pad_pad_expanded/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_clip/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_clip_default_inbounds/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_clip_default_inbounds_expanded/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_clip_default_int8_inbounds/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_clip_default_int8_inbounds_expanded/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_clip_default_int8_max/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_clip_default_int8_max_expanded/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_clip_default_int8_min/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_clip_default_int8_min_expanded/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_clip_default_max/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_clip_default_max_expanded/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_clip_default_min/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_clip_default_min_expanded/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_clip_example/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_clip_example_expanded/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_clip_expanded/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_clip_inbounds/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_clip_inbounds_expanded/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_clip_min_greater_than_max/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_clip_min_greater_than_max_expanded/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_clip_outbounds/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_clip_outbounds_expanded/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_clip_splitbounds/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_clip_splitbounds_expanded/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_col2im/model.onnx | 18 | ❌ | Unsupported op Col2Im |
| node/test_col2im_5d/model.onnx | 18 | ❌ | Unsupported op Col2Im |
| node/test_col2im_dilations/model.onnx | 18 | ❌ | Unsupported op Col2Im |
| node/test_col2im_pads/model.onnx | 18 | ❌ | Unsupported op Col2Im |
| node/test_col2im_strides/model.onnx | 18 | ❌ | Unsupported op Col2Im |
| node/test_compress_0/model.onnx | 11 | ✅ | OK (max ULP 0) |
| node/test_compress_1/model.onnx | 11 | ✅ | OK (max ULP 0) |
| node/test_compress_default_axis/model.onnx | 11 | ✅ | OK (max ULP 0) |
| node/test_compress_negative_axis/model.onnx | 11 | ✅ | OK (max ULP 0) |
| node/test_concat_1d_axis_0/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_concat_1d_axis_negative_1/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_concat_2d_axis_0/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_concat_2d_axis_1/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_concat_2d_axis_negative_1/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_concat_2d_axis_negative_2/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_concat_3d_axis_0/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_concat_3d_axis_1/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_concat_3d_axis_2/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_concat_3d_axis_negative_1/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_concat_3d_axis_negative_2/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_concat_3d_axis_negative_3/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_constant/model.onnx | 25 | ❌ | Graph must contain at least one node |
| node/test_constant_pad/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_constant_pad_axes/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_constant_pad_negative_axes/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_constantofshape_float_ones/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_constantofshape_int_shape_zero/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_constantofshape_int_zeros/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_conv_with_autopad_same/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_conv_with_strides_and_asymmetric_padding/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_conv_with_strides_no_padding/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_conv_with_strides_padding/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_convinteger_with_padding/model.onnx | 10 | ✅ | OK (max ULP 0) |
| node/test_convinteger_without_padding/model.onnx | 10 | ✅ | OK (max ULP 0) |
| node/test_convtranspose/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_convtranspose_1d/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_convtranspose_3d/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_convtranspose_autopad_same/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_convtranspose_dilations/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_convtranspose_group_2/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_convtranspose_group_2_image_3/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_convtranspose_kernel_shape/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_convtranspose_output_shape/model.onnx | 22 | ❌ | ConvTranspose output shape must be fully defined and non-negative |
| node/test_convtranspose_pad/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_convtranspose_pads/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_cos/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_cos_example/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_cosh/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_cosh_example/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_cumsum_1d/model.onnx | 14 | ✅ | OK (max ULP 0) |
| node/test_cumsum_1d_exclusive/model.onnx | 14 | ✅ | OK (max ULP 0) |
| node/test_cumsum_1d_int32_exclusive/model.onnx | 14 | ✅ | OK (max ULP 0) |
| node/test_cumsum_1d_reverse/model.onnx | 14 | ✅ | OK (max ULP 0) |
| node/test_cumsum_1d_reverse_exclusive/model.onnx | 14 | ✅ | OK (max ULP 0) |
| node/test_cumsum_2d_axis_0/model.onnx | 14 | ✅ | OK (max ULP 0) |
| node/test_cumsum_2d_axis_1/model.onnx | 14 | ✅ | OK (max ULP 0) |
| node/test_cumsum_2d_int32/model.onnx | 14 | ✅ | OK (max ULP 0) |
| node/test_cumsum_2d_negative_axis/model.onnx | 14 | ✅ | OK (max ULP 0) |
| node/test_deform_conv_with_mask_bias/model.onnx | 22 | ❌ | Unsupported op DeformConv |
| node/test_deform_conv_with_multiple_offset_groups/model.onnx | 22 | ❌ | Unsupported op DeformConv |
| node/test_depthtospace_crd_mode_example/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_depthtospace_example/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_dequantizelinear/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_dequantizelinear_axis/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_dequantizelinear_blocked/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_dequantizelinear_e4m3fn/model.onnx | 25 | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'x'. |
| node/test_dequantizelinear_e4m3fn_float16/model.onnx | 25 | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'x'. |
| node/test_dequantizelinear_e4m3fn_zero_point/model.onnx | 25 | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'x'. |
| node/test_dequantizelinear_e5m2/model.onnx | 25 | ❌ | Unsupported elem_type 19 (FLOAT8E5M2) for tensor 'x'. |
| node/test_dequantizelinear_float4e2m1/model.onnx | 25 | ❌ | Unsupported elem_type 23 (FLOAT4E2M1) for tensor 'x'. |
| node/test_dequantizelinear_int16/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_dequantizelinear_int2/model.onnx | 25 | ❌ | Unsupported elem_type 26 (INT2) for tensor 'x'. |
| node/test_dequantizelinear_int4/model.onnx | 25 | ❌ | Unsupported elem_type 22 (INT4) for tensor 'x'. |
| node/test_dequantizelinear_uint16/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_dequantizelinear_uint2/model.onnx | 25 | ❌ | Unsupported elem_type 25 (UINT2) for tensor 'x'. |
| node/test_dequantizelinear_uint4/model.onnx | 25 | ❌ | Unsupported elem_type 21 (UINT4) for tensor 'x'. |
| node/test_det_2d/model.onnx | 22 | ❌ | Unsupported op Det |
| node/test_det_nd/model.onnx | 22 | ❌ | Unsupported op Det |
| node/test_dft/model.onnx | 20 | ❌ | Unsupported op DFT |
| node/test_dft_axis/model.onnx | 20 | ❌ | Unsupported op DFT |
| node/test_dft_axis_opset19/model.onnx | 19 | ❌ | Unsupported op DFT |
| node/test_dft_inverse/model.onnx | 20 | ❌ | Unsupported op DFT |
| node/test_dft_inverse_opset19/model.onnx | 19 | ❌ | Unsupported op DFT |
| node/test_dft_opset19/model.onnx | 19 | ❌ | Unsupported op DFT |
| node/test_div/model.onnx | 14 | ✅ | OK (max ULP 0) |
| node/test_div_bcast/model.onnx | 14 | ✅ | OK (max ULP 0) |
| node/test_div_example/model.onnx | 14 | ✅ | OK (max ULP 0) |
| node/test_div_int16/model.onnx | 14 | ✅ | OK (max ULP 0) |
| node/test_div_int8/model.onnx | 14 | ✅ | OK (max ULP 0) |
| node/test_div_uint16/model.onnx | 14 | ✅ | OK (max ULP 0) |
| node/test_div_uint32/model.onnx | 14 | ✅ | OK (max ULP 0) |
| node/test_div_uint64/model.onnx | 14 | ✅ | OK (max ULP 0) |
| node/test_div_uint8/model.onnx | 14 | ✅ | OK (max ULP 0) |
| node/test_dropout_default/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_dropout_default_mask/model.onnx | 22 | ❌ | Dropout mask output is not supported |
| node/test_dropout_default_mask_ratio/model.onnx | 22 | ❌ | Dropout supports only the data input and 1 or 2 outputs |
| node/test_dropout_default_old/model.onnx | 11 | ✅ | OK (max ULP 0) |
| node/test_dropout_default_ratio/model.onnx | 22 | ❌ | Dropout supports only the data input and 1 or 2 outputs |
| node/test_dropout_random_old/model.onnx | 11 | ✅ | OK (max ULP 0) |
| node/test_dynamicquantizelinear/model.onnx | 11 | ✅ | OK (max ULP 0) |
| node/test_dynamicquantizelinear_expanded/model.onnx | 11 | ✅ | OK (max ULP 0) |
| node/test_dynamicquantizelinear_max_adjusted/model.onnx | 11 | ✅ | OK (max ULP 0) |
| node/test_dynamicquantizelinear_max_adjusted_expanded/model.onnx | 11 | ✅ | OK (max ULP 0) |
| node/test_dynamicquantizelinear_min_adjusted/model.onnx | 11 | ✅ | OK (max ULP 0) |
| node/test_dynamicquantizelinear_min_adjusted_expanded/model.onnx | 11 | ✅ | OK (max ULP 0) |
| node/test_edge_pad/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_einsum_batch_diagonal/model.onnx | 12 | ✅ | OK (max ULP 0) |
| node/test_einsum_batch_matmul/model.onnx | 12 | ✅ | OK (max ULP 0) |
| node/test_einsum_inner_prod/model.onnx | 12 | ✅ | OK (max ULP 0) |
| node/test_einsum_scalar/model.onnx | 12 | ✅ | OK (max ULP 0) |
| node/test_einsum_sum/model.onnx | 12 | ✅ | OK (max ULP 1) |
| node/test_einsum_transpose/model.onnx | 12 | ✅ | OK (max ULP 0) |
| node/test_elu/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_elu_default/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_elu_default_expanded_ver18/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_elu_example/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_elu_example_expanded_ver18/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_elu_expanded_ver18/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_equal/model.onnx | 19 | ✅ | OK (max ULP 0) |
| node/test_equal_bcast/model.onnx | 19 | ✅ | OK (max ULP 0) |
| node/test_equal_int16/model.onnx | 19 | ✅ | OK (max ULP 0) |
| node/test_equal_int8/model.onnx | 19 | ✅ | OK (max ULP 0) |
| node/test_equal_string/model.onnx | 19 | ✅ | OK (max ULP 0) |
| node/test_equal_string_broadcast/model.onnx | 19 | ✅ | OK (max ULP 0) |
| node/test_equal_uint16/model.onnx | 19 | ✅ | OK (max ULP 0) |
| node/test_equal_uint32/model.onnx | 19 | ✅ | OK (max ULP 0) |
| node/test_equal_uint64/model.onnx | 19 | ✅ | OK (max ULP 0) |
| node/test_equal_uint8/model.onnx | 19 | ✅ | OK (max ULP 0) |
| node/test_erf/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_exp/model.onnx | 13 | ✅ | OK (max ULP 1) |
| node/test_exp_example/model.onnx | 13 | ✅ | OK (max ULP 1) |
| node/test_expand_dim_changed/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_expand_dim_unchanged/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_eyelike_populate_off_main_diagonal/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_eyelike_with_dtype/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_eyelike_without_dtype/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_flatten_axis0/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_flatten_axis1/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_flatten_axis2/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_flatten_axis3/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_flatten_default_axis/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_flatten_negative_axis1/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_flatten_negative_axis2/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_flatten_negative_axis3/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_flatten_negative_axis4/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_floor/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_floor_example/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_gather_0/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_gather_1/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_gather_2d_indices/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_gather_elements_0/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_gather_elements_1/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_gather_elements_negative_indices/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_gather_negative_indices/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_gathernd_example_float32/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_gathernd_example_int32/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_gathernd_example_int32_batch_dim1/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_gelu_default_1/model.onnx | 20 | ✅ | OK (max ULP 0) |
| node/test_gelu_default_1_expanded/model.onnx | 20 | ✅ | OK (max ULP 0) |
| node/test_gelu_default_2/model.onnx | 20 | ✅ | OK (max ULP 0) |
| node/test_gelu_default_2_expanded/model.onnx | 20 | ✅ | OK (max ULP 0) |
| node/test_gelu_tanh_1/model.onnx | 20 | ❌ | Gelu only supports approximate=none |
| node/test_gelu_tanh_1_expanded/model.onnx | 20 | ✅ | OK (max ULP 0) |
| node/test_gelu_tanh_2/model.onnx | 20 | ❌ | Gelu only supports approximate=none |
| node/test_gelu_tanh_2_expanded/model.onnx | 20 | ✅ | OK (max ULP 0) |
| node/test_gemm_all_attributes/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_gemm_alpha/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_gemm_beta/model.onnx | 13 | ✅ | OK (max ULP 1) |
| node/test_gemm_default_matrix_bias/model.onnx | 13 | ✅ | OK (max ULP 1) |
| node/test_gemm_default_no_bias/model.onnx | 13 | ✅ | OK (max ULP 2) |
| node/test_gemm_default_scalar_bias/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_gemm_default_single_elem_vector_bias/model.onnx | 13 | ✅ | OK (max ULP 1) |
| node/test_gemm_default_vector_bias/model.onnx | 13 | ✅ | OK (max ULP 1) |
| node/test_gemm_default_zero_bias/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_gemm_transposeA/model.onnx | 13 | ✅ | OK (max ULP 1) |
| node/test_gemm_transposeB/model.onnx | 13 | ✅ | OK (max ULP 1) |
| node/test_globalaveragepool/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_globalaveragepool_precomputed/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_globalmaxpool/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_globalmaxpool_precomputed/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_greater/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_greater_bcast/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_greater_equal/model.onnx | 16 | ✅ | OK (max ULP 0) |
| node/test_greater_equal_bcast/model.onnx | 16 | ✅ | OK (max ULP 0) |
| node/test_greater_equal_bcast_expanded/model.onnx | 16 | ✅ | OK (max ULP 0) |
| node/test_greater_equal_expanded/model.onnx | 16 | ✅ | OK (max ULP 0) |
| node/test_greater_equal_int16/model.onnx | 16 | ✅ | OK (max ULP 0) |
| node/test_greater_equal_int16_expanded/model.onnx | 16 | ✅ | OK (max ULP 0) |
| node/test_greater_equal_int8/model.onnx | 16 | ✅ | OK (max ULP 0) |
| node/test_greater_equal_int8_expanded/model.onnx | 16 | ✅ | OK (max ULP 0) |
| node/test_greater_equal_uint16/model.onnx | 16 | ✅ | OK (max ULP 0) |
| node/test_greater_equal_uint16_expanded/model.onnx | 16 | ✅ | OK (max ULP 0) |
| node/test_greater_equal_uint32/model.onnx | 16 | ✅ | OK (max ULP 0) |
| node/test_greater_equal_uint32_expanded/model.onnx | 16 | ✅ | OK (max ULP 0) |
| node/test_greater_equal_uint64/model.onnx | 16 | ✅ | OK (max ULP 0) |
| node/test_greater_equal_uint64_expanded/model.onnx | 16 | ✅ | OK (max ULP 0) |
| node/test_greater_equal_uint8/model.onnx | 16 | ✅ | OK (max ULP 0) |
| node/test_greater_equal_uint8_expanded/model.onnx | 16 | ✅ | OK (max ULP 0) |
| node/test_greater_int16/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_greater_int8/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_greater_uint16/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_greater_uint32/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_greater_uint64/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_greater_uint8/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_gridsample/model.onnx | 22 | ✅ | OK (max ULP 2) |
| node/test_gridsample_aligncorners_true/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_gridsample_bicubic/model.onnx | 22 | ❌ | Out of tolerance (max ULP 1678) |
| node/test_gridsample_bicubic_align_corners_0_additional_1/model.onnx | 22 | ✅ | OK (max ULP 13) |
| node/test_gridsample_bicubic_align_corners_1_additional_1/model.onnx | 22 | ✅ | OK (max ULP 34) |
| node/test_gridsample_bilinear/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_gridsample_bilinear_align_corners_0_additional_1/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_gridsample_bilinear_align_corners_1_additional_1/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_gridsample_border_padding/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_gridsample_nearest/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_gridsample_nearest_align_corners_0_additional_1/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_gridsample_nearest_align_corners_1_additional_1/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_gridsample_reflection_padding/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_gridsample_volumetric_bilinear_align_corners_0/model.onnx | 22 | ✅ | OK (max ULP 1) |
| node/test_gridsample_volumetric_bilinear_align_corners_1/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_gridsample_volumetric_nearest_align_corners_0/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_gridsample_volumetric_nearest_align_corners_1/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_gridsample_zeros_padding/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_group_normalization_epsilon/model.onnx | 21 | ✅ | OK (max ULP 0) |
| node/test_group_normalization_epsilon_expanded/model.onnx | 21 | ✅ | OK (max ULP 2) |
| node/test_group_normalization_example/model.onnx | 21 | ✅ | OK (max ULP 0) |
| node/test_group_normalization_example_expanded/model.onnx | 21 | ✅ | OK (max ULP 1) |
| node/test_gru_batchwise/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_gru_defaults/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_gru_seq_length/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_gru_with_initial_bias/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_hammingwindow/model.onnx | 17 | ✅ | OK (max ULP 0) |
| node/test_hammingwindow_expanded/model.onnx | 17 | ✅ | OK (max ULP 0) |
| node/test_hammingwindow_symmetric/model.onnx | 17 | ✅ | OK (max ULP 5) |
| node/test_hammingwindow_symmetric_expanded/model.onnx | 17 | ✅ | OK (max ULP 0) |
| node/test_hannwindow/model.onnx | 17 | ❌ | Unsupported op HannWindow |
| node/test_hannwindow_expanded/model.onnx | 17 | ✅ | OK (max ULP 3) |
| node/test_hannwindow_symmetric/model.onnx | 17 | ❌ | Unsupported op HannWindow |
| node/test_hannwindow_symmetric_expanded/model.onnx | 17 | ✅ | OK (max ULP 0) |
| node/test_hardmax_axis_0/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_hardmax_axis_1/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_hardmax_axis_2/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_hardmax_default_axis/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_hardmax_example/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_hardmax_negative_axis/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_hardmax_one_hot/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_hardsigmoid/model.onnx | 22 | ❌ | HardSigmoid only supports alpha=0.2 |
| node/test_hardsigmoid_default/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_hardsigmoid_default_expanded_ver18/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_hardsigmoid_example/model.onnx | 22 | ❌ | HardSigmoid only supports alpha=0.2 |
| node/test_hardsigmoid_example_expanded_ver18/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_hardsigmoid_expanded_ver18/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_hardswish/model.onnx | 22 | ✅ | OK (max ULP 2) |
| node/test_hardswish_expanded/model.onnx | 22 | ❌ | HardSigmoid only supports alpha=0.2 |
| node/test_identity/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_identity_opt/model.onnx | 16 | ❌ | Unsupported non-tensor value 'opt_in' in op Identity. |
| node/test_identity_sequence/model.onnx | 25 | ❌ | Unsupported non-tensor value 'x' in op Identity. |
| node/test_if/model.onnx | 11 | ✅ | OK (max ULP 0) |
| node/test_if_opt/model.onnx | 16 | ✅ | OK (non-tensor outputs matched; max_abs_diff=0, max_ulp=0) |
| node/test_if_seq/model.onnx | 13 | ✅ | OK (non-tensor outputs matched; max_abs_diff=0, max_ulp=0) |
| node/test_image_decoder_decode_bmp_rgb/model.onnx | 20 | ❌ | Unsupported op ImageDecoder |
| node/test_image_decoder_decode_jpeg2k_rgb/model.onnx | 20 | ❌ | Unsupported op ImageDecoder |
| node/test_image_decoder_decode_jpeg_bgr/model.onnx | 20 | ❌ | Unsupported op ImageDecoder |
| node/test_image_decoder_decode_jpeg_grayscale/model.onnx | 20 | ❌ | Unsupported op ImageDecoder |
| node/test_image_decoder_decode_jpeg_rgb/model.onnx | 20 | ❌ | Unsupported op ImageDecoder |
| node/test_image_decoder_decode_png_rgb/model.onnx | 20 | ❌ | Unsupported op ImageDecoder |
| node/test_image_decoder_decode_pnm_rgb/model.onnx | 20 | ❌ | Unsupported op ImageDecoder |
| node/test_image_decoder_decode_tiff_rgb/model.onnx | 20 | ❌ | Unsupported op ImageDecoder |
| node/test_image_decoder_decode_webp_rgb/model.onnx | 20 | ❌ | Unsupported op ImageDecoder |
| node/test_instancenorm_epsilon/model.onnx | 22 | ✅ | OK (max ULP 2) |
| node/test_instancenorm_example/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_isinf/model.onnx | 20 | ✅ | OK (max ULP 0) |
| node/test_isinf_float16/model.onnx | 20 | ✅ | OK (max ULP 0) |
| node/test_isinf_negative/model.onnx | 20 | ✅ | OK (max ULP 0) |
| node/test_isinf_positive/model.onnx | 20 | ✅ | OK (max ULP 0) |
| node/test_isnan/model.onnx | 20 | ✅ | OK (max ULP 0) |
| node/test_isnan_float16/model.onnx | 20 | ✅ | OK (max ULP 0) |
| node/test_l1normalization_axis_0/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_l1normalization_axis_1/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_l1normalization_axis_last/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_l2normalization_axis_0/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_l2normalization_axis_1/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_layer_normalization_2d_axis0/model.onnx | 17 | ✅ | OK (max ULP 0) |
| node/test_layer_normalization_2d_axis0_expanded/model.onnx | 17 | ✅ | OK (max ULP 2) |
| node/test_layer_normalization_2d_axis0_expanded_ver18/model.onnx | 18 | ✅ | OK (max ULP 2) |
| node/test_layer_normalization_2d_axis1/model.onnx | 17 | ✅ | OK (max ULP 0) |
| node/test_layer_normalization_2d_axis1_expanded/model.onnx | 17 | ✅ | OK (max ULP 24) |
| node/test_layer_normalization_2d_axis1_expanded_ver18/model.onnx | 18 | ✅ | OK (max ULP 24) |
| node/test_layer_normalization_2d_axis_negative_1/model.onnx | 17 | ✅ | OK (max ULP 0) |
| node/test_layer_normalization_2d_axis_negative_1_expanded/model.onnx | 17 | ✅ | OK (max ULP 80) |
| node/test_layer_normalization_2d_axis_negative_1_expanded_ver18/model.onnx | 18 | ✅ | OK (max ULP 80) |
| node/test_layer_normalization_2d_axis_negative_2/model.onnx | 17 | ✅ | OK (max ULP 0) |
| node/test_layer_normalization_2d_axis_negative_2_expanded/model.onnx | 17 | ✅ | OK (max ULP 0) |
| node/test_layer_normalization_2d_axis_negative_2_expanded_ver18/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_layer_normalization_3d_axis0_epsilon/model.onnx | 17 | ✅ | OK (max ULP 0) |
| node/test_layer_normalization_3d_axis0_epsilon_expanded/model.onnx | 17 | ✅ | OK (max ULP 8) |
| node/test_layer_normalization_3d_axis0_epsilon_expanded_ver18/model.onnx | 18 | ✅ | OK (max ULP 8) |
| node/test_layer_normalization_3d_axis1_epsilon/model.onnx | 17 | ✅ | OK (max ULP 0) |
| node/test_layer_normalization_3d_axis1_epsilon_expanded/model.onnx | 17 | ✅ | OK (max ULP 16) |
| node/test_layer_normalization_3d_axis1_epsilon_expanded_ver18/model.onnx | 18 | ✅ | OK (max ULP 16) |
| node/test_layer_normalization_3d_axis2_epsilon/model.onnx | 17 | ✅ | OK (max ULP 0) |
| node/test_layer_normalization_3d_axis2_epsilon_expanded/model.onnx | 17 | ✅ | OK (max ULP 2) |
| node/test_layer_normalization_3d_axis2_epsilon_expanded_ver18/model.onnx | 18 | ✅ | OK (max ULP 2) |
| node/test_layer_normalization_3d_axis_negative_1_epsilon/model.onnx | 17 | ✅ | OK (max ULP 0) |
| node/test_layer_normalization_3d_axis_negative_1_epsilon_expanded/model.onnx | 17 | ✅ | OK (max ULP 2) |
| node/test_layer_normalization_3d_axis_negative_1_epsilon_expanded_ver18/model.onnx | 18 | ✅ | OK (max ULP 2) |
| node/test_layer_normalization_3d_axis_negative_2_epsilon/model.onnx | 17 | ✅ | OK (max ULP 0) |
| node/test_layer_normalization_3d_axis_negative_2_epsilon_expanded/model.onnx | 17 | ✅ | OK (max ULP 4) |
| node/test_layer_normalization_3d_axis_negative_2_epsilon_expanded_ver18/model.onnx | 18 | ✅ | OK (max ULP 4) |
| node/test_layer_normalization_3d_axis_negative_3_epsilon/model.onnx | 17 | ✅ | OK (max ULP 2) |
| node/test_layer_normalization_3d_axis_negative_3_epsilon_expanded/model.onnx | 17 | ✅ | OK (max ULP 9) |
| node/test_layer_normalization_3d_axis_negative_3_epsilon_expanded_ver18/model.onnx | 18 | ✅ | OK (max ULP 9) |
| node/test_layer_normalization_4d_axis0/model.onnx | 17 | ✅ | OK (max ULP 2) |
| node/test_layer_normalization_4d_axis0_expanded/model.onnx | 17 | ✅ | OK (max ULP 6) |
| node/test_layer_normalization_4d_axis0_expanded_ver18/model.onnx | 18 | ✅ | OK (max ULP 6) |
| node/test_layer_normalization_4d_axis1/model.onnx | 17 | ✅ | OK (max ULP 1) |
| node/test_layer_normalization_4d_axis1_expanded/model.onnx | 17 | ✅ | OK (max ULP 12) |
| node/test_layer_normalization_4d_axis1_expanded_ver18/model.onnx | 18 | ✅ | OK (max ULP 12) |
| node/test_layer_normalization_4d_axis2/model.onnx | 17 | ✅ | OK (max ULP 3) |
| node/test_layer_normalization_4d_axis2_expanded/model.onnx | 17 | ✅ | OK (max ULP 8) |
| node/test_layer_normalization_4d_axis2_expanded_ver18/model.onnx | 18 | ✅ | OK (max ULP 8) |
| node/test_layer_normalization_4d_axis3/model.onnx | 17 | ✅ | OK (max ULP 0) |
| node/test_layer_normalization_4d_axis3_expanded/model.onnx | 17 | ✅ | OK (max ULP 12) |
| node/test_layer_normalization_4d_axis3_expanded_ver18/model.onnx | 18 | ✅ | OK (max ULP 12) |
| node/test_layer_normalization_4d_axis_negative_1/model.onnx | 17 | ✅ | OK (max ULP 0) |
| node/test_layer_normalization_4d_axis_negative_1_expanded/model.onnx | 17 | ✅ | OK (max ULP 24) |
| node/test_layer_normalization_4d_axis_negative_1_expanded_ver18/model.onnx | 18 | ✅ | OK (max ULP 24) |
| node/test_layer_normalization_4d_axis_negative_2/model.onnx | 17 | ✅ | OK (max ULP 12) |
| node/test_layer_normalization_4d_axis_negative_2_expanded/model.onnx | 17 | ✅ | OK (max ULP 24) |
| node/test_layer_normalization_4d_axis_negative_2_expanded_ver18/model.onnx | 18 | ✅ | OK (max ULP 24) |
| node/test_layer_normalization_4d_axis_negative_3/model.onnx | 17 | ✅ | OK (max ULP 4) |
| node/test_layer_normalization_4d_axis_negative_3_expanded/model.onnx | 17 | ✅ | OK (max ULP 48) |
| node/test_layer_normalization_4d_axis_negative_3_expanded_ver18/model.onnx | 18 | ✅ | OK (max ULP 48) |
| node/test_layer_normalization_4d_axis_negative_4/model.onnx | 17 | ✅ | OK (max ULP 2) |
| node/test_layer_normalization_4d_axis_negative_4_expanded/model.onnx | 17 | ✅ | OK (max ULP 6) |
| node/test_layer_normalization_4d_axis_negative_4_expanded_ver18/model.onnx | 18 | ✅ | OK (max ULP 6) |
| node/test_layer_normalization_default_axis/model.onnx | 17 | ✅ | OK (max ULP 0) |
| node/test_layer_normalization_default_axis_expanded/model.onnx | 17 | ✅ | OK (max ULP 6) |
| node/test_layer_normalization_default_axis_expanded_ver18/model.onnx | 18 | ✅ | OK (max ULP 6) |
| node/test_leakyrelu/model.onnx | 16 | ✅ | OK (max ULP 0) |
| node/test_leakyrelu_default/model.onnx | 16 | ✅ | OK (max ULP 0) |
| node/test_leakyrelu_default_expanded/model.onnx | 16 | ✅ | OK (max ULP 0) |
| node/test_leakyrelu_example/model.onnx | 16 | ✅ | OK (max ULP 0) |
| node/test_leakyrelu_example_expanded/model.onnx | 16 | ✅ | OK (max ULP 0) |
| node/test_leakyrelu_expanded/model.onnx | 16 | ✅ | OK (max ULP 0) |
| node/test_less/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_less_bcast/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_less_equal/model.onnx | 16 | ✅ | OK (max ULP 0) |
| node/test_less_equal_bcast/model.onnx | 16 | ✅ | OK (max ULP 0) |
| node/test_less_equal_bcast_expanded/model.onnx | 16 | ✅ | OK (max ULP 0) |
| node/test_less_equal_expanded/model.onnx | 16 | ✅ | OK (max ULP 0) |
| node/test_less_equal_int16/model.onnx | 16 | ✅ | OK (max ULP 0) |
| node/test_less_equal_int16_expanded/model.onnx | 16 | ✅ | OK (max ULP 0) |
| node/test_less_equal_int8/model.onnx | 16 | ✅ | OK (max ULP 0) |
| node/test_less_equal_int8_expanded/model.onnx | 16 | ✅ | OK (max ULP 0) |
| node/test_less_equal_uint16/model.onnx | 16 | ✅ | OK (max ULP 0) |
| node/test_less_equal_uint16_expanded/model.onnx | 16 | ✅ | OK (max ULP 0) |
| node/test_less_equal_uint32/model.onnx | 16 | ✅ | OK (max ULP 0) |
| node/test_less_equal_uint32_expanded/model.onnx | 16 | ✅ | OK (max ULP 0) |
| node/test_less_equal_uint64/model.onnx | 16 | ✅ | OK (max ULP 0) |
| node/test_less_equal_uint64_expanded/model.onnx | 16 | ✅ | OK (max ULP 0) |
| node/test_less_equal_uint8/model.onnx | 16 | ✅ | OK (max ULP 0) |
| node/test_less_equal_uint8_expanded/model.onnx | 16 | ✅ | OK (max ULP 0) |
| node/test_less_int16/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_less_int8/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_less_uint16/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_less_uint32/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_less_uint64/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_less_uint8/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_log/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_log_example/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_logsoftmax_axis_0/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_logsoftmax_axis_0_expanded/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_logsoftmax_axis_0_expanded_ver18/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_logsoftmax_axis_1/model.onnx | 13 | ✅ | OK (max ULP 3) |
| node/test_logsoftmax_axis_1_expanded/model.onnx | 13 | ✅ | OK (max ULP 3) |
| node/test_logsoftmax_axis_1_expanded_ver18/model.onnx | 18 | ✅ | OK (max ULP 3) |
| node/test_logsoftmax_axis_2/model.onnx | 13 | ✅ | OK (max ULP 1) |
| node/test_logsoftmax_axis_2_expanded/model.onnx | 13 | ✅ | OK (max ULP 1) |
| node/test_logsoftmax_axis_2_expanded_ver18/model.onnx | 18 | ✅ | OK (max ULP 1) |
| node/test_logsoftmax_default_axis/model.onnx | 13 | ✅ | OK (max ULP 1) |
| node/test_logsoftmax_default_axis_expanded/model.onnx | 13 | ✅ | OK (max ULP 1) |
| node/test_logsoftmax_default_axis_expanded_ver18/model.onnx | 18 | ✅ | OK (max ULP 1) |
| node/test_logsoftmax_example_1/model.onnx | 13 | ✅ | OK (max ULP 1) |
| node/test_logsoftmax_example_1_expanded/model.onnx | 13 | ✅ | OK (max ULP 1) |
| node/test_logsoftmax_example_1_expanded_ver18/model.onnx | 18 | ✅ | OK (max ULP 1) |
| node/test_logsoftmax_large_number/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_logsoftmax_large_number_expanded/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_logsoftmax_large_number_expanded_ver18/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_logsoftmax_negative_axis/model.onnx | 13 | ✅ | OK (max ULP 1) |
| node/test_logsoftmax_negative_axis_expanded/model.onnx | 13 | ✅ | OK (max ULP 1) |
| node/test_logsoftmax_negative_axis_expanded_ver18/model.onnx | 18 | ✅ | OK (max ULP 1) |
| node/test_loop11/model.onnx | 11 | ✅ | OK (max ULP 0) |
| node/test_loop13_seq/model.onnx | 13 | ✅ | OK (non-tensor outputs matched; max_abs_diff=0, max_ulp=58382658) |
| node/test_loop16_seq_none/model.onnx | 16 | ❌ | Unsupported op Loop |
| node/test_lpnormalization_default/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_lppool_1d_default/model.onnx | 22 | ❌ | LpPool expects 2D kernel_shape |
| node/test_lppool_2d_default/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_lppool_2d_dilations/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_lppool_2d_pads/model.onnx | 22 | ✅ | OK (max ULP 1) |
| node/test_lppool_2d_same_lower/model.onnx | 22 | ❌ | LpPool supports auto_pad=NOTSET only |
| node/test_lppool_2d_same_upper/model.onnx | 22 | ❌ | LpPool supports auto_pad=NOTSET only |
| node/test_lppool_2d_strides/model.onnx | 22 | ✅ | OK (max ULP 2) |
| node/test_lppool_3d_default/model.onnx | 22 | ❌ | LpPool expects 2D kernel_shape |
| node/test_lrn/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_lrn_default/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_lstm_batchwise/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_lstm_defaults/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_lstm_with_initial_bias/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_lstm_with_peepholes/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_matmul_1d_1d/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_matmul_1d_3d/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_matmul_2d/model.onnx | 13 | ✅ | OK (max ULP 1) |
| node/test_matmul_3d/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_matmul_4d/model.onnx | 13 | ✅ | OK (max ULP 1) |
| node/test_matmul_4d_1d/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_matmul_bcast/model.onnx | 13 | ✅ | OK (max ULP 2) |
| node/test_matmulinteger/model.onnx | 10 | ✅ | OK (max ULP 0) |
| node/test_max_example/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_max_float16/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_max_float32/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_max_float64/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_max_int16/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_max_int32/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_max_int64/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_max_int8/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_max_one_input/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_max_two_inputs/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_max_uint16/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_max_uint32/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_max_uint64/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_max_uint8/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_maxpool_1d_default/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_maxpool_2d_ceil/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_maxpool_2d_ceil_output_size_reduce_by_one/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_maxpool_2d_default/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_maxpool_2d_dilations/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_maxpool_2d_pads/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_maxpool_2d_precomputed_pads/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_maxpool_2d_precomputed_same_upper/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_maxpool_2d_precomputed_strides/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_maxpool_2d_same_lower/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_maxpool_2d_same_upper/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_maxpool_2d_strides/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_maxpool_2d_uint8/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_maxpool_3d_default/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_maxpool_3d_dilations/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_maxpool_3d_dilations_use_ref_impl/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_maxpool_3d_dilations_use_ref_impl_large/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_maxpool_with_argmax_2d_precomputed_pads/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_maxpool_with_argmax_2d_precomputed_strides/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_maxunpool_export_with_output_shape/model.onnx | 22 | ❌ | Unsupported op MaxUnpool |
| node/test_maxunpool_export_without_output_shape/model.onnx | 22 | ❌ | Unsupported op MaxUnpool |
| node/test_mean_example/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_mean_one_input/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_mean_two_inputs/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_melweightmatrix/model.onnx | 17 | ❌ | Unsupported op MelWeightMatrix |
| node/test_min_example/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_min_float16/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_min_float32/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_min_float64/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_min_int16/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_min_int32/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_min_int64/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_min_int8/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_min_one_input/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_min_two_inputs/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_min_uint16/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_min_uint32/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_min_uint64/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_min_uint8/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_mish/model.onnx | 22 | ✅ | OK (max ULP 1) |
| node/test_mish_expanded/model.onnx | 22 | ✅ | OK (max ULP 1) |
| node/test_mod_broadcast/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_mod_int64_fmod/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_mod_mixed_sign_float16/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_mod_mixed_sign_float32/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_mod_mixed_sign_float64/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_mod_mixed_sign_int16/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_mod_mixed_sign_int32/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_mod_mixed_sign_int64/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_mod_mixed_sign_int8/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_mod_uint16/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_mod_uint32/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_mod_uint64/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_mod_uint8/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_momentum/model.onnx |  | ❌ | Unsupported op Momentum |
| node/test_momentum_multiple/model.onnx |  | ❌ | Unsupported op Momentum |
| node/test_mul/model.onnx | 14 | ✅ | OK (max ULP 0) |
| node/test_mul_bcast/model.onnx | 14 | ✅ | OK (max ULP 0) |
| node/test_mul_example/model.onnx | 14 | ✅ | OK (max ULP 0) |
| node/test_mul_int16/model.onnx | 14 | ✅ | OK (max ULP 0) |
| node/test_mul_int8/model.onnx | 14 | ✅ | OK (max ULP 0) |
| node/test_mul_uint16/model.onnx | 14 | ✅ | OK (max ULP 0) |
| node/test_mul_uint32/model.onnx | 14 | ✅ | OK (max ULP 0) |
| node/test_mul_uint64/model.onnx | 14 | ✅ | OK (max ULP 0) |
| node/test_mul_uint8/model.onnx | 14 | ✅ | OK (max ULP 0) |
| node/test_mvn/model.onnx | 13 | ✅ | OK (max ULP 3) |
| node/test_mvn_expanded/model.onnx | 13 | ✅ | OK (max ULP 2) |
| node/test_mvn_expanded_ver18/model.onnx | 18 | ✅ | OK (max ULP 2) |
| node/test_neg/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_neg_example/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_nesterov_momentum/model.onnx |  | ❌ | Unsupported op Momentum |
| node/test_nllloss_NC/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_nllloss_NC_expanded/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_nllloss_NCd1/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_nllloss_NCd1_expanded/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_nllloss_NCd1_ii/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_nllloss_NCd1_ii_expanded/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_nllloss_NCd1_mean_weight_negative_ii/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_nllloss_NCd1_mean_weight_negative_ii_expanded/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_nllloss_NCd1_weight/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_nllloss_NCd1_weight_expanded/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_nllloss_NCd1_weight_ii/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_nllloss_NCd1_weight_ii_expanded/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_nllloss_NCd1d2/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_nllloss_NCd1d2_expanded/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_nllloss_NCd1d2_no_weight_reduction_mean_ii/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_nllloss_NCd1d2_no_weight_reduction_mean_ii_expanded/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_nllloss_NCd1d2_reduction_mean/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_nllloss_NCd1d2_reduction_mean_expanded/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_nllloss_NCd1d2_reduction_sum/model.onnx | 22 | ✅ | OK (max ULP 3) |
| node/test_nllloss_NCd1d2_reduction_sum_expanded/model.onnx | 22 | ✅ | OK (max ULP 3) |
| node/test_nllloss_NCd1d2_with_weight/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_nllloss_NCd1d2_with_weight_expanded/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_nllloss_NCd1d2_with_weight_reduction_mean/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_nllloss_NCd1d2_with_weight_reduction_mean_expanded/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_nllloss_NCd1d2_with_weight_reduction_sum/model.onnx | 22 | ✅ | OK (max ULP 1) |
| node/test_nllloss_NCd1d2_with_weight_reduction_sum_expanded/model.onnx | 22 | ✅ | OK (max ULP 1) |
| node/test_nllloss_NCd1d2_with_weight_reduction_sum_ii/model.onnx | 22 | ✅ | OK (max ULP 3) |
| node/test_nllloss_NCd1d2_with_weight_reduction_sum_ii_expanded/model.onnx | 22 | ✅ | OK (max ULP 3) |
| node/test_nllloss_NCd1d2d3_none_no_weight_negative_ii/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_nllloss_NCd1d2d3_none_no_weight_negative_ii_expanded/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_nllloss_NCd1d2d3_sum_weight_high_ii/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_nllloss_NCd1d2d3_sum_weight_high_ii_expanded/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_nllloss_NCd1d2d3d4d5_mean_weight/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_nllloss_NCd1d2d3d4d5_mean_weight_expanded/model.onnx | 22 | ❌ | Out of tolerance (max ULP 357) |
| node/test_nllloss_NCd1d2d3d4d5_none_no_weight/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_nllloss_NCd1d2d3d4d5_none_no_weight_expanded/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_nonmaxsuppression_center_point_box_format/model.onnx | 11 | ✅ | OK (max ULP 0) |
| node/test_nonmaxsuppression_flipped_coordinates/model.onnx | 11 | ✅ | OK (max ULP 0) |
| node/test_nonmaxsuppression_identical_boxes/model.onnx | 11 | ✅ | OK (max ULP 0) |
| node/test_nonmaxsuppression_limit_output_size/model.onnx | 11 | ✅ | OK (max ULP 0) |
| node/test_nonmaxsuppression_single_box/model.onnx | 11 | ✅ | OK (max ULP 0) |
| node/test_nonmaxsuppression_suppress_by_IOU/model.onnx | 11 | ✅ | OK (max ULP 0) |
| node/test_nonmaxsuppression_suppress_by_IOU_and_scores/model.onnx | 11 | ✅ | OK (max ULP 0) |
| node/test_nonmaxsuppression_two_batches/model.onnx | 11 | ✅ | OK (max ULP 0) |
| node/test_nonmaxsuppression_two_classes/model.onnx | 11 | ✅ | OK (max ULP 0) |
| node/test_nonzero_example/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_not_2d/model.onnx | 1 | ✅ | OK (max ULP 0) |
| node/test_not_3d/model.onnx | 1 | ✅ | OK (max ULP 0) |
| node/test_not_4d/model.onnx | 1 | ✅ | OK (max ULP 0) |
| node/test_onehot_negative_indices/model.onnx | 11 | ✅ | OK (max ULP 0) |
| node/test_onehot_with_axis/model.onnx | 11 | ✅ | OK (max ULP 0) |
| node/test_onehot_with_negative_axis/model.onnx | 11 | ✅ | OK (max ULP 0) |
| node/test_onehot_without_axis/model.onnx | 11 | ✅ | OK (max ULP 0) |
| node/test_optional_get_element_optional_sequence/model.onnx | 18 | ❌ | Unsupported op OptionalGetElement |
| node/test_optional_get_element_optional_tensor/model.onnx | 18 | ❌ | Unsupported op OptionalGetElement |
| node/test_optional_get_element_sequence/model.onnx | 18 | ❌ | Unsupported op OptionalGetElement |
| node/test_optional_get_element_tensor/model.onnx | 18 | ❌ | Unsupported op OptionalGetElement |
| node/test_optional_has_element_empty_no_input_name_optional_input/model.onnx | 18 | ❌ | OptionalHasElement expects exactly one non-empty input. |
| node/test_optional_has_element_empty_no_input_name_tensor_input/model.onnx | 18 | ❌ | OptionalHasElement expects exactly one non-empty input. |
| node/test_optional_has_element_empty_no_input_optional_input/model.onnx | 18 | ❌ | OptionalHasElement expects exactly one non-empty input. |
| node/test_optional_has_element_empty_no_input_tensor_input/model.onnx | 18 | ❌ | OptionalHasElement expects exactly one non-empty input. |
| node/test_optional_has_element_empty_optional_input/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_optional_has_element_optional_input/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_optional_has_element_tensor_input/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_or2d/model.onnx | 7 | ✅ | OK (max ULP 0) |
| node/test_or3d/model.onnx | 7 | ✅ | OK (max ULP 0) |
| node/test_or4d/model.onnx | 7 | ✅ | OK (max ULP 0) |
| node/test_or_bcast3v1d/model.onnx | 7 | ✅ | OK (max ULP 0) |
| node/test_or_bcast3v2d/model.onnx | 7 | ✅ | OK (max ULP 0) |
| node/test_or_bcast4v2d/model.onnx | 7 | ✅ | OK (max ULP 0) |
| node/test_or_bcast4v3d/model.onnx | 7 | ✅ | OK (max ULP 0) |
| node/test_or_bcast4v4d/model.onnx | 7 | ✅ | OK (max ULP 0) |
| node/test_pow/model.onnx | 15 | ✅ | OK (max ULP 0) |
| node/test_pow_bcast_array/model.onnx | 15 | ✅ | OK (max ULP 0) |
| node/test_pow_bcast_scalar/model.onnx | 15 | ✅ | OK (max ULP 0) |
| node/test_pow_example/model.onnx | 15 | ✅ | OK (max ULP 0) |
| node/test_pow_types_float32_int32/model.onnx | 15 | ✅ | OK (max ULP 0) |
| node/test_pow_types_float32_int64/model.onnx | 15 | ✅ | OK (max ULP 0) |
| node/test_pow_types_float32_uint32/model.onnx | 15 | ✅ | OK (max ULP 0) |
| node/test_pow_types_float32_uint64/model.onnx | 15 | ✅ | OK (max ULP 0) |
| node/test_pow_types_int32_float32/model.onnx | 15 | ✅ | OK (max ULP 0) |
| node/test_pow_types_int32_int32/model.onnx | 15 | ✅ | OK (max ULP 0) |
| node/test_pow_types_int64_float32/model.onnx | 15 | ✅ | OK (max ULP 0) |
| node/test_pow_types_int64_int64/model.onnx | 15 | ✅ | OK (max ULP 0) |
| node/test_prelu_broadcast/model.onnx | 16 | ✅ | OK (max ULP 0) |
| node/test_prelu_broadcast_expanded/model.onnx | 16 | ✅ | OK (max ULP 0) |
| node/test_prelu_example/model.onnx | 16 | ✅ | OK (max ULP 0) |
| node/test_prelu_example_expanded/model.onnx | 16 | ✅ | OK (max ULP 0) |
| node/test_qlinearconv/model.onnx | 10 | ✅ | OK (max ULP 0) |
| node/test_qlinearmatmul_2D_int8_float16/model.onnx | 21 | ❌ | Arrays are not equal (max abs diff 148) |
| node/test_qlinearmatmul_2D_int8_float32/model.onnx | 21 | ❌ | Arrays are not equal (max abs diff 148) |
| node/test_qlinearmatmul_2D_uint8_float16/model.onnx | 21 | ✅ | OK (max ULP 0) |
| node/test_qlinearmatmul_2D_uint8_float32/model.onnx | 21 | ✅ | OK (max ULP 0) |
| node/test_qlinearmatmul_3D_int8_float16/model.onnx | 21 | ❌ | Arrays are not equal (max abs diff 247) |
| node/test_qlinearmatmul_3D_int8_float32/model.onnx | 21 | ❌ | Arrays are not equal (max abs diff 248) |
| node/test_qlinearmatmul_3D_uint8_float16/model.onnx | 21 | ✅ | OK (max ULP 0) |
| node/test_qlinearmatmul_3D_uint8_float32/model.onnx | 21 | ✅ | OK (max ULP 0) |
| node/test_quantizelinear/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_quantizelinear_axis/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_quantizelinear_blocked_asymmetric/model.onnx | 25 | ❌ | QuantizeLinear block_size is not supported |
| node/test_quantizelinear_blocked_symmetric/model.onnx | 25 | ❌ | QuantizeLinear block_size is not supported |
| node/test_quantizelinear_e4m3fn/model.onnx | 25 | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'y_zero_point'. |
| node/test_quantizelinear_e5m2/model.onnx | 25 | ❌ | Unsupported elem_type 19 (FLOAT8E5M2) for tensor 'y_zero_point'. |
| node/test_quantizelinear_float4e2m1/model.onnx | 25 | ❌ | Unsupported elem_type 23 (FLOAT4E2M1) for tensor 'y_zero_point'. |
| node/test_quantizelinear_int16/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_quantizelinear_int2/model.onnx | 25 | ❌ | Unsupported elem_type 26 (INT2) for tensor 'y_zero_point'. |
| node/test_quantizelinear_int4/model.onnx | 25 | ❌ | Unsupported elem_type 22 (INT4) for tensor 'y_zero_point'. |
| node/test_quantizelinear_uint16/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_quantizelinear_uint2/model.onnx | 25 | ❌ | Unsupported elem_type 25 (UINT2) for tensor 'y_zero_point'. |
| node/test_quantizelinear_uint4/model.onnx | 25 | ❌ | Unsupported elem_type 21 (UINT4) for tensor 'y_zero_point'. |
| node/test_range_float_type_positive_delta/model.onnx | 11 | ✅ | OK (max ULP 0) |
| node/test_range_float_type_positive_delta_expanded/model.onnx | 11 | ✅ | OK (max ULP 0) |
| node/test_range_int32_type_negative_delta/model.onnx | 11 | ✅ | OK (max ULP 0) |
| node/test_range_int32_type_negative_delta_expanded/model.onnx | 11 | ✅ | OK (max ULP 0) |
| node/test_reciprocal/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_reciprocal_example/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_reduce_l1_default_axes_keepdims_example/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_l1_default_axes_keepdims_example_expanded/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_l1_default_axes_keepdims_random/model.onnx | 18 | ✅ | OK (max ULP 1) |
| node/test_reduce_l1_default_axes_keepdims_random_expanded/model.onnx | 18 | ✅ | OK (max ULP 1) |
| node/test_reduce_l1_do_not_keepdims_example/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_l1_do_not_keepdims_example_expanded/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_l1_do_not_keepdims_random/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_l1_do_not_keepdims_random_expanded/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_l1_empty_set/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_l1_empty_set_expanded/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_l1_keep_dims_example/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_l1_keep_dims_example_expanded/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_l1_keep_dims_random/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_l1_keep_dims_random_expanded/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_l1_negative_axes_keep_dims_example/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_l1_negative_axes_keep_dims_example_expanded/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_l1_negative_axes_keep_dims_random/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_l1_negative_axes_keep_dims_random_expanded/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_l2_default_axes_keepdims_example/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_l2_default_axes_keepdims_example_expanded/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_l2_default_axes_keepdims_random/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_l2_default_axes_keepdims_random_expanded/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_l2_do_not_keepdims_example/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_l2_do_not_keepdims_example_expanded/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_l2_do_not_keepdims_random/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_l2_do_not_keepdims_random_expanded/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_l2_empty_set/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_l2_empty_set_expanded/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_l2_keep_dims_example/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_l2_keep_dims_example_expanded/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_l2_keep_dims_random/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_l2_keep_dims_random_expanded/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_l2_negative_axes_keep_dims_example/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_l2_negative_axes_keep_dims_example_expanded/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_l2_negative_axes_keep_dims_random/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_l2_negative_axes_keep_dims_random_expanded/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_log_sum_asc_axes/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_log_sum_asc_axes_expanded/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_log_sum_default/model.onnx | 18 | ✅ | OK (max ULP 1) |
| node/test_reduce_log_sum_default_expanded/model.onnx | 18 | ✅ | OK (max ULP 1) |
| node/test_reduce_log_sum_desc_axes/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_log_sum_desc_axes_expanded/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_log_sum_empty_set/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_log_sum_empty_set_expanded/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_log_sum_exp_default_axes_keepdims_example/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_log_sum_exp_default_axes_keepdims_example_expanded/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_log_sum_exp_default_axes_keepdims_random/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_log_sum_exp_default_axes_keepdims_random_expanded/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_log_sum_exp_do_not_keepdims_example/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_log_sum_exp_do_not_keepdims_example_expanded/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_log_sum_exp_do_not_keepdims_random/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_log_sum_exp_do_not_keepdims_random_expanded/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_log_sum_exp_empty_set/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_log_sum_exp_empty_set_expanded/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_log_sum_exp_keepdims_example/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_log_sum_exp_keepdims_example_expanded/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_log_sum_exp_keepdims_random/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_log_sum_exp_keepdims_random_expanded/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_log_sum_exp_negative_axes_keepdims_example/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_log_sum_exp_negative_axes_keepdims_example_expanded/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_log_sum_exp_negative_axes_keepdims_random/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_log_sum_exp_negative_axes_keepdims_random_expanded/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_log_sum_negative_axes/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_log_sum_negative_axes_expanded/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_max_bool_inputs/model.onnx | 20 | ❌ | ReduceMax does not support dtype bool |
| node/test_reduce_max_default_axes_keepdim_example/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_max_default_axes_keepdims_random/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_max_do_not_keepdims_example/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_max_do_not_keepdims_random/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_max_empty_set/model.onnx | 20 | ✅ | OK (max ULP 0) |
| node/test_reduce_max_keepdims_example/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_max_keepdims_random/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_max_negative_axes_keepdims_example/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_max_negative_axes_keepdims_random/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_mean_default_axes_keepdims_example/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_mean_default_axes_keepdims_random/model.onnx | 18 | ✅ | OK (max ULP 2) |
| node/test_reduce_mean_do_not_keepdims_example/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_mean_do_not_keepdims_random/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_mean_keepdims_example/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_mean_keepdims_random/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_mean_negative_axes_keepdims_example/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_mean_negative_axes_keepdims_random/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_min_bool_inputs/model.onnx | 20 | ❌ | ReduceMin does not support dtype bool |
| node/test_reduce_min_default_axes_keepdims_example/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_min_default_axes_keepdims_random/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_min_do_not_keepdims_example/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_min_do_not_keepdims_random/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_min_empty_set/model.onnx | 20 | ✅ | OK (max ULP 0) |
| node/test_reduce_min_keepdims_example/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_min_keepdims_random/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_min_negative_axes_keepdims_example/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_min_negative_axes_keepdims_random/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_prod_default_axes_keepdims_example/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_prod_default_axes_keepdims_random/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_prod_do_not_keepdims_example/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_prod_do_not_keepdims_random/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_prod_empty_set/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_prod_keepdims_example/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_prod_keepdims_random/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_prod_negative_axes_keepdims_example/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_prod_negative_axes_keepdims_random/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_sum_default_axes_keepdims_example/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_reduce_sum_default_axes_keepdims_random/model.onnx | 13 | ✅ | OK (max ULP 2) |
| node/test_reduce_sum_do_not_keepdims_example/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_reduce_sum_do_not_keepdims_random/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_reduce_sum_empty_axes_input_noop/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_reduce_sum_empty_axes_input_noop_example/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_reduce_sum_empty_set/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_reduce_sum_empty_set_non_reduced_axis_zero/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_reduce_sum_keepdims_example/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_reduce_sum_keepdims_random/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_reduce_sum_negative_axes_keepdims_example/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_reduce_sum_negative_axes_keepdims_random/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_reduce_sum_square_default_axes_keepdims_example/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_sum_square_default_axes_keepdims_example_expanded/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_sum_square_default_axes_keepdims_random/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_sum_square_default_axes_keepdims_random_expanded/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_sum_square_do_not_keepdims_example/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_sum_square_do_not_keepdims_example_expanded/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_sum_square_do_not_keepdims_random/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_sum_square_do_not_keepdims_random_expanded/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_sum_square_empty_set/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_sum_square_empty_set_expanded/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_sum_square_keepdims_example/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_sum_square_keepdims_example_expanded/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_sum_square_keepdims_random/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_sum_square_keepdims_random_expanded/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_sum_square_negative_axes_keepdims_example/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_sum_square_negative_axes_keepdims_example_expanded/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_sum_square_negative_axes_keepdims_random/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reduce_sum_square_negative_axes_keepdims_random_expanded/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reflect_pad/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_regex_full_match_basic/model.onnx | 20 | ❌ | Unsupported op RegexFullMatch |
| node/test_regex_full_match_email_domain/model.onnx | 20 | ❌ | Unsupported op RegexFullMatch |
| node/test_regex_full_match_empty/model.onnx | 20 | ❌ | Unsupported op RegexFullMatch |
| node/test_relu/model.onnx | 14 | ✅ | OK (max ULP 0) |
| node/test_relu_expanded_ver18/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_reshape_allowzero_reordered/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_reshape_extended_dims/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_reshape_negative_dim/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_reshape_negative_extended_dims/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_reshape_one_dim/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_reshape_reduced_dims/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_reshape_reordered_all_dims/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_reshape_reordered_last_dims/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_reshape_zero_and_negative_dim/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_reshape_zero_dim/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_resize_downsample_scales_cubic/model.onnx | 19 | ✅ | OK (max ULP 0) |
| node/test_resize_downsample_scales_cubic_A_n0p5_exclude_outside/model.onnx | 19 | ✅ | OK (max ULP 0) |
| node/test_resize_downsample_scales_cubic_align_corners/model.onnx | 19 | ❌ | Out of tolerance (max ULP 1098996) |
| node/test_resize_downsample_scales_cubic_antialias/model.onnx | 19 | ✅ | OK (max ULP 0) |
| node/test_resize_downsample_scales_linear/model.onnx | 19 | ✅ | OK (max ULP 0) |
| node/test_resize_downsample_scales_linear_align_corners/model.onnx | 19 | ❌ | Out of tolerance (max ULP 3595118) |
| node/test_resize_downsample_scales_linear_antialias/model.onnx | 19 | ✅ | OK (max ULP 0) |
| node/test_resize_downsample_scales_linear_half_pixel_symmetric/model.onnx | 19 | ✅ | OK (max ULP 0) |
| node/test_resize_downsample_scales_nearest/model.onnx | 19 | ✅ | OK (max ULP 0) |
| node/test_resize_downsample_sizes_cubic/model.onnx | 19 | ✅ | OK (max ULP 0) |
| node/test_resize_downsample_sizes_cubic_antialias/model.onnx | 19 | ✅ | OK (max ULP 0) |
| node/test_resize_downsample_sizes_linear_antialias/model.onnx | 19 | ✅ | OK (max ULP 0) |
| node/test_resize_downsample_sizes_linear_pytorch_half_pixel/model.onnx | 19 | ✅ | OK (max ULP 0) |
| node/test_resize_downsample_sizes_nearest/model.onnx | 19 | ✅ | OK (max ULP 0) |
| node/test_resize_downsample_sizes_nearest_not_larger/model.onnx | 19 | ✅ | OK (max ULP 0) |
| node/test_resize_downsample_sizes_nearest_not_smaller/model.onnx | 19 | ✅ | OK (max ULP 0) |
| node/test_resize_tf_crop_and_resize/model.onnx | 19 | ✅ | OK (max ULP 0) |
| node/test_resize_tf_crop_and_resize_axes_2_3/model.onnx | 19 | ✅ | OK (max ULP 0) |
| node/test_resize_tf_crop_and_resize_axes_3_2/model.onnx | 19 | ✅ | OK (max ULP 0) |
| node/test_resize_tf_crop_and_resize_extrapolation_value/model.onnx | 19 | ✅ | OK (max ULP 0) |
| node/test_resize_upsample_scales_cubic/model.onnx | 19 | ✅ | OK (max ULP 0) |
| node/test_resize_upsample_scales_cubic_A_n0p5_exclude_outside/model.onnx | 19 | ✅ | OK (max ULP 0) |
| node/test_resize_upsample_scales_cubic_align_corners/model.onnx | 19 | ✅ | OK (max ULP 0) |
| node/test_resize_upsample_scales_cubic_asymmetric/model.onnx | 19 | ✅ | OK (max ULP 0) |
| node/test_resize_upsample_scales_linear/model.onnx | 19 | ✅ | OK (max ULP 0) |
| node/test_resize_upsample_scales_linear_align_corners/model.onnx | 19 | ✅ | OK (max ULP 0) |
| node/test_resize_upsample_scales_linear_half_pixel_symmetric/model.onnx | 19 | ✅ | OK (max ULP 0) |
| node/test_resize_upsample_scales_nearest/model.onnx | 19 | ✅ | OK (max ULP 0) |
| node/test_resize_upsample_scales_nearest_axes_2_3/model.onnx | 19 | ✅ | OK (max ULP 0) |
| node/test_resize_upsample_scales_nearest_axes_3_2/model.onnx | 19 | ✅ | OK (max ULP 0) |
| node/test_resize_upsample_sizes_cubic/model.onnx | 19 | ✅ | OK (max ULP 0) |
| node/test_resize_upsample_sizes_nearest/model.onnx | 19 | ✅ | OK (max ULP 0) |
| node/test_resize_upsample_sizes_nearest_axes_2_3/model.onnx | 19 | ✅ | OK (max ULP 0) |
| node/test_resize_upsample_sizes_nearest_axes_3_2/model.onnx | 19 | ✅ | OK (max ULP 0) |
| node/test_resize_upsample_sizes_nearest_ceil_half_pixel/model.onnx | 19 | ✅ | OK (max ULP 0) |
| node/test_resize_upsample_sizes_nearest_floor_align_corners/model.onnx | 19 | ✅ | OK (max ULP 0) |
| node/test_resize_upsample_sizes_nearest_not_larger/model.onnx | 19 | ✅ | OK (max ULP 0) |
| node/test_resize_upsample_sizes_nearest_not_smaller/model.onnx | 19 | ✅ | OK (max ULP 0) |
| node/test_resize_upsample_sizes_nearest_round_prefer_ceil_asymmetric/model.onnx | 19 | ✅ | OK (max ULP 0) |
| node/test_reversesequence_batch/model.onnx | 10 | ✅ | OK (max ULP 0) |
| node/test_reversesequence_time/model.onnx | 10 | ✅ | OK (max ULP 0) |
| node/test_rms_normalization_2d_axis0/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_rms_normalization_2d_axis0_expanded/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_rms_normalization_2d_axis1/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_rms_normalization_2d_axis1_expanded/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_rms_normalization_2d_axis_negative_1/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_rms_normalization_2d_axis_negative_1_expanded/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_rms_normalization_2d_axis_negative_2/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_rms_normalization_2d_axis_negative_2_expanded/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_rms_normalization_3d_axis0_epsilon/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_rms_normalization_3d_axis0_epsilon_expanded/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_rms_normalization_3d_axis1_epsilon/model.onnx | 23 | ✅ | OK (max ULP 2) |
| node/test_rms_normalization_3d_axis1_epsilon_expanded/model.onnx | 23 | ✅ | OK (max ULP 2) |
| node/test_rms_normalization_3d_axis2_epsilon/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_rms_normalization_3d_axis2_epsilon_expanded/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_rms_normalization_3d_axis_negative_1_epsilon/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_rms_normalization_3d_axis_negative_1_epsilon_expanded/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_rms_normalization_3d_axis_negative_2_epsilon/model.onnx | 23 | ✅ | OK (max ULP 2) |
| node/test_rms_normalization_3d_axis_negative_2_epsilon_expanded/model.onnx | 23 | ✅ | OK (max ULP 2) |
| node/test_rms_normalization_3d_axis_negative_3_epsilon/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_rms_normalization_3d_axis_negative_3_epsilon_expanded/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_rms_normalization_4d_axis0/model.onnx | 23 | ✅ | OK (max ULP 1) |
| node/test_rms_normalization_4d_axis0_expanded/model.onnx | 23 | ✅ | OK (max ULP 1) |
| node/test_rms_normalization_4d_axis1/model.onnx | 23 | ✅ | OK (max ULP 2) |
| node/test_rms_normalization_4d_axis1_expanded/model.onnx | 23 | ✅ | OK (max ULP 2) |
| node/test_rms_normalization_4d_axis2/model.onnx | 23 | ✅ | OK (max ULP 2) |
| node/test_rms_normalization_4d_axis2_expanded/model.onnx | 23 | ✅ | OK (max ULP 2) |
| node/test_rms_normalization_4d_axis3/model.onnx | 23 | ✅ | OK (max ULP 1) |
| node/test_rms_normalization_4d_axis3_expanded/model.onnx | 23 | ✅ | OK (max ULP 1) |
| node/test_rms_normalization_4d_axis_negative_1/model.onnx | 23 | ✅ | OK (max ULP 2) |
| node/test_rms_normalization_4d_axis_negative_1_expanded/model.onnx | 23 | ✅ | OK (max ULP 2) |
| node/test_rms_normalization_4d_axis_negative_2/model.onnx | 23 | ✅ | OK (max ULP 1) |
| node/test_rms_normalization_4d_axis_negative_2_expanded/model.onnx | 23 | ✅ | OK (max ULP 1) |
| node/test_rms_normalization_4d_axis_negative_3/model.onnx | 23 | ✅ | OK (max ULP 2) |
| node/test_rms_normalization_4d_axis_negative_3_expanded/model.onnx | 23 | ✅ | OK (max ULP 2) |
| node/test_rms_normalization_4d_axis_negative_4/model.onnx | 23 | ✅ | OK (max ULP 2) |
| node/test_rms_normalization_4d_axis_negative_4_expanded/model.onnx | 23 | ✅ | OK (max ULP 2) |
| node/test_rms_normalization_default_axis/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_rms_normalization_default_axis_expanded/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_rnn_seq_length/model.onnx | 22 | ❌ | Unsupported op RNN |
| node/test_roialign_aligned_false/model.onnx | 22 | ❌ | Unsupported op RoiAlign |
| node/test_roialign_aligned_true/model.onnx | 22 | ❌ | Unsupported op RoiAlign |
| node/test_roialign_mode_max/model.onnx | 22 | ❌ | Unsupported op RoiAlign |
| node/test_rotary_embedding/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_rotary_embedding_3d_input/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_rotary_embedding_3d_input_expanded/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_rotary_embedding_expanded/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_rotary_embedding_interleaved/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_rotary_embedding_interleaved_expanded/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_rotary_embedding_no_position_ids/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_rotary_embedding_no_position_ids_expanded/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_rotary_embedding_no_position_ids_interleaved/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_rotary_embedding_no_position_ids_interleaved_expanded/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_rotary_embedding_no_position_ids_rotary_dim/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_rotary_embedding_no_position_ids_rotary_dim_expanded/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_rotary_embedding_with_interleaved_rotary_dim/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_rotary_embedding_with_interleaved_rotary_dim_expanded/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_rotary_embedding_with_rotary_dim/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_rotary_embedding_with_rotary_dim_expanded/model.onnx | 23 | ✅ | OK (max ULP 0) |
| node/test_round/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_scan9_sum/model.onnx | 9 | ✅ | OK (max ULP 0) |
| node/test_scan_sum/model.onnx | 8 | ✅ | OK (max ULP 0) |
| node/test_scatter_elements_with_axis/model.onnx | 18 | ❌ | Unsupported op ScatterElements |
| node/test_scatter_elements_with_duplicate_indices/model.onnx | 18 | ❌ | Unsupported op ScatterElements |
| node/test_scatter_elements_with_negative_indices/model.onnx | 18 | ❌ | Unsupported op ScatterElements |
| node/test_scatter_elements_with_reduction_max/model.onnx | 18 | ❌ | Unsupported op ScatterElements |
| node/test_scatter_elements_with_reduction_min/model.onnx | 18 | ❌ | Unsupported op ScatterElements |
| node/test_scatter_elements_without_axis/model.onnx | 18 | ❌ | Unsupported op ScatterElements |
| node/test_scatter_with_axis/model.onnx | 10 | ✅ | OK (max ULP 0) |
| node/test_scatter_without_axis/model.onnx | 10 | ✅ | OK (max ULP 0) |
| node/test_scatternd/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_scatternd_add/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_scatternd_max/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_scatternd_min/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_scatternd_multiply/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_sce_NCd1_mean_weight_negative_ii/model.onnx | 13 | ✅ | OK (max ULP 2) |
| node/test_sce_NCd1_mean_weight_negative_ii_expanded/model.onnx | 13 | ✅ | OK (max ULP 2) |
| node/test_sce_NCd1_mean_weight_negative_ii_log_prob/model.onnx | 13 | ✅ | OK (max ULP 2) |
| node/test_sce_NCd1_mean_weight_negative_ii_log_prob_expanded/model.onnx | 13 | ✅ | OK (max ULP 2) |
| node/test_sce_NCd1d2d3_none_no_weight_negative_ii/model.onnx | 13 | ✅ | OK (max ULP 2) |
| node/test_sce_NCd1d2d3_none_no_weight_negative_ii_expanded/model.onnx | 13 | ✅ | OK (max ULP 2) |
| node/test_sce_NCd1d2d3_none_no_weight_negative_ii_log_prob/model.onnx | 13 | ✅ | OK (max ULP 2) |
| node/test_sce_NCd1d2d3_none_no_weight_negative_ii_log_prob_expanded/model.onnx | 13 | ✅ | OK (max ULP 2) |
| node/test_sce_NCd1d2d3_sum_weight_high_ii/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_sce_NCd1d2d3_sum_weight_high_ii_expanded/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_sce_NCd1d2d3_sum_weight_high_ii_log_prob/model.onnx | 13 | ✅ | OK (max ULP 2) |
| node/test_sce_NCd1d2d3_sum_weight_high_ii_log_prob_expanded/model.onnx | 13 | ✅ | OK (max ULP 2) |
| node/test_sce_NCd1d2d3d4d5_mean_weight/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_sce_NCd1d2d3d4d5_mean_weight_expanded/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_sce_NCd1d2d3d4d5_mean_weight_log_prob/model.onnx | 13 | ✅ | OK (max ULP 2) |
| node/test_sce_NCd1d2d3d4d5_mean_weight_log_prob_expanded/model.onnx | 13 | ✅ | OK (max ULP 3) |
| node/test_sce_NCd1d2d3d4d5_none_no_weight/model.onnx | 13 | ✅ | OK (max ULP 3) |
| node/test_sce_NCd1d2d3d4d5_none_no_weight_expanded/model.onnx | 13 | ✅ | OK (max ULP 3) |
| node/test_sce_NCd1d2d3d4d5_none_no_weight_log_prob/model.onnx | 13 | ✅ | OK (max ULP 3) |
| node/test_sce_NCd1d2d3d4d5_none_no_weight_log_prob_expanded/model.onnx | 13 | ✅ | OK (max ULP 3) |
| node/test_sce_mean/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_sce_mean_3d/model.onnx | 13 | ✅ | OK (max ULP 2) |
| node/test_sce_mean_3d_expanded/model.onnx | 13 | ✅ | OK (max ULP 2) |
| node/test_sce_mean_3d_log_prob/model.onnx | 13 | ✅ | OK (max ULP 2) |
| node/test_sce_mean_3d_log_prob_expanded/model.onnx | 13 | ✅ | OK (max ULP 2) |
| node/test_sce_mean_expanded/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_sce_mean_log_prob/model.onnx | 13 | ✅ | OK (max ULP 2) |
| node/test_sce_mean_log_prob_expanded/model.onnx | 13 | ✅ | OK (max ULP 2) |
| node/test_sce_mean_no_weight_ii/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_sce_mean_no_weight_ii_3d/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_sce_mean_no_weight_ii_3d_expanded/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_sce_mean_no_weight_ii_3d_log_prob/model.onnx | 13 | ✅ | OK (max ULP 2) |
| node/test_sce_mean_no_weight_ii_3d_log_prob_expanded/model.onnx | 13 | ✅ | OK (max ULP 2) |
| node/test_sce_mean_no_weight_ii_4d/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_sce_mean_no_weight_ii_4d_expanded/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_sce_mean_no_weight_ii_4d_log_prob/model.onnx | 13 | ✅ | OK (max ULP 2) |
| node/test_sce_mean_no_weight_ii_4d_log_prob_expanded/model.onnx | 13 | ✅ | OK (max ULP 2) |
| node/test_sce_mean_no_weight_ii_expanded/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_sce_mean_no_weight_ii_log_prob/model.onnx | 13 | ✅ | OK (max ULP 2) |
| node/test_sce_mean_no_weight_ii_log_prob_expanded/model.onnx | 13 | ✅ | OK (max ULP 2) |
| node/test_sce_mean_weight/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_sce_mean_weight_expanded/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_sce_mean_weight_ii/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_sce_mean_weight_ii_3d/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_sce_mean_weight_ii_3d_expanded/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_sce_mean_weight_ii_3d_log_prob/model.onnx | 13 | ✅ | OK (max ULP 2) |
| node/test_sce_mean_weight_ii_3d_log_prob_expanded/model.onnx | 13 | ✅ | OK (max ULP 2) |
| node/test_sce_mean_weight_ii_4d/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_sce_mean_weight_ii_4d_expanded/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_sce_mean_weight_ii_4d_log_prob/model.onnx | 13 | ✅ | OK (max ULP 2) |
| node/test_sce_mean_weight_ii_4d_log_prob_expanded/model.onnx | 13 | ✅ | OK (max ULP 2) |
| node/test_sce_mean_weight_ii_expanded/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_sce_mean_weight_ii_log_prob/model.onnx | 13 | ✅ | OK (max ULP 2) |
| node/test_sce_mean_weight_ii_log_prob_expanded/model.onnx | 13 | ✅ | OK (max ULP 2) |
| node/test_sce_mean_weight_log_prob/model.onnx | 13 | ✅ | OK (max ULP 2) |
| node/test_sce_mean_weight_log_prob_expanded/model.onnx | 13 | ✅ | OK (max ULP 2) |
| node/test_sce_none/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_sce_none_expanded/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_sce_none_log_prob/model.onnx | 13 | ✅ | OK (max ULP 2) |
| node/test_sce_none_log_prob_expanded/model.onnx | 13 | ✅ | OK (max ULP 2) |
| node/test_sce_none_weights/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_sce_none_weights_expanded/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_sce_none_weights_log_prob/model.onnx | 13 | ✅ | OK (max ULP 2) |
| node/test_sce_none_weights_log_prob_expanded/model.onnx | 13 | ✅ | OK (max ULP 2) |
| node/test_sce_sum/model.onnx | 13 | ✅ | OK (max ULP 1) |
| node/test_sce_sum_expanded/model.onnx | 13 | ✅ | OK (max ULP 1) |
| node/test_sce_sum_log_prob/model.onnx | 13 | ✅ | OK (max ULP 2) |
| node/test_sce_sum_log_prob_expanded/model.onnx | 13 | ✅ | OK (max ULP 2) |
| node/test_selu/model.onnx | 22 | ❌ | Selu only supports alpha=1.6732632423543772 |
| node/test_selu_default/model.onnx | 22 | ✅ | OK (max ULP 5) |
| node/test_selu_default_expanded_ver18/model.onnx | 18 | ✅ | OK (max ULP 43) |
| node/test_selu_example/model.onnx | 22 | ❌ | Selu only supports alpha=1.6732632423543772 |
| node/test_selu_example_expanded_ver18/model.onnx | 18 | ✅ | OK (max ULP 2) |
| node/test_selu_expanded_ver18/model.onnx | 18 | ✅ | OK (max ULP 24) |
| node/test_sequence_insert_at_back/model.onnx | 11 | ✅ | OK (non-tensor outputs matched; max_abs_diff=8972011810649449542, max_ulp=0) |
| node/test_sequence_insert_at_front/model.onnx | 11 | ✅ | OK (non-tensor outputs matched; max_abs_diff=8247628798212575849, max_ulp=0) |
| node/test_sequence_map_add_1_sequence_1_tensor/model.onnx | 17 | ❌ | Unsupported op SequenceMap |
| node/test_sequence_map_add_1_sequence_1_tensor_expanded/model.onnx | 17 | ❌ | Unsupported op Loop |
| node/test_sequence_map_add_2_sequences/model.onnx | 17 | ❌ | Unsupported op SequenceMap |
| node/test_sequence_map_add_2_sequences_expanded/model.onnx | 17 | ❌ | Unsupported op Loop |
| node/test_sequence_map_extract_shapes/model.onnx | 17 | ❌ | Unsupported op SequenceMap |
| node/test_sequence_map_extract_shapes_expanded/model.onnx | 17 | ❌ | Unsupported op Loop |
| node/test_sequence_map_identity_1_sequence/model.onnx | 17 | ❌ | Unsupported op SequenceMap |
| node/test_sequence_map_identity_1_sequence_1_tensor/model.onnx | 17 | ❌ | Unsupported op SequenceMap |
| node/test_sequence_map_identity_1_sequence_1_tensor_expanded/model.onnx | 17 | ❌ | Unsupported op Loop |
| node/test_sequence_map_identity_1_sequence_expanded/model.onnx | 17 | ❌ | Unsupported op Loop |
| node/test_sequence_map_identity_2_sequences/model.onnx | 17 | ❌ | Unsupported op SequenceMap |
| node/test_sequence_map_identity_2_sequences_expanded/model.onnx | 17 | ❌ | Unsupported op Loop |
| node/test_shape/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_shape_clip_end/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_shape_clip_start/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_shape_end_1/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_shape_end_negative_1/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_shape_example/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_shape_start_1/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_shape_start_1_end_2/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_shape_start_1_end_negative_1/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_shape_start_greater_than_end/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_shape_start_negative_1/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_shrink_hard/model.onnx | 9 | ✅ | OK (max ULP 0) |
| node/test_shrink_hard_expanded_ver18/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_shrink_soft/model.onnx | 9 | ✅ | OK (max ULP 0) |
| node/test_shrink_soft_expanded_ver18/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_sigmoid/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_sigmoid_example/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_sign/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_simple_rnn_batchwise/model.onnx | 22 | ❌ | Unsupported op RNN |
| node/test_simple_rnn_defaults/model.onnx | 22 | ❌ | Unsupported op RNN |
| node/test_simple_rnn_with_initial_bias/model.onnx | 22 | ❌ | Unsupported op RNN |
| node/test_sin/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_sin_example/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_sinh/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_sinh_example/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_size/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_size_example/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_slice/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_slice_default_axes/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_slice_default_steps/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_slice_end_out_of_bounds/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_slice_neg/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_slice_neg_steps/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_slice_negative_axes/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_slice_start_out_of_bounds/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_softmax_axis_0/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_softmax_axis_0_expanded/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_softmax_axis_0_expanded_ver18/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_softmax_axis_1/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_softmax_axis_1_expanded/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_softmax_axis_1_expanded_ver18/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_softmax_axis_2/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_softmax_axis_2_expanded/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_softmax_axis_2_expanded_ver18/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_softmax_default_axis/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_softmax_default_axis_expanded/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_softmax_default_axis_expanded_ver18/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_softmax_example/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_softmax_example_expanded/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_softmax_example_expanded_ver18/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_softmax_large_number/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_softmax_large_number_expanded/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_softmax_large_number_expanded_ver18/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_softmax_negative_axis/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_softmax_negative_axis_expanded/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_softmax_negative_axis_expanded_ver18/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_softplus/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_softplus_example/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_softplus_example_expanded_ver18/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_softplus_expanded_ver18/model.onnx | 18 | ✅ | OK (max ULP 1) |
| node/test_softsign/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_softsign_example/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_softsign_example_expanded_ver18/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_softsign_expanded_ver18/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_spacetodepth/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_spacetodepth_example/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_split_1d_uneven_split_opset18/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_split_2d_uneven_split_opset18/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_split_equal_parts_1d_opset13/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_split_equal_parts_1d_opset18/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_split_equal_parts_2d/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_split_equal_parts_2d_opset13/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_split_equal_parts_default_axis_opset13/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_split_equal_parts_default_axis_opset18/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_split_to_sequence_1/model.onnx | 24 | ✅ | OK (non-tensor outputs matched; max_abs_diff=0, max_ulp=20971520) |
| node/test_split_to_sequence_2/model.onnx | 24 | ✅ | OK (non-tensor outputs matched; max_abs_diff=0, max_ulp=0) |
| node/test_split_to_sequence_nokeepdims/model.onnx | 24 | ✅ | OK (non-tensor outputs matched; max_abs_diff=0, max_ulp=0) |
| node/test_split_variable_parts_1d_opset13/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_split_variable_parts_1d_opset18/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_split_variable_parts_2d_opset13/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_split_variable_parts_2d_opset18/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_split_variable_parts_default_axis_opset13/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_split_variable_parts_default_axis_opset18/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_split_zero_size_splits_opset13/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_split_zero_size_splits_opset18/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_sqrt/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_sqrt_example/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_squeeze/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_squeeze_negative_axes/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_stft/model.onnx | 17 | ❌ | Unsupported op STFT |
| node/test_stft_with_window/model.onnx | 17 | ❌ | Unsupported op STFT |
| node/test_string_concat/model.onnx | 20 | ❌ | Unsupported op StringConcat |
| node/test_string_concat_broadcasting/model.onnx | 20 | ❌ | Unsupported op StringConcat |
| node/test_string_concat_empty_string/model.onnx | 20 | ❌ | Unsupported op StringConcat |
| node/test_string_concat_utf8/model.onnx | 20 | ❌ | Unsupported op StringConcat |
| node/test_string_concat_zero_dimensional/model.onnx | 20 | ❌ | Unsupported op StringConcat |
| node/test_string_split_basic/model.onnx | 20 | ❌ | Unsupported op StringSplit |
| node/test_string_split_consecutive_delimiters/model.onnx | 20 | ❌ | Unsupported op StringSplit |
| node/test_string_split_empty_string_delimiter/model.onnx | 20 | ❌ | Unsupported op StringSplit |
| node/test_string_split_empty_tensor/model.onnx | 20 | ❌ | Unsupported op StringSplit |
| node/test_string_split_maxsplit/model.onnx | 20 | ❌ | Unsupported op StringSplit |
| node/test_string_split_no_delimiter/model.onnx | 20 | ❌ | Unsupported op StringSplit |
| node/test_strnormalizer_export_monday_casesensintive_lower/model.onnx | 10 | ✅ | OK (max ULP 0) |
| node/test_strnormalizer_export_monday_casesensintive_nochangecase/model.onnx | 10 | ✅ | OK (max ULP 0) |
| node/test_strnormalizer_export_monday_casesensintive_upper/model.onnx | 10 | ✅ | OK (max ULP 0) |
| node/test_strnormalizer_export_monday_empty_output/model.onnx | 10 | ✅ | OK (max ULP 0) |
| node/test_strnormalizer_export_monday_insensintive_upper_twodim/model.onnx | 10 | ✅ | OK (max ULP 0) |
| node/test_strnormalizer_nostopwords_nochangecase/model.onnx | 10 | ✅ | OK (max ULP 0) |
| node/test_sub/model.onnx | 14 | ✅ | OK (max ULP 0) |
| node/test_sub_bcast/model.onnx | 14 | ✅ | OK (max ULP 0) |
| node/test_sub_example/model.onnx | 14 | ✅ | OK (max ULP 0) |
| node/test_sub_int16/model.onnx | 14 | ✅ | OK (max ULP 0) |
| node/test_sub_int8/model.onnx | 14 | ✅ | OK (max ULP 0) |
| node/test_sub_uint16/model.onnx | 14 | ✅ | OK (max ULP 0) |
| node/test_sub_uint32/model.onnx | 14 | ✅ | OK (max ULP 0) |
| node/test_sub_uint64/model.onnx | 14 | ✅ | OK (max ULP 0) |
| node/test_sub_uint8/model.onnx | 14 | ✅ | OK (max ULP 0) |
| node/test_sum_example/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_sum_one_input/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_sum_two_inputs/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_swish/model.onnx | 24 | ✅ | OK (max ULP 1) |
| node/test_swish_expanded/model.onnx | 24 | ✅ | OK (max ULP 0) |
| node/test_tan/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_tan_example/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_tanh/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_tanh_example/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_tensorscatter/model.onnx | 24 | ✅ | OK (max ULP 0) |
| node/test_tensorscatter_3d/model.onnx | 24 | ✅ | OK (max ULP 0) |
| node/test_tensorscatter_circular/model.onnx | 24 | ✅ | OK (max ULP 0) |
| node/test_tfidfvectorizer_tf_batch_onlybigrams_skip0/model.onnx | 9 | ✅ | OK (max ULP 0) |
| node/test_tfidfvectorizer_tf_batch_onlybigrams_skip5/model.onnx | 9 | ✅ | OK (max ULP 0) |
| node/test_tfidfvectorizer_tf_batch_uniandbigrams_skip5/model.onnx | 9 | ✅ | OK (max ULP 0) |
| node/test_tfidfvectorizer_tf_only_bigrams_skip0/model.onnx | 9 | ✅ | OK (max ULP 0) |
| node/test_tfidfvectorizer_tf_onlybigrams_levelempty/model.onnx | 9 | ✅ | OK (max ULP 0) |
| node/test_tfidfvectorizer_tf_onlybigrams_skip5/model.onnx | 9 | ✅ | OK (max ULP 0) |
| node/test_tfidfvectorizer_tf_uniandbigrams_skip5/model.onnx | 9 | ✅ | OK (max ULP 0) |
| node/test_thresholdedrelu/model.onnx | 22 | ❌ | ThresholdedRelu only supports alpha=1.0 |
| node/test_thresholdedrelu_default/model.onnx | 22 | ✅ | OK (max ULP 0) |
| node/test_thresholdedrelu_default_expanded_ver18/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_thresholdedrelu_example/model.onnx | 22 | ❌ | ThresholdedRelu only supports alpha=1.0 |
| node/test_thresholdedrelu_example_expanded_ver18/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_thresholdedrelu_expanded_ver18/model.onnx | 18 | ✅ | OK (max ULP 0) |
| node/test_tile/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_tile_precomputed/model.onnx | 13 | ✅ | OK (max ULP 0) |
| node/test_top_k/model.onnx | 24 | ✅ | OK (max ULP 0) |
| node/test_top_k_negative_axis/model.onnx | 24 | ✅ | OK (max ULP 0) |
| node/test_top_k_same_values/model.onnx | 24 | ✅ | OK (max ULP 0) |
| node/test_top_k_same_values_2d/model.onnx | 24 | ✅ | OK (max ULP 0) |
| node/test_top_k_same_values_largest/model.onnx | 24 | ✅ | OK (max ULP 0) |
| node/test_top_k_smallest/model.onnx | 24 | ✅ | OK (max ULP 0) |
| node/test_top_k_uint64/model.onnx | 24 | ✅ | OK (max ULP 0) |
| node/test_training_dropout/model.onnx | 22 | ❌ | Dropout supports only the data input and 1 or 2 outputs |
| node/test_training_dropout_default/model.onnx | 22 | ❌ | Dropout supports only the data input and 1 or 2 outputs |
| node/test_training_dropout_default_mask/model.onnx | 22 | ❌ | Dropout supports only the data input and 1 or 2 outputs |
| node/test_training_dropout_mask/model.onnx | 22 | ❌ | Dropout supports only the data input and 1 or 2 outputs |
| node/test_training_dropout_zero_ratio/model.onnx | 22 | ❌ | Dropout supports only the data input and 1 or 2 outputs |
| node/test_training_dropout_zero_ratio_mask/model.onnx | 22 | ❌ | Dropout supports only the data input and 1 or 2 outputs |
| node/test_transpose_all_permutations_0/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_transpose_all_permutations_1/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_transpose_all_permutations_2/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_transpose_all_permutations_3/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_transpose_all_permutations_4/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_transpose_all_permutations_5/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_transpose_default/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_tril/model.onnx | 14 | ✅ | OK (max ULP 0) |
| node/test_tril_neg/model.onnx | 14 | ✅ | OK (max ULP 0) |
| node/test_tril_one_row_neg/model.onnx | 14 | ✅ | OK (max ULP 0) |
| node/test_tril_out_neg/model.onnx | 14 | ✅ | OK (max ULP 0) |
| node/test_tril_out_pos/model.onnx | 14 | ✅ | OK (max ULP 0) |
| node/test_tril_pos/model.onnx | 14 | ✅ | OK (max ULP 0) |
| node/test_tril_square/model.onnx | 14 | ✅ | OK (max ULP 0) |
| node/test_tril_square_neg/model.onnx | 14 | ✅ | OK (max ULP 0) |
| node/test_tril_zero/model.onnx | 14 | ✅ | OK (max ULP 0) |
| node/test_triu/model.onnx | 14 | ✅ | OK (max ULP 0) |
| node/test_triu_neg/model.onnx | 14 | ✅ | OK (max ULP 0) |
| node/test_triu_one_row/model.onnx | 14 | ✅ | OK (max ULP 0) |
| node/test_triu_out_neg_out/model.onnx | 14 | ✅ | OK (max ULP 0) |
| node/test_triu_out_pos/model.onnx | 14 | ✅ | OK (max ULP 0) |
| node/test_triu_pos/model.onnx | 14 | ✅ | OK (max ULP 0) |
| node/test_triu_square/model.onnx | 14 | ✅ | OK (max ULP 0) |
| node/test_triu_square_neg/model.onnx | 14 | ✅ | OK (max ULP 0) |
| node/test_triu_zero/model.onnx | 14 | ✅ | OK (max ULP 0) |
| node/test_unique_length_1/model.onnx | 11 | ✅ | OK (max ULP 0) |
| node/test_unique_not_sorted_without_axis/model.onnx | 11 | ✅ | OK (max ULP 0) |
| node/test_unique_sorted_with_axis/model.onnx | 11 | ✅ | OK (max ULP 0) |
| node/test_unique_sorted_with_axis_3d/model.onnx | 11 | ✅ | OK (max ULP 0) |
| node/test_unique_sorted_with_negative_axis/model.onnx | 11 | ✅ | OK (max ULP 0) |
| node/test_unique_sorted_without_axis/model.onnx | 11 | ✅ | OK (max ULP 0) |
| node/test_unsqueeze_axis_0/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_unsqueeze_axis_1/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_unsqueeze_axis_2/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_unsqueeze_negative_axes/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_unsqueeze_three_axes/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_unsqueeze_two_axes/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_unsqueeze_unsorted_axes/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_upsample_nearest/model.onnx | 9 | ✅ | OK (max ULP 0) |
| node/test_where_example/model.onnx | 16 | ✅ | OK (max ULP 0) |
| node/test_where_long_example/model.onnx | 16 | ✅ | OK (max ULP 0) |
| node/test_wrap_pad/model.onnx | 25 | ✅ | OK (max ULP 0) |
| node/test_xor2d/model.onnx | 7 | ✅ | OK (max ULP 0) |
| node/test_xor3d/model.onnx | 7 | ✅ | OK (max ULP 0) |
| node/test_xor4d/model.onnx | 7 | ✅ | OK (max ULP 0) |
| node/test_xor_bcast3v1d/model.onnx | 7 | ✅ | OK (max ULP 0) |
| node/test_xor_bcast3v2d/model.onnx | 7 | ✅ | OK (max ULP 0) |
| node/test_xor_bcast4v2d/model.onnx | 7 | ✅ | OK (max ULP 0) |
| node/test_xor_bcast4v3d/model.onnx | 7 | ✅ | OK (max ULP 0) |
| node/test_xor_bcast4v4d/model.onnx | 7 | ✅ | OK (max ULP 0) |
| pytorch-converted/test_AvgPool1d/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-converted/test_AvgPool1d_stride/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-converted/test_AvgPool2d/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-converted/test_AvgPool2d_stride/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-converted/test_AvgPool3d/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-converted/test_AvgPool3d_stride/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-converted/test_AvgPool3d_stride1_pad0_gpu_input/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-converted/test_BatchNorm1d_3d_input_eval/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-converted/test_BatchNorm2d_eval/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-converted/test_BatchNorm2d_momentum_eval/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-converted/test_BatchNorm3d_eval/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-converted/test_BatchNorm3d_momentum_eval/model.onnx | 6 | ✅ | OK (max ULP 2) |
| pytorch-converted/test_ConstantPad2d/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-converted/test_Conv1d/model.onnx | 6 | ✅ | OK (max ULP 4) |
| pytorch-converted/test_Conv1d_dilated/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-converted/test_Conv1d_groups/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-converted/test_Conv1d_pad1/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-converted/test_Conv1d_pad1size1/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-converted/test_Conv1d_pad2/model.onnx | 6 | ✅ | OK (max ULP 3) |
| pytorch-converted/test_Conv1d_pad2size1/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-converted/test_Conv1d_stride/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-converted/test_Conv2d/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-converted/test_Conv2d_depthwise/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-converted/test_Conv2d_depthwise_padded/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-converted/test_Conv2d_depthwise_strided/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-converted/test_Conv2d_depthwise_with_multiplier/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-converted/test_Conv2d_dilated/model.onnx | 6 | ✅ | OK (max ULP 1) |
| pytorch-converted/test_Conv2d_groups/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-converted/test_Conv2d_groups_thnn/model.onnx | 6 | ✅ | OK (max ULP 3) |
| pytorch-converted/test_Conv2d_no_bias/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-converted/test_Conv2d_padding/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-converted/test_Conv2d_strided/model.onnx | 6 | ✅ | OK (max ULP 9) |
| pytorch-converted/test_Conv3d/model.onnx | 6 | ✅ | OK (max ULP 4) |
| pytorch-converted/test_Conv3d_dilated/model.onnx | 6 | ✅ | OK (max ULP 3) |
| pytorch-converted/test_Conv3d_dilated_strided/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-converted/test_Conv3d_groups/model.onnx | 6 | ✅ | OK (max ULP 9) |
| pytorch-converted/test_Conv3d_no_bias/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-converted/test_Conv3d_stride/model.onnx | 6 | ✅ | OK (max ULP 5) |
| pytorch-converted/test_Conv3d_stride_padding/model.onnx | 6 | ✅ | OK (max ULP 7) |
| pytorch-converted/test_ConvTranspose2d/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-converted/test_ConvTranspose2d_no_bias/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-converted/test_ELU/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-converted/test_Embedding/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-converted/test_Embedding_sparse/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-converted/test_GLU/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-converted/test_GLU_dim/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-converted/test_LeakyReLU/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-converted/test_LeakyReLU_with_negval/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-converted/test_Linear/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-converted/test_Linear_no_bias/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-converted/test_LogSoftmax/model.onnx | 6 | ✅ | OK (max ULP 2) |
| pytorch-converted/test_MaxPool1d/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-converted/test_MaxPool1d_stride/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-converted/test_MaxPool1d_stride_padding_dilation/model.onnx | 12 | ✅ | OK (max ULP 0) |
| pytorch-converted/test_MaxPool2d/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-converted/test_MaxPool2d_stride_padding_dilation/model.onnx | 12 | ✅ | OK (max ULP 0) |
| pytorch-converted/test_MaxPool3d/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-converted/test_MaxPool3d_stride/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-converted/test_MaxPool3d_stride_padding/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-converted/test_PReLU_1d/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-converted/test_PReLU_1d_multiparam/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-converted/test_PReLU_2d/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-converted/test_PReLU_2d_multiparam/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-converted/test_PReLU_3d/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-converted/test_PReLU_3d_multiparam/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-converted/test_PixelShuffle/model.onnx | 9 | ✅ | OK (max ULP 0) |
| pytorch-converted/test_PoissonNLLLLoss_no_reduce/model.onnx | 6 | ✅ | OK (max ULP 1) |
| pytorch-converted/test_ReLU/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-converted/test_ReflectionPad2d/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-converted/test_ReplicationPad2d/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-converted/test_SELU/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-converted/test_Sigmoid/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-converted/test_Softmax/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-converted/test_Softmin/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-converted/test_Softplus/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-converted/test_Softsign/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-converted/test_Tanh/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-converted/test_ZeroPad2d/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-converted/test_log_softmax_dim3/model.onnx | 6 | ✅ | OK (max ULP 1) |
| pytorch-converted/test_log_softmax_lastdim/model.onnx | 6 | ✅ | OK (max ULP 1) |
| pytorch-converted/test_softmax_functional_dim3/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-converted/test_softmax_lastdim/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-operator/test_operator_add_broadcast/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-operator/test_operator_add_size1_broadcast/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-operator/test_operator_add_size1_right_broadcast/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-operator/test_operator_add_size1_singleton_broadcast/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-operator/test_operator_addconstant/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-operator/test_operator_addmm/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-operator/test_operator_basic/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-operator/test_operator_chunk/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-operator/test_operator_clip/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-operator/test_operator_concat2/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-operator/test_operator_conv/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-operator/test_operator_convtranspose/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-operator/test_operator_exp/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-operator/test_operator_flatten/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-operator/test_operator_index/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-operator/test_operator_max/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-operator/test_operator_maxpool/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-operator/test_operator_min/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-operator/test_operator_mm/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-operator/test_operator_non_float_params/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-operator/test_operator_pad/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-operator/test_operator_params/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-operator/test_operator_permute2/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-operator/test_operator_pow/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-operator/test_operator_reduced_mean/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-operator/test_operator_reduced_mean_keepdim/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-operator/test_operator_reduced_sum/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-operator/test_operator_reduced_sum_keepdim/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-operator/test_operator_repeat/model.onnx | 9 | ✅ | OK (max ULP 0) |
| pytorch-operator/test_operator_repeat_dim_overflow/model.onnx | 9 | ✅ | OK (max ULP 0) |
| pytorch-operator/test_operator_selu/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-operator/test_operator_sqrt/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-operator/test_operator_symbolic_override/model.onnx | 6 | ✅ | OK (max ULP 9) |
| pytorch-operator/test_operator_symbolic_override_nested/model.onnx | 6 | ✅ | OK (max ULP 0) |
| pytorch-operator/test_operator_view/model.onnx | 6 | ✅ | OK (max ULP 0) |
| simple/test_expand_shape_model1/model.onnx | 9 | ✅ | OK (max ULP 0) |
| simple/test_expand_shape_model2/model.onnx | 9 | ✅ | OK (max ULP 0) |
| simple/test_expand_shape_model3/model.onnx | 9 | ✅ | OK (max ULP 0) |
| simple/test_expand_shape_model4/model.onnx | 9 | ✅ | OK (max ULP 0) |
| simple/test_gradient_of_add/model.onnx | 12 | ✅ | OK (max ULP 0) |
| simple/test_gradient_of_add_and_mul/model.onnx | 12 | ✅ | OK (max ULP 0) |
| simple/test_sequence_model1/model.onnx | 12 | ✅ | OK (max ULP 0) |
| simple/test_sequence_model2/model.onnx | 12 | ✅ | OK (max ULP 0) |
| simple/test_sequence_model3/model.onnx | 12 | ✅ | OK (max ULP 0) |
| simple/test_sequence_model4/model.onnx | 12 | ✅ | OK (max ULP 0) |
| simple/test_sequence_model5/model.onnx | 12 | ✅ | OK (max ULP 0) |
| simple/test_sequence_model6/model.onnx | 12 | ✅ | OK (max ULP 0) |
| simple/test_sequence_model7/model.onnx | 12 | ✅ | OK (max ULP 0) |
| simple/test_sequence_model8/model.onnx | 12 | ✅ | OK (max ULP 0) |
| simple/test_shrink/model.onnx | 10 | ✅ | OK (max ULP 0) |
| simple/test_sign_model/model.onnx | 9 | ✅ | OK (max ULP 0) |
| simple/test_single_relu_model/model.onnx | 9 | ✅ | OK (max ULP 0) |
| simple/test_strnorm_model_monday_casesensintive_lower/model.onnx | 10 | ✅ | OK (max ULP 0) |
| simple/test_strnorm_model_monday_casesensintive_nochangecase/model.onnx | 10 | ✅ | OK (max ULP 0) |
| simple/test_strnorm_model_monday_casesensintive_upper/model.onnx | 10 | ✅ | OK (max ULP 0) |
| simple/test_strnorm_model_monday_empty_output/model.onnx | 10 | ✅ | OK (max ULP 0) |
| simple/test_strnorm_model_monday_insensintive_upper_twodim/model.onnx | 10 | ✅ | OK (max ULP 0) |
| simple/test_strnorm_model_nostopwords_nochangecase/model.onnx | 10 | ✅ | OK (max ULP 0) |

## ONNX2C test coverage

Test directory: `onnx2c-org/test`

Coverage 118 / 125 ONNX files (94.4%).

| File | Opset | Supported | Error |
| --- | --- | --- | --- |
| benchmarks/benchmark_conv_fits_128k/model.onnx | 12 | ✅ | OK (max ULP 16) |
| benchmarks/benchmark_conv_yolov6n_biggestconv/model.onnx | 12 | ✅ | OK (max ULP 18) |
| benchmarks/benchmark_conv_yolov6n_inputlayer/model.onnx | 12 | ✅ | OK (max ULP 4) |
| benchmarks/benchmark_conv_yolov6n_lastconv/model.onnx | 12 | ✅ | OK (max ULP 8) |
| local_ops/test_gather_basic/model.onnx | 22 | ✅ | OK (max ULP 0) |
| local_ops/test_gather_output_scalar/model.onnx | 22 | ✅ | OK (max ULP 0) |
| local_ops/test_gather_scalar_axis0/model.onnx | 22 | ✅ | OK (max ULP 0) |
| local_ops/test_gather_scalar_axis1/model.onnx | 22 | ✅ | OK (max ULP 0) |
| local_ops/test_gemm_C1/model.onnx | 12 | ✅ | OK (max ULP 0) |
| local_ops/test_gemm_C1_transA/model.onnx | 12 | ✅ | OK (max ULP 0) |
| local_ops/test_gemm_C1_transB/model.onnx | 12 | ✅ | OK (max ULP 0) |
| local_ops/test_gemm_C1x1/model.onnx | 12 | ✅ | OK (max ULP 0) |
| local_ops/test_gemm_C1x1_transA/model.onnx | 12 | ✅ | OK (max ULP 0) |
| local_ops/test_gemm_C1xN/model.onnx | 12 | ✅ | OK (max ULP 0) |
| local_ops/test_gemm_C1xN_transA/model.onnx | 12 | ✅ | OK (max ULP 1) |
| local_ops/test_gemm_C1xN_transA_transB/model.onnx | 12 | ✅ | OK (max ULP 1) |
| local_ops/test_gemm_CM_transA/model.onnx | 12 | ❌ | onnxruntime failed to run onnx2c-org/test/local_ops/test_gemm_CM_transA/model.onnx: [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Non-zero status code returned while running Gemm node. Name:'sclbl-onnx-node1' Status Message: Gemm: Invalid bias shape for broadcast |
| local_ops/test_gemm_CMx1/model.onnx | 12 | ✅ | OK (max ULP 1) |
| local_ops/test_gemm_CMx1_transA/model.onnx | 12 | ✅ | OK (max ULP 0) |
| local_ops/test_gemm_CMx1_transA_transB/model.onnx | 12 | ✅ | OK (max ULP 0) |
| local_ops/test_gemm_CMxN/model.onnx | 12 | ✅ | OK (max ULP 1) |
| local_ops/test_gemm_CMxN_transA/model.onnx | 12 | ✅ | OK (max ULP 0) |
| local_ops/test_gemm_CMxN_transA_transB/model.onnx | 12 | ✅ | OK (max ULP 0) |
| local_ops/test_gemm_CMxN_transB/model.onnx | 12 | ✅ | OK (max ULP 1) |
| local_ops/test_gemm_CN/model.onnx | 12 | ✅ | OK (max ULP 0) |
| local_ops/test_gemm_CN_transA/model.onnx | 12 | ✅ | OK (max ULP 1) |
| local_ops/test_gemm_CN_transA_transB/model.onnx | 12 | ✅ | OK (max ULP 1) |
| local_ops/test_gemm_CN_transB/model.onnx | 12 | ✅ | OK (max ULP 0) |
| local_ops/test_lstm_activations/model.onnx | 11 | ✅ | OK (max ULP 0) |
| local_ops/test_lstm_all_outputs/model.onnx | 11 | ✅ | OK (max ULP 0) |
| local_ops/test_lstm_bidirectional/model.onnx | 11 | ✅ | OK (max ULP 0) |
| local_ops/test_lstm_clip/model.onnx | 11 | ✅ | OK (max ULP 0) |
| local_ops/test_lstm_intermediate_h/model.onnx | 11 | ✅ | OK (max ULP 0) |
| local_ops/test_lstm_missing_inputs/model.onnx | 12 | ✅ | OK (max ULP 0) |
| local_ops/test_lstm_reverse/model.onnx | 11 | ✅ | OK (max ULP 0) |
| local_ops/test_lstm_seq_length/model.onnx | 11 | ✅ | OK (max ULP 0) |
| local_ops/test_lstm_simple/model.onnx | 12 | ✅ | OK (max ULP 0) |
| local_ops/test_lstm_with_initial_state/model.onnx | 12 | ✅ | OK (max ULP 5) |
| local_ops/test_lstm_y_c/model.onnx | 11 | ✅ | OK (max ULP 0) |
| local_ops/test_matmul_1x1x3x4_2x3x4x5/model.onnx | 22 | ✅ | OK (max ULP 0) |
| local_ops/test_matmul_1x3x4_2x3x4x5/model.onnx | 22 | ✅ | OK (max ULP 0) |
| local_ops/test_matmul_1x3x4_3x4x5/model.onnx | 22 | ✅ | OK (max ULP 0) |
| local_ops/test_matmul_2x1x3x4_2x3x4x5/model.onnx | 22 | ✅ | OK (max ULP 0) |
| local_ops/test_matmul_2x3_3x4/model.onnx | 22 | ✅ | OK (max ULP 0) |
| local_ops/test_matmul_2x3x3x4_1x4x5/model.onnx | 22 | ✅ | OK (max ULP 0) |
| local_ops/test_matmul_2x3x4_4/model.onnx | 22 | ✅ | OK (max ULP 0) |
| local_ops/test_matmul_2x3x4_4x5/model.onnx | 22 | ✅ | OK (max ULP 0) |
| local_ops/test_matmul_2x3x4x5_5/model.onnx | 22 | ✅ | OK (max ULP 0) |
| local_ops/test_matmul_3_2x3x4/model.onnx | 22 | ✅ | OK (max ULP 0) |
| local_ops/test_matmul_3_3/model.onnx | 22 | ✅ | OK (max ULP 0) |
| local_ops/test_matmul_3_3x4/model.onnx | 22 | ✅ | OK (max ULP 0) |
| local_ops/test_matmul_3x4_2x4x5/model.onnx | 22 | ✅ | OK (max ULP 0) |
| local_ops/test_matmul_3x4_4/model.onnx | 22 | ✅ | OK (max ULP 0) |
| local_ops/test_matmul_4x5x2x3_4x5x3x4/model.onnx | 22 | ✅ | OK (max ULP 0) |
| local_ops/test_matmul_5x2x3_5x3x4/model.onnx | 22 | ✅ | OK (max ULP 0) |
| local_ops/test_matmul_precision/model.onnx | 12 | ✅ | OK (max ULP 0) |
| local_ops/test_maxpool_stride_1/model.onnx | 12 | ✅ | OK (max ULP 0) |
| local_ops/test_maxpool_stride_2/model.onnx | 12 | ✅ | OK (max ULP 0) |
| local_ops/test_nodes_out_of_order/model.onnx | 12 | ✅ | OK (max ULP 0) |
| local_ops/test_pad_constant_default/model.onnx | 12 | ✅ | OK (max ULP 0) |
| local_ops/test_pad_constant_input/model.onnx | 12 | ✅ | OK (max ULP 0) |
| local_ops/test_pad_edge/model.onnx | 12 | ✅ | OK (max ULP 0) |
| local_ops/test_pad_edge_allaxes/model.onnx | 12 | ✅ | OK (max ULP 0) |
| local_ops/test_pad_reflect_allaxes/model.onnx | 12 | ✅ | OK (max ULP 0) |
| local_ops/test_pad_reflect_nopadding/model.onnx | 12 | ✅ | OK (max ULP 0) |
| local_ops/test_qlinearadd_int8/model.onnx |  | ✅ | OK (max ULP 0) |
| local_ops/test_qlinearadd_uint8/model.onnx |  | ✅ | OK (max ULP 0) |
| local_ops/test_qlinearmul_int8/model.onnx |  | ✅ | OK (max ULP 0) |
| local_ops/test_qlinearmul_uint8/model.onnx |  | ✅ | OK (max ULP 0) |
| local_ops/test_resize_downsample_sizes_linear_1D/model.onnx | 12 | ✅ | OK (max ULP 0) |
| local_ops/test_resize_downsample_sizes_linear_1D_align/model.onnx | 12 | ✅ | OK (max ULP 1) |
| local_ops/test_scalar_input_to_node/model.onnx | 13 | ✅ | OK (max ULP 0) |
| local_ops/test_scatternd_indices_1x1x2/model.onnx | 12 | ✅ | OK (max ULP 0) |
| local_ops/test_scatternd_indices_1x2x2/model.onnx | 12 | ✅ | OK (max ULP 0) |
| local_ops/test_scatternd_indices_2x2x2/model.onnx | 12 | ✅ | OK (max ULP 0) |
| local_ops/test_scatternd_indices_3x2/model.onnx | 12 | ✅ | OK (max ULP 0) |
| local_ops/test_shape_const_out/model.onnx | 12 | ✅ | OK (max ULP 0) |
| local_ops/test_slice_end_INT64_MAX/model.onnx | 12 | ✅ | OK (max ULP 0) |
| mnist/model.onnx | 8 | ❌ | Out of tolerance (max ULP 160) |
| mnist/pytorch.onnx | 9 | ❌ | Out of tolerance (max ULP 2295973698) |
| old_onnx_backend/11/test_softmax_axis_0/model.onnx | 9 | ✅ | OK (max ULP 0) |
| old_onnx_backend/11/test_softmax_axis_1/model.onnx | 9 | ✅ | OK (max ULP 0) |
| old_onnx_backend/11/test_softmax_default_axis/model.onnx | 9 | ✅ | OK (max ULP 0) |
| old_onnx_backend/11/test_squeeze/model.onnx | 9 | ✅ | OK (max ULP 0) |
| old_onnx_backend/11/test_squeeze_negative_axes/model.onnx | 11 | ✅ | OK (max ULP 0) |
| old_onnx_backend/11/test_unsqueeze_axis_0/model.onnx | 11 | ✅ | OK (max ULP 0) |
| old_onnx_backend/11/test_unsqueeze_axis_1/model.onnx | 11 | ✅ | OK (max ULP 0) |
| old_onnx_backend/11/test_unsqueeze_axis_2/model.onnx | 11 | ✅ | OK (max ULP 0) |
| old_onnx_backend/11/test_unsqueeze_axis_3/model.onnx | 11 | ✅ | OK (max ULP 0) |
| old_onnx_backend/11/test_unsqueeze_negative_axes/model.onnx | 11 | ✅ | OK (max ULP 0) |
| old_onnx_backend/11/test_unsqueeze_three_axes/model.onnx | 11 | ✅ | OK (max ULP 0) |
| old_onnx_backend/11/test_unsqueeze_two_axes/model.onnx | 11 | ✅ | OK (max ULP 0) |
| old_onnx_backend/11/test_unsqueeze_unsorted_axes/model.onnx | 11 | ✅ | OK (max ULP 0) |
| old_onnx_backend/12/test_pow_types_float/model.onnx | 12 | ✅ | OK (max ULP 0) |
| old_onnx_backend/12/test_pow_types_int/model.onnx | 12 | ✅ | OK (max ULP 0) |
| old_onnx_backend/5/test_resize_downsample_sizes_nearest_tf_half_pixel_for_nn/model.onnx | 13 | ✅ | OK (max ULP 0) |
| old_onnx_backend/7/test_split_variable_parts_1d/model.onnx | 13 | ✅ | OK (max ULP 0) |
| old_onnx_backend/7/test_split_variable_parts_2d/model.onnx | 13 | ✅ | OK (max ULP 0) |
| old_onnx_backend/7/test_split_variable_parts_default_axis/model.onnx | 13 | ✅ | OK (max ULP 0) |
| old_onnx_backend/8/test_convtranspose_with_kernel/model.onnx | 8 | ✅ | OK (max ULP 0) |
| old_onnx_backend/9/test_slice/model.onnx | 9 | ✅ | OK (max ULP 0) |
| old_onnx_backend/9/test_slice_default_axes/model.onnx | 9 | ✅ | OK (max ULP 0) |
| old_onnx_backend/9/test_slice_end_out_of_bounds/model.onnx | 9 | ✅ | OK (max ULP 0) |
| old_onnx_backend/9/test_slice_neg/model.onnx | 9 | ✅ | OK (max ULP 0) |
| simple_networks/conv_2ch_3kernels_randombias.onnx | 11 | ✅ | OK (max ULP 1) |
| simple_networks/conv_2kernels.onnx | 11 | ✅ | OK (max ULP 0) |
| simple_networks/conv_2kernels_randombias.onnx | 11 | ✅ | OK (max ULP 2) |
| simple_networks/conv_3ch.onnx | 11 | ✅ | OK (max ULP 1) |
| simple_networks/conv_k2.onnx | 11 | ✅ | OK (max ULP 0) |
| simple_networks/conv_k2_maxpool_k2.onnx | 12 | ✅ | OK (max ULP 0) |
| simple_networks/conv_k2_s2.onnx | 11 | ✅ | OK (max ULP 0) |
| simple_networks/fp_bfloat16.onnx | 22 | ❌ | Cast input and output shapes must match |
| simple_networks/fp_float16.onnx | 22 | ❌ | Cast input and output shapes must match |
| simple_networks/lstm_k1_b1_r1.onnx | 11 | ✅ | OK (max ULP 0) |
| simple_networks/lstm_k1_b1_r1_relu.onnx | 11 | ✅ | OK (max ULP 0) |
| simple_networks/maxpool_k2.onnx | 12 | ✅ | OK (max ULP 0) |
| simple_networks/maxpool_k2_s2.onnx | 12 | ✅ | OK (max ULP 0) |
| simple_networks/random_uniform.onnx | 22 | ❌ | Unsupported op RandomUniform |
| simple_networks/scalar_abs.onnx | 13 | ✅ | OK (max ULP 0) |
| simple_networks/scalar_add.onnx | 13 | ✅ | OK (max ULP 0) |
| simple_networks/scalar_identity.onnx | 13 | ✅ | OK (max ULP 0) |
| simple_networks/treeensembleclassifier.onnx | 12 | ❌ | Unsupported op TreeEnsembleClassifier |
| tfl_helloworld/model.onnx | 9 | ✅ | OK (max ULP 0) |
| velardo/lesson14.onnx | 11 | ✅ | OK (max ULP 0) |
| velardo/lesson9.onnx | 9 | ✅ | OK (max ULP 0) |

## Local ONNX test coverage

Test directory: `tests/onnx`

Coverage 1 / 3 ONNX files (33.3%).

| File | Opset | Supported | Error |
| --- | --- | --- | --- |
| micro_kws_m_qdq.onnx | 15 | ❌ | DequantizeLinear zero_point shape must match scale shape |
| micro_kws_m_qoperator_avg_pool.onnx | 15 | ✅ | OK (max ULP 0) |
| micro_kws_m_qoperator_softmax.onnx | 15 | ❌ | Unsupported op QLinearSoftmax |
| micro_kws_m_qoperator_avg_pool.onnx | 15 | ❌ | Unsupported op QLinearAveragePool |
| micro_kws_m_qoperator_softmax.onnx | 15 | ❌ | Out of tolerance (max ULP 1058537472) |

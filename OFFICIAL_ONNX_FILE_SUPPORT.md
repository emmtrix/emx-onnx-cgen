# Official ONNX file support

Support 244 / 500 official ONNX files.

ONNX version: 1.20.1

See [`OFFICIAL_ONNX_FILE_SUPPORT_HISTOGRAM.md`](OFFICIAL_ONNX_FILE_SUPPORT_HISTOGRAM.md) for the error histogram.

| File | Supported | Error |
| --- | --- | --- |
| onnx-org/onnx/backend/test/data/light/light_bvlc_alexnet.onnx | ❌ | Testbench execution failed: exit code -11 (signal 11: SIGSEGV) |
| onnx-org/onnx/backend/test/data/light/light_densenet121.onnx | ❌ | Testbench execution failed: exit code -11 (signal 11: SIGSEGV) |
| onnx-org/onnx/backend/test/data/light/light_inception_v1.onnx | ❌ | Testbench execution failed: exit code -11 (signal 11: SIGSEGV) |
| onnx-org/onnx/backend/test/data/light/light_inception_v2.onnx | ❌ | Testbench execution failed: exit code -11 (signal 11: SIGSEGV) |
| onnx-org/onnx/backend/test/data/light/light_resnet50.onnx | ❌ | Testbench execution failed: exit code -11 (signal 11: SIGSEGV) |
| onnx-org/onnx/backend/test/data/light/light_shufflenet.onnx | ❌ | Testbench execution failed: exit code -11 (signal 11: SIGSEGV) |
| onnx-org/onnx/backend/test/data/light/light_squeezenet.onnx | ❌ | Testbench execution failed: exit code -11 (signal 11: SIGSEGV) |
| onnx-org/onnx/backend/test/data/light/light_vgg19.onnx | ❌ | Testbench execution failed: exit code -11 (signal 11: SIGSEGV) |
| onnx-org/onnx/backend/test/data/light/light_zfnet512.onnx | ❌ | Testbench execution failed: exit code -11 (signal 11: SIGSEGV) |
| onnx-org/onnx/backend/test/data/node/test_abs/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_acos/model.onnx | ✅ | OK (max ULP 3) |
| onnx-org/onnx/backend/test/data/node/test_acos_example/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_acosh/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_acosh_example/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_adagrad/model.onnx | ❌ | Unsupported op Adagrad |
| onnx-org/onnx/backend/test/data/node/test_adagrad_multiple/model.onnx | ❌ | Unsupported op Adagrad |
| onnx-org/onnx/backend/test/data/node/test_adam/model.onnx | ❌ | Unsupported op Adam |
| onnx-org/onnx/backend/test/data/node/test_adam_multiple/model.onnx | ❌ | Unsupported op Adam |
| onnx-org/onnx/backend/test/data/node/test_add/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_add_bcast/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_add_int16/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_add_int8/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_add_uint16/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_add_uint32/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_add_uint64/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_add_uint8/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_affine_grid_2d/model.onnx | ❌ | Unsupported op AffineGrid |
| onnx-org/onnx/backend/test/data/node/test_affine_grid_2d_align_corners/model.onnx | ❌ | Unsupported op AffineGrid |
| onnx-org/onnx/backend/test/data/node/test_affine_grid_2d_align_corners_expanded/model.onnx | ❌ | Unsupported op If |
| onnx-org/onnx/backend/test/data/node/test_affine_grid_2d_expanded/model.onnx | ❌ | Unsupported op If |
| onnx-org/onnx/backend/test/data/node/test_affine_grid_3d/model.onnx | ❌ | Unsupported op AffineGrid |
| onnx-org/onnx/backend/test/data/node/test_affine_grid_3d_align_corners/model.onnx | ❌ | Unsupported op AffineGrid |
| onnx-org/onnx/backend/test/data/node/test_affine_grid_3d_align_corners_expanded/model.onnx | ❌ | Unsupported op If |
| onnx-org/onnx/backend/test/data/node/test_affine_grid_3d_expanded/model.onnx | ❌ | Unsupported op If |
| onnx-org/onnx/backend/test/data/node/test_ai_onnx_ml_array_feature_extractor/model.onnx | ❌ | Unsupported op ArrayFeatureExtractor |
| onnx-org/onnx/backend/test/data/node/test_ai_onnx_ml_binarizer/model.onnx | ❌ | Unsupported op Binarizer |
| onnx-org/onnx/backend/test/data/node/test_ai_onnx_ml_label_encoder_string_int/model.onnx | ❌ | Unsupported elem_type 8 (STRING) for tensor 'X'. |
| onnx-org/onnx/backend/test/data/node/test_ai_onnx_ml_label_encoder_string_int_no_default/model.onnx | ❌ | Unsupported elem_type 8 (STRING) for tensor 'X'. |
| onnx-org/onnx/backend/test/data/node/test_ai_onnx_ml_label_encoder_tensor_mapping/model.onnx | ❌ | Unsupported elem_type 8 (STRING) for tensor 'X'. |
| onnx-org/onnx/backend/test/data/node/test_ai_onnx_ml_label_encoder_tensor_value_only_mapping/model.onnx | ❌ | Unsupported elem_type 8 (STRING) for tensor 'X'. |
| onnx-org/onnx/backend/test/data/node/test_ai_onnx_ml_tree_ensemble_set_membership/model.onnx | ❌ | Unsupported op TreeEnsemble |
| onnx-org/onnx/backend/test/data/node/test_ai_onnx_ml_tree_ensemble_single_tree/model.onnx | ❌ | Unsupported op TreeEnsemble |
| onnx-org/onnx/backend/test/data/node/test_and2d/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_and3d/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_and4d/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_and_bcast3v1d/model.onnx | ❌ | And expects identical input/output shapes |
| onnx-org/onnx/backend/test/data/node/test_and_bcast3v2d/model.onnx | ❌ | And expects identical input/output shapes |
| onnx-org/onnx/backend/test/data/node/test_and_bcast4v2d/model.onnx | ❌ | And expects identical input/output shapes |
| onnx-org/onnx/backend/test/data/node/test_and_bcast4v3d/model.onnx | ❌ | And expects identical input/output shapes |
| onnx-org/onnx/backend/test/data/node/test_and_bcast4v4d/model.onnx | ❌ | And expects identical input/output shapes |
| onnx-org/onnx/backend/test/data/node/test_argmax_default_axis_example/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_argmax_default_axis_example_select_last_index/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_argmax_default_axis_random/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_argmax_default_axis_random_select_last_index/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_argmax_keepdims_example/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_argmax_keepdims_example_select_last_index/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_argmax_keepdims_random/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_argmax_keepdims_random_select_last_index/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_argmax_negative_axis_keepdims_example/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_argmax_negative_axis_keepdims_example_select_last_index/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_argmax_negative_axis_keepdims_random/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_argmax_negative_axis_keepdims_random_select_last_index/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_argmax_no_keepdims_example/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_argmax_no_keepdims_example_select_last_index/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_argmax_no_keepdims_random/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_argmax_no_keepdims_random_select_last_index/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_argmin_default_axis_example/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_argmin_default_axis_example_select_last_index/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_argmin_default_axis_random/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_argmin_default_axis_random_select_last_index/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_argmin_keepdims_example/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_argmin_keepdims_example_select_last_index/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_argmin_keepdims_random/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_argmin_keepdims_random_select_last_index/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_argmin_negative_axis_keepdims_example/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_argmin_negative_axis_keepdims_example_select_last_index/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_argmin_negative_axis_keepdims_random/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_argmin_negative_axis_keepdims_random_select_last_index/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_argmin_no_keepdims_example/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_argmin_no_keepdims_example_select_last_index/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_argmin_no_keepdims_random/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_argmin_no_keepdims_random_select_last_index/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_asin/model.onnx | ✅ | OK (max ULP 3) |
| onnx-org/onnx/backend/test/data/node/test_asin_example/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_asinh/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_asinh_example/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_atan/model.onnx | ✅ | OK (max ULP 1) |
| onnx-org/onnx/backend/test/data/node/test_atan_example/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_atanh/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_atanh_example/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_attention_3d/model.onnx | ✅ | OK (max ULP 4) |
| onnx-org/onnx/backend/test/data/node/test_attention_3d_attn_mask/model.onnx | ✅ | OK (max ULP 3) |
| onnx-org/onnx/backend/test/data/node/test_attention_3d_attn_mask_expanded/model.onnx | ✅ | OK (max ULP 2) |
| onnx-org/onnx/backend/test/data/node/test_attention_3d_causal/model.onnx | ✅ | OK (max ULP 3) |
| onnx-org/onnx/backend/test/data/node/test_attention_3d_causal_expanded/model.onnx | ✅ | OK (max ULP 2) |
| onnx-org/onnx/backend/test/data/node/test_attention_3d_diff_heads_sizes/model.onnx | ✅ | OK (max ULP 3) |
| onnx-org/onnx/backend/test/data/node/test_attention_3d_diff_heads_sizes_attn_mask/model.onnx | ✅ | OK (max ULP 3) |
| onnx-org/onnx/backend/test/data/node/test_attention_3d_diff_heads_sizes_attn_mask_expanded/model.onnx | ✅ | OK (max ULP 4) |
| onnx-org/onnx/backend/test/data/node/test_attention_3d_diff_heads_sizes_causal/model.onnx | ✅ | OK (max ULP 2) |
| onnx-org/onnx/backend/test/data/node/test_attention_3d_diff_heads_sizes_causal_expanded/model.onnx | ✅ | OK (max ULP 3) |
| onnx-org/onnx/backend/test/data/node/test_attention_3d_diff_heads_sizes_expanded/model.onnx | ✅ | OK (max ULP 3) |
| onnx-org/onnx/backend/test/data/node/test_attention_3d_diff_heads_sizes_scaled/model.onnx | ✅ | OK (max ULP 4) |
| onnx-org/onnx/backend/test/data/node/test_attention_3d_diff_heads_sizes_scaled_expanded/model.onnx | ✅ | OK (max ULP 3) |
| onnx-org/onnx/backend/test/data/node/test_attention_3d_diff_heads_sizes_softcap/model.onnx | ✅ | OK (max ULP 4) |
| onnx-org/onnx/backend/test/data/node/test_attention_3d_diff_heads_sizes_softcap_expanded/model.onnx | ✅ | OK (max ULP 4) |
| onnx-org/onnx/backend/test/data/node/test_attention_3d_diff_heads_with_past_and_present/model.onnx | ✅ | OK (max ULP 4) |
| onnx-org/onnx/backend/test/data/node/test_attention_3d_diff_heads_with_past_and_present_expanded/model.onnx | ✅ | OK (max ULP 5) |
| onnx-org/onnx/backend/test/data/node/test_attention_3d_expanded/model.onnx | ✅ | OK (max ULP 3) |
| onnx-org/onnx/backend/test/data/node/test_attention_3d_gqa/model.onnx | ✅ | OK (max ULP 4) |
| onnx-org/onnx/backend/test/data/node/test_attention_3d_gqa_attn_mask/model.onnx | ✅ | OK (max ULP 4) |
| onnx-org/onnx/backend/test/data/node/test_attention_3d_gqa_attn_mask_expanded/model.onnx | ✅ | OK (max ULP 3) |
| onnx-org/onnx/backend/test/data/node/test_attention_3d_gqa_causal/model.onnx | ✅ | OK (max ULP 3) |
| onnx-org/onnx/backend/test/data/node/test_attention_3d_gqa_causal_expanded/model.onnx | ✅ | OK (max ULP 3) |
| onnx-org/onnx/backend/test/data/node/test_attention_3d_gqa_expanded/model.onnx | ✅ | OK (max ULP 4) |
| onnx-org/onnx/backend/test/data/node/test_attention_3d_gqa_scaled/model.onnx | ✅ | OK (max ULP 3) |
| onnx-org/onnx/backend/test/data/node/test_attention_3d_gqa_scaled_expanded/model.onnx | ✅ | OK (max ULP 3) |
| onnx-org/onnx/backend/test/data/node/test_attention_3d_gqa_softcap/model.onnx | ✅ | OK (max ULP 3) |
| onnx-org/onnx/backend/test/data/node/test_attention_3d_gqa_softcap_expanded/model.onnx | ✅ | OK (max ULP 3) |
| onnx-org/onnx/backend/test/data/node/test_attention_3d_gqa_with_past_and_present/model.onnx | ✅ | OK (max ULP 5) |
| onnx-org/onnx/backend/test/data/node/test_attention_3d_gqa_with_past_and_present_expanded/model.onnx | ✅ | OK (max ULP 4) |
| onnx-org/onnx/backend/test/data/node/test_attention_3d_scaled/model.onnx | ✅ | OK (max ULP 4) |
| onnx-org/onnx/backend/test/data/node/test_attention_3d_scaled_expanded/model.onnx | ✅ | OK (max ULP 3) |
| onnx-org/onnx/backend/test/data/node/test_attention_3d_softcap/model.onnx | ✅ | OK (max ULP 3) |
| onnx-org/onnx/backend/test/data/node/test_attention_3d_softcap_expanded/model.onnx | ✅ | OK (max ULP 3) |
| onnx-org/onnx/backend/test/data/node/test_attention_3d_transpose_verification/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_attention_3d_transpose_verification_expanded/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_attention_3d_with_past_and_present/model.onnx | ✅ | OK (max ULP 6) |
| onnx-org/onnx/backend/test/data/node/test_attention_3d_with_past_and_present_expanded/model.onnx | ✅ | OK (max ULP 3) |
| onnx-org/onnx/backend/test/data/node/test_attention_3d_with_past_and_present_qk_matmul/model.onnx | ✅ | OK (max ULP 6) |
| onnx-org/onnx/backend/test/data/node/test_attention_3d_with_past_and_present_qk_matmul_bias/model.onnx | ✅ | OK (max ULP 6) |
| onnx-org/onnx/backend/test/data/node/test_attention_3d_with_past_and_present_qk_matmul_bias_expanded/model.onnx | ✅ | OK (max ULP 3) |
| onnx-org/onnx/backend/test/data/node/test_attention_3d_with_past_and_present_qk_matmul_expanded/model.onnx | ✅ | OK (max ULP 4) |
| onnx-org/onnx/backend/test/data/node/test_attention_3d_with_past_and_present_qk_matmul_softcap/model.onnx | ✅ | OK (max ULP 4) |
| onnx-org/onnx/backend/test/data/node/test_attention_3d_with_past_and_present_qk_matmul_softcap_expanded/model.onnx | ✅ | OK (max ULP 4) |
| onnx-org/onnx/backend/test/data/node/test_attention_3d_with_past_and_present_qk_matmul_softmax/model.onnx | ✅ | OK (max ULP 6) |
| onnx-org/onnx/backend/test/data/node/test_attention_3d_with_past_and_present_qk_matmul_softmax_expanded/model.onnx | ✅ | OK (max ULP 4) |
| onnx-org/onnx/backend/test/data/node/test_attention_4d/model.onnx | ✅ | OK (max ULP 3) |
| onnx-org/onnx/backend/test/data/node/test_attention_4d_attn_mask/model.onnx | ✅ | OK (max ULP 4) |
| onnx-org/onnx/backend/test/data/node/test_attention_4d_attn_mask_3d/model.onnx | ✅ | OK (max ULP 5) |
| onnx-org/onnx/backend/test/data/node/test_attention_4d_attn_mask_3d_causal/model.onnx | ✅ | OK (max ULP 2) |
| onnx-org/onnx/backend/test/data/node/test_attention_4d_attn_mask_3d_causal_expanded/model.onnx | ✅ | OK (max ULP 3) |
| onnx-org/onnx/backend/test/data/node/test_attention_4d_attn_mask_3d_expanded/model.onnx | ✅ | OK (max ULP 3) |
| onnx-org/onnx/backend/test/data/node/test_attention_4d_attn_mask_4d/model.onnx | ✅ | OK (max ULP 4) |
| onnx-org/onnx/backend/test/data/node/test_attention_4d_attn_mask_4d_causal/model.onnx | ✅ | OK (max ULP 3) |
| onnx-org/onnx/backend/test/data/node/test_attention_4d_attn_mask_4d_causal_expanded/model.onnx | ✅ | OK (max ULP 2) |
| onnx-org/onnx/backend/test/data/node/test_attention_4d_attn_mask_4d_expanded/model.onnx | ✅ | OK (max ULP 2) |
| onnx-org/onnx/backend/test/data/node/test_attention_4d_attn_mask_bool/model.onnx | ✅ | OK (max ULP 3) |
| onnx-org/onnx/backend/test/data/node/test_attention_4d_attn_mask_bool_4d/model.onnx | ✅ | OK (max ULP 3) |
| onnx-org/onnx/backend/test/data/node/test_attention_4d_attn_mask_bool_4d_expanded/model.onnx | ✅ | OK (max ULP 3) |
| onnx-org/onnx/backend/test/data/node/test_attention_4d_attn_mask_bool_expanded/model.onnx | ✅ | OK (max ULP 3) |
| onnx-org/onnx/backend/test/data/node/test_attention_4d_attn_mask_expanded/model.onnx | ✅ | OK (max ULP 3) |
| onnx-org/onnx/backend/test/data/node/test_attention_4d_causal/model.onnx | ✅ | OK (max ULP 3) |
| onnx-org/onnx/backend/test/data/node/test_attention_4d_causal_expanded/model.onnx | ✅ | OK (max ULP 2) |
| onnx-org/onnx/backend/test/data/node/test_attention_4d_diff_heads_mask4d_padded_kv/model.onnx | ❌ | ONNX Runtime failed to run onnx-org/onnx/backend/test/data/node/test_attention_4d_diff_heads_mask4d_padded_kv/model.onnx: [ONNXRuntimeError] : 1 : FAIL : /onnxruntime_src/onnxruntime/core/graph/model.cc:181 onnxruntime::Model::Model(onnx::ModelProto&&, const onnxruntime::PathString&, const onnxruntime::IOnnxRuntimeOpSchemaRegistryList*, const onnxruntime::logging::Logger&, const onnxruntime::ModelOptions&) Unsupported model IR version: 12, max supported IR version: 11 |
| onnx-org/onnx/backend/test/data/node/test_attention_4d_diff_heads_mask4d_padded_kv_expanded/model.onnx | ❌ | Pad value input must be a scalar |
| onnx-org/onnx/backend/test/data/node/test_attention_4d_diff_heads_sizes/model.onnx | ✅ | OK (max ULP 4) |
| onnx-org/onnx/backend/test/data/node/test_attention_4d_diff_heads_sizes_attn_mask/model.onnx | ✅ | OK (max ULP 2) |
| onnx-org/onnx/backend/test/data/node/test_attention_4d_diff_heads_sizes_attn_mask_expanded/model.onnx | ✅ | OK (max ULP 3) |
| onnx-org/onnx/backend/test/data/node/test_attention_4d_diff_heads_sizes_causal/model.onnx | ✅ | OK (max ULP 3) |
| onnx-org/onnx/backend/test/data/node/test_attention_4d_diff_heads_sizes_causal_expanded/model.onnx | ✅ | OK (max ULP 3) |
| onnx-org/onnx/backend/test/data/node/test_attention_4d_diff_heads_sizes_expanded/model.onnx | ✅ | OK (max ULP 3) |
| onnx-org/onnx/backend/test/data/node/test_attention_4d_diff_heads_sizes_scaled/model.onnx | ✅ | OK (max ULP 3) |
| onnx-org/onnx/backend/test/data/node/test_attention_4d_diff_heads_sizes_scaled_expanded/model.onnx | ✅ | OK (max ULP 3) |
| onnx-org/onnx/backend/test/data/node/test_attention_4d_diff_heads_sizes_softcap/model.onnx | ✅ | OK (max ULP 3) |
| onnx-org/onnx/backend/test/data/node/test_attention_4d_diff_heads_sizes_softcap_expanded/model.onnx | ✅ | OK (max ULP 3) |
| onnx-org/onnx/backend/test/data/node/test_attention_4d_diff_heads_with_past_and_present/model.onnx | ✅ | OK (max ULP 4) |
| onnx-org/onnx/backend/test/data/node/test_attention_4d_diff_heads_with_past_and_present_expanded/model.onnx | ✅ | OK (max ULP 3) |
| onnx-org/onnx/backend/test/data/node/test_attention_4d_diff_heads_with_past_and_present_mask3d/model.onnx | ✅ | OK (max ULP 5) |
| onnx-org/onnx/backend/test/data/node/test_attention_4d_diff_heads_with_past_and_present_mask3d_expanded/model.onnx | ✅ | OK (max ULP 4) |
| onnx-org/onnx/backend/test/data/node/test_attention_4d_diff_heads_with_past_and_present_mask4d/model.onnx | ✅ | OK (max ULP 4) |
| onnx-org/onnx/backend/test/data/node/test_attention_4d_diff_heads_with_past_and_present_mask4d_expanded/model.onnx | ✅ | OK (max ULP 5) |
| onnx-org/onnx/backend/test/data/node/test_attention_4d_expanded/model.onnx | ✅ | OK (max ULP 3) |
| onnx-org/onnx/backend/test/data/node/test_attention_4d_fp16/model.onnx | ✅ | OK (max ULP 2) |
| onnx-org/onnx/backend/test/data/node/test_attention_4d_fp16_expanded/model.onnx | ✅ | OK (max ULP 3) |
| onnx-org/onnx/backend/test/data/node/test_attention_4d_gqa/model.onnx | ✅ | OK (max ULP 3) |
| onnx-org/onnx/backend/test/data/node/test_attention_4d_gqa_attn_mask/model.onnx | ✅ | OK (max ULP 4) |
| onnx-org/onnx/backend/test/data/node/test_attention_4d_gqa_attn_mask_expanded/model.onnx | ✅ | OK (max ULP 3) |
| onnx-org/onnx/backend/test/data/node/test_attention_4d_gqa_causal/model.onnx | ✅ | OK (max ULP 3) |
| onnx-org/onnx/backend/test/data/node/test_attention_4d_gqa_causal_expanded/model.onnx | ✅ | OK (max ULP 3) |
| onnx-org/onnx/backend/test/data/node/test_attention_4d_gqa_expanded/model.onnx | ✅ | OK (max ULP 3) |
| onnx-org/onnx/backend/test/data/node/test_attention_4d_gqa_scaled/model.onnx | ✅ | OK (max ULP 3) |
| onnx-org/onnx/backend/test/data/node/test_attention_4d_gqa_scaled_expanded/model.onnx | ✅ | OK (max ULP 3) |
| onnx-org/onnx/backend/test/data/node/test_attention_4d_gqa_softcap/model.onnx | ✅ | OK (max ULP 3) |
| onnx-org/onnx/backend/test/data/node/test_attention_4d_gqa_softcap_expanded/model.onnx | ✅ | OK (max ULP 3) |
| onnx-org/onnx/backend/test/data/node/test_attention_4d_gqa_with_past_and_present/model.onnx | ✅ | OK (max ULP 5) |
| onnx-org/onnx/backend/test/data/node/test_attention_4d_gqa_with_past_and_present_expanded/model.onnx | ✅ | OK (max ULP 7) |
| onnx-org/onnx/backend/test/data/node/test_attention_4d_gqa_with_past_and_present_fp16/model.onnx | ✅ | OK (max ULP 5) |
| onnx-org/onnx/backend/test/data/node/test_attention_4d_gqa_with_past_and_present_fp16_expanded/model.onnx | ✅ | OK (max ULP 4) |
| onnx-org/onnx/backend/test/data/node/test_attention_4d_scaled/model.onnx | ✅ | OK (max ULP 4) |
| onnx-org/onnx/backend/test/data/node/test_attention_4d_scaled_expanded/model.onnx | ✅ | OK (max ULP 3) |
| onnx-org/onnx/backend/test/data/node/test_attention_4d_softcap/model.onnx | ✅ | OK (max ULP 3) |
| onnx-org/onnx/backend/test/data/node/test_attention_4d_softcap_expanded/model.onnx | ✅ | OK (max ULP 4) |
| onnx-org/onnx/backend/test/data/node/test_attention_4d_with_past_and_present/model.onnx | ✅ | OK (max ULP 5) |
| onnx-org/onnx/backend/test/data/node/test_attention_4d_with_past_and_present_expanded/model.onnx | ✅ | OK (max ULP 4) |
| onnx-org/onnx/backend/test/data/node/test_attention_4d_with_past_and_present_qk_matmul/model.onnx | ✅ | OK (max ULP 4) |
| onnx-org/onnx/backend/test/data/node/test_attention_4d_with_past_and_present_qk_matmul_bias/model.onnx | ✅ | OK (max ULP 5) |
| onnx-org/onnx/backend/test/data/node/test_attention_4d_with_past_and_present_qk_matmul_bias_3d_mask/model.onnx | ✅ | OK (max ULP 3) |
| onnx-org/onnx/backend/test/data/node/test_attention_4d_with_past_and_present_qk_matmul_bias_3d_mask_causal/model.onnx | ✅ | OK (max ULP 4) |
| onnx-org/onnx/backend/test/data/node/test_attention_4d_with_past_and_present_qk_matmul_bias_3d_mask_causal_expanded/model.onnx | ✅ | OK (max ULP 5) |
| onnx-org/onnx/backend/test/data/node/test_attention_4d_with_past_and_present_qk_matmul_bias_3d_mask_expanded/model.onnx | ✅ | OK (max ULP 5) |
| onnx-org/onnx/backend/test/data/node/test_attention_4d_with_past_and_present_qk_matmul_bias_4d_mask/model.onnx | ✅ | OK (max ULP 5) |
| onnx-org/onnx/backend/test/data/node/test_attention_4d_with_past_and_present_qk_matmul_bias_4d_mask_causal/model.onnx | ✅ | OK (max ULP 5) |
| onnx-org/onnx/backend/test/data/node/test_attention_4d_with_past_and_present_qk_matmul_bias_4d_mask_causal_expanded/model.onnx | ✅ | OK (max ULP 4) |
| onnx-org/onnx/backend/test/data/node/test_attention_4d_with_past_and_present_qk_matmul_bias_4d_mask_expanded/model.onnx | ✅ | OK (max ULP 5) |
| onnx-org/onnx/backend/test/data/node/test_attention_4d_with_past_and_present_qk_matmul_bias_expanded/model.onnx | ✅ | OK (max ULP 4) |
| onnx-org/onnx/backend/test/data/node/test_attention_4d_with_past_and_present_qk_matmul_expanded/model.onnx | ✅ | OK (max ULP 4) |
| onnx-org/onnx/backend/test/data/node/test_attention_4d_with_qk_matmul/model.onnx | ✅ | OK (max ULP 3) |
| onnx-org/onnx/backend/test/data/node/test_attention_4d_with_qk_matmul_bias/model.onnx | ✅ | OK (max ULP 4) |
| onnx-org/onnx/backend/test/data/node/test_attention_4d_with_qk_matmul_bias_expanded/model.onnx | ✅ | OK (max ULP 3) |
| onnx-org/onnx/backend/test/data/node/test_attention_4d_with_qk_matmul_expanded/model.onnx | ✅ | OK (max ULP 3) |
| onnx-org/onnx/backend/test/data/node/test_attention_4d_with_qk_matmul_softcap/model.onnx | ✅ | OK (max ULP 4) |
| onnx-org/onnx/backend/test/data/node/test_attention_4d_with_qk_matmul_softcap_expanded/model.onnx | ✅ | OK (max ULP 3) |
| onnx-org/onnx/backend/test/data/node/test_attention_4d_with_qk_matmul_softmax/model.onnx | ✅ | OK (max ULP 4) |
| onnx-org/onnx/backend/test/data/node/test_attention_4d_with_qk_matmul_softmax_expanded/model.onnx | ✅ | OK (max ULP 5) |
| onnx-org/onnx/backend/test/data/node/test_averagepool_1d_default/model.onnx | ❌ | AveragePool expects 2D kernel_shape |
| onnx-org/onnx/backend/test/data/node/test_averagepool_2d_ceil/model.onnx | ❌ | AveragePool supports ceil_mode=0 only |
| onnx-org/onnx/backend/test/data/node/test_averagepool_2d_ceil_last_window_starts_on_pad/model.onnx | ❌ | AveragePool supports ceil_mode=0 only |
| onnx-org/onnx/backend/test/data/node/test_averagepool_2d_default/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_averagepool_2d_dilations/model.onnx | ❌ | AveragePool has unsupported attributes |
| onnx-org/onnx/backend/test/data/node/test_averagepool_2d_pads/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_averagepool_2d_pads_count_include_pad/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_averagepool_2d_precomputed_pads/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_averagepool_2d_precomputed_pads_count_include_pad/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_averagepool_2d_precomputed_same_upper/model.onnx | ❌ | AveragePool supports auto_pad=NOTSET only |
| onnx-org/onnx/backend/test/data/node/test_averagepool_2d_precomputed_strides/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_averagepool_2d_same_lower/model.onnx | ❌ | AveragePool supports auto_pad=NOTSET only |
| onnx-org/onnx/backend/test/data/node/test_averagepool_2d_same_upper/model.onnx | ❌ | AveragePool supports auto_pad=NOTSET only |
| onnx-org/onnx/backend/test/data/node/test_averagepool_2d_strides/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_averagepool_3d_default/model.onnx | ❌ | AveragePool expects 2D kernel_shape |
| onnx-org/onnx/backend/test/data/node/test_averagepool_3d_dilations_large_count_include_pad_is_0_ceil_mode_is_False/model.onnx | ❌ | AveragePool has unsupported attributes |
| onnx-org/onnx/backend/test/data/node/test_averagepool_3d_dilations_large_count_include_pad_is_0_ceil_mode_is_True/model.onnx | ❌ | AveragePool has unsupported attributes |
| onnx-org/onnx/backend/test/data/node/test_averagepool_3d_dilations_large_count_include_pad_is_1_ceil_mode_is_False/model.onnx | ❌ | AveragePool has unsupported attributes |
| onnx-org/onnx/backend/test/data/node/test_averagepool_3d_dilations_large_count_include_pad_is_1_ceil_mode_is_True/model.onnx | ❌ | AveragePool has unsupported attributes |
| onnx-org/onnx/backend/test/data/node/test_averagepool_3d_dilations_small/model.onnx | ❌ | AveragePool has unsupported attributes |
| onnx-org/onnx/backend/test/data/node/test_basic_conv_with_padding/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_basic_conv_without_padding/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_basic_deform_conv_with_padding/model.onnx | ❌ | Unsupported op DeformConv |
| onnx-org/onnx/backend/test/data/node/test_basic_deform_conv_without_padding/model.onnx | ❌ | Unsupported op DeformConv |
| onnx-org/onnx/backend/test/data/node/test_batchnorm_epsilon/model.onnx | ✅ | OK (max ULP 32) |
| onnx-org/onnx/backend/test/data/node/test_batchnorm_epsilon_training_mode/model.onnx | ❌ | BatchNormalization must have 5 inputs and 1 output |
| onnx-org/onnx/backend/test/data/node/test_batchnorm_example/model.onnx | ✅ | OK (max ULP 32) |
| onnx-org/onnx/backend/test/data/node/test_batchnorm_example_training_mode/model.onnx | ❌ | BatchNormalization must have 5 inputs and 1 output |
| onnx-org/onnx/backend/test/data/node/test_bernoulli/model.onnx | ❌ | Unsupported op Bernoulli |
| onnx-org/onnx/backend/test/data/node/test_bernoulli_double/model.onnx | ❌ | Unsupported op Bernoulli |
| onnx-org/onnx/backend/test/data/node/test_bernoulli_double_expanded/model.onnx | ❌ | Unsupported op RandomUniformLike |
| onnx-org/onnx/backend/test/data/node/test_bernoulli_expanded/model.onnx | ❌ | Unsupported op RandomUniformLike |
| onnx-org/onnx/backend/test/data/node/test_bernoulli_seed/model.onnx | ❌ | Unsupported op Bernoulli |
| onnx-org/onnx/backend/test/data/node/test_bernoulli_seed_expanded/model.onnx | ❌ | Unsupported op RandomUniformLike |
| onnx-org/onnx/backend/test/data/node/test_bitshift_left_uint16/model.onnx | ✅ |  |
| onnx-org/onnx/backend/test/data/node/test_bitshift_left_uint32/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_bitshift_left_uint64/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_bitshift_left_uint8/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_bitshift_right_uint16/model.onnx | ✅ |  |
| onnx-org/onnx/backend/test/data/node/test_bitshift_right_uint32/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_bitshift_right_uint64/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_bitshift_right_uint8/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_bitwise_and_i16_3d/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_bitwise_and_i32_2d/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_bitwise_and_ui64_bcast_3v1d/model.onnx | ❌ | BitwiseAnd expects identical input/output shapes |
| onnx-org/onnx/backend/test/data/node/test_bitwise_and_ui8_bcast_4v3d/model.onnx | ❌ | BitwiseAnd expects identical input/output shapes |
| onnx-org/onnx/backend/test/data/node/test_bitwise_not_2d/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_bitwise_not_3d/model.onnx | ❌ | Unsupported op BitwiseNot |
| onnx-org/onnx/backend/test/data/node/test_bitwise_not_4d/model.onnx | ❌ | Unsupported op BitwiseNot |
| onnx-org/onnx/backend/test/data/node/test_bitwise_or_i16_4d/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_bitwise_or_i32_2d/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_bitwise_or_ui64_bcast_3v1d/model.onnx | ❌ | BitwiseOr expects identical input/output shapes |
| onnx-org/onnx/backend/test/data/node/test_bitwise_or_ui8_bcast_4v3d/model.onnx | ❌ | BitwiseOr expects identical input/output shapes |
| onnx-org/onnx/backend/test/data/node/test_bitwise_xor_i16_3d/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_bitwise_xor_i32_2d/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_bitwise_xor_ui64_bcast_3v1d/model.onnx | ❌ | BitwiseXor expects identical input/output shapes |
| onnx-org/onnx/backend/test/data/node/test_bitwise_xor_ui8_bcast_4v3d/model.onnx | ❌ | BitwiseXor expects identical input/output shapes |
| onnx-org/onnx/backend/test/data/node/test_blackmanwindow/model.onnx | ❌ | Unsupported op BlackmanWindow |
| onnx-org/onnx/backend/test/data/node/test_blackmanwindow_expanded/model.onnx | ✅ | OK (max ULP 8) |
| onnx-org/onnx/backend/test/data/node/test_blackmanwindow_symmetric/model.onnx | ❌ | Unsupported op BlackmanWindow |
| onnx-org/onnx/backend/test/data/node/test_blackmanwindow_symmetric_expanded/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_cast_BFLOAT16_to_FLOAT/model.onnx | ❌ | Unsupported elem_type 16 (BFLOAT16) for tensor 'input'. |
| onnx-org/onnx/backend/test/data/node/test_cast_DOUBLE_to_FLOAT/model.onnx | ❌ | ONNX Runtime failed to run onnx-org/onnx/backend/test/data/node/test_cast_DOUBLE_to_FLOAT/model.onnx: [ONNXRuntimeError] : 1 : FAIL : /onnxruntime_src/onnxruntime/core/graph/model.cc:181 onnxruntime::Model::Model(onnx::ModelProto&&, const onnxruntime::PathString&, const onnxruntime::IOnnxRuntimeOpSchemaRegistryList*, const onnxruntime::logging::Logger&, const onnxruntime::ModelOptions&) Unsupported model IR version: 13, max supported IR version: 11 |
| onnx-org/onnx/backend/test/data/node/test_cast_DOUBLE_to_FLOAT16/model.onnx | ❌ | ONNX Runtime failed to run onnx-org/onnx/backend/test/data/node/test_cast_DOUBLE_to_FLOAT16/model.onnx: [ONNXRuntimeError] : 1 : FAIL : /onnxruntime_src/onnxruntime/core/graph/model.cc:181 onnxruntime::Model::Model(onnx::ModelProto&&, const onnxruntime::PathString&, const onnxruntime::IOnnxRuntimeOpSchemaRegistryList*, const onnxruntime::logging::Logger&, const onnxruntime::ModelOptions&) Unsupported model IR version: 13, max supported IR version: 11 |
| onnx-org/onnx/backend/test/data/node/test_cast_FLOAT16_to_DOUBLE/model.onnx | ❌ | ONNX Runtime failed to run onnx-org/onnx/backend/test/data/node/test_cast_FLOAT16_to_DOUBLE/model.onnx: [ONNXRuntimeError] : 1 : FAIL : /onnxruntime_src/onnxruntime/core/graph/model.cc:181 onnxruntime::Model::Model(onnx::ModelProto&&, const onnxruntime::PathString&, const onnxruntime::IOnnxRuntimeOpSchemaRegistryList*, const onnxruntime::logging::Logger&, const onnxruntime::ModelOptions&) Unsupported model IR version: 13, max supported IR version: 11 |
| onnx-org/onnx/backend/test/data/node/test_cast_FLOAT16_to_FLOAT/model.onnx | ❌ | ONNX Runtime failed to run onnx-org/onnx/backend/test/data/node/test_cast_FLOAT16_to_FLOAT/model.onnx: [ONNXRuntimeError] : 1 : FAIL : /onnxruntime_src/onnxruntime/core/graph/model.cc:181 onnxruntime::Model::Model(onnx::ModelProto&&, const onnxruntime::PathString&, const onnxruntime::IOnnxRuntimeOpSchemaRegistryList*, const onnxruntime::logging::Logger&, const onnxruntime::ModelOptions&) Unsupported model IR version: 13, max supported IR version: 11 |
| onnx-org/onnx/backend/test/data/node/test_cast_FLOAT16_to_FLOAT4E2M1/model.onnx | ❌ | Unsupported elem_type 23 (FLOAT4E2M1) for tensor 'output'. |
| onnx-org/onnx/backend/test/data/node/test_cast_FLOAT16_to_FLOAT8E4M3FN/model.onnx | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'output'. |
| onnx-org/onnx/backend/test/data/node/test_cast_FLOAT16_to_FLOAT8E4M3FNUZ/model.onnx | ❌ | Unsupported elem_type 18 (FLOAT8E4M3FNUZ) for tensor 'output'. |
| onnx-org/onnx/backend/test/data/node/test_cast_FLOAT16_to_FLOAT8E5M2/model.onnx | ❌ | Unsupported elem_type 19 (FLOAT8E5M2) for tensor 'output'. |
| onnx-org/onnx/backend/test/data/node/test_cast_FLOAT16_to_FLOAT8E5M2FNUZ/model.onnx | ❌ | Unsupported elem_type 20 (FLOAT8E5M2FNUZ) for tensor 'output'. |
| onnx-org/onnx/backend/test/data/node/test_cast_FLOAT16_to_INT2/model.onnx | ❌ | Unsupported elem_type 26 (INT2) for tensor 'output'. |
| onnx-org/onnx/backend/test/data/node/test_cast_FLOAT16_to_INT4/model.onnx | ❌ | Unsupported elem_type 22 (INT4) for tensor 'output'. |
| onnx-org/onnx/backend/test/data/node/test_cast_FLOAT16_to_UINT2/model.onnx | ❌ | Unsupported elem_type 25 (UINT2) for tensor 'output'. |
| onnx-org/onnx/backend/test/data/node/test_cast_FLOAT16_to_UINT4/model.onnx | ❌ | Unsupported elem_type 21 (UINT4) for tensor 'output'. |
| onnx-org/onnx/backend/test/data/node/test_cast_FLOAT4E2M1_to_FLOAT/model.onnx | ❌ | Unsupported elem_type 23 (FLOAT4E2M1) for tensor 'input'. |
| onnx-org/onnx/backend/test/data/node/test_cast_FLOAT4E2M1_to_FLOAT16/model.onnx | ❌ | Unsupported elem_type 23 (FLOAT4E2M1) for tensor 'input'. |
| onnx-org/onnx/backend/test/data/node/test_cast_FLOAT8E4M3FNUZ_to_FLOAT/model.onnx | ❌ | Unsupported elem_type 18 (FLOAT8E4M3FNUZ) for tensor 'input'. |
| onnx-org/onnx/backend/test/data/node/test_cast_FLOAT8E4M3FNUZ_to_FLOAT16/model.onnx | ❌ | Unsupported elem_type 18 (FLOAT8E4M3FNUZ) for tensor 'input'. |
| onnx-org/onnx/backend/test/data/node/test_cast_FLOAT8E4M3FN_to_FLOAT/model.onnx | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'input'. |
| onnx-org/onnx/backend/test/data/node/test_cast_FLOAT8E4M3FN_to_FLOAT16/model.onnx | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'input'. |
| onnx-org/onnx/backend/test/data/node/test_cast_FLOAT8E5M2FNUZ_to_FLOAT/model.onnx | ❌ | Unsupported elem_type 20 (FLOAT8E5M2FNUZ) for tensor 'input'. |
| onnx-org/onnx/backend/test/data/node/test_cast_FLOAT8E5M2FNUZ_to_FLOAT16/model.onnx | ❌ | Unsupported elem_type 20 (FLOAT8E5M2FNUZ) for tensor 'input'. |
| onnx-org/onnx/backend/test/data/node/test_cast_FLOAT8E5M2_to_FLOAT/model.onnx | ❌ | Unsupported elem_type 19 (FLOAT8E5M2) for tensor 'input'. |
| onnx-org/onnx/backend/test/data/node/test_cast_FLOAT8E5M2_to_FLOAT16/model.onnx | ❌ | Unsupported elem_type 19 (FLOAT8E5M2) for tensor 'input'. |
| onnx-org/onnx/backend/test/data/node/test_cast_FLOAT_to_BFLOAT16/model.onnx | ❌ | Unsupported elem_type 16 (BFLOAT16) for tensor 'output'. |
| onnx-org/onnx/backend/test/data/node/test_cast_FLOAT_to_DOUBLE/model.onnx | ❌ | ONNX Runtime failed to run onnx-org/onnx/backend/test/data/node/test_cast_FLOAT_to_DOUBLE/model.onnx: [ONNXRuntimeError] : 1 : FAIL : /onnxruntime_src/onnxruntime/core/graph/model.cc:181 onnxruntime::Model::Model(onnx::ModelProto&&, const onnxruntime::PathString&, const onnxruntime::IOnnxRuntimeOpSchemaRegistryList*, const onnxruntime::logging::Logger&, const onnxruntime::ModelOptions&) Unsupported model IR version: 13, max supported IR version: 11 |
| onnx-org/onnx/backend/test/data/node/test_cast_FLOAT_to_FLOAT16/model.onnx | ❌ | ONNX Runtime failed to run onnx-org/onnx/backend/test/data/node/test_cast_FLOAT_to_FLOAT16/model.onnx: [ONNXRuntimeError] : 1 : FAIL : /onnxruntime_src/onnxruntime/core/graph/model.cc:181 onnxruntime::Model::Model(onnx::ModelProto&&, const onnxruntime::PathString&, const onnxruntime::IOnnxRuntimeOpSchemaRegistryList*, const onnxruntime::logging::Logger&, const onnxruntime::ModelOptions&) Unsupported model IR version: 13, max supported IR version: 11 |
| onnx-org/onnx/backend/test/data/node/test_cast_FLOAT_to_FLOAT4E2M1/model.onnx | ❌ | Unsupported elem_type 23 (FLOAT4E2M1) for tensor 'output'. |
| onnx-org/onnx/backend/test/data/node/test_cast_FLOAT_to_FLOAT8E4M3FN/model.onnx | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'output'. |
| onnx-org/onnx/backend/test/data/node/test_cast_FLOAT_to_FLOAT8E4M3FNUZ/model.onnx | ❌ | Unsupported elem_type 18 (FLOAT8E4M3FNUZ) for tensor 'output'. |
| onnx-org/onnx/backend/test/data/node/test_cast_FLOAT_to_FLOAT8E5M2/model.onnx | ❌ | Unsupported elem_type 19 (FLOAT8E5M2) for tensor 'output'. |
| onnx-org/onnx/backend/test/data/node/test_cast_FLOAT_to_FLOAT8E5M2FNUZ/model.onnx | ❌ | Unsupported elem_type 20 (FLOAT8E5M2FNUZ) for tensor 'output'. |
| onnx-org/onnx/backend/test/data/node/test_cast_FLOAT_to_INT2/model.onnx | ❌ | Unsupported elem_type 26 (INT2) for tensor 'output'. |
| onnx-org/onnx/backend/test/data/node/test_cast_FLOAT_to_INT4/model.onnx | ❌ | Unsupported elem_type 22 (INT4) for tensor 'output'. |
| onnx-org/onnx/backend/test/data/node/test_cast_FLOAT_to_UINT2/model.onnx | ❌ | Unsupported elem_type 25 (UINT2) for tensor 'output'. |
| onnx-org/onnx/backend/test/data/node/test_cast_FLOAT_to_UINT4/model.onnx | ❌ | Unsupported elem_type 21 (UINT4) for tensor 'output'. |
| onnx-org/onnx/backend/test/data/node/test_cast_INT2_to_FLOAT/model.onnx | ❌ | Unsupported elem_type 26 (INT2) for tensor 'input'. |
| onnx-org/onnx/backend/test/data/node/test_cast_INT2_to_FLOAT16/model.onnx | ❌ | Unsupported elem_type 26 (INT2) for tensor 'input'. |
| onnx-org/onnx/backend/test/data/node/test_cast_INT2_to_INT8/model.onnx | ❌ | Unsupported elem_type 26 (INT2) for tensor 'input'. |
| onnx-org/onnx/backend/test/data/node/test_cast_INT4_to_FLOAT/model.onnx | ❌ | Unsupported elem_type 22 (INT4) for tensor 'input'. |
| onnx-org/onnx/backend/test/data/node/test_cast_INT4_to_FLOAT16/model.onnx | ❌ | Unsupported elem_type 22 (INT4) for tensor 'input'. |
| onnx-org/onnx/backend/test/data/node/test_cast_INT4_to_INT8/model.onnx | ❌ | Unsupported elem_type 22 (INT4) for tensor 'input'. |
| onnx-org/onnx/backend/test/data/node/test_cast_UINT2_to_FLOAT/model.onnx | ❌ | Unsupported elem_type 25 (UINT2) for tensor 'input'. |
| onnx-org/onnx/backend/test/data/node/test_cast_UINT2_to_FLOAT16/model.onnx | ❌ | Unsupported elem_type 25 (UINT2) for tensor 'input'. |
| onnx-org/onnx/backend/test/data/node/test_cast_UINT2_to_UINT8/model.onnx | ❌ | Unsupported elem_type 25 (UINT2) for tensor 'input'. |
| onnx-org/onnx/backend/test/data/node/test_cast_UINT4_to_FLOAT/model.onnx | ❌ | Unsupported elem_type 21 (UINT4) for tensor 'input'. |
| onnx-org/onnx/backend/test/data/node/test_cast_UINT4_to_FLOAT16/model.onnx | ❌ | Unsupported elem_type 21 (UINT4) for tensor 'input'. |
| onnx-org/onnx/backend/test/data/node/test_cast_UINT4_to_UINT8/model.onnx | ❌ | Unsupported elem_type 21 (UINT4) for tensor 'input'. |
| onnx-org/onnx/backend/test/data/node/test_cast_e8m0_FLOAT16_to_FLOAT8E8M0/model.onnx | ❌ | Unsupported elem_type 24 (FLOAT8E8M0) for tensor 'output'. |
| onnx-org/onnx/backend/test/data/node/test_cast_e8m0_FLOAT8E8M0_to_FLOAT/model.onnx | ❌ | Unsupported elem_type 24 (FLOAT8E8M0) for tensor 'input'. |
| onnx-org/onnx/backend/test/data/node/test_cast_e8m0_FLOAT8E8M0_to_FLOAT16/model.onnx | ❌ | Unsupported elem_type 24 (FLOAT8E8M0) for tensor 'input'. |
| onnx-org/onnx/backend/test/data/node/test_cast_e8m0_FLOAT_to_FLOAT8E8M0/model.onnx | ❌ | Unsupported elem_type 24 (FLOAT8E8M0) for tensor 'output'. |
| onnx-org/onnx/backend/test/data/node/test_cast_no_saturate_FLOAT16_to_FLOAT8E4M3FN/model.onnx | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'output'. |
| onnx-org/onnx/backend/test/data/node/test_cast_no_saturate_FLOAT16_to_FLOAT8E4M3FNUZ/model.onnx | ❌ | Unsupported elem_type 18 (FLOAT8E4M3FNUZ) for tensor 'output'. |
| onnx-org/onnx/backend/test/data/node/test_cast_no_saturate_FLOAT16_to_FLOAT8E5M2/model.onnx | ❌ | Unsupported elem_type 19 (FLOAT8E5M2) for tensor 'output'. |
| onnx-org/onnx/backend/test/data/node/test_cast_no_saturate_FLOAT16_to_FLOAT8E5M2FNUZ/model.onnx | ❌ | Unsupported elem_type 20 (FLOAT8E5M2FNUZ) for tensor 'output'. |
| onnx-org/onnx/backend/test/data/node/test_cast_no_saturate_FLOAT_to_FLOAT8E4M3FN/model.onnx | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'output'. |
| onnx-org/onnx/backend/test/data/node/test_cast_no_saturate_FLOAT_to_FLOAT8E4M3FNUZ/model.onnx | ❌ | Unsupported elem_type 18 (FLOAT8E4M3FNUZ) for tensor 'output'. |
| onnx-org/onnx/backend/test/data/node/test_cast_no_saturate_FLOAT_to_FLOAT8E5M2/model.onnx | ❌ | Unsupported elem_type 19 (FLOAT8E5M2) for tensor 'output'. |
| onnx-org/onnx/backend/test/data/node/test_cast_no_saturate_FLOAT_to_FLOAT8E5M2FNUZ/model.onnx | ❌ | Unsupported elem_type 20 (FLOAT8E5M2FNUZ) for tensor 'output'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_BFLOAT16_to_FLOAT/model.onnx | ❌ | Unsupported elem_type 16 (BFLOAT16) for tensor 'input'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_BFLOAT16_to_FLOAT_expanded/model.onnx | ❌ | Unsupported elem_type 16 (BFLOAT16) for tensor 'input'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_DOUBLE_to_FLOAT/model.onnx | ❌ | ONNX Runtime failed to run onnx-org/onnx/backend/test/data/node/test_castlike_DOUBLE_to_FLOAT/model.onnx: [ONNXRuntimeError] : 1 : FAIL : /onnxruntime_src/onnxruntime/core/graph/model.cc:181 onnxruntime::Model::Model(onnx::ModelProto&&, const onnxruntime::PathString&, const onnxruntime::IOnnxRuntimeOpSchemaRegistryList*, const onnxruntime::logging::Logger&, const onnxruntime::ModelOptions&) Unsupported model IR version: 13, max supported IR version: 11 |
| onnx-org/onnx/backend/test/data/node/test_castlike_DOUBLE_to_FLOAT16/model.onnx | ❌ | ONNX Runtime failed to run onnx-org/onnx/backend/test/data/node/test_castlike_DOUBLE_to_FLOAT16/model.onnx: [ONNXRuntimeError] : 1 : FAIL : /onnxruntime_src/onnxruntime/core/graph/model.cc:181 onnxruntime::Model::Model(onnx::ModelProto&&, const onnxruntime::PathString&, const onnxruntime::IOnnxRuntimeOpSchemaRegistryList*, const onnxruntime::logging::Logger&, const onnxruntime::ModelOptions&) Unsupported model IR version: 13, max supported IR version: 11 |
| onnx-org/onnx/backend/test/data/node/test_castlike_DOUBLE_to_FLOAT16_expanded/model.onnx | ❌ | ONNX Runtime failed to run onnx-org/onnx/backend/test/data/node/test_castlike_DOUBLE_to_FLOAT16_expanded/model.onnx: [ONNXRuntimeError] : 1 : FAIL : /onnxruntime_src/onnxruntime/core/graph/model.cc:181 onnxruntime::Model::Model(onnx::ModelProto&&, const onnxruntime::PathString&, const onnxruntime::IOnnxRuntimeOpSchemaRegistryList*, const onnxruntime::logging::Logger&, const onnxruntime::ModelOptions&) Unsupported model IR version: 13, max supported IR version: 11 |
| onnx-org/onnx/backend/test/data/node/test_castlike_DOUBLE_to_FLOAT_expanded/model.onnx | ❌ | ONNX Runtime failed to run onnx-org/onnx/backend/test/data/node/test_castlike_DOUBLE_to_FLOAT_expanded/model.onnx: [ONNXRuntimeError] : 1 : FAIL : /onnxruntime_src/onnxruntime/core/graph/model.cc:181 onnxruntime::Model::Model(onnx::ModelProto&&, const onnxruntime::PathString&, const onnxruntime::IOnnxRuntimeOpSchemaRegistryList*, const onnxruntime::logging::Logger&, const onnxruntime::ModelOptions&) Unsupported model IR version: 13, max supported IR version: 11 |
| onnx-org/onnx/backend/test/data/node/test_castlike_FLOAT16_to_DOUBLE/model.onnx | ❌ | ONNX Runtime failed to run onnx-org/onnx/backend/test/data/node/test_castlike_FLOAT16_to_DOUBLE/model.onnx: [ONNXRuntimeError] : 1 : FAIL : /onnxruntime_src/onnxruntime/core/graph/model.cc:181 onnxruntime::Model::Model(onnx::ModelProto&&, const onnxruntime::PathString&, const onnxruntime::IOnnxRuntimeOpSchemaRegistryList*, const onnxruntime::logging::Logger&, const onnxruntime::ModelOptions&) Unsupported model IR version: 13, max supported IR version: 11 |
| onnx-org/onnx/backend/test/data/node/test_castlike_FLOAT16_to_DOUBLE_expanded/model.onnx | ❌ | ONNX Runtime failed to run onnx-org/onnx/backend/test/data/node/test_castlike_FLOAT16_to_DOUBLE_expanded/model.onnx: [ONNXRuntimeError] : 1 : FAIL : /onnxruntime_src/onnxruntime/core/graph/model.cc:181 onnxruntime::Model::Model(onnx::ModelProto&&, const onnxruntime::PathString&, const onnxruntime::IOnnxRuntimeOpSchemaRegistryList*, const onnxruntime::logging::Logger&, const onnxruntime::ModelOptions&) Unsupported model IR version: 13, max supported IR version: 11 |
| onnx-org/onnx/backend/test/data/node/test_castlike_FLOAT16_to_FLOAT/model.onnx | ❌ | ONNX Runtime failed to run onnx-org/onnx/backend/test/data/node/test_castlike_FLOAT16_to_FLOAT/model.onnx: [ONNXRuntimeError] : 1 : FAIL : /onnxruntime_src/onnxruntime/core/graph/model.cc:181 onnxruntime::Model::Model(onnx::ModelProto&&, const onnxruntime::PathString&, const onnxruntime::IOnnxRuntimeOpSchemaRegistryList*, const onnxruntime::logging::Logger&, const onnxruntime::ModelOptions&) Unsupported model IR version: 13, max supported IR version: 11 |
| onnx-org/onnx/backend/test/data/node/test_castlike_FLOAT16_to_FLOAT4E2M1/model.onnx | ❌ | Unsupported elem_type 23 (FLOAT4E2M1) for tensor 'like'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_FLOAT16_to_FLOAT4E2M1_expanded/model.onnx | ❌ | Unsupported elem_type 23 (FLOAT4E2M1) for tensor 'like'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_FLOAT16_to_FLOAT8E4M3FN/model.onnx | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'like'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_FLOAT16_to_FLOAT8E4M3FNUZ/model.onnx | ❌ | Unsupported elem_type 18 (FLOAT8E4M3FNUZ) for tensor 'like'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_FLOAT16_to_FLOAT8E4M3FNUZ_expanded/model.onnx | ❌ | Unsupported elem_type 18 (FLOAT8E4M3FNUZ) for tensor 'like'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_FLOAT16_to_FLOAT8E4M3FN_expanded/model.onnx | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'like'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_FLOAT16_to_FLOAT8E5M2/model.onnx | ❌ | Unsupported elem_type 19 (FLOAT8E5M2) for tensor 'like'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_FLOAT16_to_FLOAT8E5M2FNUZ/model.onnx | ❌ | Unsupported elem_type 20 (FLOAT8E5M2FNUZ) for tensor 'like'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_FLOAT16_to_FLOAT8E5M2FNUZ_expanded/model.onnx | ❌ | Unsupported elem_type 20 (FLOAT8E5M2FNUZ) for tensor 'like'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_FLOAT16_to_FLOAT8E5M2_expanded/model.onnx | ❌ | Unsupported elem_type 19 (FLOAT8E5M2) for tensor 'like'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_FLOAT16_to_FLOAT_expanded/model.onnx | ❌ | ONNX Runtime failed to run onnx-org/onnx/backend/test/data/node/test_castlike_FLOAT16_to_FLOAT_expanded/model.onnx: [ONNXRuntimeError] : 1 : FAIL : /onnxruntime_src/onnxruntime/core/graph/model.cc:181 onnxruntime::Model::Model(onnx::ModelProto&&, const onnxruntime::PathString&, const onnxruntime::IOnnxRuntimeOpSchemaRegistryList*, const onnxruntime::logging::Logger&, const onnxruntime::ModelOptions&) Unsupported model IR version: 13, max supported IR version: 11 |
| onnx-org/onnx/backend/test/data/node/test_castlike_FLOAT16_to_INT2/model.onnx | ❌ | Unsupported elem_type 26 (INT2) for tensor 'like'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_FLOAT16_to_INT2_expanded/model.onnx | ❌ | Unsupported elem_type 26 (INT2) for tensor 'like'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_FLOAT16_to_INT4/model.onnx | ❌ | Unsupported elem_type 22 (INT4) for tensor 'like'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_FLOAT16_to_INT4_expanded/model.onnx | ❌ | Unsupported elem_type 22 (INT4) for tensor 'like'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_FLOAT16_to_UINT2/model.onnx | ❌ | Unsupported elem_type 25 (UINT2) for tensor 'like'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_FLOAT16_to_UINT2_expanded/model.onnx | ❌ | Unsupported elem_type 25 (UINT2) for tensor 'like'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_FLOAT16_to_UINT4/model.onnx | ❌ | Unsupported elem_type 21 (UINT4) for tensor 'like'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_FLOAT16_to_UINT4_expanded/model.onnx | ❌ | Unsupported elem_type 21 (UINT4) for tensor 'like'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_FLOAT4E2M1_to_FLOAT/model.onnx | ❌ | Unsupported elem_type 23 (FLOAT4E2M1) for tensor 'input'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_FLOAT4E2M1_to_FLOAT16/model.onnx | ❌ | Unsupported elem_type 23 (FLOAT4E2M1) for tensor 'input'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_FLOAT4E2M1_to_FLOAT16_expanded/model.onnx | ❌ | Unsupported elem_type 23 (FLOAT4E2M1) for tensor 'input'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_FLOAT4E2M1_to_FLOAT_expanded/model.onnx | ❌ | Unsupported elem_type 23 (FLOAT4E2M1) for tensor 'input'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_FLOAT8E4M3FNUZ_to_FLOAT/model.onnx | ❌ | Unsupported elem_type 18 (FLOAT8E4M3FNUZ) for tensor 'input'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_FLOAT8E4M3FNUZ_to_FLOAT16/model.onnx | ❌ | Unsupported elem_type 18 (FLOAT8E4M3FNUZ) for tensor 'input'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_FLOAT8E4M3FNUZ_to_FLOAT16_expanded/model.onnx | ❌ | Unsupported elem_type 18 (FLOAT8E4M3FNUZ) for tensor 'input'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_FLOAT8E4M3FNUZ_to_FLOAT_expanded/model.onnx | ❌ | Unsupported elem_type 18 (FLOAT8E4M3FNUZ) for tensor 'input'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_FLOAT8E4M3FN_to_FLOAT/model.onnx | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'input'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_FLOAT8E4M3FN_to_FLOAT16/model.onnx | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'input'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_FLOAT8E4M3FN_to_FLOAT16_expanded/model.onnx | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'input'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_FLOAT8E4M3FN_to_FLOAT_expanded/model.onnx | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'input'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_FLOAT8E5M2FNUZ_to_FLOAT/model.onnx | ❌ | Unsupported elem_type 20 (FLOAT8E5M2FNUZ) for tensor 'input'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_FLOAT8E5M2FNUZ_to_FLOAT16/model.onnx | ❌ | Unsupported elem_type 20 (FLOAT8E5M2FNUZ) for tensor 'input'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_FLOAT8E5M2FNUZ_to_FLOAT16_expanded/model.onnx | ❌ | Unsupported elem_type 20 (FLOAT8E5M2FNUZ) for tensor 'input'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_FLOAT8E5M2FNUZ_to_FLOAT_expanded/model.onnx | ❌ | Unsupported elem_type 20 (FLOAT8E5M2FNUZ) for tensor 'input'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_FLOAT8E5M2_to_FLOAT/model.onnx | ❌ | Unsupported elem_type 19 (FLOAT8E5M2) for tensor 'input'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_FLOAT8E5M2_to_FLOAT16/model.onnx | ❌ | Unsupported elem_type 19 (FLOAT8E5M2) for tensor 'input'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_FLOAT8E5M2_to_FLOAT16_expanded/model.onnx | ❌ | Unsupported elem_type 19 (FLOAT8E5M2) for tensor 'input'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_FLOAT8E5M2_to_FLOAT_expanded/model.onnx | ❌ | Unsupported elem_type 19 (FLOAT8E5M2) for tensor 'input'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_FLOAT_to_BFLOAT16/model.onnx | ❌ | Unsupported elem_type 16 (BFLOAT16) for tensor 'like'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_FLOAT_to_BFLOAT16_expanded/model.onnx | ❌ | Unsupported elem_type 16 (BFLOAT16) for tensor 'like'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_FLOAT_to_DOUBLE/model.onnx | ❌ | ONNX Runtime failed to run onnx-org/onnx/backend/test/data/node/test_castlike_FLOAT_to_DOUBLE/model.onnx: [ONNXRuntimeError] : 1 : FAIL : /onnxruntime_src/onnxruntime/core/graph/model.cc:181 onnxruntime::Model::Model(onnx::ModelProto&&, const onnxruntime::PathString&, const onnxruntime::IOnnxRuntimeOpSchemaRegistryList*, const onnxruntime::logging::Logger&, const onnxruntime::ModelOptions&) Unsupported model IR version: 13, max supported IR version: 11 |
| onnx-org/onnx/backend/test/data/node/test_castlike_FLOAT_to_DOUBLE_expanded/model.onnx | ❌ | ONNX Runtime failed to run onnx-org/onnx/backend/test/data/node/test_castlike_FLOAT_to_DOUBLE_expanded/model.onnx: [ONNXRuntimeError] : 1 : FAIL : /onnxruntime_src/onnxruntime/core/graph/model.cc:181 onnxruntime::Model::Model(onnx::ModelProto&&, const onnxruntime::PathString&, const onnxruntime::IOnnxRuntimeOpSchemaRegistryList*, const onnxruntime::logging::Logger&, const onnxruntime::ModelOptions&) Unsupported model IR version: 13, max supported IR version: 11 |
| onnx-org/onnx/backend/test/data/node/test_castlike_FLOAT_to_FLOAT16/model.onnx | ❌ | ONNX Runtime failed to run onnx-org/onnx/backend/test/data/node/test_castlike_FLOAT_to_FLOAT16/model.onnx: [ONNXRuntimeError] : 1 : FAIL : /onnxruntime_src/onnxruntime/core/graph/model.cc:181 onnxruntime::Model::Model(onnx::ModelProto&&, const onnxruntime::PathString&, const onnxruntime::IOnnxRuntimeOpSchemaRegistryList*, const onnxruntime::logging::Logger&, const onnxruntime::ModelOptions&) Unsupported model IR version: 13, max supported IR version: 11 |
| onnx-org/onnx/backend/test/data/node/test_castlike_FLOAT_to_FLOAT16_expanded/model.onnx | ❌ | ONNX Runtime failed to run onnx-org/onnx/backend/test/data/node/test_castlike_FLOAT_to_FLOAT16_expanded/model.onnx: [ONNXRuntimeError] : 1 : FAIL : /onnxruntime_src/onnxruntime/core/graph/model.cc:181 onnxruntime::Model::Model(onnx::ModelProto&&, const onnxruntime::PathString&, const onnxruntime::IOnnxRuntimeOpSchemaRegistryList*, const onnxruntime::logging::Logger&, const onnxruntime::ModelOptions&) Unsupported model IR version: 13, max supported IR version: 11 |
| onnx-org/onnx/backend/test/data/node/test_castlike_FLOAT_to_FLOAT4E2M1/model.onnx | ❌ | Unsupported elem_type 23 (FLOAT4E2M1) for tensor 'like'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_FLOAT_to_FLOAT4E2M1_expanded/model.onnx | ❌ | Unsupported elem_type 23 (FLOAT4E2M1) for tensor 'like'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_FLOAT_to_FLOAT8E4M3FN/model.onnx | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'like'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_FLOAT_to_FLOAT8E4M3FNUZ/model.onnx | ❌ | Unsupported elem_type 18 (FLOAT8E4M3FNUZ) for tensor 'like'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_FLOAT_to_FLOAT8E4M3FNUZ_expanded/model.onnx | ❌ | Unsupported elem_type 18 (FLOAT8E4M3FNUZ) for tensor 'like'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_FLOAT_to_FLOAT8E4M3FN_expanded/model.onnx | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'like'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_FLOAT_to_FLOAT8E5M2/model.onnx | ❌ | Unsupported elem_type 19 (FLOAT8E5M2) for tensor 'like'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_FLOAT_to_FLOAT8E5M2FNUZ/model.onnx | ❌ | Unsupported elem_type 20 (FLOAT8E5M2FNUZ) for tensor 'like'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_FLOAT_to_FLOAT8E5M2FNUZ_expanded/model.onnx | ❌ | Unsupported elem_type 20 (FLOAT8E5M2FNUZ) for tensor 'like'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_FLOAT_to_FLOAT8E5M2_expanded/model.onnx | ❌ | Unsupported elem_type 19 (FLOAT8E5M2) for tensor 'like'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_FLOAT_to_INT2/model.onnx | ❌ | Unsupported elem_type 26 (INT2) for tensor 'like'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_FLOAT_to_INT2_expanded/model.onnx | ❌ | Unsupported elem_type 26 (INT2) for tensor 'like'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_FLOAT_to_INT4/model.onnx | ❌ | Unsupported elem_type 22 (INT4) for tensor 'like'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_FLOAT_to_INT4_expanded/model.onnx | ❌ | Unsupported elem_type 22 (INT4) for tensor 'like'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_FLOAT_to_UINT2/model.onnx | ❌ | Unsupported elem_type 25 (UINT2) for tensor 'like'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_FLOAT_to_UINT2_expanded/model.onnx | ❌ | Unsupported elem_type 25 (UINT2) for tensor 'like'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_FLOAT_to_UINT4/model.onnx | ❌ | Unsupported elem_type 21 (UINT4) for tensor 'like'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_FLOAT_to_UINT4_expanded/model.onnx | ❌ | Unsupported elem_type 21 (UINT4) for tensor 'like'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_INT2_to_FLOAT/model.onnx | ❌ | Unsupported elem_type 26 (INT2) for tensor 'input'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_INT2_to_FLOAT16/model.onnx | ❌ | Unsupported elem_type 26 (INT2) for tensor 'input'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_INT2_to_FLOAT16_expanded/model.onnx | ❌ | Unsupported elem_type 26 (INT2) for tensor 'input'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_INT2_to_FLOAT_expanded/model.onnx | ❌ | Unsupported elem_type 26 (INT2) for tensor 'input'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_INT2_to_INT8/model.onnx | ❌ | Unsupported elem_type 26 (INT2) for tensor 'input'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_INT2_to_INT8_expanded/model.onnx | ❌ | Unsupported elem_type 26 (INT2) for tensor 'input'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_INT4_to_FLOAT/model.onnx | ❌ | Unsupported elem_type 22 (INT4) for tensor 'input'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_INT4_to_FLOAT16/model.onnx | ❌ | Unsupported elem_type 22 (INT4) for tensor 'input'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_INT4_to_FLOAT16_expanded/model.onnx | ❌ | Unsupported elem_type 22 (INT4) for tensor 'input'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_INT4_to_FLOAT_expanded/model.onnx | ❌ | Unsupported elem_type 22 (INT4) for tensor 'input'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_INT4_to_INT8/model.onnx | ❌ | Unsupported elem_type 22 (INT4) for tensor 'input'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_INT4_to_INT8_expanded/model.onnx | ❌ | Unsupported elem_type 22 (INT4) for tensor 'input'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_UINT2_to_FLOAT/model.onnx | ❌ | Unsupported elem_type 25 (UINT2) for tensor 'input'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_UINT2_to_FLOAT16/model.onnx | ❌ | Unsupported elem_type 25 (UINT2) for tensor 'input'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_UINT2_to_FLOAT16_expanded/model.onnx | ❌ | Unsupported elem_type 25 (UINT2) for tensor 'input'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_UINT2_to_FLOAT_expanded/model.onnx | ❌ | Unsupported elem_type 25 (UINT2) for tensor 'input'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_UINT2_to_UINT8/model.onnx | ❌ | Unsupported elem_type 25 (UINT2) for tensor 'input'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_UINT2_to_UINT8_expanded/model.onnx | ❌ | Unsupported elem_type 25 (UINT2) for tensor 'input'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_UINT4_to_FLOAT/model.onnx | ❌ | Unsupported elem_type 21 (UINT4) for tensor 'input'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_UINT4_to_FLOAT16/model.onnx | ❌ | Unsupported elem_type 21 (UINT4) for tensor 'input'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_UINT4_to_FLOAT16_expanded/model.onnx | ❌ | Unsupported elem_type 21 (UINT4) for tensor 'input'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_UINT4_to_FLOAT_expanded/model.onnx | ❌ | Unsupported elem_type 21 (UINT4) for tensor 'input'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_UINT4_to_UINT8/model.onnx | ❌ | Unsupported elem_type 21 (UINT4) for tensor 'input'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_UINT4_to_UINT8_expanded/model.onnx | ❌ | Unsupported elem_type 21 (UINT4) for tensor 'input'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_no_saturate_FLOAT16_to_FLOAT8E4M3FN/model.onnx | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'like'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_no_saturate_FLOAT16_to_FLOAT8E4M3FNUZ/model.onnx | ❌ | Unsupported elem_type 18 (FLOAT8E4M3FNUZ) for tensor 'like'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_no_saturate_FLOAT16_to_FLOAT8E4M3FNUZ_expanded/model.onnx | ❌ | Unsupported elem_type 18 (FLOAT8E4M3FNUZ) for tensor 'like'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_no_saturate_FLOAT16_to_FLOAT8E4M3FN_expanded/model.onnx | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'like'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_no_saturate_FLOAT16_to_FLOAT8E5M2/model.onnx | ❌ | Unsupported elem_type 19 (FLOAT8E5M2) for tensor 'like'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_no_saturate_FLOAT16_to_FLOAT8E5M2FNUZ/model.onnx | ❌ | Unsupported elem_type 20 (FLOAT8E5M2FNUZ) for tensor 'like'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_no_saturate_FLOAT16_to_FLOAT8E5M2FNUZ_expanded/model.onnx | ❌ | Unsupported elem_type 20 (FLOAT8E5M2FNUZ) for tensor 'like'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_no_saturate_FLOAT16_to_FLOAT8E5M2_expanded/model.onnx | ❌ | Unsupported elem_type 19 (FLOAT8E5M2) for tensor 'like'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_no_saturate_FLOAT_to_FLOAT8E4M3FN/model.onnx | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'like'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_no_saturate_FLOAT_to_FLOAT8E4M3FNUZ/model.onnx | ❌ | Unsupported elem_type 18 (FLOAT8E4M3FNUZ) for tensor 'like'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_no_saturate_FLOAT_to_FLOAT8E4M3FNUZ_expanded/model.onnx | ❌ | Unsupported elem_type 18 (FLOAT8E4M3FNUZ) for tensor 'like'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_no_saturate_FLOAT_to_FLOAT8E4M3FN_expanded/model.onnx | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'like'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_no_saturate_FLOAT_to_FLOAT8E5M2/model.onnx | ❌ | Unsupported elem_type 19 (FLOAT8E5M2) for tensor 'like'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_no_saturate_FLOAT_to_FLOAT8E5M2FNUZ/model.onnx | ❌ | Unsupported elem_type 20 (FLOAT8E5M2FNUZ) for tensor 'like'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_no_saturate_FLOAT_to_FLOAT8E5M2FNUZ_expanded/model.onnx | ❌ | Unsupported elem_type 20 (FLOAT8E5M2FNUZ) for tensor 'like'. |
| onnx-org/onnx/backend/test/data/node/test_castlike_no_saturate_FLOAT_to_FLOAT8E5M2_expanded/model.onnx | ❌ | Unsupported elem_type 19 (FLOAT8E5M2) for tensor 'like'. |
| onnx-org/onnx/backend/test/data/node/test_ceil/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_ceil_example/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_celu/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_celu_expanded/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_center_crop_pad_crop/model.onnx | ❌ | Unsupported op CenterCropPad |
| onnx-org/onnx/backend/test/data/node/test_center_crop_pad_crop_and_pad/model.onnx | ❌ | Unsupported op CenterCropPad |
| onnx-org/onnx/backend/test/data/node/test_center_crop_pad_crop_and_pad_expanded/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_center_crop_pad_crop_axes_chw/model.onnx | ❌ | Unsupported op CenterCropPad |
| onnx-org/onnx/backend/test/data/node/test_center_crop_pad_crop_axes_chw_expanded/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_center_crop_pad_crop_axes_hwc/model.onnx | ❌ | Unsupported op CenterCropPad |
| onnx-org/onnx/backend/test/data/node/test_center_crop_pad_crop_axes_hwc_expanded/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_center_crop_pad_crop_expanded/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_center_crop_pad_crop_negative_axes_hwc/model.onnx | ❌ | Unsupported op CenterCropPad |
| onnx-org/onnx/backend/test/data/node/test_center_crop_pad_crop_negative_axes_hwc_expanded/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_center_crop_pad_pad/model.onnx | ❌ | Unsupported op CenterCropPad |
| onnx-org/onnx/backend/test/data/node/test_center_crop_pad_pad_expanded/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_clip/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_clip_default_inbounds/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_clip_default_inbounds_expanded/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_clip_default_int8_inbounds/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_clip_default_int8_inbounds_expanded/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_clip_default_int8_max/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_clip_default_int8_max_expanded/model.onnx | ✅ |  |
| onnx-org/onnx/backend/test/data/node/test_clip_default_int8_min/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_clip_default_int8_min_expanded/model.onnx | ✅ |  |
| onnx-org/onnx/backend/test/data/node/test_clip_default_max/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_clip_default_max_expanded/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_clip_default_min/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_clip_default_min_expanded/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_clip_example/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_clip_example_expanded/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_clip_expanded/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_clip_inbounds/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_clip_inbounds_expanded/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_clip_min_greater_than_max/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_clip_min_greater_than_max_expanded/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_clip_outbounds/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_clip_outbounds_expanded/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_clip_splitbounds/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_clip_splitbounds_expanded/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_col2im/model.onnx | ❌ | Unsupported op Col2Im |
| onnx-org/onnx/backend/test/data/node/test_col2im_5d/model.onnx | ❌ | Unsupported op Col2Im |
| onnx-org/onnx/backend/test/data/node/test_col2im_dilations/model.onnx | ❌ | Unsupported op Col2Im |
| onnx-org/onnx/backend/test/data/node/test_col2im_pads/model.onnx | ❌ | Unsupported op Col2Im |
| onnx-org/onnx/backend/test/data/node/test_col2im_strides/model.onnx | ❌ | Unsupported op Col2Im |
| onnx-org/onnx/backend/test/data/node/test_compress_0/model.onnx | ❌ | Unsupported op Compress |
| onnx-org/onnx/backend/test/data/node/test_compress_1/model.onnx | ❌ | Unsupported op Compress |
| onnx-org/onnx/backend/test/data/node/test_compress_default_axis/model.onnx | ❌ | Unsupported op Compress |
| onnx-org/onnx/backend/test/data/node/test_compress_negative_axis/model.onnx | ❌ | Unsupported op Compress |
| onnx-org/onnx/backend/test/data/node/test_concat_1d_axis_0/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_concat_1d_axis_negative_1/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_concat_2d_axis_0/model.onnx | ✅ | OK (max ULP 0) |
| onnx-org/onnx/backend/test/data/node/test_concat_2d_axis_1/model.onnx | ✅ | OK (max ULP 0) |

## Local ONNX file support

Local tests: `onnx2c-org/test/local_ops`.

Support 60 / 74 local ONNX files.

| File | Supported | Error |
| --- | --- | --- |
| test_gather_basic/model.onnx | ✅ | OK (max ULP 0) |
| test_gather_output_scalar/model.onnx | ✅ | OK (max ULP 0) |
| test_gather_scalar_axis0/model.onnx | ✅ | OK (max ULP 0) |
| test_gather_scalar_axis1/model.onnx | ✅ | OK (max ULP 0) |
| test_gemm_C1/model.onnx | ✅ | OK (max ULP 1) |
| test_gemm_C1_transA/model.onnx | ✅ | OK (max ULP 0) |
| test_gemm_C1_transB/model.onnx | ✅ | OK (max ULP 0) |
| test_gemm_C1x1/model.onnx | ✅ | OK (max ULP 0) |
| test_gemm_C1x1_transA/model.onnx | ✅ | OK (max ULP 0) |
| test_gemm_C1xN/model.onnx | ✅ | OK (max ULP 0) |
| test_gemm_C1xN_transA/model.onnx | ✅ | OK (max ULP 1) |
| test_gemm_C1xN_transA_transB/model.onnx | ✅ | OK (max ULP 1) |
| test_gemm_CM_transA/model.onnx | ❌ | Gemm bias input must be broadcastable to output shape, got (2,) vs (2, 4) |
| test_gemm_CMx1/model.onnx | ✅ | OK (max ULP 1) |
| test_gemm_CMx1_transA/model.onnx | ✅ | OK (max ULP 1) |
| test_gemm_CMx1_transA_transB/model.onnx | ✅ | OK (max ULP 0) |
| test_gemm_CMxN/model.onnx | ✅ | OK (max ULP 1) |
| test_gemm_CMxN_transA/model.onnx | ✅ | OK (max ULP 0) |
| test_gemm_CMxN_transA_transB/model.onnx | ✅ | OK (max ULP 1) |
| test_gemm_CMxN_transB/model.onnx | ✅ | OK (max ULP 1) |
| test_gemm_CN/model.onnx | ✅ | OK (max ULP 0) |
| test_gemm_CN_transA/model.onnx | ✅ | OK (max ULP 1) |
| test_gemm_CN_transA_transB/model.onnx | ✅ | OK (max ULP 1) |
| test_gemm_CN_transB/model.onnx | ✅ | OK (max ULP 1) |
| test_lstm_activations/model.onnx | ✅ | OK (max ULP 0) |
| test_lstm_all_outputs/model.onnx | ✅ | OK (max ULP 2) |
| test_lstm_bidirectional/model.onnx | ❌ | Unsupported LSTM direction b'bidirectional' |
| test_lstm_clip/model.onnx | ✅ | OK (max ULP 2) |
| test_lstm_intermediate_h/model.onnx | ✅ | OK (max ULP 2) |
| test_lstm_missing_inputs/model.onnx | ✅ | OK (max ULP 1) |
| test_lstm_reverse/model.onnx | ❌ | Unsupported LSTM direction b'reverse' |
| test_lstm_seq_length/model.onnx | ✅ | OK (max ULP 3) |
| test_lstm_simple/model.onnx | ✅ | OK (max ULP 0) |
| test_lstm_with_initial_state/model.onnx | ✅ | OK (max ULP 4) |
| test_lstm_y_c/model.onnx | ✅ | OK (max ULP 2) |
| test_matmul_1x1x3x4_2x3x4x5/model.onnx | ✅ | OK (max ULP 1) |
| test_matmul_1x3x4_2x3x4x5/model.onnx | ✅ | OK (max ULP 1) |
| test_matmul_1x3x4_3x4x5/model.onnx | ✅ | OK (max ULP 1) |
| test_matmul_2x1x3x4_2x3x4x5/model.onnx | ✅ | OK (max ULP 2) |
| test_matmul_2x3_3x4/model.onnx | ✅ | OK (max ULP 1) |
| test_matmul_2x3x3x4_1x4x5/model.onnx | ✅ | OK (max ULP 1) |
| test_matmul_2x3x4_4/model.onnx | ✅ | OK (max ULP 0) |
| test_matmul_2x3x4_4x5/model.onnx | ✅ | OK (max ULP 1) |
| test_matmul_2x3x4x5_5/model.onnx | ✅ | OK (max ULP 0) |
| test_matmul_3_2x3x4/model.onnx | ✅ | OK (max ULP 0) |
| test_matmul_3_3/model.onnx | ✅ | OK (max ULP 0) |
| test_matmul_3_3x4/model.onnx | ✅ | OK (max ULP 0) |
| test_matmul_3x4_2x4x5/model.onnx | ✅ | OK (max ULP 1) |
| test_matmul_3x4_4/model.onnx | ✅ | OK (max ULP 0) |
| test_matmul_4x5x2x3_4x5x3x4/model.onnx | ✅ | OK (max ULP 2) |
| test_matmul_5x2x3_5x3x4/model.onnx | ✅ | OK (max ULP 1) |
| test_matmul_precision/model.onnx | ✅ | OK (max ULP 0) |
| test_maxpool_stride_1/model.onnx | ✅ | OK (max ULP 0) |
| test_maxpool_stride_2/model.onnx | ✅ | OK (max ULP 0) |
| test_nodes_out_of_order/model.onnx | ✅ | OK (max ULP 0) |
| test_pad_constant_default/model.onnx | ✅ | OK (max ULP 0) |
| test_pad_constant_input/model.onnx | ✅ | OK (max ULP 0) |
| test_pad_edge/model.onnx | ✅ | OK (max ULP 0) |
| test_pad_edge_allaxes/model.onnx | ✅ | OK (max ULP 0) |
| test_pad_reflect_allaxes/model.onnx | ✅ | OK (max ULP 0) |
| test_pad_reflect_nopadding/model.onnx | ✅ | OK (max ULP 0) |
| test_qlinearadd_int8/model.onnx | ❌ | Unsupported op QLinearAdd |
| test_qlinearadd_uint8/model.onnx | ❌ | Unsupported op QLinearAdd |
| test_qlinearmul_int8/model.onnx | ❌ | Unsupported op QLinearMul |
| test_qlinearmul_uint8/model.onnx | ❌ | Unsupported op QLinearMul |
| test_resize_downsample_sizes_linear_1D/model.onnx | ❌ | ONNX Runtime failed to run onnx2c-org/test/local_ops/test_resize_downsample_sizes_linear_1D/model.onnx: [ONNXRuntimeError] : 10 : INVALID_GRAPH : This is an invalid model. In Node, ("sclbl-onnx-node1", Resize, "", -1) : ("X": tensor(float),"","","sizes": tensor(int64),) -> ("Y": tensor(float),) , Error Node (sclbl-onnx-node1)'s input 1 is marked single but has an empty string in the graph |
| test_resize_downsample_sizes_linear_1D_align/model.onnx | ❌ | ONNX Runtime failed to run onnx2c-org/test/local_ops/test_resize_downsample_sizes_linear_1D_align/model.onnx: [ONNXRuntimeError] : 10 : INVALID_GRAPH : This is an invalid model. In Node, ("sclbl-onnx-node1", Resize, "", -1) : ("X": tensor(float),"","","sizes": tensor(int64),) -> ("Y": tensor(float),) , Error Node (sclbl-onnx-node1)'s input 1 is marked single but has an empty string in the graph |
| test_scalar_input_to_node/model.onnx | ✅ | OK (max ULP 0) |
| test_scatternd_indices_1x1x2/model.onnx | ❌ | Unsupported op ScatterND |
| test_scatternd_indices_1x2x2/model.onnx | ❌ | Unsupported op ScatterND |
| test_scatternd_indices_2x2x2/model.onnx | ❌ | Unsupported op ScatterND |
| test_scatternd_indices_3x2/model.onnx | ❌ | Unsupported op ScatterND |
| test_shape_const_out/model.onnx | ✅ | OK (max ULP 0) |
| test_slice_end_INT64_MAX/model.onnx | ✅ | OK (max ULP 0) |

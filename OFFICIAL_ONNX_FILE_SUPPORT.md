# Official ONNX file support

Support 922 / 1802 official ONNX files.

ONNX version: 1.20.1

See [`OFFICIAL_ONNX_FILE_SUPPORT_HISTOGRAM.md`](OFFICIAL_ONNX_FILE_SUPPORT_HISTOGRAM.md) for the error histogram.

| File | Supported | Error |
| --- | --- | --- |
| light/light_bvlc_alexnet.onnx | ✅ |  |
| light/light_densenet121.onnx | ✅ |  |
| light/light_inception_v1.onnx | ✅ |  |
| light/light_inception_v2.onnx | ✅ |  |
| light/light_resnet50.onnx | ✅ |  |
| light/light_shufflenet.onnx | ✅ |  |
| light/light_squeezenet.onnx | ✅ |  |
| light/light_vgg19.onnx | ✅ |  |
| light/light_zfnet512.onnx | ✅ |  |
| node/test_abs/model.onnx | ✅ |  |
| node/test_acos/model.onnx | ✅ |  |
| node/test_acos_example/model.onnx | ✅ |  |
| node/test_acosh/model.onnx | ✅ |  |
| node/test_acosh_example/model.onnx | ✅ |  |
| node/test_adagrad/model.onnx | ❌ | Unsupported op Adagrad |
| node/test_adagrad_multiple/model.onnx | ❌ | Unsupported op Adagrad |
| node/test_adam/model.onnx | ❌ | Unsupported op Adam |
| node/test_adam_multiple/model.onnx | ❌ | Unsupported op Adam |
| node/test_add/model.onnx | ✅ |  |
| node/test_add_bcast/model.onnx | ✅ |  |
| node/test_add_int16/model.onnx | ✅ |  |
| node/test_add_int8/model.onnx | ✅ |  |
| node/test_add_uint16/model.onnx | ✅ |  |
| node/test_add_uint32/model.onnx | ✅ |  |
| node/test_add_uint64/model.onnx | ✅ |  |
| node/test_add_uint8/model.onnx | ✅ |  |
| node/test_affine_grid_2d/model.onnx | ❌ | Unsupported op AffineGrid |
| node/test_affine_grid_2d_align_corners/model.onnx | ❌ | Unsupported op AffineGrid |
| node/test_affine_grid_2d_align_corners_expanded/model.onnx | ❌ | Dynamic dim for tensor 'AffineGrid_test_affine_grid_2d_align_corners_expanded_function_W' |
| node/test_affine_grid_2d_expanded/model.onnx | ❌ | Dynamic dim for tensor 'AffineGrid_test_affine_grid_2d_expanded_function_W' |
| node/test_affine_grid_3d/model.onnx | ❌ | Unsupported op AffineGrid |
| node/test_affine_grid_3d_align_corners/model.onnx | ❌ | Unsupported op AffineGrid |
| node/test_affine_grid_3d_align_corners_expanded/model.onnx | ❌ | Dynamic dim for tensor 'AffineGrid_test_affine_grid_3d_align_corners_expanded_function_N' |
| node/test_affine_grid_3d_expanded/model.onnx | ❌ | Dynamic dim for tensor 'AffineGrid_test_affine_grid_3d_expanded_function_N' |
| node/test_ai_onnx_ml_array_feature_extractor/model.onnx | ❌ | Unsupported op ArrayFeatureExtractor |
| node/test_ai_onnx_ml_binarizer/model.onnx | ❌ | Unsupported op Binarizer |
| node/test_ai_onnx_ml_label_encoder_string_int/model.onnx | ❌ | Unsupported elem_type 8 (STRING) for tensor 'X'. |
| node/test_ai_onnx_ml_label_encoder_string_int_no_default/model.onnx | ❌ | Unsupported elem_type 8 (STRING) for tensor 'X'. |
| node/test_ai_onnx_ml_label_encoder_tensor_mapping/model.onnx | ❌ | Unsupported elem_type 8 (STRING) for tensor 'X'. |
| node/test_ai_onnx_ml_label_encoder_tensor_value_only_mapping/model.onnx | ❌ | Unsupported elem_type 8 (STRING) for tensor 'X'. |
| node/test_ai_onnx_ml_tree_ensemble_set_membership/model.onnx | ❌ | Unsupported op TreeEnsemble |
| node/test_ai_onnx_ml_tree_ensemble_single_tree/model.onnx | ❌ | Unsupported op TreeEnsemble |
| node/test_and2d/model.onnx | ✅ |  |
| node/test_and3d/model.onnx | ✅ |  |
| node/test_and4d/model.onnx | ✅ |  |
| node/test_and_bcast3v1d/model.onnx | ✅ |  |
| node/test_and_bcast3v2d/model.onnx | ✅ |  |
| node/test_and_bcast4v2d/model.onnx | ✅ |  |
| node/test_and_bcast4v3d/model.onnx | ✅ |  |
| node/test_and_bcast4v4d/model.onnx | ✅ |  |
| node/test_argmax_default_axis_example/model.onnx | ❌ | Unsupported op ArgMax |
| node/test_argmax_default_axis_example_select_last_index/model.onnx | ❌ | Unsupported op ArgMax |
| node/test_argmax_default_axis_random/model.onnx | ❌ | Unsupported op ArgMax |
| node/test_argmax_default_axis_random_select_last_index/model.onnx | ❌ | Unsupported op ArgMax |
| node/test_argmax_keepdims_example/model.onnx | ❌ | Unsupported op ArgMax |
| node/test_argmax_keepdims_example_select_last_index/model.onnx | ❌ | Unsupported op ArgMax |
| node/test_argmax_keepdims_random/model.onnx | ❌ | Unsupported op ArgMax |
| node/test_argmax_keepdims_random_select_last_index/model.onnx | ❌ | Unsupported op ArgMax |
| node/test_argmax_negative_axis_keepdims_example/model.onnx | ❌ | Unsupported op ArgMax |
| node/test_argmax_negative_axis_keepdims_example_select_last_index/model.onnx | ❌ | Unsupported op ArgMax |
| node/test_argmax_negative_axis_keepdims_random/model.onnx | ❌ | Unsupported op ArgMax |
| node/test_argmax_negative_axis_keepdims_random_select_last_index/model.onnx | ❌ | Unsupported op ArgMax |
| node/test_argmax_no_keepdims_example/model.onnx | ❌ | Unsupported op ArgMax |
| node/test_argmax_no_keepdims_example_select_last_index/model.onnx | ❌ | Unsupported op ArgMax |
| node/test_argmax_no_keepdims_random/model.onnx | ❌ | Unsupported op ArgMax |
| node/test_argmax_no_keepdims_random_select_last_index/model.onnx | ❌ | Unsupported op ArgMax |
| node/test_argmin_default_axis_example/model.onnx | ❌ | Unsupported op ArgMin |
| node/test_argmin_default_axis_example_select_last_index/model.onnx | ❌ | Unsupported op ArgMin |
| node/test_argmin_default_axis_random/model.onnx | ❌ | Unsupported op ArgMin |
| node/test_argmin_default_axis_random_select_last_index/model.onnx | ❌ | Unsupported op ArgMin |
| node/test_argmin_keepdims_example/model.onnx | ❌ | Unsupported op ArgMin |
| node/test_argmin_keepdims_example_select_last_index/model.onnx | ❌ | Unsupported op ArgMin |
| node/test_argmin_keepdims_random/model.onnx | ❌ | Unsupported op ArgMin |
| node/test_argmin_keepdims_random_select_last_index/model.onnx | ❌ | Unsupported op ArgMin |
| node/test_argmin_negative_axis_keepdims_example/model.onnx | ❌ | Unsupported op ArgMin |
| node/test_argmin_negative_axis_keepdims_example_select_last_index/model.onnx | ❌ | Unsupported op ArgMin |
| node/test_argmin_negative_axis_keepdims_random/model.onnx | ❌ | Unsupported op ArgMin |
| node/test_argmin_negative_axis_keepdims_random_select_last_index/model.onnx | ❌ | Unsupported op ArgMin |
| node/test_argmin_no_keepdims_example/model.onnx | ❌ | Unsupported op ArgMin |
| node/test_argmin_no_keepdims_example_select_last_index/model.onnx | ❌ | Unsupported op ArgMin |
| node/test_argmin_no_keepdims_random/model.onnx | ❌ | Unsupported op ArgMin |
| node/test_argmin_no_keepdims_random_select_last_index/model.onnx | ❌ | Unsupported op ArgMin |
| node/test_asin/model.onnx | ✅ |  |
| node/test_asin_example/model.onnx | ✅ |  |
| node/test_asinh/model.onnx | ✅ |  |
| node/test_asinh_example/model.onnx | ✅ |  |
| node/test_atan/model.onnx | ✅ |  |
| node/test_atan_example/model.onnx | ✅ |  |
| node/test_atanh/model.onnx | ✅ |  |
| node/test_atanh_example/model.onnx | ✅ |  |
| node/test_attention_3d/model.onnx | ✅ |  |
| node/test_attention_3d_attn_mask/model.onnx | ✅ |  |
| node/test_attention_3d_attn_mask_expanded/model.onnx | ❌ | Dynamic dim for tensor 'Attention_test_attention_3d_attn_mask_expanded_function_KExpanded' |
| node/test_attention_3d_causal/model.onnx | ✅ |  |
| node/test_attention_3d_causal_expanded/model.onnx | ❌ | Dynamic dim for tensor 'Attention_test_attention_3d_causal_expanded_function_RangeRow' |
| node/test_attention_3d_diff_heads_sizes/model.onnx | ✅ |  |
| node/test_attention_3d_diff_heads_sizes_attn_mask/model.onnx | ✅ |  |
| node/test_attention_3d_diff_heads_sizes_attn_mask_expanded/model.onnx | ❌ | Dynamic dim for tensor 'Attention_test_attention_3d_diff_heads_sizes_attn_mask_expanded_function_KExpanded' |
| node/test_attention_3d_diff_heads_sizes_causal/model.onnx | ✅ |  |
| node/test_attention_3d_diff_heads_sizes_causal_expanded/model.onnx | ❌ | Dynamic dim for tensor 'Attention_test_attention_3d_diff_heads_sizes_causal_expanded_function_RangeRow' |
| node/test_attention_3d_diff_heads_sizes_expanded/model.onnx | ❌ | Dynamic dim for tensor 'Attention_test_attention_3d_diff_heads_sizes_expanded_function_KExpanded' |
| node/test_attention_3d_diff_heads_sizes_scaled/model.onnx | ✅ |  |
| node/test_attention_3d_diff_heads_sizes_scaled_expanded/model.onnx | ❌ | Dynamic dim for tensor 'Attention_test_attention_3d_diff_heads_sizes_scaled_expanded_function_KExpanded' |
| node/test_attention_3d_diff_heads_sizes_softcap/model.onnx | ✅ |  |
| node/test_attention_3d_diff_heads_sizes_softcap_expanded/model.onnx | ❌ | Dynamic dim for tensor 'Attention_test_attention_3d_diff_heads_sizes_softcap_expanded_function_KExpanded' |
| node/test_attention_3d_diff_heads_with_past_and_present/model.onnx | ✅ |  |
| node/test_attention_3d_diff_heads_with_past_and_present_expanded/model.onnx | ❌ | Dynamic dim for tensor 'Attention_test_attention_3d_diff_heads_with_past_and_present_expanded_function_KExpanded' |
| node/test_attention_3d_expanded/model.onnx | ❌ | Dynamic dim for tensor 'Attention_test_attention_3d_expanded_function_KExpanded' |
| node/test_attention_3d_gqa/model.onnx | ✅ |  |
| node/test_attention_3d_gqa_attn_mask/model.onnx | ✅ |  |
| node/test_attention_3d_gqa_attn_mask_expanded/model.onnx | ❌ | Dynamic dim for tensor 'Attention_test_attention_3d_gqa_attn_mask_expanded_function_KExpanded' |
| node/test_attention_3d_gqa_causal/model.onnx | ✅ |  |
| node/test_attention_3d_gqa_causal_expanded/model.onnx | ❌ | Dynamic dim for tensor 'Attention_test_attention_3d_gqa_causal_expanded_function_RangeRow' |
| node/test_attention_3d_gqa_expanded/model.onnx | ❌ | Dynamic dim for tensor 'Attention_test_attention_3d_gqa_expanded_function_KExpanded' |
| node/test_attention_3d_gqa_scaled/model.onnx | ✅ |  |
| node/test_attention_3d_gqa_scaled_expanded/model.onnx | ❌ | Dynamic dim for tensor 'Attention_test_attention_3d_gqa_scaled_expanded_function_KExpanded' |
| node/test_attention_3d_gqa_softcap/model.onnx | ✅ |  |
| node/test_attention_3d_gqa_softcap_expanded/model.onnx | ❌ | Dynamic dim for tensor 'Attention_test_attention_3d_gqa_softcap_expanded_function_KExpanded' |
| node/test_attention_3d_gqa_with_past_and_present/model.onnx | ✅ |  |
| node/test_attention_3d_gqa_with_past_and_present_expanded/model.onnx | ❌ | Dynamic dim for tensor 'Attention_test_attention_3d_gqa_with_past_and_present_expanded_function_KExpanded' |
| node/test_attention_3d_scaled/model.onnx | ✅ |  |
| node/test_attention_3d_scaled_expanded/model.onnx | ❌ | Dynamic dim for tensor 'Attention_test_attention_3d_scaled_expanded_function_KExpanded' |
| node/test_attention_3d_softcap/model.onnx | ✅ |  |
| node/test_attention_3d_softcap_expanded/model.onnx | ❌ | Dynamic dim for tensor 'Attention_test_attention_3d_softcap_expanded_function_KExpanded' |
| node/test_attention_3d_transpose_verification/model.onnx | ✅ |  |
| node/test_attention_3d_transpose_verification_expanded/model.onnx | ❌ | Dynamic dim for tensor 'Attention_test_attention_3d_transpose_verification_expanded_function_KExpanded' |
| node/test_attention_3d_with_past_and_present/model.onnx | ✅ |  |
| node/test_attention_3d_with_past_and_present_expanded/model.onnx | ❌ | Dynamic dim for tensor 'Attention_test_attention_3d_with_past_and_present_expanded_function_KExpanded' |
| node/test_attention_3d_with_past_and_present_qk_matmul/model.onnx | ✅ |  |
| node/test_attention_3d_with_past_and_present_qk_matmul_bias/model.onnx | ✅ |  |
| node/test_attention_3d_with_past_and_present_qk_matmul_bias_expanded/model.onnx | ❌ | Dynamic dim for tensor 'Attention_test_attention_3d_with_past_and_present_qk_matmul_bias_expanded_function_KExpanded' |
| node/test_attention_3d_with_past_and_present_qk_matmul_expanded/model.onnx | ❌ | Dynamic dim for tensor 'Attention_test_attention_3d_with_past_and_present_qk_matmul_expanded_function_KExpanded' |
| node/test_attention_3d_with_past_and_present_qk_matmul_softcap/model.onnx | ✅ |  |
| node/test_attention_3d_with_past_and_present_qk_matmul_softcap_expanded/model.onnx | ❌ | Dynamic dim for tensor 'Attention_test_attention_3d_with_past_and_present_qk_matmul_softcap_expanded_function_KExpanded' |
| node/test_attention_3d_with_past_and_present_qk_matmul_softmax/model.onnx | ✅ |  |
| node/test_attention_3d_with_past_and_present_qk_matmul_softmax_expanded/model.onnx | ❌ | Dynamic dim for tensor 'Attention_test_attention_3d_with_past_and_present_qk_matmul_softmax_expanded_function_KExpanded' |
| node/test_attention_4d/model.onnx | ✅ |  |
| node/test_attention_4d_attn_mask/model.onnx | ✅ |  |
| node/test_attention_4d_attn_mask_3d/model.onnx | ✅ |  |
| node/test_attention_4d_attn_mask_3d_causal/model.onnx | ✅ |  |
| node/test_attention_4d_attn_mask_3d_causal_expanded/model.onnx | ❌ | Dynamic dim for tensor 'Attention_test_attention_4d_attn_mask_3d_causal_expanded_function_RangeRow' |
| node/test_attention_4d_attn_mask_3d_expanded/model.onnx | ❌ | Dynamic dim for tensor 'Attention_test_attention_4d_attn_mask_3d_expanded_function_KExpanded' |
| node/test_attention_4d_attn_mask_4d/model.onnx | ✅ |  |
| node/test_attention_4d_attn_mask_4d_causal/model.onnx | ✅ |  |
| node/test_attention_4d_attn_mask_4d_causal_expanded/model.onnx | ❌ | Dynamic dim for tensor 'Attention_test_attention_4d_attn_mask_4d_causal_expanded_function_RangeRow' |
| node/test_attention_4d_attn_mask_4d_expanded/model.onnx | ❌ | Dynamic dim for tensor 'Attention_test_attention_4d_attn_mask_4d_expanded_function_KExpanded' |
| node/test_attention_4d_attn_mask_bool/model.onnx | ✅ |  |
| node/test_attention_4d_attn_mask_bool_4d/model.onnx | ✅ |  |
| node/test_attention_4d_attn_mask_bool_4d_expanded/model.onnx | ❌ | Dynamic dim for tensor 'Attention_test_attention_4d_attn_mask_bool_4d_expanded_function_KExpanded' |
| node/test_attention_4d_attn_mask_bool_expanded/model.onnx | ❌ | Dynamic dim for tensor 'Attention_test_attention_4d_attn_mask_bool_expanded_function_KExpanded' |
| node/test_attention_4d_attn_mask_expanded/model.onnx | ❌ | Dynamic dim for tensor 'Attention_test_attention_4d_attn_mask_expanded_function_KExpanded' |
| node/test_attention_4d_causal/model.onnx | ✅ |  |
| node/test_attention_4d_causal_expanded/model.onnx | ❌ | Dynamic dim for tensor 'Attention_test_attention_4d_causal_expanded_function_RangeRow' |
| node/test_attention_4d_diff_heads_mask4d_padded_kv/model.onnx | ✅ |  |
| node/test_attention_4d_diff_heads_mask4d_padded_kv_expanded/model.onnx | ❌ | Dynamic dim for tensor 'Attention_test_attention_4d_diff_heads_mask4d_padded_kv_expanded_function_AttnBias' |
| node/test_attention_4d_diff_heads_sizes/model.onnx | ✅ |  |
| node/test_attention_4d_diff_heads_sizes_attn_mask/model.onnx | ✅ |  |
| node/test_attention_4d_diff_heads_sizes_attn_mask_expanded/model.onnx | ❌ | Dynamic dim for tensor 'Attention_test_attention_4d_diff_heads_sizes_attn_mask_expanded_function_KExpanded' |
| node/test_attention_4d_diff_heads_sizes_causal/model.onnx | ✅ |  |
| node/test_attention_4d_diff_heads_sizes_causal_expanded/model.onnx | ❌ | Dynamic dim for tensor 'Attention_test_attention_4d_diff_heads_sizes_causal_expanded_function_RangeRow' |
| node/test_attention_4d_diff_heads_sizes_expanded/model.onnx | ❌ | Dynamic dim for tensor 'Attention_test_attention_4d_diff_heads_sizes_expanded_function_KExpanded' |
| node/test_attention_4d_diff_heads_sizes_scaled/model.onnx | ✅ |  |
| node/test_attention_4d_diff_heads_sizes_scaled_expanded/model.onnx | ❌ | Dynamic dim for tensor 'Attention_test_attention_4d_diff_heads_sizes_scaled_expanded_function_KExpanded' |
| node/test_attention_4d_diff_heads_sizes_softcap/model.onnx | ✅ |  |
| node/test_attention_4d_diff_heads_sizes_softcap_expanded/model.onnx | ❌ | Dynamic dim for tensor 'Attention_test_attention_4d_diff_heads_sizes_softcap_expanded_function_KExpanded' |
| node/test_attention_4d_diff_heads_with_past_and_present/model.onnx | ✅ |  |
| node/test_attention_4d_diff_heads_with_past_and_present_expanded/model.onnx | ❌ | Dynamic dim for tensor 'Attention_test_attention_4d_diff_heads_with_past_and_present_expanded_function_KExpanded' |
| node/test_attention_4d_diff_heads_with_past_and_present_mask3d/model.onnx | ✅ |  |
| node/test_attention_4d_diff_heads_with_past_and_present_mask3d_expanded/model.onnx | ❌ | Dynamic dim for tensor 'Attention_test_attention_4d_diff_heads_with_past_and_present_mask3d_expanded_function_KExpanded' |
| node/test_attention_4d_diff_heads_with_past_and_present_mask4d/model.onnx | ✅ |  |
| node/test_attention_4d_diff_heads_with_past_and_present_mask4d_expanded/model.onnx | ❌ | Dynamic dim for tensor 'Attention_test_attention_4d_diff_heads_with_past_and_present_mask4d_expanded_function_KExpanded' |
| node/test_attention_4d_expanded/model.onnx | ❌ | Dynamic dim for tensor 'Attention_test_attention_4d_expanded_function_KExpanded' |
| node/test_attention_4d_fp16/model.onnx | ✅ |  |
| node/test_attention_4d_fp16_expanded/model.onnx | ❌ | Dynamic dim for tensor 'Attention_test_attention_4d_fp16_expanded_function_KExpanded' |
| node/test_attention_4d_gqa/model.onnx | ✅ |  |
| node/test_attention_4d_gqa_attn_mask/model.onnx | ✅ |  |
| node/test_attention_4d_gqa_attn_mask_expanded/model.onnx | ❌ | Dynamic dim for tensor 'Attention_test_attention_4d_gqa_attn_mask_expanded_function_KExpanded' |
| node/test_attention_4d_gqa_causal/model.onnx | ✅ |  |
| node/test_attention_4d_gqa_causal_expanded/model.onnx | ❌ | Dynamic dim for tensor 'Attention_test_attention_4d_gqa_causal_expanded_function_RangeRow' |
| node/test_attention_4d_gqa_expanded/model.onnx | ❌ | Dynamic dim for tensor 'Attention_test_attention_4d_gqa_expanded_function_KExpanded' |
| node/test_attention_4d_gqa_scaled/model.onnx | ✅ |  |
| node/test_attention_4d_gqa_scaled_expanded/model.onnx | ❌ | Dynamic dim for tensor 'Attention_test_attention_4d_gqa_scaled_expanded_function_KExpanded' |
| node/test_attention_4d_gqa_softcap/model.onnx | ✅ |  |
| node/test_attention_4d_gqa_softcap_expanded/model.onnx | ❌ | Dynamic dim for tensor 'Attention_test_attention_4d_gqa_softcap_expanded_function_KExpanded' |
| node/test_attention_4d_gqa_with_past_and_present/model.onnx | ✅ |  |
| node/test_attention_4d_gqa_with_past_and_present_expanded/model.onnx | ❌ | Dynamic dim for tensor 'Attention_test_attention_4d_gqa_with_past_and_present_expanded_function_KExpanded' |
| node/test_attention_4d_gqa_with_past_and_present_fp16/model.onnx | ✅ |  |
| node/test_attention_4d_gqa_with_past_and_present_fp16_expanded/model.onnx | ❌ | Dynamic dim for tensor 'Attention_test_attention_4d_gqa_with_past_and_present_fp16_expanded_function_KExpanded' |
| node/test_attention_4d_scaled/model.onnx | ✅ |  |
| node/test_attention_4d_scaled_expanded/model.onnx | ❌ | Dynamic dim for tensor 'Attention_test_attention_4d_scaled_expanded_function_KExpanded' |
| node/test_attention_4d_softcap/model.onnx | ✅ |  |
| node/test_attention_4d_softcap_expanded/model.onnx | ❌ | Dynamic dim for tensor 'Attention_test_attention_4d_softcap_expanded_function_KExpanded' |
| node/test_attention_4d_with_past_and_present/model.onnx | ✅ |  |
| node/test_attention_4d_with_past_and_present_expanded/model.onnx | ❌ | Dynamic dim for tensor 'Attention_test_attention_4d_with_past_and_present_expanded_function_KExpanded' |
| node/test_attention_4d_with_past_and_present_qk_matmul/model.onnx | ✅ |  |
| node/test_attention_4d_with_past_and_present_qk_matmul_bias/model.onnx | ✅ |  |
| node/test_attention_4d_with_past_and_present_qk_matmul_bias_3d_mask/model.onnx | ✅ |  |
| node/test_attention_4d_with_past_and_present_qk_matmul_bias_3d_mask_causal/model.onnx | ✅ |  |
| node/test_attention_4d_with_past_and_present_qk_matmul_bias_3d_mask_causal_expanded/model.onnx | ❌ | Dynamic dim for tensor 'Attention_test_attention_4d_with_past_and_present_qk_matmul_bias_3d_mask_causal_expanded_function_RangeRow' |
| node/test_attention_4d_with_past_and_present_qk_matmul_bias_3d_mask_expanded/model.onnx | ❌ | Dynamic dim for tensor 'Attention_test_attention_4d_with_past_and_present_qk_matmul_bias_3d_mask_expanded_function_KExpanded' |
| node/test_attention_4d_with_past_and_present_qk_matmul_bias_4d_mask/model.onnx | ✅ |  |
| node/test_attention_4d_with_past_and_present_qk_matmul_bias_4d_mask_causal/model.onnx | ✅ |  |
| node/test_attention_4d_with_past_and_present_qk_matmul_bias_4d_mask_causal_expanded/model.onnx | ❌ | Dynamic dim for tensor 'Attention_test_attention_4d_with_past_and_present_qk_matmul_bias_4d_mask_causal_expanded_function_RangeRow' |
| node/test_attention_4d_with_past_and_present_qk_matmul_bias_4d_mask_expanded/model.onnx | ❌ | Dynamic dim for tensor 'Attention_test_attention_4d_with_past_and_present_qk_matmul_bias_4d_mask_expanded_function_KExpanded' |
| node/test_attention_4d_with_past_and_present_qk_matmul_bias_expanded/model.onnx | ❌ | Dynamic dim for tensor 'Attention_test_attention_4d_with_past_and_present_qk_matmul_bias_expanded_function_KExpanded' |
| node/test_attention_4d_with_past_and_present_qk_matmul_expanded/model.onnx | ❌ | Dynamic dim for tensor 'Attention_test_attention_4d_with_past_and_present_qk_matmul_expanded_function_KExpanded' |
| node/test_attention_4d_with_qk_matmul/model.onnx | ✅ |  |
| node/test_attention_4d_with_qk_matmul_bias/model.onnx | ✅ |  |
| node/test_attention_4d_with_qk_matmul_bias_expanded/model.onnx | ❌ | Dynamic dim for tensor 'Attention_test_attention_4d_with_qk_matmul_bias_expanded_function_KExpanded' |
| node/test_attention_4d_with_qk_matmul_expanded/model.onnx | ❌ | Dynamic dim for tensor 'Attention_test_attention_4d_with_qk_matmul_expanded_function_KExpanded' |
| node/test_attention_4d_with_qk_matmul_softcap/model.onnx | ✅ |  |
| node/test_attention_4d_with_qk_matmul_softcap_expanded/model.onnx | ❌ | Dynamic dim for tensor 'Attention_test_attention_4d_with_qk_matmul_softcap_expanded_function_KExpanded' |
| node/test_attention_4d_with_qk_matmul_softmax/model.onnx | ✅ |  |
| node/test_attention_4d_with_qk_matmul_softmax_expanded/model.onnx | ❌ | Dynamic dim for tensor 'Attention_test_attention_4d_with_qk_matmul_softmax_expanded_function_KExpanded' |
| node/test_averagepool_1d_default/model.onnx | ❌ | AveragePool expects 2D kernel_shape |
| node/test_averagepool_2d_ceil/model.onnx | ❌ | AveragePool supports ceil_mode=0 only |
| node/test_averagepool_2d_ceil_last_window_starts_on_pad/model.onnx | ❌ | AveragePool supports ceil_mode=0 only |
| node/test_averagepool_2d_default/model.onnx | ✅ |  |
| node/test_averagepool_2d_dilations/model.onnx | ❌ | AveragePool has unsupported attributes |
| node/test_averagepool_2d_pads/model.onnx | ✅ |  |
| node/test_averagepool_2d_pads_count_include_pad/model.onnx | ✅ |  |
| node/test_averagepool_2d_precomputed_pads/model.onnx | ✅ |  |
| node/test_averagepool_2d_precomputed_pads_count_include_pad/model.onnx | ✅ |  |
| node/test_averagepool_2d_precomputed_same_upper/model.onnx | ❌ | AveragePool supports auto_pad=NOTSET only |
| node/test_averagepool_2d_precomputed_strides/model.onnx | ✅ |  |
| node/test_averagepool_2d_same_lower/model.onnx | ❌ | AveragePool supports auto_pad=NOTSET only |
| node/test_averagepool_2d_same_upper/model.onnx | ❌ | AveragePool supports auto_pad=NOTSET only |
| node/test_averagepool_2d_strides/model.onnx | ✅ |  |
| node/test_averagepool_3d_default/model.onnx | ❌ | AveragePool expects 2D kernel_shape |
| node/test_averagepool_3d_dilations_large_count_include_pad_is_0_ceil_mode_is_False/model.onnx | ❌ | AveragePool has unsupported attributes |
| node/test_averagepool_3d_dilations_large_count_include_pad_is_0_ceil_mode_is_True/model.onnx | ❌ | AveragePool has unsupported attributes |
| node/test_averagepool_3d_dilations_large_count_include_pad_is_1_ceil_mode_is_False/model.onnx | ❌ | AveragePool has unsupported attributes |
| node/test_averagepool_3d_dilations_large_count_include_pad_is_1_ceil_mode_is_True/model.onnx | ❌ | AveragePool has unsupported attributes |
| node/test_averagepool_3d_dilations_small/model.onnx | ❌ | AveragePool has unsupported attributes |
| node/test_basic_conv_with_padding/model.onnx | ✅ |  |
| node/test_basic_conv_without_padding/model.onnx | ✅ |  |
| node/test_basic_deform_conv_with_padding/model.onnx | ❌ | Unsupported op DeformConv |
| node/test_basic_deform_conv_without_padding/model.onnx | ❌ | Unsupported op DeformConv |
| node/test_batchnorm_epsilon/model.onnx | ✅ |  |
| node/test_batchnorm_epsilon_training_mode/model.onnx | ❌ | BatchNormalization must have 5 inputs and 1 output |
| node/test_batchnorm_example/model.onnx | ✅ |  |
| node/test_batchnorm_example_training_mode/model.onnx | ❌ | BatchNormalization must have 5 inputs and 1 output |
| node/test_bernoulli/model.onnx | ❌ | Unsupported op Bernoulli |
| node/test_bernoulli_double/model.onnx | ❌ | Unsupported op Bernoulli |
| node/test_bernoulli_double_expanded/model.onnx | ❌ | Unsupported op RandomUniformLike |
| node/test_bernoulli_expanded/model.onnx | ❌ | Unsupported op RandomUniformLike |
| node/test_bernoulli_seed/model.onnx | ❌ | Unsupported op Bernoulli |
| node/test_bernoulli_seed_expanded/model.onnx | ❌ | Unsupported op RandomUniformLike |
| node/test_bitshift_left_uint16/model.onnx | ✅ |  |
| node/test_bitshift_left_uint32/model.onnx | ✅ |  |
| node/test_bitshift_left_uint64/model.onnx | ✅ |  |
| node/test_bitshift_left_uint8/model.onnx | ✅ |  |
| node/test_bitshift_right_uint16/model.onnx | ✅ |  |
| node/test_bitshift_right_uint32/model.onnx | ✅ |  |
| node/test_bitshift_right_uint64/model.onnx | ✅ |  |
| node/test_bitshift_right_uint8/model.onnx | ✅ |  |
| node/test_bitwise_and_i16_3d/model.onnx | ✅ |  |
| node/test_bitwise_and_i32_2d/model.onnx | ✅ |  |
| node/test_bitwise_and_ui64_bcast_3v1d/model.onnx | ✅ |  |
| node/test_bitwise_and_ui8_bcast_4v3d/model.onnx | ✅ |  |
| node/test_bitwise_not_2d/model.onnx | ✅ |  |
| node/test_bitwise_not_3d/model.onnx | ❌ | Unsupported op BitwiseNot |
| node/test_bitwise_not_4d/model.onnx | ❌ | Unsupported op BitwiseNot |
| node/test_bitwise_or_i16_4d/model.onnx | ✅ |  |
| node/test_bitwise_or_i32_2d/model.onnx | ✅ |  |
| node/test_bitwise_or_ui64_bcast_3v1d/model.onnx | ✅ |  |
| node/test_bitwise_or_ui8_bcast_4v3d/model.onnx | ✅ |  |
| node/test_bitwise_xor_i16_3d/model.onnx | ✅ |  |
| node/test_bitwise_xor_i32_2d/model.onnx | ✅ |  |
| node/test_bitwise_xor_ui64_bcast_3v1d/model.onnx | ✅ |  |
| node/test_bitwise_xor_ui8_bcast_4v3d/model.onnx | ✅ |  |
| node/test_blackmanwindow/model.onnx | ❌ | Unsupported op BlackmanWindow |
| node/test_blackmanwindow_expanded/model.onnx | ❌ | Dynamic dim for tensor 'BlackmanWindow_test_blackmanwindow_expanded_function_Range' |
| node/test_blackmanwindow_symmetric/model.onnx | ❌ | Unsupported op BlackmanWindow |
| node/test_blackmanwindow_symmetric_expanded/model.onnx | ❌ | Dynamic dim for tensor 'BlackmanWindow_test_blackmanwindow_symmetric_expanded_function_Range' |
| node/test_cast_BFLOAT16_to_FLOAT/model.onnx | ❌ | Unsupported elem_type 16 (BFLOAT16) for tensor 'input'. |
| node/test_cast_DOUBLE_to_FLOAT/model.onnx | ✅ |  |
| node/test_cast_DOUBLE_to_FLOAT16/model.onnx | ✅ |  |
| node/test_cast_FLOAT16_to_DOUBLE/model.onnx | ✅ |  |
| node/test_cast_FLOAT16_to_FLOAT/model.onnx | ✅ |  |
| node/test_cast_FLOAT16_to_FLOAT4E2M1/model.onnx | ❌ | Unsupported elem_type 23 (FLOAT4E2M1) for tensor 'output'. |
| node/test_cast_FLOAT16_to_FLOAT8E4M3FN/model.onnx | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'output'. |
| node/test_cast_FLOAT16_to_FLOAT8E4M3FNUZ/model.onnx | ❌ | Unsupported elem_type 18 (FLOAT8E4M3FNUZ) for tensor 'output'. |
| node/test_cast_FLOAT16_to_FLOAT8E5M2/model.onnx | ❌ | Unsupported elem_type 19 (FLOAT8E5M2) for tensor 'output'. |
| node/test_cast_FLOAT16_to_FLOAT8E5M2FNUZ/model.onnx | ❌ | Unsupported elem_type 20 (FLOAT8E5M2FNUZ) for tensor 'output'. |
| node/test_cast_FLOAT16_to_INT2/model.onnx | ❌ | Unsupported elem_type 26 (INT2) for tensor 'output'. |
| node/test_cast_FLOAT16_to_INT4/model.onnx | ❌ | Unsupported elem_type 22 (INT4) for tensor 'output'. |
| node/test_cast_FLOAT16_to_UINT2/model.onnx | ❌ | Unsupported elem_type 25 (UINT2) for tensor 'output'. |
| node/test_cast_FLOAT16_to_UINT4/model.onnx | ❌ | Unsupported elem_type 21 (UINT4) for tensor 'output'. |
| node/test_cast_FLOAT4E2M1_to_FLOAT/model.onnx | ❌ | Unsupported elem_type 23 (FLOAT4E2M1) for tensor 'input'. |
| node/test_cast_FLOAT4E2M1_to_FLOAT16/model.onnx | ❌ | Unsupported elem_type 23 (FLOAT4E2M1) for tensor 'input'. |
| node/test_cast_FLOAT8E4M3FNUZ_to_FLOAT/model.onnx | ❌ | Unsupported elem_type 18 (FLOAT8E4M3FNUZ) for tensor 'input'. |
| node/test_cast_FLOAT8E4M3FNUZ_to_FLOAT16/model.onnx | ❌ | Unsupported elem_type 18 (FLOAT8E4M3FNUZ) for tensor 'input'. |
| node/test_cast_FLOAT8E4M3FN_to_FLOAT/model.onnx | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'input'. |
| node/test_cast_FLOAT8E4M3FN_to_FLOAT16/model.onnx | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'input'. |
| node/test_cast_FLOAT8E5M2FNUZ_to_FLOAT/model.onnx | ❌ | Unsupported elem_type 20 (FLOAT8E5M2FNUZ) for tensor 'input'. |
| node/test_cast_FLOAT8E5M2FNUZ_to_FLOAT16/model.onnx | ❌ | Unsupported elem_type 20 (FLOAT8E5M2FNUZ) for tensor 'input'. |
| node/test_cast_FLOAT8E5M2_to_FLOAT/model.onnx | ❌ | Unsupported elem_type 19 (FLOAT8E5M2) for tensor 'input'. |
| node/test_cast_FLOAT8E5M2_to_FLOAT16/model.onnx | ❌ | Unsupported elem_type 19 (FLOAT8E5M2) for tensor 'input'. |
| node/test_cast_FLOAT_to_BFLOAT16/model.onnx | ❌ | Unsupported elem_type 16 (BFLOAT16) for tensor 'output'. |
| node/test_cast_FLOAT_to_DOUBLE/model.onnx | ✅ |  |
| node/test_cast_FLOAT_to_FLOAT16/model.onnx | ✅ |  |
| node/test_cast_FLOAT_to_FLOAT4E2M1/model.onnx | ❌ | Unsupported elem_type 23 (FLOAT4E2M1) for tensor 'output'. |
| node/test_cast_FLOAT_to_FLOAT8E4M3FN/model.onnx | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'output'. |
| node/test_cast_FLOAT_to_FLOAT8E4M3FNUZ/model.onnx | ❌ | Unsupported elem_type 18 (FLOAT8E4M3FNUZ) for tensor 'output'. |
| node/test_cast_FLOAT_to_FLOAT8E5M2/model.onnx | ❌ | Unsupported elem_type 19 (FLOAT8E5M2) for tensor 'output'. |
| node/test_cast_FLOAT_to_FLOAT8E5M2FNUZ/model.onnx | ❌ | Unsupported elem_type 20 (FLOAT8E5M2FNUZ) for tensor 'output'. |
| node/test_cast_FLOAT_to_INT2/model.onnx | ❌ | Unsupported elem_type 26 (INT2) for tensor 'output'. |
| node/test_cast_FLOAT_to_INT4/model.onnx | ❌ | Unsupported elem_type 22 (INT4) for tensor 'output'. |
| node/test_cast_FLOAT_to_UINT2/model.onnx | ❌ | Unsupported elem_type 25 (UINT2) for tensor 'output'. |
| node/test_cast_FLOAT_to_UINT4/model.onnx | ❌ | Unsupported elem_type 21 (UINT4) for tensor 'output'. |
| node/test_cast_INT2_to_FLOAT/model.onnx | ❌ | Unsupported elem_type 26 (INT2) for tensor 'input'. |
| node/test_cast_INT2_to_FLOAT16/model.onnx | ❌ | Unsupported elem_type 26 (INT2) for tensor 'input'. |
| node/test_cast_INT2_to_INT8/model.onnx | ❌ | Unsupported elem_type 26 (INT2) for tensor 'input'. |
| node/test_cast_INT4_to_FLOAT/model.onnx | ❌ | Unsupported elem_type 22 (INT4) for tensor 'input'. |
| node/test_cast_INT4_to_FLOAT16/model.onnx | ❌ | Unsupported elem_type 22 (INT4) for tensor 'input'. |
| node/test_cast_INT4_to_INT8/model.onnx | ❌ | Unsupported elem_type 22 (INT4) for tensor 'input'. |
| node/test_cast_UINT2_to_FLOAT/model.onnx | ❌ | Unsupported elem_type 25 (UINT2) for tensor 'input'. |
| node/test_cast_UINT2_to_FLOAT16/model.onnx | ❌ | Unsupported elem_type 25 (UINT2) for tensor 'input'. |
| node/test_cast_UINT2_to_UINT8/model.onnx | ❌ | Unsupported elem_type 25 (UINT2) for tensor 'input'. |
| node/test_cast_UINT4_to_FLOAT/model.onnx | ❌ | Unsupported elem_type 21 (UINT4) for tensor 'input'. |
| node/test_cast_UINT4_to_FLOAT16/model.onnx | ❌ | Unsupported elem_type 21 (UINT4) for tensor 'input'. |
| node/test_cast_UINT4_to_UINT8/model.onnx | ❌ | Unsupported elem_type 21 (UINT4) for tensor 'input'. |
| node/test_cast_e8m0_FLOAT16_to_FLOAT8E8M0/model.onnx | ❌ | Unsupported elem_type 24 (FLOAT8E8M0) for tensor 'output'. |
| node/test_cast_e8m0_FLOAT8E8M0_to_FLOAT/model.onnx | ❌ | Unsupported elem_type 24 (FLOAT8E8M0) for tensor 'input'. |
| node/test_cast_e8m0_FLOAT8E8M0_to_FLOAT16/model.onnx | ❌ | Unsupported elem_type 24 (FLOAT8E8M0) for tensor 'input'. |
| node/test_cast_e8m0_FLOAT_to_FLOAT8E8M0/model.onnx | ❌ | Unsupported elem_type 24 (FLOAT8E8M0) for tensor 'output'. |
| node/test_cast_no_saturate_FLOAT16_to_FLOAT8E4M3FN/model.onnx | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'output'. |
| node/test_cast_no_saturate_FLOAT16_to_FLOAT8E4M3FNUZ/model.onnx | ❌ | Unsupported elem_type 18 (FLOAT8E4M3FNUZ) for tensor 'output'. |
| node/test_cast_no_saturate_FLOAT16_to_FLOAT8E5M2/model.onnx | ❌ | Unsupported elem_type 19 (FLOAT8E5M2) for tensor 'output'. |
| node/test_cast_no_saturate_FLOAT16_to_FLOAT8E5M2FNUZ/model.onnx | ❌ | Unsupported elem_type 20 (FLOAT8E5M2FNUZ) for tensor 'output'. |
| node/test_cast_no_saturate_FLOAT_to_FLOAT8E4M3FN/model.onnx | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'output'. |
| node/test_cast_no_saturate_FLOAT_to_FLOAT8E4M3FNUZ/model.onnx | ❌ | Unsupported elem_type 18 (FLOAT8E4M3FNUZ) for tensor 'output'. |
| node/test_cast_no_saturate_FLOAT_to_FLOAT8E5M2/model.onnx | ❌ | Unsupported elem_type 19 (FLOAT8E5M2) for tensor 'output'. |
| node/test_cast_no_saturate_FLOAT_to_FLOAT8E5M2FNUZ/model.onnx | ❌ | Unsupported elem_type 20 (FLOAT8E5M2FNUZ) for tensor 'output'. |
| node/test_castlike_BFLOAT16_to_FLOAT/model.onnx | ❌ | Unsupported elem_type 16 (BFLOAT16) for tensor 'input'. |
| node/test_castlike_BFLOAT16_to_FLOAT_expanded/model.onnx | ❌ | Unsupported elem_type 16 (BFLOAT16) for tensor 'input'. |
| node/test_castlike_DOUBLE_to_FLOAT/model.onnx | ✅ |  |
| node/test_castlike_DOUBLE_to_FLOAT16/model.onnx | ✅ |  |
| node/test_castlike_DOUBLE_to_FLOAT16_expanded/model.onnx | ✅ |  |
| node/test_castlike_DOUBLE_to_FLOAT_expanded/model.onnx | ✅ |  |
| node/test_castlike_FLOAT16_to_DOUBLE/model.onnx | ✅ |  |
| node/test_castlike_FLOAT16_to_DOUBLE_expanded/model.onnx | ✅ |  |
| node/test_castlike_FLOAT16_to_FLOAT/model.onnx | ✅ |  |
| node/test_castlike_FLOAT16_to_FLOAT4E2M1/model.onnx | ❌ | Unsupported elem_type 23 (FLOAT4E2M1) for tensor 'like'. |
| node/test_castlike_FLOAT16_to_FLOAT4E2M1_expanded/model.onnx | ❌ | Unsupported elem_type 23 (FLOAT4E2M1) for tensor 'like'. |
| node/test_castlike_FLOAT16_to_FLOAT8E4M3FN/model.onnx | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'like'. |
| node/test_castlike_FLOAT16_to_FLOAT8E4M3FNUZ/model.onnx | ❌ | Unsupported elem_type 18 (FLOAT8E4M3FNUZ) for tensor 'like'. |
| node/test_castlike_FLOAT16_to_FLOAT8E4M3FNUZ_expanded/model.onnx | ❌ | Unsupported elem_type 18 (FLOAT8E4M3FNUZ) for tensor 'like'. |
| node/test_castlike_FLOAT16_to_FLOAT8E4M3FN_expanded/model.onnx | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'like'. |
| node/test_castlike_FLOAT16_to_FLOAT8E5M2/model.onnx | ❌ | Unsupported elem_type 19 (FLOAT8E5M2) for tensor 'like'. |
| node/test_castlike_FLOAT16_to_FLOAT8E5M2FNUZ/model.onnx | ❌ | Unsupported elem_type 20 (FLOAT8E5M2FNUZ) for tensor 'like'. |
| node/test_castlike_FLOAT16_to_FLOAT8E5M2FNUZ_expanded/model.onnx | ❌ | Unsupported elem_type 20 (FLOAT8E5M2FNUZ) for tensor 'like'. |
| node/test_castlike_FLOAT16_to_FLOAT8E5M2_expanded/model.onnx | ❌ | Unsupported elem_type 19 (FLOAT8E5M2) for tensor 'like'. |
| node/test_castlike_FLOAT16_to_FLOAT_expanded/model.onnx | ✅ |  |
| node/test_castlike_FLOAT16_to_INT2/model.onnx | ❌ | Unsupported elem_type 26 (INT2) for tensor 'like'. |
| node/test_castlike_FLOAT16_to_INT2_expanded/model.onnx | ❌ | Unsupported elem_type 26 (INT2) for tensor 'like'. |
| node/test_castlike_FLOAT16_to_INT4/model.onnx | ❌ | Unsupported elem_type 22 (INT4) for tensor 'like'. |
| node/test_castlike_FLOAT16_to_INT4_expanded/model.onnx | ❌ | Unsupported elem_type 22 (INT4) for tensor 'like'. |
| node/test_castlike_FLOAT16_to_UINT2/model.onnx | ❌ | Unsupported elem_type 25 (UINT2) for tensor 'like'. |
| node/test_castlike_FLOAT16_to_UINT2_expanded/model.onnx | ❌ | Unsupported elem_type 25 (UINT2) for tensor 'like'. |
| node/test_castlike_FLOAT16_to_UINT4/model.onnx | ❌ | Unsupported elem_type 21 (UINT4) for tensor 'like'. |
| node/test_castlike_FLOAT16_to_UINT4_expanded/model.onnx | ❌ | Unsupported elem_type 21 (UINT4) for tensor 'like'. |
| node/test_castlike_FLOAT4E2M1_to_FLOAT/model.onnx | ❌ | Unsupported elem_type 23 (FLOAT4E2M1) for tensor 'input'. |
| node/test_castlike_FLOAT4E2M1_to_FLOAT16/model.onnx | ❌ | Unsupported elem_type 23 (FLOAT4E2M1) for tensor 'input'. |
| node/test_castlike_FLOAT4E2M1_to_FLOAT16_expanded/model.onnx | ❌ | Unsupported elem_type 23 (FLOAT4E2M1) for tensor 'input'. |
| node/test_castlike_FLOAT4E2M1_to_FLOAT_expanded/model.onnx | ❌ | Unsupported elem_type 23 (FLOAT4E2M1) for tensor 'input'. |
| node/test_castlike_FLOAT8E4M3FNUZ_to_FLOAT/model.onnx | ❌ | Unsupported elem_type 18 (FLOAT8E4M3FNUZ) for tensor 'input'. |
| node/test_castlike_FLOAT8E4M3FNUZ_to_FLOAT16/model.onnx | ❌ | Unsupported elem_type 18 (FLOAT8E4M3FNUZ) for tensor 'input'. |
| node/test_castlike_FLOAT8E4M3FNUZ_to_FLOAT16_expanded/model.onnx | ❌ | Unsupported elem_type 18 (FLOAT8E4M3FNUZ) for tensor 'input'. |
| node/test_castlike_FLOAT8E4M3FNUZ_to_FLOAT_expanded/model.onnx | ❌ | Unsupported elem_type 18 (FLOAT8E4M3FNUZ) for tensor 'input'. |
| node/test_castlike_FLOAT8E4M3FN_to_FLOAT/model.onnx | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'input'. |
| node/test_castlike_FLOAT8E4M3FN_to_FLOAT16/model.onnx | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'input'. |
| node/test_castlike_FLOAT8E4M3FN_to_FLOAT16_expanded/model.onnx | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'input'. |
| node/test_castlike_FLOAT8E4M3FN_to_FLOAT_expanded/model.onnx | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'input'. |
| node/test_castlike_FLOAT8E5M2FNUZ_to_FLOAT/model.onnx | ❌ | Unsupported elem_type 20 (FLOAT8E5M2FNUZ) for tensor 'input'. |
| node/test_castlike_FLOAT8E5M2FNUZ_to_FLOAT16/model.onnx | ❌ | Unsupported elem_type 20 (FLOAT8E5M2FNUZ) for tensor 'input'. |
| node/test_castlike_FLOAT8E5M2FNUZ_to_FLOAT16_expanded/model.onnx | ❌ | Unsupported elem_type 20 (FLOAT8E5M2FNUZ) for tensor 'input'. |
| node/test_castlike_FLOAT8E5M2FNUZ_to_FLOAT_expanded/model.onnx | ❌ | Unsupported elem_type 20 (FLOAT8E5M2FNUZ) for tensor 'input'. |
| node/test_castlike_FLOAT8E5M2_to_FLOAT/model.onnx | ❌ | Unsupported elem_type 19 (FLOAT8E5M2) for tensor 'input'. |
| node/test_castlike_FLOAT8E5M2_to_FLOAT16/model.onnx | ❌ | Unsupported elem_type 19 (FLOAT8E5M2) for tensor 'input'. |
| node/test_castlike_FLOAT8E5M2_to_FLOAT16_expanded/model.onnx | ❌ | Unsupported elem_type 19 (FLOAT8E5M2) for tensor 'input'. |
| node/test_castlike_FLOAT8E5M2_to_FLOAT_expanded/model.onnx | ❌ | Unsupported elem_type 19 (FLOAT8E5M2) for tensor 'input'. |
| node/test_castlike_FLOAT_to_BFLOAT16/model.onnx | ❌ | Unsupported elem_type 16 (BFLOAT16) for tensor 'like'. |
| node/test_castlike_FLOAT_to_BFLOAT16_expanded/model.onnx | ❌ | Unsupported elem_type 16 (BFLOAT16) for tensor 'like'. |
| node/test_castlike_FLOAT_to_DOUBLE/model.onnx | ✅ |  |
| node/test_castlike_FLOAT_to_DOUBLE_expanded/model.onnx | ✅ |  |
| node/test_castlike_FLOAT_to_FLOAT16/model.onnx | ✅ |  |
| node/test_castlike_FLOAT_to_FLOAT16_expanded/model.onnx | ✅ |  |
| node/test_castlike_FLOAT_to_FLOAT4E2M1/model.onnx | ❌ | Unsupported elem_type 23 (FLOAT4E2M1) for tensor 'like'. |
| node/test_castlike_FLOAT_to_FLOAT4E2M1_expanded/model.onnx | ❌ | Unsupported elem_type 23 (FLOAT4E2M1) for tensor 'like'. |
| node/test_castlike_FLOAT_to_FLOAT8E4M3FN/model.onnx | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'like'. |
| node/test_castlike_FLOAT_to_FLOAT8E4M3FNUZ/model.onnx | ❌ | Unsupported elem_type 18 (FLOAT8E4M3FNUZ) for tensor 'like'. |
| node/test_castlike_FLOAT_to_FLOAT8E4M3FNUZ_expanded/model.onnx | ❌ | Unsupported elem_type 18 (FLOAT8E4M3FNUZ) for tensor 'like'. |
| node/test_castlike_FLOAT_to_FLOAT8E4M3FN_expanded/model.onnx | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'like'. |
| node/test_castlike_FLOAT_to_FLOAT8E5M2/model.onnx | ❌ | Unsupported elem_type 19 (FLOAT8E5M2) for tensor 'like'. |
| node/test_castlike_FLOAT_to_FLOAT8E5M2FNUZ/model.onnx | ❌ | Unsupported elem_type 20 (FLOAT8E5M2FNUZ) for tensor 'like'. |
| node/test_castlike_FLOAT_to_FLOAT8E5M2FNUZ_expanded/model.onnx | ❌ | Unsupported elem_type 20 (FLOAT8E5M2FNUZ) for tensor 'like'. |
| node/test_castlike_FLOAT_to_FLOAT8E5M2_expanded/model.onnx | ❌ | Unsupported elem_type 19 (FLOAT8E5M2) for tensor 'like'. |
| node/test_castlike_FLOAT_to_INT2/model.onnx | ❌ | Unsupported elem_type 26 (INT2) for tensor 'like'. |
| node/test_castlike_FLOAT_to_INT2_expanded/model.onnx | ❌ | Unsupported elem_type 26 (INT2) for tensor 'like'. |
| node/test_castlike_FLOAT_to_INT4/model.onnx | ❌ | Unsupported elem_type 22 (INT4) for tensor 'like'. |
| node/test_castlike_FLOAT_to_INT4_expanded/model.onnx | ❌ | Unsupported elem_type 22 (INT4) for tensor 'like'. |
| node/test_castlike_FLOAT_to_UINT2/model.onnx | ❌ | Unsupported elem_type 25 (UINT2) for tensor 'like'. |
| node/test_castlike_FLOAT_to_UINT2_expanded/model.onnx | ❌ | Unsupported elem_type 25 (UINT2) for tensor 'like'. |
| node/test_castlike_FLOAT_to_UINT4/model.onnx | ❌ | Unsupported elem_type 21 (UINT4) for tensor 'like'. |
| node/test_castlike_FLOAT_to_UINT4_expanded/model.onnx | ❌ | Unsupported elem_type 21 (UINT4) for tensor 'like'. |
| node/test_castlike_INT2_to_FLOAT/model.onnx | ❌ | Unsupported elem_type 26 (INT2) for tensor 'input'. |
| node/test_castlike_INT2_to_FLOAT16/model.onnx | ❌ | Unsupported elem_type 26 (INT2) for tensor 'input'. |
| node/test_castlike_INT2_to_FLOAT16_expanded/model.onnx | ❌ | Unsupported elem_type 26 (INT2) for tensor 'input'. |
| node/test_castlike_INT2_to_FLOAT_expanded/model.onnx | ❌ | Unsupported elem_type 26 (INT2) for tensor 'input'. |
| node/test_castlike_INT2_to_INT8/model.onnx | ❌ | Unsupported elem_type 26 (INT2) for tensor 'input'. |
| node/test_castlike_INT2_to_INT8_expanded/model.onnx | ❌ | Unsupported elem_type 26 (INT2) for tensor 'input'. |
| node/test_castlike_INT4_to_FLOAT/model.onnx | ❌ | Unsupported elem_type 22 (INT4) for tensor 'input'. |
| node/test_castlike_INT4_to_FLOAT16/model.onnx | ❌ | Unsupported elem_type 22 (INT4) for tensor 'input'. |
| node/test_castlike_INT4_to_FLOAT16_expanded/model.onnx | ❌ | Unsupported elem_type 22 (INT4) for tensor 'input'. |
| node/test_castlike_INT4_to_FLOAT_expanded/model.onnx | ❌ | Unsupported elem_type 22 (INT4) for tensor 'input'. |
| node/test_castlike_INT4_to_INT8/model.onnx | ❌ | Unsupported elem_type 22 (INT4) for tensor 'input'. |
| node/test_castlike_INT4_to_INT8_expanded/model.onnx | ❌ | Unsupported elem_type 22 (INT4) for tensor 'input'. |
| node/test_castlike_UINT2_to_FLOAT/model.onnx | ❌ | Unsupported elem_type 25 (UINT2) for tensor 'input'. |
| node/test_castlike_UINT2_to_FLOAT16/model.onnx | ❌ | Unsupported elem_type 25 (UINT2) for tensor 'input'. |
| node/test_castlike_UINT2_to_FLOAT16_expanded/model.onnx | ❌ | Unsupported elem_type 25 (UINT2) for tensor 'input'. |
| node/test_castlike_UINT2_to_FLOAT_expanded/model.onnx | ❌ | Unsupported elem_type 25 (UINT2) for tensor 'input'. |
| node/test_castlike_UINT2_to_UINT8/model.onnx | ❌ | Unsupported elem_type 25 (UINT2) for tensor 'input'. |
| node/test_castlike_UINT2_to_UINT8_expanded/model.onnx | ❌ | Unsupported elem_type 25 (UINT2) for tensor 'input'. |
| node/test_castlike_UINT4_to_FLOAT/model.onnx | ❌ | Unsupported elem_type 21 (UINT4) for tensor 'input'. |
| node/test_castlike_UINT4_to_FLOAT16/model.onnx | ❌ | Unsupported elem_type 21 (UINT4) for tensor 'input'. |
| node/test_castlike_UINT4_to_FLOAT16_expanded/model.onnx | ❌ | Unsupported elem_type 21 (UINT4) for tensor 'input'. |
| node/test_castlike_UINT4_to_FLOAT_expanded/model.onnx | ❌ | Unsupported elem_type 21 (UINT4) for tensor 'input'. |
| node/test_castlike_UINT4_to_UINT8/model.onnx | ❌ | Unsupported elem_type 21 (UINT4) for tensor 'input'. |
| node/test_castlike_UINT4_to_UINT8_expanded/model.onnx | ❌ | Unsupported elem_type 21 (UINT4) for tensor 'input'. |
| node/test_castlike_no_saturate_FLOAT16_to_FLOAT8E4M3FN/model.onnx | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'like'. |
| node/test_castlike_no_saturate_FLOAT16_to_FLOAT8E4M3FNUZ/model.onnx | ❌ | Unsupported elem_type 18 (FLOAT8E4M3FNUZ) for tensor 'like'. |
| node/test_castlike_no_saturate_FLOAT16_to_FLOAT8E4M3FNUZ_expanded/model.onnx | ❌ | Unsupported elem_type 18 (FLOAT8E4M3FNUZ) for tensor 'like'. |
| node/test_castlike_no_saturate_FLOAT16_to_FLOAT8E4M3FN_expanded/model.onnx | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'like'. |
| node/test_castlike_no_saturate_FLOAT16_to_FLOAT8E5M2/model.onnx | ❌ | Unsupported elem_type 19 (FLOAT8E5M2) for tensor 'like'. |
| node/test_castlike_no_saturate_FLOAT16_to_FLOAT8E5M2FNUZ/model.onnx | ❌ | Unsupported elem_type 20 (FLOAT8E5M2FNUZ) for tensor 'like'. |
| node/test_castlike_no_saturate_FLOAT16_to_FLOAT8E5M2FNUZ_expanded/model.onnx | ❌ | Unsupported elem_type 20 (FLOAT8E5M2FNUZ) for tensor 'like'. |
| node/test_castlike_no_saturate_FLOAT16_to_FLOAT8E5M2_expanded/model.onnx | ❌ | Unsupported elem_type 19 (FLOAT8E5M2) for tensor 'like'. |
| node/test_castlike_no_saturate_FLOAT_to_FLOAT8E4M3FN/model.onnx | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'like'. |
| node/test_castlike_no_saturate_FLOAT_to_FLOAT8E4M3FNUZ/model.onnx | ❌ | Unsupported elem_type 18 (FLOAT8E4M3FNUZ) for tensor 'like'. |
| node/test_castlike_no_saturate_FLOAT_to_FLOAT8E4M3FNUZ_expanded/model.onnx | ❌ | Unsupported elem_type 18 (FLOAT8E4M3FNUZ) for tensor 'like'. |
| node/test_castlike_no_saturate_FLOAT_to_FLOAT8E4M3FN_expanded/model.onnx | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'like'. |
| node/test_castlike_no_saturate_FLOAT_to_FLOAT8E5M2/model.onnx | ❌ | Unsupported elem_type 19 (FLOAT8E5M2) for tensor 'like'. |
| node/test_castlike_no_saturate_FLOAT_to_FLOAT8E5M2FNUZ/model.onnx | ❌ | Unsupported elem_type 20 (FLOAT8E5M2FNUZ) for tensor 'like'. |
| node/test_castlike_no_saturate_FLOAT_to_FLOAT8E5M2FNUZ_expanded/model.onnx | ❌ | Unsupported elem_type 20 (FLOAT8E5M2FNUZ) for tensor 'like'. |
| node/test_castlike_no_saturate_FLOAT_to_FLOAT8E5M2_expanded/model.onnx | ❌ | Unsupported elem_type 19 (FLOAT8E5M2) for tensor 'like'. |
| node/test_ceil/model.onnx | ✅ |  |
| node/test_ceil_example/model.onnx | ✅ |  |
| node/test_celu/model.onnx | ✅ |  |
| node/test_celu_expanded/model.onnx | ✅ |  |
| node/test_center_crop_pad_crop/model.onnx | ❌ | Unsupported op CenterCropPad |
| node/test_center_crop_pad_crop_and_pad/model.onnx | ❌ | Unsupported op CenterCropPad |
| node/test_center_crop_pad_crop_and_pad_expanded/model.onnx | ❌ | Dynamic dim for tensor 'CenterCropPad_test_center_crop_pad_crop_and_pad_expanded_function_padded_input' |
| node/test_center_crop_pad_crop_axes_chw/model.onnx | ❌ | Unsupported op CenterCropPad |
| node/test_center_crop_pad_crop_axes_chw_expanded/model.onnx | ❌ | Dynamic dim for tensor 'CenterCropPad_test_center_crop_pad_crop_axes_chw_expanded_function_padded_input' |
| node/test_center_crop_pad_crop_axes_hwc/model.onnx | ❌ | Unsupported op CenterCropPad |
| node/test_center_crop_pad_crop_axes_hwc_expanded/model.onnx | ❌ | Dynamic dim for tensor 'CenterCropPad_test_center_crop_pad_crop_axes_hwc_expanded_function_padded_input' |
| node/test_center_crop_pad_crop_expanded/model.onnx | ❌ | Dynamic dim for tensor 'CenterCropPad_test_center_crop_pad_crop_expanded_function_padded_input' |
| node/test_center_crop_pad_crop_negative_axes_hwc/model.onnx | ❌ | Unsupported op CenterCropPad |
| node/test_center_crop_pad_crop_negative_axes_hwc_expanded/model.onnx | ❌ | Dynamic dim for tensor 'CenterCropPad_test_center_crop_pad_crop_negative_axes_hwc_expanded_function_padded_input' |
| node/test_center_crop_pad_pad/model.onnx | ❌ | Unsupported op CenterCropPad |
| node/test_center_crop_pad_pad_expanded/model.onnx | ❌ | Dynamic dim for tensor 'CenterCropPad_test_center_crop_pad_pad_expanded_function_padded_input' |
| node/test_clip/model.onnx | ✅ |  |
| node/test_clip_default_inbounds/model.onnx | ✅ |  |
| node/test_clip_default_inbounds_expanded/model.onnx | ✅ |  |
| node/test_clip_default_int8_inbounds/model.onnx | ✅ |  |
| node/test_clip_default_int8_inbounds_expanded/model.onnx | ✅ |  |
| node/test_clip_default_int8_max/model.onnx | ✅ |  |
| node/test_clip_default_int8_max_expanded/model.onnx | ✅ |  |
| node/test_clip_default_int8_min/model.onnx | ✅ |  |
| node/test_clip_default_int8_min_expanded/model.onnx | ✅ |  |
| node/test_clip_default_max/model.onnx | ✅ |  |
| node/test_clip_default_max_expanded/model.onnx | ✅ |  |
| node/test_clip_default_min/model.onnx | ✅ |  |
| node/test_clip_default_min_expanded/model.onnx | ✅ |  |
| node/test_clip_example/model.onnx | ✅ |  |
| node/test_clip_example_expanded/model.onnx | ✅ |  |
| node/test_clip_expanded/model.onnx | ✅ |  |
| node/test_clip_inbounds/model.onnx | ✅ |  |
| node/test_clip_inbounds_expanded/model.onnx | ✅ |  |
| node/test_clip_min_greater_than_max/model.onnx | ✅ |  |
| node/test_clip_min_greater_than_max_expanded/model.onnx | ✅ |  |
| node/test_clip_outbounds/model.onnx | ✅ |  |
| node/test_clip_outbounds_expanded/model.onnx | ✅ |  |
| node/test_clip_splitbounds/model.onnx | ✅ |  |
| node/test_clip_splitbounds_expanded/model.onnx | ✅ |  |
| node/test_col2im/model.onnx | ❌ | Unsupported op Col2Im |
| node/test_col2im_5d/model.onnx | ❌ | Unsupported op Col2Im |
| node/test_col2im_dilations/model.onnx | ❌ | Unsupported op Col2Im |
| node/test_col2im_pads/model.onnx | ❌ | Unsupported op Col2Im |
| node/test_col2im_strides/model.onnx | ❌ | Unsupported op Col2Im |
| node/test_compress_0/model.onnx | ❌ | Unsupported op Compress |
| node/test_compress_1/model.onnx | ❌ | Unsupported op Compress |
| node/test_compress_default_axis/model.onnx | ❌ | Unsupported op Compress |
| node/test_compress_negative_axis/model.onnx | ❌ | Unsupported op Compress |
| node/test_concat_1d_axis_0/model.onnx | ✅ |  |
| node/test_concat_1d_axis_negative_1/model.onnx | ✅ |  |
| node/test_concat_2d_axis_0/model.onnx | ✅ |  |
| node/test_concat_2d_axis_1/model.onnx | ✅ |  |
| node/test_concat_2d_axis_negative_1/model.onnx | ✅ |  |
| node/test_concat_2d_axis_negative_2/model.onnx | ✅ |  |
| node/test_concat_3d_axis_0/model.onnx | ✅ |  |
| node/test_concat_3d_axis_1/model.onnx | ✅ |  |
| node/test_concat_3d_axis_2/model.onnx | ✅ |  |
| node/test_concat_3d_axis_negative_1/model.onnx | ✅ |  |
| node/test_concat_3d_axis_negative_2/model.onnx | ✅ |  |
| node/test_concat_3d_axis_negative_3/model.onnx | ✅ |  |
| node/test_constant/model.onnx | ❌ | Graph must contain at least one node |
| node/test_constant_pad/model.onnx | ❌ | Unsupported op Pad |
| node/test_constant_pad_axes/model.onnx | ❌ | Unsupported op Pad |
| node/test_constant_pad_negative_axes/model.onnx | ❌ | Unsupported op Pad |
| node/test_constantofshape_float_ones/model.onnx | ✅ |  |
| node/test_constantofshape_int_shape_zero/model.onnx | ❌ | Dynamic or zero dims are not supported |
| node/test_constantofshape_int_zeros/model.onnx | ✅ |  |
| node/test_conv_with_autopad_same/model.onnx | ✅ |  |
| node/test_conv_with_strides_and_asymmetric_padding/model.onnx | ✅ |  |
| node/test_conv_with_strides_no_padding/model.onnx | ✅ |  |
| node/test_conv_with_strides_padding/model.onnx | ✅ |  |
| node/test_convinteger_with_padding/model.onnx | ❌ | Unsupported op ConvInteger |
| node/test_convinteger_without_padding/model.onnx | ❌ | Unsupported op ConvInteger |
| node/test_convtranspose/model.onnx | ❌ | Unsupported op ConvTranspose |
| node/test_convtranspose_1d/model.onnx | ❌ | Unsupported op ConvTranspose |
| node/test_convtranspose_3d/model.onnx | ❌ | Unsupported op ConvTranspose |
| node/test_convtranspose_autopad_same/model.onnx | ❌ | Unsupported op ConvTranspose |
| node/test_convtranspose_dilations/model.onnx | ❌ | Unsupported op ConvTranspose |
| node/test_convtranspose_group_2/model.onnx | ❌ | Unsupported op ConvTranspose |
| node/test_convtranspose_group_2_image_3/model.onnx | ❌ | Unsupported op ConvTranspose |
| node/test_convtranspose_kernel_shape/model.onnx | ❌ | Unsupported op ConvTranspose |
| node/test_convtranspose_output_shape/model.onnx | ❌ | Unsupported op ConvTranspose |
| node/test_convtranspose_pad/model.onnx | ❌ | Unsupported op ConvTranspose |
| node/test_convtranspose_pads/model.onnx | ❌ | Unsupported op ConvTranspose |
| node/test_cos/model.onnx | ✅ |  |
| node/test_cos_example/model.onnx | ✅ |  |
| node/test_cosh/model.onnx | ✅ |  |
| node/test_cosh_example/model.onnx | ✅ |  |
| node/test_cumsum_1d/model.onnx | ❌ | Unsupported op CumSum |
| node/test_cumsum_1d_exclusive/model.onnx | ❌ | Unsupported op CumSum |
| node/test_cumsum_1d_int32_exclusive/model.onnx | ❌ | Unsupported op CumSum |
| node/test_cumsum_1d_reverse/model.onnx | ❌ | Unsupported op CumSum |
| node/test_cumsum_1d_reverse_exclusive/model.onnx | ❌ | Unsupported op CumSum |
| node/test_cumsum_2d_axis_0/model.onnx | ❌ | Unsupported op CumSum |
| node/test_cumsum_2d_axis_1/model.onnx | ❌ | Unsupported op CumSum |
| node/test_cumsum_2d_int32/model.onnx | ❌ | Unsupported op CumSum |
| node/test_cumsum_2d_negative_axis/model.onnx | ❌ | Unsupported op CumSum |
| node/test_deform_conv_with_mask_bias/model.onnx | ❌ | Unsupported op DeformConv |
| node/test_deform_conv_with_multiple_offset_groups/model.onnx | ❌ | Unsupported op DeformConv |
| node/test_depthtospace_crd_mode_example/model.onnx | ❌ | Unsupported op DepthToSpace |
| node/test_depthtospace_example/model.onnx | ❌ | Unsupported op DepthToSpace |
| node/test_dequantizelinear/model.onnx | ❌ | Unsupported op DequantizeLinear |
| node/test_dequantizelinear_axis/model.onnx | ❌ | Unsupported op DequantizeLinear |
| node/test_dequantizelinear_blocked/model.onnx | ❌ | Unsupported op DequantizeLinear |
| node/test_dequantizelinear_e4m3fn/model.onnx | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'x'. |
| node/test_dequantizelinear_e4m3fn_float16/model.onnx | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'x'. |
| node/test_dequantizelinear_e4m3fn_zero_point/model.onnx | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'x'. |
| node/test_dequantizelinear_e5m2/model.onnx | ❌ | Unsupported elem_type 19 (FLOAT8E5M2) for tensor 'x'. |
| node/test_dequantizelinear_float4e2m1/model.onnx | ❌ | Unsupported elem_type 23 (FLOAT4E2M1) for tensor 'x'. |
| node/test_dequantizelinear_int16/model.onnx | ❌ | Unsupported op DequantizeLinear |
| node/test_dequantizelinear_int2/model.onnx | ❌ | Unsupported elem_type 26 (INT2) for tensor 'x'. |
| node/test_dequantizelinear_int4/model.onnx | ❌ | Unsupported elem_type 22 (INT4) for tensor 'x'. |
| node/test_dequantizelinear_uint16/model.onnx | ❌ | Unsupported op DequantizeLinear |
| node/test_dequantizelinear_uint2/model.onnx | ❌ | Unsupported elem_type 25 (UINT2) for tensor 'x'. |
| node/test_dequantizelinear_uint4/model.onnx | ❌ | Unsupported elem_type 21 (UINT4) for tensor 'x'. |
| node/test_det_2d/model.onnx | ❌ | Unsupported op Det |
| node/test_det_nd/model.onnx | ❌ | Unsupported op Det |
| node/test_dft/model.onnx | ❌ | Unsupported op DFT |
| node/test_dft_axis/model.onnx | ❌ | Unsupported op DFT |
| node/test_dft_axis_opset19/model.onnx | ❌ | Unsupported op DFT |
| node/test_dft_inverse/model.onnx | ❌ | Unsupported op DFT |
| node/test_dft_inverse_opset19/model.onnx | ❌ | Unsupported op DFT |
| node/test_dft_opset19/model.onnx | ❌ | Unsupported op DFT |
| node/test_div/model.onnx | ✅ |  |
| node/test_div_bcast/model.onnx | ✅ |  |
| node/test_div_example/model.onnx | ✅ |  |
| node/test_div_int16/model.onnx | ✅ |  |
| node/test_div_int8/model.onnx | ✅ |  |
| node/test_div_uint16/model.onnx | ✅ |  |
| node/test_div_uint32/model.onnx | ✅ |  |
| node/test_div_uint64/model.onnx | ✅ |  |
| node/test_div_uint8/model.onnx | ✅ |  |
| node/test_dropout_default/model.onnx | ✅ |  |
| node/test_dropout_default_mask/model.onnx | ❌ | Dropout mask output is not supported |
| node/test_dropout_default_mask_ratio/model.onnx | ❌ | Dropout supports only the data input and 1 or 2 outputs |
| node/test_dropout_default_old/model.onnx | ✅ |  |
| node/test_dropout_default_ratio/model.onnx | ❌ | Dropout supports only the data input and 1 or 2 outputs |
| node/test_dropout_random_old/model.onnx | ✅ |  |
| node/test_dynamicquantizelinear/model.onnx | ❌ | Unsupported op DynamicQuantizeLinear |
| node/test_dynamicquantizelinear_expanded/model.onnx | ❌ | Unsupported op Identity |
| node/test_dynamicquantizelinear_max_adjusted/model.onnx | ❌ | Unsupported op DynamicQuantizeLinear |
| node/test_dynamicquantizelinear_max_adjusted_expanded/model.onnx | ❌ | Unsupported op Identity |
| node/test_dynamicquantizelinear_min_adjusted/model.onnx | ❌ | Unsupported op DynamicQuantizeLinear |
| node/test_dynamicquantizelinear_min_adjusted_expanded/model.onnx | ❌ | Unsupported op Identity |
| node/test_edge_pad/model.onnx | ❌ | Unsupported op Pad |
| node/test_einsum_batch_diagonal/model.onnx | ❌ | Unsupported op Einsum |
| node/test_einsum_batch_matmul/model.onnx | ❌ | Unsupported op Einsum |
| node/test_einsum_inner_prod/model.onnx | ❌ | Unsupported op Einsum |
| node/test_einsum_scalar/model.onnx | ❌ | Unsupported op Einsum |
| node/test_einsum_sum/model.onnx | ❌ | Unsupported op Einsum |
| node/test_einsum_transpose/model.onnx | ❌ | Unsupported op Einsum |
| node/test_elu/model.onnx | ❌ | Elu only supports alpha=1.0 |
| node/test_elu_default/model.onnx | ✅ |  |
| node/test_elu_default_expanded_ver18/model.onnx | ✅ |  |
| node/test_elu_example/model.onnx | ❌ | Elu only supports alpha=1.0 |
| node/test_elu_example_expanded_ver18/model.onnx | ✅ |  |
| node/test_elu_expanded_ver18/model.onnx | ✅ |  |
| node/test_equal/model.onnx | ✅ |  |
| node/test_equal_bcast/model.onnx | ✅ |  |
| node/test_equal_int16/model.onnx | ✅ |  |
| node/test_equal_int8/model.onnx | ✅ |  |
| node/test_equal_string/model.onnx | ❌ | Unsupported elem_type 8 (STRING) for tensor 'x'. |
| node/test_equal_string_broadcast/model.onnx | ❌ | Unsupported elem_type 8 (STRING) for tensor 'x'. |
| node/test_equal_uint16/model.onnx | ✅ |  |
| node/test_equal_uint32/model.onnx | ✅ |  |
| node/test_equal_uint64/model.onnx | ✅ |  |
| node/test_equal_uint8/model.onnx | ✅ |  |
| node/test_erf/model.onnx | ✅ |  |
| node/test_exp/model.onnx | ✅ |  |
| node/test_exp_example/model.onnx | ✅ |  |
| node/test_expand_dim_changed/model.onnx | ❌ | Unsupported op Expand |
| node/test_expand_dim_unchanged/model.onnx | ❌ | Unsupported op Expand |
| node/test_eyelike_populate_off_main_diagonal/model.onnx | ❌ | Unsupported op EyeLike |
| node/test_eyelike_with_dtype/model.onnx | ❌ | Unsupported op EyeLike |
| node/test_eyelike_without_dtype/model.onnx | ❌ | Unsupported op EyeLike |
| node/test_flatten_axis0/model.onnx | ✅ |  |
| node/test_flatten_axis1/model.onnx | ✅ |  |
| node/test_flatten_axis2/model.onnx | ✅ |  |
| node/test_flatten_axis3/model.onnx | ✅ |  |
| node/test_flatten_default_axis/model.onnx | ✅ |  |
| node/test_flatten_negative_axis1/model.onnx | ✅ |  |
| node/test_flatten_negative_axis2/model.onnx | ✅ |  |
| node/test_flatten_negative_axis3/model.onnx | ✅ |  |
| node/test_flatten_negative_axis4/model.onnx | ✅ |  |
| node/test_floor/model.onnx | ✅ |  |
| node/test_floor_example/model.onnx | ✅ |  |
| node/test_gather_0/model.onnx | ✅ |  |
| node/test_gather_1/model.onnx | ✅ |  |
| node/test_gather_2d_indices/model.onnx | ✅ |  |
| node/test_gather_elements_0/model.onnx | ✅ |  |
| node/test_gather_elements_1/model.onnx | ✅ |  |
| node/test_gather_elements_negative_indices/model.onnx | ✅ |  |
| node/test_gather_negative_indices/model.onnx | ✅ |  |
| node/test_gathernd_example_float32/model.onnx | ❌ | Unsupported op GatherND |
| node/test_gathernd_example_int32/model.onnx | ❌ | Unsupported op GatherND |
| node/test_gathernd_example_int32_batch_dim1/model.onnx | ❌ | Unsupported op GatherND |
| node/test_gelu_default_1/model.onnx | ✅ |  |
| node/test_gelu_default_1_expanded/model.onnx | ✅ |  |
| node/test_gelu_default_2/model.onnx | ✅ |  |
| node/test_gelu_default_2_expanded/model.onnx | ✅ |  |
| node/test_gelu_tanh_1/model.onnx | ❌ | Gelu only supports approximate=none |
| node/test_gelu_tanh_1_expanded/model.onnx | ✅ |  |
| node/test_gelu_tanh_2/model.onnx | ❌ | Gelu only supports approximate=none |
| node/test_gelu_tanh_2_expanded/model.onnx | ✅ |  |
| node/test_gemm_all_attributes/model.onnx | ✅ |  |
| node/test_gemm_alpha/model.onnx | ✅ |  |
| node/test_gemm_beta/model.onnx | ✅ |  |
| node/test_gemm_default_matrix_bias/model.onnx | ✅ |  |
| node/test_gemm_default_no_bias/model.onnx | ✅ |  |
| node/test_gemm_default_scalar_bias/model.onnx | ✅ |  |
| node/test_gemm_default_single_elem_vector_bias/model.onnx | ✅ |  |
| node/test_gemm_default_vector_bias/model.onnx | ✅ |  |
| node/test_gemm_default_zero_bias/model.onnx | ✅ |  |
| node/test_gemm_transposeA/model.onnx | ✅ |  |
| node/test_gemm_transposeB/model.onnx | ✅ |  |
| node/test_globalaveragepool/model.onnx | ✅ |  |
| node/test_globalaveragepool_precomputed/model.onnx | ✅ |  |
| node/test_globalmaxpool/model.onnx | ❌ | Unsupported op GlobalMaxPool |
| node/test_globalmaxpool_precomputed/model.onnx | ❌ | Unsupported op GlobalMaxPool |
| node/test_greater/model.onnx | ✅ |  |
| node/test_greater_bcast/model.onnx | ✅ |  |
| node/test_greater_equal/model.onnx | ✅ |  |
| node/test_greater_equal_bcast/model.onnx | ✅ |  |
| node/test_greater_equal_bcast_expanded/model.onnx | ✅ |  |
| node/test_greater_equal_expanded/model.onnx | ✅ |  |
| node/test_greater_equal_int16/model.onnx | ✅ |  |
| node/test_greater_equal_int16_expanded/model.onnx | ✅ |  |
| node/test_greater_equal_int8/model.onnx | ✅ |  |
| node/test_greater_equal_int8_expanded/model.onnx | ✅ |  |
| node/test_greater_equal_uint16/model.onnx | ✅ |  |
| node/test_greater_equal_uint16_expanded/model.onnx | ✅ |  |
| node/test_greater_equal_uint32/model.onnx | ✅ |  |
| node/test_greater_equal_uint32_expanded/model.onnx | ✅ |  |
| node/test_greater_equal_uint64/model.onnx | ✅ |  |
| node/test_greater_equal_uint64_expanded/model.onnx | ✅ |  |
| node/test_greater_equal_uint8/model.onnx | ✅ |  |
| node/test_greater_equal_uint8_expanded/model.onnx | ✅ |  |
| node/test_greater_int16/model.onnx | ✅ |  |
| node/test_greater_int8/model.onnx | ✅ |  |
| node/test_greater_uint16/model.onnx | ✅ |  |
| node/test_greater_uint32/model.onnx | ✅ |  |
| node/test_greater_uint64/model.onnx | ✅ |  |
| node/test_greater_uint8/model.onnx | ✅ |  |
| node/test_gridsample/model.onnx | ❌ | Unsupported op GridSample |
| node/test_gridsample_aligncorners_true/model.onnx | ❌ | Unsupported op GridSample |
| node/test_gridsample_bicubic/model.onnx | ❌ | Unsupported op GridSample |
| node/test_gridsample_bicubic_align_corners_0_additional_1/model.onnx | ❌ | Unsupported op GridSample |
| node/test_gridsample_bicubic_align_corners_1_additional_1/model.onnx | ❌ | Unsupported op GridSample |
| node/test_gridsample_bilinear/model.onnx | ❌ | Unsupported op GridSample |
| node/test_gridsample_bilinear_align_corners_0_additional_1/model.onnx | ❌ | Unsupported op GridSample |
| node/test_gridsample_bilinear_align_corners_1_additional_1/model.onnx | ❌ | Unsupported op GridSample |
| node/test_gridsample_border_padding/model.onnx | ❌ | Unsupported op GridSample |
| node/test_gridsample_nearest/model.onnx | ❌ | Unsupported op GridSample |
| node/test_gridsample_nearest_align_corners_0_additional_1/model.onnx | ❌ | Unsupported op GridSample |
| node/test_gridsample_nearest_align_corners_1_additional_1/model.onnx | ❌ | Unsupported op GridSample |
| node/test_gridsample_reflection_padding/model.onnx | ❌ | Unsupported op GridSample |
| node/test_gridsample_volumetric_bilinear_align_corners_0/model.onnx | ❌ | Unsupported op GridSample |
| node/test_gridsample_volumetric_bilinear_align_corners_1/model.onnx | ❌ | Unsupported op GridSample |
| node/test_gridsample_volumetric_nearest_align_corners_0/model.onnx | ❌ | Unsupported op GridSample |
| node/test_gridsample_volumetric_nearest_align_corners_1/model.onnx | ❌ | Unsupported op GridSample |
| node/test_gridsample_zeros_padding/model.onnx | ❌ | Unsupported op GridSample |
| node/test_group_normalization_epsilon/model.onnx | ❌ | Unsupported op GroupNormalization |
| node/test_group_normalization_epsilon_expanded/model.onnx | ❌ | Dynamic dim for tensor 'GroupNormalization_test_group_normalization_epsilon_expanded_function_XReshaped' |
| node/test_group_normalization_example/model.onnx | ❌ | Unsupported op GroupNormalization |
| node/test_group_normalization_example_expanded/model.onnx | ❌ | Dynamic dim for tensor 'GroupNormalization_test_group_normalization_example_expanded_function_XReshaped' |
| node/test_gru_batchwise/model.onnx | ❌ | Unsupported op GRU |
| node/test_gru_defaults/model.onnx | ❌ | Unsupported op GRU |
| node/test_gru_seq_length/model.onnx | ❌ | Unsupported op GRU |
| node/test_gru_with_initial_bias/model.onnx | ❌ | Unsupported op GRU |
| node/test_hammingwindow/model.onnx | ❌ | Unsupported op HammingWindow |
| node/test_hammingwindow_expanded/model.onnx | ❌ | Dynamic dim for tensor 'HammingWindow_test_hammingwindow_expanded_function_Range' |
| node/test_hammingwindow_symmetric/model.onnx | ❌ | Unsupported op HammingWindow |
| node/test_hammingwindow_symmetric_expanded/model.onnx | ❌ | Dynamic dim for tensor 'HammingWindow_test_hammingwindow_symmetric_expanded_function_Range' |
| node/test_hannwindow/model.onnx | ❌ | Unsupported op HannWindow |
| node/test_hannwindow_expanded/model.onnx | ❌ | Dynamic dim for tensor 'HannWindow_test_hannwindow_expanded_function_Range' |
| node/test_hannwindow_symmetric/model.onnx | ❌ | Unsupported op HannWindow |
| node/test_hannwindow_symmetric_expanded/model.onnx | ❌ | Dynamic dim for tensor 'HannWindow_test_hannwindow_symmetric_expanded_function_Range' |
| node/test_hardmax_axis_0/model.onnx | ❌ | Unsupported op Hardmax |
| node/test_hardmax_axis_1/model.onnx | ❌ | Unsupported op Hardmax |
| node/test_hardmax_axis_2/model.onnx | ❌ | Unsupported op Hardmax |
| node/test_hardmax_default_axis/model.onnx | ❌ | Unsupported op Hardmax |
| node/test_hardmax_example/model.onnx | ❌ | Unsupported op Hardmax |
| node/test_hardmax_negative_axis/model.onnx | ❌ | Unsupported op Hardmax |
| node/test_hardmax_one_hot/model.onnx | ❌ | Unsupported op Hardmax |
| node/test_hardsigmoid/model.onnx | ❌ | HardSigmoid only supports alpha=0.2 |
| node/test_hardsigmoid_default/model.onnx | ✅ |  |
| node/test_hardsigmoid_default_expanded_ver18/model.onnx | ✅ |  |
| node/test_hardsigmoid_example/model.onnx | ❌ | HardSigmoid only supports alpha=0.2 |
| node/test_hardsigmoid_example_expanded_ver18/model.onnx | ✅ |  |
| node/test_hardsigmoid_expanded_ver18/model.onnx | ✅ |  |
| node/test_hardswish/model.onnx | ✅ |  |
| node/test_hardswish_expanded/model.onnx | ❌ | HardSigmoid only supports alpha=0.2 |
| node/test_identity/model.onnx | ✅ |  |
| node/test_identity_opt/model.onnx | ❌ | Unsupported value type 'optional_type' for 'opt_in'. Hint: export the model with tensor inputs/outputs. |
| node/test_identity_sequence/model.onnx | ❌ | Unsupported value type 'sequence_type' for 'x'. Hint: export the model with tensor inputs/outputs. |
| node/test_if/model.onnx | ❌ | Unsupported op If |
| node/test_if_opt/model.onnx | ❌ | Unsupported value type 'optional_type' for 'sequence'. Hint: export the model with tensor inputs/outputs. |
| node/test_if_seq/model.onnx | ❌ | Unsupported value type 'sequence_type' for 'res'. Hint: export the model with tensor inputs/outputs. |
| node/test_image_decoder_decode_bmp_rgb/model.onnx | ❌ | Unsupported op ImageDecoder |
| node/test_image_decoder_decode_jpeg2k_rgb/model.onnx | ❌ | Unsupported op ImageDecoder |
| node/test_image_decoder_decode_jpeg_bgr/model.onnx | ❌ | Unsupported op ImageDecoder |
| node/test_image_decoder_decode_jpeg_grayscale/model.onnx | ❌ | Unsupported op ImageDecoder |
| node/test_image_decoder_decode_jpeg_rgb/model.onnx | ❌ | Unsupported op ImageDecoder |
| node/test_image_decoder_decode_png_rgb/model.onnx | ❌ | Unsupported op ImageDecoder |
| node/test_image_decoder_decode_pnm_rgb/model.onnx | ❌ | Unsupported op ImageDecoder |
| node/test_image_decoder_decode_tiff_rgb/model.onnx | ❌ | Unsupported op ImageDecoder |
| node/test_image_decoder_decode_webp_rgb/model.onnx | ❌ | Unsupported op ImageDecoder |
| node/test_instancenorm_epsilon/model.onnx | ❌ | Unsupported op InstanceNormalization |
| node/test_instancenorm_example/model.onnx | ❌ | Unsupported op InstanceNormalization |
| node/test_isinf/model.onnx | ✅ |  |
| node/test_isinf_float16/model.onnx | ✅ |  |
| node/test_isinf_negative/model.onnx | ✅ |  |
| node/test_isinf_positive/model.onnx | ✅ |  |
| node/test_isnan/model.onnx | ✅ |  |
| node/test_isnan_float16/model.onnx | ✅ |  |
| node/test_l1normalization_axis_0/model.onnx | ❌ | Unsupported op LpNormalization |
| node/test_l1normalization_axis_1/model.onnx | ❌ | Unsupported op LpNormalization |
| node/test_l1normalization_axis_last/model.onnx | ❌ | Unsupported op LpNormalization |
| node/test_l2normalization_axis_0/model.onnx | ❌ | Unsupported op LpNormalization |
| node/test_l2normalization_axis_1/model.onnx | ❌ | Unsupported op LpNormalization |
| node/test_layer_normalization_2d_axis0/model.onnx | ❌ | Unsupported op LayerNormalization |
| node/test_layer_normalization_2d_axis0_expanded/model.onnx | ❌ | Dynamic or zero dims are not supported |
| node/test_layer_normalization_2d_axis0_expanded_ver18/model.onnx | ❌ | Dynamic or zero dims are not supported |
| node/test_layer_normalization_2d_axis1/model.onnx | ❌ | Unsupported op LayerNormalization |
| node/test_layer_normalization_2d_axis1_expanded/model.onnx | ✅ |  |
| node/test_layer_normalization_2d_axis1_expanded_ver18/model.onnx | ✅ |  |
| node/test_layer_normalization_2d_axis_negative_1/model.onnx | ❌ | Unsupported op LayerNormalization |
| node/test_layer_normalization_2d_axis_negative_1_expanded/model.onnx | ❌ | Dynamic dim for tensor 'LayerNormalization_test_layer_normalization_2d_axis_negative_1_expanded_function_SuffixShape' |
| node/test_layer_normalization_2d_axis_negative_1_expanded_ver18/model.onnx | ❌ | Dynamic dim for tensor 'LayerNormalization_test_layer_normalization_2d_axis_negative_1_expanded_function_SuffixShape' |
| node/test_layer_normalization_2d_axis_negative_2/model.onnx | ❌ | Unsupported op LayerNormalization |
| node/test_layer_normalization_2d_axis_negative_2_expanded/model.onnx | ❌ | Dynamic dim for tensor 'LayerNormalization_test_layer_normalization_2d_axis_negative_2_expanded_function_SuffixShape' |
| node/test_layer_normalization_2d_axis_negative_2_expanded_ver18/model.onnx | ❌ | Dynamic dim for tensor 'LayerNormalization_test_layer_normalization_2d_axis_negative_2_expanded_function_SuffixShape' |
| node/test_layer_normalization_3d_axis0_epsilon/model.onnx | ❌ | Unsupported op LayerNormalization |
| node/test_layer_normalization_3d_axis0_epsilon_expanded/model.onnx | ❌ | Dynamic or zero dims are not supported |
| node/test_layer_normalization_3d_axis0_epsilon_expanded_ver18/model.onnx | ❌ | Dynamic or zero dims are not supported |
| node/test_layer_normalization_3d_axis1_epsilon/model.onnx | ❌ | Unsupported op LayerNormalization |
| node/test_layer_normalization_3d_axis1_epsilon_expanded/model.onnx | ✅ |  |
| node/test_layer_normalization_3d_axis1_epsilon_expanded_ver18/model.onnx | ✅ |  |
| node/test_layer_normalization_3d_axis2_epsilon/model.onnx | ❌ | Unsupported op LayerNormalization |
| node/test_layer_normalization_3d_axis2_epsilon_expanded/model.onnx | ✅ |  |
| node/test_layer_normalization_3d_axis2_epsilon_expanded_ver18/model.onnx | ✅ |  |
| node/test_layer_normalization_3d_axis_negative_1_epsilon/model.onnx | ❌ | Unsupported op LayerNormalization |
| node/test_layer_normalization_3d_axis_negative_1_epsilon_expanded/model.onnx | ❌ | Dynamic dim for tensor 'LayerNormalization_test_layer_normalization_3d_axis_negative_1_epsilon_expanded_function_SuffixShape' |
| node/test_layer_normalization_3d_axis_negative_1_epsilon_expanded_ver18/model.onnx | ❌ | Dynamic dim for tensor 'LayerNormalization_test_layer_normalization_3d_axis_negative_1_epsilon_expanded_function_SuffixShape' |
| node/test_layer_normalization_3d_axis_negative_2_epsilon/model.onnx | ❌ | Unsupported op LayerNormalization |
| node/test_layer_normalization_3d_axis_negative_2_epsilon_expanded/model.onnx | ❌ | Dynamic dim for tensor 'LayerNormalization_test_layer_normalization_3d_axis_negative_2_epsilon_expanded_function_SuffixShape' |
| node/test_layer_normalization_3d_axis_negative_2_epsilon_expanded_ver18/model.onnx | ❌ | Dynamic dim for tensor 'LayerNormalization_test_layer_normalization_3d_axis_negative_2_epsilon_expanded_function_SuffixShape' |
| node/test_layer_normalization_3d_axis_negative_3_epsilon/model.onnx | ❌ | Unsupported op LayerNormalization |
| node/test_layer_normalization_3d_axis_negative_3_epsilon_expanded/model.onnx | ❌ | Dynamic dim for tensor 'LayerNormalization_test_layer_normalization_3d_axis_negative_3_epsilon_expanded_function_SuffixShape' |
| node/test_layer_normalization_3d_axis_negative_3_epsilon_expanded_ver18/model.onnx | ❌ | Dynamic dim for tensor 'LayerNormalization_test_layer_normalization_3d_axis_negative_3_epsilon_expanded_function_SuffixShape' |
| node/test_layer_normalization_4d_axis0/model.onnx | ❌ | Unsupported op LayerNormalization |
| node/test_layer_normalization_4d_axis0_expanded/model.onnx | ❌ | Dynamic or zero dims are not supported |
| node/test_layer_normalization_4d_axis0_expanded_ver18/model.onnx | ❌ | Dynamic or zero dims are not supported |
| node/test_layer_normalization_4d_axis1/model.onnx | ❌ | Unsupported op LayerNormalization |
| node/test_layer_normalization_4d_axis1_expanded/model.onnx | ✅ |  |
| node/test_layer_normalization_4d_axis1_expanded_ver18/model.onnx | ✅ |  |
| node/test_layer_normalization_4d_axis2/model.onnx | ❌ | Unsupported op LayerNormalization |
| node/test_layer_normalization_4d_axis2_expanded/model.onnx | ✅ |  |
| node/test_layer_normalization_4d_axis2_expanded_ver18/model.onnx | ✅ |  |
| node/test_layer_normalization_4d_axis3/model.onnx | ❌ | Unsupported op LayerNormalization |
| node/test_layer_normalization_4d_axis3_expanded/model.onnx | ✅ |  |
| node/test_layer_normalization_4d_axis3_expanded_ver18/model.onnx | ✅ |  |
| node/test_layer_normalization_4d_axis_negative_1/model.onnx | ❌ | Unsupported op LayerNormalization |
| node/test_layer_normalization_4d_axis_negative_1_expanded/model.onnx | ❌ | Dynamic dim for tensor 'LayerNormalization_test_layer_normalization_4d_axis_negative_1_expanded_function_SuffixShape' |
| node/test_layer_normalization_4d_axis_negative_1_expanded_ver18/model.onnx | ❌ | Dynamic dim for tensor 'LayerNormalization_test_layer_normalization_4d_axis_negative_1_expanded_function_SuffixShape' |
| node/test_layer_normalization_4d_axis_negative_2/model.onnx | ❌ | Unsupported op LayerNormalization |
| node/test_layer_normalization_4d_axis_negative_2_expanded/model.onnx | ❌ | Dynamic dim for tensor 'LayerNormalization_test_layer_normalization_4d_axis_negative_2_expanded_function_SuffixShape' |
| node/test_layer_normalization_4d_axis_negative_2_expanded_ver18/model.onnx | ❌ | Dynamic dim for tensor 'LayerNormalization_test_layer_normalization_4d_axis_negative_2_expanded_function_SuffixShape' |
| node/test_layer_normalization_4d_axis_negative_3/model.onnx | ❌ | Unsupported op LayerNormalization |
| node/test_layer_normalization_4d_axis_negative_3_expanded/model.onnx | ❌ | Dynamic dim for tensor 'LayerNormalization_test_layer_normalization_4d_axis_negative_3_expanded_function_SuffixShape' |
| node/test_layer_normalization_4d_axis_negative_3_expanded_ver18/model.onnx | ❌ | Dynamic dim for tensor 'LayerNormalization_test_layer_normalization_4d_axis_negative_3_expanded_function_SuffixShape' |
| node/test_layer_normalization_4d_axis_negative_4/model.onnx | ❌ | Unsupported op LayerNormalization |
| node/test_layer_normalization_4d_axis_negative_4_expanded/model.onnx | ❌ | Dynamic dim for tensor 'LayerNormalization_test_layer_normalization_4d_axis_negative_4_expanded_function_SuffixShape' |
| node/test_layer_normalization_4d_axis_negative_4_expanded_ver18/model.onnx | ❌ | Dynamic dim for tensor 'LayerNormalization_test_layer_normalization_4d_axis_negative_4_expanded_function_SuffixShape' |
| node/test_layer_normalization_default_axis/model.onnx | ❌ | Unsupported op LayerNormalization |
| node/test_layer_normalization_default_axis_expanded/model.onnx | ❌ | Dynamic dim for tensor 'LayerNormalization_test_layer_normalization_default_axis_expanded_function_SuffixShape' |
| node/test_layer_normalization_default_axis_expanded_ver18/model.onnx | ❌ | Dynamic dim for tensor 'LayerNormalization_test_layer_normalization_default_axis_expanded_function_SuffixShape' |
| node/test_leakyrelu/model.onnx | ❌ | LeakyRelu only supports alpha=0.01 |
| node/test_leakyrelu_default/model.onnx | ✅ |  |
| node/test_leakyrelu_default_expanded/model.onnx | ✅ |  |
| node/test_leakyrelu_example/model.onnx | ❌ | LeakyRelu only supports alpha=0.01 |
| node/test_leakyrelu_example_expanded/model.onnx | ✅ |  |
| node/test_leakyrelu_expanded/model.onnx | ✅ |  |
| node/test_less/model.onnx | ✅ |  |
| node/test_less_bcast/model.onnx | ✅ |  |
| node/test_less_equal/model.onnx | ✅ |  |
| node/test_less_equal_bcast/model.onnx | ✅ |  |
| node/test_less_equal_bcast_expanded/model.onnx | ✅ |  |
| node/test_less_equal_expanded/model.onnx | ✅ |  |
| node/test_less_equal_int16/model.onnx | ✅ |  |
| node/test_less_equal_int16_expanded/model.onnx | ✅ |  |
| node/test_less_equal_int8/model.onnx | ✅ |  |
| node/test_less_equal_int8_expanded/model.onnx | ✅ |  |
| node/test_less_equal_uint16/model.onnx | ✅ |  |
| node/test_less_equal_uint16_expanded/model.onnx | ✅ |  |
| node/test_less_equal_uint32/model.onnx | ✅ |  |
| node/test_less_equal_uint32_expanded/model.onnx | ✅ |  |
| node/test_less_equal_uint64/model.onnx | ✅ |  |
| node/test_less_equal_uint64_expanded/model.onnx | ✅ |  |
| node/test_less_equal_uint8/model.onnx | ✅ |  |
| node/test_less_equal_uint8_expanded/model.onnx | ✅ |  |
| node/test_less_int16/model.onnx | ✅ |  |
| node/test_less_int8/model.onnx | ✅ |  |
| node/test_less_uint16/model.onnx | ✅ |  |
| node/test_less_uint32/model.onnx | ✅ |  |
| node/test_less_uint64/model.onnx | ✅ |  |
| node/test_less_uint8/model.onnx | ✅ |  |
| node/test_log/model.onnx | ✅ |  |
| node/test_log_example/model.onnx | ✅ |  |
| node/test_logsoftmax_axis_0/model.onnx | ✅ |  |
| node/test_logsoftmax_axis_0_expanded/model.onnx | ✅ |  |
| node/test_logsoftmax_axis_0_expanded_ver18/model.onnx | ✅ |  |
| node/test_logsoftmax_axis_1/model.onnx | ✅ |  |
| node/test_logsoftmax_axis_1_expanded/model.onnx | ✅ |  |
| node/test_logsoftmax_axis_1_expanded_ver18/model.onnx | ✅ |  |
| node/test_logsoftmax_axis_2/model.onnx | ✅ |  |
| node/test_logsoftmax_axis_2_expanded/model.onnx | ✅ |  |
| node/test_logsoftmax_axis_2_expanded_ver18/model.onnx | ✅ |  |
| node/test_logsoftmax_default_axis/model.onnx | ✅ |  |
| node/test_logsoftmax_default_axis_expanded/model.onnx | ✅ |  |
| node/test_logsoftmax_default_axis_expanded_ver18/model.onnx | ✅ |  |
| node/test_logsoftmax_example_1/model.onnx | ✅ |  |
| node/test_logsoftmax_example_1_expanded/model.onnx | ✅ |  |
| node/test_logsoftmax_example_1_expanded_ver18/model.onnx | ✅ |  |
| node/test_logsoftmax_large_number/model.onnx | ✅ |  |
| node/test_logsoftmax_large_number_expanded/model.onnx | ✅ |  |
| node/test_logsoftmax_large_number_expanded_ver18/model.onnx | ✅ |  |
| node/test_logsoftmax_negative_axis/model.onnx | ✅ |  |
| node/test_logsoftmax_negative_axis_expanded/model.onnx | ✅ |  |
| node/test_logsoftmax_negative_axis_expanded_ver18/model.onnx | ✅ |  |
| node/test_loop11/model.onnx | ❌ | Unsupported op Loop |
| node/test_loop13_seq/model.onnx | ❌ | Unsupported value type 'sequence_type' for 'seq_empty'. Hint: export the model with tensor inputs/outputs. |
| node/test_loop16_seq_none/model.onnx | ❌ | Unsupported value type 'optional_type' for 'opt_seq'. Hint: export the model with tensor inputs/outputs. |
| node/test_lpnormalization_default/model.onnx | ❌ | Unsupported op LpNormalization |
| node/test_lppool_1d_default/model.onnx | ❌ | Unsupported op LpPool |
| node/test_lppool_2d_default/model.onnx | ❌ | Unsupported op LpPool |
| node/test_lppool_2d_dilations/model.onnx | ❌ | Unsupported op LpPool |
| node/test_lppool_2d_pads/model.onnx | ❌ | Unsupported op LpPool |
| node/test_lppool_2d_same_lower/model.onnx | ❌ | Unsupported op LpPool |
| node/test_lppool_2d_same_upper/model.onnx | ❌ | Unsupported op LpPool |
| node/test_lppool_2d_strides/model.onnx | ❌ | Unsupported op LpPool |
| node/test_lppool_3d_default/model.onnx | ❌ | Unsupported op LpPool |
| node/test_lrn/model.onnx | ✅ |  |
| node/test_lrn_default/model.onnx | ✅ |  |
| node/test_lstm_batchwise/model.onnx | ✅ |  |
| node/test_lstm_defaults/model.onnx | ✅ |  |
| node/test_lstm_with_initial_bias/model.onnx | ✅ |  |
| node/test_lstm_with_peepholes/model.onnx | ✅ |  |
| node/test_matmul_1d_1d/model.onnx | ✅ |  |
| node/test_matmul_1d_3d/model.onnx | ✅ |  |
| node/test_matmul_2d/model.onnx | ✅ |  |
| node/test_matmul_3d/model.onnx | ✅ |  |
| node/test_matmul_4d/model.onnx | ✅ |  |
| node/test_matmul_4d_1d/model.onnx | ✅ |  |
| node/test_matmul_bcast/model.onnx | ✅ |  |
| node/test_matmulinteger/model.onnx | ❌ | Unsupported op MatMulInteger |
| node/test_max_example/model.onnx | ❌ | Max must have 2 inputs and 1 output |
| node/test_max_float16/model.onnx | ❌ | <ScalarType.F16: 'f16'> |
| node/test_max_float32/model.onnx | ✅ |  |
| node/test_max_float64/model.onnx | ✅ |  |
| node/test_max_int16/model.onnx | ✅ |  |
| node/test_max_int32/model.onnx | ✅ |  |
| node/test_max_int64/model.onnx | ✅ |  |
| node/test_max_int8/model.onnx | ✅ |  |
| node/test_max_one_input/model.onnx | ❌ | Max must have 2 inputs and 1 output |
| node/test_max_two_inputs/model.onnx | ✅ |  |
| node/test_max_uint16/model.onnx | ✅ |  |
| node/test_max_uint32/model.onnx | ✅ |  |
| node/test_max_uint64/model.onnx | ✅ |  |
| node/test_max_uint8/model.onnx | ✅ |  |
| node/test_maxpool_1d_default/model.onnx | ✅ |  |
| node/test_maxpool_2d_ceil/model.onnx | ✅ |  |
| node/test_maxpool_2d_ceil_output_size_reduce_by_one/model.onnx | ✅ |  |
| node/test_maxpool_2d_default/model.onnx | ✅ |  |
| node/test_maxpool_2d_dilations/model.onnx | ✅ |  |
| node/test_maxpool_2d_pads/model.onnx | ✅ |  |
| node/test_maxpool_2d_precomputed_pads/model.onnx | ✅ |  |
| node/test_maxpool_2d_precomputed_same_upper/model.onnx | ✅ |  |
| node/test_maxpool_2d_precomputed_strides/model.onnx | ✅ |  |
| node/test_maxpool_2d_same_lower/model.onnx | ✅ |  |
| node/test_maxpool_2d_same_upper/model.onnx | ✅ |  |
| node/test_maxpool_2d_strides/model.onnx | ✅ |  |
| node/test_maxpool_2d_uint8/model.onnx | ✅ |  |
| node/test_maxpool_3d_default/model.onnx | ✅ |  |
| node/test_maxpool_3d_dilations/model.onnx | ✅ |  |
| node/test_maxpool_3d_dilations_use_ref_impl/model.onnx | ✅ |  |
| node/test_maxpool_3d_dilations_use_ref_impl_large/model.onnx | ✅ |  |
| node/test_maxpool_with_argmax_2d_precomputed_pads/model.onnx | ✅ |  |
| node/test_maxpool_with_argmax_2d_precomputed_strides/model.onnx | ✅ |  |
| node/test_maxunpool_export_with_output_shape/model.onnx | ❌ | Unsupported op MaxUnpool |
| node/test_maxunpool_export_without_output_shape/model.onnx | ❌ | Unsupported op MaxUnpool |
| node/test_mean_example/model.onnx | ❌ | Mean must have 2 inputs and 1 output |
| node/test_mean_one_input/model.onnx | ❌ | Mean must have 2 inputs and 1 output |
| node/test_mean_two_inputs/model.onnx | ✅ |  |
| node/test_melweightmatrix/model.onnx | ❌ | Unsupported op MelWeightMatrix |
| node/test_min_example/model.onnx | ❌ | Min must have 2 inputs and 1 output |
| node/test_min_float16/model.onnx | ❌ | <ScalarType.F16: 'f16'> |
| node/test_min_float32/model.onnx | ✅ |  |
| node/test_min_float64/model.onnx | ✅ |  |
| node/test_min_int16/model.onnx | ✅ |  |
| node/test_min_int32/model.onnx | ✅ |  |
| node/test_min_int64/model.onnx | ✅ |  |
| node/test_min_int8/model.onnx | ✅ |  |
| node/test_min_one_input/model.onnx | ❌ | Min must have 2 inputs and 1 output |
| node/test_min_two_inputs/model.onnx | ✅ |  |
| node/test_min_uint16/model.onnx | ✅ |  |
| node/test_min_uint32/model.onnx | ✅ |  |
| node/test_min_uint64/model.onnx | ✅ |  |
| node/test_min_uint8/model.onnx | ✅ |  |
| node/test_mish/model.onnx | ❌ | Unsupported op Mish |
| node/test_mish_expanded/model.onnx | ✅ |  |
| node/test_mod_broadcast/model.onnx | ✅ |  |
| node/test_mod_int64_fmod/model.onnx | ✅ |  |
| node/test_mod_mixed_sign_float16/model.onnx | ❌ | <ScalarType.F16: 'f16'> |
| node/test_mod_mixed_sign_float32/model.onnx | ✅ |  |
| node/test_mod_mixed_sign_float64/model.onnx | ✅ |  |
| node/test_mod_mixed_sign_int16/model.onnx | ✅ |  |
| node/test_mod_mixed_sign_int32/model.onnx | ✅ |  |
| node/test_mod_mixed_sign_int64/model.onnx | ✅ |  |
| node/test_mod_mixed_sign_int8/model.onnx | ✅ |  |
| node/test_mod_uint16/model.onnx | ✅ |  |
| node/test_mod_uint32/model.onnx | ✅ |  |
| node/test_mod_uint64/model.onnx | ✅ |  |
| node/test_mod_uint8/model.onnx | ✅ |  |
| node/test_momentum/model.onnx | ❌ | Unsupported op Momentum |
| node/test_momentum_multiple/model.onnx | ❌ | Unsupported op Momentum |
| node/test_mul/model.onnx | ✅ |  |
| node/test_mul_bcast/model.onnx | ✅ |  |
| node/test_mul_example/model.onnx | ✅ |  |
| node/test_mul_int16/model.onnx | ✅ |  |
| node/test_mul_int8/model.onnx | ✅ |  |
| node/test_mul_uint16/model.onnx | ✅ |  |
| node/test_mul_uint32/model.onnx | ✅ |  |
| node/test_mul_uint64/model.onnx | ✅ |  |
| node/test_mul_uint8/model.onnx | ✅ |  |
| node/test_mvn/model.onnx | ❌ | Unsupported op MeanVarianceNormalization |
| node/test_mvn_expanded/model.onnx | ✅ |  |
| node/test_mvn_expanded_ver18/model.onnx | ✅ |  |
| node/test_neg/model.onnx | ✅ |  |
| node/test_neg_example/model.onnx | ✅ |  |
| node/test_nesterov_momentum/model.onnx | ❌ | Unsupported op Momentum |
| node/test_nllloss_NC/model.onnx | ✅ |  |
| node/test_nllloss_NC_expanded/model.onnx | ✅ |  |
| node/test_nllloss_NCd1/model.onnx | ✅ |  |
| node/test_nllloss_NCd1_expanded/model.onnx | ✅ |  |
| node/test_nllloss_NCd1_ii/model.onnx | ✅ |  |
| node/test_nllloss_NCd1_ii_expanded/model.onnx | ✅ |  |
| node/test_nllloss_NCd1_mean_weight_negative_ii/model.onnx | ✅ |  |
| node/test_nllloss_NCd1_mean_weight_negative_ii_expanded/model.onnx | ✅ |  |
| node/test_nllloss_NCd1_weight/model.onnx | ✅ |  |
| node/test_nllloss_NCd1_weight_expanded/model.onnx | ✅ |  |
| node/test_nllloss_NCd1_weight_ii/model.onnx | ✅ |  |
| node/test_nllloss_NCd1_weight_ii_expanded/model.onnx | ✅ |  |
| node/test_nllloss_NCd1d2/model.onnx | ✅ |  |
| node/test_nllloss_NCd1d2_expanded/model.onnx | ✅ |  |
| node/test_nllloss_NCd1d2_no_weight_reduction_mean_ii/model.onnx | ✅ |  |
| node/test_nllloss_NCd1d2_no_weight_reduction_mean_ii_expanded/model.onnx | ✅ |  |
| node/test_nllloss_NCd1d2_reduction_mean/model.onnx | ✅ |  |
| node/test_nllloss_NCd1d2_reduction_mean_expanded/model.onnx | ✅ |  |
| node/test_nllloss_NCd1d2_reduction_sum/model.onnx | ✅ |  |
| node/test_nllloss_NCd1d2_reduction_sum_expanded/model.onnx | ✅ |  |
| node/test_nllloss_NCd1d2_with_weight/model.onnx | ✅ |  |
| node/test_nllloss_NCd1d2_with_weight_expanded/model.onnx | ✅ |  |
| node/test_nllloss_NCd1d2_with_weight_reduction_mean/model.onnx | ✅ |  |
| node/test_nllloss_NCd1d2_with_weight_reduction_mean_expanded/model.onnx | ✅ |  |
| node/test_nllloss_NCd1d2_with_weight_reduction_sum/model.onnx | ✅ |  |
| node/test_nllloss_NCd1d2_with_weight_reduction_sum_expanded/model.onnx | ✅ |  |
| node/test_nllloss_NCd1d2_with_weight_reduction_sum_ii/model.onnx | ✅ |  |
| node/test_nllloss_NCd1d2_with_weight_reduction_sum_ii_expanded/model.onnx | ✅ |  |
| node/test_nllloss_NCd1d2d3_none_no_weight_negative_ii/model.onnx | ✅ |  |
| node/test_nllloss_NCd1d2d3_none_no_weight_negative_ii_expanded/model.onnx | ✅ |  |
| node/test_nllloss_NCd1d2d3_sum_weight_high_ii/model.onnx | ✅ |  |
| node/test_nllloss_NCd1d2d3_sum_weight_high_ii_expanded/model.onnx | ✅ |  |
| node/test_nllloss_NCd1d2d3d4d5_mean_weight/model.onnx | ✅ |  |
| node/test_nllloss_NCd1d2d3d4d5_mean_weight_expanded/model.onnx | ✅ |  |
| node/test_nllloss_NCd1d2d3d4d5_none_no_weight/model.onnx | ✅ |  |
| node/test_nllloss_NCd1d2d3d4d5_none_no_weight_expanded/model.onnx | ✅ |  |
| node/test_nonmaxsuppression_center_point_box_format/model.onnx | ❌ | Unsupported op NonMaxSuppression |
| node/test_nonmaxsuppression_flipped_coordinates/model.onnx | ❌ | Unsupported op NonMaxSuppression |
| node/test_nonmaxsuppression_identical_boxes/model.onnx | ❌ | Unsupported op NonMaxSuppression |
| node/test_nonmaxsuppression_limit_output_size/model.onnx | ❌ | Unsupported op NonMaxSuppression |
| node/test_nonmaxsuppression_single_box/model.onnx | ❌ | Unsupported op NonMaxSuppression |
| node/test_nonmaxsuppression_suppress_by_IOU/model.onnx | ❌ | Unsupported op NonMaxSuppression |
| node/test_nonmaxsuppression_suppress_by_IOU_and_scores/model.onnx | ❌ | Unsupported op NonMaxSuppression |
| node/test_nonmaxsuppression_two_batches/model.onnx | ❌ | Unsupported op NonMaxSuppression |
| node/test_nonmaxsuppression_two_classes/model.onnx | ❌ | Unsupported op NonMaxSuppression |
| node/test_nonzero_example/model.onnx | ❌ | Unsupported op NonZero |
| node/test_not_2d/model.onnx | ✅ |  |
| node/test_not_3d/model.onnx | ✅ |  |
| node/test_not_4d/model.onnx | ✅ |  |
| node/test_onehot_negative_indices/model.onnx | ❌ | Unsupported op OneHot |
| node/test_onehot_with_axis/model.onnx | ❌ | Unsupported op OneHot |
| node/test_onehot_with_negative_axis/model.onnx | ❌ | Unsupported op OneHot |
| node/test_onehot_without_axis/model.onnx | ❌ | Unsupported op OneHot |
| node/test_optional_get_element_optional_sequence/model.onnx | ❌ | Unsupported value type 'optional_type' for 'optional_input'. Hint: export the model with tensor inputs/outputs. |
| node/test_optional_get_element_optional_tensor/model.onnx | ❌ | Unsupported value type 'optional_type' for 'optional_input'. Hint: export the model with tensor inputs/outputs. |
| node/test_optional_get_element_sequence/model.onnx | ❌ | Unsupported value type 'sequence_type' for 'optional_input'. Hint: export the model with tensor inputs/outputs. |
| node/test_optional_get_element_tensor/model.onnx | ❌ | Unsupported op OptionalGetElement |
| node/test_optional_has_element_empty_no_input_name_optional_input/model.onnx | ❌ | Unsupported op OptionalHasElement |
| node/test_optional_has_element_empty_no_input_name_tensor_input/model.onnx | ❌ | Unsupported op OptionalHasElement |
| node/test_optional_has_element_empty_no_input_optional_input/model.onnx | ❌ | Unsupported op OptionalHasElement |
| node/test_optional_has_element_empty_no_input_tensor_input/model.onnx | ❌ | Unsupported op OptionalHasElement |
| node/test_optional_has_element_empty_optional_input/model.onnx | ❌ | Unsupported value type 'optional_type' for 'optional_input'. Hint: export the model with tensor inputs/outputs. |
| node/test_optional_has_element_optional_input/model.onnx | ❌ | Unsupported value type 'optional_type' for 'optional_input'. Hint: export the model with tensor inputs/outputs. |
| node/test_optional_has_element_tensor_input/model.onnx | ❌ | Unsupported value type 'optional_type' for 'optional_input'. Hint: export the model with tensor inputs/outputs. |
| node/test_or2d/model.onnx | ✅ |  |
| node/test_or3d/model.onnx | ✅ |  |
| node/test_or4d/model.onnx | ✅ |  |
| node/test_or_bcast3v1d/model.onnx | ✅ |  |
| node/test_or_bcast3v2d/model.onnx | ✅ |  |
| node/test_or_bcast4v2d/model.onnx | ✅ |  |
| node/test_or_bcast4v3d/model.onnx | ✅ |  |
| node/test_or_bcast4v4d/model.onnx | ✅ |  |
| node/test_pow/model.onnx | ✅ |  |
| node/test_pow_bcast_array/model.onnx | ✅ |  |
| node/test_pow_bcast_scalar/model.onnx | ✅ |  |
| node/test_pow_example/model.onnx | ✅ |  |
| node/test_pow_types_float32_int32/model.onnx | ❌ | Pow expects matching dtypes, got float, int32 |
| node/test_pow_types_float32_int64/model.onnx | ❌ | Pow expects matching dtypes, got float, int64 |
| node/test_pow_types_float32_uint32/model.onnx | ❌ | Pow expects matching dtypes, got float, uint32 |
| node/test_pow_types_float32_uint64/model.onnx | ❌ | Pow expects matching dtypes, got float, uint64 |
| node/test_pow_types_int32_float32/model.onnx | ❌ | Pow expects matching dtypes, got float, int32 |
| node/test_pow_types_int32_int32/model.onnx | ✅ |  |
| node/test_pow_types_int64_float32/model.onnx | ❌ | Pow expects matching dtypes, got float, int64 |
| node/test_pow_types_int64_int64/model.onnx | ✅ |  |
| node/test_prelu_broadcast/model.onnx | ✅ |  |
| node/test_prelu_broadcast_expanded/model.onnx | ✅ |  |
| node/test_prelu_example/model.onnx | ✅ |  |
| node/test_prelu_example_expanded/model.onnx | ✅ |  |
| node/test_qlinearconv/model.onnx | ❌ | Unsupported op QLinearConv |
| node/test_qlinearmatmul_2D_int8_float16/model.onnx | ❌ | Unsupported op QLinearMatMul |
| node/test_qlinearmatmul_2D_int8_float32/model.onnx | ❌ | Unsupported op QLinearMatMul |
| node/test_qlinearmatmul_2D_uint8_float16/model.onnx | ❌ | Unsupported op QLinearMatMul |
| node/test_qlinearmatmul_2D_uint8_float32/model.onnx | ❌ | Unsupported op QLinearMatMul |
| node/test_qlinearmatmul_3D_int8_float16/model.onnx | ❌ | Unsupported op QLinearMatMul |
| node/test_qlinearmatmul_3D_int8_float32/model.onnx | ❌ | Unsupported op QLinearMatMul |
| node/test_qlinearmatmul_3D_uint8_float16/model.onnx | ❌ | Unsupported op QLinearMatMul |
| node/test_qlinearmatmul_3D_uint8_float32/model.onnx | ❌ | Unsupported op QLinearMatMul |
| node/test_quantizelinear/model.onnx | ❌ | Unsupported op QuantizeLinear |
| node/test_quantizelinear_axis/model.onnx | ❌ | Unsupported op QuantizeLinear |
| node/test_quantizelinear_blocked_asymmetric/model.onnx | ❌ | Unsupported op QuantizeLinear |
| node/test_quantizelinear_blocked_symmetric/model.onnx | ❌ | Unsupported op QuantizeLinear |
| node/test_quantizelinear_e4m3fn/model.onnx | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'y_zero_point'. |
| node/test_quantizelinear_e5m2/model.onnx | ❌ | Unsupported elem_type 19 (FLOAT8E5M2) for tensor 'y_zero_point'. |
| node/test_quantizelinear_float4e2m1/model.onnx | ❌ | Unsupported elem_type 23 (FLOAT4E2M1) for tensor 'y_zero_point'. |
| node/test_quantizelinear_int16/model.onnx | ❌ | Unsupported op QuantizeLinear |
| node/test_quantizelinear_int2/model.onnx | ❌ | Unsupported elem_type 26 (INT2) for tensor 'y_zero_point'. |
| node/test_quantizelinear_int4/model.onnx | ❌ | Unsupported elem_type 22 (INT4) for tensor 'y_zero_point'. |
| node/test_quantizelinear_uint16/model.onnx | ❌ | Unsupported op QuantizeLinear |
| node/test_quantizelinear_uint2/model.onnx | ❌ | Unsupported elem_type 25 (UINT2) for tensor 'y_zero_point'. |
| node/test_quantizelinear_uint4/model.onnx | ❌ | Unsupported elem_type 21 (UINT4) for tensor 'y_zero_point'. |
| node/test_range_float_type_positive_delta/model.onnx | ❌ | Unsupported op Range |
| node/test_range_float_type_positive_delta_expanded/model.onnx | ❌ | Unsupported op Loop |
| node/test_range_int32_type_negative_delta/model.onnx | ❌ | Unsupported op Range |
| node/test_range_int32_type_negative_delta_expanded/model.onnx | ❌ | Unsupported op Loop |
| node/test_reciprocal/model.onnx | ✅ |  |
| node/test_reciprocal_example/model.onnx | ✅ |  |
| node/test_reduce_l1_default_axes_keepdims_example/model.onnx | ✅ |  |
| node/test_reduce_l1_default_axes_keepdims_example_expanded/model.onnx | ✅ |  |
| node/test_reduce_l1_default_axes_keepdims_random/model.onnx | ✅ |  |
| node/test_reduce_l1_default_axes_keepdims_random_expanded/model.onnx | ✅ |  |
| node/test_reduce_l1_do_not_keepdims_example/model.onnx | ✅ |  |
| node/test_reduce_l1_do_not_keepdims_example_expanded/model.onnx | ✅ |  |
| node/test_reduce_l1_do_not_keepdims_random/model.onnx | ✅ |  |
| node/test_reduce_l1_do_not_keepdims_random_expanded/model.onnx | ✅ |  |
| node/test_reduce_l1_empty_set/model.onnx | ❌ | Dynamic or zero dims are not supported |
| node/test_reduce_l1_empty_set_expanded/model.onnx | ❌ | Dynamic or zero dims are not supported |
| node/test_reduce_l1_keep_dims_example/model.onnx | ✅ |  |
| node/test_reduce_l1_keep_dims_example_expanded/model.onnx | ✅ |  |
| node/test_reduce_l1_keep_dims_random/model.onnx | ✅ |  |
| node/test_reduce_l1_keep_dims_random_expanded/model.onnx | ✅ |  |
| node/test_reduce_l1_negative_axes_keep_dims_example/model.onnx | ✅ |  |
| node/test_reduce_l1_negative_axes_keep_dims_example_expanded/model.onnx | ✅ |  |
| node/test_reduce_l1_negative_axes_keep_dims_random/model.onnx | ✅ |  |
| node/test_reduce_l1_negative_axes_keep_dims_random_expanded/model.onnx | ✅ |  |
| node/test_reduce_l2_default_axes_keepdims_example/model.onnx | ✅ |  |
| node/test_reduce_l2_default_axes_keepdims_example_expanded/model.onnx | ❌ | ReduceSum output shape must be (1, 1, 1), got () |
| node/test_reduce_l2_default_axes_keepdims_random/model.onnx | ✅ |  |
| node/test_reduce_l2_default_axes_keepdims_random_expanded/model.onnx | ❌ | ReduceSum output shape must be (1, 1, 1), got () |
| node/test_reduce_l2_do_not_keepdims_example/model.onnx | ✅ |  |
| node/test_reduce_l2_do_not_keepdims_example_expanded/model.onnx | ❌ | CastLike input and output shapes must match |
| node/test_reduce_l2_do_not_keepdims_random/model.onnx | ✅ |  |
| node/test_reduce_l2_do_not_keepdims_random_expanded/model.onnx | ❌ | CastLike input and output shapes must match |
| node/test_reduce_l2_empty_set/model.onnx | ❌ | Dynamic or zero dims are not supported |
| node/test_reduce_l2_empty_set_expanded/model.onnx | ❌ | ReduceSum output shape rank must match input rank |
| node/test_reduce_l2_keep_dims_example/model.onnx | ✅ |  |
| node/test_reduce_l2_keep_dims_example_expanded/model.onnx | ❌ | ReduceSum output shape rank must match input rank |
| node/test_reduce_l2_keep_dims_random/model.onnx | ✅ |  |
| node/test_reduce_l2_keep_dims_random_expanded/model.onnx | ❌ | ReduceSum output shape rank must match input rank |
| node/test_reduce_l2_negative_axes_keep_dims_example/model.onnx | ✅ |  |
| node/test_reduce_l2_negative_axes_keep_dims_example_expanded/model.onnx | ❌ | ReduceSum output shape rank must match input rank |
| node/test_reduce_l2_negative_axes_keep_dims_random/model.onnx | ✅ |  |
| node/test_reduce_l2_negative_axes_keep_dims_random_expanded/model.onnx | ❌ | ReduceSum output shape rank must match input rank |
| node/test_reduce_log_sum_asc_axes/model.onnx | ✅ |  |
| node/test_reduce_log_sum_asc_axes_expanded/model.onnx | ✅ |  |
| node/test_reduce_log_sum_default/model.onnx | ✅ |  |
| node/test_reduce_log_sum_default_expanded/model.onnx | ❌ | ReduceSum output shape must be (1, 1, 1), got () |
| node/test_reduce_log_sum_desc_axes/model.onnx | ✅ |  |
| node/test_reduce_log_sum_desc_axes_expanded/model.onnx | ✅ |  |
| node/test_reduce_log_sum_empty_set/model.onnx | ❌ | Dynamic or zero dims are not supported |
| node/test_reduce_log_sum_empty_set_expanded/model.onnx | ❌ | ReduceSum output shape rank must match input rank |
| node/test_reduce_log_sum_exp_default_axes_keepdims_example/model.onnx | ✅ |  |
| node/test_reduce_log_sum_exp_default_axes_keepdims_example_expanded/model.onnx | ❌ | ReduceSum output shape must be (1, 1, 1), got () |
| node/test_reduce_log_sum_exp_default_axes_keepdims_random/model.onnx | ✅ |  |
| node/test_reduce_log_sum_exp_default_axes_keepdims_random_expanded/model.onnx | ❌ | ReduceSum output shape must be (1, 1, 1), got () |
| node/test_reduce_log_sum_exp_do_not_keepdims_example/model.onnx | ✅ |  |
| node/test_reduce_log_sum_exp_do_not_keepdims_example_expanded/model.onnx | ❌ | CastLike input and output shapes must match |
| node/test_reduce_log_sum_exp_do_not_keepdims_random/model.onnx | ✅ |  |
| node/test_reduce_log_sum_exp_do_not_keepdims_random_expanded/model.onnx | ❌ | CastLike input and output shapes must match |
| node/test_reduce_log_sum_exp_empty_set/model.onnx | ❌ | Dynamic or zero dims are not supported |
| node/test_reduce_log_sum_exp_empty_set_expanded/model.onnx | ❌ | ReduceSum output shape rank must match input rank |
| node/test_reduce_log_sum_exp_keepdims_example/model.onnx | ✅ |  |
| node/test_reduce_log_sum_exp_keepdims_example_expanded/model.onnx | ❌ | ReduceSum output shape rank must match input rank |
| node/test_reduce_log_sum_exp_keepdims_random/model.onnx | ✅ |  |
| node/test_reduce_log_sum_exp_keepdims_random_expanded/model.onnx | ❌ | ReduceSum output shape rank must match input rank |
| node/test_reduce_log_sum_exp_negative_axes_keepdims_example/model.onnx | ✅ |  |
| node/test_reduce_log_sum_exp_negative_axes_keepdims_example_expanded/model.onnx | ❌ | ReduceSum output shape rank must match input rank |
| node/test_reduce_log_sum_exp_negative_axes_keepdims_random/model.onnx | ✅ |  |
| node/test_reduce_log_sum_exp_negative_axes_keepdims_random_expanded/model.onnx | ❌ | ReduceSum output shape rank must match input rank |
| node/test_reduce_log_sum_negative_axes/model.onnx | ✅ |  |
| node/test_reduce_log_sum_negative_axes_expanded/model.onnx | ❌ | ReduceSum output shape rank must match input rank |
| node/test_reduce_max_bool_inputs/model.onnx | ❌ | ReduceMax does not support dtype bool |
| node/test_reduce_max_default_axes_keepdim_example/model.onnx | ✅ |  |
| node/test_reduce_max_default_axes_keepdims_random/model.onnx | ✅ |  |
| node/test_reduce_max_do_not_keepdims_example/model.onnx | ✅ |  |
| node/test_reduce_max_do_not_keepdims_random/model.onnx | ✅ |  |
| node/test_reduce_max_empty_set/model.onnx | ❌ | Dynamic or zero dims are not supported |
| node/test_reduce_max_keepdims_example/model.onnx | ✅ |  |
| node/test_reduce_max_keepdims_random/model.onnx | ✅ |  |
| node/test_reduce_max_negative_axes_keepdims_example/model.onnx | ✅ |  |
| node/test_reduce_max_negative_axes_keepdims_random/model.onnx | ✅ |  |
| node/test_reduce_mean_default_axes_keepdims_example/model.onnx | ✅ |  |
| node/test_reduce_mean_default_axes_keepdims_random/model.onnx | ✅ |  |
| node/test_reduce_mean_do_not_keepdims_example/model.onnx | ✅ |  |
| node/test_reduce_mean_do_not_keepdims_random/model.onnx | ✅ |  |
| node/test_reduce_mean_keepdims_example/model.onnx | ✅ |  |
| node/test_reduce_mean_keepdims_random/model.onnx | ✅ |  |
| node/test_reduce_mean_negative_axes_keepdims_example/model.onnx | ✅ |  |
| node/test_reduce_mean_negative_axes_keepdims_random/model.onnx | ✅ |  |
| node/test_reduce_min_bool_inputs/model.onnx | ❌ | ReduceMin does not support dtype bool |
| node/test_reduce_min_default_axes_keepdims_example/model.onnx | ✅ |  |
| node/test_reduce_min_default_axes_keepdims_random/model.onnx | ✅ |  |
| node/test_reduce_min_do_not_keepdims_example/model.onnx | ✅ |  |
| node/test_reduce_min_do_not_keepdims_random/model.onnx | ✅ |  |
| node/test_reduce_min_empty_set/model.onnx | ❌ | Dynamic or zero dims are not supported |
| node/test_reduce_min_keepdims_example/model.onnx | ✅ |  |
| node/test_reduce_min_keepdims_random/model.onnx | ✅ |  |
| node/test_reduce_min_negative_axes_keepdims_example/model.onnx | ✅ |  |
| node/test_reduce_min_negative_axes_keepdims_random/model.onnx | ✅ |  |
| node/test_reduce_prod_default_axes_keepdims_example/model.onnx | ✅ |  |
| node/test_reduce_prod_default_axes_keepdims_random/model.onnx | ✅ |  |
| node/test_reduce_prod_do_not_keepdims_example/model.onnx | ✅ |  |
| node/test_reduce_prod_do_not_keepdims_random/model.onnx | ✅ |  |
| node/test_reduce_prod_empty_set/model.onnx | ❌ | Dynamic or zero dims are not supported |
| node/test_reduce_prod_keepdims_example/model.onnx | ✅ |  |
| node/test_reduce_prod_keepdims_random/model.onnx | ✅ |  |
| node/test_reduce_prod_negative_axes_keepdims_example/model.onnx | ✅ |  |
| node/test_reduce_prod_negative_axes_keepdims_random/model.onnx | ✅ |  |
| node/test_reduce_sum_default_axes_keepdims_example/model.onnx | ✅ |  |
| node/test_reduce_sum_default_axes_keepdims_random/model.onnx | ✅ |  |
| node/test_reduce_sum_do_not_keepdims_example/model.onnx | ✅ |  |
| node/test_reduce_sum_do_not_keepdims_random/model.onnx | ✅ |  |
| node/test_reduce_sum_empty_axes_input_noop/model.onnx | ✅ |  |
| node/test_reduce_sum_empty_axes_input_noop_example/model.onnx | ✅ |  |
| node/test_reduce_sum_empty_set/model.onnx | ❌ | Dynamic or zero dims are not supported |
| node/test_reduce_sum_empty_set_non_reduced_axis_zero/model.onnx | ❌ | Dynamic or zero dims are not supported |
| node/test_reduce_sum_keepdims_example/model.onnx | ✅ |  |
| node/test_reduce_sum_keepdims_random/model.onnx | ✅ |  |
| node/test_reduce_sum_negative_axes_keepdims_example/model.onnx | ✅ |  |
| node/test_reduce_sum_negative_axes_keepdims_random/model.onnx | ✅ |  |
| node/test_reduce_sum_square_default_axes_keepdims_example/model.onnx | ✅ |  |
| node/test_reduce_sum_square_default_axes_keepdims_example_expanded/model.onnx | ✅ |  |
| node/test_reduce_sum_square_default_axes_keepdims_random/model.onnx | ✅ |  |
| node/test_reduce_sum_square_default_axes_keepdims_random_expanded/model.onnx | ✅ |  |
| node/test_reduce_sum_square_do_not_keepdims_example/model.onnx | ✅ |  |
| node/test_reduce_sum_square_do_not_keepdims_example_expanded/model.onnx | ✅ |  |
| node/test_reduce_sum_square_do_not_keepdims_random/model.onnx | ✅ |  |
| node/test_reduce_sum_square_do_not_keepdims_random_expanded/model.onnx | ✅ |  |
| node/test_reduce_sum_square_empty_set/model.onnx | ❌ | Dynamic or zero dims are not supported |
| node/test_reduce_sum_square_empty_set_expanded/model.onnx | ❌ | Dynamic or zero dims are not supported |
| node/test_reduce_sum_square_keepdims_example/model.onnx | ✅ |  |
| node/test_reduce_sum_square_keepdims_example_expanded/model.onnx | ✅ |  |
| node/test_reduce_sum_square_keepdims_random/model.onnx | ✅ |  |
| node/test_reduce_sum_square_keepdims_random_expanded/model.onnx | ✅ |  |
| node/test_reduce_sum_square_negative_axes_keepdims_example/model.onnx | ✅ |  |
| node/test_reduce_sum_square_negative_axes_keepdims_example_expanded/model.onnx | ✅ |  |
| node/test_reduce_sum_square_negative_axes_keepdims_random/model.onnx | ✅ |  |
| node/test_reduce_sum_square_negative_axes_keepdims_random_expanded/model.onnx | ✅ |  |
| node/test_reflect_pad/model.onnx | ❌ | Unsupported op Pad |
| node/test_regex_full_match_basic/model.onnx | ❌ | Unsupported elem_type 8 (STRING) for tensor 'X'. |
| node/test_regex_full_match_email_domain/model.onnx | ❌ | Unsupported elem_type 8 (STRING) for tensor 'X'. |
| node/test_regex_full_match_empty/model.onnx | ❌ | Unsupported elem_type 8 (STRING) for tensor 'X'. |
| node/test_relu/model.onnx | ✅ |  |
| node/test_relu_expanded_ver18/model.onnx | ✅ |  |
| node/test_reshape_allowzero_reordered/model.onnx | ❌ | Dynamic or zero dims are not supported |
| node/test_reshape_extended_dims/model.onnx | ✅ |  |
| node/test_reshape_negative_dim/model.onnx | ✅ |  |
| node/test_reshape_negative_extended_dims/model.onnx | ✅ |  |
| node/test_reshape_one_dim/model.onnx | ✅ |  |
| node/test_reshape_reduced_dims/model.onnx | ✅ |  |
| node/test_reshape_reordered_all_dims/model.onnx | ✅ |  |
| node/test_reshape_reordered_last_dims/model.onnx | ✅ |  |
| node/test_reshape_zero_and_negative_dim/model.onnx | ✅ |  |
| node/test_reshape_zero_dim/model.onnx | ✅ |  |
| node/test_resize_downsample_scales_cubic/model.onnx | ✅ |  |
| node/test_resize_downsample_scales_cubic_A_n0p5_exclude_outside/model.onnx | ✅ |  |
| node/test_resize_downsample_scales_cubic_align_corners/model.onnx | ✅ |  |
| node/test_resize_downsample_scales_cubic_antialias/model.onnx | ✅ |  |
| node/test_resize_downsample_scales_linear/model.onnx | ✅ |  |
| node/test_resize_downsample_scales_linear_align_corners/model.onnx | ✅ |  |
| node/test_resize_downsample_scales_linear_antialias/model.onnx | ✅ |  |
| node/test_resize_downsample_scales_linear_half_pixel_symmetric/model.onnx | ✅ |  |
| node/test_resize_downsample_scales_nearest/model.onnx | ✅ |  |
| node/test_resize_downsample_sizes_cubic/model.onnx | ✅ |  |
| node/test_resize_downsample_sizes_cubic_antialias/model.onnx | ✅ |  |
| node/test_resize_downsample_sizes_linear_antialias/model.onnx | ✅ |  |
| node/test_resize_downsample_sizes_linear_pytorch_half_pixel/model.onnx | ✅ |  |
| node/test_resize_downsample_sizes_nearest/model.onnx | ✅ |  |
| node/test_resize_downsample_sizes_nearest_not_larger/model.onnx | ✅ |  |
| node/test_resize_downsample_sizes_nearest_not_smaller/model.onnx | ✅ |  |
| node/test_resize_tf_crop_and_resize/model.onnx | ✅ |  |
| node/test_resize_tf_crop_and_resize_axes_2_3/model.onnx | ✅ |  |
| node/test_resize_tf_crop_and_resize_axes_3_2/model.onnx | ✅ |  |
| node/test_resize_tf_crop_and_resize_extrapolation_value/model.onnx | ✅ |  |
| node/test_resize_upsample_scales_cubic/model.onnx | ✅ |  |
| node/test_resize_upsample_scales_cubic_A_n0p5_exclude_outside/model.onnx | ✅ |  |
| node/test_resize_upsample_scales_cubic_align_corners/model.onnx | ✅ |  |
| node/test_resize_upsample_scales_cubic_asymmetric/model.onnx | ✅ |  |
| node/test_resize_upsample_scales_linear/model.onnx | ✅ |  |
| node/test_resize_upsample_scales_linear_align_corners/model.onnx | ✅ |  |
| node/test_resize_upsample_scales_linear_half_pixel_symmetric/model.onnx | ✅ |  |
| node/test_resize_upsample_scales_nearest/model.onnx | ✅ |  |
| node/test_resize_upsample_scales_nearest_axes_2_3/model.onnx | ✅ |  |
| node/test_resize_upsample_scales_nearest_axes_3_2/model.onnx | ✅ |  |
| node/test_resize_upsample_sizes_cubic/model.onnx | ✅ |  |
| node/test_resize_upsample_sizes_nearest/model.onnx | ✅ |  |
| node/test_resize_upsample_sizes_nearest_axes_2_3/model.onnx | ✅ |  |
| node/test_resize_upsample_sizes_nearest_axes_3_2/model.onnx | ✅ |  |
| node/test_resize_upsample_sizes_nearest_ceil_half_pixel/model.onnx | ✅ |  |
| node/test_resize_upsample_sizes_nearest_floor_align_corners/model.onnx | ✅ |  |
| node/test_resize_upsample_sizes_nearest_not_larger/model.onnx | ✅ |  |
| node/test_resize_upsample_sizes_nearest_not_smaller/model.onnx | ✅ |  |
| node/test_resize_upsample_sizes_nearest_round_prefer_ceil_asymmetric/model.onnx | ✅ |  |
| node/test_reversesequence_batch/model.onnx | ❌ | Unsupported op ReverseSequence |
| node/test_reversesequence_time/model.onnx | ❌ | Unsupported op ReverseSequence |
| node/test_rms_normalization_2d_axis0/model.onnx | ❌ | Unsupported op RMSNormalization |
| node/test_rms_normalization_2d_axis0_expanded/model.onnx | ❌ | Dynamic dim for tensor 'RMSNormalization_test_rms_normalization_2d_axis0_expanded_function_ReduceAxes' |
| node/test_rms_normalization_2d_axis1/model.onnx | ❌ | Unsupported op RMSNormalization |
| node/test_rms_normalization_2d_axis1_expanded/model.onnx | ❌ | Dynamic dim for tensor 'RMSNormalization_test_rms_normalization_2d_axis1_expanded_function_ReduceAxes' |
| node/test_rms_normalization_2d_axis_negative_1/model.onnx | ❌ | Unsupported op RMSNormalization |
| node/test_rms_normalization_2d_axis_negative_1_expanded/model.onnx | ❌ | Dynamic dim for tensor 'RMSNormalization_test_rms_normalization_2d_axis_negative_1_expanded_function_ReduceAxes' |
| node/test_rms_normalization_2d_axis_negative_2/model.onnx | ❌ | Unsupported op RMSNormalization |
| node/test_rms_normalization_2d_axis_negative_2_expanded/model.onnx | ❌ | Dynamic dim for tensor 'RMSNormalization_test_rms_normalization_2d_axis_negative_2_expanded_function_ReduceAxes' |
| node/test_rms_normalization_3d_axis0_epsilon/model.onnx | ❌ | Unsupported op RMSNormalization |
| node/test_rms_normalization_3d_axis0_epsilon_expanded/model.onnx | ❌ | Dynamic dim for tensor 'RMSNormalization_test_rms_normalization_3d_axis0_epsilon_expanded_function_ReduceAxes' |
| node/test_rms_normalization_3d_axis1_epsilon/model.onnx | ❌ | Unsupported op RMSNormalization |
| node/test_rms_normalization_3d_axis1_epsilon_expanded/model.onnx | ❌ | Dynamic dim for tensor 'RMSNormalization_test_rms_normalization_3d_axis1_epsilon_expanded_function_ReduceAxes' |
| node/test_rms_normalization_3d_axis2_epsilon/model.onnx | ❌ | Unsupported op RMSNormalization |
| node/test_rms_normalization_3d_axis2_epsilon_expanded/model.onnx | ❌ | Dynamic dim for tensor 'RMSNormalization_test_rms_normalization_3d_axis2_epsilon_expanded_function_ReduceAxes' |
| node/test_rms_normalization_3d_axis_negative_1_epsilon/model.onnx | ❌ | Unsupported op RMSNormalization |
| node/test_rms_normalization_3d_axis_negative_1_epsilon_expanded/model.onnx | ❌ | Dynamic dim for tensor 'RMSNormalization_test_rms_normalization_3d_axis_negative_1_epsilon_expanded_function_ReduceAxes' |
| node/test_rms_normalization_3d_axis_negative_2_epsilon/model.onnx | ❌ | Unsupported op RMSNormalization |
| node/test_rms_normalization_3d_axis_negative_2_epsilon_expanded/model.onnx | ❌ | Dynamic dim for tensor 'RMSNormalization_test_rms_normalization_3d_axis_negative_2_epsilon_expanded_function_ReduceAxes' |
| node/test_rms_normalization_3d_axis_negative_3_epsilon/model.onnx | ❌ | Unsupported op RMSNormalization |
| node/test_rms_normalization_3d_axis_negative_3_epsilon_expanded/model.onnx | ❌ | Dynamic dim for tensor 'RMSNormalization_test_rms_normalization_3d_axis_negative_3_epsilon_expanded_function_ReduceAxes' |
| node/test_rms_normalization_4d_axis0/model.onnx | ❌ | Unsupported op RMSNormalization |
| node/test_rms_normalization_4d_axis0_expanded/model.onnx | ❌ | Dynamic dim for tensor 'RMSNormalization_test_rms_normalization_4d_axis0_expanded_function_ReduceAxes' |
| node/test_rms_normalization_4d_axis1/model.onnx | ❌ | Unsupported op RMSNormalization |
| node/test_rms_normalization_4d_axis1_expanded/model.onnx | ❌ | Dynamic dim for tensor 'RMSNormalization_test_rms_normalization_4d_axis1_expanded_function_ReduceAxes' |
| node/test_rms_normalization_4d_axis2/model.onnx | ❌ | Unsupported op RMSNormalization |
| node/test_rms_normalization_4d_axis2_expanded/model.onnx | ❌ | Dynamic dim for tensor 'RMSNormalization_test_rms_normalization_4d_axis2_expanded_function_ReduceAxes' |
| node/test_rms_normalization_4d_axis3/model.onnx | ❌ | Unsupported op RMSNormalization |
| node/test_rms_normalization_4d_axis3_expanded/model.onnx | ❌ | Dynamic dim for tensor 'RMSNormalization_test_rms_normalization_4d_axis3_expanded_function_ReduceAxes' |
| node/test_rms_normalization_4d_axis_negative_1/model.onnx | ❌ | Unsupported op RMSNormalization |
| node/test_rms_normalization_4d_axis_negative_1_expanded/model.onnx | ❌ | Dynamic dim for tensor 'RMSNormalization_test_rms_normalization_4d_axis_negative_1_expanded_function_ReduceAxes' |
| node/test_rms_normalization_4d_axis_negative_2/model.onnx | ❌ | Unsupported op RMSNormalization |
| node/test_rms_normalization_4d_axis_negative_2_expanded/model.onnx | ❌ | Dynamic dim for tensor 'RMSNormalization_test_rms_normalization_4d_axis_negative_2_expanded_function_ReduceAxes' |
| node/test_rms_normalization_4d_axis_negative_3/model.onnx | ❌ | Unsupported op RMSNormalization |
| node/test_rms_normalization_4d_axis_negative_3_expanded/model.onnx | ❌ | Dynamic dim for tensor 'RMSNormalization_test_rms_normalization_4d_axis_negative_3_expanded_function_ReduceAxes' |
| node/test_rms_normalization_4d_axis_negative_4/model.onnx | ❌ | Unsupported op RMSNormalization |
| node/test_rms_normalization_4d_axis_negative_4_expanded/model.onnx | ❌ | Dynamic dim for tensor 'RMSNormalization_test_rms_normalization_4d_axis_negative_4_expanded_function_ReduceAxes' |
| node/test_rms_normalization_default_axis/model.onnx | ❌ | Unsupported op RMSNormalization |
| node/test_rms_normalization_default_axis_expanded/model.onnx | ❌ | Dynamic dim for tensor 'RMSNormalization_test_rms_normalization_default_axis_expanded_function_ReduceAxes' |
| node/test_rnn_seq_length/model.onnx | ❌ | Unsupported op RNN |
| node/test_roialign_aligned_false/model.onnx | ❌ | Unsupported op RoiAlign |
| node/test_roialign_aligned_true/model.onnx | ❌ | Unsupported op RoiAlign |
| node/test_roialign_mode_max/model.onnx | ❌ | Unsupported op RoiAlign |
| node/test_rotary_embedding/model.onnx | ❌ | Unsupported op RotaryEmbedding |
| node/test_rotary_embedding_3d_input/model.onnx | ❌ | Unsupported op RotaryEmbedding |
| node/test_rotary_embedding_3d_input_expanded/model.onnx | ❌ | Dynamic dim for tensor 'RotaryEmbedding_test_rotary_embedding_3d_input_expanded_function_CosCacheSliced' |
| node/test_rotary_embedding_expanded/model.onnx | ❌ | Dynamic dim for tensor 'RotaryEmbedding_test_rotary_embedding_expanded_function_CosCacheSliced' |
| node/test_rotary_embedding_interleaved/model.onnx | ❌ | Unsupported op RotaryEmbedding |
| node/test_rotary_embedding_interleaved_expanded/model.onnx | ❌ | Dynamic dim for tensor 'RotaryEmbedding_test_rotary_embedding_interleaved_expanded_function_CosCacheSliced' |
| node/test_rotary_embedding_no_position_ids/model.onnx | ❌ | Unsupported op RotaryEmbedding |
| node/test_rotary_embedding_no_position_ids_expanded/model.onnx | ❌ | Dynamic dim for tensor 'RotaryEmbedding_test_rotary_embedding_no_position_ids_expanded_function_CosCacheSliced' |
| node/test_rotary_embedding_no_position_ids_interleaved/model.onnx | ❌ | Unsupported op RotaryEmbedding |
| node/test_rotary_embedding_no_position_ids_interleaved_expanded/model.onnx | ❌ | Dynamic dim for tensor 'RotaryEmbedding_test_rotary_embedding_no_position_ids_interleaved_expanded_function_CosCacheSliced' |
| node/test_rotary_embedding_no_position_ids_rotary_dim/model.onnx | ❌ | Unsupported op RotaryEmbedding |
| node/test_rotary_embedding_no_position_ids_rotary_dim_expanded/model.onnx | ❌ | Dynamic dim for tensor 'RotaryEmbedding_test_rotary_embedding_no_position_ids_rotary_dim_expanded_function_CosCacheSliced' |
| node/test_rotary_embedding_with_interleaved_rotary_dim/model.onnx | ❌ | Unsupported op RotaryEmbedding |
| node/test_rotary_embedding_with_interleaved_rotary_dim_expanded/model.onnx | ❌ | Dynamic dim for tensor 'RotaryEmbedding_test_rotary_embedding_with_interleaved_rotary_dim_expanded_function_CosCacheSliced' |
| node/test_rotary_embedding_with_rotary_dim/model.onnx | ❌ | Unsupported op RotaryEmbedding |
| node/test_rotary_embedding_with_rotary_dim_expanded/model.onnx | ❌ | Dynamic dim for tensor 'RotaryEmbedding_test_rotary_embedding_with_rotary_dim_expanded_function_CosCacheSliced' |
| node/test_round/model.onnx | ✅ |  |
| node/test_scan9_sum/model.onnx | ❌ | Unsupported op Scan |
| node/test_scan_sum/model.onnx | ❌ | Unsupported op Scan |
| node/test_scatter_elements_with_axis/model.onnx | ❌ | Unsupported op ScatterElements |
| node/test_scatter_elements_with_duplicate_indices/model.onnx | ❌ | Unsupported op ScatterElements |
| node/test_scatter_elements_with_negative_indices/model.onnx | ❌ | Unsupported op ScatterElements |
| node/test_scatter_elements_with_reduction_max/model.onnx | ❌ | Unsupported op ScatterElements |
| node/test_scatter_elements_with_reduction_min/model.onnx | ❌ | Unsupported op ScatterElements |
| node/test_scatter_elements_without_axis/model.onnx | ❌ | Unsupported op ScatterElements |
| node/test_scatter_with_axis/model.onnx | ❌ | Unsupported op Scatter |
| node/test_scatter_without_axis/model.onnx | ❌ | Unsupported op Scatter |
| node/test_scatternd/model.onnx | ❌ | Unsupported op ScatterND |
| node/test_scatternd_add/model.onnx | ❌ | Unsupported op ScatterND |
| node/test_scatternd_max/model.onnx | ❌ | Unsupported op ScatterND |
| node/test_scatternd_min/model.onnx | ❌ | Unsupported op ScatterND |
| node/test_scatternd_multiply/model.onnx | ❌ | Unsupported op ScatterND |
| node/test_sce_NCd1_mean_weight_negative_ii/model.onnx | ✅ |  |
| node/test_sce_NCd1_mean_weight_negative_ii_expanded/model.onnx | ✅ |  |
| node/test_sce_NCd1_mean_weight_negative_ii_log_prob/model.onnx | ✅ |  |
| node/test_sce_NCd1_mean_weight_negative_ii_log_prob_expanded/model.onnx | ✅ |  |
| node/test_sce_NCd1d2d3_none_no_weight_negative_ii/model.onnx | ✅ |  |
| node/test_sce_NCd1d2d3_none_no_weight_negative_ii_expanded/model.onnx | ✅ |  |
| node/test_sce_NCd1d2d3_none_no_weight_negative_ii_log_prob/model.onnx | ✅ |  |
| node/test_sce_NCd1d2d3_none_no_weight_negative_ii_log_prob_expanded/model.onnx | ✅ |  |
| node/test_sce_NCd1d2d3_sum_weight_high_ii/model.onnx | ✅ |  |
| node/test_sce_NCd1d2d3_sum_weight_high_ii_expanded/model.onnx | ✅ |  |
| node/test_sce_NCd1d2d3_sum_weight_high_ii_log_prob/model.onnx | ✅ |  |
| node/test_sce_NCd1d2d3_sum_weight_high_ii_log_prob_expanded/model.onnx | ✅ |  |
| node/test_sce_NCd1d2d3d4d5_mean_weight/model.onnx | ✅ |  |
| node/test_sce_NCd1d2d3d4d5_mean_weight_expanded/model.onnx | ✅ |  |
| node/test_sce_NCd1d2d3d4d5_mean_weight_log_prob/model.onnx | ✅ |  |
| node/test_sce_NCd1d2d3d4d5_mean_weight_log_prob_expanded/model.onnx | ✅ |  |
| node/test_sce_NCd1d2d3d4d5_none_no_weight/model.onnx | ✅ |  |
| node/test_sce_NCd1d2d3d4d5_none_no_weight_expanded/model.onnx | ✅ |  |
| node/test_sce_NCd1d2d3d4d5_none_no_weight_log_prob/model.onnx | ✅ |  |
| node/test_sce_NCd1d2d3d4d5_none_no_weight_log_prob_expanded/model.onnx | ✅ |  |
| node/test_sce_mean/model.onnx | ✅ |  |
| node/test_sce_mean_3d/model.onnx | ✅ |  |
| node/test_sce_mean_3d_expanded/model.onnx | ✅ |  |
| node/test_sce_mean_3d_log_prob/model.onnx | ✅ |  |
| node/test_sce_mean_3d_log_prob_expanded/model.onnx | ✅ |  |
| node/test_sce_mean_expanded/model.onnx | ✅ |  |
| node/test_sce_mean_log_prob/model.onnx | ✅ |  |
| node/test_sce_mean_log_prob_expanded/model.onnx | ✅ |  |
| node/test_sce_mean_no_weight_ii/model.onnx | ✅ |  |
| node/test_sce_mean_no_weight_ii_3d/model.onnx | ✅ |  |
| node/test_sce_mean_no_weight_ii_3d_expanded/model.onnx | ✅ |  |
| node/test_sce_mean_no_weight_ii_3d_log_prob/model.onnx | ✅ |  |
| node/test_sce_mean_no_weight_ii_3d_log_prob_expanded/model.onnx | ✅ |  |
| node/test_sce_mean_no_weight_ii_4d/model.onnx | ✅ |  |
| node/test_sce_mean_no_weight_ii_4d_expanded/model.onnx | ✅ |  |
| node/test_sce_mean_no_weight_ii_4d_log_prob/model.onnx | ✅ |  |
| node/test_sce_mean_no_weight_ii_4d_log_prob_expanded/model.onnx | ✅ |  |
| node/test_sce_mean_no_weight_ii_expanded/model.onnx | ✅ |  |
| node/test_sce_mean_no_weight_ii_log_prob/model.onnx | ✅ |  |
| node/test_sce_mean_no_weight_ii_log_prob_expanded/model.onnx | ✅ |  |
| node/test_sce_mean_weight/model.onnx | ✅ |  |
| node/test_sce_mean_weight_expanded/model.onnx | ✅ |  |
| node/test_sce_mean_weight_ii/model.onnx | ✅ |  |
| node/test_sce_mean_weight_ii_3d/model.onnx | ✅ |  |
| node/test_sce_mean_weight_ii_3d_expanded/model.onnx | ✅ |  |
| node/test_sce_mean_weight_ii_3d_log_prob/model.onnx | ✅ |  |
| node/test_sce_mean_weight_ii_3d_log_prob_expanded/model.onnx | ✅ |  |
| node/test_sce_mean_weight_ii_4d/model.onnx | ✅ |  |
| node/test_sce_mean_weight_ii_4d_expanded/model.onnx | ✅ |  |
| node/test_sce_mean_weight_ii_4d_log_prob/model.onnx | ✅ |  |
| node/test_sce_mean_weight_ii_4d_log_prob_expanded/model.onnx | ✅ |  |
| node/test_sce_mean_weight_ii_expanded/model.onnx | ✅ |  |
| node/test_sce_mean_weight_ii_log_prob/model.onnx | ✅ |  |
| node/test_sce_mean_weight_ii_log_prob_expanded/model.onnx | ✅ |  |
| node/test_sce_mean_weight_log_prob/model.onnx | ✅ |  |
| node/test_sce_mean_weight_log_prob_expanded/model.onnx | ✅ |  |
| node/test_sce_none/model.onnx | ✅ |  |
| node/test_sce_none_expanded/model.onnx | ✅ |  |
| node/test_sce_none_log_prob/model.onnx | ✅ |  |
| node/test_sce_none_log_prob_expanded/model.onnx | ✅ |  |
| node/test_sce_none_weights/model.onnx | ✅ |  |
| node/test_sce_none_weights_expanded/model.onnx | ✅ |  |
| node/test_sce_none_weights_log_prob/model.onnx | ✅ |  |
| node/test_sce_none_weights_log_prob_expanded/model.onnx | ✅ |  |
| node/test_sce_sum/model.onnx | ✅ |  |
| node/test_sce_sum_expanded/model.onnx | ✅ |  |
| node/test_sce_sum_log_prob/model.onnx | ✅ |  |
| node/test_sce_sum_log_prob_expanded/model.onnx | ✅ |  |
| node/test_selu/model.onnx | ❌ | Selu only supports alpha=1.6732632423543772 |
| node/test_selu_default/model.onnx | ✅ |  |
| node/test_selu_default_expanded_ver18/model.onnx | ✅ |  |
| node/test_selu_example/model.onnx | ❌ | Selu only supports alpha=1.6732632423543772 |
| node/test_selu_example_expanded_ver18/model.onnx | ✅ |  |
| node/test_selu_expanded_ver18/model.onnx | ✅ |  |
| node/test_sequence_insert_at_back/model.onnx | ❌ | Unsupported value type 'sequence_type' for 'sequence'. Hint: export the model with tensor inputs/outputs. |
| node/test_sequence_insert_at_front/model.onnx | ❌ | Unsupported value type 'sequence_type' for 'sequence'. Hint: export the model with tensor inputs/outputs. |
| node/test_sequence_map_add_1_sequence_1_tensor/model.onnx | ❌ | Unsupported value type 'sequence_type' for 'x0'. Hint: export the model with tensor inputs/outputs. |
| node/test_sequence_map_add_1_sequence_1_tensor_expanded/model.onnx | ❌ | Unsupported value type 'sequence_type' for 'x0'. Hint: export the model with tensor inputs/outputs. |
| node/test_sequence_map_add_2_sequences/model.onnx | ❌ | Unsupported value type 'sequence_type' for 'x0'. Hint: export the model with tensor inputs/outputs. |
| node/test_sequence_map_add_2_sequences_expanded/model.onnx | ❌ | Unsupported value type 'sequence_type' for 'x0'. Hint: export the model with tensor inputs/outputs. |
| node/test_sequence_map_extract_shapes/model.onnx | ❌ | Unsupported value type 'sequence_type' for 'in_seq'. Hint: export the model with tensor inputs/outputs. |
| node/test_sequence_map_extract_shapes_expanded/model.onnx | ❌ | Unsupported value type 'sequence_type' for 'in_seq'. Hint: export the model with tensor inputs/outputs. |
| node/test_sequence_map_identity_1_sequence/model.onnx | ❌ | Unsupported value type 'sequence_type' for 'x'. Hint: export the model with tensor inputs/outputs. |
| node/test_sequence_map_identity_1_sequence_1_tensor/model.onnx | ❌ | Unsupported value type 'sequence_type' for 'x0'. Hint: export the model with tensor inputs/outputs. |
| node/test_sequence_map_identity_1_sequence_1_tensor_expanded/model.onnx | ❌ | Unsupported value type 'sequence_type' for 'x0'. Hint: export the model with tensor inputs/outputs. |
| node/test_sequence_map_identity_1_sequence_expanded/model.onnx | ❌ | Unsupported value type 'sequence_type' for 'x'. Hint: export the model with tensor inputs/outputs. |
| node/test_sequence_map_identity_2_sequences/model.onnx | ❌ | Unsupported value type 'sequence_type' for 'x0'. Hint: export the model with tensor inputs/outputs. |
| node/test_sequence_map_identity_2_sequences_expanded/model.onnx | ❌ | Unsupported value type 'sequence_type' for 'x0'. Hint: export the model with tensor inputs/outputs. |
| node/test_shape/model.onnx | ✅ |  |
| node/test_shape_clip_end/model.onnx | ✅ |  |
| node/test_shape_clip_start/model.onnx | ✅ |  |
| node/test_shape_end_1/model.onnx | ✅ |  |
| node/test_shape_end_negative_1/model.onnx | ✅ |  |
| node/test_shape_example/model.onnx | ✅ |  |
| node/test_shape_start_1/model.onnx | ✅ |  |
| node/test_shape_start_1_end_2/model.onnx | ✅ |  |
| node/test_shape_start_1_end_negative_1/model.onnx | ✅ |  |
| node/test_shape_start_greater_than_end/model.onnx | ❌ | Dynamic or zero dims are not supported |
| node/test_shape_start_negative_1/model.onnx | ✅ |  |
| node/test_shrink_hard/model.onnx | ✅ |  |
| node/test_shrink_hard_expanded_ver18/model.onnx | ✅ |  |
| node/test_shrink_soft/model.onnx | ✅ |  |
| node/test_shrink_soft_expanded_ver18/model.onnx | ✅ |  |
| node/test_sigmoid/model.onnx | ✅ |  |
| node/test_sigmoid_example/model.onnx | ✅ |  |
| node/test_sign/model.onnx | ✅ |  |
| node/test_simple_rnn_batchwise/model.onnx | ❌ | Unsupported op RNN |
| node/test_simple_rnn_defaults/model.onnx | ❌ | Unsupported op RNN |
| node/test_simple_rnn_with_initial_bias/model.onnx | ❌ | Unsupported op RNN |
| node/test_sin/model.onnx | ✅ |  |
| node/test_sin_example/model.onnx | ✅ |  |
| node/test_sinh/model.onnx | ✅ |  |
| node/test_sinh_example/model.onnx | ✅ |  |
| node/test_size/model.onnx | ✅ |  |
| node/test_size_example/model.onnx | ✅ |  |
| node/test_slice/model.onnx | ❌ | Slice starts input must be a constant initializer |
| node/test_slice_default_axes/model.onnx | ❌ | Slice starts input must be a constant initializer |
| node/test_slice_default_steps/model.onnx | ❌ | Slice starts input must be a constant initializer |
| node/test_slice_end_out_of_bounds/model.onnx | ❌ | Slice starts input must be a constant initializer |
| node/test_slice_neg/model.onnx | ❌ | Slice starts input must be a constant initializer |
| node/test_slice_neg_steps/model.onnx | ❌ | Slice starts input must be a constant initializer |
| node/test_slice_negative_axes/model.onnx | ❌ | Slice starts input must be a constant initializer |
| node/test_slice_start_out_of_bounds/model.onnx | ❌ | Dynamic or zero dims are not supported |
| node/test_softmax_axis_0/model.onnx | ✅ |  |
| node/test_softmax_axis_0_expanded/model.onnx | ✅ |  |
| node/test_softmax_axis_0_expanded_ver18/model.onnx | ✅ |  |
| node/test_softmax_axis_1/model.onnx | ✅ |  |
| node/test_softmax_axis_1_expanded/model.onnx | ✅ |  |
| node/test_softmax_axis_1_expanded_ver18/model.onnx | ✅ |  |
| node/test_softmax_axis_2/model.onnx | ✅ |  |
| node/test_softmax_axis_2_expanded/model.onnx | ✅ |  |
| node/test_softmax_axis_2_expanded_ver18/model.onnx | ✅ |  |
| node/test_softmax_default_axis/model.onnx | ✅ |  |
| node/test_softmax_default_axis_expanded/model.onnx | ✅ |  |
| node/test_softmax_default_axis_expanded_ver18/model.onnx | ✅ |  |
| node/test_softmax_example/model.onnx | ✅ |  |
| node/test_softmax_example_expanded/model.onnx | ✅ |  |
| node/test_softmax_example_expanded_ver18/model.onnx | ✅ |  |
| node/test_softmax_large_number/model.onnx | ✅ |  |
| node/test_softmax_large_number_expanded/model.onnx | ✅ |  |
| node/test_softmax_large_number_expanded_ver18/model.onnx | ✅ |  |
| node/test_softmax_negative_axis/model.onnx | ✅ |  |
| node/test_softmax_negative_axis_expanded/model.onnx | ✅ |  |
| node/test_softmax_negative_axis_expanded_ver18/model.onnx | ✅ |  |
| node/test_softplus/model.onnx | ✅ |  |
| node/test_softplus_example/model.onnx | ✅ |  |
| node/test_softplus_example_expanded_ver18/model.onnx | ✅ |  |
| node/test_softplus_expanded_ver18/model.onnx | ✅ |  |
| node/test_softsign/model.onnx | ✅ |  |
| node/test_softsign_example/model.onnx | ✅ |  |
| node/test_softsign_example_expanded_ver18/model.onnx | ✅ |  |
| node/test_softsign_expanded_ver18/model.onnx | ✅ |  |
| node/test_spacetodepth/model.onnx | ❌ | Unsupported op SpaceToDepth |
| node/test_spacetodepth_example/model.onnx | ❌ | Unsupported op SpaceToDepth |
| node/test_split_1d_uneven_split_opset18/model.onnx | ❌ | Unsupported op Split |
| node/test_split_2d_uneven_split_opset18/model.onnx | ❌ | Unsupported op Split |
| node/test_split_equal_parts_1d_opset13/model.onnx | ❌ | Unsupported op Split |
| node/test_split_equal_parts_1d_opset18/model.onnx | ❌ | Unsupported op Split |
| node/test_split_equal_parts_2d/model.onnx | ❌ | Unsupported op Split |
| node/test_split_equal_parts_2d_opset13/model.onnx | ❌ | Unsupported op Split |
| node/test_split_equal_parts_default_axis_opset13/model.onnx | ❌ | Unsupported op Split |
| node/test_split_equal_parts_default_axis_opset18/model.onnx | ❌ | Unsupported op Split |
| node/test_split_to_sequence_1/model.onnx | ❌ | Unsupported value type 'sequence_type' for 'seq'. Hint: export the model with tensor inputs/outputs. |
| node/test_split_to_sequence_2/model.onnx | ❌ | Unsupported value type 'sequence_type' for 'seq'. Hint: export the model with tensor inputs/outputs. |
| node/test_split_to_sequence_nokeepdims/model.onnx | ❌ | Unsupported value type 'sequence_type' for 'seq'. Hint: export the model with tensor inputs/outputs. |
| node/test_split_variable_parts_1d_opset13/model.onnx | ❌ | Unsupported op Split |
| node/test_split_variable_parts_1d_opset18/model.onnx | ❌ | Unsupported op Split |
| node/test_split_variable_parts_2d_opset13/model.onnx | ❌ | Unsupported op Split |
| node/test_split_variable_parts_2d_opset18/model.onnx | ❌ | Unsupported op Split |
| node/test_split_variable_parts_default_axis_opset13/model.onnx | ❌ | Unsupported op Split |
| node/test_split_variable_parts_default_axis_opset18/model.onnx | ❌ | Unsupported op Split |
| node/test_split_zero_size_splits_opset13/model.onnx | ❌ | Dynamic or zero dims are not supported |
| node/test_split_zero_size_splits_opset18/model.onnx | ❌ | Dynamic or zero dims are not supported |
| node/test_sqrt/model.onnx | ✅ |  |
| node/test_sqrt_example/model.onnx | ✅ |  |
| node/test_squeeze/model.onnx | ✅ |  |
| node/test_squeeze_negative_axes/model.onnx | ✅ |  |
| node/test_stft/model.onnx | ❌ | Unsupported op STFT |
| node/test_stft_with_window/model.onnx | ❌ | Unsupported op STFT |
| node/test_string_concat/model.onnx | ❌ | Unsupported elem_type 8 (STRING) for tensor 'x'. |
| node/test_string_concat_broadcasting/model.onnx | ❌ | Unsupported elem_type 8 (STRING) for tensor 'x'. |
| node/test_string_concat_empty_string/model.onnx | ❌ | Unsupported elem_type 8 (STRING) for tensor 'x'. |
| node/test_string_concat_utf8/model.onnx | ❌ | Unsupported elem_type 8 (STRING) for tensor 'x'. |
| node/test_string_concat_zero_dimensional/model.onnx | ❌ | Unsupported elem_type 8 (STRING) for tensor 'x'. |
| node/test_string_split_basic/model.onnx | ❌ | Unsupported elem_type 8 (STRING) for tensor 'x'. |
| node/test_string_split_consecutive_delimiters/model.onnx | ❌ | Unsupported elem_type 8 (STRING) for tensor 'x'. |
| node/test_string_split_empty_string_delimiter/model.onnx | ❌ | Unsupported elem_type 8 (STRING) for tensor 'x'. |
| node/test_string_split_empty_tensor/model.onnx | ❌ | Unsupported elem_type 8 (STRING) for tensor 'x'. |
| node/test_string_split_maxsplit/model.onnx | ❌ | Unsupported elem_type 8 (STRING) for tensor 'x'. |
| node/test_string_split_no_delimiter/model.onnx | ❌ | Unsupported elem_type 8 (STRING) for tensor 'x'. |
| node/test_strnormalizer_export_monday_casesensintive_lower/model.onnx | ❌ | Unsupported elem_type 8 (STRING) for tensor 'x'. |
| node/test_strnormalizer_export_monday_casesensintive_nochangecase/model.onnx | ❌ | Unsupported elem_type 8 (STRING) for tensor 'x'. |
| node/test_strnormalizer_export_monday_casesensintive_upper/model.onnx | ❌ | Unsupported elem_type 8 (STRING) for tensor 'x'. |
| node/test_strnormalizer_export_monday_empty_output/model.onnx | ❌ | Unsupported elem_type 8 (STRING) for tensor 'x'. |
| node/test_strnormalizer_export_monday_insensintive_upper_twodim/model.onnx | ❌ | Unsupported elem_type 8 (STRING) for tensor 'x'. |
| node/test_strnormalizer_nostopwords_nochangecase/model.onnx | ❌ | Unsupported elem_type 8 (STRING) for tensor 'x'. |
| node/test_sub/model.onnx | ✅ |  |
| node/test_sub_bcast/model.onnx | ✅ |  |
| node/test_sub_example/model.onnx | ✅ |  |
| node/test_sub_int16/model.onnx | ✅ |  |
| node/test_sub_int8/model.onnx | ✅ |  |
| node/test_sub_uint16/model.onnx | ✅ |  |
| node/test_sub_uint32/model.onnx | ✅ |  |
| node/test_sub_uint64/model.onnx | ✅ |  |
| node/test_sub_uint8/model.onnx | ✅ |  |
| node/test_sum_example/model.onnx | ❌ | Sum must have 2 inputs and 1 output |
| node/test_sum_one_input/model.onnx | ❌ | Sum must have 2 inputs and 1 output |
| node/test_sum_two_inputs/model.onnx | ✅ |  |
| node/test_swish/model.onnx | ✅ |  |
| node/test_swish_expanded/model.onnx | ✅ |  |
| node/test_tan/model.onnx | ✅ |  |
| node/test_tan_example/model.onnx | ✅ |  |
| node/test_tanh/model.onnx | ✅ |  |
| node/test_tanh_example/model.onnx | ✅ |  |
| node/test_tensorscatter/model.onnx | ❌ | Unsupported op TensorScatter |
| node/test_tensorscatter_3d/model.onnx | ❌ | Unsupported op TensorScatter |
| node/test_tensorscatter_circular/model.onnx | ❌ | Unsupported op TensorScatter |
| node/test_tfidfvectorizer_tf_batch_onlybigrams_skip0/model.onnx | ❌ | Unsupported op TfIdfVectorizer |
| node/test_tfidfvectorizer_tf_batch_onlybigrams_skip5/model.onnx | ❌ | Unsupported op TfIdfVectorizer |
| node/test_tfidfvectorizer_tf_batch_uniandbigrams_skip5/model.onnx | ❌ | Unsupported op TfIdfVectorizer |
| node/test_tfidfvectorizer_tf_only_bigrams_skip0/model.onnx | ❌ | Unsupported op TfIdfVectorizer |
| node/test_tfidfvectorizer_tf_onlybigrams_levelempty/model.onnx | ❌ | Unsupported op TfIdfVectorizer |
| node/test_tfidfvectorizer_tf_onlybigrams_skip5/model.onnx | ❌ | Unsupported op TfIdfVectorizer |
| node/test_tfidfvectorizer_tf_uniandbigrams_skip5/model.onnx | ❌ | Unsupported op TfIdfVectorizer |
| node/test_thresholdedrelu/model.onnx | ❌ | ThresholdedRelu only supports alpha=1.0 |
| node/test_thresholdedrelu_default/model.onnx | ✅ |  |
| node/test_thresholdedrelu_default_expanded_ver18/model.onnx | ✅ |  |
| node/test_thresholdedrelu_example/model.onnx | ❌ | ThresholdedRelu only supports alpha=1.0 |
| node/test_thresholdedrelu_example_expanded_ver18/model.onnx | ✅ |  |
| node/test_thresholdedrelu_expanded_ver18/model.onnx | ✅ |  |
| node/test_tile/model.onnx | ❌ | Unsupported op Tile |
| node/test_tile_precomputed/model.onnx | ❌ | Unsupported op Tile |
| node/test_top_k/model.onnx | ❌ | Unsupported op TopK |
| node/test_top_k_negative_axis/model.onnx | ❌ | Unsupported op TopK |
| node/test_top_k_same_values/model.onnx | ❌ | Unsupported op TopK |
| node/test_top_k_same_values_2d/model.onnx | ❌ | Unsupported op TopK |
| node/test_top_k_same_values_largest/model.onnx | ❌ | Unsupported op TopK |
| node/test_top_k_smallest/model.onnx | ❌ | Unsupported op TopK |
| node/test_top_k_uint64/model.onnx | ❌ | Unsupported op TopK |
| node/test_training_dropout/model.onnx | ❌ | Dropout supports only the data input and 1 or 2 outputs |
| node/test_training_dropout_default/model.onnx | ❌ | Dropout supports only the data input and 1 or 2 outputs |
| node/test_training_dropout_default_mask/model.onnx | ❌ | Dropout supports only the data input and 1 or 2 outputs |
| node/test_training_dropout_mask/model.onnx | ❌ | Dropout supports only the data input and 1 or 2 outputs |
| node/test_training_dropout_zero_ratio/model.onnx | ❌ | Dropout supports only the data input and 1 or 2 outputs |
| node/test_training_dropout_zero_ratio_mask/model.onnx | ❌ | Dropout supports only the data input and 1 or 2 outputs |
| node/test_transpose_all_permutations_0/model.onnx | ✅ |  |
| node/test_transpose_all_permutations_1/model.onnx | ✅ |  |
| node/test_transpose_all_permutations_2/model.onnx | ✅ |  |
| node/test_transpose_all_permutations_3/model.onnx | ✅ |  |
| node/test_transpose_all_permutations_4/model.onnx | ✅ |  |
| node/test_transpose_all_permutations_5/model.onnx | ✅ |  |
| node/test_transpose_default/model.onnx | ✅ |  |
| node/test_tril/model.onnx | ❌ | Unsupported op Trilu |
| node/test_tril_neg/model.onnx | ❌ | Unsupported op Trilu |
| node/test_tril_one_row_neg/model.onnx | ❌ | Unsupported op Trilu |
| node/test_tril_out_neg/model.onnx | ❌ | Unsupported op Trilu |
| node/test_tril_out_pos/model.onnx | ❌ | Unsupported op Trilu |
| node/test_tril_pos/model.onnx | ❌ | Unsupported op Trilu |
| node/test_tril_square/model.onnx | ❌ | Unsupported op Trilu |
| node/test_tril_square_neg/model.onnx | ❌ | Unsupported op Trilu |
| node/test_tril_zero/model.onnx | ❌ | Dynamic or zero dims are not supported |
| node/test_triu/model.onnx | ❌ | Unsupported op Trilu |
| node/test_triu_neg/model.onnx | ❌ | Unsupported op Trilu |
| node/test_triu_one_row/model.onnx | ❌ | Unsupported op Trilu |
| node/test_triu_out_neg_out/model.onnx | ❌ | Unsupported op Trilu |
| node/test_triu_out_pos/model.onnx | ❌ | Unsupported op Trilu |
| node/test_triu_pos/model.onnx | ❌ | Unsupported op Trilu |
| node/test_triu_square/model.onnx | ❌ | Unsupported op Trilu |
| node/test_triu_square_neg/model.onnx | ❌ | Unsupported op Trilu |
| node/test_triu_zero/model.onnx | ❌ | Dynamic or zero dims are not supported |
| node/test_unique_length_1/model.onnx | ❌ | Unsupported op Unique |
| node/test_unique_not_sorted_without_axis/model.onnx | ❌ | Unsupported op Unique |
| node/test_unique_sorted_with_axis/model.onnx | ❌ | Unsupported op Unique |
| node/test_unique_sorted_with_axis_3d/model.onnx | ❌ | Unsupported op Unique |
| node/test_unique_sorted_with_negative_axis/model.onnx | ❌ | Unsupported op Unique |
| node/test_unique_sorted_without_axis/model.onnx | ❌ | Unsupported op Unique |
| node/test_unsqueeze_axis_0/model.onnx | ✅ |  |
| node/test_unsqueeze_axis_1/model.onnx | ✅ |  |
| node/test_unsqueeze_axis_2/model.onnx | ✅ |  |
| node/test_unsqueeze_negative_axes/model.onnx | ✅ |  |
| node/test_unsqueeze_three_axes/model.onnx | ✅ |  |
| node/test_unsqueeze_two_axes/model.onnx | ✅ |  |
| node/test_unsqueeze_unsorted_axes/model.onnx | ✅ |  |
| node/test_upsample_nearest/model.onnx | ❌ | Unsupported op Upsample |
| node/test_where_example/model.onnx | ✅ |  |
| node/test_where_long_example/model.onnx | ✅ |  |
| node/test_wrap_pad/model.onnx | ❌ | Unsupported op Pad |
| node/test_xor2d/model.onnx | ✅ |  |
| node/test_xor3d/model.onnx | ✅ |  |
| node/test_xor4d/model.onnx | ✅ |  |
| node/test_xor_bcast3v1d/model.onnx | ✅ |  |
| node/test_xor_bcast3v2d/model.onnx | ✅ |  |
| node/test_xor_bcast4v2d/model.onnx | ✅ |  |
| node/test_xor_bcast4v3d/model.onnx | ✅ |  |
| node/test_xor_bcast4v4d/model.onnx | ✅ |  |
| pytorch-converted/test_AvgPool1d/model.onnx | ✅ |  |
| pytorch-converted/test_AvgPool1d_stride/model.onnx | ✅ |  |
| pytorch-converted/test_AvgPool2d/model.onnx | ✅ |  |
| pytorch-converted/test_AvgPool2d_stride/model.onnx | ✅ |  |
| pytorch-converted/test_AvgPool3d/model.onnx | ❌ | AveragePool expects 2D kernel_shape |
| pytorch-converted/test_AvgPool3d_stride/model.onnx | ❌ | AveragePool expects 2D kernel_shape |
| pytorch-converted/test_AvgPool3d_stride1_pad0_gpu_input/model.onnx | ❌ | AveragePool expects 2D kernel_shape |
| pytorch-converted/test_BatchNorm1d_3d_input_eval/model.onnx | ✅ |  |
| pytorch-converted/test_BatchNorm2d_eval/model.onnx | ✅ |  |
| pytorch-converted/test_BatchNorm2d_momentum_eval/model.onnx | ✅ |  |
| pytorch-converted/test_BatchNorm3d_eval/model.onnx | ✅ |  |
| pytorch-converted/test_BatchNorm3d_momentum_eval/model.onnx | ✅ |  |
| pytorch-converted/test_ConstantPad2d/model.onnx | ❌ | Unsupported op Pad |
| pytorch-converted/test_Conv1d/model.onnx | ✅ |  |
| pytorch-converted/test_Conv1d_dilated/model.onnx | ✅ |  |
| pytorch-converted/test_Conv1d_groups/model.onnx | ✅ |  |
| pytorch-converted/test_Conv1d_pad1/model.onnx | ✅ |  |
| pytorch-converted/test_Conv1d_pad1size1/model.onnx | ✅ |  |
| pytorch-converted/test_Conv1d_pad2/model.onnx | ✅ |  |
| pytorch-converted/test_Conv1d_pad2size1/model.onnx | ✅ |  |
| pytorch-converted/test_Conv1d_stride/model.onnx | ✅ |  |
| pytorch-converted/test_Conv2d/model.onnx | ✅ |  |
| pytorch-converted/test_Conv2d_depthwise/model.onnx | ✅ |  |
| pytorch-converted/test_Conv2d_depthwise_padded/model.onnx | ✅ |  |
| pytorch-converted/test_Conv2d_depthwise_strided/model.onnx | ✅ |  |
| pytorch-converted/test_Conv2d_depthwise_with_multiplier/model.onnx | ✅ |  |
| pytorch-converted/test_Conv2d_dilated/model.onnx | ✅ |  |
| pytorch-converted/test_Conv2d_groups/model.onnx | ✅ |  |
| pytorch-converted/test_Conv2d_groups_thnn/model.onnx | ✅ |  |
| pytorch-converted/test_Conv2d_no_bias/model.onnx | ✅ |  |
| pytorch-converted/test_Conv2d_padding/model.onnx | ✅ |  |
| pytorch-converted/test_Conv2d_strided/model.onnx | ✅ |  |
| pytorch-converted/test_Conv3d/model.onnx | ✅ |  |
| pytorch-converted/test_Conv3d_dilated/model.onnx | ✅ |  |
| pytorch-converted/test_Conv3d_dilated_strided/model.onnx | ✅ |  |
| pytorch-converted/test_Conv3d_groups/model.onnx | ✅ |  |
| pytorch-converted/test_Conv3d_no_bias/model.onnx | ✅ |  |
| pytorch-converted/test_Conv3d_stride/model.onnx | ✅ |  |
| pytorch-converted/test_Conv3d_stride_padding/model.onnx | ✅ |  |
| pytorch-converted/test_ConvTranspose2d/model.onnx | ❌ | Unsupported op ConvTranspose |
| pytorch-converted/test_ConvTranspose2d_no_bias/model.onnx | ❌ | Unsupported op ConvTranspose |
| pytorch-converted/test_ELU/model.onnx | ❌ | Elu only supports alpha=1.0 |
| pytorch-converted/test_Embedding/model.onnx | ✅ |  |
| pytorch-converted/test_Embedding_sparse/model.onnx | ✅ |  |
| pytorch-converted/test_GLU/model.onnx | ❌ | Unsupported op Split |
| pytorch-converted/test_GLU_dim/model.onnx | ❌ | Unsupported op Split |
| pytorch-converted/test_LeakyReLU/model.onnx | ✅ |  |
| pytorch-converted/test_LeakyReLU_with_negval/model.onnx | ❌ | LeakyRelu only supports alpha=0.01 |
| pytorch-converted/test_Linear/model.onnx | ✅ |  |
| pytorch-converted/test_Linear_no_bias/model.onnx | ✅ |  |
| pytorch-converted/test_LogSoftmax/model.onnx | ✅ |  |
| pytorch-converted/test_MaxPool1d/model.onnx | ✅ |  |
| pytorch-converted/test_MaxPool1d_stride/model.onnx | ✅ |  |
| pytorch-converted/test_MaxPool1d_stride_padding_dilation/model.onnx | ✅ |  |
| pytorch-converted/test_MaxPool2d/model.onnx | ✅ |  |
| pytorch-converted/test_MaxPool2d_stride_padding_dilation/model.onnx | ✅ |  |
| pytorch-converted/test_MaxPool3d/model.onnx | ✅ |  |
| pytorch-converted/test_MaxPool3d_stride/model.onnx | ✅ |  |
| pytorch-converted/test_MaxPool3d_stride_padding/model.onnx | ✅ |  |
| pytorch-converted/test_PReLU_1d/model.onnx | ✅ |  |
| pytorch-converted/test_PReLU_1d_multiparam/model.onnx | ✅ |  |
| pytorch-converted/test_PReLU_2d/model.onnx | ✅ |  |
| pytorch-converted/test_PReLU_2d_multiparam/model.onnx | ✅ |  |
| pytorch-converted/test_PReLU_3d/model.onnx | ✅ |  |
| pytorch-converted/test_PReLU_3d_multiparam/model.onnx | ✅ |  |
| pytorch-converted/test_PixelShuffle/model.onnx | ✅ |  |
| pytorch-converted/test_PoissonNLLLLoss_no_reduce/model.onnx | ✅ |  |
| pytorch-converted/test_ReLU/model.onnx | ✅ |  |
| pytorch-converted/test_ReflectionPad2d/model.onnx | ❌ | Unsupported op Pad |
| pytorch-converted/test_ReplicationPad2d/model.onnx | ❌ | Unsupported op Pad |
| pytorch-converted/test_SELU/model.onnx | ✅ |  |
| pytorch-converted/test_Sigmoid/model.onnx | ✅ |  |
| pytorch-converted/test_Softmax/model.onnx | ✅ |  |
| pytorch-converted/test_Softmin/model.onnx | ✅ |  |
| pytorch-converted/test_Softplus/model.onnx | ✅ |  |
| pytorch-converted/test_Softsign/model.onnx | ✅ |  |
| pytorch-converted/test_Tanh/model.onnx | ✅ |  |
| pytorch-converted/test_ZeroPad2d/model.onnx | ❌ | Unsupported op Pad |
| pytorch-converted/test_log_softmax_dim3/model.onnx | ✅ |  |
| pytorch-converted/test_log_softmax_lastdim/model.onnx | ✅ |  |
| pytorch-converted/test_softmax_functional_dim3/model.onnx | ✅ |  |
| pytorch-converted/test_softmax_lastdim/model.onnx | ✅ |  |
| pytorch-operator/test_operator_add_broadcast/model.onnx | ✅ |  |
| pytorch-operator/test_operator_add_size1_broadcast/model.onnx | ✅ |  |
| pytorch-operator/test_operator_add_size1_right_broadcast/model.onnx | ✅ |  |
| pytorch-operator/test_operator_add_size1_singleton_broadcast/model.onnx | ✅ |  |
| pytorch-operator/test_operator_addconstant/model.onnx | ✅ |  |
| pytorch-operator/test_operator_addmm/model.onnx | ✅ |  |
| pytorch-operator/test_operator_basic/model.onnx | ✅ |  |
| pytorch-operator/test_operator_chunk/model.onnx | ❌ | Unsupported op Split |
| pytorch-operator/test_operator_clip/model.onnx | ✅ |  |
| pytorch-operator/test_operator_concat2/model.onnx | ✅ |  |
| pytorch-operator/test_operator_conv/model.onnx | ✅ |  |
| pytorch-operator/test_operator_convtranspose/model.onnx | ❌ | Unsupported op ConvTranspose |
| pytorch-operator/test_operator_exp/model.onnx | ✅ |  |
| pytorch-operator/test_operator_flatten/model.onnx | ✅ |  |
| pytorch-operator/test_operator_index/model.onnx | ✅ |  |
| pytorch-operator/test_operator_max/model.onnx | ✅ |  |
| pytorch-operator/test_operator_maxpool/model.onnx | ✅ |  |
| pytorch-operator/test_operator_min/model.onnx | ✅ |  |
| pytorch-operator/test_operator_mm/model.onnx | ✅ |  |
| pytorch-operator/test_operator_non_float_params/model.onnx | ✅ |  |
| pytorch-operator/test_operator_pad/model.onnx | ❌ | Unsupported op Pad |
| pytorch-operator/test_operator_params/model.onnx | ✅ |  |
| pytorch-operator/test_operator_permute2/model.onnx | ✅ |  |
| pytorch-operator/test_operator_pow/model.onnx | ✅ |  |
| pytorch-operator/test_operator_reduced_mean/model.onnx | ✅ |  |
| pytorch-operator/test_operator_reduced_mean_keepdim/model.onnx | ✅ |  |
| pytorch-operator/test_operator_reduced_sum/model.onnx | ✅ |  |
| pytorch-operator/test_operator_reduced_sum_keepdim/model.onnx | ✅ |  |
| pytorch-operator/test_operator_repeat/model.onnx | ❌ | Unsupported op Tile |
| pytorch-operator/test_operator_repeat_dim_overflow/model.onnx | ❌ | Unsupported op Tile |
| pytorch-operator/test_operator_selu/model.onnx | ✅ |  |
| pytorch-operator/test_operator_sqrt/model.onnx | ✅ |  |
| pytorch-operator/test_operator_symbolic_override/model.onnx | ❌ | Unsupported op InstanceNormalization |
| pytorch-operator/test_operator_symbolic_override_nested/model.onnx | ❌ | Sum must have 2 inputs and 1 output |
| pytorch-operator/test_operator_view/model.onnx | ✅ |  |
| simple/test_expand_shape_model1/model.onnx | ❌ | Unsupported op Expand |
| simple/test_expand_shape_model2/model.onnx | ❌ | Unsupported op Expand |
| simple/test_expand_shape_model3/model.onnx | ❌ | Unsupported op Expand |
| simple/test_expand_shape_model4/model.onnx | ❌ | Unsupported op Expand |
| simple/test_gradient_of_add/model.onnx | ❌ | Unsupported op Gradient |
| simple/test_gradient_of_add_and_mul/model.onnx | ❌ | Unsupported op Gradient |
| simple/test_sequence_model1/model.onnx | ❌ | Dynamic dim for tensor 'out' |
| simple/test_sequence_model2/model.onnx | ❌ | Unsupported value type 'sequence_type' for 'seq_1'. Hint: export the model with tensor inputs/outputs. |
| simple/test_sequence_model3/model.onnx | ❌ | Unsupported value type 'sequence_type' for 'seq_1'. Hint: export the model with tensor inputs/outputs. |
| simple/test_sequence_model4/model.onnx | ❌ | Dynamic dim for tensor 'out' |
| simple/test_sequence_model5/model.onnx | ❌ | Unsupported value type 'sequence_type' for 'seq_1'. Hint: export the model with tensor inputs/outputs. |
| simple/test_sequence_model6/model.onnx | ❌ | Unsupported value type 'sequence_type' for 'seq_1'. Hint: export the model with tensor inputs/outputs. |
| simple/test_sequence_model7/model.onnx | ❌ | Unsupported value type 'sequence_type' for 'seq_1'. Hint: export the model with tensor inputs/outputs. |
| simple/test_sequence_model8/model.onnx | ❌ | Dynamic dim for tensor 'X' |
| simple/test_shrink/model.onnx | ✅ |  |
| simple/test_sign_model/model.onnx | ✅ |  |
| simple/test_single_relu_model/model.onnx | ✅ |  |
| simple/test_strnorm_model_monday_casesensintive_lower/model.onnx | ❌ | Unsupported elem_type 8 (STRING) for tensor 'x'. |
| simple/test_strnorm_model_monday_casesensintive_nochangecase/model.onnx | ❌ | Unsupported elem_type 8 (STRING) for tensor 'x'. |
| simple/test_strnorm_model_monday_casesensintive_upper/model.onnx | ❌ | Unsupported elem_type 8 (STRING) for tensor 'x'. |
| simple/test_strnorm_model_monday_empty_output/model.onnx | ❌ | Unsupported elem_type 8 (STRING) for tensor 'x'. |
| simple/test_strnorm_model_monday_insensintive_upper_twodim/model.onnx | ❌ | Unsupported elem_type 8 (STRING) for tensor 'x'. |
| simple/test_strnorm_model_nostopwords_nochangecase/model.onnx | ❌ | Unsupported elem_type 8 (STRING) for tensor 'x'. |

## Local ONNX file support

Local tests: `onnx2c-org/test/local_ops`.

Support 56 / 74 local ONNX files.

| File | Supported | Error |
| --- | --- | --- |
| test_gather_basic/model.onnx | ✅ |  |
| test_gather_output_scalar/model.onnx | ✅ |  |
| test_gather_scalar_axis0/model.onnx | ✅ |  |
| test_gather_scalar_axis1/model.onnx | ✅ |  |
| test_gemm_C1/model.onnx | ✅ |  |
| test_gemm_C1_transA/model.onnx | ✅ |  |
| test_gemm_C1_transB/model.onnx | ✅ |  |
| test_gemm_C1x1/model.onnx | ✅ |  |
| test_gemm_C1x1_transA/model.onnx | ✅ |  |
| test_gemm_C1xN/model.onnx | ✅ |  |
| test_gemm_C1xN_transA/model.onnx | ✅ |  |
| test_gemm_C1xN_transA_transB/model.onnx | ✅ |  |
| test_gemm_CM_transA/model.onnx | ❌ | Gemm bias input must be broadcastable to output shape, got (2,) vs (2, 4) |
| test_gemm_CMx1/model.onnx | ✅ |  |
| test_gemm_CMx1_transA/model.onnx | ✅ |  |
| test_gemm_CMx1_transA_transB/model.onnx | ✅ |  |
| test_gemm_CMxN/model.onnx | ✅ |  |
| test_gemm_CMxN_transA/model.onnx | ✅ |  |
| test_gemm_CMxN_transA_transB/model.onnx | ✅ |  |
| test_gemm_CMxN_transB/model.onnx | ✅ |  |
| test_gemm_CN/model.onnx | ✅ |  |
| test_gemm_CN_transA/model.onnx | ✅ |  |
| test_gemm_CN_transA_transB/model.onnx | ✅ |  |
| test_gemm_CN_transB/model.onnx | ✅ |  |
| test_lstm_activations/model.onnx | ✅ |  |
| test_lstm_all_outputs/model.onnx | ✅ |  |
| test_lstm_bidirectional/model.onnx | ❌ | Unsupported LSTM direction b'bidirectional' |
| test_lstm_clip/model.onnx | ✅ |  |
| test_lstm_intermediate_h/model.onnx | ✅ |  |
| test_lstm_missing_inputs/model.onnx | ✅ |  |
| test_lstm_reverse/model.onnx | ❌ | Unsupported LSTM direction b'reverse' |
| test_lstm_seq_length/model.onnx | ✅ |  |
| test_lstm_simple/model.onnx | ✅ |  |
| test_lstm_with_initial_state/model.onnx | ✅ |  |
| test_lstm_y_c/model.onnx | ✅ |  |
| test_matmul_1x1x3x4_2x3x4x5/model.onnx | ✅ |  |
| test_matmul_1x3x4_2x3x4x5/model.onnx | ✅ |  |
| test_matmul_1x3x4_3x4x5/model.onnx | ✅ |  |
| test_matmul_2x1x3x4_2x3x4x5/model.onnx | ✅ |  |
| test_matmul_2x3_3x4/model.onnx | ✅ |  |
| test_matmul_2x3x3x4_1x4x5/model.onnx | ✅ |  |
| test_matmul_2x3x4_4/model.onnx | ✅ |  |
| test_matmul_2x3x4_4x5/model.onnx | ✅ |  |
| test_matmul_2x3x4x5_5/model.onnx | ✅ |  |
| test_matmul_3_2x3x4/model.onnx | ✅ |  |
| test_matmul_3_3/model.onnx | ✅ |  |
| test_matmul_3_3x4/model.onnx | ✅ |  |
| test_matmul_3x4_2x4x5/model.onnx | ✅ |  |
| test_matmul_3x4_4/model.onnx | ✅ |  |
| test_matmul_4x5x2x3_4x5x3x4/model.onnx | ✅ |  |
| test_matmul_5x2x3_5x3x4/model.onnx | ✅ |  |
| test_matmul_precision/model.onnx | ✅ |  |
| test_maxpool_stride_1/model.onnx | ✅ |  |
| test_maxpool_stride_2/model.onnx | ✅ |  |
| test_nodes_out_of_order/model.onnx | ✅ |  |
| test_pad_constant_default/model.onnx | ❌ | Unsupported op Pad |
| test_pad_constant_input/model.onnx | ❌ | Unsupported op Pad |
| test_pad_edge/model.onnx | ❌ | Unsupported op Pad |
| test_pad_edge_allaxes/model.onnx | ❌ | Unsupported op Pad |
| test_pad_reflect_allaxes/model.onnx | ❌ | Unsupported op Pad |
| test_pad_reflect_nopadding/model.onnx | ❌ | Unsupported op Pad |
| test_qlinearadd_int8/model.onnx | ❌ | Unsupported op QLinearAdd |
| test_qlinearadd_uint8/model.onnx | ❌ | Unsupported op QLinearAdd |
| test_qlinearmul_int8/model.onnx | ❌ | Unsupported op QLinearMul |
| test_qlinearmul_uint8/model.onnx | ❌ | Unsupported op QLinearMul |
| test_resize_downsample_sizes_linear_1D/model.onnx | ✅ |  |
| test_resize_downsample_sizes_linear_1D_align/model.onnx | ✅ |  |
| test_scalar_input_to_node/model.onnx | ✅ |  |
| test_scatternd_indices_1x1x2/model.onnx | ❌ | Unsupported op ScatterND |
| test_scatternd_indices_1x2x2/model.onnx | ❌ | Unsupported op ScatterND |
| test_scatternd_indices_2x2x2/model.onnx | ❌ | Unsupported op ScatterND |
| test_scatternd_indices_3x2/model.onnx | ❌ | Unsupported op ScatterND |
| test_shape_const_out/model.onnx | ❌ | Unsupported op Expand |
| test_slice_end_INT64_MAX/model.onnx | ✅ |  |
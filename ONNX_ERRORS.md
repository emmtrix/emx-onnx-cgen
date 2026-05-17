<!-- AUTO-GENERATED FILE. DO NOT EDIT. -->
<!-- Regenerate with: UPDATE_REFS=1 pytest -q tests/test_official_onnx_files_docs.py::test_official_onnx_file_support_doc -->

# ONNX verification errors

Aggregates non-success verification outcomes.

| Error message | Count | Opset versions |
| --- | --- | --- |
| Out of tolerance | 40 | 7, 17, 22 |
| Unsupported op com.microsoft.CausalConvWithState | 29 |  |
| Unsupported op com.microsoft.LinearAttention | 26 |  |
| Unsupported value type '*' for '*'. Hint: export the model with tensor inputs/outputs. | 8 |  |
| Unsupported op com.microsoft.GroupQueryAttention | 6 |  |

## Error frequency by opset

| Error message | Opset | Count |
| --- | --- | --- |
| Out of tolerance | 7 | 4 |
| Out of tolerance | 17 | 13 |
| Out of tolerance | 22 | 1 |

## Failing ONNX files

Lists every ONNX file with a non-success verification outcome.

| File | Opset | Verification | Supported | Error |
| --- | --- | --- | --- | --- |
| node/test_adam_multiple/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 62311) |
| node/test_l2normalization_axis_0/model.onnx | 22 | Data/Data | ❌ | Out of tolerance (max ULP 4294967295) |
| test/contrib_ops/attention_op_test/AttentionPastState_dynamic_run0/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 86151) |
| test/contrib_ops/attention_op_test/Attention_Mask1D_Fp32_B2_S64_run0/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 1020241) |
| test/contrib_ops/attention_op_test/Attention_Mask2D_Fp32_B2_S32_run0/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 980755) |
| test/contrib_ops/bifurcation_detector_op_test/SuffixMatchAtEndOfSrc_run0/model.onnx |  | Data/Data | ❌ | Out of tolerance (max abs diff 4) |
| test/contrib_ops/causal_conv_with_state_op_test/BasicNoStateNoBias_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.CausalConvWithState |
| test/contrib_ops/causal_conv_with_state_op_test/BasicNoStateNoBias_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.CausalConvWithState |
| test/contrib_ops/causal_conv_with_state_op_test/BasicWithBias_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.CausalConvWithState |
| test/contrib_ops/causal_conv_with_state_op_test/BasicWithBias_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.CausalConvWithState |
| test/contrib_ops/causal_conv_with_state_op_test/BasicWithState_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.CausalConvWithState |
| test/contrib_ops/causal_conv_with_state_op_test/BasicWithState_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.CausalConvWithState |
| test/contrib_ops/causal_conv_with_state_op_test/KernelSize2_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.CausalConvWithState |
| test/contrib_ops/causal_conv_with_state_op_test/KernelSize2_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.CausalConvWithState |
| test/contrib_ops/causal_conv_with_state_op_test/KernelSize4_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.CausalConvWithState |
| test/contrib_ops/causal_conv_with_state_op_test/KernelSize4_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.CausalConvWithState |
| test/contrib_ops/causal_conv_with_state_op_test/LargerDimensions_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.CausalConvWithState |
| test/contrib_ops/causal_conv_with_state_op_test/LargerDimensions_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.CausalConvWithState |
| test/contrib_ops/causal_conv_with_state_op_test/MultiBatch_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.CausalConvWithState |
| test/contrib_ops/causal_conv_with_state_op_test/MultiBatch_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.CausalConvWithState |
| test/contrib_ops/causal_conv_with_state_op_test/SiluActivationNoState_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.CausalConvWithState |
| test/contrib_ops/causal_conv_with_state_op_test/SiluActivationNoState_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.CausalConvWithState |
| test/contrib_ops/causal_conv_with_state_op_test/SiluActivationWithBiasAndState_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.CausalConvWithState |
| test/contrib_ops/causal_conv_with_state_op_test/SiluActivationWithBiasAndState_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.CausalConvWithState |
| test/contrib_ops/causal_conv_with_state_op_test/SiluActivationWithState_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.CausalConvWithState |
| test/contrib_ops/causal_conv_with_state_op_test/SiluActivationWithState_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.CausalConvWithState |
| test/contrib_ops/causal_conv_with_state_op_test/SingleTokenDecodeMultiBatch_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.CausalConvWithState |
| test/contrib_ops/causal_conv_with_state_op_test/SingleTokenDecodeMultiBatch_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.CausalConvWithState |
| test/contrib_ops/causal_conv_with_state_op_test/SingleTokenDecode_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.CausalConvWithState |
| test/contrib_ops/causal_conv_with_state_op_test/SingleTokenDecode_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.CausalConvWithState |
| test/contrib_ops/causal_conv_with_state_op_test/StateContinuity_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.CausalConvWithState |
| test/contrib_ops/causal_conv_with_state_op_test/StateContinuity_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.CausalConvWithState |
| test/contrib_ops/causal_conv_with_state_op_test/StateContinuity_run2/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.CausalConvWithState |
| test/contrib_ops/causal_conv_with_state_op_test/WithStateAndBias_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.CausalConvWithState |
| test/contrib_ops/causal_conv_with_state_op_test/WithStateAndBias_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.CausalConvWithState |
| test/contrib_ops/group_query_attention_op_test/BoundaryValidSeqlensKWithLargerPast_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.GroupQueryAttention |
| test/contrib_ops/group_query_attention_op_test/BoundaryValidSeqlensK_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.GroupQueryAttention |
| test/contrib_ops/group_query_attention_op_test/MaxBoundarySeqlensK_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.GroupQueryAttention |
| test/contrib_ops/group_query_attention_op_test/SeqlensKLegacy2DShapeMultiBatch_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.GroupQueryAttention |
| test/contrib_ops/group_query_attention_op_test/SeqlensKLegacy2DShapeTrailingBatch_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.GroupQueryAttention |
| test/contrib_ops/group_query_attention_op_test/SeqlensKLegacy2DShape_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.GroupQueryAttention |
| test/contrib_ops/layer_norm_op_test/BERTLayerNorm_NoBias_run0/model.onnx | 17 | Data/Data | ❌ | Out of tolerance (max ULP 2272) |
| test/contrib_ops/layer_norm_op_test/BERTLayerNorm_run0/model.onnx | 17 | Data/Data | ❌ | Out of tolerance (max ULP 5632) |
| test/contrib_ops/layer_norm_op_test/LayerNorm17_double_run0/model.onnx | 17 | Data/Data | ❌ | Out of tolerance (max ULP 160715044774) |
| test/contrib_ops/layer_norm_op_test/LayerNorm17_opset_run0/model.onnx | 17 | Data/Data | ❌ | Out of tolerance (max ULP 299) |
| test/contrib_ops/layer_norm_op_test/LayerNorm17_opset_run1/model.onnx | 17 | Data/Data | ❌ | Out of tolerance (max ULP 299) |
| test/contrib_ops/layer_norm_op_test/LayerNorm_Scale_Bias_4D_OuterInnerBroadcast_Axis3_run0/model.onnx | 17 | Data/Data | ❌ | Out of tolerance (max ULP 1613) |
| test/contrib_ops/layer_norm_op_test/LayerNorm_Scale_Bias_4D_OuterInnerBroadcast_run0/model.onnx | 17 | Data/Data | ❌ | Out of tolerance (max ULP 1613) |
| test/contrib_ops/layer_norm_op_test/LayerNorm_Scale_Bias_Axis2_run0/model.onnx | 17 | Data/Data | ❌ | Out of tolerance (max ULP 336) |
| test/contrib_ops/layer_norm_op_test/LayerNorm_Scale_Bias_Broadcast_Dim0_run0/model.onnx | 7 | Data/Data | ❌ | Out of tolerance (max ULP 286) |
| test/contrib_ops/layer_norm_op_test/LayerNorm_Scale_Bias_Broadcast_Dim1_run0/model.onnx | 7 | Data/Data | ❌ | Out of tolerance (max ULP 286) |
| test/contrib_ops/layer_norm_op_test/LayerNorm_Scale_Bias_PerLastDim_run0/model.onnx | 17 | Data/Data | ❌ | Out of tolerance (max ULP 336) |
| test/contrib_ops/layer_norm_op_test/LayerNorm_Scale_Bias_Scalar_Axis2_run0/model.onnx | 17 | Data/Data | ❌ | Out of tolerance (max ULP 252) |
| test/contrib_ops/layer_norm_op_test/LayerNorm_Scale_Bias_Scalar_run0/model.onnx | 17 | Data/Data | ❌ | Out of tolerance (max ULP 252) |
| test/contrib_ops/layer_norm_op_test/LayerNorm_Scale_Bias_run0/model.onnx | 7 | Data/Data | ❌ | Out of tolerance (max ULP 10874) |
| test/contrib_ops/layer_norm_op_test/LayerNorm_Scale_Scalar_NoBias_Axis2_run0/model.onnx | 17 | Data/Data | ❌ | Out of tolerance (max ULP 252) |
| test/contrib_ops/layer_norm_op_test/LayerNorm_Scale_Scalar_NoBias_run0/model.onnx | 17 | Data/Data | ❌ | Out of tolerance (max ULP 252) |
| test/contrib_ops/layer_norm_op_test/LayerNorm_run0/model.onnx | 7 | Data/Data | ❌ | Out of tolerance (max ULP 299) |
| test/contrib_ops/linear_attention_op_test/DeltaRule_MultiToken_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.LinearAttention |
| test/contrib_ops/linear_attention_op_test/DeltaRule_SingleToken_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.LinearAttention |
| test/contrib_ops/linear_attention_op_test/GatedDeltaRule_BroadcastDecay_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.LinearAttention |
| test/contrib_ops/linear_attention_op_test/GatedDeltaRule_InverseGQA_LargerDims_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.LinearAttention |
| test/contrib_ops/linear_attention_op_test/GatedDeltaRule_InverseGQA_Small_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.LinearAttention |
| test/contrib_ops/linear_attention_op_test/GatedDeltaRule_KGQA_Small_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.LinearAttention |
| test/contrib_ops/linear_attention_op_test/GatedDeltaRule_LargerDims_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.LinearAttention |
| test/contrib_ops/linear_attention_op_test/GatedDeltaRule_LongerSequence_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.LinearAttention |
| test/contrib_ops/linear_attention_op_test/GatedDeltaRule_MultiBatchMultiHead_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.LinearAttention |
| test/contrib_ops/linear_attention_op_test/GatedDeltaRule_MultiToken_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.LinearAttention |
| test/contrib_ops/linear_attention_op_test/GatedDeltaRule_NonPowerOf2DK_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.LinearAttention |
| test/contrib_ops/linear_attention_op_test/GatedDeltaRule_Qwen35Like_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.LinearAttention |
| test/contrib_ops/linear_attention_op_test/GatedDeltaRule_Qwen35_KGQA_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.LinearAttention |
| test/contrib_ops/linear_attention_op_test/GatedDeltaRule_SingleToken_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.LinearAttention |
| test/contrib_ops/linear_attention_op_test/GatedRule_BroadcastDecay_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.LinearAttention |
| test/contrib_ops/linear_attention_op_test/GatedRule_LargerDims_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.LinearAttention |
| test/contrib_ops/linear_attention_op_test/GatedRule_MultiToken_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.LinearAttention |
| test/contrib_ops/linear_attention_op_test/GatedRule_SingleToken_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.LinearAttention |
| test/contrib_ops/linear_attention_op_test/LinearRule_DefaultScale_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.LinearAttention |
| test/contrib_ops/linear_attention_op_test/LinearRule_InverseGQA_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.LinearAttention |
| test/contrib_ops/linear_attention_op_test/LinearRule_KGQA_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.LinearAttention |
| test/contrib_ops/linear_attention_op_test/LinearRule_LargerDims_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.LinearAttention |
| test/contrib_ops/linear_attention_op_test/LinearRule_MultiBatchMultiHead_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.LinearAttention |
| test/contrib_ops/linear_attention_op_test/LinearRule_MultiToken_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.LinearAttention |
| test/contrib_ops/linear_attention_op_test/LinearRule_SingleToken_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.LinearAttention |
| test/contrib_ops/linear_attention_op_test/LinearRule_WithInitialState_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.LinearAttention |
| test/contrib_ops/math/matmul_sparse_test/TestCoo_run0/model.onnx |  | Data/Data | ❌ | Unsupported value type 'sparse_tensor_type' for 'A'. Hint: export the model with tensor inputs/outputs. |
| test/contrib_ops/math/matmul_sparse_test/TestCoo_run1/model.onnx |  | Data/Data | ❌ | Unsupported value type 'sparse_tensor_type' for 'A'. Hint: export the model with tensor inputs/outputs. |
| test/contrib_ops/math/matmul_sparse_test/TestCoo_run2/model.onnx |  | Data/Data | ❌ | Unsupported value type 'sparse_tensor_type' for 'A'. Hint: export the model with tensor inputs/outputs. |
| test/contrib_ops/math/matmul_sparse_test/TestCoo_run3/model.onnx |  | Data/Data | ❌ | Unsupported value type 'sparse_tensor_type' for 'A'. Hint: export the model with tensor inputs/outputs. |
| test/contrib_ops/math/matmul_sparse_test/TestCsr_run0/model.onnx |  | Data/Data | ❌ | Unsupported value type 'sparse_tensor_type' for 'A'. Hint: export the model with tensor inputs/outputs. |
| test/contrib_ops/math/matmul_sparse_test/TestCsr_run1/model.onnx |  | Data/Data | ❌ | Unsupported value type 'sparse_tensor_type' for 'A'. Hint: export the model with tensor inputs/outputs. |
| test/contrib_ops/math/matmul_sparse_test/TestCsr_run2/model.onnx |  | Data/Data | ❌ | Unsupported value type 'sparse_tensor_type' for 'A'. Hint: export the model with tensor inputs/outputs. |
| test/contrib_ops/math/matmul_sparse_test/TestCsr_run3/model.onnx |  | Data/Data | ❌ | Unsupported value type 'sparse_tensor_type' for 'A'. Hint: export the model with tensor inputs/outputs. |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run145/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 10976) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run146/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 11529) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run147/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 10976) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run148/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 11961) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run149/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 11529) |
| test/contrib_ops/multihead_attention_op_test/CrossAttention_WithPastPassedInDirectly_NoMask_run0/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 64137) |
| test/contrib_ops/multihead_attention_op_test/CrossAttention_WithPastPassedInDirectly_NoMask_run1/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 64137) |
| test/contrib_ops/qlinear_lookup_table_test/QLinearSoftmax_Int8_v13_run0/model.onnx |  | Data/Data | ❌ | Out of tolerance (max abs diff 1) |
| test/contrib_ops/qlinear_lookup_table_test/QLinearSoftmax_Int8_v13_run1/model.onnx |  | Data/Data | ❌ | Out of tolerance (max abs diff 1) |
| test/contrib_ops/qlinear_lookup_table_test/QLinearSoftmax_UInt8_v13_run0/model.onnx |  | Data/Data | ❌ | Out of tolerance (max abs diff 1) |
| test/contrib_ops/qlinear_lookup_table_test/QLinearSoftmax_UInt8_v13_run1/model.onnx |  | Data/Data | ❌ | Out of tolerance (max abs diff 1) |
| test/contrib_ops/rotary_embedding_op_test/RotaryEmbedding_CustomRotaryDim_SmallData_Phi_Packed_Batching_run0/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 6754) |
| test/contrib_ops/rotary_embedding_op_test/RotaryEmbedding_CustomRotaryDim_SmallData_Phi_run0/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 6754) |
| test/contrib_ops/rotary_embedding_op_test/RotaryEmbedding_Interleaved_LargeData_LlamaMSFT_run0/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 177823) |
| test/contrib_ops/rotary_embedding_op_test/RotaryEmbedding_Interleaved_SmallData_LlamaMSFT_run0/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 6391) |
| test/contrib_ops/rotary_embedding_op_test/RotaryEmbedding_NotInterleaved_LargeData_LlamaMSFT_run0/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 52786) |
| test/contrib_ops/rotary_embedding_op_test/RotaryEmbedding_NotInterleaved_SmallData_LlamaMSFT_run0/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 26739) |

<!-- AUTO-GENERATED FILE. DO NOT EDIT. -->
<!-- Regenerate with: UPDATE_REFS=1 pytest -q tests/test_official_onnx_files_docs.py::test_official_onnx_file_support_doc -->

# ONNX verification errors

Aggregates non-success verification outcomes.

| Error message | Count | Opset versions |
| --- | --- | --- |
| Out of tolerance | 32 | 7, 17 |
| Testbench execution failed: exit code 1 | 10 |  |
| GatherBlockQuantized axis -1 is out of range for rank 3 | 8 |  |
| GatherBlockQuantized indices must be int32 or int64 | 8 |  |
| Unsupported value type '*' for '*'. Hint: export the model with tensor inputs/outputs. | 8 |  |
| GridSample mode '*' is not supported | 7 |  |
| Unsupported op com.microsoft.AttnLSTM | 6 |  |
| Unsupported op com.microsoft.WordConvEmbedding | 6 |  |
| RotaryEmbedding inputs must share the same dtype | 5 | 23 |
| GatherBlockQuantized axis 3 is out of range for rank 3 | 4 |  |
| GatherBlockQuantized scales shape (2, 2, 1) does not match expected (2, 3, 1) (data_shape=(2, 3, 4), quantize_axis=2, block_size=16, packed=False) | 4 |  |
| GatherBlockQuantized scales shape (2, 3, 1) does not match expected (2, 3, 2) (data_shape=(2, 3, 4), quantize_axis=2, block_size=16, packed=True) | 4 |  |
| QLinearSoftmax axis -2 is out of bounds for shape () | 4 |  |
| Unique must have 1 input and 4 outputs | 4 |  |
| GatherBlockQuantized data dtype must be integer, got int64 | 2 |  |
| GatherBlockQuantized data dtype must be integer, got uint64 | 2 |  |
| GatherBlockQuantized scales shape (2, 2, 1) does not match expected (2, 3, 1) (data_shape=(2, 3, 2), quantize_axis=2, block_size=16, packed=True) | 2 |  |
| GatherBlockQuantized supports bits in [4, 8], got 1 | 2 |  |
| GatherBlockQuantized supports bits in [4, 8], got 2 | 2 |  |
| GatherBlockQuantized supports bits in [4, 8], got 3 | 2 |  |
| GatherBlockQuantized supports bits in [4, 8], got 5 | 2 |  |
| GatherBlockQuantized supports bits in [4, 8], got 6 | 2 |  |
| GatherBlockQuantized supports bits in [4, 8], got 7 | 2 |  |
| Output shape mismatch for output (actual_shape=(1, 3, 8), actual_size=24, expected_shape=(1, 3, 4), expected_size=12, output=output) | 2 |  |
| Output shape mismatch for output (actual_shape=(2, 1, 4), actual_size=8, expected_shape=(1, 3, 4), expected_size=12, output=output) | 2 |  |
| Unsupported op com.microsoft.BifurcationDetector | 2 |  |
| Unsupported op com.microsoft.DecoderMaskedMultiHeadAttention | 2 |  |
| Unsupported op com.microsoft.UnfoldTensor | 2 |  |
| CropAndResize method '*' is not supported | 1 |  |
| MurmurHash3 does not support input dtype int8; supported: int32, int64, float, double, string | 1 |  |
| Unsupported op com.microsoft.ConvTransposeWithDynamicPads | 1 |  |
| Unsupported op com.microsoft.MaxpoolWithMask | 1 |  |
| Unsupported op com.microsoft.MoE | 1 |  |
| Unsupported op com.microsoft.SampleOp | 1 |  |

## Error frequency by opset

| Error message | Opset | Count |
| --- | --- | --- |
| Out of tolerance | 7 | 7 |
| Out of tolerance | 17 | 13 |
| RotaryEmbedding inputs must share the same dtype | 23 | 5 |

## Failing ONNX files

Lists every ONNX file with a non-success verification outcome.

| File | Opset | Verification | Supported | Error |
| --- | --- | --- | --- | --- |
| node/test_adam_multiple/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 62311) |
| node/test_rotary_embedding/model.onnx | 23 | Data/Data | ❌ | RotaryEmbedding inputs must share the same dtype |
| node/test_rotary_embedding_3d_input/model.onnx | 23 | Data/Data | ❌ | RotaryEmbedding inputs must share the same dtype |
| node/test_rotary_embedding_interleaved/model.onnx | 23 | Data/Data | ❌ | RotaryEmbedding inputs must share the same dtype |
| node/test_rotary_embedding_with_interleaved_rotary_dim/model.onnx | 23 | Data/Data | ❌ | RotaryEmbedding inputs must share the same dtype |
| node/test_rotary_embedding_with_rotary_dim/model.onnx | 23 | Data/Data | ❌ | RotaryEmbedding inputs must share the same dtype |
| test/contrib_ops/attention_lstm_op_test/BidirectionLstmWithBahdanauAM2BatchShortenSeqLen_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.AttnLSTM |
| test/contrib_ops/attention_lstm_op_test/BidirectionLstmWithBahdanauAMShortenSeqLength_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.AttnLSTM |
| test/contrib_ops/attention_lstm_op_test/ForwardLstmWithBahdanauAMShortenSeqLength_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.AttnLSTM |
| test/contrib_ops/attention_lstm_op_test/ForwardLstmWithBahdanauAMZeroAttention_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.AttnLSTM |
| test/contrib_ops/attention_lstm_op_test/ForwardLstmWithBahdanauAM_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.AttnLSTM |
| test/contrib_ops/attention_lstm_op_test/ReverseLstmWithBahdanauAMShortenSeqLength_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.AttnLSTM |
| test/contrib_ops/attention_op_test/AttentionPastState_dynamic_run0/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 58877) |
| test/contrib_ops/attention_op_test/Attention_Mask1D_Fp32_B2_S64_run0/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 739520) |
| test/contrib_ops/attention_op_test/Attention_Mask2D_Fp32_B2_S32_run0/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 980293) |
| test/contrib_ops/bifurcation_detector_op_test/Test1_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.BifurcationDetector |
| test/contrib_ops/bifurcation_detector_op_test/Test2_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.BifurcationDetector |
| test/contrib_ops/conv_transpose_with_dynamic_pads_test/ConvTransposeWithDynamicPads_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.ConvTransposeWithDynamicPads |
| test/contrib_ops/crop_and_resize_op_test/CropAndResize_1133_run2/model.onnx |  | Data/Data | ❌ | CropAndResize method 'nearest' is not supported |
| test/contrib_ops/decoder_masked_multihead_attention_op_test/cpu_cross_attn_fp32_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.DecoderMaskedMultiHeadAttention |
| test/contrib_ops/decoder_masked_multihead_attention_op_test/cpu_self_attn_fp32_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.DecoderMaskedMultiHeadAttention |
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
| test/contrib_ops/gather_block_quantized_op_test/UnsupportedTypes_run2/model.onnx |  | Data/Data | ❌ | Testbench execution failed: exit code 1 |
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
| test/contrib_ops/gridsample_test/gridsample_paddingmode_border_run0/model.onnx |  | Data/Data | ❌ | GridSample mode 'bilinear' is not supported |
| test/contrib_ops/gridsample_test/gridsample_paddingmode_reflection_run0/model.onnx |  | Data/Data | ❌ | GridSample mode 'bilinear' is not supported |
| test/contrib_ops/gridsample_test/gridsample_paddingmode_zeros_run0/model.onnx |  | Data/Data | ❌ | GridSample mode 'bilinear' is not supported |
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
| test/contrib_ops/math/matmul_sparse_test/TestCoo_run0/model.onnx |  | Data/Data | ❌ | Unsupported value type 'sparse_tensor_type' for 'A'. Hint: export the model with tensor inputs/outputs. |
| test/contrib_ops/math/matmul_sparse_test/TestCoo_run1/model.onnx |  | Data/Data | ❌ | Unsupported value type 'sparse_tensor_type' for 'A'. Hint: export the model with tensor inputs/outputs. |
| test/contrib_ops/math/matmul_sparse_test/TestCoo_run2/model.onnx |  | Data/Data | ❌ | Unsupported value type 'sparse_tensor_type' for 'A'. Hint: export the model with tensor inputs/outputs. |
| test/contrib_ops/math/matmul_sparse_test/TestCoo_run3/model.onnx |  | Data/Data | ❌ | Unsupported value type 'sparse_tensor_type' for 'A'. Hint: export the model with tensor inputs/outputs. |
| test/contrib_ops/math/matmul_sparse_test/TestCsr_run0/model.onnx |  | Data/Data | ❌ | Unsupported value type 'sparse_tensor_type' for 'A'. Hint: export the model with tensor inputs/outputs. |
| test/contrib_ops/math/matmul_sparse_test/TestCsr_run1/model.onnx |  | Data/Data | ❌ | Unsupported value type 'sparse_tensor_type' for 'A'. Hint: export the model with tensor inputs/outputs. |
| test/contrib_ops/math/matmul_sparse_test/TestCsr_run2/model.onnx |  | Data/Data | ❌ | Unsupported value type 'sparse_tensor_type' for 'A'. Hint: export the model with tensor inputs/outputs. |
| test/contrib_ops/math/matmul_sparse_test/TestCsr_run3/model.onnx |  | Data/Data | ❌ | Unsupported value type 'sparse_tensor_type' for 'A'. Hint: export the model with tensor inputs/outputs. |
| test/contrib_ops/maxpool_mask_test/MaxPoolWithMask_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MaxpoolWithMask |
| test/contrib_ops/moe_test/MoECpuTest_BasicSwiGLU_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MoE |
| test/contrib_ops/murmur_hash3_test/UnsupportedInputType_run0/model.onnx |  | Data/Data | ❌ | MurmurHash3 does not support input dtype int8; supported: int32, int64, float, double, string |
| test/contrib_ops/qlinear_lookup_table_test/QLinearSoftmax_Int8_v12_run1/model.onnx |  | Data/Data | ❌ | QLinearSoftmax axis -2 is out of bounds for shape () |
| test/contrib_ops/qlinear_lookup_table_test/QLinearSoftmax_Int8_v13_run0/model.onnx |  | Data/Data | ❌ | Out of tolerance (max abs diff 1) |
| test/contrib_ops/qlinear_lookup_table_test/QLinearSoftmax_Int8_v13_run1/model.onnx |  | Data/Data | ❌ | QLinearSoftmax axis -2 is out of bounds for shape () |
| test/contrib_ops/qlinear_lookup_table_test/QLinearSoftmax_UInt8_v12_run1/model.onnx |  | Data/Data | ❌ | QLinearSoftmax axis -2 is out of bounds for shape () |
| test/contrib_ops/qlinear_lookup_table_test/QLinearSoftmax_UInt8_v13_run0/model.onnx |  | Data/Data | ❌ | Out of tolerance (max abs diff 1) |
| test/contrib_ops/qlinear_lookup_table_test/QLinearSoftmax_UInt8_v13_run1/model.onnx |  | Data/Data | ❌ | QLinearSoftmax axis -2 is out of bounds for shape () |
| test/contrib_ops/rotary_embedding_op_test/RotaryEmbedding_CustomRotaryDim_SmallData_Phi_Packed_Batching_run0/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 6754) |
| test/contrib_ops/rotary_embedding_op_test/RotaryEmbedding_CustomRotaryDim_SmallData_Phi_run0/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 6754) |
| test/contrib_ops/rotary_embedding_op_test/RotaryEmbedding_Interleaved_LargeData_LlamaMSFT_run0/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 177823) |
| test/contrib_ops/rotary_embedding_op_test/RotaryEmbedding_Interleaved_SmallData_LlamaMSFT_run0/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 6391) |
| test/contrib_ops/rotary_embedding_op_test/RotaryEmbedding_NotInterleaved_LargeData_LlamaMSFT_run0/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 52786) |
| test/contrib_ops/rotary_embedding_op_test/RotaryEmbedding_NotInterleaved_SmallData_LlamaMSFT_run0/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 26739) |
| test/contrib_ops/sample_op_test/SampleOpFloat_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.SampleOp |
| test/contrib_ops/tensor_op_test/LastDim_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.UnfoldTensor |
| test/contrib_ops/tensor_op_test/MeanVarianceNormalizationCPUTest_Version1_TO_8_run0/model.onnx | 7 | Data/Data | ❌ | Out of tolerance (max ULP 7137119) |
| test/contrib_ops/tensor_op_test/MeanVarianceNormalizationCPUTest_Version1_TO_8_run1/model.onnx | 7 | Data/Data | ❌ | Out of tolerance (max ULP 21399263) |
| test/contrib_ops/tensor_op_test/MeanVarianceNormalizationCPUTest_Version1_TO_8_run2/model.onnx | 7 | Data/Data | ❌ | Out of tolerance (max ULP 15806857) |
| test/contrib_ops/tensor_op_test/NormalDim_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.UnfoldTensor |
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

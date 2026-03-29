<!-- AUTO-GENERATED FILE. DO NOT EDIT. -->
<!-- Regenerate with: UPDATE_REFS=1 pytest -q tests/test_official_onnx_files_docs.py::test_official_onnx_file_support_doc -->

# ONNX verification errors

Aggregates non-success verification outcomes.

| Error message | Count | Opset versions |
| --- | --- | --- |
| MatMulNBits g_idx (input 4) is not supported | 238 |  |
| AveragePool has unsupported attributes | 72 |  |
| Out of tolerance | 72 | 7, 17 |
| Unsupported op com.microsoft.MatMulIntegerToFloat | 48 |  |
| Unsupported op com.microsoft.DynamicQuantizeMatMul | 39 |  |
| Unsupported op com.microsoft.QLinearConcat | 36 |  |
| Unsupported op com.microsoft.QLinearGlobalAveragePool | 36 |  |
| Unsupported op com.microsoft.MultiHeadAttention | 28 |  |
| Unsupported op com.microsoft.Tokenizer | 28 |  |
| Unsupported op com.microsoft.QAttention | 24 |  |
| Unsupported op com.microsoft.SkipLayerNormalization | 17 |  |
| LayerNormalization scale rank must match normalized rank | 15 | 7, 17 |
| RotaryEmbedding inputs must share the same dtype | 12 |  |
| Unsupported op ai.onnx.DynamicSlice | 12 | 1 |
| Unsupported op com.microsoft.QEmbedLayerNormalization | 12 |  |
| Unsupported op com.microsoft.ExpandDims | 11 |  |
| Testbench execution failed: exit code 1 | 10 |  |
| Unsupported op com.microsoft.CropAndResize | 10 |  |
| Unsupported op com.microsoft.EmbedLayerNormalization | 10 |  |
| Unsupported op com.microsoft.QLinearWhere | 10 |  |
| GatherBlockQuantized axis -1 is out of range for rank 3 | 8 |  |
| GatherBlockQuantized indices must be int32 or int64 | 8 |  |
| Unsupported value type '*' for '*'. Hint: export the model with tensor inputs/outputs. | 8 |  |
| GridSample mode '*' is not supported | 7 |  |
| Unsupported op com.microsoft.AttnLSTM | 6 |  |
| Unsupported op com.microsoft.BiasGelu | 6 |  |
| Unsupported op com.microsoft.WordConvEmbedding | 6 |  |
| Unsupported op com.microsoft.Inverse | 5 |  |
| FusedMatMul batch dimensions are not broadcastable: (2,) vs (4,) | 4 |  |
| GatherBlockQuantized axis 3 is out of range for rank 3 | 4 |  |
| GatherBlockQuantized scales shape (2, 2, 1) does not match expected (2, 3, 1) (data_shape=(2, 3, 4), quantize_axis=2, block_size=16, packed=False) | 4 |  |
| GatherBlockQuantized scales shape (2, 3, 1) does not match expected (2, 3, 2) (data_shape=(2, 3, 4), quantize_axis=2, block_size=16, packed=True) | 4 |  |
| QLinearSoftmax axis -2 is out of bounds for shape () | 4 |  |
| Unique must have 1 input and 4 outputs | 4 |  |
| Unsupported op ai.onnx.Crop | 4 | 1, 7 |
| Unsupported op com.microsoft.FusedConv | 4 |  |
| Unsupported op com.microsoft.MatMulInteger16 | 4 |  |
| Unsupported op com.microsoft.QLinearSigmoid | 4 |  |
| FusedMatMul batch dimensions are not broadcastable: (1, 3) vs (3, 2) | 3 |  |
| Output shape mismatch for Y (actual_shape=(3, 2, 3, 2), actual_size=36, expected_shape=(3, 2, 3, 1), expected_size=18, output=Y) | 3 |  |
| Unsupported op com.microsoft.CDist | 3 |  |
| Unsupported op com.microsoft.FastGelu | 3 |  |
| Failed to build testbench (model.c:125:31: error: passing argument 3 of ‘node0_node1’ from incompatible pointer type [-Wincompatible-pointer-types]). | 2 |  |
| Failed to build testbench (model.c:137:31: error: passing argument 3 of ‘node0_node1’ from incompatible pointer type [-Wincompatible-pointer-types]). | 2 |  |
| Failed to build testbench (model.c:139:31: error: passing argument 3 of ‘node0_node1’ from incompatible pointer type [-Wincompatible-pointer-types]). | 2 |  |
| FusedMatMul inner dimensions must match after transposition, got effective shapes (2, 2, 3) and (3, 1, 4) (original: (2, 2, 3) and (3, 1, 4)) | 2 |  |
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
| Unsupported op ai.onnx.Affine | 2 | 7 |
| Unsupported op ai.onnx.Scale | 2 | 7 |
| Unsupported op com.microsoft.BifurcationDetector | 2 |  |
| Unsupported op com.microsoft.DecoderMaskedMultiHeadAttention | 2 |  |
| Unsupported op com.microsoft.QLinearLeakyRelu | 2 |  |
| Unsupported op com.microsoft.UnfoldTensor | 2 |  |
| FusedMatMul inner dimensions must match after transposition, got effective shapes (1, 2, 3) and (3, 1, 4) (original: (1, 2, 3) and (3, 1, 4)) | 1 |  |
| FusedMatMul inner dimensions must match after transposition, got effective shapes (1, 2, 3) and (3, 1, 4) (original: (1, 3, 2) and (3, 1, 4)) | 1 |  |
| FusedMatMul inner dimensions must match after transposition, got effective shapes (2, 1, 3) and (3, 1, 4) (original: (2, 1, 3) and (3, 1, 4)) | 1 |  |
| FusedMatMul inner dimensions must match after transposition, got effective shapes (2, 2, 3) and (3, 1, 4) (original: (2, 3, 2) and (3, 1, 4)) | 1 |  |
| FusedMatMul inner dimensions must match after transposition, got effective shapes (3, 2, 1) and (1, 3, 4) (original: (3, 1, 2) and (1, 3, 4)) | 1 |  |
| FusedMatMul inner dimensions must match after transposition, got effective shapes (3, 2, 1) and (1, 3, 4) (original: (3, 1, 2) and (1, 4, 3)) | 1 |  |
| FusedMatMul inner dimensions must match after transposition, got effective shapes (3, 2, 1) and (4, 3, 1) (original: (3, 1, 2) and (4, 1, 3)) | 1 |  |
| FusedMatMul inner dimensions must match after transposition, got effective shapes (3, 2, 2) and (1, 3, 4) (original: (3, 2, 2) and (1, 3, 4)) | 1 |  |
| FusedMatMul inner dimensions must match after transposition, got effective shapes (3, 2, 2) and (1, 3, 4) (original: (3, 2, 2) and (1, 4, 3)) | 1 |  |
| FusedMatMul inner dimensions must match after transposition, got effective shapes (3, 2, 2) and (3, 1, 4) (original: (3, 2, 2) and (3, 1, 4)) | 1 |  |
| FusedMatMul inner dimensions must match after transposition, got effective shapes (3, 2, 2) and (4, 3, 1) (original: (3, 2, 2) and (4, 1, 3)) | 1 |  |
| MurmurHash3 does not support input dtype int8; supported: int32, int64, float, double, string | 1 |  |
| Output shape mismatch for Y (actual_shape=(3, 2, 4), actual_size=24, expected_shape=(1, 2, 4), expected_size=8, output=Y) | 1 |  |
| Output shape mismatch for Y (actual_shape=(3, 3, 2, 2), actual_size=36, expected_shape=(3, 2, 3, 1), expected_size=18, output=Y) | 1 |  |
| Unsupported op ai.onnx.ImageScaler | 1 | 7 |
| Unsupported op com.microsoft.ConvTransposeWithDynamicPads | 1 |  |
| Unsupported op com.microsoft.DynamicTimeWarping | 1 |  |
| Unsupported op com.microsoft.MaxpoolWithMask | 1 |  |
| Unsupported op com.microsoft.MoE | 1 |  |
| Unsupported op com.microsoft.NGramRepeatBlock | 1 |  |
| Unsupported op com.microsoft.SampleOp | 1 |  |

## Error frequency by opset

| Error message | Opset | Count |
| --- | --- | --- |
| Unsupported op ai.onnx.DynamicSlice | 1 | 12 |
| Unsupported op ai.onnx.Crop | 1 | 2 |
| LayerNormalization scale rank must match normalized rank | 7 | 9 |
| Out of tolerance | 7 | 5 |
| Unsupported op ai.onnx.Affine | 7 | 2 |
| Unsupported op ai.onnx.Crop | 7 | 2 |
| Unsupported op ai.onnx.Scale | 7 | 2 |
| Unsupported op ai.onnx.ImageScaler | 7 | 1 |
| Out of tolerance | 17 | 7 |
| LayerNormalization scale rank must match normalized rank | 17 | 6 |

## Failing ONNX files

Lists every ONNX file with a non-success verification outcome.

| File | Opset | Verification | Supported | Error |
| --- | --- | --- | --- | --- |
| node/test_adam_multiple/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 62311) |
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
| test/contrib_ops/crop_op_test/Crop_Border_run0/model.onnx | 1 | Data/Data | ❌ | Unsupported op ai.onnx.Crop |
| test/contrib_ops/crop_op_test/Crop_Scale_run0/model.onnx | 1 | Data/Data | ❌ | Unsupported op ai.onnx.Crop |
| test/contrib_ops/decoder_masked_multihead_attention_op_test/cpu_cross_attn_fp32_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.DecoderMaskedMultiHeadAttention |
| test/contrib_ops/decoder_masked_multihead_attention_op_test/cpu_self_attn_fp32_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.DecoderMaskedMultiHeadAttention |
| test/contrib_ops/dynamic_quantize_matmul_test/Int8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.DynamicQuantizeMatMul |
| test/contrib_ops/dynamic_quantize_matmul_test/Int8_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.DynamicQuantizeMatMul |
| test/contrib_ops/dynamic_quantize_matmul_test/Int8_run10/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.DynamicQuantizeMatMul |
| test/contrib_ops/dynamic_quantize_matmul_test/Int8_run11/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.DynamicQuantizeMatMul |
| test/contrib_ops/dynamic_quantize_matmul_test/Int8_run12/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.DynamicQuantizeMatMul |
| test/contrib_ops/dynamic_quantize_matmul_test/Int8_run13/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.DynamicQuantizeMatMul |
| test/contrib_ops/dynamic_quantize_matmul_test/Int8_run14/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.DynamicQuantizeMatMul |
| test/contrib_ops/dynamic_quantize_matmul_test/Int8_run15/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.DynamicQuantizeMatMul |
| test/contrib_ops/dynamic_quantize_matmul_test/Int8_run2/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.DynamicQuantizeMatMul |
| test/contrib_ops/dynamic_quantize_matmul_test/Int8_run3/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.DynamicQuantizeMatMul |
| test/contrib_ops/dynamic_quantize_matmul_test/Int8_run4/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.DynamicQuantizeMatMul |
| test/contrib_ops/dynamic_quantize_matmul_test/Int8_run5/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.DynamicQuantizeMatMul |
| test/contrib_ops/dynamic_quantize_matmul_test/Int8_run6/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.DynamicQuantizeMatMul |
| test/contrib_ops/dynamic_quantize_matmul_test/Int8_run7/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.DynamicQuantizeMatMul |
| test/contrib_ops/dynamic_quantize_matmul_test/Int8_run8/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.DynamicQuantizeMatMul |
| test/contrib_ops/dynamic_quantize_matmul_test/Int8_run9/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.DynamicQuantizeMatMul |
| test/contrib_ops/dynamic_quantize_matmul_test/UInt8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.DynamicQuantizeMatMul |
| test/contrib_ops/dynamic_quantize_matmul_test/UInt8_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.DynamicQuantizeMatMul |
| test/contrib_ops/dynamic_quantize_matmul_test/UInt8_run10/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.DynamicQuantizeMatMul |
| test/contrib_ops/dynamic_quantize_matmul_test/UInt8_run11/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.DynamicQuantizeMatMul |
| test/contrib_ops/dynamic_quantize_matmul_test/UInt8_run12/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.DynamicQuantizeMatMul |
| test/contrib_ops/dynamic_quantize_matmul_test/UInt8_run13/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.DynamicQuantizeMatMul |
| test/contrib_ops/dynamic_quantize_matmul_test/UInt8_run14/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.DynamicQuantizeMatMul |
| test/contrib_ops/dynamic_quantize_matmul_test/UInt8_run15/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.DynamicQuantizeMatMul |
| test/contrib_ops/dynamic_quantize_matmul_test/UInt8_run2/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.DynamicQuantizeMatMul |
| test/contrib_ops/dynamic_quantize_matmul_test/UInt8_run3/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.DynamicQuantizeMatMul |
| test/contrib_ops/dynamic_quantize_matmul_test/UInt8_run4/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.DynamicQuantizeMatMul |
| test/contrib_ops/dynamic_quantize_matmul_test/UInt8_run5/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.DynamicQuantizeMatMul |
| test/contrib_ops/dynamic_quantize_matmul_test/UInt8_run6/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.DynamicQuantizeMatMul |
| test/contrib_ops/dynamic_quantize_matmul_test/UInt8_run7/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.DynamicQuantizeMatMul |
| test/contrib_ops/dynamic_quantize_matmul_test/UInt8_run8/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.DynamicQuantizeMatMul |
| test/contrib_ops/dynamic_quantize_matmul_test/UInt8_run9/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.DynamicQuantizeMatMul |
| test/contrib_ops/dynamic_quantize_matmul_test/UInt8_test_with_empty_input_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.DynamicQuantizeMatMul |
| test/contrib_ops/dynamic_quantize_matmul_test/WithConstantBInputs_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.DynamicQuantizeMatMul |
| test/contrib_ops/dynamic_quantize_matmul_test/WithConstantBInputs_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.DynamicQuantizeMatMul |
| test/contrib_ops/dynamic_quantize_matmul_test/WithConstantBInputs_run2/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.DynamicQuantizeMatMul |
| test/contrib_ops/dynamic_quantize_matmul_test/WithConstantBInputs_run3/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.DynamicQuantizeMatMul |
| test/contrib_ops/dynamic_quantize_matmul_test/WithConstantBInputs_run4/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.DynamicQuantizeMatMul |
| test/contrib_ops/dynamic_quantize_matmul_test/WithConstantBInputs_run5/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.DynamicQuantizeMatMul |
| test/contrib_ops/dynamic_slice_op_test/dynamic_slice_ends_out_of_bounds_run0/model.onnx | 1 | Data/Data | ❌ | Unsupported op ai.onnx.DynamicSlice |
| test/contrib_ops/dynamic_slice_op_test/dynamic_slice_full_axes_run0/model.onnx | 1 | Data/Data | ❌ | Unsupported op ai.onnx.DynamicSlice |
| test/contrib_ops/dynamic_slice_op_test/dynamic_slice_full_axes_run1/model.onnx | 1 | Data/Data | ❌ | Unsupported op ai.onnx.DynamicSlice |
| test/contrib_ops/dynamic_slice_op_test/dynamic_slice_varied_types_run0/model.onnx | 1 | Data/Data | ❌ | Unsupported op ai.onnx.DynamicSlice |
| test/contrib_ops/dynamic_slice_op_test/dynamic_slice_varied_types_run1/model.onnx | 1 | Data/Data | ❌ | Unsupported op ai.onnx.DynamicSlice |
| test/contrib_ops/dynamic_slice_op_test/dynamic_slice_varied_types_run2/model.onnx | 1 | Data/Data | ❌ | Unsupported op ai.onnx.DynamicSlice |
| test/contrib_ops/dynamic_slice_op_test/dynamic_slice_varied_types_run3/model.onnx | 1 | Data/Data | ❌ | Unsupported op ai.onnx.DynamicSlice |
| test/contrib_ops/dynamic_slice_op_test/dynamic_slice_varied_types_run4/model.onnx | 1 | Data/Data | ❌ | Unsupported op ai.onnx.DynamicSlice |
| test/contrib_ops/dynamic_slice_op_test/dynamic_slice_with_axes_run0/model.onnx | 1 | Data/Data | ❌ | Unsupported op ai.onnx.DynamicSlice |
| test/contrib_ops/dynamic_slice_op_test/dynamic_slice_with_axes_run1/model.onnx | 1 | Data/Data | ❌ | Unsupported op ai.onnx.DynamicSlice |
| test/contrib_ops/dynamic_slice_op_test/dynamic_slice_with_negative_axes_run0/model.onnx | 1 | Data/Data | ❌ | Unsupported op ai.onnx.DynamicSlice |
| test/contrib_ops/dynamic_slice_op_test/dynamic_slice_with_negative_axes_run1/model.onnx | 1 | Data/Data | ❌ | Unsupported op ai.onnx.DynamicSlice |
| test/contrib_ops/dynamic_time_warping_op_test/simple_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.DynamicTimeWarping |
| test/contrib_ops/element_wise_ops_test/AffineDefaultAttributes_run0/model.onnx | 7 | Data/Data | ❌ | Unsupported op ai.onnx.Affine |
| test/contrib_ops/element_wise_ops_test/Affine_run0/model.onnx | 7 | Data/Data | ❌ | Unsupported op ai.onnx.Affine |
| test/contrib_ops/element_wise_ops_test/Float_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.BiasGelu |
| test/contrib_ops/element_wise_ops_test/Float_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.BiasGelu |
| test/contrib_ops/element_wise_ops_test/Float_run2/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.BiasGelu |
| test/contrib_ops/element_wise_ops_test/Float_run3/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.BiasGelu |
| test/contrib_ops/element_wise_ops_test/Float_run4/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.BiasGelu |
| test/contrib_ops/element_wise_ops_test/Float_run5/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.BiasGelu |
| test/contrib_ops/element_wise_ops_test/Scale_Default_run0/model.onnx | 7 | Data/Data | ❌ | Unsupported op ai.onnx.Scale |
| test/contrib_ops/element_wise_ops_test/Scale_run0/model.onnx | 7 | Data/Data | ❌ | Unsupported op ai.onnx.Scale |
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
| test/contrib_ops/expand_dims_test/Basic_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.ExpandDims |
| test/contrib_ops/expand_dims_test/Basic_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.ExpandDims |
| test/contrib_ops/expand_dims_test/Basic_run2/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.ExpandDims |
| test/contrib_ops/expand_dims_test/MaxAxis_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.ExpandDims |
| test/contrib_ops/expand_dims_test/MaxAxis_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.ExpandDims |
| test/contrib_ops/expand_dims_test/MinAxis_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.ExpandDims |
| test/contrib_ops/expand_dims_test/MinAxis_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.ExpandDims |
| test/contrib_ops/expand_dims_test/NegativeAxisOutOfRange_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.ExpandDims |
| test/contrib_ops/expand_dims_test/NegativeAxisOutOfRange_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.ExpandDims |
| test/contrib_ops/expand_dims_test/PositiveAxisOutOfRange_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.ExpandDims |
| test/contrib_ops/expand_dims_test/PositiveAxisOutOfRange_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.ExpandDims |
| test/contrib_ops/fastgelu_op_test/FastGeluWithBiasFloat32_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.FastGelu |
| test/contrib_ops/fastgelu_op_test/FastGeluWithNullInput_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.FastGelu |
| test/contrib_ops/fastgelu_op_test/FastGeluWithoutBiasFloat32_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.FastGelu |
| test/contrib_ops/fused_conv_test/Conv2D_Bias_Relu_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.FusedConv |
| test/contrib_ops/fused_conv_test/Conv2D_HardSigmoid_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.FusedConv |
| test/contrib_ops/fused_conv_test/Conv2D_Relu_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.FusedConv |
| test/contrib_ops/fused_conv_test/Cpu_Conv2D_Bias_Z_Relu_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.FusedConv |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run0/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 6684672) |
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
| test/contrib_ops/inverse_test/four_by_four_batches_float_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.Inverse |
| test/contrib_ops/inverse_test/four_by_four_float_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.Inverse |
| test/contrib_ops/inverse_test/two_by_two_double_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.Inverse |
| test/contrib_ops/inverse_test/two_by_two_float16_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.Inverse |
| test/contrib_ops/inverse_test/two_by_two_float_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.Inverse |
| test/contrib_ops/layer_norm_op_test/BERTLayerNorm_NoBias_run0/model.onnx | 17 | Data/Data | ❌ | Out of tolerance (max ULP 2272) |
| test/contrib_ops/layer_norm_op_test/BERTLayerNorm_run0/model.onnx | 17 | Data/Data | ❌ | Out of tolerance (max ULP 5632) |
| test/contrib_ops/layer_norm_op_test/LayerNorm17_double_run0/model.onnx | 17 | Data/Data | ❌ | Out of tolerance (max ULP 160715044774) |
| test/contrib_ops/layer_norm_op_test/LayerNorm17_opset_run0/model.onnx | 17 | Data/Data | ❌ | Out of tolerance (max ULP 299) |
| test/contrib_ops/layer_norm_op_test/LayerNorm17_opset_run1/model.onnx | 17 | Data/Data | ❌ | Out of tolerance (max ULP 299) |
| test/contrib_ops/layer_norm_op_test/LayerNorm_Scale_Bias_4D_OuterInnerBroadcast_Axis3_run0/model.onnx | 17 | Data/Data | ❌ | LayerNormalization scale rank must match normalized rank |
| test/contrib_ops/layer_norm_op_test/LayerNorm_Scale_Bias_4D_OuterInnerBroadcast_run0/model.onnx | 17 | Data/Data | ❌ | LayerNormalization scale rank must match normalized rank |
| test/contrib_ops/layer_norm_op_test/LayerNorm_Scale_Bias_Axis2_run0/model.onnx | 17 | Data/Data | ❌ | Out of tolerance (max ULP 336) |
| test/contrib_ops/layer_norm_op_test/LayerNorm_Scale_Bias_Broadcast_Dim0_Fp16_run0/model.onnx | 7 | Data/Data | ❌ | LayerNormalization scale rank must match normalized rank |
| test/contrib_ops/layer_norm_op_test/LayerNorm_Scale_Bias_Broadcast_Dim0_run0/model.onnx | 7 | Data/Data | ❌ | LayerNormalization scale rank must match normalized rank |
| test/contrib_ops/layer_norm_op_test/LayerNorm_Scale_Bias_Broadcast_Dim1_Fp16_run0/model.onnx | 7 | Data/Data | ❌ | LayerNormalization scale rank must match normalized rank |
| test/contrib_ops/layer_norm_op_test/LayerNorm_Scale_Bias_Broadcast_Dim1_run0/model.onnx | 7 | Data/Data | ❌ | LayerNormalization scale rank must match normalized rank |
| test/contrib_ops/layer_norm_op_test/LayerNorm_Scale_Bias_Broadcast_Fp16_run0/model.onnx | 7 | Data/Data | ❌ | LayerNormalization scale rank must match normalized rank |
| test/contrib_ops/layer_norm_op_test/LayerNorm_Scale_Bias_Broadcast_Fp16_run1/model.onnx | 7 | Data/Data | ❌ | LayerNormalization scale rank must match normalized rank |
| test/contrib_ops/layer_norm_op_test/LayerNorm_Scale_Bias_NoBroadcast_Fp16_run0/model.onnx | 7 | Data/Data | ❌ | LayerNormalization scale rank must match normalized rank |
| test/contrib_ops/layer_norm_op_test/LayerNorm_Scale_Bias_NoBroadcast_run0/model.onnx | 7 | Data/Data | ❌ | LayerNormalization scale rank must match normalized rank |
| test/contrib_ops/layer_norm_op_test/LayerNorm_Scale_Bias_PerLastDim_run0/model.onnx | 17 | Data/Data | ❌ | Out of tolerance (max ULP 336) |
| test/contrib_ops/layer_norm_op_test/LayerNorm_Scale_Bias_Scalar_Axis2_run0/model.onnx | 17 | Data/Data | ❌ | LayerNormalization scale rank must match normalized rank |
| test/contrib_ops/layer_norm_op_test/LayerNorm_Scale_Bias_Scalar_run0/model.onnx | 17 | Data/Data | ❌ | LayerNormalization scale rank must match normalized rank |
| test/contrib_ops/layer_norm_op_test/LayerNorm_Scale_Bias_run0/model.onnx | 7 | Data/Data | ❌ | Out of tolerance (max ULP 10874) |
| test/contrib_ops/layer_norm_op_test/LayerNorm_Scale_Scalar_NoBias_Axis2_run0/model.onnx | 17 | Data/Data | ❌ | LayerNormalization scale rank must match normalized rank |
| test/contrib_ops/layer_norm_op_test/LayerNorm_Scale_Scalar_NoBias_run0/model.onnx | 17 | Data/Data | ❌ | LayerNormalization scale rank must match normalized rank |
| test/contrib_ops/layer_norm_op_test/LayerNorm_ValidScaleBias_Broadcast_run0/model.onnx | 7 | Data/Data | ❌ | LayerNormalization scale rank must match normalized rank |
| test/contrib_ops/layer_norm_op_test/LayerNorm_run0/model.onnx | 7 | Data/Data | ❌ | Out of tolerance (max ULP 299) |
| test/contrib_ops/math/matmul_sparse_test/TestCoo_run0/model.onnx |  | Data/Data | ❌ | Unsupported value type 'sparse_tensor_type' for 'A'. Hint: export the model with tensor inputs/outputs. |
| test/contrib_ops/math/matmul_sparse_test/TestCoo_run1/model.onnx |  | Data/Data | ❌ | Unsupported value type 'sparse_tensor_type' for 'A'. Hint: export the model with tensor inputs/outputs. |
| test/contrib_ops/math/matmul_sparse_test/TestCoo_run2/model.onnx |  | Data/Data | ❌ | Unsupported value type 'sparse_tensor_type' for 'A'. Hint: export the model with tensor inputs/outputs. |
| test/contrib_ops/math/matmul_sparse_test/TestCoo_run3/model.onnx |  | Data/Data | ❌ | Unsupported value type 'sparse_tensor_type' for 'A'. Hint: export the model with tensor inputs/outputs. |
| test/contrib_ops/math/matmul_sparse_test/TestCsr_run0/model.onnx |  | Data/Data | ❌ | Unsupported value type 'sparse_tensor_type' for 'A'. Hint: export the model with tensor inputs/outputs. |
| test/contrib_ops/math/matmul_sparse_test/TestCsr_run1/model.onnx |  | Data/Data | ❌ | Unsupported value type 'sparse_tensor_type' for 'A'. Hint: export the model with tensor inputs/outputs. |
| test/contrib_ops/math/matmul_sparse_test/TestCsr_run2/model.onnx |  | Data/Data | ❌ | Unsupported value type 'sparse_tensor_type' for 'A'. Hint: export the model with tensor inputs/outputs. |
| test/contrib_ops/math/matmul_sparse_test/TestCsr_run3/model.onnx |  | Data/Data | ❌ | Unsupported value type 'sparse_tensor_type' for 'A'. Hint: export the model with tensor inputs/outputs. |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy0_run11/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 135) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy0_run12/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy0_run13/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy0_run14/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 135) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy0_run17/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy0_run18/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy0_run2/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy0_run22/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy0_run23/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy0_run25/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 861) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy0_run26/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 7454) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy0_run27/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy0_run28/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy0_run29/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 7454) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy0_run3/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy0_run32/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy0_run33/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy0_run35/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 1730) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy0_run36/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 9141) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy0_run37/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy0_run38/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy0_run39/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 9141) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy0_run7/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy0_run8/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run100/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 861) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run101/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 7454) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run102/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run103/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run104/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 7454) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run107/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run108/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run112/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run113/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run115/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 1730) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run116/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 9141) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run117/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run118/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run119/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 9141) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run12/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run13/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run17/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run18/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run2/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run22/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run23/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run27/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run28/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run3/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run32/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run33/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run37/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run38/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run42/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run43/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run47/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run48/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run51/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 135) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run52/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run53/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run54/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 135) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run57/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run58/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run62/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run63/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run67/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run68/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run7/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run72/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run73/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run77/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run78/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run8/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run82/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run83/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run87/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run88/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run92/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run93/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run95/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 6080) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run96/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 9218) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run97/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run98/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy2_run99/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 9218) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run102/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run103/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run105/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 6080) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run106/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 9218) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run107/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run108/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run109/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 9218) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run110/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 861) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run111/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 7454) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run112/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run113/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run114/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 7454) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run117/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run118/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run12/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run122/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run123/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run127/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run128/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run13/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run130/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 1730) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run131/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 9141) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run132/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run133/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run134/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 9141) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run135/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 10976) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run136/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 11529) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run137/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run138/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run139/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 11529) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run17/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run18/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run2/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run22/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run23/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run27/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run28/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run3/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run32/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run33/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run37/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run38/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run42/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run43/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run47/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run48/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run51/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 135) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run52/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run53/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run54/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 135) |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run57/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run58/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run62/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run63/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run67/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run68/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run7/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run72/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run73/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run77/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run78/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run8/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run82/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run83/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run87/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run88/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run92/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run93/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run97/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float16_4b_Accuracy4_run98/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run102/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run103/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run107/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run108/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run112/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run113/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run117/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run118/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run12/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run13/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run17/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run18/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run2/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run22/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run23/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run27/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run28/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run3/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run32/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run33/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run37/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run38/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run42/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run43/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run47/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run48/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run52/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run53/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run57/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run58/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run62/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run63/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run67/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run68/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run7/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run72/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run73/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run77/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run78/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run8/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run82/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run83/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run87/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run88/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run92/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run93/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run97/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy0_run98/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy1_run12/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy1_run13/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy1_run17/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy1_run18/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy1_run2/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy1_run22/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy1_run23/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy1_run27/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy1_run28/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy1_run3/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy1_run32/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy1_run33/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy1_run37/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy1_run38/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy1_run7/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy1_run8/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run102/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run103/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run107/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run108/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run112/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run113/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run117/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run118/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run12/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run122/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run123/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run13/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run17/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run18/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run2/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run22/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run23/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run27/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run28/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run3/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run32/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run33/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run37/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run38/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run42/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run43/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run47/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run48/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run52/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run53/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run57/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run58/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run62/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run63/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run67/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run68/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run7/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run72/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run73/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run77/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run78/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run8/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run82/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run83/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run87/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run88/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run92/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run93/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run97/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/Float32_4b_Accuracy4_run98/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/LegacyShape_4b_run0/model.onnx |  | Data/Data | ❌ | Failed to build testbench (model.c:125:31: error: passing argument 3 of ‘node0_node1’ from incompatible pointer type [-Wincompatible-pointer-types]). |
| test/contrib_ops/matmul_4bits_test/LegacyShape_4b_run1/model.onnx |  | Data/Data | ❌ | Failed to build testbench (model.c:139:31: error: passing argument 3 of ‘node0_node1’ from incompatible pointer type [-Wincompatible-pointer-types]). |
| test/contrib_ops/matmul_4bits_test/LegacyShape_4b_run2/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/LegacyShape_4b_run3/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/LegacyShape_4b_run4/model.onnx |  | Data/Data | ❌ | Failed to build testbench (model.c:137:31: error: passing argument 3 of ‘node0_node1’ from incompatible pointer type [-Wincompatible-pointer-types]). |
| test/contrib_ops/matmul_4bits_test/LegacyShape_4b_run5/model.onnx |  | Data/Data | ❌ | Failed to build testbench (model.c:125:31: error: passing argument 3 of ‘node0_node1’ from incompatible pointer type [-Wincompatible-pointer-types]). |
| test/contrib_ops/matmul_4bits_test/LegacyShape_4b_run6/model.onnx |  | Data/Data | ❌ | Failed to build testbench (model.c:139:31: error: passing argument 3 of ‘node0_node1’ from incompatible pointer type [-Wincompatible-pointer-types]). |
| test/contrib_ops/matmul_4bits_test/LegacyShape_4b_run7/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/LegacyShape_4b_run8/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/LegacyShape_4b_run9/model.onnx |  | Data/Data | ❌ | Failed to build testbench (model.c:137:31: error: passing argument 3 of ‘node0_node1’ from incompatible pointer type [-Wincompatible-pointer-types]). |
| test/contrib_ops/matmul_integer16_test/MatMulInteger16_1_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MatMulInteger16 |
| test/contrib_ops/matmul_integer16_test/MatMulInteger16_2_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MatMulInteger16 |
| test/contrib_ops/matmul_integer16_test/MatMulInteger16_3_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MatMulInteger16 |
| test/contrib_ops/matmul_integer16_test/MatMulInteger16_Empty_input_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MatMulInteger16 |
| test/contrib_ops/matmul_integer_to_float_test/HasZeroPoint_HasBias_test_S8S8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/HasZeroPoint_HasBias_test_S8S8_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/HasZeroPoint_HasBias_test_S8S8_run2/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/HasZeroPoint_HasBias_test_S8S8_run3/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/HasZeroPoint_HasBias_test_U8S8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/HasZeroPoint_HasBias_test_U8S8_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/HasZeroPoint_HasBias_test_U8S8_run2/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/HasZeroPoint_HasBias_test_U8S8_run3/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/HasZeroPoint_HasBias_test_U8X8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/HasZeroPoint_HasBias_test_U8X8_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/HasZeroPoint_HasBias_test_U8X8_run2/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/HasZeroPoint_HasBias_test_U8X8_run3/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/HasZeroPoint_NoBias_test_S8S8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/HasZeroPoint_NoBias_test_S8S8_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/HasZeroPoint_NoBias_test_S8S8_run2/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/HasZeroPoint_NoBias_test_S8S8_run3/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/HasZeroPoint_NoBias_test_U8S8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/HasZeroPoint_NoBias_test_U8S8_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/HasZeroPoint_NoBias_test_U8S8_run2/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/HasZeroPoint_NoBias_test_U8S8_run3/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/HasZeroPoint_NoBias_test_U8U8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/HasZeroPoint_NoBias_test_U8U8_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/HasZeroPoint_NoBias_test_U8U8_run2/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/HasZeroPoint_NoBias_test_U8U8_run3/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/NoZeroPoint_HasBias_test_S8S8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/NoZeroPoint_HasBias_test_S8S8_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/NoZeroPoint_HasBias_test_S8S8_run2/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/NoZeroPoint_HasBias_test_S8S8_run3/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/NoZeroPoint_HasBias_test_U8S8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/NoZeroPoint_HasBias_test_U8S8_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/NoZeroPoint_HasBias_test_U8S8_run2/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/NoZeroPoint_HasBias_test_U8S8_run3/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/NoZeroPoint_HasBias_test_U8U8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/NoZeroPoint_HasBias_test_U8U8_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/NoZeroPoint_HasBias_test_U8U8_run2/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/NoZeroPoint_HasBias_test_U8U8_run3/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/NoZeroPoint_NoBias_test_S8S8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/NoZeroPoint_NoBias_test_S8S8_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/NoZeroPoint_NoBias_test_S8S8_run2/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/NoZeroPoint_NoBias_test_S8S8_run3/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/NoZeroPoint_NoBias_test_U8S8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/NoZeroPoint_NoBias_test_U8S8_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/NoZeroPoint_NoBias_test_U8S8_run2/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/NoZeroPoint_NoBias_test_U8S8_run3/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/NoZeroPoint_NoBias_test_U8U8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/NoZeroPoint_NoBias_test_U8U8_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/NoZeroPoint_NoBias_test_U8U8_run2/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/NoZeroPoint_NoBias_test_U8U8_run3/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MatMulIntegerToFloat |
| test/contrib_ops/maxpool_mask_test/MaxPoolWithMask_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MaxpoolWithMask |
| test/contrib_ops/moe_test/MoECpuTest_BasicSwiGLU_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MoE |
| test/contrib_ops/multihead_attention_op_test/CrossAttentionWithPast_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MultiHeadAttention |
| test/contrib_ops/multihead_attention_op_test/CrossAttentionWithPast_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MultiHeadAttention |
| test/contrib_ops/multihead_attention_op_test/CrossAttention_Batch1_HeadSize16_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MultiHeadAttention |
| test/contrib_ops/multihead_attention_op_test/CrossAttention_Batch1_HeadSize16_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MultiHeadAttention |
| test/contrib_ops/multihead_attention_op_test/CrossAttention_Batch1_HeadSize16_run2/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MultiHeadAttention |
| test/contrib_ops/multihead_attention_op_test/CrossAttention_Batch1_HeadSize16_run3/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MultiHeadAttention |
| test/contrib_ops/multihead_attention_op_test/CrossAttention_Batch1_HeadSize8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MultiHeadAttention |
| test/contrib_ops/multihead_attention_op_test/CrossAttention_Batch1_HeadSize8_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MultiHeadAttention |
| test/contrib_ops/multihead_attention_op_test/CrossAttention_Batch2_HeadSize16_8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MultiHeadAttention |
| test/contrib_ops/multihead_attention_op_test/CrossAttention_Batch2_HeadSize16_8_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MultiHeadAttention |
| test/contrib_ops/multihead_attention_op_test/CrossAttention_Batch2_HeadSize16_8_run2/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MultiHeadAttention |
| test/contrib_ops/multihead_attention_op_test/CrossAttention_Batch2_HeadSize16_8_run3/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MultiHeadAttention |
| test/contrib_ops/multihead_attention_op_test/CrossAttention_Batch2_HeadSize40_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MultiHeadAttention |
| test/contrib_ops/multihead_attention_op_test/CrossAttention_Batch2_HeadSize40_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MultiHeadAttention |
| test/contrib_ops/multihead_attention_op_test/CrossAttention_Batch2_HeadSize40_run2/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MultiHeadAttention |
| test/contrib_ops/multihead_attention_op_test/CrossAttention_Batch2_HeadSize40_run3/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MultiHeadAttention |
| test/contrib_ops/multihead_attention_op_test/CrossAttention_DiffSequenceLengths_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MultiHeadAttention |
| test/contrib_ops/multihead_attention_op_test/CrossAttention_DiffSequenceLengths_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MultiHeadAttention |
| test/contrib_ops/multihead_attention_op_test/CrossAttention_DiffSequenceLengths_run2/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MultiHeadAttention |
| test/contrib_ops/multihead_attention_op_test/CrossAttention_DiffSequenceLengths_run3/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MultiHeadAttention |
| test/contrib_ops/multihead_attention_op_test/CrossAttention_DiffSequenceLengths_run4/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MultiHeadAttention |
| test/contrib_ops/multihead_attention_op_test/CrossAttention_DiffSequenceLengths_run5/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MultiHeadAttention |
| test/contrib_ops/multihead_attention_op_test/SelfAttention_WithPastAndPresent_NoMask_NoRelPosBias_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MultiHeadAttention |
| test/contrib_ops/multihead_attention_op_test/SelfAttention_WithPastAndPresent_NoMask_NoRelPosBias_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MultiHeadAttention |
| test/contrib_ops/multihead_attention_op_test/SelfAttention_WithPastAndPresent_NoMask_NoRelPosBias_run2/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MultiHeadAttention |
| test/contrib_ops/multihead_attention_op_test/SelfAttention_WithPastAndPresent_NoMask_NoRelPosBias_run3/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MultiHeadAttention |
| test/contrib_ops/multihead_attention_op_test/SelfAttention_WithPastAndPresent_NoMask_NoRelPosBias_run4/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MultiHeadAttention |
| test/contrib_ops/multihead_attention_op_test/SelfAttention_WithPastAndPresent_NoMask_NoRelPosBias_run5/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.MultiHeadAttention |
| test/contrib_ops/murmur_hash3_test/UnsupportedInputType_run0/model.onnx |  | Data/Data | ❌ | MurmurHash3 does not support input dtype int8; supported: int32, int64, float, double, string |
| test/contrib_ops/ngram_repeat_block_op_test/NGramSize_3_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.NGramRepeatBlock |
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
| test/contrib_ops/qlinear_concat_test/ExpectFail_MixedConstDynamic_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearConcat |
| test/contrib_ops/qlinear_concat_test/ExpectFail_MixedConstDynamic_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearConcat |
| test/contrib_ops/qlinear_concat_test/ExpectFail_MixedConstDynamic_run2/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearConcat |
| test/contrib_ops/qlinear_concat_test/ExpectFail_MixedConstDynamic_run3/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearConcat |
| test/contrib_ops/qlinear_concat_test/ExpectFail_MixedConstDynamic_run4/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearConcat |
| test/contrib_ops/qlinear_concat_test/ExpectFail_MixedConstDynamic_run5/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearConcat |
| test/contrib_ops/qlinear_concat_test/ExpectFail_MixedConstDynamic_run6/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearConcat |
| test/contrib_ops/qlinear_concat_test/ExpectFail_MixedConstDynamic_run7/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearConcat |
| test/contrib_ops/qlinear_concat_test/ExpectFail_WrongScaleType_0_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearConcat |
| test/contrib_ops/qlinear_concat_test/ExpectFail_WrongScaleType_0_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearConcat |
| test/contrib_ops/qlinear_concat_test/ExpectFail_WrongScaleType_1_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearConcat |
| test/contrib_ops/qlinear_concat_test/ExpectFail_WrongScaleType_1_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearConcat |
| test/contrib_ops/qlinear_concat_test/ExpectFail_WrongTensorType_0_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearConcat |
| test/contrib_ops/qlinear_concat_test/ExpectFail_WrongTensorType_0_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearConcat |
| test/contrib_ops/qlinear_concat_test/ExpectFail_WrongTensorType_1_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearConcat |
| test/contrib_ops/qlinear_concat_test/ExpectFail_WrongTensorType_1_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearConcat |
| test/contrib_ops/qlinear_concat_test/ExpectFail_WrongZeroPointType_0_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearConcat |
| test/contrib_ops/qlinear_concat_test/ExpectFail_WrongZeroPointType_0_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearConcat |
| test/contrib_ops/qlinear_concat_test/ExpectFail_WrongZeroPointType_1_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearConcat |
| test/contrib_ops/qlinear_concat_test/ExpectFail_WrongZeroPointType_1_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearConcat |
| test/contrib_ops/qlinear_concat_test/Input3_ConstConstConst_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearConcat |
| test/contrib_ops/qlinear_concat_test/Input3_ConstConstConst_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearConcat |
| test/contrib_ops/qlinear_concat_test/Input3_DynamicDynamicDynamic_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearConcat |
| test/contrib_ops/qlinear_concat_test/Input3_DynamicDynamicDynamic_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearConcat |
| test/contrib_ops/qlinear_concat_test/Input3_MixedConstDynamic_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearConcat |
| test/contrib_ops/qlinear_concat_test/Input3_MixedConstDynamic_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearConcat |
| test/contrib_ops/qlinear_concat_test/Input3_MixedConstDynamic_run2/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearConcat |
| test/contrib_ops/qlinear_concat_test/Input3_MixedConstDynamic_run3/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearConcat |
| test/contrib_ops/qlinear_concat_test/Input3_MixedConstDynamic_run4/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearConcat |
| test/contrib_ops/qlinear_concat_test/Input3_MixedConstDynamic_run5/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearConcat |
| test/contrib_ops/qlinear_concat_test/Input3_MixedConstDynamic_run6/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearConcat |
| test/contrib_ops/qlinear_concat_test/Input3_MixedConstDynamic_run7/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearConcat |
| test/contrib_ops/qlinear_concat_test/InputOne_Const_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearConcat |
| test/contrib_ops/qlinear_concat_test/InputOne_Const_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearConcat |
| test/contrib_ops/qlinear_concat_test/InputOne_Dynamic_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearConcat |
| test/contrib_ops/qlinear_concat_test/InputOne_Dynamic_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearConcat |
| test/contrib_ops/qlinear_global_average_pool_test/Nchw_1x32x32x1_S8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearGlobalAveragePool |
| test/contrib_ops/qlinear_global_average_pool_test/Nchw_1x32x32x1_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearGlobalAveragePool |
| test/contrib_ops/qlinear_global_average_pool_test/Nchw_1x7x7x255_S8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearGlobalAveragePool |
| test/contrib_ops/qlinear_global_average_pool_test/Nchw_1x7x7x255_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearGlobalAveragePool |
| test/contrib_ops/qlinear_global_average_pool_test/Nchw_1x7x7x256_S8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearGlobalAveragePool |
| test/contrib_ops/qlinear_global_average_pool_test/Nchw_1x7x7x256_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearGlobalAveragePool |
| test/contrib_ops/qlinear_global_average_pool_test/Nchw_1x8x8x255_S8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearGlobalAveragePool |
| test/contrib_ops/qlinear_global_average_pool_test/Nchw_1x8x8x255_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearGlobalAveragePool |
| test/contrib_ops/qlinear_global_average_pool_test/Nchw_1x8x8x256_S8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearGlobalAveragePool |
| test/contrib_ops/qlinear_global_average_pool_test/Nchw_1x8x8x256_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearGlobalAveragePool |
| test/contrib_ops/qlinear_global_average_pool_test/Nchw_3x7x7x255_S8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearGlobalAveragePool |
| test/contrib_ops/qlinear_global_average_pool_test/Nchw_3x7x7x255_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearGlobalAveragePool |
| test/contrib_ops/qlinear_global_average_pool_test/Nchw_3x7x7x256_S8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearGlobalAveragePool |
| test/contrib_ops/qlinear_global_average_pool_test/Nchw_3x7x7x256_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearGlobalAveragePool |
| test/contrib_ops/qlinear_global_average_pool_test/Nchw_3x8x8x255_S8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearGlobalAveragePool |
| test/contrib_ops/qlinear_global_average_pool_test/Nchw_3x8x8x255_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearGlobalAveragePool |
| test/contrib_ops/qlinear_global_average_pool_test/Nchw_3x8x8x256_S8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearGlobalAveragePool |
| test/contrib_ops/qlinear_global_average_pool_test/Nchw_3x8x8x256_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearGlobalAveragePool |
| test/contrib_ops/qlinear_global_average_pool_test/Nhwc_1x1x32x32_S8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearGlobalAveragePool |
| test/contrib_ops/qlinear_global_average_pool_test/Nhwc_1x1x32x32_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearGlobalAveragePool |
| test/contrib_ops/qlinear_global_average_pool_test/Nhwc_1x255x7x7_S8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearGlobalAveragePool |
| test/contrib_ops/qlinear_global_average_pool_test/Nhwc_1x255x7x7_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearGlobalAveragePool |
| test/contrib_ops/qlinear_global_average_pool_test/Nhwc_1x255x8x8_S8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearGlobalAveragePool |
| test/contrib_ops/qlinear_global_average_pool_test/Nhwc_1x255x8x8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearGlobalAveragePool |
| test/contrib_ops/qlinear_global_average_pool_test/Nhwc_1x256x7x7_S8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearGlobalAveragePool |
| test/contrib_ops/qlinear_global_average_pool_test/Nhwc_1x256x7x7_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearGlobalAveragePool |
| test/contrib_ops/qlinear_global_average_pool_test/Nhwc_1x256x8x8_S8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearGlobalAveragePool |
| test/contrib_ops/qlinear_global_average_pool_test/Nhwc_1x256x8x8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearGlobalAveragePool |
| test/contrib_ops/qlinear_global_average_pool_test/Nhwc_3x255x7x7_S8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearGlobalAveragePool |
| test/contrib_ops/qlinear_global_average_pool_test/Nhwc_3x255x7x7_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearGlobalAveragePool |
| test/contrib_ops/qlinear_global_average_pool_test/Nhwc_3x255x8x8_S8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearGlobalAveragePool |
| test/contrib_ops/qlinear_global_average_pool_test/Nhwc_3x255x8x8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearGlobalAveragePool |
| test/contrib_ops/qlinear_global_average_pool_test/Nhwc_3x256x7x7_S8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearGlobalAveragePool |
| test/contrib_ops/qlinear_global_average_pool_test/Nhwc_3x256x7x7_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearGlobalAveragePool |
| test/contrib_ops/qlinear_global_average_pool_test/Nhwc_3x256x8x8_S8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearGlobalAveragePool |
| test/contrib_ops/qlinear_global_average_pool_test/Nhwc_3x256x8x8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearGlobalAveragePool |
| test/contrib_ops/qlinear_lookup_table_test/QLinearLeakyRelu_Int8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearLeakyRelu |
| test/contrib_ops/qlinear_lookup_table_test/QLinearLeakyRelu_UInt8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearLeakyRelu |
| test/contrib_ops/qlinear_lookup_table_test/QLinearSigmoid_Int8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearSigmoid |
| test/contrib_ops/qlinear_lookup_table_test/QLinearSigmoid_UInt8_0_Y_ZP_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearSigmoid |
| test/contrib_ops/qlinear_lookup_table_test/QLinearSigmoid_UInt8_0_Y_ZP_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearSigmoid |
| test/contrib_ops/qlinear_lookup_table_test/QLinearSigmoid_UInt8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QLinearSigmoid |
| test/contrib_ops/qlinear_lookup_table_test/QLinearSoftmax_Int8_v12_run1/model.onnx |  | Data/Data | ❌ | QLinearSoftmax axis -2 is out of bounds for shape () |
| test/contrib_ops/qlinear_lookup_table_test/QLinearSoftmax_Int8_v13_run0/model.onnx |  | Data/Data | ❌ | Out of tolerance (max abs diff 1) |
| test/contrib_ops/qlinear_lookup_table_test/QLinearSoftmax_Int8_v13_run1/model.onnx |  | Data/Data | ❌ | QLinearSoftmax axis -2 is out of bounds for shape () |
| test/contrib_ops/qlinear_lookup_table_test/QLinearSoftmax_UInt8_v12_run1/model.onnx |  | Data/Data | ❌ | QLinearSoftmax axis -2 is out of bounds for shape () |
| test/contrib_ops/qlinear_lookup_table_test/QLinearSoftmax_UInt8_v13_run0/model.onnx |  | Data/Data | ❌ | Out of tolerance (max abs diff 1) |
| test/contrib_ops/qlinear_lookup_table_test/QLinearSoftmax_UInt8_v13_run1/model.onnx |  | Data/Data | ❌ | QLinearSoftmax axis -2 is out of bounds for shape () |
| test/contrib_ops/qlinear_pool_test/AveragePool1D_ExcludePadPixel_S8_run0/model.onnx |  | Data/Data | ❌ | AveragePool has unsupported attributes |
| test/contrib_ops/qlinear_pool_test/AveragePool1D_ExcludePadPixel_S8_run1/model.onnx |  | Data/Data | ❌ | AveragePool has unsupported attributes |
| test/contrib_ops/qlinear_pool_test/AveragePool1D_ExcludePadPixel_S8_run2/model.onnx |  | Data/Data | ❌ | AveragePool has unsupported attributes |
| test/contrib_ops/qlinear_pool_test/AveragePool1D_ExcludePadPixel_nhwc_S8_run0/model.onnx |  | Data/Data | ❌ | AveragePool has unsupported attributes |
| test/contrib_ops/qlinear_pool_test/AveragePool1D_ExcludePadPixel_nhwc_run0/model.onnx |  | Data/Data | ❌ | AveragePool has unsupported attributes |
| test/contrib_ops/qlinear_pool_test/AveragePool1D_ExcludePadPixel_run0/model.onnx |  | Data/Data | ❌ | AveragePool has unsupported attributes |
| test/contrib_ops/qlinear_pool_test/AveragePool1D_ExcludePadPixel_run1/model.onnx |  | Data/Data | ❌ | AveragePool has unsupported attributes |
| test/contrib_ops/qlinear_pool_test/AveragePool1D_ExcludePadPixel_run2/model.onnx |  | Data/Data | ❌ | AveragePool has unsupported attributes |
| test/contrib_ops/qlinear_pool_test/AveragePool1D_IncludePadPixel_S8_run0/model.onnx |  | Data/Data | ❌ | AveragePool has unsupported attributes |
| test/contrib_ops/qlinear_pool_test/AveragePool1D_IncludePadPixel_S8_run1/model.onnx |  | Data/Data | ❌ | AveragePool has unsupported attributes |
| test/contrib_ops/qlinear_pool_test/AveragePool1D_IncludePadPixel_S8_run2/model.onnx |  | Data/Data | ❌ | AveragePool has unsupported attributes |
| test/contrib_ops/qlinear_pool_test/AveragePool1D_IncludePadPixel_nhwc_S8_run0/model.onnx |  | Data/Data | ❌ | AveragePool has unsupported attributes |
| test/contrib_ops/qlinear_pool_test/AveragePool1D_IncludePadPixel_nhwc_run0/model.onnx |  | Data/Data | ❌ | AveragePool has unsupported attributes |
| test/contrib_ops/qlinear_pool_test/AveragePool1D_IncludePadPixel_run0/model.onnx |  | Data/Data | ❌ | AveragePool has unsupported attributes |
| test/contrib_ops/qlinear_pool_test/AveragePool1D_IncludePadPixel_run1/model.onnx |  | Data/Data | ❌ | AveragePool has unsupported attributes |
| test/contrib_ops/qlinear_pool_test/AveragePool1D_IncludePadPixel_run2/model.onnx |  | Data/Data | ❌ | AveragePool has unsupported attributes |
| test/contrib_ops/qlinear_pool_test/AveragePool2D_BigImage_S8_run0/model.onnx |  | Data/Data | ❌ | AveragePool has unsupported attributes |
| test/contrib_ops/qlinear_pool_test/AveragePool2D_BigImage_S8_run1/model.onnx |  | Data/Data | ❌ | AveragePool has unsupported attributes |
| test/contrib_ops/qlinear_pool_test/AveragePool2D_BigImage_S8_run2/model.onnx |  | Data/Data | ❌ | AveragePool has unsupported attributes |
| test/contrib_ops/qlinear_pool_test/AveragePool2D_BigImage_nhwc_S8_run0/model.onnx |  | Data/Data | ❌ | AveragePool has unsupported attributes |
| test/contrib_ops/qlinear_pool_test/AveragePool2D_BigImage_nhwc_run0/model.onnx |  | Data/Data | ❌ | AveragePool has unsupported attributes |
| test/contrib_ops/qlinear_pool_test/AveragePool2D_BigImage_run0/model.onnx |  | Data/Data | ❌ | AveragePool has unsupported attributes |
| test/contrib_ops/qlinear_pool_test/AveragePool2D_BigImage_run1/model.onnx |  | Data/Data | ❌ | AveragePool has unsupported attributes |
| test/contrib_ops/qlinear_pool_test/AveragePool2D_BigImage_run2/model.onnx |  | Data/Data | ❌ | AveragePool has unsupported attributes |
| test/contrib_ops/qlinear_pool_test/AveragePool2D_ExcludePadPixel_S8_run0/model.onnx |  | Data/Data | ❌ | AveragePool has unsupported attributes |
| test/contrib_ops/qlinear_pool_test/AveragePool2D_ExcludePadPixel_S8_run1/model.onnx |  | Data/Data | ❌ | AveragePool has unsupported attributes |
| test/contrib_ops/qlinear_pool_test/AveragePool2D_ExcludePadPixel_S8_run2/model.onnx |  | Data/Data | ❌ | AveragePool has unsupported attributes |
| test/contrib_ops/qlinear_pool_test/AveragePool2D_ExcludePadPixel_nhwc_S8_run0/model.onnx |  | Data/Data | ❌ | AveragePool has unsupported attributes |
| test/contrib_ops/qlinear_pool_test/AveragePool2D_ExcludePadPixel_nhwc_run0/model.onnx |  | Data/Data | ❌ | AveragePool has unsupported attributes |
| test/contrib_ops/qlinear_pool_test/AveragePool2D_ExcludePadPixel_run0/model.onnx |  | Data/Data | ❌ | AveragePool has unsupported attributes |
| test/contrib_ops/qlinear_pool_test/AveragePool2D_ExcludePadPixel_run1/model.onnx |  | Data/Data | ❌ | AveragePool has unsupported attributes |
| test/contrib_ops/qlinear_pool_test/AveragePool2D_ExcludePadPixel_run2/model.onnx |  | Data/Data | ❌ | AveragePool has unsupported attributes |
| test/contrib_ops/qlinear_pool_test/AveragePool2D_Global_S8_run0/model.onnx |  | Data/Data | ❌ | AveragePool has unsupported attributes |
| test/contrib_ops/qlinear_pool_test/AveragePool2D_Global_S8_run1/model.onnx |  | Data/Data | ❌ | AveragePool has unsupported attributes |
| test/contrib_ops/qlinear_pool_test/AveragePool2D_Global_S8_run2/model.onnx |  | Data/Data | ❌ | AveragePool has unsupported attributes |
| test/contrib_ops/qlinear_pool_test/AveragePool2D_Global_nhwc_S8_run0/model.onnx |  | Data/Data | ❌ | AveragePool has unsupported attributes |
| test/contrib_ops/qlinear_pool_test/AveragePool2D_Global_nhwc_run0/model.onnx |  | Data/Data | ❌ | AveragePool has unsupported attributes |
| test/contrib_ops/qlinear_pool_test/AveragePool2D_Global_run0/model.onnx |  | Data/Data | ❌ | AveragePool has unsupported attributes |
| test/contrib_ops/qlinear_pool_test/AveragePool2D_Global_run1/model.onnx |  | Data/Data | ❌ | AveragePool has unsupported attributes |
| test/contrib_ops/qlinear_pool_test/AveragePool2D_Global_run2/model.onnx |  | Data/Data | ❌ | AveragePool has unsupported attributes |
| test/contrib_ops/qlinear_pool_test/AveragePool2D_IncludePadPixel_S8_run0/model.onnx |  | Data/Data | ❌ | AveragePool has unsupported attributes |
| test/contrib_ops/qlinear_pool_test/AveragePool2D_IncludePadPixel_S8_run1/model.onnx |  | Data/Data | ❌ | AveragePool has unsupported attributes |
| test/contrib_ops/qlinear_pool_test/AveragePool2D_IncludePadPixel_S8_run2/model.onnx |  | Data/Data | ❌ | AveragePool has unsupported attributes |
| test/contrib_ops/qlinear_pool_test/AveragePool2D_IncludePadPixel_nhwc_S8_run0/model.onnx |  | Data/Data | ❌ | AveragePool has unsupported attributes |
| test/contrib_ops/qlinear_pool_test/AveragePool2D_IncludePadPixel_nhwc_run0/model.onnx |  | Data/Data | ❌ | AveragePool has unsupported attributes |
| test/contrib_ops/qlinear_pool_test/AveragePool2D_IncludePadPixel_run0/model.onnx |  | Data/Data | ❌ | AveragePool has unsupported attributes |
| test/contrib_ops/qlinear_pool_test/AveragePool2D_IncludePadPixel_run1/model.onnx |  | Data/Data | ❌ | AveragePool has unsupported attributes |
| test/contrib_ops/qlinear_pool_test/AveragePool2D_IncludePadPixel_run2/model.onnx |  | Data/Data | ❌ | AveragePool has unsupported attributes |
| test/contrib_ops/qlinear_pool_test/AveragePool2D_MultiChannel_S8_run0/model.onnx |  | Data/Data | ❌ | AveragePool has unsupported attributes |
| test/contrib_ops/qlinear_pool_test/AveragePool2D_MultiChannel_S8_run1/model.onnx |  | Data/Data | ❌ | AveragePool has unsupported attributes |
| test/contrib_ops/qlinear_pool_test/AveragePool2D_MultiChannel_S8_run2/model.onnx |  | Data/Data | ❌ | AveragePool has unsupported attributes |
| test/contrib_ops/qlinear_pool_test/AveragePool2D_MultiChannel_nhwc_S8_run0/model.onnx |  | Data/Data | ❌ | AveragePool has unsupported attributes |
| test/contrib_ops/qlinear_pool_test/AveragePool2D_MultiChannel_nhwc_run0/model.onnx |  | Data/Data | ❌ | AveragePool has unsupported attributes |
| test/contrib_ops/qlinear_pool_test/AveragePool2D_MultiChannel_run0/model.onnx |  | Data/Data | ❌ | AveragePool has unsupported attributes |
| test/contrib_ops/qlinear_pool_test/AveragePool2D_MultiChannel_run1/model.onnx |  | Data/Data | ❌ | AveragePool has unsupported attributes |
| test/contrib_ops/qlinear_pool_test/AveragePool2D_MultiChannel_run2/model.onnx |  | Data/Data | ❌ | AveragePool has unsupported attributes |
| test/contrib_ops/qlinear_pool_test/AveragePool3D_ExcludePadPixel_S8_run0/model.onnx |  | Data/Data | ❌ | AveragePool has unsupported attributes |
| test/contrib_ops/qlinear_pool_test/AveragePool3D_ExcludePadPixel_S8_run1/model.onnx |  | Data/Data | ❌ | AveragePool has unsupported attributes |
| test/contrib_ops/qlinear_pool_test/AveragePool3D_ExcludePadPixel_S8_run2/model.onnx |  | Data/Data | ❌ | AveragePool has unsupported attributes |
| test/contrib_ops/qlinear_pool_test/AveragePool3D_ExcludePadPixel_nhwc_S8_run0/model.onnx |  | Data/Data | ❌ | AveragePool has unsupported attributes |
| test/contrib_ops/qlinear_pool_test/AveragePool3D_ExcludePadPixel_nhwc_run0/model.onnx |  | Data/Data | ❌ | AveragePool has unsupported attributes |
| test/contrib_ops/qlinear_pool_test/AveragePool3D_ExcludePadPixel_run0/model.onnx |  | Data/Data | ❌ | AveragePool has unsupported attributes |
| test/contrib_ops/qlinear_pool_test/AveragePool3D_ExcludePadPixel_run1/model.onnx |  | Data/Data | ❌ | AveragePool has unsupported attributes |
| test/contrib_ops/qlinear_pool_test/AveragePool3D_ExcludePadPixel_run2/model.onnx |  | Data/Data | ❌ | AveragePool has unsupported attributes |
| test/contrib_ops/qlinear_pool_test/AveragePool3D_IncludePadPixel_S8_run0/model.onnx |  | Data/Data | ❌ | AveragePool has unsupported attributes |
| test/contrib_ops/qlinear_pool_test/AveragePool3D_IncludePadPixel_S8_run1/model.onnx |  | Data/Data | ❌ | AveragePool has unsupported attributes |
| test/contrib_ops/qlinear_pool_test/AveragePool3D_IncludePadPixel_S8_run2/model.onnx |  | Data/Data | ❌ | AveragePool has unsupported attributes |
| test/contrib_ops/qlinear_pool_test/AveragePool3D_IncludePadPixel_nhwc_S8_run0/model.onnx |  | Data/Data | ❌ | AveragePool has unsupported attributes |
| test/contrib_ops/qlinear_pool_test/AveragePool3D_IncludePadPixel_nhwc_run0/model.onnx |  | Data/Data | ❌ | AveragePool has unsupported attributes |
| test/contrib_ops/qlinear_pool_test/AveragePool3D_IncludePadPixel_run0/model.onnx |  | Data/Data | ❌ | AveragePool has unsupported attributes |
| test/contrib_ops/qlinear_pool_test/AveragePool3D_IncludePadPixel_run1/model.onnx |  | Data/Data | ❌ | AveragePool has unsupported attributes |
| test/contrib_ops/qlinear_pool_test/AveragePool3D_IncludePadPixel_run2/model.onnx |  | Data/Data | ❌ | AveragePool has unsupported attributes |
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
| test/contrib_ops/quantize_attention_op_test/QAttentionBatch1_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QAttention |
| test/contrib_ops/quantize_attention_op_test/QAttentionBatch1_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QAttention |
| test/contrib_ops/quantize_attention_op_test/QAttentionBatch2_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QAttention |
| test/contrib_ops/quantize_attention_op_test/QAttentionBatch2_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QAttention |
| test/contrib_ops/quantize_attention_op_test/QAttentionMaskExceedSequence_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QAttention |
| test/contrib_ops/quantize_attention_op_test/QAttentionMaskExceedSequence_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QAttention |
| test/contrib_ops/quantize_attention_op_test/QAttentionMaskPartialSequence_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QAttention |
| test/contrib_ops/quantize_attention_op_test/QAttentionMaskPartialSequence_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QAttention |
| test/contrib_ops/quantize_attention_op_test/QAttentionNoMaskIndex_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QAttention |
| test/contrib_ops/quantize_attention_op_test/QAttentionNoMaskIndex_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QAttention |
| test/contrib_ops/quantize_attention_op_test/QAttentionPastState_u8s8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QAttention |
| test/contrib_ops/quantize_attention_op_test/QAttentionPastState_u8s8_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QAttention |
| test/contrib_ops/quantize_attention_op_test/QAttentionPastState_u8s8_run2/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QAttention |
| test/contrib_ops/quantize_attention_op_test/QAttentionPastState_u8s8_run3/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QAttention |
| test/contrib_ops/quantize_attention_op_test/QAttentionPastState_u8u8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QAttention |
| test/contrib_ops/quantize_attention_op_test/QAttentionPastState_u8u8_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QAttention |
| test/contrib_ops/quantize_attention_op_test/QAttentionPastState_u8u8_run2/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QAttention |
| test/contrib_ops/quantize_attention_op_test/QAttentionPastState_u8u8_run3/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QAttention |
| test/contrib_ops/quantize_attention_op_test/QAttentionPrunedModel_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QAttention |
| test/contrib_ops/quantize_attention_op_test/QAttentionPrunedModel_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QAttention |
| test/contrib_ops/quantize_attention_op_test/QAttentionUnidirectional_U8S8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QAttention |
| test/contrib_ops/quantize_attention_op_test/QAttentionUnidirectional_U8U8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QAttention |
| test/contrib_ops/quantize_attention_op_test/SharedPrepackedWeights_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QAttention |
| test/contrib_ops/quantize_attention_op_test/SharedPrepackedWeights_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.QAttention |
| test/contrib_ops/rotary_embedding_op_test/RotaryEmbedding_CustomRotaryDim_SmallData_Phi_Packed_Batching_run0/model.onnx |  | Data/Data | ❌ | RotaryEmbedding inputs must share the same dtype |
| test/contrib_ops/rotary_embedding_op_test/RotaryEmbedding_CustomRotaryDim_SmallData_Phi_Packed_Batching_run1/model.onnx |  | Data/Data | ❌ | RotaryEmbedding inputs must share the same dtype |
| test/contrib_ops/rotary_embedding_op_test/RotaryEmbedding_CustomRotaryDim_SmallData_Phi_run0/model.onnx |  | Data/Data | ❌ | RotaryEmbedding inputs must share the same dtype |
| test/contrib_ops/rotary_embedding_op_test/RotaryEmbedding_CustomRotaryDim_SmallData_Phi_run1/model.onnx |  | Data/Data | ❌ | RotaryEmbedding inputs must share the same dtype |
| test/contrib_ops/rotary_embedding_op_test/RotaryEmbedding_Interleaved_LargeData_LlamaMSFT_run0/model.onnx |  | Data/Data | ❌ | RotaryEmbedding inputs must share the same dtype |
| test/contrib_ops/rotary_embedding_op_test/RotaryEmbedding_Interleaved_LargeData_LlamaMSFT_run1/model.onnx |  | Data/Data | ❌ | RotaryEmbedding inputs must share the same dtype |
| test/contrib_ops/rotary_embedding_op_test/RotaryEmbedding_Interleaved_SmallData_LlamaMSFT_run0/model.onnx |  | Data/Data | ❌ | RotaryEmbedding inputs must share the same dtype |
| test/contrib_ops/rotary_embedding_op_test/RotaryEmbedding_Interleaved_SmallData_LlamaMSFT_run1/model.onnx |  | Data/Data | ❌ | RotaryEmbedding inputs must share the same dtype |
| test/contrib_ops/rotary_embedding_op_test/RotaryEmbedding_NotInterleaved_LargeData_LlamaMSFT_run0/model.onnx |  | Data/Data | ❌ | RotaryEmbedding inputs must share the same dtype |
| test/contrib_ops/rotary_embedding_op_test/RotaryEmbedding_NotInterleaved_LargeData_LlamaMSFT_run1/model.onnx |  | Data/Data | ❌ | RotaryEmbedding inputs must share the same dtype |
| test/contrib_ops/rotary_embedding_op_test/RotaryEmbedding_NotInterleaved_SmallData_LlamaMSFT_run0/model.onnx |  | Data/Data | ❌ | RotaryEmbedding inputs must share the same dtype |
| test/contrib_ops/rotary_embedding_op_test/RotaryEmbedding_NotInterleaved_SmallData_LlamaMSFT_run1/model.onnx |  | Data/Data | ❌ | RotaryEmbedding inputs must share the same dtype |
| test/contrib_ops/sample_op_test/SampleOpFloat_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.SampleOp |
| test/contrib_ops/skiplayernorm_op_test/SkipLayerNormBatch1_NoBeta_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.SkipLayerNormalization |
| test/contrib_ops/skiplayernorm_op_test/SkipLayerNormBatch1_NoBeta_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.SkipLayerNormalization |
| test/contrib_ops/skiplayernorm_op_test/SkipLayerNormBatch1_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.SkipLayerNormalization |
| test/contrib_ops/skiplayernorm_op_test/SkipLayerNormBatch1_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.SkipLayerNormalization |
| test/contrib_ops/skiplayernorm_op_test/SkipLayerNormBatch2_Bias_ProducingOptionalOutput_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.SkipLayerNormalization |
| test/contrib_ops/skiplayernorm_op_test/SkipLayerNormBatch2_Bias_ProducingOptionalOutput_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.SkipLayerNormalization |
| test/contrib_ops/skiplayernorm_op_test/SkipLayerNormBatch2_Bias_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.SkipLayerNormalization |
| test/contrib_ops/skiplayernorm_op_test/SkipLayerNormBatch2_Bias_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.SkipLayerNormalization |
| test/contrib_ops/skiplayernorm_op_test/SkipLayerNormBatch2_Skip_Broadcast_Batch_Size_1_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.SkipLayerNormalization |
| test/contrib_ops/skiplayernorm_op_test/SkipLayerNormBatch2_Skip_Broadcast_No_Batch_Size_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.SkipLayerNormalization |
| test/contrib_ops/skiplayernorm_op_test/SkipLayerNormBatch2_TokenCount_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.SkipLayerNormalization |
| test/contrib_ops/skiplayernorm_op_test/SkipLayerNormBatch2_TokenCount_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.SkipLayerNormalization |
| test/contrib_ops/skiplayernorm_op_test/SkipLayerNormBatch2_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.SkipLayerNormalization |
| test/contrib_ops/skiplayernorm_op_test/SkipLayerNormBatch2_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.SkipLayerNormalization |
| test/contrib_ops/skiplayernorm_op_test/SkipLayerNormNullInput_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.SkipLayerNormalization |
| test/contrib_ops/skiplayernorm_op_test/SkipLayerNormNullInput_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.SkipLayerNormalization |
| test/contrib_ops/skiplayernorm_op_test/SkipLayerNormPrePack_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.SkipLayerNormalization |
| test/contrib_ops/tensor_op_test/CropBorderAndScale_run0/model.onnx | 7 | Data/Data | ❌ | Unsupported op ai.onnx.Crop |
| test/contrib_ops/tensor_op_test/CropBorderOnly_run0/model.onnx | 7 | Data/Data | ❌ | Unsupported op ai.onnx.Crop |
| test/contrib_ops/tensor_op_test/ImageScalerTest_run0/model.onnx | 7 | Data/Data | ❌ | Unsupported op ai.onnx.ImageScaler |
| test/contrib_ops/tensor_op_test/LastDim_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.UnfoldTensor |
| test/contrib_ops/tensor_op_test/MeanVarianceNormalizationCPUTest_Version1_TO_8_run0/model.onnx | 7 | Data/Data | ❌ | Out of tolerance (max ULP 7137119) |
| test/contrib_ops/tensor_op_test/MeanVarianceNormalizationCPUTest_Version1_TO_8_run1/model.onnx | 7 | Data/Data | ❌ | Out of tolerance (max ULP 21399263) |
| test/contrib_ops/tensor_op_test/MeanVarianceNormalizationCPUTest_Version1_TO_8_run2/model.onnx | 7 | Data/Data | ❌ | Out of tolerance (max ULP 15806857) |
| test/contrib_ops/tensor_op_test/NormalDim_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.UnfoldTensor |
| test/contrib_ops/tokenizer_test/TokenizerCharLevel_CyrillicCharsWithMarkersC_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.Tokenizer |
| test/contrib_ops/tokenizer_test/TokenizerCharLevel_EmptyOutputC_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.Tokenizer |
| test/contrib_ops/tokenizer_test/TokenizerCharLevel_EmptyOutputNC_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.Tokenizer |
| test/contrib_ops/tokenizer_test/TokenizerCharLevel_LatinCharsNoMarkersC_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.Tokenizer |
| test/contrib_ops/tokenizer_test/TokenizerCharLevel_LatinCharsNoMarkersNC_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.Tokenizer |
| test/contrib_ops/tokenizer_test/TokenizerCharLevel_LatinCharsWithMarkersC_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.Tokenizer |
| test/contrib_ops/tokenizer_test/TokenizerCharLevel_LatinCharsWithMarkersNC_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.Tokenizer |
| test/contrib_ops/tokenizer_test/TokenizerCharLevel_MixedCharsWithMarkersC_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.Tokenizer |
| test/contrib_ops/tokenizer_test/TokenizerExpression_Grouping_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.Tokenizer |
| test/contrib_ops/tokenizer_test/TokenizerExpression_RegChar_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.Tokenizer |
| test/contrib_ops/tokenizer_test/TokenizerExpression_RegDot_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.Tokenizer |
| test/contrib_ops/tokenizer_test/TokenizerExpression_RegEx_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.Tokenizer |
| test/contrib_ops/tokenizer_test/TokenizerExpression_RegRep_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.Tokenizer |
| test/contrib_ops/tokenizer_test/TokenizerWithSeparators_MixCharCommonPrefixC_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.Tokenizer |
| test/contrib_ops/tokenizer_test/TokenizerWithSeparators_MixCharsNoMarkersSeparatorsOverlapLongFirstC_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.Tokenizer |
| test/contrib_ops/tokenizer_test/TokenizerWithSeparators_MixCharsNoMarkersSeparatorsOverlapLongFirstRepeatedShortC_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.Tokenizer |
| test/contrib_ops/tokenizer_test/TokenizerWithSeparators_MixCharsNoMarkersSeparatorsOverlapShortFirstC_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.Tokenizer |
| test/contrib_ops/tokenizer_test/TokenizerWithSeparators_MixCharsNoMarkersSeparatorsOverlapingMatchC_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.Tokenizer |
| test/contrib_ops/tokenizer_test/TokenizerWithSeparators_MixCharsWithMarkersC_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.Tokenizer |
| test/contrib_ops/tokenizer_test/TokenizerWithSeparators_MixCharsWithMarkersCompleteMatchEmptyOutputC_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.Tokenizer |
| test/contrib_ops/tokenizer_test/TokenizerWithSeparators_MixCharsWithMarkersEmptyInputEmptyOutputC_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.Tokenizer |
| test/contrib_ops/tokenizer_test/TokenizerWithSeparators_MixCharsWithMarkersEmptyInputEmptyOutputNC_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.Tokenizer |
| test/contrib_ops/tokenizer_test/TokenizerWithSeparators_MixCharsWithMarkersEndMatchAtLeast4CharsC_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.Tokenizer |
| test/contrib_ops/tokenizer_test/TokenizerWithSeparators_MixCharsWithMarkersEndMatchC_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.Tokenizer |
| test/contrib_ops/tokenizer_test/TokenizerWithSeparators_MixCharsWithMarkersStartMatchC_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.Tokenizer |
| test/contrib_ops/tokenizer_test/Tokenizer_EmptyInput_run0/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.Tokenizer |
| test/contrib_ops/tokenizer_test/Tokenizer_EmptyInput_run1/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.Tokenizer |
| test/contrib_ops/tokenizer_test/Tokenizer_EmptyInput_run2/model.onnx |  | Data/Data | ❌ | Unsupported op com.microsoft.Tokenizer |
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

<!-- AUTO-GENERATED FILE. DO NOT EDIT. -->
<!-- Regenerate with: UPDATE_REFS=1 pytest -q tests/test_official_onnx_files_docs.py::test_official_onnx_file_support_doc -->

# ONNX verification errors

Aggregates non-success verification outcomes.

| Error message | Count | Opset versions |
| --- | --- | --- |
| Unsupported op MatMulBnb4 | 768 |  |
| Unsupported op QGemm | 624 |  |
| Unsupported op NhwcMaxPool | 562 |  |
| MatMulNBits g_idx (input 4) is not supported | 238 |  |
| Unsupported op GatherBlockQuantized | 222 |  |
| Unsupported op FusedMatMul | 192 |  |
| Out of tolerance | 98 | 7, 17 |
| AveragePool has unsupported attributes | 72 |  |
| Unsupported op Attention | 65 |  |
| Unsupported op MatMulIntegerToFloat | 48 |  |
| Unsupported op DynamicQuantizeMatMul | 39 |  |
| Unsupported op QLinearConcat | 36 |  |
| Unsupported op QLinearGlobalAveragePool | 36 |  |
| Unsupported op MultiHeadAttention | 28 |  |
| Unsupported op Tokenizer | 28 |  |
| Unsupported op QAttention | 24 |  |
| Unsupported op MurmurHash3 | 17 |  |
| Unsupported op SkipLayerNormalization | 17 |  |
| LayerNormalization scale rank must match normalized rank | 15 | 7, 17 |
| RotaryEmbedding inputs must share the same dtype | 12 |  |
| Unsupported op DynamicSlice | 12 | 1 |
| Unsupported op QEmbedLayerNormalization | 12 |  |
| Unsupported op ExpandDims | 11 |  |
| Unsupported op CropAndResize | 10 |  |
| Unsupported op EmbedLayerNormalization | 10 |  |
| Unsupported op QLinearWhere | 10 |  |
| Unsupported value type '*' for '*'. Hint: export the model with tensor inputs/outputs. | 8 |  |
| GridSample mode '*' is not supported | 7 |  |
| Unsupported op AttnLSTM | 6 |  |
| Unsupported op BiasGelu | 6 |  |
| Unsupported op WordConvEmbedding | 6 |  |
| Unsupported op Inverse | 5 |  |
| QLinearSoftmax axis -2 is out of bounds for shape () | 4 |  |
| Unique must have 1 input and 4 outputs | 4 |  |
| Unsupported op Crop | 4 | 1, 7 |
| Unsupported op FusedConv | 4 |  |
| Unsupported op MatMulInteger16 | 4 |  |
| Unsupported op QLinearSigmoid | 4 |  |
| Unsupported op CDist | 3 |  |
| Unsupported op FastGelu | 3 |  |
| Failed to build testbench (model.c:125:31: error: passing argument 3 of ‘node0_node1’ from incompatible pointer type [-Werror=incompatible-pointer-types]). | 2 |  |
| Failed to build testbench (model.c:137:31: error: passing argument 3 of ‘node0_node1’ from incompatible pointer type [-Werror=incompatible-pointer-types]). | 2 |  |
| Failed to build testbench (model.c:139:31: error: passing argument 3 of ‘node0_node1’ from incompatible pointer type [-Werror=incompatible-pointer-types]). | 2 |  |
| Unsupported op Affine | 2 | 7 |
| Unsupported op BifurcationDetector | 2 |  |
| Unsupported op DecoderMaskedMultiHeadAttention | 2 |  |
| Unsupported op QLinearLeakyRelu | 2 |  |
| Unsupported op Scale | 2 | 7 |
| Unsupported op UnfoldTensor | 2 |  |
| Unsupported op ConvTransposeWithDynamicPads | 1 |  |
| Unsupported op DynamicTimeWarping | 1 |  |
| Unsupported op ImageScaler | 1 | 7 |
| Unsupported op MaxpoolWithMask | 1 |  |
| Unsupported op MoE | 1 |  |
| Unsupported op NGramRepeatBlock | 1 |  |
| Unsupported op SampleOp | 1 |  |
| Unsupported test-data sequence input for verify: variable sequence element shapes are not supported | 1 | 9 |
| onnxruntime failed to run onnx2c-org/test/local_ops/test_gemm_CM_transA/model.onnx: [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Non-zero status code returned while running Gemm node. Name:'*' Status Message: Gemm: Invalid bias shape for broadcast | 1 | 12 |

## Error frequency by opset

| Error message | Opset | Count |
| --- | --- | --- |
| Unsupported op DynamicSlice | 1 | 12 |
| Unsupported op Crop | 1 | 2 |
| LayerNormalization scale rank must match normalized rank | 7 | 9 |
| Out of tolerance | 7 | 5 |
| Unsupported op Affine | 7 | 2 |
| Unsupported op Crop | 7 | 2 |
| Unsupported op Scale | 7 | 2 |
| Unsupported op ImageScaler | 7 | 1 |
| Unsupported test-data sequence input for verify: variable sequence element shapes are not supported | 9 | 1 |
| onnxruntime failed to run onnx2c-org/test/local_ops/test_gemm_CM_transA/model.onnx: [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Non-zero status code returned while running Gemm node. Name:'*' Status Message: Gemm: Invalid bias shape for broadcast | 12 | 1 |
| Out of tolerance | 17 | 7 |
| LayerNormalization scale rank must match normalized rank | 17 | 6 |

## Failing ONNX files

Lists every ONNX file with a non-success verification outcome.

| File | Opset | Verification | Supported | Error |
| --- | --- | --- | --- | --- |
| local_ops/test_gemm_CM_transA/model.onnx | 12 | Data/Data | ❌ | onnxruntime failed to run onnx2c-org/test/local_ops/test_gemm_CM_transA/model.onnx: [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Non-zero status code returned while running Gemm node. Name:'sclbl-onnx-node1' Status Message: Gemm: Invalid bias shape for broadcast |
| mnist/pytorch.onnx | 9 | Data/Data | ❌ | Unsupported test-data sequence input for verify: variable sequence element shapes are not supported |
| node/test_adam_multiple/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 62311) |
| test/contrib_ops/attention_lstm_op_test/BidirectionLstmWithBahdanauAM2BatchShortenSeqLen_run0/model.onnx |  | Data/Data | ❌ | Unsupported op AttnLSTM |
| test/contrib_ops/attention_lstm_op_test/BidirectionLstmWithBahdanauAMShortenSeqLength_run0/model.onnx |  | Data/Data | ❌ | Unsupported op AttnLSTM |
| test/contrib_ops/attention_lstm_op_test/ForwardLstmWithBahdanauAMShortenSeqLength_run0/model.onnx |  | Data/Data | ❌ | Unsupported op AttnLSTM |
| test/contrib_ops/attention_lstm_op_test/ForwardLstmWithBahdanauAMZeroAttention_run0/model.onnx |  | Data/Data | ❌ | Unsupported op AttnLSTM |
| test/contrib_ops/attention_lstm_op_test/ForwardLstmWithBahdanauAM_run0/model.onnx |  | Data/Data | ❌ | Unsupported op AttnLSTM |
| test/contrib_ops/attention_lstm_op_test/ReverseLstmWithBahdanauAMShortenSeqLength_run0/model.onnx |  | Data/Data | ❌ | Unsupported op AttnLSTM |
| test/contrib_ops/attention_op_test/Attention3DMask_run0/model.onnx |  | Data/Data | ❌ | Unsupported op Attention |
| test/contrib_ops/attention_op_test/Attention3DMask_run1/model.onnx |  | Data/Data | ❌ | Unsupported op Attention |
| test/contrib_ops/attention_op_test/AttentionBatch1AttentionBias_run0/model.onnx |  | Data/Data | ❌ | Unsupported op Attention |
| test/contrib_ops/attention_op_test/AttentionBatch1AttentionBias_run1/model.onnx |  | Data/Data | ❌ | Unsupported op Attention |
| test/contrib_ops/attention_op_test/AttentionBatch1WithQKVAttr1_run0/model.onnx |  | Data/Data | ❌ | Unsupported op Attention |
| test/contrib_ops/attention_op_test/AttentionBatch1WithQKVAttr1_run1/model.onnx |  | Data/Data | ❌ | Unsupported op Attention |
| test/contrib_ops/attention_op_test/AttentionBatch1WithQKVAttr2_run0/model.onnx |  | Data/Data | ❌ | Unsupported op Attention |
| test/contrib_ops/attention_op_test/AttentionBatch1WithQKVAttr2_run1/model.onnx |  | Data/Data | ❌ | Unsupported op Attention |
| test/contrib_ops/attention_op_test/AttentionBatch1_run0/model.onnx |  | Data/Data | ❌ | Unsupported op Attention |
| test/contrib_ops/attention_op_test/AttentionBatch1_run1/model.onnx |  | Data/Data | ❌ | Unsupported op Attention |
| test/contrib_ops/attention_op_test/AttentionBatch2AttentionBias_run0/model.onnx |  | Data/Data | ❌ | Unsupported op Attention |
| test/contrib_ops/attention_op_test/AttentionBatch2AttentionBias_run1/model.onnx |  | Data/Data | ❌ | Unsupported op Attention |
| test/contrib_ops/attention_op_test/AttentionBatch2AttentionMask_run0/model.onnx |  | Data/Data | ❌ | Unsupported op Attention |
| test/contrib_ops/attention_op_test/AttentionBatch2AttentionMask_run1/model.onnx |  | Data/Data | ❌ | Unsupported op Attention |
| test/contrib_ops/attention_op_test/AttentionBatch2LeftPaddingMaskIndex2_run0/model.onnx |  | Data/Data | ❌ | Unsupported op Attention |
| test/contrib_ops/attention_op_test/AttentionBatch2LeftPaddingMaskIndex2_run1/model.onnx |  | Data/Data | ❌ | Unsupported op Attention |
| test/contrib_ops/attention_op_test/AttentionBatch2MaskIndex2_run0/model.onnx |  | Data/Data | ❌ | Unsupported op Attention |
| test/contrib_ops/attention_op_test/AttentionBatch2MaskIndex2_run1/model.onnx |  | Data/Data | ❌ | Unsupported op Attention |
| test/contrib_ops/attention_op_test/AttentionBatch2_run0/model.onnx |  | Data/Data | ❌ | Unsupported op Attention |
| test/contrib_ops/attention_op_test/AttentionBatch2_run1/model.onnx |  | Data/Data | ❌ | Unsupported op Attention |
| test/contrib_ops/attention_op_test/AttentionDummyMask2D_run0/model.onnx |  | Data/Data | ❌ | Unsupported op Attention |
| test/contrib_ops/attention_op_test/AttentionDummyMask2D_run1/model.onnx |  | Data/Data | ❌ | Unsupported op Attention |
| test/contrib_ops/attention_op_test/AttentionEmptyPastState_run0/model.onnx |  | Data/Data | ❌ | Unsupported op Attention |
| test/contrib_ops/attention_op_test/AttentionEmptyPastState_run1/model.onnx |  | Data/Data | ❌ | Unsupported op Attention |
| test/contrib_ops/attention_op_test/AttentionLeftPaddingMaskIndex2_run0/model.onnx |  | Data/Data | ❌ | Unsupported op Attention |
| test/contrib_ops/attention_op_test/AttentionLeftPaddingMaskIndex2_run1/model.onnx |  | Data/Data | ❌ | Unsupported op Attention |
| test/contrib_ops/attention_op_test/AttentionMask1DEndNoWord_run0/model.onnx |  | Data/Data | ❌ | Unsupported op Attention |
| test/contrib_ops/attention_op_test/AttentionMask1DEndNoWord_run1/model.onnx |  | Data/Data | ❌ | Unsupported op Attention |
| test/contrib_ops/attention_op_test/AttentionMask1DNoWord_run0/model.onnx |  | Data/Data | ❌ | Unsupported op Attention |
| test/contrib_ops/attention_op_test/AttentionMask1DNoWord_run1/model.onnx |  | Data/Data | ❌ | Unsupported op Attention |
| test/contrib_ops/attention_op_test/AttentionMask2DNoWord_run0/model.onnx |  | Data/Data | ❌ | Unsupported op Attention |
| test/contrib_ops/attention_op_test/AttentionMask2DNoWord_run1/model.onnx |  | Data/Data | ❌ | Unsupported op Attention |
| test/contrib_ops/attention_op_test/AttentionMask3DNoWord_run0/model.onnx |  | Data/Data | ❌ | Unsupported op Attention |
| test/contrib_ops/attention_op_test/AttentionMask3DNoWord_run1/model.onnx |  | Data/Data | ❌ | Unsupported op Attention |
| test/contrib_ops/attention_op_test/AttentionMaskExceedSequence_run0/model.onnx |  | Data/Data | ❌ | Unsupported op Attention |
| test/contrib_ops/attention_op_test/AttentionMaskExceedSequence_run1/model.onnx |  | Data/Data | ❌ | Unsupported op Attention |
| test/contrib_ops/attention_op_test/AttentionMaskIndexOutOfRange_run0/model.onnx |  | Data/Data | ❌ | Unsupported op Attention |
| test/contrib_ops/attention_op_test/AttentionMaskIndexOutOfRange_run1/model.onnx |  | Data/Data | ❌ | Unsupported op Attention |
| test/contrib_ops/attention_op_test/AttentionMaskPartialSequence_run0/model.onnx |  | Data/Data | ❌ | Unsupported op Attention |
| test/contrib_ops/attention_op_test/AttentionMaskPartialSequence_run1/model.onnx |  | Data/Data | ❌ | Unsupported op Attention |
| test/contrib_ops/attention_op_test/AttentionNoMaskIndex_run0/model.onnx |  | Data/Data | ❌ | Unsupported op Attention |
| test/contrib_ops/attention_op_test/AttentionNoMaskIndex_run1/model.onnx |  | Data/Data | ❌ | Unsupported op Attention |
| test/contrib_ops/attention_op_test/AttentionPastStateBatch1_run0/model.onnx |  | Data/Data | ❌ | Unsupported op Attention |
| test/contrib_ops/attention_op_test/AttentionPastStateBatch1_run1/model.onnx |  | Data/Data | ❌ | Unsupported op Attention |
| test/contrib_ops/attention_op_test/AttentionPastStateBatch2WithPadding_run0/model.onnx |  | Data/Data | ❌ | Unsupported op Attention |
| test/contrib_ops/attention_op_test/AttentionPastStateBatch2WithPadding_run1/model.onnx |  | Data/Data | ❌ | Unsupported op Attention |
| test/contrib_ops/attention_op_test/AttentionPastStateBatch2_run0/model.onnx |  | Data/Data | ❌ | Unsupported op Attention |
| test/contrib_ops/attention_op_test/AttentionPastStateBatch2_run1/model.onnx |  | Data/Data | ❌ | Unsupported op Attention |
| test/contrib_ops/attention_op_test/AttentionPastState_dynamic_run0/model.onnx |  | Data/Data | ❌ | Unsupported op Attention |
| test/contrib_ops/attention_op_test/AttentionPrunedModel_run0/model.onnx |  | Data/Data | ❌ | Unsupported op Attention |
| test/contrib_ops/attention_op_test/AttentionPrunedModel_run1/model.onnx |  | Data/Data | ❌ | Unsupported op Attention |
| test/contrib_ops/attention_op_test/AttentionRightPaddingMaskIndex2_run0/model.onnx |  | Data/Data | ❌ | Unsupported op Attention |
| test/contrib_ops/attention_op_test/AttentionRightPaddingMaskIndex2_run1/model.onnx |  | Data/Data | ❌ | Unsupported op Attention |
| test/contrib_ops/attention_op_test/AttentionUnidirectional3DMask_run0/model.onnx |  | Data/Data | ❌ | Unsupported op Attention |
| test/contrib_ops/attention_op_test/AttentionUnidirectional3DMask_run1/model.onnx |  | Data/Data | ❌ | Unsupported op Attention |
| test/contrib_ops/attention_op_test/AttentionUnidirectionalAttentionMask_run0/model.onnx |  | Data/Data | ❌ | Unsupported op Attention |
| test/contrib_ops/attention_op_test/AttentionUnidirectionalAttentionMask_run1/model.onnx |  | Data/Data | ❌ | Unsupported op Attention |
| test/contrib_ops/attention_op_test/AttentionUnidirectional_run0/model.onnx |  | Data/Data | ❌ | Unsupported op Attention |
| test/contrib_ops/attention_op_test/AttentionUnidirectional_run1/model.onnx |  | Data/Data | ❌ | Unsupported op Attention |
| test/contrib_ops/attention_op_test/AttentionWithNormFactor_run0/model.onnx |  | Data/Data | ❌ | Unsupported op Attention |
| test/contrib_ops/attention_op_test/AttentionWithNormFactor_run1/model.onnx |  | Data/Data | ❌ | Unsupported op Attention |
| test/contrib_ops/attention_op_test/Attention_Mask1D_Fp32_B2_S64_run0/model.onnx |  | Data/Data | ❌ | Unsupported op Attention |
| test/contrib_ops/attention_op_test/Attention_Mask2D_Fp32_B2_S32_run0/model.onnx |  | Data/Data | ❌ | Unsupported op Attention |
| test/contrib_ops/attention_op_test/SharedPrepackedWeights_run0/model.onnx |  | Data/Data | ❌ | Unsupported op Attention |
| test/contrib_ops/attention_op_test/SharedPrepackedWeights_run1/model.onnx |  | Data/Data | ❌ | Unsupported op Attention |
| test/contrib_ops/bifurcation_detector_op_test/Test1_run0/model.onnx |  | Data/Data | ❌ | Unsupported op BifurcationDetector |
| test/contrib_ops/bifurcation_detector_op_test/Test2_run0/model.onnx |  | Data/Data | ❌ | Unsupported op BifurcationDetector |
| test/contrib_ops/cdist_op_test/DoubleEuclidean_run0/model.onnx |  | Data/Data | ❌ | Unsupported op CDist |
| test/contrib_ops/cdist_op_test/Euclidean_run0/model.onnx |  | Data/Data | ❌ | Unsupported op CDist |
| test/contrib_ops/cdist_op_test/Sqeuclidean_run0/model.onnx |  | Data/Data | ❌ | Unsupported op CDist |
| test/contrib_ops/conv_transpose_with_dynamic_pads_test/ConvTransposeWithDynamicPads_run0/model.onnx |  | Data/Data | ❌ | Unsupported op ConvTransposeWithDynamicPads |
| test/contrib_ops/crop_and_resize_op_test/CropAndResize_1122_run0/model.onnx |  | Data/Data | ❌ | Unsupported op CropAndResize |
| test/contrib_ops/crop_and_resize_op_test/CropAndResize_1122_run1/model.onnx |  | Data/Data | ❌ | Unsupported op CropAndResize |
| test/contrib_ops/crop_and_resize_op_test/CropAndResize_1122_run2/model.onnx |  | Data/Data | ❌ | Unsupported op CropAndResize |
| test/contrib_ops/crop_and_resize_op_test/CropAndResize_1133_run0/model.onnx |  | Data/Data | ❌ | Unsupported op CropAndResize |
| test/contrib_ops/crop_and_resize_op_test/CropAndResize_1133_run1/model.onnx |  | Data/Data | ❌ | Unsupported op CropAndResize |
| test/contrib_ops/crop_and_resize_op_test/CropAndResize_1133_run2/model.onnx |  | Data/Data | ❌ | Unsupported op CropAndResize |
| test/contrib_ops/crop_and_resize_op_test/CropAndResize_1222_run0/model.onnx |  | Data/Data | ❌ | Unsupported op CropAndResize |
| test/contrib_ops/crop_and_resize_op_test/CropAndResize_1222_run1/model.onnx |  | Data/Data | ❌ | Unsupported op CropAndResize |
| test/contrib_ops/crop_and_resize_op_test/CropAndResize_2122_run0/model.onnx |  | Data/Data | ❌ | Unsupported op CropAndResize |
| test/contrib_ops/crop_and_resize_op_test/CropAndResize_2122_run1/model.onnx |  | Data/Data | ❌ | Unsupported op CropAndResize |
| test/contrib_ops/crop_op_test/Crop_Border_run0/model.onnx | 1 | Data/Data | ❌ | Unsupported op Crop |
| test/contrib_ops/crop_op_test/Crop_Scale_run0/model.onnx | 1 | Data/Data | ❌ | Unsupported op Crop |
| test/contrib_ops/decoder_masked_multihead_attention_op_test/cpu_cross_attn_fp32_run0/model.onnx |  | Data/Data | ❌ | Unsupported op DecoderMaskedMultiHeadAttention |
| test/contrib_ops/decoder_masked_multihead_attention_op_test/cpu_self_attn_fp32_run0/model.onnx |  | Data/Data | ❌ | Unsupported op DecoderMaskedMultiHeadAttention |
| test/contrib_ops/dynamic_quantize_matmul_test/Int8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op DynamicQuantizeMatMul |
| test/contrib_ops/dynamic_quantize_matmul_test/Int8_run1/model.onnx |  | Data/Data | ❌ | Unsupported op DynamicQuantizeMatMul |
| test/contrib_ops/dynamic_quantize_matmul_test/Int8_run10/model.onnx |  | Data/Data | ❌ | Unsupported op DynamicQuantizeMatMul |
| test/contrib_ops/dynamic_quantize_matmul_test/Int8_run11/model.onnx |  | Data/Data | ❌ | Unsupported op DynamicQuantizeMatMul |
| test/contrib_ops/dynamic_quantize_matmul_test/Int8_run12/model.onnx |  | Data/Data | ❌ | Unsupported op DynamicQuantizeMatMul |
| test/contrib_ops/dynamic_quantize_matmul_test/Int8_run13/model.onnx |  | Data/Data | ❌ | Unsupported op DynamicQuantizeMatMul |
| test/contrib_ops/dynamic_quantize_matmul_test/Int8_run14/model.onnx |  | Data/Data | ❌ | Unsupported op DynamicQuantizeMatMul |
| test/contrib_ops/dynamic_quantize_matmul_test/Int8_run15/model.onnx |  | Data/Data | ❌ | Unsupported op DynamicQuantizeMatMul |
| test/contrib_ops/dynamic_quantize_matmul_test/Int8_run2/model.onnx |  | Data/Data | ❌ | Unsupported op DynamicQuantizeMatMul |
| test/contrib_ops/dynamic_quantize_matmul_test/Int8_run3/model.onnx |  | Data/Data | ❌ | Unsupported op DynamicQuantizeMatMul |
| test/contrib_ops/dynamic_quantize_matmul_test/Int8_run4/model.onnx |  | Data/Data | ❌ | Unsupported op DynamicQuantizeMatMul |
| test/contrib_ops/dynamic_quantize_matmul_test/Int8_run5/model.onnx |  | Data/Data | ❌ | Unsupported op DynamicQuantizeMatMul |
| test/contrib_ops/dynamic_quantize_matmul_test/Int8_run6/model.onnx |  | Data/Data | ❌ | Unsupported op DynamicQuantizeMatMul |
| test/contrib_ops/dynamic_quantize_matmul_test/Int8_run7/model.onnx |  | Data/Data | ❌ | Unsupported op DynamicQuantizeMatMul |
| test/contrib_ops/dynamic_quantize_matmul_test/Int8_run8/model.onnx |  | Data/Data | ❌ | Unsupported op DynamicQuantizeMatMul |
| test/contrib_ops/dynamic_quantize_matmul_test/Int8_run9/model.onnx |  | Data/Data | ❌ | Unsupported op DynamicQuantizeMatMul |
| test/contrib_ops/dynamic_quantize_matmul_test/UInt8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op DynamicQuantizeMatMul |
| test/contrib_ops/dynamic_quantize_matmul_test/UInt8_run1/model.onnx |  | Data/Data | ❌ | Unsupported op DynamicQuantizeMatMul |
| test/contrib_ops/dynamic_quantize_matmul_test/UInt8_run10/model.onnx |  | Data/Data | ❌ | Unsupported op DynamicQuantizeMatMul |
| test/contrib_ops/dynamic_quantize_matmul_test/UInt8_run11/model.onnx |  | Data/Data | ❌ | Unsupported op DynamicQuantizeMatMul |
| test/contrib_ops/dynamic_quantize_matmul_test/UInt8_run12/model.onnx |  | Data/Data | ❌ | Unsupported op DynamicQuantizeMatMul |
| test/contrib_ops/dynamic_quantize_matmul_test/UInt8_run13/model.onnx |  | Data/Data | ❌ | Unsupported op DynamicQuantizeMatMul |
| test/contrib_ops/dynamic_quantize_matmul_test/UInt8_run14/model.onnx |  | Data/Data | ❌ | Unsupported op DynamicQuantizeMatMul |
| test/contrib_ops/dynamic_quantize_matmul_test/UInt8_run15/model.onnx |  | Data/Data | ❌ | Unsupported op DynamicQuantizeMatMul |
| test/contrib_ops/dynamic_quantize_matmul_test/UInt8_run2/model.onnx |  | Data/Data | ❌ | Unsupported op DynamicQuantizeMatMul |
| test/contrib_ops/dynamic_quantize_matmul_test/UInt8_run3/model.onnx |  | Data/Data | ❌ | Unsupported op DynamicQuantizeMatMul |
| test/contrib_ops/dynamic_quantize_matmul_test/UInt8_run4/model.onnx |  | Data/Data | ❌ | Unsupported op DynamicQuantizeMatMul |
| test/contrib_ops/dynamic_quantize_matmul_test/UInt8_run5/model.onnx |  | Data/Data | ❌ | Unsupported op DynamicQuantizeMatMul |
| test/contrib_ops/dynamic_quantize_matmul_test/UInt8_run6/model.onnx |  | Data/Data | ❌ | Unsupported op DynamicQuantizeMatMul |
| test/contrib_ops/dynamic_quantize_matmul_test/UInt8_run7/model.onnx |  | Data/Data | ❌ | Unsupported op DynamicQuantizeMatMul |
| test/contrib_ops/dynamic_quantize_matmul_test/UInt8_run8/model.onnx |  | Data/Data | ❌ | Unsupported op DynamicQuantizeMatMul |
| test/contrib_ops/dynamic_quantize_matmul_test/UInt8_run9/model.onnx |  | Data/Data | ❌ | Unsupported op DynamicQuantizeMatMul |
| test/contrib_ops/dynamic_quantize_matmul_test/UInt8_test_with_empty_input_run0/model.onnx |  | Data/Data | ❌ | Unsupported op DynamicQuantizeMatMul |
| test/contrib_ops/dynamic_quantize_matmul_test/WithConstantBInputs_run0/model.onnx |  | Data/Data | ❌ | Unsupported op DynamicQuantizeMatMul |
| test/contrib_ops/dynamic_quantize_matmul_test/WithConstantBInputs_run1/model.onnx |  | Data/Data | ❌ | Unsupported op DynamicQuantizeMatMul |
| test/contrib_ops/dynamic_quantize_matmul_test/WithConstantBInputs_run2/model.onnx |  | Data/Data | ❌ | Unsupported op DynamicQuantizeMatMul |
| test/contrib_ops/dynamic_quantize_matmul_test/WithConstantBInputs_run3/model.onnx |  | Data/Data | ❌ | Unsupported op DynamicQuantizeMatMul |
| test/contrib_ops/dynamic_quantize_matmul_test/WithConstantBInputs_run4/model.onnx |  | Data/Data | ❌ | Unsupported op DynamicQuantizeMatMul |
| test/contrib_ops/dynamic_quantize_matmul_test/WithConstantBInputs_run5/model.onnx |  | Data/Data | ❌ | Unsupported op DynamicQuantizeMatMul |
| test/contrib_ops/dynamic_slice_op_test/dynamic_slice_ends_out_of_bounds_run0/model.onnx | 1 | Data/Data | ❌ | Unsupported op DynamicSlice |
| test/contrib_ops/dynamic_slice_op_test/dynamic_slice_full_axes_run0/model.onnx | 1 | Data/Data | ❌ | Unsupported op DynamicSlice |
| test/contrib_ops/dynamic_slice_op_test/dynamic_slice_full_axes_run1/model.onnx | 1 | Data/Data | ❌ | Unsupported op DynamicSlice |
| test/contrib_ops/dynamic_slice_op_test/dynamic_slice_varied_types_run0/model.onnx | 1 | Data/Data | ❌ | Unsupported op DynamicSlice |
| test/contrib_ops/dynamic_slice_op_test/dynamic_slice_varied_types_run1/model.onnx | 1 | Data/Data | ❌ | Unsupported op DynamicSlice |
| test/contrib_ops/dynamic_slice_op_test/dynamic_slice_varied_types_run2/model.onnx | 1 | Data/Data | ❌ | Unsupported op DynamicSlice |
| test/contrib_ops/dynamic_slice_op_test/dynamic_slice_varied_types_run3/model.onnx | 1 | Data/Data | ❌ | Unsupported op DynamicSlice |
| test/contrib_ops/dynamic_slice_op_test/dynamic_slice_varied_types_run4/model.onnx | 1 | Data/Data | ❌ | Unsupported op DynamicSlice |
| test/contrib_ops/dynamic_slice_op_test/dynamic_slice_with_axes_run0/model.onnx | 1 | Data/Data | ❌ | Unsupported op DynamicSlice |
| test/contrib_ops/dynamic_slice_op_test/dynamic_slice_with_axes_run1/model.onnx | 1 | Data/Data | ❌ | Unsupported op DynamicSlice |
| test/contrib_ops/dynamic_slice_op_test/dynamic_slice_with_negative_axes_run0/model.onnx | 1 | Data/Data | ❌ | Unsupported op DynamicSlice |
| test/contrib_ops/dynamic_slice_op_test/dynamic_slice_with_negative_axes_run1/model.onnx | 1 | Data/Data | ❌ | Unsupported op DynamicSlice |
| test/contrib_ops/dynamic_time_warping_op_test/simple_run0/model.onnx |  | Data/Data | ❌ | Unsupported op DynamicTimeWarping |
| test/contrib_ops/element_wise_ops_test/AffineDefaultAttributes_run0/model.onnx | 7 | Data/Data | ❌ | Unsupported op Affine |
| test/contrib_ops/element_wise_ops_test/Affine_run0/model.onnx | 7 | Data/Data | ❌ | Unsupported op Affine |
| test/contrib_ops/element_wise_ops_test/Float_run0/model.onnx |  | Data/Data | ❌ | Unsupported op BiasGelu |
| test/contrib_ops/element_wise_ops_test/Float_run1/model.onnx |  | Data/Data | ❌ | Unsupported op BiasGelu |
| test/contrib_ops/element_wise_ops_test/Float_run2/model.onnx |  | Data/Data | ❌ | Unsupported op BiasGelu |
| test/contrib_ops/element_wise_ops_test/Float_run3/model.onnx |  | Data/Data | ❌ | Unsupported op BiasGelu |
| test/contrib_ops/element_wise_ops_test/Float_run4/model.onnx |  | Data/Data | ❌ | Unsupported op BiasGelu |
| test/contrib_ops/element_wise_ops_test/Float_run5/model.onnx |  | Data/Data | ❌ | Unsupported op BiasGelu |
| test/contrib_ops/element_wise_ops_test/Scale_Default_run0/model.onnx | 7 | Data/Data | ❌ | Unsupported op Scale |
| test/contrib_ops/element_wise_ops_test/Scale_run0/model.onnx | 7 | Data/Data | ❌ | Unsupported op Scale |
| test/contrib_ops/embed_layer_norm_op_test/EmbedLayerNormBatch1_EmbeddingSum_NoMaskIndex_run0/model.onnx |  | Data/Data | ❌ | Unsupported op EmbedLayerNormalization |
| test/contrib_ops/embed_layer_norm_op_test/EmbedLayerNormBatch1_EmbeddingSum_run0/model.onnx |  | Data/Data | ❌ | Unsupported op EmbedLayerNormalization |
| test/contrib_ops/embed_layer_norm_op_test/EmbedLayerNormBatch1_PositionIdsDiffOrder_run0/model.onnx |  | Data/Data | ❌ | Unsupported op EmbedLayerNormalization |
| test/contrib_ops/embed_layer_norm_op_test/EmbedLayerNormBatch1_PositionIds_run0/model.onnx |  | Data/Data | ❌ | Unsupported op EmbedLayerNormalization |
| test/contrib_ops/embed_layer_norm_op_test/EmbedLayerNormBatch1_run0/model.onnx |  | Data/Data | ❌ | Unsupported op EmbedLayerNormalization |
| test/contrib_ops/embed_layer_norm_op_test/EmbedLayerNormBatch2_NoMask_run0/model.onnx |  | Data/Data | ❌ | Unsupported op EmbedLayerNormalization |
| test/contrib_ops/embed_layer_norm_op_test/EmbedLayerNormBatch2_run0/model.onnx |  | Data/Data | ❌ | Unsupported op EmbedLayerNormalization |
| test/contrib_ops/embed_layer_norm_op_test/EmbedLayerNormBatch3_PositionIds_BroadCast_run0/model.onnx |  | Data/Data | ❌ | Unsupported op EmbedLayerNormalization |
| test/contrib_ops/embed_layer_norm_op_test/EmbedLayerNormBatch_Distill_run0/model.onnx |  | Data/Data | ❌ | Unsupported op EmbedLayerNormalization |
| test/contrib_ops/embed_layer_norm_op_test/EmbedLayerNormLargeBatchSmallHiddenSize_run0/model.onnx |  | Data/Data | ❌ | Unsupported op EmbedLayerNormalization |
| test/contrib_ops/expand_dims_test/Basic_run0/model.onnx |  | Data/Data | ❌ | Unsupported op ExpandDims |
| test/contrib_ops/expand_dims_test/Basic_run1/model.onnx |  | Data/Data | ❌ | Unsupported op ExpandDims |
| test/contrib_ops/expand_dims_test/Basic_run2/model.onnx |  | Data/Data | ❌ | Unsupported op ExpandDims |
| test/contrib_ops/expand_dims_test/MaxAxis_run0/model.onnx |  | Data/Data | ❌ | Unsupported op ExpandDims |
| test/contrib_ops/expand_dims_test/MaxAxis_run1/model.onnx |  | Data/Data | ❌ | Unsupported op ExpandDims |
| test/contrib_ops/expand_dims_test/MinAxis_run0/model.onnx |  | Data/Data | ❌ | Unsupported op ExpandDims |
| test/contrib_ops/expand_dims_test/MinAxis_run1/model.onnx |  | Data/Data | ❌ | Unsupported op ExpandDims |
| test/contrib_ops/expand_dims_test/NegativeAxisOutOfRange_run0/model.onnx |  | Data/Data | ❌ | Unsupported op ExpandDims |
| test/contrib_ops/expand_dims_test/NegativeAxisOutOfRange_run1/model.onnx |  | Data/Data | ❌ | Unsupported op ExpandDims |
| test/contrib_ops/expand_dims_test/PositiveAxisOutOfRange_run0/model.onnx |  | Data/Data | ❌ | Unsupported op ExpandDims |
| test/contrib_ops/expand_dims_test/PositiveAxisOutOfRange_run1/model.onnx |  | Data/Data | ❌ | Unsupported op ExpandDims |
| test/contrib_ops/fastgelu_op_test/FastGeluWithBiasFloat32_run0/model.onnx |  | Data/Data | ❌ | Unsupported op FastGelu |
| test/contrib_ops/fastgelu_op_test/FastGeluWithNullInput_run0/model.onnx |  | Data/Data | ❌ | Unsupported op FastGelu |
| test/contrib_ops/fastgelu_op_test/FastGeluWithoutBiasFloat32_run0/model.onnx |  | Data/Data | ❌ | Unsupported op FastGelu |
| test/contrib_ops/fused_conv_test/Conv2D_Bias_Relu_run0/model.onnx |  | Data/Data | ❌ | Unsupported op FusedConv |
| test/contrib_ops/fused_conv_test/Conv2D_HardSigmoid_run0/model.onnx |  | Data/Data | ❌ | Unsupported op FusedConv |
| test/contrib_ops/fused_conv_test/Conv2D_Relu_run0/model.onnx |  | Data/Data | ❌ | Unsupported op FusedConv |
| test/contrib_ops/fused_conv_test/Cpu_Conv2D_Bias_Z_Relu_run0/model.onnx |  | Data/Data | ❌ | Unsupported op FusedConv |
| test/contrib_ops/fused_matmul_op_test/FloatTypeNoTranspose_run0/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeNoTranspose_run1/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeNoTranspose_run10/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeNoTranspose_run11/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeNoTranspose_run2/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeNoTranspose_run3/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeNoTranspose_run4/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeNoTranspose_run5/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeNoTranspose_run6/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeNoTranspose_run7/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeNoTranspose_run8/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeNoTranspose_run9/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run0/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run1/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run10/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run11/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run12/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run13/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run14/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run15/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run16/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run17/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run18/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run19/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run2/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run20/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run21/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run22/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run23/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run24/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run25/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run26/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run27/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run28/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run29/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run3/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run30/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run31/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run32/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run33/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run34/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run35/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run36/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run37/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run38/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run39/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run4/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run40/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run41/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run42/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run43/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run44/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run45/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run46/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run47/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run48/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run49/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run5/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run50/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run51/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run52/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run53/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run54/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run55/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run56/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run57/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run58/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run59/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run6/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run60/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run61/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run62/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run63/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run64/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run65/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run66/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run67/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run68/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run69/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run7/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run70/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run71/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run8/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeScale_run9/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeAB_run0/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeAB_run1/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeAB_run10/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeAB_run11/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeAB_run12/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeAB_run13/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeAB_run14/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeAB_run15/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeAB_run16/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeAB_run17/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeAB_run18/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeAB_run19/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeAB_run2/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeAB_run20/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeAB_run21/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeAB_run22/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeAB_run23/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeAB_run3/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeAB_run4/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeAB_run5/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeAB_run6/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeAB_run7/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeAB_run8/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeAB_run9/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeA_run0/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeA_run1/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeA_run10/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeA_run11/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeA_run2/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeA_run3/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeA_run4/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeA_run5/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeA_run6/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeA_run7/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeA_run8/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeA_run9/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeB_run0/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeB_run1/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeB_run10/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeB_run11/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeB_run12/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeB_run13/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeB_run14/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeB_run15/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeB_run16/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeB_run17/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeB_run18/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeB_run19/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeB_run2/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeB_run20/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeB_run21/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeB_run22/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeB_run23/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeB_run3/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeB_run4/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeB_run5/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeB_run6/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeB_run7/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeB_run8/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeB_run9/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run0/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run1/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run10/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run11/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run12/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run13/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run14/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run15/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run16/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run17/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run18/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run19/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run2/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run20/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run21/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run22/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run23/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run24/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run25/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run26/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run27/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run28/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run29/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run3/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run30/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run31/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run32/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run33/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run34/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run35/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run36/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run37/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run38/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run39/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run4/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run40/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run41/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run42/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run43/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run44/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run45/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run46/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run47/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run5/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run6/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run7/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run8/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/fused_matmul_op_test/FloatTypeTransposeBatch_run9/model.onnx |  | Data/Data | ❌ | Unsupported op FusedMatMul |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_4Bits_run0/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_4Bits_run1/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_4Bits_run10/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_4Bits_run11/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_4Bits_run12/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_4Bits_run13/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_4Bits_run14/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_4Bits_run15/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_4Bits_run2/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_4Bits_run3/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_4Bits_run4/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_4Bits_run5/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_4Bits_run6/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_4Bits_run7/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_4Bits_run8/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_4Bits_run9/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_8Bits_run0/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_8Bits_run1/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_8Bits_run2/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_8Bits_run3/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_8Bits_run4/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_8Bits_run5/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_8Bits_run6/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_8Bits_run7/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_run0/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_run1/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_run10/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_run11/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_run12/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_run13/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_run14/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_run15/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_run2/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_run3/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_run4/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_run5/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_run6/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_run7/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_run8/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0NoZeroPoints_run9/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_4Bits_run0/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_4Bits_run1/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_4Bits_run2/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_4Bits_run3/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_4Bits_run4/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_4Bits_run5/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_4Bits_run6/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_4Bits_run7/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_8Bits_run0/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_8Bits_run1/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_8Bits_run2/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_8Bits_run3/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_8Bits_run4/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_8Bits_run5/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_8Bits_run6/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_8Bits_run7/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run0/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run1/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run10/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run11/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run12/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run13/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run14/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run15/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run16/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run17/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run18/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run19/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run2/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run20/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run21/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run22/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run23/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run24/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run25/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run26/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run27/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run28/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run29/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run3/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run30/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run31/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run4/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run5/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run6/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run7/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run8/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis0WithZeroPoints_run9/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run0/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run1/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run10/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run11/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run12/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run13/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run14/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run15/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run16/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run17/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run18/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run19/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run2/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run20/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run21/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run22/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run23/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run24/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run25/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run26/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run27/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run28/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run29/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run3/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run30/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run31/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run4/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run5/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run6/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run7/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run8/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis1_run9/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run0/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run1/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run10/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run11/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run12/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run13/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run14/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run15/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run16/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run17/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run18/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run19/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run2/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run20/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run21/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run22/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run23/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run24/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run25/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run26/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run27/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run28/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run29/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run3/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run30/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run31/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run4/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run5/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run6/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run7/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run8/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/GatherAxis2_run9/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/InvalidBlockSize_run0/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/InvalidBlockSize_run1/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/InvalidBlockSize_run2/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/InvalidBlockSize_run3/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/InvalidBlockSize_run4/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/InvalidBlockSize_run5/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/InvalidGatherAxis_run0/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/InvalidGatherAxis_run1/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/InvalidGatherAxis_run2/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/InvalidGatherAxis_run3/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/InvalidGatherAxis_run4/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/InvalidGatherAxis_run5/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/InvalidIndices_run0/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/InvalidIndices_run1/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/InvalidIndices_run2/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/InvalidIndices_run3/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/InvalidIndices_run4/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/InvalidIndices_run5/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/InvalidQuantizeAxis_run0/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/InvalidQuantizeAxis_run1/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/InvalidQuantizeAxis_run2/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/InvalidQuantizeAxis_run3/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/InvalidQuantizeAxis_run4/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/InvalidQuantizeAxis_run5/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/NotSupportedBits_run0/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/NotSupportedBits_run1/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/NotSupportedBits_run10/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/NotSupportedBits_run11/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/NotSupportedBits_run2/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/NotSupportedBits_run3/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/NotSupportedBits_run4/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/NotSupportedBits_run5/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/NotSupportedBits_run6/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/NotSupportedBits_run7/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/NotSupportedBits_run8/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/NotSupportedBits_run9/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/ShapeMismatch_run0/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/ShapeMismatch_run1/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/ShapeMismatch_run2/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/ShapeMismatch_run3/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/ShapeMismatch_run4/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/ShapeMismatch_run5/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/UnsupportedTypes_run0/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/UnsupportedTypes_run1/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/UnsupportedTypes_run10/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/UnsupportedTypes_run11/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/UnsupportedTypes_run12/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/UnsupportedTypes_run13/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/UnsupportedTypes_run14/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/UnsupportedTypes_run15/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/UnsupportedTypes_run16/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/UnsupportedTypes_run17/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/UnsupportedTypes_run18/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/UnsupportedTypes_run19/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/UnsupportedTypes_run2/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/UnsupportedTypes_run20/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/UnsupportedTypes_run21/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/UnsupportedTypes_run22/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/UnsupportedTypes_run23/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/UnsupportedTypes_run3/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/UnsupportedTypes_run4/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/UnsupportedTypes_run5/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/UnsupportedTypes_run6/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/UnsupportedTypes_run7/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/UnsupportedTypes_run8/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/UnsupportedTypes_run9/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/UnsupportedUInt8DataType_run0/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/UnsupportedUInt8DataType_run1/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/UnsupportedUInt8DataType_run2/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gather_block_quantized_op_test/UnsupportedUInt8DataType_run3/model.onnx |  | Data/Data | ❌ | Unsupported op GatherBlockQuantized |
| test/contrib_ops/gridsample_test/gridsample_aligncorners_true_run0/model.onnx |  | Data/Data | ❌ | GridSample mode 'bilinear' is not supported |
| test/contrib_ops/gridsample_test/gridsample_default_run0/model.onnx |  | Data/Data | ❌ | GridSample mode 'bilinear' is not supported |
| test/contrib_ops/gridsample_test/gridsample_mode_bicubic_run0/model.onnx |  | Data/Data | ❌ | GridSample mode 'bicubic' is not supported |
| test/contrib_ops/gridsample_test/gridsample_mode_bilinear_run0/model.onnx |  | Data/Data | ❌ | GridSample mode 'bilinear' is not supported |
| test/contrib_ops/gridsample_test/gridsample_paddingmode_border_run0/model.onnx |  | Data/Data | ❌ | GridSample mode 'bilinear' is not supported |
| test/contrib_ops/gridsample_test/gridsample_paddingmode_reflection_run0/model.onnx |  | Data/Data | ❌ | GridSample mode 'bilinear' is not supported |
| test/contrib_ops/gridsample_test/gridsample_paddingmode_zeros_run0/model.onnx |  | Data/Data | ❌ | GridSample mode 'bilinear' is not supported |
| test/contrib_ops/inverse_test/four_by_four_batches_float_run0/model.onnx |  | Data/Data | ❌ | Unsupported op Inverse |
| test/contrib_ops/inverse_test/four_by_four_float_run0/model.onnx |  | Data/Data | ❌ | Unsupported op Inverse |
| test/contrib_ops/inverse_test/two_by_two_double_run0/model.onnx |  | Data/Data | ❌ | Unsupported op Inverse |
| test/contrib_ops/inverse_test/two_by_two_float16_run0/model.onnx |  | Data/Data | ❌ | Unsupported op Inverse |
| test/contrib_ops/inverse_test/two_by_two_float_run0/model.onnx |  | Data/Data | ❌ | Unsupported op Inverse |
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
| test/contrib_ops/matmul_4bits_test/LegacyShape_4b_run0/model.onnx |  | Data/Data | ❌ | Failed to build testbench (model.c:125:31: error: passing argument 3 of ‘node0_node1’ from incompatible pointer type [-Werror=incompatible-pointer-types]). |
| test/contrib_ops/matmul_4bits_test/LegacyShape_4b_run1/model.onnx |  | Data/Data | ❌ | Failed to build testbench (model.c:139:31: error: passing argument 3 of ‘node0_node1’ from incompatible pointer type [-Werror=incompatible-pointer-types]). |
| test/contrib_ops/matmul_4bits_test/LegacyShape_4b_run2/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/LegacyShape_4b_run3/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/LegacyShape_4b_run4/model.onnx |  | Data/Data | ❌ | Failed to build testbench (model.c:137:31: error: passing argument 3 of ‘node0_node1’ from incompatible pointer type [-Werror=incompatible-pointer-types]). |
| test/contrib_ops/matmul_4bits_test/LegacyShape_4b_run5/model.onnx |  | Data/Data | ❌ | Failed to build testbench (model.c:125:31: error: passing argument 3 of ‘node0_node1’ from incompatible pointer type [-Werror=incompatible-pointer-types]). |
| test/contrib_ops/matmul_4bits_test/LegacyShape_4b_run6/model.onnx |  | Data/Data | ❌ | Failed to build testbench (model.c:139:31: error: passing argument 3 of ‘node0_node1’ from incompatible pointer type [-Werror=incompatible-pointer-types]). |
| test/contrib_ops/matmul_4bits_test/LegacyShape_4b_run7/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/LegacyShape_4b_run8/model.onnx |  | Data/Data | ❌ | MatMulNBits g_idx (input 4) is not supported |
| test/contrib_ops/matmul_4bits_test/LegacyShape_4b_run9/model.onnx |  | Data/Data | ❌ | Failed to build testbench (model.c:137:31: error: passing argument 3 of ‘node0_node1’ from incompatible pointer type [-Werror=incompatible-pointer-types]). |
| test/contrib_ops/matmul_bnb4_test/Float32_run0/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run1/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run10/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run100/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run101/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run102/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run103/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run104/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run105/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run106/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run107/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run108/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run109/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run11/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run110/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run111/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run112/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run113/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run114/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run115/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run116/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run117/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run118/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run119/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run12/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run120/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run121/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run122/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run123/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run124/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run125/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run126/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run127/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run128/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run129/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run13/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run130/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run131/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run132/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run133/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run134/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run135/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run136/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run137/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run138/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run139/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run14/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run140/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run141/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run142/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run143/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run144/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run145/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run146/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run147/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run148/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run149/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run15/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run150/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run151/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run152/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run153/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run154/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run155/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run156/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run157/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run158/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run159/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run16/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run160/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run161/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run162/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run163/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run164/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run165/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run166/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run167/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run168/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run169/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run17/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run170/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run171/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run172/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run173/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run174/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run175/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run176/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run177/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run178/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run179/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run18/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run180/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run181/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run182/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run183/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run184/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run185/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run186/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run187/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run188/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run189/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run19/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run190/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run191/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run192/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run193/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run194/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run195/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run196/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run197/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run198/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run199/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run2/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run20/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run200/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run201/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run202/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run203/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run204/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run205/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run206/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run207/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run208/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run209/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run21/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run210/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run211/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run212/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run213/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run214/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run215/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run216/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run217/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run218/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run219/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run22/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run220/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run221/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run222/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run223/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run224/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run225/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run226/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run227/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run228/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run229/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run23/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run230/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run231/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run232/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run233/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run234/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run235/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run236/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run237/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run238/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run239/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run24/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run240/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run241/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run242/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run243/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run244/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run245/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run246/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run247/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run248/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run249/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run25/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run250/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run251/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run252/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run253/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run254/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run255/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run256/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run257/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run258/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run259/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run26/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run260/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run261/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run262/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run263/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run264/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run265/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run266/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run267/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run268/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run269/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run27/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run270/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run271/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run272/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run273/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run274/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run275/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run276/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run277/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run278/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run279/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run28/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run280/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run281/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run282/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run283/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run284/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run285/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run286/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run287/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run288/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run289/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run29/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run290/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run291/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run292/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run293/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run294/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run295/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run296/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run297/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run298/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run299/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run3/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run30/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run300/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run301/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run302/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run303/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run304/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run305/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run306/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run307/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run308/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run309/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run31/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run310/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run311/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run312/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run313/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run314/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run315/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run316/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run317/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run318/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run319/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run32/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run320/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run321/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run322/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run323/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run324/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run325/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run326/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run327/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run328/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run329/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run33/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run330/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run331/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run332/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run333/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run334/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run335/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run336/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run337/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run338/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run339/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run34/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run340/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run341/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run342/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run343/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run344/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run345/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run346/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run347/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run348/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run349/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run35/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run350/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run351/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run352/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run353/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run354/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run355/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run356/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run357/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run358/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run359/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run36/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run360/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run361/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run362/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run363/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run364/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run365/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run366/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run367/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run368/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run369/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run37/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run370/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run371/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run372/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run373/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run374/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run375/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run376/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run377/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run378/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run379/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run38/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run380/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run381/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run382/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run383/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run384/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run385/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run386/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run387/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run388/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run389/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run39/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run390/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run391/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run392/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run393/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run394/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run395/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run396/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run397/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run398/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run399/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run4/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run40/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run400/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run401/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run402/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run403/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run404/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run405/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run406/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run407/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run408/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run409/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run41/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run410/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run411/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run412/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run413/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run414/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run415/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run416/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run417/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run418/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run419/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run42/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run420/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run421/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run422/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run423/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run424/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run425/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run426/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run427/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run428/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run429/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run43/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run430/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run431/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run432/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run433/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run434/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run435/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run436/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run437/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run438/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run439/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run44/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run440/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run441/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run442/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run443/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run444/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run445/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run446/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run447/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run448/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run449/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run45/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run450/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run451/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run452/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run453/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run454/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run455/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run456/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run457/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run458/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run459/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run46/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run460/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run461/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run462/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run463/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run464/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run465/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run466/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run467/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run468/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run469/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run47/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run470/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run471/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run472/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run473/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run474/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run475/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run476/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run477/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run478/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run479/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run48/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run480/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run481/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run482/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run483/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run484/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run485/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run486/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run487/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run488/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run489/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run49/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run490/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run491/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run492/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run493/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run494/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run495/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run496/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run497/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run498/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run499/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run5/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run50/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run500/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run501/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run502/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run503/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run504/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run505/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run506/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run507/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run508/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run509/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run51/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run510/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run511/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run512/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run513/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run514/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run515/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run516/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run517/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run518/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run519/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run52/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run520/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run521/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run522/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run523/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run524/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run525/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run526/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run527/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run528/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run529/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run53/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run530/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run531/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run532/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run533/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run534/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run535/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run536/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run537/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run538/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run539/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run54/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run540/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run541/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run542/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run543/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run544/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run545/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run546/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run547/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run548/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run549/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run55/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run550/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run551/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run552/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run553/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run554/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run555/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run556/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run557/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run558/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run559/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run56/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run560/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run561/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run562/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run563/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run564/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run565/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run566/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run567/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run568/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run569/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run57/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run570/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run571/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run572/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run573/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run574/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run575/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run576/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run577/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run578/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run579/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run58/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run580/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run581/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run582/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run583/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run584/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run585/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run586/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run587/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run588/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run589/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run59/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run590/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run591/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run592/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run593/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run594/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run595/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run596/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run597/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run598/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run599/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run6/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run60/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run600/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run601/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run602/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run603/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run604/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run605/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run606/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run607/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run608/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run609/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run61/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run610/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run611/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run612/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run613/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run614/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run615/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run616/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run617/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run618/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run619/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run62/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run620/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run621/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run622/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run623/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run624/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run625/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run626/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run627/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run628/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run629/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run63/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run630/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run631/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run632/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run633/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run634/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run635/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run636/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run637/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run638/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run639/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run64/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run640/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run641/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run642/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run643/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run644/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run645/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run646/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run647/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run648/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run649/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run65/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run650/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run651/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run652/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run653/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run654/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run655/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run656/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run657/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run658/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run659/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run66/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run660/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run661/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run662/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run663/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run664/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run665/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run666/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run667/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run668/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run669/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run67/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run670/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run671/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run672/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run673/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run674/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run675/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run676/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run677/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run678/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run679/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run68/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run680/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run681/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run682/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run683/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run684/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run685/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run686/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run687/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run688/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run689/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run69/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run690/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run691/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run692/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run693/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run694/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run695/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run696/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run697/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run698/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run699/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run7/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run70/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run700/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run701/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run702/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run703/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run704/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run705/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run706/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run707/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run708/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run709/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run71/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run710/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run711/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run712/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run713/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run714/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run715/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run716/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run717/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run718/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run719/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run72/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run720/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run721/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run722/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run723/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run724/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run725/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run726/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run727/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run728/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run729/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run73/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run730/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run731/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run732/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run733/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run734/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run735/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run736/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run737/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run738/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run739/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run74/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run740/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run741/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run742/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run743/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run744/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run745/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run746/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run747/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run748/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run749/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run75/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run750/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run751/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run752/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run753/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run754/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run755/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run756/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run757/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run758/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run759/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run76/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run760/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run761/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run762/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run763/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run764/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run765/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run766/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run767/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run77/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run78/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run79/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run8/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run80/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run81/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run82/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run83/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run84/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run85/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run86/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run87/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run88/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run89/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run9/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run90/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run91/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run92/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run93/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run94/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run95/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run96/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run97/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run98/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_bnb4_test/Float32_run99/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulBnb4 |
| test/contrib_ops/matmul_integer16_test/MatMulInteger16_1_run0/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulInteger16 |
| test/contrib_ops/matmul_integer16_test/MatMulInteger16_2_run0/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulInteger16 |
| test/contrib_ops/matmul_integer16_test/MatMulInteger16_3_run0/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulInteger16 |
| test/contrib_ops/matmul_integer16_test/MatMulInteger16_Empty_input_run0/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulInteger16 |
| test/contrib_ops/matmul_integer_to_float_test/HasZeroPoint_HasBias_test_S8S8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/HasZeroPoint_HasBias_test_S8S8_run1/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/HasZeroPoint_HasBias_test_S8S8_run2/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/HasZeroPoint_HasBias_test_S8S8_run3/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/HasZeroPoint_HasBias_test_U8S8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/HasZeroPoint_HasBias_test_U8S8_run1/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/HasZeroPoint_HasBias_test_U8S8_run2/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/HasZeroPoint_HasBias_test_U8S8_run3/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/HasZeroPoint_HasBias_test_U8X8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/HasZeroPoint_HasBias_test_U8X8_run1/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/HasZeroPoint_HasBias_test_U8X8_run2/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/HasZeroPoint_HasBias_test_U8X8_run3/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/HasZeroPoint_NoBias_test_S8S8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/HasZeroPoint_NoBias_test_S8S8_run1/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/HasZeroPoint_NoBias_test_S8S8_run2/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/HasZeroPoint_NoBias_test_S8S8_run3/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/HasZeroPoint_NoBias_test_U8S8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/HasZeroPoint_NoBias_test_U8S8_run1/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/HasZeroPoint_NoBias_test_U8S8_run2/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/HasZeroPoint_NoBias_test_U8S8_run3/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/HasZeroPoint_NoBias_test_U8U8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/HasZeroPoint_NoBias_test_U8U8_run1/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/HasZeroPoint_NoBias_test_U8U8_run2/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/HasZeroPoint_NoBias_test_U8U8_run3/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/NoZeroPoint_HasBias_test_S8S8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/NoZeroPoint_HasBias_test_S8S8_run1/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/NoZeroPoint_HasBias_test_S8S8_run2/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/NoZeroPoint_HasBias_test_S8S8_run3/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/NoZeroPoint_HasBias_test_U8S8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/NoZeroPoint_HasBias_test_U8S8_run1/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/NoZeroPoint_HasBias_test_U8S8_run2/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/NoZeroPoint_HasBias_test_U8S8_run3/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/NoZeroPoint_HasBias_test_U8U8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/NoZeroPoint_HasBias_test_U8U8_run1/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/NoZeroPoint_HasBias_test_U8U8_run2/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/NoZeroPoint_HasBias_test_U8U8_run3/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/NoZeroPoint_NoBias_test_S8S8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/NoZeroPoint_NoBias_test_S8S8_run1/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/NoZeroPoint_NoBias_test_S8S8_run2/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/NoZeroPoint_NoBias_test_S8S8_run3/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/NoZeroPoint_NoBias_test_U8S8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/NoZeroPoint_NoBias_test_U8S8_run1/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/NoZeroPoint_NoBias_test_U8S8_run2/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/NoZeroPoint_NoBias_test_U8S8_run3/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/NoZeroPoint_NoBias_test_U8U8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/NoZeroPoint_NoBias_test_U8U8_run1/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/NoZeroPoint_NoBias_test_U8U8_run2/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulIntegerToFloat |
| test/contrib_ops/matmul_integer_to_float_test/NoZeroPoint_NoBias_test_U8U8_run3/model.onnx |  | Data/Data | ❌ | Unsupported op MatMulIntegerToFloat |
| test/contrib_ops/maxpool_mask_test/MaxPoolWithMask_run0/model.onnx |  | Data/Data | ❌ | Unsupported op MaxpoolWithMask |
| test/contrib_ops/moe_test/MoECpuTest_BasicSwiGLU_run0/model.onnx |  | Data/Data | ❌ | Unsupported op MoE |
| test/contrib_ops/multihead_attention_op_test/CrossAttentionWithPast_run0/model.onnx |  | Data/Data | ❌ | Unsupported op MultiHeadAttention |
| test/contrib_ops/multihead_attention_op_test/CrossAttentionWithPast_run1/model.onnx |  | Data/Data | ❌ | Unsupported op MultiHeadAttention |
| test/contrib_ops/multihead_attention_op_test/CrossAttention_Batch1_HeadSize16_run0/model.onnx |  | Data/Data | ❌ | Unsupported op MultiHeadAttention |
| test/contrib_ops/multihead_attention_op_test/CrossAttention_Batch1_HeadSize16_run1/model.onnx |  | Data/Data | ❌ | Unsupported op MultiHeadAttention |
| test/contrib_ops/multihead_attention_op_test/CrossAttention_Batch1_HeadSize16_run2/model.onnx |  | Data/Data | ❌ | Unsupported op MultiHeadAttention |
| test/contrib_ops/multihead_attention_op_test/CrossAttention_Batch1_HeadSize16_run3/model.onnx |  | Data/Data | ❌ | Unsupported op MultiHeadAttention |
| test/contrib_ops/multihead_attention_op_test/CrossAttention_Batch1_HeadSize8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op MultiHeadAttention |
| test/contrib_ops/multihead_attention_op_test/CrossAttention_Batch1_HeadSize8_run1/model.onnx |  | Data/Data | ❌ | Unsupported op MultiHeadAttention |
| test/contrib_ops/multihead_attention_op_test/CrossAttention_Batch2_HeadSize16_8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op MultiHeadAttention |
| test/contrib_ops/multihead_attention_op_test/CrossAttention_Batch2_HeadSize16_8_run1/model.onnx |  | Data/Data | ❌ | Unsupported op MultiHeadAttention |
| test/contrib_ops/multihead_attention_op_test/CrossAttention_Batch2_HeadSize16_8_run2/model.onnx |  | Data/Data | ❌ | Unsupported op MultiHeadAttention |
| test/contrib_ops/multihead_attention_op_test/CrossAttention_Batch2_HeadSize16_8_run3/model.onnx |  | Data/Data | ❌ | Unsupported op MultiHeadAttention |
| test/contrib_ops/multihead_attention_op_test/CrossAttention_Batch2_HeadSize40_run0/model.onnx |  | Data/Data | ❌ | Unsupported op MultiHeadAttention |
| test/contrib_ops/multihead_attention_op_test/CrossAttention_Batch2_HeadSize40_run1/model.onnx |  | Data/Data | ❌ | Unsupported op MultiHeadAttention |
| test/contrib_ops/multihead_attention_op_test/CrossAttention_Batch2_HeadSize40_run2/model.onnx |  | Data/Data | ❌ | Unsupported op MultiHeadAttention |
| test/contrib_ops/multihead_attention_op_test/CrossAttention_Batch2_HeadSize40_run3/model.onnx |  | Data/Data | ❌ | Unsupported op MultiHeadAttention |
| test/contrib_ops/multihead_attention_op_test/CrossAttention_DiffSequenceLengths_run0/model.onnx |  | Data/Data | ❌ | Unsupported op MultiHeadAttention |
| test/contrib_ops/multihead_attention_op_test/CrossAttention_DiffSequenceLengths_run1/model.onnx |  | Data/Data | ❌ | Unsupported op MultiHeadAttention |
| test/contrib_ops/multihead_attention_op_test/CrossAttention_DiffSequenceLengths_run2/model.onnx |  | Data/Data | ❌ | Unsupported op MultiHeadAttention |
| test/contrib_ops/multihead_attention_op_test/CrossAttention_DiffSequenceLengths_run3/model.onnx |  | Data/Data | ❌ | Unsupported op MultiHeadAttention |
| test/contrib_ops/multihead_attention_op_test/CrossAttention_DiffSequenceLengths_run4/model.onnx |  | Data/Data | ❌ | Unsupported op MultiHeadAttention |
| test/contrib_ops/multihead_attention_op_test/CrossAttention_DiffSequenceLengths_run5/model.onnx |  | Data/Data | ❌ | Unsupported op MultiHeadAttention |
| test/contrib_ops/multihead_attention_op_test/SelfAttention_WithPastAndPresent_NoMask_NoRelPosBias_run0/model.onnx |  | Data/Data | ❌ | Unsupported op MultiHeadAttention |
| test/contrib_ops/multihead_attention_op_test/SelfAttention_WithPastAndPresent_NoMask_NoRelPosBias_run1/model.onnx |  | Data/Data | ❌ | Unsupported op MultiHeadAttention |
| test/contrib_ops/multihead_attention_op_test/SelfAttention_WithPastAndPresent_NoMask_NoRelPosBias_run2/model.onnx |  | Data/Data | ❌ | Unsupported op MultiHeadAttention |
| test/contrib_ops/multihead_attention_op_test/SelfAttention_WithPastAndPresent_NoMask_NoRelPosBias_run3/model.onnx |  | Data/Data | ❌ | Unsupported op MultiHeadAttention |
| test/contrib_ops/multihead_attention_op_test/SelfAttention_WithPastAndPresent_NoMask_NoRelPosBias_run4/model.onnx |  | Data/Data | ❌ | Unsupported op MultiHeadAttention |
| test/contrib_ops/multihead_attention_op_test/SelfAttention_WithPastAndPresent_NoMask_NoRelPosBias_run5/model.onnx |  | Data/Data | ❌ | Unsupported op MultiHeadAttention |
| test/contrib_ops/murmur_hash3_test/DefaultSeed_run0/model.onnx |  | Data/Data | ❌ | Unsupported op MurmurHash3 |
| test/contrib_ops/murmur_hash3_test/MoreDataFloat_run0/model.onnx |  | Data/Data | ❌ | Unsupported op MurmurHash3 |
| test/contrib_ops/murmur_hash3_test/MoreDataInt_run0/model.onnx |  | Data/Data | ❌ | Unsupported op MurmurHash3 |
| test/contrib_ops/murmur_hash3_test/MultipleStringsKeyUIntResult_run0/model.onnx |  | Data/Data | ❌ | Unsupported op MurmurHash3 |
| test/contrib_ops/murmur_hash3_test/NonZeroSeedUIntResult_run0/model.onnx |  | Data/Data | ❌ | Unsupported op MurmurHash3 |
| test/contrib_ops/murmur_hash3_test/NonZeroSeed_run0/model.onnx |  | Data/Data | ❌ | Unsupported op MurmurHash3 |
| test/contrib_ops/murmur_hash3_test/StringKeyIntResult_run0/model.onnx |  | Data/Data | ❌ | Unsupported op MurmurHash3 |
| test/contrib_ops/murmur_hash3_test/StringKeyIntWithSeed42_run0/model.onnx |  | Data/Data | ❌ | Unsupported op MurmurHash3 |
| test/contrib_ops/murmur_hash3_test/StringKeyUIntResult_run0/model.onnx |  | Data/Data | ❌ | Unsupported op MurmurHash3 |
| test/contrib_ops/murmur_hash3_test/StringKeyUIntWithSeed42_run0/model.onnx |  | Data/Data | ❌ | Unsupported op MurmurHash3 |
| test/contrib_ops/murmur_hash3_test/UnsupportedInputType_run0/model.onnx |  | Data/Data | ❌ | Unsupported op MurmurHash3 |
| test/contrib_ops/murmur_hash3_test/ZeroSeedDoubleResult_run0/model.onnx |  | Data/Data | ❌ | Unsupported op MurmurHash3 |
| test/contrib_ops/murmur_hash3_test/ZeroSeedFloatResult_run0/model.onnx |  | Data/Data | ❌ | Unsupported op MurmurHash3 |
| test/contrib_ops/murmur_hash3_test/ZeroSeedUIntResult2_run0/model.onnx |  | Data/Data | ❌ | Unsupported op MurmurHash3 |
| test/contrib_ops/murmur_hash3_test/ZeroSeedUIntResult3_run0/model.onnx |  | Data/Data | ❌ | Unsupported op MurmurHash3 |
| test/contrib_ops/murmur_hash3_test/ZeroSeedUIntResult_run0/model.onnx |  | Data/Data | ❌ | Unsupported op MurmurHash3 |
| test/contrib_ops/murmur_hash3_test/ZeroSeed_run0/model.onnx |  | Data/Data | ❌ | Unsupported op MurmurHash3 |
| test/contrib_ops/ngram_repeat_block_op_test/NGramSize_3_run0/model.onnx |  | Data/Data | ❌ | Unsupported op NGramRepeatBlock |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run1/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run10/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run11/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run12/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run13/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run14/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run15/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run16/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run17/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run18/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run19/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run2/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run20/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run21/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run22/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run23/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run24/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run25/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run26/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run27/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run28/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run29/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run3/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run30/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run31/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run32/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run33/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run34/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run35/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run36/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run37/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run38/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run39/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run4/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run40/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run41/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run42/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run43/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run44/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run45/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run46/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run47/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run48/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run49/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run5/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run50/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run51/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run52/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run53/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run54/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run55/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run56/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run57/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run58/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run59/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run6/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run60/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run61/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run62/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run63/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run64/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run65/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run66/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run67/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run68/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run69/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run7/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run70/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run71/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run72/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run73/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run74/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run75/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run76/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run77/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run78/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run79/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run8/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run80/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run81/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run82/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run83/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run84/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run85/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run86/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run87/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run88/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run89/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run9/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run90/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run91/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_S8_run92/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run0/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run1/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run10/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run11/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run12/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run13/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run14/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run15/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run16/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run17/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run18/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run19/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run2/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run20/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run21/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run22/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run23/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run24/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run25/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run26/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run27/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run28/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run29/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run3/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run30/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run31/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run32/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run33/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run34/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run35/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run36/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run37/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run38/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run39/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run4/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run40/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run41/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run42/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run43/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run44/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run45/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run46/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run47/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run48/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run49/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run5/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run50/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run51/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run52/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run53/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run54/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run55/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run56/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run57/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run58/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run59/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run6/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run60/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run61/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run62/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run63/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run64/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run65/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run66/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run67/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run68/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run69/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run7/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run70/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run71/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run72/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run73/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run74/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run75/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run76/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run77/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run78/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run79/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run8/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run80/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run81/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run82/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run83/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run84/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run85/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run86/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run87/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run88/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run89/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run9/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run90/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run91/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool1D_run92/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run1/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run10/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run11/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run12/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run13/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run14/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run15/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run16/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run17/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run18/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run19/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run2/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run20/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run21/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run22/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run23/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run24/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run25/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run26/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run27/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run28/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run29/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run3/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run30/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run31/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run32/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run33/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run34/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run35/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run36/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run37/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run38/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run39/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run4/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run40/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run41/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run42/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run43/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run44/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run45/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run46/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run47/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run48/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run49/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run5/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run50/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run51/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run52/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run53/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run54/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run55/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run56/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run57/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run58/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run59/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run6/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run60/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run61/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run62/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run63/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run64/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run65/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run66/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run67/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run68/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run69/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run7/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run70/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run71/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run72/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run73/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run74/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run75/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run76/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run77/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run78/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run79/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run8/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run80/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run81/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run82/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run83/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run84/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run85/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run86/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run87/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run88/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run89/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run9/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run90/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run91/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_S8_run92/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run0/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run1/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run10/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run11/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run12/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run13/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run14/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run15/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run16/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run17/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run18/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run19/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run2/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run20/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run21/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run22/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run23/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run24/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run25/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run26/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run27/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run28/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run29/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run3/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run30/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run31/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run32/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run33/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run34/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run35/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run36/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run37/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run38/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run39/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run4/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run40/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run41/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run42/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run43/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run44/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run45/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run46/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run47/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run48/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run49/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run5/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run50/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run51/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run52/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run53/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run54/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run55/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run56/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run57/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run58/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run59/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run6/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run60/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run61/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run62/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run63/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run64/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run65/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run66/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run67/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run68/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run69/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run7/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run70/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run71/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run72/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run73/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run74/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run75/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run76/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run77/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run78/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run79/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run8/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run80/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run81/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run82/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run83/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run84/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run85/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run86/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run87/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run88/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run89/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run9/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run90/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run91/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool2D_run92/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run1/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run10/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run11/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run12/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run13/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run14/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run15/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run16/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run17/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run18/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run19/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run2/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run20/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run21/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run22/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run23/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run24/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run25/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run26/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run27/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run28/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run29/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run3/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run30/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run31/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run32/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run33/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run34/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run35/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run36/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run37/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run38/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run39/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run4/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run40/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run41/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run42/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run43/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run44/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run45/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run46/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run47/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run48/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run49/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run5/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run50/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run51/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run52/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run53/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run54/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run55/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run56/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run57/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run58/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run59/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run6/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run60/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run61/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run62/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run63/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run64/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run65/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run66/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run67/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run68/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run69/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run7/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run70/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run71/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run72/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run73/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run74/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run75/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run76/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run77/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run78/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run79/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run8/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run80/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run81/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run82/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run83/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run84/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run85/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run86/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run87/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run88/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run89/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run9/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run90/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run91/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_S8_run92/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run0/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run1/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run10/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run11/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run12/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run13/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run14/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run15/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run16/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run17/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run18/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run19/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run2/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run20/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run21/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run22/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run23/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run24/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run25/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run26/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run27/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run28/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run29/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run3/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run30/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run31/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run32/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run33/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run34/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run35/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run36/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run37/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run38/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run39/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run4/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run40/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run41/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run42/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run43/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run44/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run45/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run46/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run47/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run48/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run49/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run5/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run50/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run51/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run52/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run53/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run54/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run55/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run56/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run57/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run58/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run59/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run6/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run60/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run61/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run62/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run63/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run64/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run65/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run66/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run67/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run68/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run69/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run7/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run70/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run71/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run72/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run73/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run74/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run75/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run76/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run77/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run78/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run79/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run8/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run80/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run81/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run82/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run83/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run84/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run85/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run86/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run87/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run88/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run89/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run9/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run90/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run91/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPool3D_run92/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPoolDilations_S8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPoolDilations_run0/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPoolStrides_S8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/nhwc_maxpool_op_test/MaxPoolStrides_run0/model.onnx |  | Data/Data | ❌ | Unsupported op NhwcMaxPool |
| test/contrib_ops/qembed_layer_norm_op_test/EmbedLayerNormBatch1_Float16_run0/model.onnx |  | Data/Data | ❌ | Unsupported op QEmbedLayerNormalization |
| test/contrib_ops/qembed_layer_norm_op_test/EmbedLayerNormBatch1_Float16_run1/model.onnx |  | Data/Data | ❌ | Unsupported op QEmbedLayerNormalization |
| test/contrib_ops/qembed_layer_norm_op_test/EmbedLayerNormBatch1_run0/model.onnx |  | Data/Data | ❌ | Unsupported op QEmbedLayerNormalization |
| test/contrib_ops/qembed_layer_norm_op_test/EmbedLayerNormBatch1_run1/model.onnx |  | Data/Data | ❌ | Unsupported op QEmbedLayerNormalization |
| test/contrib_ops/qembed_layer_norm_op_test/EmbedLayerNormBatch2_NoMask_run0/model.onnx |  | Data/Data | ❌ | Unsupported op QEmbedLayerNormalization |
| test/contrib_ops/qembed_layer_norm_op_test/EmbedLayerNormBatch2_NoMask_run1/model.onnx |  | Data/Data | ❌ | Unsupported op QEmbedLayerNormalization |
| test/contrib_ops/qembed_layer_norm_op_test/EmbedLayerNormBatch2_run0/model.onnx |  | Data/Data | ❌ | Unsupported op QEmbedLayerNormalization |
| test/contrib_ops/qembed_layer_norm_op_test/EmbedLayerNormBatch2_run1/model.onnx |  | Data/Data | ❌ | Unsupported op QEmbedLayerNormalization |
| test/contrib_ops/qembed_layer_norm_op_test/EmbedLayerNormBatch_Distill_run0/model.onnx |  | Data/Data | ❌ | Unsupported op QEmbedLayerNormalization |
| test/contrib_ops/qembed_layer_norm_op_test/EmbedLayerNormBatch_Distill_run1/model.onnx |  | Data/Data | ❌ | Unsupported op QEmbedLayerNormalization |
| test/contrib_ops/qembed_layer_norm_op_test/EmbedLayerNormLargeBatchSmallHiddenSize_run0/model.onnx |  | Data/Data | ❌ | Unsupported op QEmbedLayerNormalization |
| test/contrib_ops/qembed_layer_norm_op_test/EmbedLayerNormLargeBatchSmallHiddenSize_run1/model.onnx |  | Data/Data | ❌ | Unsupported op QEmbedLayerNormalization |
| test/contrib_ops/qlinear_concat_test/ExpectFail_MixedConstDynamic_run0/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearConcat |
| test/contrib_ops/qlinear_concat_test/ExpectFail_MixedConstDynamic_run1/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearConcat |
| test/contrib_ops/qlinear_concat_test/ExpectFail_MixedConstDynamic_run2/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearConcat |
| test/contrib_ops/qlinear_concat_test/ExpectFail_MixedConstDynamic_run3/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearConcat |
| test/contrib_ops/qlinear_concat_test/ExpectFail_MixedConstDynamic_run4/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearConcat |
| test/contrib_ops/qlinear_concat_test/ExpectFail_MixedConstDynamic_run5/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearConcat |
| test/contrib_ops/qlinear_concat_test/ExpectFail_MixedConstDynamic_run6/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearConcat |
| test/contrib_ops/qlinear_concat_test/ExpectFail_MixedConstDynamic_run7/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearConcat |
| test/contrib_ops/qlinear_concat_test/ExpectFail_WrongScaleType_0_run0/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearConcat |
| test/contrib_ops/qlinear_concat_test/ExpectFail_WrongScaleType_0_run1/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearConcat |
| test/contrib_ops/qlinear_concat_test/ExpectFail_WrongScaleType_1_run0/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearConcat |
| test/contrib_ops/qlinear_concat_test/ExpectFail_WrongScaleType_1_run1/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearConcat |
| test/contrib_ops/qlinear_concat_test/ExpectFail_WrongTensorType_0_run0/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearConcat |
| test/contrib_ops/qlinear_concat_test/ExpectFail_WrongTensorType_0_run1/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearConcat |
| test/contrib_ops/qlinear_concat_test/ExpectFail_WrongTensorType_1_run0/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearConcat |
| test/contrib_ops/qlinear_concat_test/ExpectFail_WrongTensorType_1_run1/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearConcat |
| test/contrib_ops/qlinear_concat_test/ExpectFail_WrongZeroPointType_0_run0/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearConcat |
| test/contrib_ops/qlinear_concat_test/ExpectFail_WrongZeroPointType_0_run1/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearConcat |
| test/contrib_ops/qlinear_concat_test/ExpectFail_WrongZeroPointType_1_run0/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearConcat |
| test/contrib_ops/qlinear_concat_test/ExpectFail_WrongZeroPointType_1_run1/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearConcat |
| test/contrib_ops/qlinear_concat_test/Input3_ConstConstConst_run0/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearConcat |
| test/contrib_ops/qlinear_concat_test/Input3_ConstConstConst_run1/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearConcat |
| test/contrib_ops/qlinear_concat_test/Input3_DynamicDynamicDynamic_run0/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearConcat |
| test/contrib_ops/qlinear_concat_test/Input3_DynamicDynamicDynamic_run1/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearConcat |
| test/contrib_ops/qlinear_concat_test/Input3_MixedConstDynamic_run0/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearConcat |
| test/contrib_ops/qlinear_concat_test/Input3_MixedConstDynamic_run1/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearConcat |
| test/contrib_ops/qlinear_concat_test/Input3_MixedConstDynamic_run2/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearConcat |
| test/contrib_ops/qlinear_concat_test/Input3_MixedConstDynamic_run3/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearConcat |
| test/contrib_ops/qlinear_concat_test/Input3_MixedConstDynamic_run4/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearConcat |
| test/contrib_ops/qlinear_concat_test/Input3_MixedConstDynamic_run5/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearConcat |
| test/contrib_ops/qlinear_concat_test/Input3_MixedConstDynamic_run6/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearConcat |
| test/contrib_ops/qlinear_concat_test/Input3_MixedConstDynamic_run7/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearConcat |
| test/contrib_ops/qlinear_concat_test/InputOne_Const_run0/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearConcat |
| test/contrib_ops/qlinear_concat_test/InputOne_Const_run1/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearConcat |
| test/contrib_ops/qlinear_concat_test/InputOne_Dynamic_run0/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearConcat |
| test/contrib_ops/qlinear_concat_test/InputOne_Dynamic_run1/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearConcat |
| test/contrib_ops/qlinear_global_average_pool_test/Nchw_1x32x32x1_S8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearGlobalAveragePool |
| test/contrib_ops/qlinear_global_average_pool_test/Nchw_1x32x32x1_run0/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearGlobalAveragePool |
| test/contrib_ops/qlinear_global_average_pool_test/Nchw_1x7x7x255_S8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearGlobalAveragePool |
| test/contrib_ops/qlinear_global_average_pool_test/Nchw_1x7x7x255_run0/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearGlobalAveragePool |
| test/contrib_ops/qlinear_global_average_pool_test/Nchw_1x7x7x256_S8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearGlobalAveragePool |
| test/contrib_ops/qlinear_global_average_pool_test/Nchw_1x7x7x256_run0/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearGlobalAveragePool |
| test/contrib_ops/qlinear_global_average_pool_test/Nchw_1x8x8x255_S8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearGlobalAveragePool |
| test/contrib_ops/qlinear_global_average_pool_test/Nchw_1x8x8x255_run0/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearGlobalAveragePool |
| test/contrib_ops/qlinear_global_average_pool_test/Nchw_1x8x8x256_S8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearGlobalAveragePool |
| test/contrib_ops/qlinear_global_average_pool_test/Nchw_1x8x8x256_run0/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearGlobalAveragePool |
| test/contrib_ops/qlinear_global_average_pool_test/Nchw_3x7x7x255_S8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearGlobalAveragePool |
| test/contrib_ops/qlinear_global_average_pool_test/Nchw_3x7x7x255_run0/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearGlobalAveragePool |
| test/contrib_ops/qlinear_global_average_pool_test/Nchw_3x7x7x256_S8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearGlobalAveragePool |
| test/contrib_ops/qlinear_global_average_pool_test/Nchw_3x7x7x256_run0/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearGlobalAveragePool |
| test/contrib_ops/qlinear_global_average_pool_test/Nchw_3x8x8x255_S8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearGlobalAveragePool |
| test/contrib_ops/qlinear_global_average_pool_test/Nchw_3x8x8x255_run0/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearGlobalAveragePool |
| test/contrib_ops/qlinear_global_average_pool_test/Nchw_3x8x8x256_S8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearGlobalAveragePool |
| test/contrib_ops/qlinear_global_average_pool_test/Nchw_3x8x8x256_run0/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearGlobalAveragePool |
| test/contrib_ops/qlinear_global_average_pool_test/Nhwc_1x1x32x32_S8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearGlobalAveragePool |
| test/contrib_ops/qlinear_global_average_pool_test/Nhwc_1x1x32x32_run0/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearGlobalAveragePool |
| test/contrib_ops/qlinear_global_average_pool_test/Nhwc_1x255x7x7_S8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearGlobalAveragePool |
| test/contrib_ops/qlinear_global_average_pool_test/Nhwc_1x255x7x7_run0/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearGlobalAveragePool |
| test/contrib_ops/qlinear_global_average_pool_test/Nhwc_1x255x8x8_S8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearGlobalAveragePool |
| test/contrib_ops/qlinear_global_average_pool_test/Nhwc_1x255x8x8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearGlobalAveragePool |
| test/contrib_ops/qlinear_global_average_pool_test/Nhwc_1x256x7x7_S8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearGlobalAveragePool |
| test/contrib_ops/qlinear_global_average_pool_test/Nhwc_1x256x7x7_run0/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearGlobalAveragePool |
| test/contrib_ops/qlinear_global_average_pool_test/Nhwc_1x256x8x8_S8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearGlobalAveragePool |
| test/contrib_ops/qlinear_global_average_pool_test/Nhwc_1x256x8x8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearGlobalAveragePool |
| test/contrib_ops/qlinear_global_average_pool_test/Nhwc_3x255x7x7_S8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearGlobalAveragePool |
| test/contrib_ops/qlinear_global_average_pool_test/Nhwc_3x255x7x7_run0/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearGlobalAveragePool |
| test/contrib_ops/qlinear_global_average_pool_test/Nhwc_3x255x8x8_S8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearGlobalAveragePool |
| test/contrib_ops/qlinear_global_average_pool_test/Nhwc_3x255x8x8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearGlobalAveragePool |
| test/contrib_ops/qlinear_global_average_pool_test/Nhwc_3x256x7x7_S8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearGlobalAveragePool |
| test/contrib_ops/qlinear_global_average_pool_test/Nhwc_3x256x7x7_run0/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearGlobalAveragePool |
| test/contrib_ops/qlinear_global_average_pool_test/Nhwc_3x256x8x8_S8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearGlobalAveragePool |
| test/contrib_ops/qlinear_global_average_pool_test/Nhwc_3x256x8x8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearGlobalAveragePool |
| test/contrib_ops/qlinear_lookup_table_test/QLinearLeakyRelu_Int8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearLeakyRelu |
| test/contrib_ops/qlinear_lookup_table_test/QLinearLeakyRelu_UInt8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearLeakyRelu |
| test/contrib_ops/qlinear_lookup_table_test/QLinearSigmoid_Int8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearSigmoid |
| test/contrib_ops/qlinear_lookup_table_test/QLinearSigmoid_UInt8_0_Y_ZP_run0/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearSigmoid |
| test/contrib_ops/qlinear_lookup_table_test/QLinearSigmoid_UInt8_0_Y_ZP_run1/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearSigmoid |
| test/contrib_ops/qlinear_lookup_table_test/QLinearSigmoid_UInt8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearSigmoid |
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
| test/contrib_ops/qlinear_where_test/QLinearWhereMatrixAll_run0/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearWhere |
| test/contrib_ops/qlinear_where_test/QLinearWhereMatrixAll_run1/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearWhere |
| test/contrib_ops/qlinear_where_test/QLinearWhereScalarAll_run0/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearWhere |
| test/contrib_ops/qlinear_where_test/QLinearWhereScalarAll_run1/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearWhere |
| test/contrib_ops/qlinear_where_test/QLinearWhereScalarX_VectorY_MatrixCondition_run0/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearWhere |
| test/contrib_ops/qlinear_where_test/QLinearWhereScalarX_VectorY_MatrixCondition_run1/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearWhere |
| test/contrib_ops/qlinear_where_test/QLinearWhereVectorAll_run0/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearWhere |
| test/contrib_ops/qlinear_where_test/QLinearWhereVectorAll_run1/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearWhere |
| test/contrib_ops/qlinear_where_test/QLinearWhereVectorX_VectorY_MatrixCondition_run0/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearWhere |
| test/contrib_ops/qlinear_where_test/QLinearWhereVectorX_VectorY_MatrixCondition_run1/model.onnx |  | Data/Data | ❌ | Unsupported op QLinearWhere |
| test/contrib_ops/quant_gemm_test/GEMM_run0/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run1/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run10/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run100/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run101/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run102/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run103/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run104/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run105/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run106/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run107/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run108/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run109/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run11/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run110/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run111/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run112/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run113/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run114/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run115/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run116/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run117/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run118/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run119/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run12/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run120/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run121/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run122/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run123/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run124/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run125/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run126/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run127/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run128/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run129/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run13/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run130/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run131/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run132/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run133/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run134/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run135/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run136/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run137/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run138/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run139/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run14/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run140/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run141/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run142/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run143/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run144/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run145/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run146/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run147/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run148/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run149/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run15/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run150/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run151/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run152/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run153/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run154/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run155/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run156/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run157/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run158/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run159/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run16/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run160/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run161/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run162/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run163/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run164/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run165/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run166/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run167/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run168/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run169/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run17/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run170/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run171/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run172/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run173/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run174/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run175/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run176/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run177/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run178/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run179/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run18/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run180/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run181/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run182/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run183/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run184/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run185/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run186/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run187/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run188/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run189/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run19/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run190/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run191/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run2/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run20/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run21/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run22/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run23/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run24/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run25/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run26/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run27/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run28/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run29/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run3/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run30/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run31/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run32/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run33/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run34/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run35/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run36/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run37/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run38/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run39/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run4/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run40/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run41/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run42/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run43/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run44/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run45/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run46/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run47/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run48/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run49/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run5/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run50/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run51/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run52/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run53/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run54/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run55/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run56/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run57/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run58/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run59/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run6/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run60/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run61/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run62/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run63/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run64/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run65/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run66/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run67/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run68/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run69/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run7/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run70/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run71/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run72/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run73/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run74/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run75/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run76/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run77/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run78/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run79/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run8/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run80/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run81/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run82/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run83/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run84/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run85/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run86/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run87/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run88/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run89/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run9/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run90/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run91/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run92/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run93/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run94/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run95/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run96/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run97/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run98/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMM_run99/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run0/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run1/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run10/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run100/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run101/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run102/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run103/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run104/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run105/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run106/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run107/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run108/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run109/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run11/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run110/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run111/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run112/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run113/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run114/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run115/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run116/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run117/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run118/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run119/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run12/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run120/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run121/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run122/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run123/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run124/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run125/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run126/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run127/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run128/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run129/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run13/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run130/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run131/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run132/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run133/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run134/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run135/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run136/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run137/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run138/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run139/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run14/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run140/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run141/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run142/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run143/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run144/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run145/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run146/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run147/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run148/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run149/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run15/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run150/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run151/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run152/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run153/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run154/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run155/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run156/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run157/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run158/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run159/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run16/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run160/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run161/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run162/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run163/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run164/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run165/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run166/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run167/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run168/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run169/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run17/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run170/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run171/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run172/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run173/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run174/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run175/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run176/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run177/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run178/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run179/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run18/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run180/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run181/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run182/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run183/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run184/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run185/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run186/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run187/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run188/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run189/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run19/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run190/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run191/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run192/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run193/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run194/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run195/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run196/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run197/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run198/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run199/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run2/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run20/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run200/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run201/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run202/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run203/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run204/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run205/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run206/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run207/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run208/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run209/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run21/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run210/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run211/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run212/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run213/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run214/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run215/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run216/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run217/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run218/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run219/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run22/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run220/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run221/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run222/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run223/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run224/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run225/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run226/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run227/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run228/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run229/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run23/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run230/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run231/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run232/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run233/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run234/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run235/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run236/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run237/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run238/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run239/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run24/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run240/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run241/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run242/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run243/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run244/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run245/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run246/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run247/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run248/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run249/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run25/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run250/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run251/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run252/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run253/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run254/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run255/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run256/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run257/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run258/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run259/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run26/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run260/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run261/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run262/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run263/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run264/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run265/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run266/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run267/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run268/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run269/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run27/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run270/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run271/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run272/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run273/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run274/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run275/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run276/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run277/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run278/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run279/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run28/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run280/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run281/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run282/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run283/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run284/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run285/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run286/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run287/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run29/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run3/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run30/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run31/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run32/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run33/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run34/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run35/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run36/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run37/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run38/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run39/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run4/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run40/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run41/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run42/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run43/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run44/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run45/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run46/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run47/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run48/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run49/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run5/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run50/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run51/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run52/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run53/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run54/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run55/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run56/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run57/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run58/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run59/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run6/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run60/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run61/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run62/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run63/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run64/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run65/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run66/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run67/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run68/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run69/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run7/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run70/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run71/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run72/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run73/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run74/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run75/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run76/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run77/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run78/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run79/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run8/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run80/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run81/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run82/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run83/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run84/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run85/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run86/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run87/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run88/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run89/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run9/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run90/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run91/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run92/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run93/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run94/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run95/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run96/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run97/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run98/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/GEMV_run99/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run0/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run1/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run10/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run100/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run101/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run102/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run103/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run104/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run105/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run106/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run107/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run108/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run109/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run11/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run110/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run111/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run112/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run113/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run114/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run115/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run116/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run117/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run118/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run119/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run12/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run120/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run121/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run122/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run123/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run124/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run125/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run126/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run127/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run128/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run129/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run13/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run130/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run131/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run132/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run133/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run134/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run135/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run136/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run137/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run138/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run139/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run14/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run140/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run141/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run142/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run143/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run15/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run16/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run17/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run18/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run19/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run2/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run20/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run21/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run22/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run23/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run24/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run25/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run26/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run27/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run28/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run29/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run3/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run30/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run31/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run32/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run33/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run34/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run35/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run36/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run37/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run38/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run39/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run4/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run40/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run41/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run42/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run43/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run44/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run45/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run46/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run47/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run48/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run49/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run5/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run50/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run51/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run52/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run53/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run54/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run55/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run56/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run57/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run58/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run59/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run6/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run60/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run61/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run62/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run63/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run64/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run65/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run66/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run67/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run68/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run69/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run7/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run70/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run71/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run72/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run73/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run74/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run75/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run76/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run77/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run78/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run79/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run8/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run80/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run81/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run82/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run83/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run84/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run85/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run86/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run87/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run88/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run89/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run9/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run90/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run91/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run92/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run93/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run94/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run95/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run96/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run97/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run98/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quant_gemm_test/Scalar_run99/model.onnx |  | Data/Data | ❌ | Unsupported op QGemm |
| test/contrib_ops/quantize_attention_op_test/QAttentionBatch1_run0/model.onnx |  | Data/Data | ❌ | Unsupported op QAttention |
| test/contrib_ops/quantize_attention_op_test/QAttentionBatch1_run1/model.onnx |  | Data/Data | ❌ | Unsupported op QAttention |
| test/contrib_ops/quantize_attention_op_test/QAttentionBatch2_run0/model.onnx |  | Data/Data | ❌ | Unsupported op QAttention |
| test/contrib_ops/quantize_attention_op_test/QAttentionBatch2_run1/model.onnx |  | Data/Data | ❌ | Unsupported op QAttention |
| test/contrib_ops/quantize_attention_op_test/QAttentionMaskExceedSequence_run0/model.onnx |  | Data/Data | ❌ | Unsupported op QAttention |
| test/contrib_ops/quantize_attention_op_test/QAttentionMaskExceedSequence_run1/model.onnx |  | Data/Data | ❌ | Unsupported op QAttention |
| test/contrib_ops/quantize_attention_op_test/QAttentionMaskPartialSequence_run0/model.onnx |  | Data/Data | ❌ | Unsupported op QAttention |
| test/contrib_ops/quantize_attention_op_test/QAttentionMaskPartialSequence_run1/model.onnx |  | Data/Data | ❌ | Unsupported op QAttention |
| test/contrib_ops/quantize_attention_op_test/QAttentionNoMaskIndex_run0/model.onnx |  | Data/Data | ❌ | Unsupported op QAttention |
| test/contrib_ops/quantize_attention_op_test/QAttentionNoMaskIndex_run1/model.onnx |  | Data/Data | ❌ | Unsupported op QAttention |
| test/contrib_ops/quantize_attention_op_test/QAttentionPastState_u8s8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op QAttention |
| test/contrib_ops/quantize_attention_op_test/QAttentionPastState_u8s8_run1/model.onnx |  | Data/Data | ❌ | Unsupported op QAttention |
| test/contrib_ops/quantize_attention_op_test/QAttentionPastState_u8s8_run2/model.onnx |  | Data/Data | ❌ | Unsupported op QAttention |
| test/contrib_ops/quantize_attention_op_test/QAttentionPastState_u8s8_run3/model.onnx |  | Data/Data | ❌ | Unsupported op QAttention |
| test/contrib_ops/quantize_attention_op_test/QAttentionPastState_u8u8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op QAttention |
| test/contrib_ops/quantize_attention_op_test/QAttentionPastState_u8u8_run1/model.onnx |  | Data/Data | ❌ | Unsupported op QAttention |
| test/contrib_ops/quantize_attention_op_test/QAttentionPastState_u8u8_run2/model.onnx |  | Data/Data | ❌ | Unsupported op QAttention |
| test/contrib_ops/quantize_attention_op_test/QAttentionPastState_u8u8_run3/model.onnx |  | Data/Data | ❌ | Unsupported op QAttention |
| test/contrib_ops/quantize_attention_op_test/QAttentionPrunedModel_run0/model.onnx |  | Data/Data | ❌ | Unsupported op QAttention |
| test/contrib_ops/quantize_attention_op_test/QAttentionPrunedModel_run1/model.onnx |  | Data/Data | ❌ | Unsupported op QAttention |
| test/contrib_ops/quantize_attention_op_test/QAttentionUnidirectional_U8S8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op QAttention |
| test/contrib_ops/quantize_attention_op_test/QAttentionUnidirectional_U8U8_run0/model.onnx |  | Data/Data | ❌ | Unsupported op QAttention |
| test/contrib_ops/quantize_attention_op_test/SharedPrepackedWeights_run0/model.onnx |  | Data/Data | ❌ | Unsupported op QAttention |
| test/contrib_ops/quantize_attention_op_test/SharedPrepackedWeights_run1/model.onnx |  | Data/Data | ❌ | Unsupported op QAttention |
| test/contrib_ops/quantize_lstm_op_test/LargeSize_run0/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 1979397945) |
| test/contrib_ops/quantize_lstm_op_test/LargeSize_run1/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 1981732033) |
| test/contrib_ops/quantize_lstm_op_test/LargeSize_run10/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 2003232705) |
| test/contrib_ops/quantize_lstm_op_test/LargeSize_run11/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 1992534977) |
| test/contrib_ops/quantize_lstm_op_test/LargeSize_run12/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 38208390) |
| test/contrib_ops/quantize_lstm_op_test/LargeSize_run13/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 1985478361) |
| test/contrib_ops/quantize_lstm_op_test/LargeSize_run14/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 38208390) |
| test/contrib_ops/quantize_lstm_op_test/LargeSize_run15/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 1985478361) |
| test/contrib_ops/quantize_lstm_op_test/LargeSize_run16/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 2003150481) |
| test/contrib_ops/quantize_lstm_op_test/LargeSize_run17/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 1991509409) |
| test/contrib_ops/quantize_lstm_op_test/LargeSize_run18/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 2003150481) |
| test/contrib_ops/quantize_lstm_op_test/LargeSize_run19/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 1991509409) |
| test/contrib_ops/quantize_lstm_op_test/LargeSize_run2/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 1979397945) |
| test/contrib_ops/quantize_lstm_op_test/LargeSize_run20/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 1989256941) |
| test/contrib_ops/quantize_lstm_op_test/LargeSize_run21/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 1973217073) |
| test/contrib_ops/quantize_lstm_op_test/LargeSize_run22/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 1989256941) |
| test/contrib_ops/quantize_lstm_op_test/LargeSize_run23/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 1973217073) |
| test/contrib_ops/quantize_lstm_op_test/LargeSize_run3/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 1981732033) |
| test/contrib_ops/quantize_lstm_op_test/LargeSize_run4/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 1953277249) |
| test/contrib_ops/quantize_lstm_op_test/LargeSize_run5/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 1948480513) |
| test/contrib_ops/quantize_lstm_op_test/LargeSize_run6/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 1953277249) |
| test/contrib_ops/quantize_lstm_op_test/LargeSize_run7/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 1948480513) |
| test/contrib_ops/quantize_lstm_op_test/LargeSize_run8/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 2003232705) |
| test/contrib_ops/quantize_lstm_op_test/LargeSize_run9/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 1992534977) |
| test/contrib_ops/quantize_lstm_op_test/SharedPrepackedWeights_run0/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 429391) |
| test/contrib_ops/quantize_lstm_op_test/SharedPrepackedWeights_run1/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 429391) |
| test/contrib_ops/quantize_lstm_op_test/SmallSize_run0/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 429391) |
| test/contrib_ops/quantize_lstm_op_test/SmallSize_run1/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 792871) |
| test/contrib_ops/quantize_lstm_op_test/SmallSize_run10/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 549012) |
| test/contrib_ops/quantize_lstm_op_test/SmallSize_run11/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 795324) |
| test/contrib_ops/quantize_lstm_op_test/SmallSize_run12/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 358217) |
| test/contrib_ops/quantize_lstm_op_test/SmallSize_run13/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 1115054) |
| test/contrib_ops/quantize_lstm_op_test/SmallSize_run14/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 358217) |
| test/contrib_ops/quantize_lstm_op_test/SmallSize_run15/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 1115054) |
| test/contrib_ops/quantize_lstm_op_test/SmallSize_run16/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 651496) |
| test/contrib_ops/quantize_lstm_op_test/SmallSize_run17/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 796392) |
| test/contrib_ops/quantize_lstm_op_test/SmallSize_run18/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 651496) |
| test/contrib_ops/quantize_lstm_op_test/SmallSize_run19/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 796392) |
| test/contrib_ops/quantize_lstm_op_test/SmallSize_run2/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 429391) |
| test/contrib_ops/quantize_lstm_op_test/SmallSize_run20/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 351903) |
| test/contrib_ops/quantize_lstm_op_test/SmallSize_run21/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 1296640) |
| test/contrib_ops/quantize_lstm_op_test/SmallSize_run22/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 351903) |
| test/contrib_ops/quantize_lstm_op_test/SmallSize_run23/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 1296640) |
| test/contrib_ops/quantize_lstm_op_test/SmallSize_run3/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 792871) |
| test/contrib_ops/quantize_lstm_op_test/SmallSize_run4/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 707705) |
| test/contrib_ops/quantize_lstm_op_test/SmallSize_run5/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 3248832) |
| test/contrib_ops/quantize_lstm_op_test/SmallSize_run6/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 707705) |
| test/contrib_ops/quantize_lstm_op_test/SmallSize_run7/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 3248832) |
| test/contrib_ops/quantize_lstm_op_test/SmallSize_run8/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 549012) |
| test/contrib_ops/quantize_lstm_op_test/SmallSize_run9/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 795324) |
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
| test/contrib_ops/sample_op_test/SampleOpFloat_run0/model.onnx |  | Data/Data | ❌ | Unsupported op SampleOp |
| test/contrib_ops/skiplayernorm_op_test/SkipLayerNormBatch1_NoBeta_run0/model.onnx |  | Data/Data | ❌ | Unsupported op SkipLayerNormalization |
| test/contrib_ops/skiplayernorm_op_test/SkipLayerNormBatch1_NoBeta_run1/model.onnx |  | Data/Data | ❌ | Unsupported op SkipLayerNormalization |
| test/contrib_ops/skiplayernorm_op_test/SkipLayerNormBatch1_run0/model.onnx |  | Data/Data | ❌ | Unsupported op SkipLayerNormalization |
| test/contrib_ops/skiplayernorm_op_test/SkipLayerNormBatch1_run1/model.onnx |  | Data/Data | ❌ | Unsupported op SkipLayerNormalization |
| test/contrib_ops/skiplayernorm_op_test/SkipLayerNormBatch2_Bias_ProducingOptionalOutput_run0/model.onnx |  | Data/Data | ❌ | Unsupported op SkipLayerNormalization |
| test/contrib_ops/skiplayernorm_op_test/SkipLayerNormBatch2_Bias_ProducingOptionalOutput_run1/model.onnx |  | Data/Data | ❌ | Unsupported op SkipLayerNormalization |
| test/contrib_ops/skiplayernorm_op_test/SkipLayerNormBatch2_Bias_run0/model.onnx |  | Data/Data | ❌ | Unsupported op SkipLayerNormalization |
| test/contrib_ops/skiplayernorm_op_test/SkipLayerNormBatch2_Bias_run1/model.onnx |  | Data/Data | ❌ | Unsupported op SkipLayerNormalization |
| test/contrib_ops/skiplayernorm_op_test/SkipLayerNormBatch2_Skip_Broadcast_Batch_Size_1_run0/model.onnx |  | Data/Data | ❌ | Unsupported op SkipLayerNormalization |
| test/contrib_ops/skiplayernorm_op_test/SkipLayerNormBatch2_Skip_Broadcast_No_Batch_Size_run0/model.onnx |  | Data/Data | ❌ | Unsupported op SkipLayerNormalization |
| test/contrib_ops/skiplayernorm_op_test/SkipLayerNormBatch2_TokenCount_run0/model.onnx |  | Data/Data | ❌ | Unsupported op SkipLayerNormalization |
| test/contrib_ops/skiplayernorm_op_test/SkipLayerNormBatch2_TokenCount_run1/model.onnx |  | Data/Data | ❌ | Unsupported op SkipLayerNormalization |
| test/contrib_ops/skiplayernorm_op_test/SkipLayerNormBatch2_run0/model.onnx |  | Data/Data | ❌ | Unsupported op SkipLayerNormalization |
| test/contrib_ops/skiplayernorm_op_test/SkipLayerNormBatch2_run1/model.onnx |  | Data/Data | ❌ | Unsupported op SkipLayerNormalization |
| test/contrib_ops/skiplayernorm_op_test/SkipLayerNormNullInput_run0/model.onnx |  | Data/Data | ❌ | Unsupported op SkipLayerNormalization |
| test/contrib_ops/skiplayernorm_op_test/SkipLayerNormNullInput_run1/model.onnx |  | Data/Data | ❌ | Unsupported op SkipLayerNormalization |
| test/contrib_ops/skiplayernorm_op_test/SkipLayerNormPrePack_run0/model.onnx |  | Data/Data | ❌ | Unsupported op SkipLayerNormalization |
| test/contrib_ops/tensor_op_test/CropBorderAndScale_run0/model.onnx | 7 | Data/Data | ❌ | Unsupported op Crop |
| test/contrib_ops/tensor_op_test/CropBorderOnly_run0/model.onnx | 7 | Data/Data | ❌ | Unsupported op Crop |
| test/contrib_ops/tensor_op_test/ImageScalerTest_run0/model.onnx | 7 | Data/Data | ❌ | Unsupported op ImageScaler |
| test/contrib_ops/tensor_op_test/LastDim_run0/model.onnx |  | Data/Data | ❌ | Unsupported op UnfoldTensor |
| test/contrib_ops/tensor_op_test/MeanVarianceNormalizationCPUTest_Version1_TO_8_run0/model.onnx | 7 | Data/Data | ❌ | Out of tolerance (max ULP 7137119) |
| test/contrib_ops/tensor_op_test/MeanVarianceNormalizationCPUTest_Version1_TO_8_run1/model.onnx | 7 | Data/Data | ❌ | Out of tolerance (max ULP 21399263) |
| test/contrib_ops/tensor_op_test/MeanVarianceNormalizationCPUTest_Version1_TO_8_run2/model.onnx | 7 | Data/Data | ❌ | Out of tolerance (max ULP 15806857) |
| test/contrib_ops/tensor_op_test/NormalDim_run0/model.onnx |  | Data/Data | ❌ | Unsupported op UnfoldTensor |
| test/contrib_ops/tokenizer_test/TokenizerCharLevel_CyrillicCharsWithMarkersC_run0/model.onnx |  | Data/Data | ❌ | Unsupported op Tokenizer |
| test/contrib_ops/tokenizer_test/TokenizerCharLevel_EmptyOutputC_run0/model.onnx |  | Data/Data | ❌ | Unsupported op Tokenizer |
| test/contrib_ops/tokenizer_test/TokenizerCharLevel_EmptyOutputNC_run0/model.onnx |  | Data/Data | ❌ | Unsupported op Tokenizer |
| test/contrib_ops/tokenizer_test/TokenizerCharLevel_LatinCharsNoMarkersC_run0/model.onnx |  | Data/Data | ❌ | Unsupported op Tokenizer |
| test/contrib_ops/tokenizer_test/TokenizerCharLevel_LatinCharsNoMarkersNC_run0/model.onnx |  | Data/Data | ❌ | Unsupported op Tokenizer |
| test/contrib_ops/tokenizer_test/TokenizerCharLevel_LatinCharsWithMarkersC_run0/model.onnx |  | Data/Data | ❌ | Unsupported op Tokenizer |
| test/contrib_ops/tokenizer_test/TokenizerCharLevel_LatinCharsWithMarkersNC_run0/model.onnx |  | Data/Data | ❌ | Unsupported op Tokenizer |
| test/contrib_ops/tokenizer_test/TokenizerCharLevel_MixedCharsWithMarkersC_run0/model.onnx |  | Data/Data | ❌ | Unsupported op Tokenizer |
| test/contrib_ops/tokenizer_test/TokenizerExpression_Grouping_run0/model.onnx |  | Data/Data | ❌ | Unsupported op Tokenizer |
| test/contrib_ops/tokenizer_test/TokenizerExpression_RegChar_run0/model.onnx |  | Data/Data | ❌ | Unsupported op Tokenizer |
| test/contrib_ops/tokenizer_test/TokenizerExpression_RegDot_run0/model.onnx |  | Data/Data | ❌ | Unsupported op Tokenizer |
| test/contrib_ops/tokenizer_test/TokenizerExpression_RegEx_run0/model.onnx |  | Data/Data | ❌ | Unsupported op Tokenizer |
| test/contrib_ops/tokenizer_test/TokenizerExpression_RegRep_run0/model.onnx |  | Data/Data | ❌ | Unsupported op Tokenizer |
| test/contrib_ops/tokenizer_test/TokenizerWithSeparators_MixCharCommonPrefixC_run0/model.onnx |  | Data/Data | ❌ | Unsupported op Tokenizer |
| test/contrib_ops/tokenizer_test/TokenizerWithSeparators_MixCharsNoMarkersSeparatorsOverlapLongFirstC_run0/model.onnx |  | Data/Data | ❌ | Unsupported op Tokenizer |
| test/contrib_ops/tokenizer_test/TokenizerWithSeparators_MixCharsNoMarkersSeparatorsOverlapLongFirstRepeatedShortC_run0/model.onnx |  | Data/Data | ❌ | Unsupported op Tokenizer |
| test/contrib_ops/tokenizer_test/TokenizerWithSeparators_MixCharsNoMarkersSeparatorsOverlapShortFirstC_run0/model.onnx |  | Data/Data | ❌ | Unsupported op Tokenizer |
| test/contrib_ops/tokenizer_test/TokenizerWithSeparators_MixCharsNoMarkersSeparatorsOverlapingMatchC_run0/model.onnx |  | Data/Data | ❌ | Unsupported op Tokenizer |
| test/contrib_ops/tokenizer_test/TokenizerWithSeparators_MixCharsWithMarkersC_run0/model.onnx |  | Data/Data | ❌ | Unsupported op Tokenizer |
| test/contrib_ops/tokenizer_test/TokenizerWithSeparators_MixCharsWithMarkersCompleteMatchEmptyOutputC_run0/model.onnx |  | Data/Data | ❌ | Unsupported op Tokenizer |
| test/contrib_ops/tokenizer_test/TokenizerWithSeparators_MixCharsWithMarkersEmptyInputEmptyOutputC_run0/model.onnx |  | Data/Data | ❌ | Unsupported op Tokenizer |
| test/contrib_ops/tokenizer_test/TokenizerWithSeparators_MixCharsWithMarkersEmptyInputEmptyOutputNC_run0/model.onnx |  | Data/Data | ❌ | Unsupported op Tokenizer |
| test/contrib_ops/tokenizer_test/TokenizerWithSeparators_MixCharsWithMarkersEndMatchAtLeast4CharsC_run0/model.onnx |  | Data/Data | ❌ | Unsupported op Tokenizer |
| test/contrib_ops/tokenizer_test/TokenizerWithSeparators_MixCharsWithMarkersEndMatchC_run0/model.onnx |  | Data/Data | ❌ | Unsupported op Tokenizer |
| test/contrib_ops/tokenizer_test/TokenizerWithSeparators_MixCharsWithMarkersStartMatchC_run0/model.onnx |  | Data/Data | ❌ | Unsupported op Tokenizer |
| test/contrib_ops/tokenizer_test/Tokenizer_EmptyInput_run0/model.onnx |  | Data/Data | ❌ | Unsupported op Tokenizer |
| test/contrib_ops/tokenizer_test/Tokenizer_EmptyInput_run1/model.onnx |  | Data/Data | ❌ | Unsupported op Tokenizer |
| test/contrib_ops/tokenizer_test/Tokenizer_EmptyInput_run2/model.onnx |  | Data/Data | ❌ | Unsupported op Tokenizer |
| test/contrib_ops/unique_op_test/Unique_AllUniqueElements_run0/model.onnx |  | Data/Data | ❌ | Unique must have 1 input and 4 outputs |
| test/contrib_ops/unique_op_test/Unique_Complicated_Example_run0/model.onnx |  | Data/Data | ❌ | Unique must have 1 input and 4 outputs |
| test/contrib_ops/unique_op_test/Unique_Example_SingleElement_run0/model.onnx |  | Data/Data | ❌ | Unique must have 1 input and 4 outputs |
| test/contrib_ops/unique_op_test/Unique_Spec_Example_run0/model.onnx |  | Data/Data | ❌ | Unique must have 1 input and 4 outputs |
| test/contrib_ops/word_conv_embedding_test/WordConvEmbedding_char_embedding_shape_conv_shape_not_match_run0/model.onnx |  | Data/Data | ❌ | Unsupported op WordConvEmbedding |
| test/contrib_ops/word_conv_embedding_test/WordConvEmbedding_char_embedding_size_mismatch_run0/model.onnx |  | Data/Data | ❌ | Unsupported op WordConvEmbedding |
| test/contrib_ops/word_conv_embedding_test/WordConvEmbedding_conv_window_size_mismatch_run0/model.onnx |  | Data/Data | ❌ | Unsupported op WordConvEmbedding |
| test/contrib_ops/word_conv_embedding_test/WordConvEmbedding_embedding_size_mismatch_run0/model.onnx |  | Data/Data | ❌ | Unsupported op WordConvEmbedding |
| test/contrib_ops/word_conv_embedding_test/WordConvEmbedding_run0/model.onnx |  | Data/Data | ❌ | Unsupported op WordConvEmbedding |
| test/contrib_ops/word_conv_embedding_test/WordConvEmbedding_valid_attribute_run0/model.onnx |  | Data/Data | ❌ | Unsupported op WordConvEmbedding |

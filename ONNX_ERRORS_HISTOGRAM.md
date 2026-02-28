# Error frequency

This histogram is test-suite-overarching.

| Error message | Count | Opset versions |
| --- | --- | --- |
| Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor '*'. | 22 | 25 |
| Unsupported elem_type 19 (FLOAT8E5M2) for tensor '*'. | 20 | 25 |
| Unsupported elem_type 18 (FLOAT8E4M3FNUZ) for tensor '*'. | 18 | 25 |
| Unsupported elem_type 20 (FLOAT8E5M2FNUZ) for tensor '*'. | 18 | 25 |
| Unsupported elem_type 21 (UINT4) for tensor '*'. | 17 | 25 |
| Unsupported elem_type 22 (INT4) for tensor '*'. | 17 | 25 |
| Unsupported elem_type 25 (UINT2) for tensor '*'. | 17 | 25 |
| Unsupported elem_type 26 (INT2) for tensor '*'. | 17 | 25 |
| Unsupported elem_type 23 (FLOAT4E2M1) for tensor '*'. | 14 | 25 |
| Out of tolerance | 9 | 8, 9, 15, 19, 22 |
| Unsupported op ImageDecoder | 9 | 20 |
| Dropout supports only the data input and 1 or 2 outputs | 8 | 22 |
| Unsupported op Loop | 7 | 16, 17 |
| Unsupported op CenterCropPad | 6 | 18 |
| Unsupported op DFT | 6 | 19, 20 |
| Unsupported op ScatterElements | 6 | 18 |
| Unsupported op SequenceMap | 6 | 17 |
| Unsupported op StringSplit | 6 | 20 |
| Unsupported op Col2Im | 5 | 18 |
| Unsupported op StringConcat | 5 | 20 |
| OptionalHasElement expects exactly one non-empty input. | 4 | 18 |
| Unsupported elem_type 24 (FLOAT8E8M0) for tensor '*'. | 4 | 25 |
| Unsupported op AffineGrid | 4 | 20 |
| Unsupported op DeformConv | 4 | 22 |
| Unsupported op LabelEncoder | 4 |  |
| Unsupported op OptionalGetElement | 4 | 18 |
| Unsupported op RNN | 4 | 22 |
| HardSigmoid only supports alpha=0.2 | 3 | 22 |
| Unsupported op Momentum | 3 |  |
| Unsupported op RandomUniformLike | 3 | 22 |
| Unsupported op RegexFullMatch | 3 | 20 |
| Unsupported op RoiAlign | 3 | 22 |
| Arrays are not equal (max abs diff 148) | 2 | 21 |
| Cast input and output shapes must match | 2 | 22 |
| Gelu only supports approximate=none | 2 | 20 |
| LpPool expects 2D kernel_shape | 2 | 22 |
| LpPool supports auto_pad=NOTSET only | 2 | 22 |
| QuantizeLinear block_size is not supported | 2 | 25 |
| Selu only supports alpha=1.6732632423543772 | 2 | 22 |
| Split output shape must be (1,), got (2,) | 2 | 20 |
| ThresholdedRelu only supports alpha=1.0 | 2 | 22 |
| Unsupported non-tensor value '*' in op Identity. | 2 | 16, 25 |
| Unsupported op Adam | 2 |  |
| Unsupported op BitwiseNot | 2 | 18 |
| Unsupported op BlackmanWindow | 2 | 17 |
| Unsupported op Det | 2 | 22 |
| Unsupported op HannWindow | 2 | 17 |
| Unsupported op MaxUnpool | 2 | 22 |
| Unsupported op STFT | 2 | 17 |
| Unsupported op TreeEnsemble | 2 |  |
| Where inputs must be broadcastable, got ((), (1,), (0,)) | 2 | 20 |
| Arrays are not equal (max abs diff 247) | 1 | 21 |
| Arrays are not equal (max abs diff 248) | 1 | 21 |
| ConvTranspose output shape must be fully defined and non-negative | 1 | 22 |
| DequantizeLinear zero_point shape must match scale shape | 1 | 15 |
| Dropout mask output is not supported | 1 | 22 |
| Graph must contain at least one node | 1 | 25 |
| Pad value input must be a scalar | 1 | 24 |
| ReduceMax does not support dtype bool | 1 | 20 |
| ReduceMin does not support dtype bool | 1 | 20 |
| Unsupported op ArrayFeatureExtractor | 1 |  |
| Unsupported op Binarizer | 1 |  |
| Unsupported op MelWeightMatrix | 1 | 17 |
| Unsupported op QLinearSoftmax | 1 | 15 |
| Unsupported op QLinearAveragePool | 1 | 15 |
| Unsupported op RandomUniform | 1 | 22 |
| Unsupported op TreeEnsembleClassifier | 1 | 12 |
| onnxruntime failed to run onnx2c-org/test/local_ops/test_gemm_CM_transA/model.onnx: [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Non-zero status code returned while running Gemm node. Name:'*' Status Message: Gemm: Invalid bias shape for broadcast | 1 | 12 |

## Error frequency by opset

| Error message | Opset | Count |
| --- | --- | --- |
| Out of tolerance | 8 | 1 |
| Out of tolerance | 9 | 1 |
| Unsupported op TreeEnsembleClassifier | 12 | 1 |
| onnxruntime failed to run onnx2c-org/test/local_ops/test_gemm_CM_transA/model.onnx: [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Non-zero status code returned while running Gemm node. Name:'*' Status Message: Gemm: Invalid bias shape for broadcast | 12 | 1 |
| DequantizeLinear zero_point shape must match scale shape | 15 | 1 |
| Unsupported op QLinearSoftmax | 15 | 1 |
| Out of tolerance | 15 | 1 |
| Unsupported op QLinearAveragePool | 15 | 1 |
| Unsupported non-tensor value '*' in op Identity. | 16 | 1 |
| Unsupported op Loop | 16 | 1 |
| Unsupported op Loop | 17 | 6 |
| Unsupported op SequenceMap | 17 | 6 |
| Unsupported op BlackmanWindow | 17 | 2 |
| Unsupported op HannWindow | 17 | 2 |
| Unsupported op STFT | 17 | 2 |
| Unsupported op MelWeightMatrix | 17 | 1 |
| Unsupported op CenterCropPad | 18 | 6 |
| Unsupported op ScatterElements | 18 | 6 |
| Unsupported op Col2Im | 18 | 5 |
| OptionalHasElement expects exactly one non-empty input. | 18 | 4 |
| Unsupported op OptionalGetElement | 18 | 4 |
| Unsupported op BitwiseNot | 18 | 2 |
| Unsupported op DFT | 19 | 3 |
| Out of tolerance | 19 | 2 |
| Unsupported op ImageDecoder | 20 | 9 |
| Unsupported op StringSplit | 20 | 6 |
| Unsupported op StringConcat | 20 | 5 |
| Unsupported op AffineGrid | 20 | 4 |
| Unsupported op DFT | 20 | 3 |
| Unsupported op RegexFullMatch | 20 | 3 |
| Gelu only supports approximate=none | 20 | 2 |
| Split output shape must be (1,), got (2,) | 20 | 2 |
| Where inputs must be broadcastable, got ((), (1,), (0,)) | 20 | 2 |
| ReduceMax does not support dtype bool | 20 | 1 |
| ReduceMin does not support dtype bool | 20 | 1 |
| Arrays are not equal (max abs diff 148) | 21 | 2 |
| Arrays are not equal (max abs diff 247) | 21 | 1 |
| Arrays are not equal (max abs diff 248) | 21 | 1 |
| Dropout supports only the data input and 1 or 2 outputs | 22 | 8 |
| Out of tolerance | 22 | 4 |
| Unsupported op DeformConv | 22 | 4 |
| Unsupported op RNN | 22 | 4 |
| HardSigmoid only supports alpha=0.2 | 22 | 3 |
| Unsupported op RandomUniformLike | 22 | 3 |
| Unsupported op RoiAlign | 22 | 3 |
| Cast input and output shapes must match | 22 | 2 |
| LpPool expects 2D kernel_shape | 22 | 2 |
| LpPool supports auto_pad=NOTSET only | 22 | 2 |
| Selu only supports alpha=1.6732632423543772 | 22 | 2 |
| ThresholdedRelu only supports alpha=1.0 | 22 | 2 |
| Unsupported op Det | 22 | 2 |
| Unsupported op MaxUnpool | 22 | 2 |
| ConvTranspose output shape must be fully defined and non-negative | 22 | 1 |
| Dropout mask output is not supported | 22 | 1 |
| Unsupported op RandomUniform | 22 | 1 |
| Pad value input must be a scalar | 24 | 1 |
| Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor '*'. | 25 | 22 |
| Unsupported elem_type 19 (FLOAT8E5M2) for tensor '*'. | 25 | 20 |
| Unsupported elem_type 18 (FLOAT8E4M3FNUZ) for tensor '*'. | 25 | 18 |
| Unsupported elem_type 20 (FLOAT8E5M2FNUZ) for tensor '*'. | 25 | 18 |
| Unsupported elem_type 21 (UINT4) for tensor '*'. | 25 | 17 |
| Unsupported elem_type 22 (INT4) for tensor '*'. | 25 | 17 |
| Unsupported elem_type 25 (UINT2) for tensor '*'. | 25 | 17 |
| Unsupported elem_type 26 (INT2) for tensor '*'. | 25 | 17 |
| Unsupported elem_type 23 (FLOAT4E2M1) for tensor '*'. | 25 | 14 |
| Unsupported elem_type 24 (FLOAT8E8M0) for tensor '*'. | 25 | 4 |
| QuantizeLinear block_size is not supported | 25 | 2 |
| Graph must contain at least one node | 25 | 1 |
| Unsupported non-tensor value '*' in op Identity. | 25 | 1 |

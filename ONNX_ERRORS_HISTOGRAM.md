# Error frequency

| Error message | Count | Opset versions |
| --- | --- | --- |
| Unsupported value type '*' for '*'. Hint: export the model with tensor inputs/outputs. | 28 | 11, 12, 13, 17, 18, 24, 25 |
| Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor '*'. | 22 | 25 |
| Unsupported elem_type 19 (FLOAT8E5M2) for tensor '*'. | 20 | 25 |
| Unsupported elem_type 18 (FLOAT8E4M3FNUZ) for tensor '*'. | 18 | 25 |
| Unsupported elem_type 20 (FLOAT8E5M2FNUZ) for tensor '*'. | 18 | 25 |
| Unsupported elem_type 21 (UINT4) for tensor '*'. | 17 | 25 |
| Unsupported elem_type 22 (INT4) for tensor '*'. | 17 | 25 |
| Unsupported elem_type 25 (UINT2) for tensor '*'. | 17 | 25 |
| Unsupported elem_type 26 (INT2) for tensor '*'. | 17 | 25 |
| Unsupported dtype string | 14 | 10, 19 |
| Unsupported elem_type 23 (FLOAT4E2M1) for tensor '*'. | 14 | 25 |
| Unsupported op ImageDecoder | 9 | 20 |
| Dropout supports only the data input and 1 or 2 outputs | 8 | 22 |
| Out of tolerance | 7 | 9, 19, 22 |
| Unsupported elem_type 16 (BFLOAT16) for tensor '*'. | 6 | 25 |
| Unsupported op CenterCropPad | 6 | 18 |
| Unsupported op DFT | 6 | 19, 20 |
| Unsupported op ScatterElements | 6 | 18 |
| Unsupported op StringSplit | 6 | 20 |
| Unsupported op Unique | 6 | 11 |
| Unsupported op Col2Im | 5 | 18 |
| Unsupported op If | 5 | 11, 20 |
| Unsupported op StringConcat | 5 | 20 |
| OptionalHasElement expects exactly one non-empty input. | 4 | 18 |
| Unsupported elem_type 24 (FLOAT8E8M0) for tensor '*'. | 4 | 25 |
| Unsupported op AffineGrid | 4 | 20 |
| Unsupported op Compress | 4 | 11 |
| Unsupported op DeformConv | 4 | 22 |
| Unsupported op LabelEncoder | 4 |  |
| Unsupported op RNN | 4 | 22 |
| Unsupported optional element type '*' for '*'. Hint: export the model with optional tensor inputs/outputs. | 4 | 16, 18 |
| HardSigmoid only supports alpha=0.2 | 3 | 22 |
| Unsupported op DynamicQuantizeLinear | 3 | 11 |
| Unsupported op Loop | 3 | 11 |
| Unsupported op Momentum | 3 |  |
| Unsupported op RandomUniformLike | 3 | 22 |
| Unsupported op RegexFullMatch | 3 | 20 |
| Unsupported op RoiAlign | 3 | 22 |
| name '*' is not defined | 3 |  |
| BatchNormalization must have 5 inputs and 1 output | 2 | 15 |
| Failed to build testbench. | 2 | 12 |
| Gelu only supports approximate=none | 2 | 20 |
| LpPool expects 2D kernel_shape | 2 | 22 |
| LpPool supports auto_pad=NOTSET only | 2 | 22 |
| QuantizeLinear block_size is not supported | 2 | 25 |
| Selu only supports alpha=1.6732632423543772 | 2 | 22 |
| ThresholdedRelu only supports alpha=1.0 | 2 | 22 |
| Unsupported op Adam | 2 |  |
| Unsupported op BitwiseNot | 2 | 18 |
| Unsupported op BlackmanWindow | 2 | 17 |
| Unsupported op Det | 2 | 22 |
| Unsupported op Gradient | 2 | 12 |
| Unsupported op HannWindow | 2 | 17 |
| Unsupported op MaxUnpool | 2 | 22 |
| Unsupported op OptionalGetElement | 2 | 18 |
| Unsupported op ReverseSequence | 2 | 10 |
| Unsupported op STFT | 2 | 17 |
| Unsupported op Scatter | 2 | 10 |
| Unsupported op TreeEnsemble | 2 |  |
| ConvTranspose output shape must be fully defined and non-negative | 1 | 22 |
| Dropout mask output is not supported | 1 | 22 |
| Dynamic dim for tensor '*' | 1 | 12 |
| Graph must contain at least one node | 1 | 25 |
| Pad value input must be a scalar | 1 | 24 |
| ReduceMax does not support dtype bool | 1 | 20 |
| ReduceMin does not support dtype bool | 1 | 20 |
| Testbench execution failed: exit code 1 | 1 | 13 |
| Unsupported op ArrayFeatureExtractor | 1 |  |
| Unsupported op Binarizer | 1 |  |
| Unsupported op MatMulInteger | 1 | 10 |
| Unsupported op MelWeightMatrix | 1 | 17 |
| Unsupported op QLinearConv | 1 | 10 |

## Error frequency by opset

| Error message | Opset | Count |
| --- | --- | --- |
| Out of tolerance | 9 | 1 |
| Unsupported dtype string | 10 | 12 |
| Unsupported op ReverseSequence | 10 | 2 |
| Unsupported op Scatter | 10 | 2 |
| Unsupported op MatMulInteger | 10 | 1 |
| Unsupported op QLinearConv | 10 | 1 |
| Unsupported op Unique | 11 | 6 |
| Unsupported op Compress | 11 | 4 |
| Unsupported op DynamicQuantizeLinear | 11 | 3 |
| Unsupported op Loop | 11 | 3 |
| Unsupported value type '*' for '*'. Hint: export the model with tensor inputs/outputs. | 11 | 2 |
| Unsupported op If | 11 | 1 |
| Unsupported value type '*' for '*'. Hint: export the model with tensor inputs/outputs. | 12 | 7 |
| Failed to build testbench. | 12 | 2 |
| Unsupported op Gradient | 12 | 2 |
| Dynamic dim for tensor '*' | 12 | 1 |
| Unsupported value type '*' for '*'. Hint: export the model with tensor inputs/outputs. | 13 | 2 |
| Testbench execution failed: exit code 1 | 13 | 1 |
| BatchNormalization must have 5 inputs and 1 output | 15 | 2 |
| Unsupported optional element type '*' for '*'. Hint: export the model with optional tensor inputs/outputs. | 16 | 3 |
| Unsupported value type '*' for '*'. Hint: export the model with tensor inputs/outputs. | 17 | 12 |
| Unsupported op BlackmanWindow | 17 | 2 |
| Unsupported op HannWindow | 17 | 2 |
| Unsupported op STFT | 17 | 2 |
| Unsupported op MelWeightMatrix | 17 | 1 |
| Unsupported op CenterCropPad | 18 | 6 |
| Unsupported op ScatterElements | 18 | 6 |
| Unsupported op Col2Im | 18 | 5 |
| OptionalHasElement expects exactly one non-empty input. | 18 | 4 |
| Unsupported op BitwiseNot | 18 | 2 |
| Unsupported op OptionalGetElement | 18 | 2 |
| Unsupported optional element type '*' for '*'. Hint: export the model with optional tensor inputs/outputs. | 18 | 1 |
| Unsupported value type '*' for '*'. Hint: export the model with tensor inputs/outputs. | 18 | 1 |
| Unsupported op DFT | 19 | 3 |
| Out of tolerance | 19 | 2 |
| Unsupported dtype string | 19 | 2 |
| Unsupported op ImageDecoder | 20 | 9 |
| Unsupported op StringSplit | 20 | 6 |
| Unsupported op StringConcat | 20 | 5 |
| Unsupported op AffineGrid | 20 | 4 |
| Unsupported op If | 20 | 4 |
| Unsupported op DFT | 20 | 3 |
| Unsupported op RegexFullMatch | 20 | 3 |
| Gelu only supports approximate=none | 20 | 2 |
| ReduceMax does not support dtype bool | 20 | 1 |
| ReduceMin does not support dtype bool | 20 | 1 |
| Dropout supports only the data input and 1 or 2 outputs | 22 | 8 |
| Out of tolerance | 22 | 4 |
| Unsupported op DeformConv | 22 | 4 |
| Unsupported op RNN | 22 | 4 |
| HardSigmoid only supports alpha=0.2 | 22 | 3 |
| Unsupported op RandomUniformLike | 22 | 3 |
| Unsupported op RoiAlign | 22 | 3 |
| LpPool expects 2D kernel_shape | 22 | 2 |
| LpPool supports auto_pad=NOTSET only | 22 | 2 |
| Selu only supports alpha=1.6732632423543772 | 22 | 2 |
| ThresholdedRelu only supports alpha=1.0 | 22 | 2 |
| Unsupported op Det | 22 | 2 |
| Unsupported op MaxUnpool | 22 | 2 |
| ConvTranspose output shape must be fully defined and non-negative | 22 | 1 |
| Dropout mask output is not supported | 22 | 1 |
| Unsupported value type '*' for '*'. Hint: export the model with tensor inputs/outputs. | 24 | 3 |
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
| Unsupported elem_type 16 (BFLOAT16) for tensor '*'. | 25 | 6 |
| Unsupported elem_type 24 (FLOAT8E8M0) for tensor '*'. | 25 | 4 |
| QuantizeLinear block_size is not supported | 25 | 2 |
| Graph must contain at least one node | 25 | 1 |
| Unsupported value type '*' for '*'. Hint: export the model with tensor inputs/outputs. | 25 | 1 |

## Local ONNX file support histogram

### Error frequency

| Error message | Count | Opset versions |
| --- | --- | --- |
| Unsupported LSTM direction b'*' | 2 | 11 |
| Unsupported op QLinearAdd | 2 |  |
| Gemm bias input must be broadcastable to output shape, got (2,) vs (2, 4) | 1 | 12 |

## Error frequency by opset

| Error message | Opset | Count |
| --- | --- | --- |
| Unsupported LSTM direction b'*' | 11 | 2 |
| Gemm bias input must be broadcastable to output shape, got (2,) vs (2, 4) | 12 | 1 |

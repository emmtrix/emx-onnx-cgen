# Error frequency

| Error message | Count | Opset versions |
| --- | --- | --- |
| Unsupported elem_type 8 (STRING) for tensor '*'. | 32 | 10, 19, 20 |
| Unsupported value type '*' for '*'. Hint: export the model with tensor inputs/outputs. | 28 | 11, 12, 13, 17, 18, 24, 25 |
| Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor '*'. | 22 | 25 |
| Unsupported elem_type 19 (FLOAT8E5M2) for tensor '*'. | 20 | 25 |
| Unsupported elem_type 18 (FLOAT8E4M3FNUZ) for tensor '*'. | 18 | 25 |
| Unsupported elem_type 20 (FLOAT8E5M2FNUZ) for tensor '*'. | 18 | 25 |
| Unsupported elem_type 21 (UINT4) for tensor '*'. | 17 | 25 |
| Unsupported elem_type 22 (INT4) for tensor '*'. | 17 | 25 |
| Unsupported elem_type 25 (UINT2) for tensor '*'. | 17 | 25 |
| Unsupported elem_type 26 (INT2) for tensor '*'. | 17 | 25 |
| Unsupported elem_type 23 (FLOAT4E2M1) for tensor '*'. | 14 | 25 |
| Unsupported op ImageDecoder | 9 | 20 |
| Dropout supports only the data input and 1 or 2 outputs | 8 | 22 |
| Out of tolerance | 8 | 6, 9, 19, 22 |
| Unsupported op TfIdfVectorizer | 7 | 9 |
| Unsupported elem_type 16 (BFLOAT16) for tensor '*'. | 6 | 25 |
| Unsupported op CenterCropPad | 6 | 18 |
| Unsupported op DFT | 6 | 19, 20 |
| Unsupported op ScatterElements | 6 | 18 |
| Unsupported op Unique | 6 | 11 |
| Unsupported op Col2Im | 5 | 18 |
| Unsupported op If | 5 | 11, 20 |
| OptionalHasElement expects exactly one non-empty input. | 4 | 18 |
| Unsupported elem_type 24 (FLOAT8E8M0) for tensor '*'. | 4 | 25 |
| Unsupported op AffineGrid | 4 | 20 |
| Unsupported op Compress | 4 | 11 |
| Unsupported op DeformConv | 4 | 22 |
| Unsupported op RNN | 4 | 22 |
| Unsupported optional element type '*' for '*'. Hint: export the model with optional tensor inputs/outputs. | 4 | 16, 18 |
| Elu only supports alpha=1.0 | 3 | 6, 22 |
| HardSigmoid only supports alpha=0.2 | 3 | 22 |
| LeakyRelu only supports alpha=0.01 | 3 | 6, 16 |
| Unsupported op DynamicQuantizeLinear | 3 | 11 |
| Unsupported op Loop | 3 | 11 |
| Unsupported op Momentum | 3 |  |
| Unsupported op RandomUniformLike | 3 | 22 |
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
| Unsupported op ConvInteger | 2 | 10 |
| Unsupported op Det | 2 | 22 |
| Unsupported op Gradient | 2 | 12 |
| Unsupported op HannWindow | 2 | 17 |
| Unsupported op MaxUnpool | 2 | 22 |
| Unsupported op OptionalGetElement | 2 | 18 |
| Unsupported op ReverseSequence | 2 | 10 |
| Unsupported op STFT | 2 | 17 |
| Unsupported op Scan | 2 | 8, 9 |
| Unsupported op Scatter | 2 | 10 |
| Unsupported op TreeEnsemble | 2 |  |
| ConvTranspose output shape must be fully defined and non-negative | 1 | 22 |
| Dropout mask output is not supported | 1 | 22 |
| Dynamic dim for tensor '*' | 1 | 12 |
| Graph must contain at least one node | 1 | 25 |
| Pad value input must be a scalar | 1 | 24 |
| ReduceMax does not support dtype bool | 1 | 20 |
| ReduceMin does not support dtype bool | 1 | 20 |
| Unsupported op ArrayFeatureExtractor | 1 |  |
| Unsupported op Binarizer | 1 |  |
| Unsupported op MatMulInteger | 1 | 10 |
| Unsupported op MelWeightMatrix | 1 | 17 |
| Unsupported op QLinearConv | 1 | 10 |
| Unsupported op Upsample | 1 | 9 |

## Local ONNX file support histogram

### Error frequency

| Error message | Count | Opset versions |
| --- | --- | --- |
| Unsupported LSTM direction b'*' | 2 | 11 |
| Unsupported op QLinearAdd | 2 |  |
| Gemm bias input must be broadcastable to output shape, got (2,) vs (2, 4) | 1 | 12 |

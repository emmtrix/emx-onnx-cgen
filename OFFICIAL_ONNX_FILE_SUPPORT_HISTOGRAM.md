# Error frequency

| Error message | Count | Histogram |
| --- | --- | --- |
| ONNX Runtime failed to run | 81 | ██████████████████████████████ |
| Missing output 1 in testbench data | 36 | █████████████ |
| Failed to build testbench. | 34 | █████████████ |
| Unsupported elem_type 8 (STRING) for tensor '*'. | 32 | ████████████ |
| Test data input count does not match model inputs: 1 vs 3. | 27 | ██████████ |
| Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor '*'. | 22 | ████████ |
| Unsupported elem_type 19 (FLOAT8E5M2) for tensor '*'. | 20 | ███████ |
| Currently not supporting loading segments. | 19 | ███████ |
| Unsupported elem_type 18 (FLOAT8E4M3FNUZ) for tensor '*'. | 18 | ███████ |
| Unsupported elem_type 20 (FLOAT8E5M2FNUZ) for tensor '*'. | 18 | ███████ |
| Test data input count does not match model inputs: 1 vs 2. | 17 | ██████ |
| Unsupported elem_type 21 (UINT4) for tensor '*'. | 17 | ██████ |
| Unsupported elem_type 22 (INT4) for tensor '*'. | 17 | ██████ |
| Unsupported elem_type 25 (UINT2) for tensor '*'. | 17 | ██████ |
| Unsupported elem_type 26 (INT2) for tensor '*'. | 17 | ██████ |
| Out of tolerance | 14 | █████ |
| Unsupported elem_type 23 (FLOAT4E2M1) for tensor '*'. | 14 | █████ |
| Output shape must be fully defined | 9 | ███ |
| Unsupported op ImageDecoder | 9 | ███ |
| Unsupported op NonMaxSuppression | 9 | ███ |
| Unsupported value type '*' for '*'. Hint: export the model with tensor inputs/outputs. | 9 | ███ |
| Dropout supports only the data input and 1 or 2 outputs | 8 | ███ |
| Unsupported op QLinearMatMul | 8 | ███ |
| Unsupported op RotaryEmbedding | 8 | ███ |
| tuple index out of range | 8 | ███ |
| Unsupported op TfIdfVectorizer | 7 | ███ |
| Unsupported op TopK | 7 | ███ |
| AveragePool has unsupported attributes | 6 | ██ |
| Missing output 2 in testbench data | 6 | ██ |
| Unsupported elem_type 16 (BFLOAT16) for tensor '*'. | 6 | ██ |
| Unsupported op CenterCropPad | 6 | ██ |
| Unsupported op DFT | 6 | ██ |
| Unsupported op Einsum | 6 | ██ |
| Unsupported op ScatterElements | 6 | ██ |
| Unsupported op Unique | 6 | ██ |
| And expects identical input/output shapes | 5 | ██ |
| AveragePool expects 2D kernel_shape | 5 | ██ |
| Or expects identical input/output shapes | 5 | ██ |
| Test data input count does not match model inputs: 1 vs 5. | 5 | ██ |
| Unsupported op Col2Im | 5 | ██ |
| Unsupported op DequantizeLinear | 5 | ██ |
| Unsupported op If | 5 | ██ |
| Unsupported op ScatterND | 5 | ██ |
| Xor expects identical input/output shapes | 5 | ██ |
| Sum expects identical input/output shapes | 4 | █ |
| Unsupported elem_type 24 (FLOAT8E8M0) for tensor '*'. | 4 | █ |
| Unsupported op AffineGrid | 4 | █ |
| Unsupported op Compress | 4 | █ |
| Unsupported op DeformConv | 4 | █ |
| Unsupported op GRU | 4 | █ |
| Unsupported op OneHot | 4 | █ |
| Unsupported op OptionalHasElement | 4 | █ |
| Unsupported op RNN | 4 | █ |
| '*' | 3 | █ |
| AveragePool supports auto_pad=NOTSET only | 3 | █ |
| Elu only supports alpha=1.0 | 3 | █ |
| HardSigmoid only supports alpha=0.2 | 3 | █ |
| LeakyRelu only supports alpha=0.01 | 3 | █ |
| Min expects identical input/output shapes | 3 | █ |
| Unsupported op Bernoulli | 3 | █ |
| Unsupported op DynamicQuantizeLinear | 3 | █ |
| Unsupported op GatherND | 3 | █ |
| Unsupported op Loop | 3 | █ |
| Unsupported op Momentum | 3 | █ |
| Unsupported op RandomUniformLike | 3 | █ |
| Unsupported op RoiAlign | 3 | █ |
| Unsupported op TensorScatter | 3 | █ |
| AveragePool supports ceil_mode=0 only | 2 | █ |
| BatchNormalization must have 5 inputs and 1 output | 2 | █ |
| BitwiseAnd expects identical input/output shapes | 2 | █ |
| BitwiseOr expects identical input/output shapes | 2 | █ |
| BitwiseXor expects identical input/output shapes | 2 | █ |
| Gelu only supports approximate=none | 2 | █ |
| LpPool expects 2D kernel_shape | 2 | █ |
| LpPool supports auto_pad=NOTSET only | 2 | █ |
| Pow expects matching dtypes, got float, int32 | 2 | █ |
| Pow expects matching dtypes, got float, int64 | 2 | █ |
| QuantizeLinear block_size is not supported | 2 | █ |
| Selu only supports alpha=1.6732632423543772 | 2 | █ |
| Test data input count does not match model inputs: 3 vs 5. | 2 | █ |
| ThresholdedRelu only supports alpha=1.0 | 2 | █ |
| Tile repeats input must be a constant initializer | 2 | █ |
| Unsupported op Adagrad | 2 | █ |
| Unsupported op Adam | 2 | █ |
| Unsupported op BitwiseNot | 2 | █ |
| Unsupported op BlackmanWindow | 2 | █ |
| Unsupported op ConvInteger | 2 | █ |
| Unsupported op Det | 2 | █ |
| Unsupported op Gradient | 2 | █ |
| Unsupported op HammingWindow | 2 | █ |
| Unsupported op HannWindow | 2 | █ |
| Unsupported op MaxUnpool | 2 | █ |
| Unsupported op ReverseSequence | 2 | █ |
| Unsupported op STFT | 2 | █ |
| Unsupported op Scan | 2 | █ |
| Unsupported op Scatter | 2 | █ |
| Unsupported op TreeEnsemble | 2 | █ |
| 
Arrays are not equal

Mismatched elements: 1 / 6 (16.7%)
 ACTUAL: array([False, False,  True, False,  True,  True])
 DESIRED: array([False, False,  True, False, False,  True]) | 1 | █ |
| 
Arrays are not equal

Mismatched elements: 1 / 6 (16.7%)
Max absolute difference among violations: 1
Max relative difference among violations: 0.00558659
 ACTUAL: array([153, 255,   0,  26, 221, 178], dtype=uint8)
 DESIRED: array([153, 255,   0,  26, 221, 179], dtype=uint8) | 1 | █ |
| 
Arrays are not equal

Mismatched elements: 2 / 6 (33.3%)
 ACTUAL: array([False, False,  True, False,  True,  True])
 DESIRED: array([False, False, False, False,  True, False]) | 1 | █ |
| ConvTranspose output shape must be fully defined and non-negative | 1 | █ |
| Dropout mask output is not supported | 1 | █ |
| Graph must contain at least one node | 1 | █ |
| LpPool supports dilations=1 only | 1 | █ |
| Max expects identical input/output shapes | 1 | █ |
| Max must have at least 2 inputs | 1 | █ |
| Mean must have at least 2 inputs | 1 | █ |
| Min must have at least 2 inputs | 1 | █ |
| Missing output 3 in testbench data | 1 | █ |
| Missing output 4 in testbench data | 1 | █ |
| Missing output 5 in testbench data | 1 | █ |
| Pad value input must be a scalar | 1 | █ |
| Pow expects matching dtypes, got float, uint32 | 1 | █ |
| Pow expects matching dtypes, got float, uint64 | 1 | █ |
| ReduceMax does not support dtype bool | 1 | █ |
| ReduceMin does not support dtype bool | 1 | █ |
| Sum must have at least 2 inputs | 1 | █ |
| Test data input count does not match model inputs: 3 vs 6. | 1 | █ |
| The element type in the input tensor is UNDEFINED. | 1 | █ |
| Unsupported op ArrayFeatureExtractor | 1 | █ |
| Unsupported op Binarizer | 1 | █ |
| Unsupported op MatMulInteger | 1 | █ |
| Unsupported op MelWeightMatrix | 1 | █ |
| Unsupported op NonZero | 1 | █ |
| Unsupported op OptionalGetElement | 1 | █ |
| Unsupported op QLinearConv | 1 | █ |
| Unsupported op Upsample | 1 | █ |
| cannot reshape array of size 0 into shape (115,101,113,95,101,109,112,116,121) | 1 | █ |
| cannot reshape array of size 12 into shape (111,112,116,95,115,101,113) | 1 | █ |
| cannot reshape array of size 26 into shape (111,112,116,105,111,110,97,108,95,105,110,112,117,116) | 1 | █ |
| cannot reshape array of size 27 into shape (111,112,116,95,105,110) | 1 | █ |

## Local ONNX file support histogram

### Error frequency

| Error message | Count | Histogram |
| --- | --- | --- |
| Unsupported op ScatterND | 4 | ██████████████████████████████ |
| ONNX Runtime failed to run | 2 | ███████████████ |
| Unsupported LSTM direction b'*' | 2 | ███████████████ |
| Unsupported op QLinearAdd | 2 | ███████████████ |
| Unsupported op QLinearMul | 2 | ███████████████ |
| Gemm bias input must be broadcastable to output shape, got (2,) vs (2, 4) | 1 | ████████ |

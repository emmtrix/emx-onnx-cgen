<!-- AUTO-GENERATED FILE. DO NOT EDIT. -->
<!-- Regenerate with: UPDATE_REFS=1 pytest -q tests/test_official_onnx_files_docs.py::test_official_onnx_file_support_doc -->

# Supported operators

Operators are marked supported when they appear in an ONNX file with a successful verify result.

The `Notes` column links to implementation notes in [`docs/operator-notes.md`](docs/operator-notes.md); the `Relevant CLI options` column lists CLI flags that specifically affect the operator's generated code. Both are maintained in [`docs/operator-metadata.json`](docs/operator-metadata.json).

Supported operators: 264 / 265

| Operator | Supported | Notes | Relevant CLI options |
| --- | --- | --- | --- |
| Abs | ✅ |  |  |
| Acos | ✅ |  |  |
| Acosh | ✅ |  |  |
| Add | ✅ |  |  |
| Affine | ✅ |  |  |
| AffineGrid | ✅ |  | `--fp32-accumulation-strategy`, `--fp16-accumulation-strategy` |
| And | ✅ |  |  |
| ArgMax | ✅ |  |  |
| ArgMin | ✅ |  |  |
| Asin | ✅ |  |  |
| Asinh | ✅ |  |  |
| Atan | ✅ |  |  |
| Atanh | ✅ |  |  |
| Attention | ✅ | [Notes](docs/operator-notes.md#attention-and-transformers) | `--fp32-accumulation-strategy`, `--fp16-accumulation-strategy` |
| AveragePool | ✅ |  |  |
| BatchNormalization | ✅ | [Notes](docs/operator-notes.md#normalization-variants) |  |
| Bernoulli | ✅ | [Notes](docs/operator-notes.md#random-number-generation-randomuniform-bernoulli) |  |
| BitCast | ✅ |  |  |
| BitShift | ✅ |  |  |
| BitwiseAnd | ✅ |  |  |
| BitwiseNot | ✅ |  |  |
| BitwiseOr | ✅ |  |  |
| BitwiseXor | ✅ |  |  |
| BlackmanWindow | ✅ |  |  |
| Cast | ✅ |  |  |
| CastLike | ✅ |  |  |
| CausalConvWithState | ✅ |  | `--fp32-accumulation-strategy`, `--fp16-accumulation-strategy` |
| Ceil | ✅ |  |  |
| Celu | ✅ |  |  |
| CenterCropPad | ✅ |  |  |
| Clip | ✅ |  |  |
| Col2Im | ✅ | [Notes](docs/operator-notes.md#reconstruction-col2im-maxunpool) |  |
| Compress | ✅ | [Notes](docs/operator-notes.md#dynamic-output-operators-unique-nonzero-compress) |  |
| Concat | ✅ |  |  |
| ConcatFromSequence | ✅ |  | `--sequence-element-shape` |
| Constant | ✅ |  |  |
| ConstantOfShape | ✅ |  |  |
| Conv | ✅ | [Notes](docs/operator-notes.md#convolution-family-conv-convtranspose-deformconv) | `--fp32-accumulation-strategy`, `--fp16-accumulation-strategy` |
| ConvInteger | ✅ |  |  |
| ConvTranspose | ✅ | [Notes](docs/operator-notes.md#convolution-family-conv-convtranspose-deformconv) |  |
| Cos | ✅ |  |  |
| Cosh | ✅ |  |  |
| Crop | ✅ |  |  |
| CumProd | ✅ |  |  |
| CumSum | ✅ |  |  |
| DFT | ✅ | [Notes](docs/operator-notes.md#fft-related-operators-dft-stft) |  |
| DeformConv | ✅ | [Notes](docs/operator-notes.md#convolution-family-conv-convtranspose-deformconv) | `--fp32-accumulation-strategy`, `--fp16-accumulation-strategy` |
| DepthToSpace | ✅ |  |  |
| DequantizeLinear | ✅ | [Notes](docs/operator-notes.md#quantization-quantizelinear-dequantizelinear) |  |
| Det | ✅ |  |  |
| Div | ✅ |  |  |
| Dropout | ✅ |  |  |
| DynamicQuantizeLinear | ✅ |  |  |
| DynamicSlice | ✅ |  |  |
| Einsum | ✅ | [Notes](docs/operator-notes.md#einsum-limited-patterns) | `--fp32-accumulation-strategy`, `--fp16-accumulation-strategy` |
| Elu | ✅ |  |  |
| Equal | ✅ |  |  |
| Erf | ✅ |  |  |
| Exp | ✅ |  |  |
| Expand | ✅ |  |  |
| EyeLike | ✅ |  |  |
| Flatten | ✅ |  |  |
| Floor | ✅ |  |  |
| GRU | ✅ | [Notes](docs/operator-notes.md#recurrent-networks-lstm-gru-rnn) |  |
| Gather | ✅ |  |  |
| GatherElements | ✅ |  |  |
| GatherND | ✅ |  |  |
| Gelu | ✅ |  |  |
| Gemm | ✅ |  | `--fp32-accumulation-strategy`, `--fp16-accumulation-strategy` |
| GlobalAveragePool | ✅ |  |  |
| GlobalMaxPool | ✅ |  |  |
| Greater | ✅ |  |  |
| GreaterOrEqual | ✅ |  |  |
| GridSample | ✅ | [Notes](docs/operator-notes.md#spatial-sampling-resize-gridsample-roialign) |  |
| GroupNormalization | ✅ | [Notes](docs/operator-notes.md#normalization-variants) |  |
| HammingWindow | ✅ |  |  |
| HannWindow | ✅ |  |  |
| HardSigmoid | ✅ |  |  |
| HardSwish | ✅ |  |  |
| Hardmax | ✅ |  |  |
| Identity | ✅ |  |  |
| If | ✅ |  |  |
| ImageDecoder | ✅ | [Notes](docs/operator-notes.md#image-decoding-imagedecoder) | `--image-decoder-libs` |
| ImageScaler | ✅ |  |  |
| InstanceNormalization | ✅ | [Notes](docs/operator-notes.md#normalization-variants) |  |
| IsInf | ✅ |  |  |
| IsNaN | ✅ |  |  |
| LRN | ✅ |  |  |
| LSTM | ✅ | [Notes](docs/operator-notes.md#recurrent-networks-lstm-gru-rnn) |  |
| LayerNormalization | ✅ | [Notes](docs/operator-notes.md#normalization-variants) | `--fp32-accumulation-strategy`, `--fp16-accumulation-strategy` |
| LeakyRelu | ✅ |  |  |
| Less | ✅ |  |  |
| LessOrEqual | ✅ |  |  |
| LinearAttention | ✅ |  |  |
| Log | ✅ |  |  |
| LogSoftmax | ✅ | [Notes](docs/operator-notes.md#softmax-and-logsoftmax) |  |
| Loop | ✅ | [Notes](docs/operator-notes.md#control-flow-loop-scan) |  |
| LpNormalization | ✅ |  |  |
| LpPool | ✅ |  |  |
| MatMul | ✅ |  | `--fp32-accumulation-strategy`, `--fp16-accumulation-strategy` |
| MatMulInteger | ✅ |  |  |
| Max | ✅ |  |  |
| MaxPool | ✅ |  |  |
| MaxUnpool | ✅ | [Notes](docs/operator-notes.md#reconstruction-col2im-maxunpool) |  |
| Mean | ✅ |  |  |
| MeanVarianceNormalization | ✅ |  |  |
| MelWeightMatrix | ✅ | [Notes](docs/operator-notes.md#mel-frequency-filter-banks-melweightmatrix) |  |
| Min | ✅ |  |  |
| Mish | ✅ |  |  |
| Mod | ✅ |  |  |
| Mul | ✅ |  |  |
| Neg | ✅ |  |  |
| NegativeLogLikelihoodLoss | ✅ |  | `--fp32-accumulation-strategy`, `--fp16-accumulation-strategy` |
| NonMaxSuppression | ✅ | [Notes](docs/operator-notes.md#selection-and-filtering-topk-nonmaxsuppression) |  |
| NonZero | ✅ | [Notes](docs/operator-notes.md#dynamic-output-operators-unique-nonzero-compress) |  |
| Not | ✅ |  |  |
| OneHot | ✅ |  |  |
| OptionalGetElement | ✅ |  |  |
| OptionalHasElement | ✅ |  |  |
| Or | ✅ |  |  |
| PRelu | ✅ |  |  |
| Pad | ✅ |  |  |
| Pow | ✅ |  |  |
| QLinearConv | ✅ |  |  |
| QLinearMatMul | ✅ |  |  |
| QuantizeLinear | ✅ | [Notes](docs/operator-notes.md#quantization-quantizelinear-dequantizelinear) |  |
| RMSNormalization | ✅ | [Notes](docs/operator-notes.md#normalization-variants) |  |
| RNN | ✅ | [Notes](docs/operator-notes.md#recurrent-networks-lstm-gru-rnn) |  |
| RandomUniformLike | ✅ | [Notes](docs/operator-notes.md#random-number-generation-randomuniform-bernoulli) |  |
| Range | ✅ |  |  |
| Reciprocal | ✅ |  |  |
| ReduceL1 | ✅ |  | `--fp32-accumulation-strategy`, `--fp16-accumulation-strategy` |
| ReduceL2 | ✅ |  | `--fp32-accumulation-strategy`, `--fp16-accumulation-strategy` |
| ReduceLogSum | ✅ |  | `--fp32-accumulation-strategy`, `--fp16-accumulation-strategy` |
| ReduceLogSumExp | ✅ |  | `--fp32-accumulation-strategy`, `--fp16-accumulation-strategy` |
| ReduceMax | ✅ |  |  |
| ReduceMean | ✅ |  | `--fp32-accumulation-strategy`, `--fp16-accumulation-strategy` |
| ReduceMin | ✅ |  |  |
| ReduceProd | ✅ |  | `--fp32-accumulation-strategy`, `--fp16-accumulation-strategy` |
| ReduceSum | ✅ |  | `--fp32-accumulation-strategy`, `--fp16-accumulation-strategy` |
| ReduceSumSquare | ✅ |  | `--fp32-accumulation-strategy`, `--fp16-accumulation-strategy` |
| RegexFullMatch | ✅ |  |  |
| Relu | ✅ |  |  |
| Reshape | ✅ |  |  |
| Resize | ✅ | [Notes](docs/operator-notes.md#spatial-sampling-resize-gridsample-roialign) |  |
| ReverseSequence | ✅ |  |  |
| RoiAlign | ✅ | [Notes](docs/operator-notes.md#spatial-sampling-resize-gridsample-roialign) |  |
| RotaryEmbedding | ✅ | [Notes](docs/operator-notes.md#attention-and-transformers) |  |
| Round | ✅ |  |  |
| STFT | ✅ | [Notes](docs/operator-notes.md#fft-related-operators-dft-stft) |  |
| Scale | ✅ |  |  |
| Scan | ✅ | [Notes](docs/operator-notes.md#control-flow-loop-scan) |  |
| Scatter | ✅ |  |  |
| ScatterElements | ✅ |  |  |
| ScatterND | ✅ |  |  |
| Selu | ✅ |  |  |
| SequenceAt | ✅ |  | `--sequence-element-shape` |
| SequenceConstruct | ✅ |  | `--sequence-element-shape` |
| SequenceEmpty | ✅ |  | `--sequence-element-shape` |
| SequenceErase | ✅ |  | `--sequence-element-shape` |
| SequenceInsert | ✅ |  | `--sequence-element-shape` |
| SequenceLength | ✅ |  | `--sequence-element-shape` |
| SequenceMap | ✅ |  | `--sequence-element-shape` |
| Shape | ✅ |  |  |
| Shrink | ✅ |  |  |
| Sigmoid | ✅ |  |  |
| Sign | ✅ |  |  |
| Sin | ✅ |  |  |
| Sinh | ✅ |  |  |
| Size | ✅ |  |  |
| Slice | ✅ |  |  |
| Softmax | ✅ | [Notes](docs/operator-notes.md#softmax-and-logsoftmax) |  |
| SoftmaxCrossEntropyLoss | ✅ |  | `--fp32-accumulation-strategy`, `--fp16-accumulation-strategy` |
| Softplus | ✅ |  |  |
| Softsign | ✅ |  |  |
| SpaceToDepth | ✅ |  |  |
| Split | ✅ |  |  |
| SplitToSequence | ✅ |  | `--sequence-element-shape` |
| Sqrt | ✅ |  |  |
| Squeeze | ✅ |  |  |
| StringConcat | ✅ |  |  |
| StringNormalizer | ✅ |  |  |
| StringSplit | ✅ |  |  |
| Sub | ✅ |  |  |
| Sum | ✅ |  |  |
| Swish | ✅ |  |  |
| Tan | ✅ |  |  |
| Tanh | ✅ |  |  |
| TensorScatter | ✅ |  |  |
| TfIdfVectorizer | ✅ | [Notes](docs/operator-notes.md#text-processing-tfidfvectorizer-labelencoder) |  |
| ThresholdedRelu | ✅ |  |  |
| Tile | ✅ |  |  |
| TopK | ✅ | [Notes](docs/operator-notes.md#selection-and-filtering-topk-nonmaxsuppression) |  |
| Transpose | ✅ |  |  |
| Trilu | ✅ |  |  |
| Unique | ✅ | [Notes](docs/operator-notes.md#dynamic-output-operators-unique-nonzero-compress) |  |
| Unsqueeze | ✅ |  |  |
| Upsample | ✅ |  |  |
| Where | ✅ |  |  |
| Xor | ✅ |  |  |
| ai.onnx.ml.ArrayFeatureExtractor | ✅ |  |  |
| ai.onnx.ml.Binarizer | ✅ |  |  |
| ai.onnx.ml.LabelEncoder | ✅ | [Notes](docs/operator-notes.md#text-processing-tfidfvectorizer-labelencoder) |  |
| ai.onnx.ml.TreeEnsemble | ✅ | [Notes](docs/operator-notes.md#tree-based-models-treeensemble-treeensembleclassifier) |  |
| ai.onnx.preview.FlexAttention | ✅ |  |  |
| ai.onnx.preview.training.Adagrad | ✅ |  |  |
| ai.onnx.preview.training.Adam | ✅ |  |  |
| ai.onnx.preview.training.Gradient | ✅ |  |  |
| ai.onnx.preview.training.Momentum | ✅ |  |  |
| com.microsoft.Attention | ✅ | [Notes](docs/operator-notes.md#attention-and-transformers) | `--fp32-accumulation-strategy`, `--fp16-accumulation-strategy` |
| com.microsoft.AttnLSTM | ✅ |  |  |
| com.microsoft.BiasGelu | ✅ |  |  |
| com.microsoft.BifurcationDetector | ✅ |  |  |
| com.microsoft.CDist | ✅ |  |  |
| com.microsoft.CausalConvWithState | ✅ |  | `--fp32-accumulation-strategy`, `--fp16-accumulation-strategy` |
| com.microsoft.ConvTransposeWithDynamicPads | ✅ |  |  |
| com.microsoft.CropAndResize | ✅ |  |  |
| com.microsoft.DecoderMaskedMultiHeadAttention | ✅ |  |  |
| com.microsoft.DequantizeLinear | ✅ | [Notes](docs/operator-notes.md#quantization-quantizelinear-dequantizelinear) |  |
| com.microsoft.DynamicQuantizeLSTM | ✅ |  |  |
| com.microsoft.DynamicQuantizeMatMul | ✅ |  |  |
| com.microsoft.DynamicTimeWarping | ✅ |  |  |
| com.microsoft.EmbedLayerNormalization | ✅ |  | `--fp32-accumulation-strategy`, `--fp16-accumulation-strategy` |
| com.microsoft.ExpandDims | ✅ |  |  |
| com.microsoft.FastGelu | ✅ |  |  |
| com.microsoft.FusedConv | ✅ | [Notes](docs/operator-notes.md#convolution-family-conv-convtranspose-deformconv) | `--fp32-accumulation-strategy`, `--fp16-accumulation-strategy` |
| com.microsoft.FusedMatMul | ✅ |  | `--fp32-accumulation-strategy`, `--fp16-accumulation-strategy` |
| com.microsoft.GatherBlockQuantized | ✅ |  |  |
| com.microsoft.GridSample | ✅ | [Notes](docs/operator-notes.md#spatial-sampling-resize-gridsample-roialign) |  |
| com.microsoft.GroupQueryAttention | ✅ | [Notes](docs/operator-notes.md#attention-and-transformers) | `--fp32-accumulation-strategy`, `--fp16-accumulation-strategy` |
| com.microsoft.Inverse | ✅ |  |  |
| com.microsoft.LinearAttention | ✅ |  |  |
| com.microsoft.MatMulBnb4 | ✅ |  | `--fp32-accumulation-strategy`, `--fp16-accumulation-strategy` |
| com.microsoft.MatMulInteger16 | ✅ |  |  |
| com.microsoft.MatMulIntegerToFloat | ✅ |  |  |
| com.microsoft.MatMulNBits | ✅ |  | `--fp32-accumulation-strategy`, `--fp16-accumulation-strategy` |
| com.microsoft.MaxpoolWithMask | ✅ |  |  |
| com.microsoft.MoE | ✅ |  |  |
| com.microsoft.MultiHeadAttention | ✅ | [Notes](docs/operator-notes.md#attention-and-transformers) | `--fp32-accumulation-strategy`, `--fp16-accumulation-strategy` |
| com.microsoft.MurmurHash3 | ✅ |  |  |
| com.microsoft.NGramRepeatBlock | ✅ |  |  |
| com.microsoft.NhwcMaxPool | ✅ |  |  |
| com.microsoft.QAttention | ✅ |  |  |
| com.microsoft.QEmbedLayerNormalization | ✅ |  | `--fp32-accumulation-strategy`, `--fp16-accumulation-strategy` |
| com.microsoft.QGemm | ✅ |  |  |
| com.microsoft.QLinearAdd | ✅ |  |  |
| com.microsoft.QLinearAveragePool | ✅ |  |  |
| com.microsoft.QLinearConcat | ✅ |  |  |
| com.microsoft.QLinearGlobalAveragePool | ✅ |  |  |
| com.microsoft.QLinearLeakyRelu | ✅ |  |  |
| com.microsoft.QLinearMul | ✅ |  |  |
| com.microsoft.QLinearSigmoid | ✅ |  |  |
| com.microsoft.QLinearSoftmax | ✅ | [Notes](docs/operator-notes.md#softmax-and-logsoftmax) | `--replicate-ort-bugs` |
| com.microsoft.QLinearWhere | ✅ |  |  |
| com.microsoft.QMoE | ✅ |  |  |
| com.microsoft.QuantizeLinear | ✅ | [Notes](docs/operator-notes.md#quantization-quantizelinear-dequantizelinear) |  |
| com.microsoft.RotaryEmbedding | ✅ | [Notes](docs/operator-notes.md#attention-and-transformers) |  |
| com.microsoft.SampleOp | ✅ |  |  |
| com.microsoft.SkipLayerNormalization | ✅ |  | `--fp32-accumulation-strategy`, `--fp16-accumulation-strategy` |
| com.microsoft.SparseToDenseMatMul | ❌ |  |  |
| com.microsoft.Tokenizer | ✅ |  |  |
| com.microsoft.Trilu | ✅ |  |  |
| com.microsoft.UnfoldTensor | ✅ |  |  |
| com.microsoft.Unique | ✅ | [Notes](docs/operator-notes.md#dynamic-output-operators-unique-nonzero-compress) |  |
| com.microsoft.WordConvEmbedding | ✅ |  |  |

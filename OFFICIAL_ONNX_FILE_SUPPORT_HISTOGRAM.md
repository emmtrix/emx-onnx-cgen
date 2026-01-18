# Error frequency

| Error message | Count | Histogram |
| --- | --- | --- |
| Testbench execution failed:  | 5 | ██████████████████████████████ |
| And expects identical input/output shapes | 5 | ██████████████████████████████ |
| Testbench execution failed: exit code -11 (signal 11: SIGSEGV) | 4 | ████████████████████████ |
| Unsupported op AffineGrid | 4 | ████████████████████████ |
| Unsupported op If | 4 | ████████████████████████ |
| Unsupported elem_type 8 (STRING) for tensor '*'. | 4 | ████████████████████████ |
| Unsupported op Adagrad | 2 | ████████████ |
| Unsupported op Adam | 2 | ████████████ |
| Unsupported op TreeEnsemble | 2 | ████████████ |
| Out of tolerance (max ULP 4294967295) | 2 | ████████████ |
| Where output shape must be (1, 1), got (1,) | 2 | ████████████ |
| Out of tolerance (max ULP 2143208269) | 1 | ██████ |
| Unsupported op ArrayFeatureExtractor | 1 | ██████ |
| Unsupported op Binarizer | 1 | ██████ |

## Local ONNX file support histogram

### Error frequency

| Error message | Count | Histogram |
| --- | --- | --- |
| Unsupported op ScatterND | 4 | ██████████████████████████████ |
| Unsupported LSTM direction b'*' | 2 | ███████████████ |
| Unsupported op QLinearAdd | 2 | ███████████████ |
| Unsupported op QLinearMul | 2 | ███████████████ |
| Gemm bias input must be broadcastable to output shape, got (2,) vs (2, 4) | 1 | ████████ |
| Out of tolerance (max ULP 591626278) | 1 | ████████ |
| ONNX Runtime failed to run onnx2c-org/test/local_ops/test_resize_downsample_sizes_linear_1D/model.onnx: [ONNXRuntimeError] : 10 : INVALID_GRAPH : This is an invalid model. In Node, ("sclbl-onnx-node1", Resize, "", -1) : ("X": tensor(float),"","","sizes": tensor(int64),) -> ("Y": tensor(float),) , Error Node (sclbl-onnx-node1)'s input 1 is marked single but has an empty string in the graph | 1 | ████████ |
| ONNX Runtime failed to run onnx2c-org/test/local_ops/test_resize_downsample_sizes_linear_1D_align/model.onnx: [ONNXRuntimeError] : 10 : INVALID_GRAPH : This is an invalid model. In Node, ("sclbl-onnx-node1", Resize, "", -1) : ("X": tensor(float),"","","sizes": tensor(int64),) -> ("Y": tensor(float),) , Error Node (sclbl-onnx-node1)'s input 1 is marked single but has an empty string in the graph | 1 | ████████ |

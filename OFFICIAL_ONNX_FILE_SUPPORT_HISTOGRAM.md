# Error frequency

| Error message | Count | Histogram |
| --- | --- | --- |
| Testbench execution failed:  | 9 | ██████████████████████████████ |
| And expects identical input/output shapes | 5 | █████████████████ |
| Unsupported op AffineGrid | 4 | █████████████ |
| Unsupported op If | 4 | █████████████ |
| Unsupported elem_type 8 (STRING) for tensor '*'. | 4 | █████████████ |
| Unsupported op Adagrad | 2 | ███████ |
| Unsupported op Adam | 2 | ███████ |
| Unsupported op TreeEnsemble | 2 | ███████ |
| Where output shape must be (1, 1), got (1,) | 2 | ███████ |
| 
Not equal to tolerance rtol=0.0001, atol=1e-05

Mismatched elements: 55 / 60 (91.7%)
Max absolute difference among violations: 3.490335
Max relative difference among violations: 84.4747
 ACTUAL: array([[[ 1.091592,  0.040604,  0.165592,  0.514611,  2.044984],
        [-0.977278,  0.950088, -0.151357,  1.660834,  0.810756],
        [ 1.122782,  3.695167,  2.628596, -0.855603,  1.393952],...
 DESIRED: array([[[ 1.091592,  0.040604,  0.165592,  0.514611,  2.044984],
        [-1.649738,  0.590535, -0.964504, -1.829501,  0.588025],
        [-0.528417,  1.09472 , -0.052109, -1.604608,  0.621289],... | 1 | ███ |
| Unsupported op ArrayFeatureExtractor | 1 | ███ |
| Unsupported op Binarizer | 1 | ███ |
| 
Not equal to tolerance rtol=0.0001, atol=1e-05

nan location mismatch:
 ACTUAL: array([[[nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
         nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
        [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,...
 DESIRED: array([[[0.216069, 0.568473, 0.478865, 0.274843, 0.307853, 0.475815,
         0.316015, 0.479875, 0.704013, 0.482877, 0.520319, 0.628602,
         0.370084, 0.575148, 0.582313, 0.681792, 0.646259, 0.418965,... | 1 | ███ |
| 
Not equal to tolerance rtol=0.0001, atol=1e-05

nan location mismatch:
 ACTUAL: array([[[nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
         nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
         nan, nan, nan, nan, nan, nan],...
 DESIRED: array([[[0.313025, 0.256659, 0.415442, 0.438531, 0.707696, 0.54395 ,
         0.253743, 0.544114, 0.597431, 0.692815, 0.471582, 0.519003,
         0.636791, 0.309616, 0.413838, 0.251433, 0.391316, 0.472403,... | 1 | ███ |

## Local ONNX file support histogram

### Error frequency

| Error message | Count | Histogram |
| --- | --- | --- |
| Unsupported op ScatterND | 4 | ██████████████████████████████ |
| Unsupported LSTM direction b'*' | 2 | ███████████████ |
| Unsupported op QLinearAdd | 2 | ███████████████ |
| Unsupported op QLinearMul | 2 | ███████████████ |
| Gemm bias input must be broadcastable to output shape, got (2,) vs (2, 4) | 1 | ████████ |
| ONNX Runtime failed to run onnx2c-org/test/local_ops/test_resize_downsample_sizes_linear_1D/model.onnx: [ONNXRuntimeError] : 10 : INVALID_GRAPH : This is an invalid model. In Node, ("sclbl-onnx-node1", Resize, "", -1) : ("X": tensor(float),"","","sizes": tensor(int64),) -> ("Y": tensor(float),) , Error Node (sclbl-onnx-node1)'s input 1 is marked single but has an empty string in the graph | 1 | ████████ |
| ONNX Runtime failed to run onnx2c-org/test/local_ops/test_resize_downsample_sizes_linear_1D_align/model.onnx: [ONNXRuntimeError] : 10 : INVALID_GRAPH : This is an invalid model. In Node, ("sclbl-onnx-node1", Resize, "", -1) : ("X": tensor(float),"","","sizes": tensor(int64),) -> ("Y": tensor(float),) , Error Node (sclbl-onnx-node1)'s input 1 is marked single but has an empty string in the graph | 1 | ████████ |

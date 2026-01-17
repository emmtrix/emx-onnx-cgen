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
Max absolute difference among violations: 0.9259485
Max relative difference among violations: 1.6373023
 ACTUAL: array([[[1.47981 , 1.349239, 1.24502 , 0.787559, 0.45875 ],
        [0.447105, 0.008127, 0.849414, 1.100449, 1.115628],
        [0.832698, 1.169886, 1.395116, 0.46017 , 0.498977],...
 DESIRED: array([[[1.47981 , 1.349239, 1.24502 , 0.787559, 0.45875 ],
        [1.366137, 0.759867, 1.775362, 0.893475, 0.56212 ],
        [1.432659, 1.487871, 1.906305, 0.366869, 0.534841],... | 1 | ███ |
| Unsupported op ArrayFeatureExtractor | 1 | ███ |
| Unsupported op Binarizer | 1 | ███ |
| 
Not equal to tolerance rtol=0.0001, atol=1e-05

nan location mismatch:
 ACTUAL: array([[[nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
         nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
        [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,...
 DESIRED: array([[[0.397623, 0.462622, 0.446816, 0.56216 , 0.446149, 0.602571,
         0.572337, 0.521388, 0.543304, 0.57788 , 0.74103 , 0.75054 ,
         0.424829, 0.643997, 0.382766, 0.392968, 0.570121, 0.342225,... | 1 | ███ |
| 
Not equal to tolerance rtol=0.0001, atol=1e-05

nan location mismatch:
 ACTUAL: array([[[nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
         nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
         nan, nan, nan, nan, nan, nan],...
 DESIRED: array([[[0.598068, 0.555451, 0.475542, 0.572956, 0.524081, 0.611961,
         0.513773, 0.337044, 0.523152, 0.589548, 0.475683, 0.597341,
         0.380195, 0.432963, 0.410573, 0.56287 , 0.731879, 0.314342,... | 1 | ███ |

## Local ONNX file support histogram

### Error frequency

| Error message | Count | Histogram |
| --- | --- | --- |
| Unsupported op ScatterND | 4 | ██████████████████████████████ |
| Unsupported LSTM direction b'*' | 2 | ███████████████ |
| Unsupported op QLinearAdd | 2 | ███████████████ |
| Unsupported op QLinearMul | 2 | ███████████████ |
| Gemm bias input must be broadcastable to output shape, got (2,) vs (2, 4) | 1 | ████████ |

<!-- AUTO-GENERATED FILE. DO NOT EDIT. -->
<!-- Regenerate with: UPDATE_REFS=1 pytest -q tests/test_official_onnx_files_docs.py::test_official_onnx_file_support_doc -->

# Error frequency

Aggregates non-success verification outcomes.

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
| Out of tolerance | 6 | 15, 20, 22 |
| OptionalHasElement expects exactly one non-empty input. | 4 | 18 |
| Unsupported elem_type 24 (FLOAT8E8M0) for tensor '*'. | 4 | 25 |
| Unsupported op RNN | 4 | 22 |
| Unsupported op RandomUniformLike | 3 | 22 |
| Unsupported op RegexFullMatch | 3 | 20 |
| Unsupported op RoiAlign | 3 | 22 |
| '*' | 2 |  |
| QuantizeLinear block_size is not supported | 2 | 25 |
| ThresholdedRelu only supports alpha=1.0 | 2 | 22 |
| Unsupported op Adam | 2 |  |
| Unsupported op Det | 2 | 22 |
| Unsupported op MaxUnpool | 2 | 22 |
| Graph must contain at least one node | 1 | 25 |
| Pad value input must be a scalar | 1 | 24 |
| Unsupported op ArrayFeatureExtractor | 1 |  |
| Unsupported op Binarizer | 1 |  |
| Unsupported op Loop | 1 | 16 |
| Unsupported op RandomUniform | 1 | 22 |
| onnxruntime failed to run onnx2c-org/test/local_ops/test_gemm_CM_transA/model.onnx: [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Non-zero status code returned while running Gemm node. Name:'*' Status Message: Gemm: Invalid bias shape for broadcast | 1 | 12 |

## Error frequency by opset

| Error message | Opset | Count |
| --- | --- | --- |
| onnxruntime failed to run onnx2c-org/test/local_ops/test_gemm_CM_transA/model.onnx: [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Non-zero status code returned while running Gemm node. Name:'*' Status Message: Gemm: Invalid bias shape for broadcast | 12 | 1 |
| Out of tolerance | 15 | 1 |
| Unsupported op Loop | 16 | 1 |
| OptionalHasElement expects exactly one non-empty input. | 18 | 4 |
| Unsupported op RegexFullMatch | 20 | 3 |
| Out of tolerance | 20 | 2 |
| Unsupported op RNN | 22 | 4 |
| Out of tolerance | 22 | 3 |
| Unsupported op RandomUniformLike | 22 | 3 |
| Unsupported op RoiAlign | 22 | 3 |
| Unsupported op Det | 22 | 2 |
| Unsupported op MaxUnpool | 22 | 2 |
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
| Graph must contain at least one node | 25 | 1 |

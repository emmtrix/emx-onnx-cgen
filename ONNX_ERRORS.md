<!-- AUTO-GENERATED FILE. DO NOT EDIT. -->
<!-- Regenerate with: UPDATE_REFS=1 pytest -q tests/test_official_onnx_files_docs.py::test_official_onnx_file_support_doc -->

# ONNX verification errors

Aggregates non-success verification outcomes.

| Error message | Count | Opset versions |
| --- | --- | --- |
| Unsupported test-data sequence input for verify: variable sequence element shapes are not supported | 11 | 9, 11, 17 |
| Out of tolerance | 1 |  |
| onnxruntime failed to run onnx2c-org/test/local_ops/test_gemm_CM_transA/model.onnx: [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Non-zero status code returned while running Gemm node. Name:'*' Status Message: Gemm: Invalid bias shape for broadcast | 1 | 12 |

## Error frequency by opset

| Error message | Opset | Count |
| --- | --- | --- |
| Unsupported test-data sequence input for verify: variable sequence element shapes are not supported | 9 | 1 |
| Unsupported test-data sequence input for verify: variable sequence element shapes are not supported | 11 | 2 |
| onnxruntime failed to run onnx2c-org/test/local_ops/test_gemm_CM_transA/model.onnx: [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Non-zero status code returned while running Gemm node. Name:'*' Status Message: Gemm: Invalid bias shape for broadcast | 12 | 1 |
| Unsupported test-data sequence input for verify: variable sequence element shapes are not supported | 17 | 8 |

## Failing ONNX files

Lists every ONNX file with a non-success verification outcome.

| File | Opset | Verification | Supported | Error |
| --- | --- | --- | --- | --- |
| local_ops/test_gemm_CM_transA/model.onnx | 12 | Data/Data | ❌ | onnxruntime failed to run onnx2c-org/test/local_ops/test_gemm_CM_transA/model.onnx: [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Non-zero status code returned while running Gemm node. Name:'sclbl-onnx-node1' Status Message: Gemm: Invalid bias shape for broadcast |
| mnist/pytorch.onnx | 9 | Data/Data | ❌ | Unsupported test-data sequence input for verify: variable sequence element shapes are not supported |
| node/test_adam_multiple/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 62311) |
| node/test_sequence_insert_at_back/model.onnx | 11 | Data/Data | ❌ | Unsupported test-data sequence input for verify: variable sequence element shapes are not supported |
| node/test_sequence_insert_at_front/model.onnx | 11 | Data/Data | ❌ | Unsupported test-data sequence input for verify: variable sequence element shapes are not supported |
| node/test_sequence_map_add_2_sequences/model.onnx | 17 | Data/Data | ❌ | Unsupported test-data sequence input for verify: variable sequence element shapes are not supported |
| node/test_sequence_map_add_2_sequences_expanded/model.onnx | 17 | Data/Data | ❌ | Unsupported test-data sequence input for verify: variable sequence element shapes are not supported |
| node/test_sequence_map_extract_shapes/model.onnx | 17 | Data/Data | ❌ | Unsupported test-data sequence input for verify: variable sequence element shapes are not supported |
| node/test_sequence_map_extract_shapes_expanded/model.onnx | 17 | Data/Data | ❌ | Unsupported test-data sequence input for verify: variable sequence element shapes are not supported |
| node/test_sequence_map_identity_1_sequence_1_tensor/model.onnx | 17 | Data/Data | ❌ | Unsupported test-data sequence input for verify: variable sequence element shapes are not supported |
| node/test_sequence_map_identity_1_sequence_1_tensor_expanded/model.onnx | 17 | Data/Data | ❌ | Unsupported test-data sequence input for verify: variable sequence element shapes are not supported |
| node/test_sequence_map_identity_2_sequences/model.onnx | 17 | Data/Data | ❌ | Unsupported test-data sequence input for verify: variable sequence element shapes are not supported |
| node/test_sequence_map_identity_2_sequences_expanded/model.onnx | 17 | Data/Data | ❌ | Unsupported test-data sequence input for verify: variable sequence element shapes are not supported |

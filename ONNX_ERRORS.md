<!-- AUTO-GENERATED FILE. DO NOT EDIT. -->
<!-- Regenerate with: UPDATE_REFS=1 pytest -q tests/test_official_onnx_files_docs.py::test_official_onnx_file_support_doc -->

# ONNX verification errors

Aggregates non-success verification outcomes.

| Error message | Count | Opset versions |
| --- | --- | --- |
| Testbench execution failed: exit code 1 | 2 | 17 |
| Out of tolerance | 1 |  |
| Unsupported op Loop | 1 | 16 |
| Unsupported test-data sequence input for verify: variable sequence element shapes are not supported | 1 | 9 |
| onnxruntime failed to run onnx2c-org/test/local_ops/test_gemm_CM_transA/model.onnx: [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Non-zero status code returned while running Gemm node. Name:'*' Status Message: Gemm: Invalid bias shape for broadcast | 1 | 12 |

## Error frequency by opset

| Error message | Opset | Count |
| --- | --- | --- |
| Unsupported test-data sequence input for verify: variable sequence element shapes are not supported | 9 | 1 |
| onnxruntime failed to run onnx2c-org/test/local_ops/test_gemm_CM_transA/model.onnx: [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Non-zero status code returned while running Gemm node. Name:'*' Status Message: Gemm: Invalid bias shape for broadcast | 12 | 1 |
| Unsupported op Loop | 16 | 1 |
| Testbench execution failed: exit code 1 | 17 | 2 |

## Failing ONNX files

Lists every ONNX file with a non-success verification outcome.

| File | Opset | Verification | Supported | Error |
| --- | --- | --- | --- | --- |
| local_ops/test_gemm_CM_transA/model.onnx | 12 | Data/Data | ❌ | onnxruntime failed to run onnx2c-org/test/local_ops/test_gemm_CM_transA/model.onnx: [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Non-zero status code returned while running Gemm node. Name:'sclbl-onnx-node1' Status Message: Gemm: Invalid bias shape for broadcast |
| mnist/pytorch.onnx | 9 | Data/Data | ❌ | Unsupported test-data sequence input for verify: variable sequence element shapes are not supported |
| node/test_adam_multiple/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 62311) |
| node/test_loop16_seq_none/model.onnx | 16 | Data/Data | ❌ | Unsupported op Loop |
| node/test_sequence_map_identity_1_sequence_1_tensor/model.onnx (--sequence-element-shape x0=[<=9]) | 17 | Data/Data | ❌ | Testbench execution failed: exit code 1 |
| node/test_sequence_map_identity_1_sequence_1_tensor_expanded/model.onnx (--sequence-element-shape x0=[<=9]) | 17 | Data/Data | ❌ | Testbench execution failed: exit code 1 |

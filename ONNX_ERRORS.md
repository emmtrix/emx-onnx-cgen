<!-- AUTO-GENERATED FILE. DO NOT EDIT. -->
<!-- Regenerate with: UPDATE_REFS=1 pytest -q tests/test_official_onnx_files_docs.py::test_official_onnx_file_support_doc -->

# ONNX verification errors

Aggregates non-success verification outcomes.

| Error message | Count | Opset versions |
| --- | --- | --- |
| Code generation requires explicit ragged-sequence bounds. Reason: sequence '*' has unknown or dynamic element dimensions. Hint: pass --sequence-element-shape x=[...] | 3 | 17, 25 |
| Code generation requires explicit ragged-sequence bounds. Reason: sequence '*' has unknown or dynamic element dimensions. Hint: pass --sequence-element-shape x0=[...] | 2 | 17 |
| Out of tolerance | 2 | 12 |
| Output shape mismatch for seq[0] (actual_shape=(0, 10), actual_size=0, expected_shape=(3, 2), expected_size=6, output=seq[0]) | 1 | 24 |
| Output shape mismatch for seq[0] (actual_shape=(0, 6), actual_size=0, expected_shape=(1, 6), expected_size=6, output=seq[0]) | 1 | 24 |
| Sequence length mismatch for seq_res (actual_count=10, expected_count=5, output=seq_res) | 1 | 13 |
| Sequence length mismatch for seq_res (actual_count=5, expected_count=6, output=seq_res) | 1 | 16 |
| Unsupported test-data sequence input for verify: variable sequence element shapes are not supported | 1 | 9 |
| onnxruntime failed to run onnx2c-org/test/local_ops/test_gemm_CM_transA/model.onnx: [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Non-zero status code returned while running Gemm node. Name:'*' Status Message: Gemm: Invalid bias shape for broadcast | 1 | 12 |

## Error frequency by opset

| Error message | Opset | Count |
| --- | --- | --- |
| Unsupported test-data sequence input for verify: variable sequence element shapes are not supported | 9 | 1 |
| Out of tolerance | 12 | 1 |
| onnxruntime failed to run onnx2c-org/test/local_ops/test_gemm_CM_transA/model.onnx: [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Non-zero status code returned while running Gemm node. Name:'*' Status Message: Gemm: Invalid bias shape for broadcast | 12 | 1 |
| Sequence length mismatch for seq_res (actual_count=10, expected_count=5, output=seq_res) | 13 | 1 |
| Sequence length mismatch for seq_res (actual_count=5, expected_count=6, output=seq_res) | 16 | 1 |
| Code generation requires explicit ragged-sequence bounds. Reason: sequence '*' has unknown or dynamic element dimensions. Hint: pass --sequence-element-shape x0=[...] | 17 | 2 |
| Code generation requires explicit ragged-sequence bounds. Reason: sequence '*' has unknown or dynamic element dimensions. Hint: pass --sequence-element-shape x=[...] | 17 | 2 |
| Output shape mismatch for seq[0] (actual_shape=(0, 10), actual_size=0, expected_shape=(3, 2), expected_size=6, output=seq[0]) | 24 | 1 |
| Output shape mismatch for seq[0] (actual_shape=(0, 6), actual_size=0, expected_shape=(1, 6), expected_size=6, output=seq[0]) | 24 | 1 |
| Code generation requires explicit ragged-sequence bounds. Reason: sequence '*' has unknown or dynamic element dimensions. Hint: pass --sequence-element-shape x=[...] | 25 | 1 |

## Failing ONNX files

Lists every ONNX file with a non-success verification outcome.

| File | Opset | Verification | Supported | Error |
| --- | --- | --- | --- | --- |
| local_ops/test_gemm_CM_transA/model.onnx | 12 | Data/Data | ❌ | onnxruntime failed to run onnx2c-org/test/local_ops/test_gemm_CM_transA/model.onnx: [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Non-zero status code returned while running Gemm node. Name:'sclbl-onnx-node1' Status Message: Gemm: Invalid bias shape for broadcast |
| mnist/pytorch.onnx | 9 | Data/Data | ❌ | Unsupported test-data sequence input for verify: variable sequence element shapes are not supported |
| node/test_adam_multiple/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 62311) |
| node/test_identity_sequence/model.onnx | 25 | Data/Data | ❌ | Code generation requires explicit ragged-sequence bounds. Reason: sequence 'x' has unknown or dynamic element dimensions. Hint: pass --sequence-element-shape x=[...] |
| node/test_loop13_seq/model.onnx | 13 | Data/Data | ❌ | Sequence length mismatch for seq_res (actual_count=10, expected_count=5, output=seq_res) |
| node/test_loop16_seq_none/model.onnx | 16 | Data/Data | ❌ | Sequence length mismatch for seq_res (actual_count=5, expected_count=6, output=seq_res) |
| node/test_sequence_map_add_1_sequence_1_tensor/model.onnx | 17 | Data/Data | ❌ | Code generation requires explicit ragged-sequence bounds. Reason: sequence 'x0' has unknown or dynamic element dimensions. Hint: pass --sequence-element-shape x0=[...] |
| node/test_sequence_map_add_1_sequence_1_tensor_expanded/model.onnx | 17 | Data/Data | ❌ | Code generation requires explicit ragged-sequence bounds. Reason: sequence 'x0' has unknown or dynamic element dimensions. Hint: pass --sequence-element-shape x0=[...] |
| node/test_sequence_map_identity_1_sequence/model.onnx | 17 | Data/Data | ❌ | Code generation requires explicit ragged-sequence bounds. Reason: sequence 'x' has unknown or dynamic element dimensions. Hint: pass --sequence-element-shape x=[...] |
| node/test_sequence_map_identity_1_sequence_expanded/model.onnx | 17 | Data/Data | ❌ | Code generation requires explicit ragged-sequence bounds. Reason: sequence 'x' has unknown or dynamic element dimensions. Hint: pass --sequence-element-shape x=[...] |
| node/test_split_to_sequence_1/model.onnx | 24 | Data/Data | ❌ | Output shape mismatch for seq[0] (actual_shape=(0, 10), actual_size=0, expected_shape=(3, 2), expected_size=6, output=seq[0]) |
| node/test_split_to_sequence_2/model.onnx | 24 | Data/Data | ❌ | Output shape mismatch for seq[0] (actual_shape=(0, 6), actual_size=0, expected_shape=(1, 6), expected_size=6, output=seq[0]) |
| simple/test_sequence_model1/model.onnx | 12 | Data/Data | ❌ | Out of tolerance (max ULP 2007321857) |

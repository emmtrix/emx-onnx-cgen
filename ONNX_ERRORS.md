<!-- AUTO-GENERATED FILE. DO NOT EDIT. -->
<!-- Regenerate with: UPDATE_REFS=1 pytest -q tests/test_official_onnx_files_docs.py::test_official_onnx_file_support_doc -->

# ONNX verification errors

Aggregates non-success verification outcomes.

| Error message | Count | Opset versions |
| --- | --- | --- |
| Test data inputs require normalization/reshaping for the testbench format, so verify would no longer compare against the official output_*.pb fixtures. Re-export the model with static shapes or rerun with --test-data-inputs-only to opt into runtime-reference verification explicitly. | 11 | 9, 11, 17 |
| Out of tolerance | 1 |  |
| onnxruntime failed to run onnx2c-org/test/local_ops/test_gemm_CM_transA/model.onnx: [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Non-zero status code returned while running Gemm node. Name:'*' Status Message: Gemm: Invalid bias shape for broadcast | 1 | 12 |

## Error frequency by opset

| Error message | Opset | Count |
| --- | --- | --- |
| Test data inputs require normalization/reshaping for the testbench format, so verify would no longer compare against the official output_*.pb fixtures. Re-export the model with static shapes or rerun with --test-data-inputs-only to opt into runtime-reference verification explicitly. | 9 | 1 |
| Test data inputs require normalization/reshaping for the testbench format, so verify would no longer compare against the official output_*.pb fixtures. Re-export the model with static shapes or rerun with --test-data-inputs-only to opt into runtime-reference verification explicitly. | 11 | 2 |
| onnxruntime failed to run onnx2c-org/test/local_ops/test_gemm_CM_transA/model.onnx: [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Non-zero status code returned while running Gemm node. Name:'*' Status Message: Gemm: Invalid bias shape for broadcast | 12 | 1 |
| Test data inputs require normalization/reshaping for the testbench format, so verify would no longer compare against the official output_*.pb fixtures. Re-export the model with static shapes or rerun with --test-data-inputs-only to opt into runtime-reference verification explicitly. | 17 | 8 |

## Failing ONNX files

Lists every ONNX file with a non-success verification outcome.

| File | Opset | Verification | Supported | Error |
| --- | --- | --- | --- | --- |
| local_ops/test_gemm_CM_transA/model.onnx | 12 | Data/Data | ❌ | onnxruntime failed to run onnx2c-org/test/local_ops/test_gemm_CM_transA/model.onnx: [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Non-zero status code returned while running Gemm node. Name:'sclbl-onnx-node1' Status Message: Gemm: Invalid bias shape for broadcast |
| mnist/pytorch.onnx | 9 | Data/Data | ❌ | Test data inputs require normalization/reshaping for the testbench format, so verify would no longer compare against the official output_*.pb fixtures. Re-export the model with static shapes or rerun with --test-data-inputs-only to opt into runtime-reference verification explicitly. |
| node/test_adam_multiple/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 62311) |
| node/test_sequence_insert_at_back/model.onnx | 11 | Data/Data | ❌ | Test data inputs require normalization/reshaping for the testbench format, so verify would no longer compare against the official output_*.pb fixtures. Re-export the model with static shapes or rerun with --test-data-inputs-only to opt into runtime-reference verification explicitly. |
| node/test_sequence_insert_at_front/model.onnx | 11 | Data/Data | ❌ | Test data inputs require normalization/reshaping for the testbench format, so verify would no longer compare against the official output_*.pb fixtures. Re-export the model with static shapes or rerun with --test-data-inputs-only to opt into runtime-reference verification explicitly. |
| node/test_sequence_map_add_2_sequences/model.onnx | 17 | Data/Data | ❌ | Test data inputs require normalization/reshaping for the testbench format, so verify would no longer compare against the official output_*.pb fixtures. Re-export the model with static shapes or rerun with --test-data-inputs-only to opt into runtime-reference verification explicitly. |
| node/test_sequence_map_add_2_sequences_expanded/model.onnx | 17 | Data/Data | ❌ | Test data inputs require normalization/reshaping for the testbench format, so verify would no longer compare against the official output_*.pb fixtures. Re-export the model with static shapes or rerun with --test-data-inputs-only to opt into runtime-reference verification explicitly. |
| node/test_sequence_map_extract_shapes/model.onnx | 17 | Data/Data | ❌ | Test data inputs require normalization/reshaping for the testbench format, so verify would no longer compare against the official output_*.pb fixtures. Re-export the model with static shapes or rerun with --test-data-inputs-only to opt into runtime-reference verification explicitly. |
| node/test_sequence_map_extract_shapes_expanded/model.onnx | 17 | Data/Data | ❌ | Test data inputs require normalization/reshaping for the testbench format, so verify would no longer compare against the official output_*.pb fixtures. Re-export the model with static shapes or rerun with --test-data-inputs-only to opt into runtime-reference verification explicitly. |
| node/test_sequence_map_identity_1_sequence_1_tensor/model.onnx | 17 | Data/Data | ❌ | Test data inputs require normalization/reshaping for the testbench format, so verify would no longer compare against the official output_*.pb fixtures. Re-export the model with static shapes or rerun with --test-data-inputs-only to opt into runtime-reference verification explicitly. |
| node/test_sequence_map_identity_1_sequence_1_tensor_expanded/model.onnx | 17 | Data/Data | ❌ | Test data inputs require normalization/reshaping for the testbench format, so verify would no longer compare against the official output_*.pb fixtures. Re-export the model with static shapes or rerun with --test-data-inputs-only to opt into runtime-reference verification explicitly. |
| node/test_sequence_map_identity_2_sequences/model.onnx | 17 | Data/Data | ❌ | Test data inputs require normalization/reshaping for the testbench format, so verify would no longer compare against the official output_*.pb fixtures. Re-export the model with static shapes or rerun with --test-data-inputs-only to opt into runtime-reference verification explicitly. |
| node/test_sequence_map_identity_2_sequences_expanded/model.onnx | 17 | Data/Data | ❌ | Test data inputs require normalization/reshaping for the testbench format, so verify would no longer compare against the official output_*.pb fixtures. Re-export the model with static shapes or rerun with --test-data-inputs-only to opt into runtime-reference verification explicitly. |

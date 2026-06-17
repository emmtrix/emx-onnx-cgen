<!-- AUTO-GENERATED FILE. DO NOT EDIT. -->
<!-- Regenerate with: UPDATE_REFS=1 pytest -q tests/test_official_onnx_files_docs.py::test_official_onnx_file_support_doc -->

# ONNX verification errors

Aggregates non-success verification outcomes.

| Error message | Count | Opset versions |
| --- | --- | --- |
| Unsupported value type '*' for '*'. Hint: export the model with tensor inputs/outputs. | 8 |  |
| Out of tolerance | 5 |  |

## Failing ONNX files

Lists every ONNX file with a non-success verification outcome.

| File | Opset | Verification | Supported | Error |
| --- | --- | --- | --- | --- |
| test/contrib_ops/attention_op_test/AttentionPastState_dynamic_run0/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 86151) |
| test/contrib_ops/attention_op_test/Attention_Mask1D_Fp32_B2_S64_run0/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 1020241) |
| test/contrib_ops/attention_op_test/Attention_Mask2D_Fp32_B2_S32_run0/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 980755) |
| test/contrib_ops/math/matmul_sparse_test/TestCoo_run0/model.onnx |  | Data/Data | ❌ | Unsupported value type 'sparse_tensor_type' for 'A'. Hint: export the model with tensor inputs/outputs. |
| test/contrib_ops/math/matmul_sparse_test/TestCoo_run1/model.onnx |  | Data/Data | ❌ | Unsupported value type 'sparse_tensor_type' for 'A'. Hint: export the model with tensor inputs/outputs. |
| test/contrib_ops/math/matmul_sparse_test/TestCoo_run2/model.onnx |  | Data/Data | ❌ | Unsupported value type 'sparse_tensor_type' for 'A'. Hint: export the model with tensor inputs/outputs. |
| test/contrib_ops/math/matmul_sparse_test/TestCoo_run3/model.onnx |  | Data/Data | ❌ | Unsupported value type 'sparse_tensor_type' for 'A'. Hint: export the model with tensor inputs/outputs. |
| test/contrib_ops/math/matmul_sparse_test/TestCsr_run0/model.onnx |  | Data/Data | ❌ | Unsupported value type 'sparse_tensor_type' for 'A'. Hint: export the model with tensor inputs/outputs. |
| test/contrib_ops/math/matmul_sparse_test/TestCsr_run1/model.onnx |  | Data/Data | ❌ | Unsupported value type 'sparse_tensor_type' for 'A'. Hint: export the model with tensor inputs/outputs. |
| test/contrib_ops/math/matmul_sparse_test/TestCsr_run2/model.onnx |  | Data/Data | ❌ | Unsupported value type 'sparse_tensor_type' for 'A'. Hint: export the model with tensor inputs/outputs. |
| test/contrib_ops/math/matmul_sparse_test/TestCsr_run3/model.onnx |  | Data/Data | ❌ | Unsupported value type 'sparse_tensor_type' for 'A'. Hint: export the model with tensor inputs/outputs. |
| test/contrib_ops/multihead_attention_op_test/CrossAttention_WithPastPassedInDirectly_NoMask_run0/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 64137) |
| test/contrib_ops/multihead_attention_op_test/CrossAttention_WithPastPassedInDirectly_NoMask_run1/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 64137) |

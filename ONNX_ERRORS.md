<!-- AUTO-GENERATED FILE. DO NOT EDIT. -->
<!-- Regenerate with: UPDATE_REFS=1 pytest -q tests/test_official_onnx_files_docs.py::test_official_onnx_file_support_doc -->

# ONNX verification errors

Aggregates non-success verification outcomes.

| Error message | Count | Opset versions |
| --- | --- | --- |
| Unsupported value type '*' for '*'. Hint: export the model with tensor inputs/outputs. | 8 |  |

## Failing ONNX files

Lists every ONNX file with a non-success verification outcome.

| File | Opset | Verification | Supported | Error |
| --- | --- | --- | --- | --- |
| test/contrib_ops/math/matmul_sparse_test/TestCoo_run0/model.onnx |  | Data/Data | ❌ | Unsupported value type 'sparse_tensor_type' for 'A'. Hint: export the model with tensor inputs/outputs. |
| test/contrib_ops/math/matmul_sparse_test/TestCoo_run1/model.onnx |  | Data/Data | ❌ | Unsupported value type 'sparse_tensor_type' for 'A'. Hint: export the model with tensor inputs/outputs. |
| test/contrib_ops/math/matmul_sparse_test/TestCoo_run2/model.onnx |  | Data/Data | ❌ | Unsupported value type 'sparse_tensor_type' for 'A'. Hint: export the model with tensor inputs/outputs. |
| test/contrib_ops/math/matmul_sparse_test/TestCoo_run3/model.onnx |  | Data/Data | ❌ | Unsupported value type 'sparse_tensor_type' for 'A'. Hint: export the model with tensor inputs/outputs. |
| test/contrib_ops/math/matmul_sparse_test/TestCsr_run0/model.onnx |  | Data/Data | ❌ | Unsupported value type 'sparse_tensor_type' for 'A'. Hint: export the model with tensor inputs/outputs. |
| test/contrib_ops/math/matmul_sparse_test/TestCsr_run1/model.onnx |  | Data/Data | ❌ | Unsupported value type 'sparse_tensor_type' for 'A'. Hint: export the model with tensor inputs/outputs. |
| test/contrib_ops/math/matmul_sparse_test/TestCsr_run2/model.onnx |  | Data/Data | ❌ | Unsupported value type 'sparse_tensor_type' for 'A'. Hint: export the model with tensor inputs/outputs. |
| test/contrib_ops/math/matmul_sparse_test/TestCsr_run3/model.onnx |  | Data/Data | ❌ | Unsupported value type 'sparse_tensor_type' for 'A'. Hint: export the model with tensor inputs/outputs. |

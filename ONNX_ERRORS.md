<!-- AUTO-GENERATED FILE. DO NOT EDIT. -->
<!-- Regenerate with: UPDATE_REFS=1 pytest -q tests/test_official_onnx_files_docs.py::test_official_onnx_file_support_doc -->

# ONNX verification errors

Aggregates non-success verification outcomes.

| Error message | Count | Opset versions |
| --- | --- | --- |
| Absolute diff only supports integer and bool dtypes, got object | 22 |  |
| Out of tolerance | 1 |  |
| onnxruntime failed to run onnx2c-org/test/local_ops/test_gemm_CM_transA/model.onnx: [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Non-zero status code returned while running Gemm node. Name:'*' Status Message: Gemm: Invalid bias shape for broadcast | 1 | 12 |

## Error frequency by opset

| Error message | Opset | Count |
| --- | --- | --- |
| onnxruntime failed to run onnx2c-org/test/local_ops/test_gemm_CM_transA/model.onnx: [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Non-zero status code returned while running Gemm node. Name:'*' Status Message: Gemm: Invalid bias shape for broadcast | 12 | 1 |

## Failing ONNX files

Lists every ONNX file with a non-success verification outcome.

| File | Opset | Verification | Supported | Error |
| --- | --- | --- | --- | --- |
| local_ops/test_gemm_CM_transA/model.onnx | 12 | Data/Data | ❌ | onnxruntime failed to run onnx2c-org/test/local_ops/test_gemm_CM_transA/model.onnx: [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Non-zero status code returned while running Gemm node. Name:'sclbl-onnx-node1' Status Message: Gemm: Invalid bias shape for broadcast |
| node/test_adam_multiple/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 62311) |
| node/test_string_concat/model.onnx |  |  | ❌ | Absolute diff only supports integer and bool dtypes, got object |
| node/test_string_concat_broadcasting/model.onnx |  |  | ❌ | Absolute diff only supports integer and bool dtypes, got object |
| node/test_string_concat_empty_string/model.onnx |  |  | ❌ | Absolute diff only supports integer and bool dtypes, got object |
| node/test_string_concat_utf8/model.onnx |  |  | ❌ | Absolute diff only supports integer and bool dtypes, got object |
| node/test_string_concat_zero_dimensional/model.onnx |  |  | ❌ | Absolute diff only supports integer and bool dtypes, got object |
| node/test_string_split_basic/model.onnx |  |  | ❌ | Absolute diff only supports integer and bool dtypes, got object |
| node/test_string_split_consecutive_delimiters/model.onnx |  |  | ❌ | Absolute diff only supports integer and bool dtypes, got object |
| node/test_string_split_empty_string_delimiter/model.onnx |  |  | ❌ | Absolute diff only supports integer and bool dtypes, got object |
| node/test_string_split_maxsplit/model.onnx |  |  | ❌ | Absolute diff only supports integer and bool dtypes, got object |
| node/test_string_split_no_delimiter/model.onnx |  |  | ❌ | Absolute diff only supports integer and bool dtypes, got object |
| node/test_strnormalizer_export_monday_casesensintive_lower/model.onnx |  |  | ❌ | Absolute diff only supports integer and bool dtypes, got object |
| node/test_strnormalizer_export_monday_casesensintive_nochangecase/model.onnx |  |  | ❌ | Absolute diff only supports integer and bool dtypes, got object |
| node/test_strnormalizer_export_monday_casesensintive_upper/model.onnx |  |  | ❌ | Absolute diff only supports integer and bool dtypes, got object |
| node/test_strnormalizer_export_monday_empty_output/model.onnx |  |  | ❌ | Absolute diff only supports integer and bool dtypes, got object |
| node/test_strnormalizer_export_monday_insensintive_upper_twodim/model.onnx |  |  | ❌ | Absolute diff only supports integer and bool dtypes, got object |
| node/test_strnormalizer_nostopwords_nochangecase/model.onnx |  |  | ❌ | Absolute diff only supports integer and bool dtypes, got object |
| simple/test_strnorm_model_monday_casesensintive_lower/model.onnx |  |  | ❌ | Absolute diff only supports integer and bool dtypes, got object |
| simple/test_strnorm_model_monday_casesensintive_nochangecase/model.onnx |  |  | ❌ | Absolute diff only supports integer and bool dtypes, got object |
| simple/test_strnorm_model_monday_casesensintive_upper/model.onnx |  |  | ❌ | Absolute diff only supports integer and bool dtypes, got object |
| simple/test_strnorm_model_monday_empty_output/model.onnx |  |  | ❌ | Absolute diff only supports integer and bool dtypes, got object |
| simple/test_strnorm_model_monday_insensintive_upper_twodim/model.onnx |  |  | ❌ | Absolute diff only supports integer and bool dtypes, got object |
| simple/test_strnorm_model_nostopwords_nochangecase/model.onnx |  |  | ❌ | Absolute diff only supports integer and bool dtypes, got object |

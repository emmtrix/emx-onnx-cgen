<!-- AUTO-GENERATED FILE. DO NOT EDIT. -->
<!-- Regenerate with: UPDATE_REFS=1 pytest -q tests/test_official_onnx_files_docs.py::test_official_onnx_file_support_doc -->

# ONNX verification errors

Aggregates non-success verification outcomes.

| Error message | Count | Opset versions |
| --- | --- | --- |
| Output value mismatch for Y (got=hello, index=(0, 0), output=Y, reference=hello\x00world) | 1 |  |

## Failing ONNX files

Lists every ONNX file with a non-success verification outcome.

| File | Opset | Verification | Supported | Error |
| --- | --- | --- | --- | --- |
| test/contrib_ops/tokenizer_test/Tokenizer_EmbeddedNullBytes_run0/model.onnx |  | Data/Data | ❌ | Output value mismatch for Y (got=hello, index=(0, 0), output=Y, reference=hello\x00world) |

<!-- AUTO-GENERATED FILE. DO NOT EDIT. -->
<!-- Regenerate with: UPDATE_REFS=1 pytest -q tests/test_official_onnx_files_docs.py::test_official_onnx_file_support_doc -->

# ONNX verification errors

Aggregates non-success verification outcomes.

| Error message | Count | Opset versions |
| --- | --- | --- |
| Unsupported elem_type 23 (FLOAT4E2M1) for tensor '*'. | 14 | 25 |
| Out of tolerance | 8 | 22, 25 |
| DequantizeLinear supports int2/uint2/int4/uint4/int8/uint8/int16/uint16/int32/uint32 inputs only | 4 | 25 |
| QuantizeLinear supports int2/uint2/int4/uint4/int8/uint8/int16/uint16 outputs only | 2 | 25 |
| onnxruntime failed to run onnx2c-org/test/local_ops/test_gemm_CM_transA/model.onnx: [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Non-zero status code returned while running Gemm node. Name:'*' Status Message: Gemm: Invalid bias shape for broadcast | 1 | 12 |

## Error frequency by opset

| Error message | Opset | Count |
| --- | --- | --- |
| onnxruntime failed to run onnx2c-org/test/local_ops/test_gemm_CM_transA/model.onnx: [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Non-zero status code returned while running Gemm node. Name:'*' Status Message: Gemm: Invalid bias shape for broadcast | 12 | 1 |
| Out of tolerance | 22 | 1 |
| Unsupported elem_type 23 (FLOAT4E2M1) for tensor '*'. | 25 | 14 |
| Out of tolerance | 25 | 6 |
| DequantizeLinear supports int2/uint2/int4/uint4/int8/uint8/int16/uint16/int32/uint32 inputs only | 25 | 4 |
| QuantizeLinear supports int2/uint2/int4/uint4/int8/uint8/int16/uint16 outputs only | 25 | 2 |

## Failing ONNX files

Lists every ONNX file with a non-success verification outcome.

| File | Opset | Verification | Supported | Error |
| --- | --- | --- | --- | --- |
| local_ops/test_gemm_CM_transA/model.onnx | 12 | Data | ❌ | onnxruntime failed to run onnx2c-org/test/local_ops/test_gemm_CM_transA/model.onnx: [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Non-zero status code returned while running Gemm node. Name:'sclbl-onnx-node1' Status Message: Gemm: Invalid bias shape for broadcast |
| node/test_adam_multiple/model.onnx |  | Data | ❌ | Out of tolerance (max ULP 62311) |
| node/test_cast_FLOAT16_to_FLOAT4E2M1/model.onnx | 25 | Data | ❌ | Unsupported elem_type 23 (FLOAT4E2M1) for tensor 'output'. |
| node/test_cast_FLOAT4E2M1_to_FLOAT/model.onnx | 25 | Data | ❌ | Unsupported elem_type 23 (FLOAT4E2M1) for tensor 'input'. |
| node/test_cast_FLOAT4E2M1_to_FLOAT16/model.onnx | 25 | Data | ❌ | Unsupported elem_type 23 (FLOAT4E2M1) for tensor 'input'. |
| node/test_cast_FLOAT_to_FLOAT4E2M1/model.onnx | 25 | Data | ❌ | Unsupported elem_type 23 (FLOAT4E2M1) for tensor 'output'. |
| node/test_cast_no_saturate_FLOAT16_to_FLOAT8E5M2/model.onnx | 25 | Data | ❌ | Out of tolerance (max abs diff inf) |
| node/test_cast_no_saturate_FLOAT_to_FLOAT8E5M2/model.onnx | 25 | Data | ❌ | Out of tolerance (max abs diff inf) |
| node/test_castlike_FLOAT16_to_FLOAT4E2M1/model.onnx | 25 | Data | ❌ | Unsupported elem_type 23 (FLOAT4E2M1) for tensor 'like'. |
| node/test_castlike_FLOAT16_to_FLOAT4E2M1_expanded/model.onnx | 25 | Data | ❌ | Unsupported elem_type 23 (FLOAT4E2M1) for tensor 'like'. |
| node/test_castlike_FLOAT4E2M1_to_FLOAT/model.onnx | 25 | Data | ❌ | Unsupported elem_type 23 (FLOAT4E2M1) for tensor 'input'. |
| node/test_castlike_FLOAT4E2M1_to_FLOAT16/model.onnx | 25 | Data | ❌ | Unsupported elem_type 23 (FLOAT4E2M1) for tensor 'input'. |
| node/test_castlike_FLOAT4E2M1_to_FLOAT16_expanded/model.onnx | 25 | Data | ❌ | Unsupported elem_type 23 (FLOAT4E2M1) for tensor 'input'. |
| node/test_castlike_FLOAT4E2M1_to_FLOAT_expanded/model.onnx | 25 | Data | ❌ | Unsupported elem_type 23 (FLOAT4E2M1) for tensor 'input'. |
| node/test_castlike_FLOAT_to_FLOAT4E2M1/model.onnx | 25 | Data | ❌ | Unsupported elem_type 23 (FLOAT4E2M1) for tensor 'like'. |
| node/test_castlike_FLOAT_to_FLOAT4E2M1_expanded/model.onnx | 25 | Data | ❌ | Unsupported elem_type 23 (FLOAT4E2M1) for tensor 'like'. |
| node/test_castlike_no_saturate_FLOAT16_to_FLOAT8E5M2/model.onnx | 25 | Data | ❌ | Out of tolerance (max abs diff inf) |
| node/test_castlike_no_saturate_FLOAT16_to_FLOAT8E5M2_expanded/model.onnx | 25 | Data | ❌ | Out of tolerance (max abs diff inf) |
| node/test_castlike_no_saturate_FLOAT_to_FLOAT8E5M2/model.onnx | 25 | Data | ❌ | Out of tolerance (max abs diff inf) |
| node/test_castlike_no_saturate_FLOAT_to_FLOAT8E5M2_expanded/model.onnx | 25 | Data | ❌ | Out of tolerance (max abs diff inf) |
| node/test_dequantizelinear_e4m3fn/model.onnx | 25 | Data | ❌ | DequantizeLinear supports int2/uint2/int4/uint4/int8/uint8/int16/uint16/int32/uint32 inputs only |
| node/test_dequantizelinear_e4m3fn_float16/model.onnx | 25 | Data | ❌ | DequantizeLinear supports int2/uint2/int4/uint4/int8/uint8/int16/uint16/int32/uint32 inputs only |
| node/test_dequantizelinear_e4m3fn_zero_point/model.onnx | 25 | Data | ❌ | DequantizeLinear supports int2/uint2/int4/uint4/int8/uint8/int16/uint16/int32/uint32 inputs only |
| node/test_dequantizelinear_e5m2/model.onnx | 25 | Data | ❌ | DequantizeLinear supports int2/uint2/int4/uint4/int8/uint8/int16/uint16/int32/uint32 inputs only |
| node/test_dequantizelinear_float4e2m1/model.onnx | 25 | Data | ❌ | Unsupported elem_type 23 (FLOAT4E2M1) for tensor 'x'. |
| node/test_nllloss_NCd1d2d3d4d5_mean_weight_expanded/model.onnx | 22 | Data | ❌ | Out of tolerance (max ULP 357) |
| node/test_quantizelinear_e4m3fn/model.onnx | 25 | Data | ❌ | QuantizeLinear supports int2/uint2/int4/uint4/int8/uint8/int16/uint16 outputs only |
| node/test_quantizelinear_e5m2/model.onnx | 25 | Data | ❌ | QuantizeLinear supports int2/uint2/int4/uint4/int8/uint8/int16/uint16 outputs only |
| node/test_quantizelinear_float4e2m1/model.onnx | 25 | Data | ❌ | Unsupported elem_type 23 (FLOAT4E2M1) for tensor 'y_zero_point'. |

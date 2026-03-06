<!-- AUTO-GENERATED FILE. DO NOT EDIT. -->
<!-- Regenerate with: UPDATE_REFS=1 pytest -q tests/test_official_onnx_files_docs.py::test_official_onnx_file_support_doc -->

# ONNX verification errors

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
| Out of tolerance | 7 | 15, 20, 22 |
| Unsupported elem_type 24 (FLOAT8E8M0) for tensor '*'. | 4 | 25 |
| Unsupported op RNN | 4 | 22 |
| Unsupported op RandomUniformLike | 3 | 22 |
| Unsupported op RegexFullMatch | 3 | 20 |
| '*' | 2 |  |
| Unsupported op Det | 2 | 22 |
| Unsupported op MaxUnpool | 2 | 22 |
| Graph must contain at least one node | 1 | 25 |
| Pad value input must be a scalar | 1 | 24 |
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
| Unsupported op RegexFullMatch | 20 | 3 |
| Out of tolerance | 20 | 2 |
| Unsupported op RNN | 22 | 4 |
| Out of tolerance | 22 | 3 |
| Unsupported op RandomUniformLike | 22 | 3 |
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

## Failing ONNX files

Lists every ONNX file with a non-success verification outcome.

| File | Opset | Verification | Supported | Error |
| --- | --- | --- | --- | --- |
| local_ops/test_gemm_CM_transA/model.onnx | 12 | Data | ❌ | onnxruntime failed to run onnx2c-org/test/local_ops/test_gemm_CM_transA/model.onnx: [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Non-zero status code returned while running Gemm node. Name:'sclbl-onnx-node1' Status Message: Gemm: Invalid bias shape for broadcast |
| micro_kws_m_static_qoperator.onnx | 15 | Random+ORT | ❌ | Out of tolerance (max ULP 998244352) |
| node/test_adam_multiple/model.onnx |  | Data | ❌ | Out of tolerance (max ULP 62311) |
| node/test_affine_grid_3d/model.onnx | 20 | Data | ❌ | Out of tolerance (max ULP 151) |
| node/test_affine_grid_3d_expanded/model.onnx | 20 | Data | ❌ | Out of tolerance (max ULP 169) |
| node/test_ai_onnx_ml_binarizer/model.onnx |  | Data | ❌ | Unsupported op Binarizer |
| node/test_attention_4d_diff_heads_mask4d_padded_kv_expanded/model.onnx | 24 | Data | ❌ | Pad value input must be a scalar |
| node/test_averagepool_2d_ceil_last_window_starts_on_pad/model.onnx | 22 | Data | ❌ | Out of tolerance (max ULP 2983) |
| node/test_bernoulli_double_expanded/model.onnx | 22 | Data | ❌ | Unsupported op RandomUniformLike |
| node/test_bernoulli_expanded/model.onnx | 22 | Data | ❌ | Unsupported op RandomUniformLike |
| node/test_bernoulli_seed_expanded/model.onnx | 22 | Data | ❌ | Unsupported op RandomUniformLike |
| node/test_cast_FLOAT16_to_FLOAT4E2M1/model.onnx | 25 | Data | ❌ | Unsupported elem_type 23 (FLOAT4E2M1) for tensor 'output'. |
| node/test_cast_FLOAT16_to_FLOAT8E4M3FN/model.onnx | 25 | Data | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'output'. |
| node/test_cast_FLOAT16_to_FLOAT8E4M3FNUZ/model.onnx | 25 | Data | ❌ | Unsupported elem_type 18 (FLOAT8E4M3FNUZ) for tensor 'output'. |
| node/test_cast_FLOAT16_to_FLOAT8E5M2/model.onnx | 25 | Data | ❌ | Unsupported elem_type 19 (FLOAT8E5M2) for tensor 'output'. |
| node/test_cast_FLOAT16_to_FLOAT8E5M2FNUZ/model.onnx | 25 | Data | ❌ | Unsupported elem_type 20 (FLOAT8E5M2FNUZ) for tensor 'output'. |
| node/test_cast_FLOAT16_to_INT2/model.onnx | 25 | Data | ❌ | Unsupported elem_type 26 (INT2) for tensor 'output'. |
| node/test_cast_FLOAT16_to_INT4/model.onnx | 25 | Data | ❌ | Unsupported elem_type 22 (INT4) for tensor 'output'. |
| node/test_cast_FLOAT16_to_UINT2/model.onnx | 25 | Data | ❌ | Unsupported elem_type 25 (UINT2) for tensor 'output'. |
| node/test_cast_FLOAT16_to_UINT4/model.onnx | 25 | Data | ❌ | Unsupported elem_type 21 (UINT4) for tensor 'output'. |
| node/test_cast_FLOAT4E2M1_to_FLOAT/model.onnx | 25 | Data | ❌ | Unsupported elem_type 23 (FLOAT4E2M1) for tensor 'input'. |
| node/test_cast_FLOAT4E2M1_to_FLOAT16/model.onnx | 25 | Data | ❌ | Unsupported elem_type 23 (FLOAT4E2M1) for tensor 'input'. |
| node/test_cast_FLOAT8E4M3FNUZ_to_FLOAT/model.onnx | 25 | Data | ❌ | Unsupported elem_type 18 (FLOAT8E4M3FNUZ) for tensor 'input'. |
| node/test_cast_FLOAT8E4M3FNUZ_to_FLOAT16/model.onnx | 25 | Data | ❌ | Unsupported elem_type 18 (FLOAT8E4M3FNUZ) for tensor 'input'. |
| node/test_cast_FLOAT8E4M3FN_to_FLOAT/model.onnx | 25 | Data | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'input'. |
| node/test_cast_FLOAT8E4M3FN_to_FLOAT16/model.onnx | 25 | Data | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'input'. |
| node/test_cast_FLOAT8E5M2FNUZ_to_FLOAT/model.onnx | 25 | Data | ❌ | Unsupported elem_type 20 (FLOAT8E5M2FNUZ) for tensor 'input'. |
| node/test_cast_FLOAT8E5M2FNUZ_to_FLOAT16/model.onnx | 25 | Data | ❌ | Unsupported elem_type 20 (FLOAT8E5M2FNUZ) for tensor 'input'. |
| node/test_cast_FLOAT8E5M2_to_FLOAT/model.onnx | 25 | Data | ❌ | Unsupported elem_type 19 (FLOAT8E5M2) for tensor 'input'. |
| node/test_cast_FLOAT8E5M2_to_FLOAT16/model.onnx | 25 | Data | ❌ | Unsupported elem_type 19 (FLOAT8E5M2) for tensor 'input'. |
| node/test_cast_FLOAT_to_FLOAT4E2M1/model.onnx | 25 | Data | ❌ | Unsupported elem_type 23 (FLOAT4E2M1) for tensor 'output'. |
| node/test_cast_FLOAT_to_FLOAT8E4M3FN/model.onnx | 25 | Data | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'output'. |
| node/test_cast_FLOAT_to_FLOAT8E4M3FNUZ/model.onnx | 25 | Data | ❌ | Unsupported elem_type 18 (FLOAT8E4M3FNUZ) for tensor 'output'. |
| node/test_cast_FLOAT_to_FLOAT8E5M2/model.onnx | 25 | Data | ❌ | Unsupported elem_type 19 (FLOAT8E5M2) for tensor 'output'. |
| node/test_cast_FLOAT_to_FLOAT8E5M2FNUZ/model.onnx | 25 | Data | ❌ | Unsupported elem_type 20 (FLOAT8E5M2FNUZ) for tensor 'output'. |
| node/test_cast_FLOAT_to_INT2/model.onnx | 25 | Data | ❌ | Unsupported elem_type 26 (INT2) for tensor 'output'. |
| node/test_cast_FLOAT_to_INT4/model.onnx | 25 | Data | ❌ | Unsupported elem_type 22 (INT4) for tensor 'output'. |
| node/test_cast_FLOAT_to_UINT2/model.onnx | 25 | Data | ❌ | Unsupported elem_type 25 (UINT2) for tensor 'output'. |
| node/test_cast_FLOAT_to_UINT4/model.onnx | 25 | Data | ❌ | Unsupported elem_type 21 (UINT4) for tensor 'output'. |
| node/test_cast_INT2_to_FLOAT/model.onnx | 25 | Data | ❌ | Unsupported elem_type 26 (INT2) for tensor 'input'. |
| node/test_cast_INT2_to_FLOAT16/model.onnx | 25 | Data | ❌ | Unsupported elem_type 26 (INT2) for tensor 'input'. |
| node/test_cast_INT2_to_INT8/model.onnx | 25 | Data | ❌ | Unsupported elem_type 26 (INT2) for tensor 'input'. |
| node/test_cast_INT4_to_FLOAT/model.onnx | 25 | Data | ❌ | Unsupported elem_type 22 (INT4) for tensor 'input'. |
| node/test_cast_INT4_to_FLOAT16/model.onnx | 25 | Data | ❌ | Unsupported elem_type 22 (INT4) for tensor 'input'. |
| node/test_cast_INT4_to_INT8/model.onnx | 25 | Data | ❌ | Unsupported elem_type 22 (INT4) for tensor 'input'. |
| node/test_cast_UINT2_to_FLOAT/model.onnx | 25 | Data | ❌ | Unsupported elem_type 25 (UINT2) for tensor 'input'. |
| node/test_cast_UINT2_to_FLOAT16/model.onnx | 25 | Data | ❌ | Unsupported elem_type 25 (UINT2) for tensor 'input'. |
| node/test_cast_UINT2_to_UINT8/model.onnx | 25 | Data | ❌ | Unsupported elem_type 25 (UINT2) for tensor 'input'. |
| node/test_cast_UINT4_to_FLOAT/model.onnx | 25 | Data | ❌ | Unsupported elem_type 21 (UINT4) for tensor 'input'. |
| node/test_cast_UINT4_to_FLOAT16/model.onnx | 25 | Data | ❌ | Unsupported elem_type 21 (UINT4) for tensor 'input'. |
| node/test_cast_UINT4_to_UINT8/model.onnx | 25 | Data | ❌ | Unsupported elem_type 21 (UINT4) for tensor 'input'. |
| node/test_cast_e8m0_FLOAT16_to_FLOAT8E8M0/model.onnx | 25 | Data | ❌ | Unsupported elem_type 24 (FLOAT8E8M0) for tensor 'output'. |
| node/test_cast_e8m0_FLOAT8E8M0_to_FLOAT/model.onnx | 25 | Data | ❌ | Unsupported elem_type 24 (FLOAT8E8M0) for tensor 'input'. |
| node/test_cast_e8m0_FLOAT8E8M0_to_FLOAT16/model.onnx | 25 | Data | ❌ | Unsupported elem_type 24 (FLOAT8E8M0) for tensor 'input'. |
| node/test_cast_e8m0_FLOAT_to_FLOAT8E8M0/model.onnx | 25 | Data | ❌ | Unsupported elem_type 24 (FLOAT8E8M0) for tensor 'output'. |
| node/test_cast_no_saturate_FLOAT16_to_FLOAT8E4M3FN/model.onnx | 25 | Data | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'output'. |
| node/test_cast_no_saturate_FLOAT16_to_FLOAT8E4M3FNUZ/model.onnx | 25 | Data | ❌ | Unsupported elem_type 18 (FLOAT8E4M3FNUZ) for tensor 'output'. |
| node/test_cast_no_saturate_FLOAT16_to_FLOAT8E5M2/model.onnx | 25 | Data | ❌ | Unsupported elem_type 19 (FLOAT8E5M2) for tensor 'output'. |
| node/test_cast_no_saturate_FLOAT16_to_FLOAT8E5M2FNUZ/model.onnx | 25 | Data | ❌ | Unsupported elem_type 20 (FLOAT8E5M2FNUZ) for tensor 'output'. |
| node/test_cast_no_saturate_FLOAT_to_FLOAT8E4M3FN/model.onnx | 25 | Data | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'output'. |
| node/test_cast_no_saturate_FLOAT_to_FLOAT8E4M3FNUZ/model.onnx | 25 | Data | ❌ | Unsupported elem_type 18 (FLOAT8E4M3FNUZ) for tensor 'output'. |
| node/test_cast_no_saturate_FLOAT_to_FLOAT8E5M2/model.onnx | 25 | Data | ❌ | Unsupported elem_type 19 (FLOAT8E5M2) for tensor 'output'. |
| node/test_cast_no_saturate_FLOAT_to_FLOAT8E5M2FNUZ/model.onnx | 25 | Data | ❌ | Unsupported elem_type 20 (FLOAT8E5M2FNUZ) for tensor 'output'. |
| node/test_castlike_FLOAT16_to_FLOAT4E2M1/model.onnx | 25 | Data | ❌ | Unsupported elem_type 23 (FLOAT4E2M1) for tensor 'like'. |
| node/test_castlike_FLOAT16_to_FLOAT4E2M1_expanded/model.onnx | 25 | Data | ❌ | Unsupported elem_type 23 (FLOAT4E2M1) for tensor 'like'. |
| node/test_castlike_FLOAT16_to_FLOAT8E4M3FN/model.onnx | 25 | Data | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'like'. |
| node/test_castlike_FLOAT16_to_FLOAT8E4M3FNUZ/model.onnx | 25 | Data | ❌ | Unsupported elem_type 18 (FLOAT8E4M3FNUZ) for tensor 'like'. |
| node/test_castlike_FLOAT16_to_FLOAT8E4M3FNUZ_expanded/model.onnx | 25 | Data | ❌ | Unsupported elem_type 18 (FLOAT8E4M3FNUZ) for tensor 'like'. |
| node/test_castlike_FLOAT16_to_FLOAT8E4M3FN_expanded/model.onnx | 25 | Data | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'like'. |
| node/test_castlike_FLOAT16_to_FLOAT8E5M2/model.onnx | 25 | Data | ❌ | Unsupported elem_type 19 (FLOAT8E5M2) for tensor 'like'. |
| node/test_castlike_FLOAT16_to_FLOAT8E5M2FNUZ/model.onnx | 25 | Data | ❌ | Unsupported elem_type 20 (FLOAT8E5M2FNUZ) for tensor 'like'. |
| node/test_castlike_FLOAT16_to_FLOAT8E5M2FNUZ_expanded/model.onnx | 25 | Data | ❌ | Unsupported elem_type 20 (FLOAT8E5M2FNUZ) for tensor 'like'. |
| node/test_castlike_FLOAT16_to_FLOAT8E5M2_expanded/model.onnx | 25 | Data | ❌ | Unsupported elem_type 19 (FLOAT8E5M2) for tensor 'like'. |
| node/test_castlike_FLOAT16_to_INT2/model.onnx | 25 | Data | ❌ | Unsupported elem_type 26 (INT2) for tensor 'like'. |
| node/test_castlike_FLOAT16_to_INT2_expanded/model.onnx | 25 | Data | ❌ | Unsupported elem_type 26 (INT2) for tensor 'like'. |
| node/test_castlike_FLOAT16_to_INT4/model.onnx | 25 | Data | ❌ | Unsupported elem_type 22 (INT4) for tensor 'like'. |
| node/test_castlike_FLOAT16_to_INT4_expanded/model.onnx | 25 | Data | ❌ | Unsupported elem_type 22 (INT4) for tensor 'like'. |
| node/test_castlike_FLOAT16_to_UINT2/model.onnx | 25 | Data | ❌ | Unsupported elem_type 25 (UINT2) for tensor 'like'. |
| node/test_castlike_FLOAT16_to_UINT2_expanded/model.onnx | 25 | Data | ❌ | Unsupported elem_type 25 (UINT2) for tensor 'like'. |
| node/test_castlike_FLOAT16_to_UINT4/model.onnx | 25 | Data | ❌ | Unsupported elem_type 21 (UINT4) for tensor 'like'. |
| node/test_castlike_FLOAT16_to_UINT4_expanded/model.onnx | 25 | Data | ❌ | Unsupported elem_type 21 (UINT4) for tensor 'like'. |
| node/test_castlike_FLOAT4E2M1_to_FLOAT/model.onnx | 25 | Data | ❌ | Unsupported elem_type 23 (FLOAT4E2M1) for tensor 'input'. |
| node/test_castlike_FLOAT4E2M1_to_FLOAT16/model.onnx | 25 | Data | ❌ | Unsupported elem_type 23 (FLOAT4E2M1) for tensor 'input'. |
| node/test_castlike_FLOAT4E2M1_to_FLOAT16_expanded/model.onnx | 25 | Data | ❌ | Unsupported elem_type 23 (FLOAT4E2M1) for tensor 'input'. |
| node/test_castlike_FLOAT4E2M1_to_FLOAT_expanded/model.onnx | 25 | Data | ❌ | Unsupported elem_type 23 (FLOAT4E2M1) for tensor 'input'. |
| node/test_castlike_FLOAT8E4M3FNUZ_to_FLOAT/model.onnx | 25 | Data | ❌ | Unsupported elem_type 18 (FLOAT8E4M3FNUZ) for tensor 'input'. |
| node/test_castlike_FLOAT8E4M3FNUZ_to_FLOAT16/model.onnx | 25 | Data | ❌ | Unsupported elem_type 18 (FLOAT8E4M3FNUZ) for tensor 'input'. |
| node/test_castlike_FLOAT8E4M3FNUZ_to_FLOAT16_expanded/model.onnx | 25 | Data | ❌ | Unsupported elem_type 18 (FLOAT8E4M3FNUZ) for tensor 'input'. |
| node/test_castlike_FLOAT8E4M3FNUZ_to_FLOAT_expanded/model.onnx | 25 | Data | ❌ | Unsupported elem_type 18 (FLOAT8E4M3FNUZ) for tensor 'input'. |
| node/test_castlike_FLOAT8E4M3FN_to_FLOAT/model.onnx | 25 | Data | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'input'. |
| node/test_castlike_FLOAT8E4M3FN_to_FLOAT16/model.onnx | 25 | Data | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'input'. |
| node/test_castlike_FLOAT8E4M3FN_to_FLOAT16_expanded/model.onnx | 25 | Data | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'input'. |
| node/test_castlike_FLOAT8E4M3FN_to_FLOAT_expanded/model.onnx | 25 | Data | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'input'. |
| node/test_castlike_FLOAT8E5M2FNUZ_to_FLOAT/model.onnx | 25 | Data | ❌ | Unsupported elem_type 20 (FLOAT8E5M2FNUZ) for tensor 'input'. |
| node/test_castlike_FLOAT8E5M2FNUZ_to_FLOAT16/model.onnx | 25 | Data | ❌ | Unsupported elem_type 20 (FLOAT8E5M2FNUZ) for tensor 'input'. |
| node/test_castlike_FLOAT8E5M2FNUZ_to_FLOAT16_expanded/model.onnx | 25 | Data | ❌ | Unsupported elem_type 20 (FLOAT8E5M2FNUZ) for tensor 'input'. |
| node/test_castlike_FLOAT8E5M2FNUZ_to_FLOAT_expanded/model.onnx | 25 | Data | ❌ | Unsupported elem_type 20 (FLOAT8E5M2FNUZ) for tensor 'input'. |
| node/test_castlike_FLOAT8E5M2_to_FLOAT/model.onnx | 25 | Data | ❌ | Unsupported elem_type 19 (FLOAT8E5M2) for tensor 'input'. |
| node/test_castlike_FLOAT8E5M2_to_FLOAT16/model.onnx | 25 | Data | ❌ | Unsupported elem_type 19 (FLOAT8E5M2) for tensor 'input'. |
| node/test_castlike_FLOAT8E5M2_to_FLOAT16_expanded/model.onnx | 25 | Data | ❌ | Unsupported elem_type 19 (FLOAT8E5M2) for tensor 'input'. |
| node/test_castlike_FLOAT8E5M2_to_FLOAT_expanded/model.onnx | 25 | Data | ❌ | Unsupported elem_type 19 (FLOAT8E5M2) for tensor 'input'. |
| node/test_castlike_FLOAT_to_FLOAT4E2M1/model.onnx | 25 | Data | ❌ | Unsupported elem_type 23 (FLOAT4E2M1) for tensor 'like'. |
| node/test_castlike_FLOAT_to_FLOAT4E2M1_expanded/model.onnx | 25 | Data | ❌ | Unsupported elem_type 23 (FLOAT4E2M1) for tensor 'like'. |
| node/test_castlike_FLOAT_to_FLOAT8E4M3FN/model.onnx | 25 | Data | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'like'. |
| node/test_castlike_FLOAT_to_FLOAT8E4M3FNUZ/model.onnx | 25 | Data | ❌ | Unsupported elem_type 18 (FLOAT8E4M3FNUZ) for tensor 'like'. |
| node/test_castlike_FLOAT_to_FLOAT8E4M3FNUZ_expanded/model.onnx | 25 | Data | ❌ | Unsupported elem_type 18 (FLOAT8E4M3FNUZ) for tensor 'like'. |
| node/test_castlike_FLOAT_to_FLOAT8E4M3FN_expanded/model.onnx | 25 | Data | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'like'. |
| node/test_castlike_FLOAT_to_FLOAT8E5M2/model.onnx | 25 | Data | ❌ | Unsupported elem_type 19 (FLOAT8E5M2) for tensor 'like'. |
| node/test_castlike_FLOAT_to_FLOAT8E5M2FNUZ/model.onnx | 25 | Data | ❌ | Unsupported elem_type 20 (FLOAT8E5M2FNUZ) for tensor 'like'. |
| node/test_castlike_FLOAT_to_FLOAT8E5M2FNUZ_expanded/model.onnx | 25 | Data | ❌ | Unsupported elem_type 20 (FLOAT8E5M2FNUZ) for tensor 'like'. |
| node/test_castlike_FLOAT_to_FLOAT8E5M2_expanded/model.onnx | 25 | Data | ❌ | Unsupported elem_type 19 (FLOAT8E5M2) for tensor 'like'. |
| node/test_castlike_FLOAT_to_INT2/model.onnx | 25 | Data | ❌ | Unsupported elem_type 26 (INT2) for tensor 'like'. |
| node/test_castlike_FLOAT_to_INT2_expanded/model.onnx | 25 | Data | ❌ | Unsupported elem_type 26 (INT2) for tensor 'like'. |
| node/test_castlike_FLOAT_to_INT4/model.onnx | 25 | Data | ❌ | Unsupported elem_type 22 (INT4) for tensor 'like'. |
| node/test_castlike_FLOAT_to_INT4_expanded/model.onnx | 25 | Data | ❌ | Unsupported elem_type 22 (INT4) for tensor 'like'. |
| node/test_castlike_FLOAT_to_UINT2/model.onnx | 25 | Data | ❌ | Unsupported elem_type 25 (UINT2) for tensor 'like'. |
| node/test_castlike_FLOAT_to_UINT2_expanded/model.onnx | 25 | Data | ❌ | Unsupported elem_type 25 (UINT2) for tensor 'like'. |
| node/test_castlike_FLOAT_to_UINT4/model.onnx | 25 | Data | ❌ | Unsupported elem_type 21 (UINT4) for tensor 'like'. |
| node/test_castlike_FLOAT_to_UINT4_expanded/model.onnx | 25 | Data | ❌ | Unsupported elem_type 21 (UINT4) for tensor 'like'. |
| node/test_castlike_INT2_to_FLOAT/model.onnx | 25 | Data | ❌ | Unsupported elem_type 26 (INT2) for tensor 'input'. |
| node/test_castlike_INT2_to_FLOAT16/model.onnx | 25 | Data | ❌ | Unsupported elem_type 26 (INT2) for tensor 'input'. |
| node/test_castlike_INT2_to_FLOAT16_expanded/model.onnx | 25 | Data | ❌ | Unsupported elem_type 26 (INT2) for tensor 'input'. |
| node/test_castlike_INT2_to_FLOAT_expanded/model.onnx | 25 | Data | ❌ | Unsupported elem_type 26 (INT2) for tensor 'input'. |
| node/test_castlike_INT2_to_INT8/model.onnx | 25 | Data | ❌ | Unsupported elem_type 26 (INT2) for tensor 'input'. |
| node/test_castlike_INT2_to_INT8_expanded/model.onnx | 25 | Data | ❌ | Unsupported elem_type 26 (INT2) for tensor 'input'. |
| node/test_castlike_INT4_to_FLOAT/model.onnx | 25 | Data | ❌ | Unsupported elem_type 22 (INT4) for tensor 'input'. |
| node/test_castlike_INT4_to_FLOAT16/model.onnx | 25 | Data | ❌ | Unsupported elem_type 22 (INT4) for tensor 'input'. |
| node/test_castlike_INT4_to_FLOAT16_expanded/model.onnx | 25 | Data | ❌ | Unsupported elem_type 22 (INT4) for tensor 'input'. |
| node/test_castlike_INT4_to_FLOAT_expanded/model.onnx | 25 | Data | ❌ | Unsupported elem_type 22 (INT4) for tensor 'input'. |
| node/test_castlike_INT4_to_INT8/model.onnx | 25 | Data | ❌ | Unsupported elem_type 22 (INT4) for tensor 'input'. |
| node/test_castlike_INT4_to_INT8_expanded/model.onnx | 25 | Data | ❌ | Unsupported elem_type 22 (INT4) for tensor 'input'. |
| node/test_castlike_UINT2_to_FLOAT/model.onnx | 25 | Data | ❌ | Unsupported elem_type 25 (UINT2) for tensor 'input'. |
| node/test_castlike_UINT2_to_FLOAT16/model.onnx | 25 | Data | ❌ | Unsupported elem_type 25 (UINT2) for tensor 'input'. |
| node/test_castlike_UINT2_to_FLOAT16_expanded/model.onnx | 25 | Data | ❌ | Unsupported elem_type 25 (UINT2) for tensor 'input'. |
| node/test_castlike_UINT2_to_FLOAT_expanded/model.onnx | 25 | Data | ❌ | Unsupported elem_type 25 (UINT2) for tensor 'input'. |
| node/test_castlike_UINT2_to_UINT8/model.onnx | 25 | Data | ❌ | Unsupported elem_type 25 (UINT2) for tensor 'input'. |
| node/test_castlike_UINT2_to_UINT8_expanded/model.onnx | 25 | Data | ❌ | Unsupported elem_type 25 (UINT2) for tensor 'input'. |
| node/test_castlike_UINT4_to_FLOAT/model.onnx | 25 | Data | ❌ | Unsupported elem_type 21 (UINT4) for tensor 'input'. |
| node/test_castlike_UINT4_to_FLOAT16/model.onnx | 25 | Data | ❌ | Unsupported elem_type 21 (UINT4) for tensor 'input'. |
| node/test_castlike_UINT4_to_FLOAT16_expanded/model.onnx | 25 | Data | ❌ | Unsupported elem_type 21 (UINT4) for tensor 'input'. |
| node/test_castlike_UINT4_to_FLOAT_expanded/model.onnx | 25 | Data | ❌ | Unsupported elem_type 21 (UINT4) for tensor 'input'. |
| node/test_castlike_UINT4_to_UINT8/model.onnx | 25 | Data | ❌ | Unsupported elem_type 21 (UINT4) for tensor 'input'. |
| node/test_castlike_UINT4_to_UINT8_expanded/model.onnx | 25 | Data | ❌ | Unsupported elem_type 21 (UINT4) for tensor 'input'. |
| node/test_castlike_no_saturate_FLOAT16_to_FLOAT8E4M3FN/model.onnx | 25 | Data | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'like'. |
| node/test_castlike_no_saturate_FLOAT16_to_FLOAT8E4M3FNUZ/model.onnx | 25 | Data | ❌ | Unsupported elem_type 18 (FLOAT8E4M3FNUZ) for tensor 'like'. |
| node/test_castlike_no_saturate_FLOAT16_to_FLOAT8E4M3FNUZ_expanded/model.onnx | 25 | Data | ❌ | Unsupported elem_type 18 (FLOAT8E4M3FNUZ) for tensor 'like'. |
| node/test_castlike_no_saturate_FLOAT16_to_FLOAT8E4M3FN_expanded/model.onnx | 25 | Data | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'like'. |
| node/test_castlike_no_saturate_FLOAT16_to_FLOAT8E5M2/model.onnx | 25 | Data | ❌ | Unsupported elem_type 19 (FLOAT8E5M2) for tensor 'like'. |
| node/test_castlike_no_saturate_FLOAT16_to_FLOAT8E5M2FNUZ/model.onnx | 25 | Data | ❌ | Unsupported elem_type 20 (FLOAT8E5M2FNUZ) for tensor 'like'. |
| node/test_castlike_no_saturate_FLOAT16_to_FLOAT8E5M2FNUZ_expanded/model.onnx | 25 | Data | ❌ | Unsupported elem_type 20 (FLOAT8E5M2FNUZ) for tensor 'like'. |
| node/test_castlike_no_saturate_FLOAT16_to_FLOAT8E5M2_expanded/model.onnx | 25 | Data | ❌ | Unsupported elem_type 19 (FLOAT8E5M2) for tensor 'like'. |
| node/test_castlike_no_saturate_FLOAT_to_FLOAT8E4M3FN/model.onnx | 25 | Data | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'like'. |
| node/test_castlike_no_saturate_FLOAT_to_FLOAT8E4M3FNUZ/model.onnx | 25 | Data | ❌ | Unsupported elem_type 18 (FLOAT8E4M3FNUZ) for tensor 'like'. |
| node/test_castlike_no_saturate_FLOAT_to_FLOAT8E4M3FNUZ_expanded/model.onnx | 25 | Data | ❌ | Unsupported elem_type 18 (FLOAT8E4M3FNUZ) for tensor 'like'. |
| node/test_castlike_no_saturate_FLOAT_to_FLOAT8E4M3FN_expanded/model.onnx | 25 | Data | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'like'. |
| node/test_castlike_no_saturate_FLOAT_to_FLOAT8E5M2/model.onnx | 25 | Data | ❌ | Unsupported elem_type 19 (FLOAT8E5M2) for tensor 'like'. |
| node/test_castlike_no_saturate_FLOAT_to_FLOAT8E5M2FNUZ/model.onnx | 25 | Data | ❌ | Unsupported elem_type 20 (FLOAT8E5M2FNUZ) for tensor 'like'. |
| node/test_castlike_no_saturate_FLOAT_to_FLOAT8E5M2FNUZ_expanded/model.onnx | 25 | Data | ❌ | Unsupported elem_type 20 (FLOAT8E5M2FNUZ) for tensor 'like'. |
| node/test_castlike_no_saturate_FLOAT_to_FLOAT8E5M2_expanded/model.onnx | 25 | Data | ❌ | Unsupported elem_type 19 (FLOAT8E5M2) for tensor 'like'. |
| node/test_constant/model.onnx | 25 | Random+ORT | ❌ | Graph must contain at least one node |
| node/test_dequantizelinear_e4m3fn/model.onnx | 25 | Data | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'x'. |
| node/test_dequantizelinear_e4m3fn_float16/model.onnx | 25 | Data | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'x'. |
| node/test_dequantizelinear_e4m3fn_zero_point/model.onnx | 25 | Data | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'x'. |
| node/test_dequantizelinear_e5m2/model.onnx | 25 | Data | ❌ | Unsupported elem_type 19 (FLOAT8E5M2) for tensor 'x'. |
| node/test_dequantizelinear_float4e2m1/model.onnx | 25 | Data | ❌ | Unsupported elem_type 23 (FLOAT4E2M1) for tensor 'x'. |
| node/test_dequantizelinear_int2/model.onnx | 25 | Data | ❌ | Unsupported elem_type 26 (INT2) for tensor 'x'. |
| node/test_dequantizelinear_int4/model.onnx | 25 | Data | ❌ | Unsupported elem_type 22 (INT4) for tensor 'x'. |
| node/test_dequantizelinear_uint2/model.onnx | 25 | Data | ❌ | Unsupported elem_type 25 (UINT2) for tensor 'x'. |
| node/test_dequantizelinear_uint4/model.onnx | 25 | Data | ❌ | Unsupported elem_type 21 (UINT4) for tensor 'x'. |
| node/test_det_2d/model.onnx | 22 | Data | ❌ | Unsupported op Det |
| node/test_det_nd/model.onnx | 22 | Data | ❌ | Unsupported op Det |
| node/test_gridsample_bicubic/model.onnx | 22 | Data | ❌ | Out of tolerance (max ULP 1678) |
| node/test_loop16_seq_none/model.onnx | 16 | Data | ❌ | Unsupported op Loop |
| node/test_maxunpool_export_with_output_shape/model.onnx | 22 | Data | ❌ | Unsupported op MaxUnpool |
| node/test_maxunpool_export_without_output_shape/model.onnx | 22 | Data | ❌ | Unsupported op MaxUnpool |
| node/test_nllloss_NCd1d2d3d4d5_mean_weight_expanded/model.onnx | 22 | Data | ❌ | Out of tolerance (max ULP 357) |
| node/test_quantizelinear_e4m3fn/model.onnx | 25 | Data | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'y_zero_point'. |
| node/test_quantizelinear_e5m2/model.onnx | 25 | Data | ❌ | Unsupported elem_type 19 (FLOAT8E5M2) for tensor 'y_zero_point'. |
| node/test_quantizelinear_float4e2m1/model.onnx | 25 | Data | ❌ | Unsupported elem_type 23 (FLOAT4E2M1) for tensor 'y_zero_point'. |
| node/test_quantizelinear_int2/model.onnx | 25 | Data | ❌ | Unsupported elem_type 26 (INT2) for tensor 'y_zero_point'. |
| node/test_quantizelinear_int4/model.onnx | 25 | Data | ❌ | Unsupported elem_type 22 (INT4) for tensor 'y_zero_point'. |
| node/test_quantizelinear_uint2/model.onnx | 25 | Data | ❌ | Unsupported elem_type 25 (UINT2) for tensor 'y_zero_point'. |
| node/test_quantizelinear_uint4/model.onnx | 25 | Data | ❌ | Unsupported elem_type 21 (UINT4) for tensor 'y_zero_point'. |
| node/test_regex_full_match_basic/model.onnx | 20 | Data | ❌ | Unsupported op RegexFullMatch |
| node/test_regex_full_match_email_domain/model.onnx | 20 | Data | ❌ | Unsupported op RegexFullMatch |
| node/test_regex_full_match_empty/model.onnx | 20 | Data | ❌ | Unsupported op RegexFullMatch |
| node/test_rnn_seq_length/model.onnx | 22 | Data | ❌ | Unsupported op RNN |
| node/test_sequence_map_extract_shapes/model.onnx |  |  | ❌ | 'SequenceMap_0_in_0' |
| node/test_sequence_map_extract_shapes_expanded/model.onnx |  |  | ❌ | 'SequenceMap_test_sequence_map_extract_shapes_expanded_function_x' |
| node/test_simple_rnn_batchwise/model.onnx | 22 | Data | ❌ | Unsupported op RNN |
| node/test_simple_rnn_defaults/model.onnx | 22 | Data | ❌ | Unsupported op RNN |
| node/test_simple_rnn_with_initial_bias/model.onnx | 22 | Data | ❌ | Unsupported op RNN |
| simple_networks/random_uniform.onnx | 22 | Random+ORT | ❌ | Unsupported op RandomUniform |

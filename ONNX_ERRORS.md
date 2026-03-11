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
| ReduceSum output shape rank must match input rank | 12 | 18 |
| Code generation needs explicit shape concretization, but no --shape-inference-shapes were provided. Reason: tensor '*' has dynamic dimensions ('*', None, None, None). Hint: pass --shape-inference-shapes with explicit input specs (for example x=1x3x224x224;size=[1,3,224,224]) to compile/verify, or export the model with static shapes. | 11 | 11, 12, 15 |
| CastLike input and output shapes must match | 8 | 18 |
| Explicit --shape-inference-shapes were provided for input name(s) that do not require them: '*'. Remove those entries and rerun. | 4 | 17 |
| Unsupported elem_type 24 (FLOAT8E8M0) for tensor '*'. | 4 | 25 |
| Code generation needs explicit shape concretization, but no --shape-inference-shapes were provided. Reason: tensor '*' has dynamic dimensions ('*', None). Hint: pass --shape-inference-shapes with explicit input specs (for example x=1x3x224x224;size=[1,3,224,224]) to compile/verify, or export the model with static shapes. | 3 | 9, 11 |
| Explicit --shape-inference-shapes were provided for input name(s) that do not require them: '*', '*'. Remove those entries and rerun. | 3 | 12, 24 |
| Code generation needs explicit shape concretization, but no --shape-inference-shapes were provided. Reason: tensor '*' has dynamic dimensions ('*', '*', None). Hint: pass --shape-inference-shapes with explicit input specs (for example x=1x3x224x224;size=[1,3,224,224]) to compile/verify, or export the model with static shapes. | 2 | 11 |
| Out of tolerance | 2 | 22 |
| Code generation still has unresolved dynamic shapes after shape concretization. Reason: tensor '*' has dynamic dimensions ('*', '*', '*', '*'). Hint: provide more representative --shape-inference-shapes or export the model with static shapes. | 1 | 24 |
| Output shape mismatch for reduced (actual_shape=(1,), actual_size=1, expected_shape=(3,), expected_size=3, output=reduced) | 1 | 18 |
| Output shape mismatch for reduced (actual_shape=(1,), actual_size=1, expected_shape=(5,), expected_size=5, output=reduced) | 1 | 18 |
| Testbench execution failed: exit code 1 | 1 | 12 |
| onnxruntime failed to run onnx2c-org/test/local_ops/test_gemm_CM_transA/model.onnx: [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Non-zero status code returned while running Gemm node. Name:'*' Status Message: Gemm: Invalid bias shape for broadcast | 1 | 12 |

## Error frequency by opset

| Error message | Opset | Count |
| --- | --- | --- |
| Code generation needs explicit shape concretization, but no --shape-inference-shapes were provided. Reason: tensor '*' has dynamic dimensions ('*', None). Hint: pass --shape-inference-shapes with explicit input specs (for example x=1x3x224x224;size=[1,3,224,224]) to compile/verify, or export the model with static shapes. | 9 | 2 |
| Code generation needs explicit shape concretization, but no --shape-inference-shapes were provided. Reason: tensor '*' has dynamic dimensions ('*', None, None, None). Hint: pass --shape-inference-shapes with explicit input specs (for example x=1x3x224x224;size=[1,3,224,224]) to compile/verify, or export the model with static shapes. | 11 | 6 |
| Code generation needs explicit shape concretization, but no --shape-inference-shapes were provided. Reason: tensor '*' has dynamic dimensions ('*', '*', None). Hint: pass --shape-inference-shapes with explicit input specs (for example x=1x3x224x224;size=[1,3,224,224]) to compile/verify, or export the model with static shapes. | 11 | 2 |
| Code generation needs explicit shape concretization, but no --shape-inference-shapes were provided. Reason: tensor '*' has dynamic dimensions ('*', None). Hint: pass --shape-inference-shapes with explicit input specs (for example x=1x3x224x224;size=[1,3,224,224]) to compile/verify, or export the model with static shapes. | 11 | 1 |
| Code generation needs explicit shape concretization, but no --shape-inference-shapes were provided. Reason: tensor '*' has dynamic dimensions ('*', None, None, None). Hint: pass --shape-inference-shapes with explicit input specs (for example x=1x3x224x224;size=[1,3,224,224]) to compile/verify, or export the model with static shapes. | 12 | 3 |
| Explicit --shape-inference-shapes were provided for input name(s) that do not require them: '*', '*'. Remove those entries and rerun. | 12 | 1 |
| Testbench execution failed: exit code 1 | 12 | 1 |
| onnxruntime failed to run onnx2c-org/test/local_ops/test_gemm_CM_transA/model.onnx: [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Non-zero status code returned while running Gemm node. Name:'*' Status Message: Gemm: Invalid bias shape for broadcast | 12 | 1 |
| Code generation needs explicit shape concretization, but no --shape-inference-shapes were provided. Reason: tensor '*' has dynamic dimensions ('*', None, None, None). Hint: pass --shape-inference-shapes with explicit input specs (for example x=1x3x224x224;size=[1,3,224,224]) to compile/verify, or export the model with static shapes. | 15 | 2 |
| Explicit --shape-inference-shapes were provided for input name(s) that do not require them: '*'. Remove those entries and rerun. | 17 | 4 |
| ReduceSum output shape rank must match input rank | 18 | 12 |
| CastLike input and output shapes must match | 18 | 8 |
| Output shape mismatch for reduced (actual_shape=(1,), actual_size=1, expected_shape=(3,), expected_size=3, output=reduced) | 18 | 1 |
| Output shape mismatch for reduced (actual_shape=(1,), actual_size=1, expected_shape=(5,), expected_size=5, output=reduced) | 18 | 1 |
| Out of tolerance | 22 | 1 |
| Explicit --shape-inference-shapes were provided for input name(s) that do not require them: '*', '*'. Remove those entries and rerun. | 24 | 2 |
| Code generation still has unresolved dynamic shapes after shape concretization. Reason: tensor '*' has dynamic dimensions ('*', '*', '*', '*'). Hint: provide more representative --shape-inference-shapes or export the model with static shapes. | 24 | 1 |
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

## Failing ONNX files

Lists every ONNX file with a non-success verification outcome.

| File | Opset | Verification | Supported | Error |
| --- | --- | --- | --- | --- |
| local_ops/test_gemm_CM_transA/model.onnx | 12 | Data | ❌ | onnxruntime failed to run onnx2c-org/test/local_ops/test_gemm_CM_transA/model.onnx: [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Non-zero status code returned while running Gemm node. Name:'sclbl-onnx-node1' Status Message: Gemm: Invalid bias shape for broadcast |
| micro_kws_m_qdq.onnx | 15 | Random+ORT | ❌ | Code generation needs explicit shape concretization, but no --shape-inference-shapes were provided. Reason: tensor 'micro_kws/conv2d/Relu;micro_kws/conv2d/BiasAdd;micro_kws/conv2d_2/Conv2D;micro_kws/conv2d/Conv2D;micro_kws/conv2d/BiasAdd/ReadVariableOp__28:0' has dynamic dimensions ('unk__0', None, None, None). Hint: pass --shape-inference-shapes with explicit input specs (for example x=1x3x224x224;size=[1,3,224,224]) to compile/verify, or export the model with static shapes. |
| micro_kws_m_static_qdq.onnx | 15 | Random+ORT | ❌ | Code generation needs explicit shape concretization, but no --shape-inference-shapes were provided. Reason: tensor 'micro_kws/conv2d/Relu;micro_kws/conv2d/BiasAdd;micro_kws/conv2d_2/Conv2D;micro_kws/conv2d/Conv2D;micro_kws/conv2d/BiasAdd/ReadVariableOp__28:0' has dynamic dimensions ('unk__0', None, None, None). Hint: pass --shape-inference-shapes with explicit input specs (for example x=1x3x224x224;size=[1,3,224,224]) to compile/verify, or export the model with static shapes. |
| node/test_adam_multiple/model.onnx |  | Data | ❌ | Out of tolerance (max ULP 62311) |
| node/test_attention_4d_diff_heads_mask4d_padded_kv_expanded/model.onnx | 24 | Data | ❌ | Code generation still has unresolved dynamic shapes after shape concretization. Reason: tensor 'Attention_test_attention_4d_diff_heads_mask4d_padded_kv_expanded_function_AttnBias' has dynamic dimensions ('unk__0', 'unk__1', 'unk__2', 'unk__3'). Hint: provide more representative --shape-inference-shapes or export the model with static shapes. |
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
| node/test_dequantizelinear_e4m3fn/model.onnx | 25 | Data | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'x'. |
| node/test_dequantizelinear_e4m3fn_float16/model.onnx | 25 | Data | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'x'. |
| node/test_dequantizelinear_e4m3fn_zero_point/model.onnx | 25 | Data | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'x'. |
| node/test_dequantizelinear_e5m2/model.onnx | 25 | Data | ❌ | Unsupported elem_type 19 (FLOAT8E5M2) for tensor 'x'. |
| node/test_dequantizelinear_float4e2m1/model.onnx | 25 | Data | ❌ | Unsupported elem_type 23 (FLOAT4E2M1) for tensor 'x'. |
| node/test_dequantizelinear_int2/model.onnx | 25 | Data | ❌ | Unsupported elem_type 26 (INT2) for tensor 'x'. |
| node/test_dequantizelinear_int4/model.onnx | 25 | Data | ❌ | Unsupported elem_type 22 (INT4) for tensor 'x'. |
| node/test_dequantizelinear_uint2/model.onnx | 25 | Data | ❌ | Unsupported elem_type 25 (UINT2) for tensor 'x'. |
| node/test_dequantizelinear_uint4/model.onnx | 25 | Data | ❌ | Unsupported elem_type 21 (UINT4) for tensor 'x'. |
| node/test_nllloss_NCd1d2d3d4d5_mean_weight_expanded/model.onnx | 22 | Data | ❌ | Out of tolerance (max ULP 357) |
| node/test_quantizelinear_e4m3fn/model.onnx | 25 | Data | ❌ | Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor 'y_zero_point'. |
| node/test_quantizelinear_e5m2/model.onnx | 25 | Data | ❌ | Unsupported elem_type 19 (FLOAT8E5M2) for tensor 'y_zero_point'. |
| node/test_quantizelinear_float4e2m1/model.onnx | 25 | Data | ❌ | Unsupported elem_type 23 (FLOAT4E2M1) for tensor 'y_zero_point'. |
| node/test_quantizelinear_int2/model.onnx | 25 | Data | ❌ | Unsupported elem_type 26 (INT2) for tensor 'y_zero_point'. |
| node/test_quantizelinear_int4/model.onnx | 25 | Data | ❌ | Unsupported elem_type 22 (INT4) for tensor 'y_zero_point'. |
| node/test_quantizelinear_uint2/model.onnx | 25 | Data | ❌ | Unsupported elem_type 25 (UINT2) for tensor 'y_zero_point'. |
| node/test_quantizelinear_uint4/model.onnx | 25 | Data | ❌ | Unsupported elem_type 21 (UINT4) for tensor 'y_zero_point'. |
| node/test_reduce_l2_default_axes_keepdims_example_expanded/model.onnx | 18 | Data | ❌ | CastLike input and output shapes must match |
| node/test_reduce_l2_default_axes_keepdims_random_expanded/model.onnx | 18 | Data | ❌ | CastLike input and output shapes must match |
| node/test_reduce_l2_do_not_keepdims_example_expanded/model.onnx | 18 | Data | ❌ | CastLike input and output shapes must match |
| node/test_reduce_l2_do_not_keepdims_random_expanded/model.onnx | 18 | Data | ❌ | CastLike input and output shapes must match |
| node/test_reduce_l2_empty_set_expanded/model.onnx | 18 | Data | ❌ | ReduceSum output shape rank must match input rank |
| node/test_reduce_l2_keep_dims_example_expanded/model.onnx | 18 | Data | ❌ | ReduceSum output shape rank must match input rank |
| node/test_reduce_l2_keep_dims_random_expanded/model.onnx | 18 | Data | ❌ | ReduceSum output shape rank must match input rank |
| node/test_reduce_l2_negative_axes_keep_dims_example_expanded/model.onnx | 18 | Data | ❌ | ReduceSum output shape rank must match input rank |
| node/test_reduce_l2_negative_axes_keep_dims_random_expanded/model.onnx | 18 | Data | ❌ | ReduceSum output shape rank must match input rank |
| node/test_reduce_log_sum_asc_axes_expanded/model.onnx | 18 | Data | ❌ | Output shape mismatch for reduced (actual_shape=(1,), actual_size=1, expected_shape=(5,), expected_size=5, output=reduced) |
| node/test_reduce_log_sum_desc_axes_expanded/model.onnx | 18 | Data | ❌ | Output shape mismatch for reduced (actual_shape=(1,), actual_size=1, expected_shape=(3,), expected_size=3, output=reduced) |
| node/test_reduce_log_sum_empty_set_expanded/model.onnx | 18 | Data | ❌ | ReduceSum output shape rank must match input rank |
| node/test_reduce_log_sum_exp_default_axes_keepdims_example_expanded/model.onnx | 18 | Data | ❌ | CastLike input and output shapes must match |
| node/test_reduce_log_sum_exp_default_axes_keepdims_random_expanded/model.onnx | 18 | Data | ❌ | CastLike input and output shapes must match |
| node/test_reduce_log_sum_exp_do_not_keepdims_example_expanded/model.onnx | 18 | Data | ❌ | CastLike input and output shapes must match |
| node/test_reduce_log_sum_exp_do_not_keepdims_random_expanded/model.onnx | 18 | Data | ❌ | CastLike input and output shapes must match |
| node/test_reduce_log_sum_exp_empty_set_expanded/model.onnx | 18 | Data | ❌ | ReduceSum output shape rank must match input rank |
| node/test_reduce_log_sum_exp_keepdims_example_expanded/model.onnx | 18 | Data | ❌ | ReduceSum output shape rank must match input rank |
| node/test_reduce_log_sum_exp_keepdims_random_expanded/model.onnx | 18 | Data | ❌ | ReduceSum output shape rank must match input rank |
| node/test_reduce_log_sum_exp_negative_axes_keepdims_example_expanded/model.onnx | 18 | Data | ❌ | ReduceSum output shape rank must match input rank |
| node/test_reduce_log_sum_exp_negative_axes_keepdims_random_expanded/model.onnx | 18 | Data | ❌ | ReduceSum output shape rank must match input rank |
| node/test_reduce_log_sum_negative_axes_expanded/model.onnx | 18 | Data | ❌ | ReduceSum output shape rank must match input rank |
| node/test_sequence_map_add_1_sequence_1_tensor/model.onnx | 17 | Data | ❌ | Explicit --shape-inference-shapes were provided for input name(s) that do not require them: 'x1'. Remove those entries and rerun. |
| node/test_sequence_map_add_1_sequence_1_tensor_expanded/model.onnx | 17 | Data | ❌ | Explicit --shape-inference-shapes were provided for input name(s) that do not require them: 'x1'. Remove those entries and rerun. |
| node/test_sequence_map_identity_1_sequence_1_tensor/model.onnx | 17 | Data | ❌ | Explicit --shape-inference-shapes were provided for input name(s) that do not require them: 'x1'. Remove those entries and rerun. |
| node/test_sequence_map_identity_1_sequence_1_tensor_expanded/model.onnx | 17 | Data | ❌ | Explicit --shape-inference-shapes were provided for input name(s) that do not require them: 'x1'. Remove those entries and rerun. |
| node/test_split_to_sequence_1/model.onnx | 24 | Data | ❌ | Explicit --shape-inference-shapes were provided for input name(s) that do not require them: 'data', 'split'. Remove those entries and rerun. |
| node/test_split_to_sequence_2/model.onnx | 24 | Data | ❌ | Explicit --shape-inference-shapes were provided for input name(s) that do not require them: 'data', 'split'. Remove those entries and rerun. |
| simple/test_sequence_model4/model.onnx | 12 | Data | ❌ | Testbench execution failed: exit code 1 |
| simple/test_sequence_model8/model.onnx | 12 | Data | ❌ | Explicit --shape-inference-shapes were provided for input name(s) that do not require them: 'X', 'Splits'. Remove those entries and rerun. |
| simple_networks/conv_2ch_3kernels_randombias.onnx | 11 | Random+ORT | ❌ | Code generation needs explicit shape concretization, but no --shape-inference-shapes were provided. Reason: tensor 'adjusted_input' has dynamic dimensions ('N', None, None, None). Hint: pass --shape-inference-shapes with explicit input specs (for example x=1x3x224x224;size=[1,3,224,224]) to compile/verify, or export the model with static shapes. |
| simple_networks/conv_2kernels.onnx | 11 | Random+ORT | ❌ | Code generation needs explicit shape concretization, but no --shape-inference-shapes were provided. Reason: tensor 'adjusted_input' has dynamic dimensions ('N', None, None, None). Hint: pass --shape-inference-shapes with explicit input specs (for example x=1x3x224x224;size=[1,3,224,224]) to compile/verify, or export the model with static shapes. |
| simple_networks/conv_2kernels_randombias.onnx | 11 | Random+ORT | ❌ | Code generation needs explicit shape concretization, but no --shape-inference-shapes were provided. Reason: tensor 'adjusted_input' has dynamic dimensions ('N', None, None, None). Hint: pass --shape-inference-shapes with explicit input specs (for example x=1x3x224x224;size=[1,3,224,224]) to compile/verify, or export the model with static shapes. |
| simple_networks/conv_3ch.onnx | 11 | Random+ORT | ❌ | Code generation needs explicit shape concretization, but no --shape-inference-shapes were provided. Reason: tensor 'adjusted_input' has dynamic dimensions ('N', None, None, None). Hint: pass --shape-inference-shapes with explicit input specs (for example x=1x3x224x224;size=[1,3,224,224]) to compile/verify, or export the model with static shapes. |
| simple_networks/conv_k2.onnx | 11 | Random+ORT | ❌ | Code generation needs explicit shape concretization, but no --shape-inference-shapes were provided. Reason: tensor 'adjusted_input' has dynamic dimensions ('N', None, None, None). Hint: pass --shape-inference-shapes with explicit input specs (for example x=1x3x224x224;size=[1,3,224,224]) to compile/verify, or export the model with static shapes. |
| simple_networks/conv_k2_maxpool_k2.onnx | 12 | Random+ORT | ❌ | Code generation needs explicit shape concretization, but no --shape-inference-shapes were provided. Reason: tensor 'adjusted_input' has dynamic dimensions ('N', None, None, None). Hint: pass --shape-inference-shapes with explicit input specs (for example x=1x3x224x224;size=[1,3,224,224]) to compile/verify, or export the model with static shapes. |
| simple_networks/conv_k2_s2.onnx | 11 | Random+ORT | ❌ | Code generation needs explicit shape concretization, but no --shape-inference-shapes were provided. Reason: tensor 'adjusted_input' has dynamic dimensions ('N', None, None, None). Hint: pass --shape-inference-shapes with explicit input specs (for example x=1x3x224x224;size=[1,3,224,224]) to compile/verify, or export the model with static shapes. |
| simple_networks/lstm_k1_b1_r1.onnx | 11 | Random+ORT | ❌ | Code generation needs explicit shape concretization, but no --shape-inference-shapes were provided. Reason: tensor 'lstm_X' has dynamic dimensions ('M1', 'N', None). Hint: pass --shape-inference-shapes with explicit input specs (for example x=1x3x224x224;size=[1,3,224,224]) to compile/verify, or export the model with static shapes. |
| simple_networks/lstm_k1_b1_r1_relu.onnx | 11 | Random+ORT | ❌ | Code generation needs explicit shape concretization, but no --shape-inference-shapes were provided. Reason: tensor 'lstm_X' has dynamic dimensions ('M1', 'N', None). Hint: pass --shape-inference-shapes with explicit input specs (for example x=1x3x224x224;size=[1,3,224,224]) to compile/verify, or export the model with static shapes. |
| simple_networks/maxpool_k2.onnx | 12 | Random+ORT | ❌ | Code generation needs explicit shape concretization, but no --shape-inference-shapes were provided. Reason: tensor 'input_transposed' has dynamic dimensions ('N', None, None, None). Hint: pass --shape-inference-shapes with explicit input specs (for example x=1x3x224x224;size=[1,3,224,224]) to compile/verify, or export the model with static shapes. |
| simple_networks/maxpool_k2_s2.onnx | 12 | Random+ORT | ❌ | Code generation needs explicit shape concretization, but no --shape-inference-shapes were provided. Reason: tensor 'input_transposed' has dynamic dimensions ('N', None, None, None). Hint: pass --shape-inference-shapes with explicit input specs (for example x=1x3x224x224;size=[1,3,224,224]) to compile/verify, or export the model with static shapes. |
| tfl_helloworld/model.onnx | 9 | Random+ORT | ❌ | Code generation needs explicit shape concretization, but no --shape-inference-shapes were provided. Reason: tensor 'dense_20' has dynamic dimensions ('N', None). Hint: pass --shape-inference-shapes with explicit input specs (for example x=1x3x224x224;size=[1,3,224,224]) to compile/verify, or export the model with static shapes. |
| velardo/lesson14.onnx | 11 | Random+ORT | ❌ | Code generation needs explicit shape concretization, but no --shape-inference-shapes were provided. Reason: tensor 'flatten/Reshape:0' has dynamic dimensions ('N', None). Hint: pass --shape-inference-shapes with explicit input specs (for example x=1x3x224x224;size=[1,3,224,224]) to compile/verify, or export the model with static shapes. |
| velardo/lesson9.onnx | 9 | Random+ORT | ❌ | Code generation needs explicit shape concretization, but no --shape-inference-shapes were provided. Reason: tensor 'dense0' has dynamic dimensions ('N', None). Hint: pass --shape-inference-shapes with explicit input specs (for example x=1x3x224x224;size=[1,3,224,224]) to compile/verify, or export the model with static shapes. |

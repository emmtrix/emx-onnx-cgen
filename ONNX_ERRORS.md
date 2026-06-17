<!-- AUTO-GENERATED FILE. DO NOT EDIT. -->
<!-- Regenerate with: UPDATE_REFS=1 pytest -q tests/test_official_onnx_files_docs.py::test_official_onnx_file_support_doc -->

# ONNX verification errors

Aggregates non-success verification outcomes.

| Error message | Count | Opset versions |
| --- | --- | --- |
| Unsupported value type '*' for '*'. Hint: export the model with tensor inputs/outputs. | 8 |  |
| Out of tolerance | 5 |  |
| Range does not support dtype bfloat16 (while lowering node_index=0, op_type=Range, name=<unnamed>, inputs=[start: tensor[dtype=bfloat16, shape=()], limit: tensor[dtype=bfloat16, shape=()], delta: tensor[dtype=bfloat16, shape=()]], outputs=[output: tensor[dtype=bfloat16, shape=(2,), dim_params=(None,)]]) | 1 | 27 |
| Range does not support dtype float16 (while lowering node_index=0, op_type=Range, name=<unnamed>, inputs=[start: tensor[dtype=float16, shape=()], limit: tensor[dtype=float16, shape=()], delta: tensor[dtype=float16, shape=()]], outputs=[output: tensor[dtype=float16, shape=(2,), dim_params=(None,)]]) | 1 | 27 |
| Unsupported op Loop (while lowering node_index=9, op_type=Loop, name=<unnamed>, inputs=[Range_test_range_bfloat16_type_positive_delta_expanded_function_n: tensor[dtype=int64, shape=()], Range_test_range_bfloat16_type_positive_delta_expanded_function_loop_cond: tensor[dtype=bool, shape=()], Range_test_range_bfloat16_type_positive_delta_expanded_function_start_s: tensor[dtype=float, shape=()]], outputs=[Range_test_range_bfloat16_type_positive_delta_expanded_function_variadic_output: tensor[dtype=float, shape=()], Range_test_range_bfloat16_type_positive_delta_expanded_function_output_s: tensor[dtype=float, shape=()]]) | 1 | 27 |
| Unsupported op Loop (while lowering node_index=9, op_type=Loop, name=<unnamed>, inputs=[Range_test_range_float16_type_positive_delta_expanded_function_n: tensor[dtype=int64, shape=()], Range_test_range_float16_type_positive_delta_expanded_function_loop_cond: tensor[dtype=bool, shape=()], Range_test_range_float16_type_positive_delta_expanded_function_start_s: tensor[dtype=float, shape=()]], outputs=[Range_test_range_float16_type_positive_delta_expanded_function_variadic_output: tensor[dtype=float, shape=()], Range_test_range_float16_type_positive_delta_expanded_function_output_s: tensor[dtype=float, shape=()]]) | 1 | 27 |

## Error frequency by opset

| Error message | Opset | Count |
| --- | --- | --- |
| Range does not support dtype bfloat16 (while lowering node_index=0, op_type=Range, name=<unnamed>, inputs=[start: tensor[dtype=bfloat16, shape=()], limit: tensor[dtype=bfloat16, shape=()], delta: tensor[dtype=bfloat16, shape=()]], outputs=[output: tensor[dtype=bfloat16, shape=(2,), dim_params=(None,)]]) | 27 | 1 |
| Range does not support dtype float16 (while lowering node_index=0, op_type=Range, name=<unnamed>, inputs=[start: tensor[dtype=float16, shape=()], limit: tensor[dtype=float16, shape=()], delta: tensor[dtype=float16, shape=()]], outputs=[output: tensor[dtype=float16, shape=(2,), dim_params=(None,)]]) | 27 | 1 |
| Unsupported op Loop (while lowering node_index=9, op_type=Loop, name=<unnamed>, inputs=[Range_test_range_bfloat16_type_positive_delta_expanded_function_n: tensor[dtype=int64, shape=()], Range_test_range_bfloat16_type_positive_delta_expanded_function_loop_cond: tensor[dtype=bool, shape=()], Range_test_range_bfloat16_type_positive_delta_expanded_function_start_s: tensor[dtype=float, shape=()]], outputs=[Range_test_range_bfloat16_type_positive_delta_expanded_function_variadic_output: tensor[dtype=float, shape=()], Range_test_range_bfloat16_type_positive_delta_expanded_function_output_s: tensor[dtype=float, shape=()]]) | 27 | 1 |
| Unsupported op Loop (while lowering node_index=9, op_type=Loop, name=<unnamed>, inputs=[Range_test_range_float16_type_positive_delta_expanded_function_n: tensor[dtype=int64, shape=()], Range_test_range_float16_type_positive_delta_expanded_function_loop_cond: tensor[dtype=bool, shape=()], Range_test_range_float16_type_positive_delta_expanded_function_start_s: tensor[dtype=float, shape=()]], outputs=[Range_test_range_float16_type_positive_delta_expanded_function_variadic_output: tensor[dtype=float, shape=()], Range_test_range_float16_type_positive_delta_expanded_function_output_s: tensor[dtype=float, shape=()]]) | 27 | 1 |

## Failing ONNX files

Lists every ONNX file with a non-success verification outcome.

| File | Opset | Verification | Supported | Error |
| --- | --- | --- | --- | --- |
| node/test_range_bfloat16_type_positive_delta/model.onnx | 27 | Data/Data | ❌ | Range does not support dtype bfloat16 (while lowering node_index=0, op_type=Range, name=<unnamed>, inputs=[start: tensor[dtype=bfloat16, shape=()], limit: tensor[dtype=bfloat16, shape=()], delta: tensor[dtype=bfloat16, shape=()]], outputs=[output: tensor[dtype=bfloat16, shape=(2,), dim_params=(None,)]]) |
| node/test_range_bfloat16_type_positive_delta_expanded/model.onnx | 27 | Data/Data | ❌ | Unsupported op Loop (while lowering node_index=9, op_type=Loop, name=<unnamed>, inputs=[Range_test_range_bfloat16_type_positive_delta_expanded_function_n: tensor[dtype=int64, shape=()], Range_test_range_bfloat16_type_positive_delta_expanded_function_loop_cond: tensor[dtype=bool, shape=()], Range_test_range_bfloat16_type_positive_delta_expanded_function_start_s: tensor[dtype=float, shape=()]], outputs=[Range_test_range_bfloat16_type_positive_delta_expanded_function_variadic_output: tensor[dtype=float, shape=()], Range_test_range_bfloat16_type_positive_delta_expanded_function_output_s: tensor[dtype=float, shape=()]]) |
| node/test_range_float16_type_positive_delta/model.onnx | 27 | Data/Data | ❌ | Range does not support dtype float16 (while lowering node_index=0, op_type=Range, name=<unnamed>, inputs=[start: tensor[dtype=float16, shape=()], limit: tensor[dtype=float16, shape=()], delta: tensor[dtype=float16, shape=()]], outputs=[output: tensor[dtype=float16, shape=(2,), dim_params=(None,)]]) |
| node/test_range_float16_type_positive_delta_expanded/model.onnx | 27 | Data/Data | ❌ | Unsupported op Loop (while lowering node_index=9, op_type=Loop, name=<unnamed>, inputs=[Range_test_range_float16_type_positive_delta_expanded_function_n: tensor[dtype=int64, shape=()], Range_test_range_float16_type_positive_delta_expanded_function_loop_cond: tensor[dtype=bool, shape=()], Range_test_range_float16_type_positive_delta_expanded_function_start_s: tensor[dtype=float, shape=()]], outputs=[Range_test_range_float16_type_positive_delta_expanded_function_variadic_output: tensor[dtype=float, shape=()], Range_test_range_float16_type_positive_delta_expanded_function_output_s: tensor[dtype=float, shape=()]]) |
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

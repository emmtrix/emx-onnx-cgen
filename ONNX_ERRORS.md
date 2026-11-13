<!-- AUTO-GENERATED FILE. DO NOT EDIT. -->
<!-- Regenerate with: UPDATE_REFS=1 pytest -q tests/test_official_onnx_files_docs.py::test_official_onnx_file_support_doc -->

# ONNX verification errors

Aggregates non-success verification outcomes.

| Error message | Count | Opset versions |
| --- | --- | --- |
| Unsupported value type '*' for '*'. Hint: export the model with tensor inputs/outputs. | 8 |  |
| Out of tolerance | 3 | 17 |
| MoE: only activation_type='*' is supported, got '*' (while lowering node_index=0, op_type=MoE, name=node1, inputs=[input: tensor[dtype=float, shape=(4, 8), dim_params=(None, None)], router_probs: tensor[dtype=float, shape=(4, 4), dim_params=(None, None)], fc1_experts_weights: tensor[dtype=float, shape=(4, 16, 8), dim_params=(None, None, None)], fc1_experts_bias: tensor[dtype=float, shape=(4, 16), dim_params=(None, None)], fc2_experts_weights: tensor[dtype=float, shape=(4, 8, 16), dim_params=(None, None, None)], fc2_experts_bias: tensor[dtype=float, shape=(4, 8), dim_params=(None, None)]], outputs=[output: tensor[dtype=float, shape=(4, 8), dim_params=(None, None)]]) | 2 |  |
| Output value mismatch for Y (got=hello, index=(0, 0), output=Y, reference=hello\x00world) | 1 |  |
| Testbench execution failed: exit code 1 | 1 |  |

## Error frequency by opset

| Error message | Opset | Count |
| --- | --- | --- |
| Out of tolerance | 17 | 1 |

## Failing ONNX files

Lists every ONNX file with a non-success verification outcome.

| File | Opset | Verification | Supported | Error |
| --- | --- | --- | --- | --- |
| test/contrib_ops/attention_op_test/Attention_Mask1D_Fp32_B2_S64_run0/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 1020241) |
| test/contrib_ops/attention_op_test/Attention_Mask2D_Fp32_B2_S32_run0/model.onnx |  | Data/Data | ❌ | Out of tolerance (max ULP 980755) |
| test/contrib_ops/layer_norm_op_test/BERTLayerNorm_NoBias_run0/model.onnx (--test-data-inputs-only) | 17 | Data/ORT | ❌ | Out of tolerance (max ULP 3910) |
| test/contrib_ops/math/matmul_sparse_test/TestCoo_run0/model.onnx |  | Data/Data | ❌ | Unsupported value type 'sparse_tensor_type' for 'A'. Hint: export the model with tensor inputs/outputs. |
| test/contrib_ops/math/matmul_sparse_test/TestCoo_run1/model.onnx |  | Data/Data | ❌ | Unsupported value type 'sparse_tensor_type' for 'A'. Hint: export the model with tensor inputs/outputs. |
| test/contrib_ops/math/matmul_sparse_test/TestCoo_run2/model.onnx |  | Data/Data | ❌ | Unsupported value type 'sparse_tensor_type' for 'A'. Hint: export the model with tensor inputs/outputs. |
| test/contrib_ops/math/matmul_sparse_test/TestCoo_run3/model.onnx |  | Data/Data | ❌ | Unsupported value type 'sparse_tensor_type' for 'A'. Hint: export the model with tensor inputs/outputs. |
| test/contrib_ops/math/matmul_sparse_test/TestCsr_run0/model.onnx |  | Data/Data | ❌ | Unsupported value type 'sparse_tensor_type' for 'A'. Hint: export the model with tensor inputs/outputs. |
| test/contrib_ops/math/matmul_sparse_test/TestCsr_run1/model.onnx |  | Data/Data | ❌ | Unsupported value type 'sparse_tensor_type' for 'A'. Hint: export the model with tensor inputs/outputs. |
| test/contrib_ops/math/matmul_sparse_test/TestCsr_run2/model.onnx |  | Data/Data | ❌ | Unsupported value type 'sparse_tensor_type' for 'A'. Hint: export the model with tensor inputs/outputs. |
| test/contrib_ops/math/matmul_sparse_test/TestCsr_run3/model.onnx |  | Data/Data | ❌ | Unsupported value type 'sparse_tensor_type' for 'A'. Hint: export the model with tensor inputs/outputs. |
| test/contrib_ops/moe_test/MoETest_Gelu_run0/model.onnx |  | Data/Data | ❌ | MoE: only activation_type='swiglu' is supported, got 'gelu' (while lowering node_index=0, op_type=MoE, name=node1, inputs=[input: tensor[dtype=float, shape=(4, 8), dim_params=(None, None)], router_probs: tensor[dtype=float, shape=(4, 4), dim_params=(None, None)], fc1_experts_weights: tensor[dtype=float, shape=(4, 16, 8), dim_params=(None, None, None)], fc1_experts_bias: tensor[dtype=float, shape=(4, 16), dim_params=(None, None)], fc2_experts_weights: tensor[dtype=float, shape=(4, 8, 16), dim_params=(None, None, None)], fc2_experts_bias: tensor[dtype=float, shape=(4, 8), dim_params=(None, None)]], outputs=[output: tensor[dtype=float, shape=(4, 8), dim_params=(None, None)]]) |
| test/contrib_ops/moe_test/MoETest_Relu_run0/model.onnx |  | Data/Data | ❌ | MoE: only activation_type='swiglu' is supported, got 'relu' (while lowering node_index=0, op_type=MoE, name=node1, inputs=[input: tensor[dtype=float, shape=(4, 8), dim_params=(None, None)], router_probs: tensor[dtype=float, shape=(4, 4), dim_params=(None, None)], fc1_experts_weights: tensor[dtype=float, shape=(4, 16, 8), dim_params=(None, None, None)], fc1_experts_bias: tensor[dtype=float, shape=(4, 16), dim_params=(None, None)], fc2_experts_weights: tensor[dtype=float, shape=(4, 8, 16), dim_params=(None, None, None)], fc2_experts_bias: tensor[dtype=float, shape=(4, 8), dim_params=(None, None)]], outputs=[output: tensor[dtype=float, shape=(4, 8), dim_params=(None, None)]]) |
| test/contrib_ops/tokenizer_test/TokenizerWithSeparators_EmptyMatchRegex_run0/model.onnx |  | Data/Data | ❌ | Testbench execution failed: exit code 1 |
| test/contrib_ops/tokenizer_test/Tokenizer_EmbeddedNullBytes_run0/model.onnx |  | Data/Data | ❌ | Output value mismatch for Y (got=hello, index=(0, 0), output=Y, reference=hello\x00world) |

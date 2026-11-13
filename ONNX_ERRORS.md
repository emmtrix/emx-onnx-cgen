<!-- AUTO-GENERATED FILE. DO NOT EDIT. -->
<!-- Regenerate with: UPDATE_REFS=1 pytest -q tests/test_official_onnx_files_docs.py::test_official_onnx_file_support_doc -->

# ONNX verification errors

Aggregates non-success verification outcomes.

| Error message | Count | Opset versions |
| --- | --- | --- |
| Unsupported value type '*' for '*'. Hint: export the model with tensor inputs/outputs. | 8 |  |
| Out of tolerance | 3 | 17 |
| Unsupported op ai.onnx.SwiGLU | 3 | 28 |
| Unique inverse_indices must have shape (6,) (while lowering node_index=0, op_type=Unique, name=&lt;unnamed&gt;, inputs=[X: tensor[dtype=float, shape=(6,), dim_params=(None,)]], outputs=[Y: tensor[dtype=float, shape=(-1,), dim_params=('*',)], indices: tensor[dtype=int64, shape=(-1,), dim_params=('*',)], inverse_indices: tensor[dtype=int64, shape=(-1,), dim_params=('*',)], counts: tensor[dtype=int64, shape=(-1,), dim_params=('*',)]]) | 2 | 28 |
| Unique inverse_indices must have shape (1,) (while lowering node_index=0, op_type=Unique, name=&lt;unnamed&gt;, inputs=[X: tensor[dtype=int64, shape=(1,), dim_params=(None,)]], outputs=[Y: tensor[dtype=int64, shape=(-1,), dim_params=('*',)], indices: tensor[dtype=int64, shape=(-1,), dim_params=('*',)], inverse_indices: tensor[dtype=int64, shape=(-1,), dim_params=('*',)], counts: tensor[dtype=int64, shape=(-1,), dim_params=('*',)]]) | 1 | 28 |
| Unique inverse_indices must have shape (3,) (while lowering node_index=0, op_type=Unique, name=&lt;unnamed&gt;, inputs=[X: tensor[dtype=float, shape=(3, 3), dim_params=(None, None)]], outputs=[Y: tensor[dtype=float, shape=(-1, 3), dim_params=('*', None)], indices: tensor[dtype=int64, shape=(-1,), dim_params=('*',)], inverse_indices: tensor[dtype=int64, shape=(-1,), dim_params=('*',)], counts: tensor[dtype=int64, shape=(-1,), dim_params=('*',)]]) | 1 | 28 |
| Unique inverse_indices must have shape (3,) (while lowering node_index=0, op_type=Unique, name=&lt;unnamed&gt;, inputs=[X: tensor[dtype=float, shape=(3, 3), dim_params=(None, None)]], outputs=[Y: tensor[dtype=float, shape=(3, -1), dim_params=(None, '*')], indices: tensor[dtype=int64, shape=(-1,), dim_params=('*',)], inverse_indices: tensor[dtype=int64, shape=(-1,), dim_params=('*',)], counts: tensor[dtype=int64, shape=(-1,), dim_params=('*',)]]) | 1 | 28 |
| Unique inverse_indices must have shape (4,) (while lowering node_index=0, op_type=Unique, name=&lt;unnamed&gt;, inputs=[X: tensor[dtype=float, shape=(2, 4, 2), dim_params=(None, None, None)]], outputs=[Y: tensor[dtype=float, shape=(2, -1, 2), dim_params=(None, '*', None)], indices: tensor[dtype=int64, shape=(-1,), dim_params=('*',)], inverse_indices: tensor[dtype=int64, shape=(-1,), dim_params=('*',)], counts: tensor[dtype=int64, shape=(-1,), dim_params=('*',)]]) | 1 | 28 |

## Error frequency by opset

| Error message | Opset | Count |
| --- | --- | --- |
| Out of tolerance | 17 | 1 |
| Unsupported op ai.onnx.SwiGLU | 28 | 3 |
| Unique inverse_indices must have shape (6,) (while lowering node_index=0, op_type=Unique, name=&lt;unnamed&gt;, inputs=[X: tensor[dtype=float, shape=(6,), dim_params=(None,)]], outputs=[Y: tensor[dtype=float, shape=(-1,), dim_params=('*',)], indices: tensor[dtype=int64, shape=(-1,), dim_params=('*',)], inverse_indices: tensor[dtype=int64, shape=(-1,), dim_params=('*',)], counts: tensor[dtype=int64, shape=(-1,), dim_params=('*',)]]) | 28 | 2 |
| Unique inverse_indices must have shape (1,) (while lowering node_index=0, op_type=Unique, name=&lt;unnamed&gt;, inputs=[X: tensor[dtype=int64, shape=(1,), dim_params=(None,)]], outputs=[Y: tensor[dtype=int64, shape=(-1,), dim_params=('*',)], indices: tensor[dtype=int64, shape=(-1,), dim_params=('*',)], inverse_indices: tensor[dtype=int64, shape=(-1,), dim_params=('*',)], counts: tensor[dtype=int64, shape=(-1,), dim_params=('*',)]]) | 28 | 1 |
| Unique inverse_indices must have shape (3,) (while lowering node_index=0, op_type=Unique, name=&lt;unnamed&gt;, inputs=[X: tensor[dtype=float, shape=(3, 3), dim_params=(None, None)]], outputs=[Y: tensor[dtype=float, shape=(-1, 3), dim_params=('*', None)], indices: tensor[dtype=int64, shape=(-1,), dim_params=('*',)], inverse_indices: tensor[dtype=int64, shape=(-1,), dim_params=('*',)], counts: tensor[dtype=int64, shape=(-1,), dim_params=('*',)]]) | 28 | 1 |
| Unique inverse_indices must have shape (3,) (while lowering node_index=0, op_type=Unique, name=&lt;unnamed&gt;, inputs=[X: tensor[dtype=float, shape=(3, 3), dim_params=(None, None)]], outputs=[Y: tensor[dtype=float, shape=(3, -1), dim_params=(None, '*')], indices: tensor[dtype=int64, shape=(-1,), dim_params=('*',)], inverse_indices: tensor[dtype=int64, shape=(-1,), dim_params=('*',)], counts: tensor[dtype=int64, shape=(-1,), dim_params=('*',)]]) | 28 | 1 |
| Unique inverse_indices must have shape (4,) (while lowering node_index=0, op_type=Unique, name=&lt;unnamed&gt;, inputs=[X: tensor[dtype=float, shape=(2, 4, 2), dim_params=(None, None, None)]], outputs=[Y: tensor[dtype=float, shape=(2, -1, 2), dim_params=(None, '*', None)], indices: tensor[dtype=int64, shape=(-1,), dim_params=('*',)], inverse_indices: tensor[dtype=int64, shape=(-1,), dim_params=('*',)], counts: tensor[dtype=int64, shape=(-1,), dim_params=('*',)]]) | 28 | 1 |

## Failing ONNX files

Lists every ONNX file with a non-success verification outcome.

| File | Opset | Verification | Supported | Error |
| --- | --- | --- | --- | --- |
| node/test_swiglu/model.onnx | 28 | Data/Data | ❌ | Unsupported op ai.onnx.SwiGLU |
| node/test_swiglu_alpha/model.onnx | 28 | Data/Data | ❌ | Unsupported op ai.onnx.SwiGLU |
| node/test_swiglu_float16/model.onnx | 28 | Data/Data | ❌ | Unsupported op ai.onnx.SwiGLU |
| node/test_unique_length_1/model.onnx | 28 | Data/Data | ❌ | Unique inverse_indices must have shape (1,) (while lowering node_index=0, op_type=Unique, name=&lt;unnamed&gt;, inputs=[X: tensor[dtype=int64, shape=(1,), dim_params=(None,)]], outputs=[Y: tensor[dtype=int64, shape=(-1,), dim_params=('unk__0',)], indices: tensor[dtype=int64, shape=(-1,), dim_params=('unk__1',)], inverse_indices: tensor[dtype=int64, shape=(-1,), dim_params=('unk__2',)], counts: tensor[dtype=int64, shape=(-1,), dim_params=('unk__3',)]]) |
| node/test_unique_not_sorted_without_axis/model.onnx | 28 | Data/Data | ❌ | Unique inverse_indices must have shape (6,) (while lowering node_index=0, op_type=Unique, name=&lt;unnamed&gt;, inputs=[X: tensor[dtype=float, shape=(6,), dim_params=(None,)]], outputs=[Y: tensor[dtype=float, shape=(-1,), dim_params=('unk__0',)], indices: tensor[dtype=int64, shape=(-1,), dim_params=('unk__1',)], inverse_indices: tensor[dtype=int64, shape=(-1,), dim_params=('unk__2',)], counts: tensor[dtype=int64, shape=(-1,), dim_params=('unk__3',)]]) |
| node/test_unique_sorted_with_axis/model.onnx | 28 | Data/Data | ❌ | Unique inverse_indices must have shape (3,) (while lowering node_index=0, op_type=Unique, name=&lt;unnamed&gt;, inputs=[X: tensor[dtype=float, shape=(3, 3), dim_params=(None, None)]], outputs=[Y: tensor[dtype=float, shape=(-1, 3), dim_params=('unk__0', None)], indices: tensor[dtype=int64, shape=(-1,), dim_params=('unk__1',)], inverse_indices: tensor[dtype=int64, shape=(-1,), dim_params=('unk__2',)], counts: tensor[dtype=int64, shape=(-1,), dim_params=('unk__3',)]]) |
| node/test_unique_sorted_with_axis_3d/model.onnx | 28 | Data/Data | ❌ | Unique inverse_indices must have shape (4,) (while lowering node_index=0, op_type=Unique, name=&lt;unnamed&gt;, inputs=[X: tensor[dtype=float, shape=(2, 4, 2), dim_params=(None, None, None)]], outputs=[Y: tensor[dtype=float, shape=(2, -1, 2), dim_params=(None, 'unk__0', None)], indices: tensor[dtype=int64, shape=(-1,), dim_params=('unk__1',)], inverse_indices: tensor[dtype=int64, shape=(-1,), dim_params=('unk__2',)], counts: tensor[dtype=int64, shape=(-1,), dim_params=('unk__3',)]]) |
| node/test_unique_sorted_with_negative_axis/model.onnx | 28 | Data/Data | ❌ | Unique inverse_indices must have shape (3,) (while lowering node_index=0, op_type=Unique, name=&lt;unnamed&gt;, inputs=[X: tensor[dtype=float, shape=(3, 3), dim_params=(None, None)]], outputs=[Y: tensor[dtype=float, shape=(3, -1), dim_params=(None, 'unk__0')], indices: tensor[dtype=int64, shape=(-1,), dim_params=('unk__1',)], inverse_indices: tensor[dtype=int64, shape=(-1,), dim_params=('unk__2',)], counts: tensor[dtype=int64, shape=(-1,), dim_params=('unk__3',)]]) |
| node/test_unique_sorted_without_axis/model.onnx | 28 | Data/Data | ❌ | Unique inverse_indices must have shape (6,) (while lowering node_index=0, op_type=Unique, name=&lt;unnamed&gt;, inputs=[X: tensor[dtype=float, shape=(6,), dim_params=(None,)]], outputs=[Y: tensor[dtype=float, shape=(-1,), dim_params=('unk__0',)], indices: tensor[dtype=int64, shape=(-1,), dim_params=('unk__1',)], inverse_indices: tensor[dtype=int64, shape=(-1,), dim_params=('unk__2',)], counts: tensor[dtype=int64, shape=(-1,), dim_params=('unk__3',)]]) |
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

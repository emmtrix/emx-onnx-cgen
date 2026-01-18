# Pytest Speed Report

Generated: 2026-01-18T08:21:44.048201+00:00

## Run Summary
- Command: `/root/.pyenv/versions/3.12.12/bin/python -m pytest --durations=0 --durations-min=0 -n auto -q`
- Exit code: 0

## Slowest Durations

| Rank | Duration (s) | Phase | Test |
| ---: | ---: | --- | --- |
| 1 | 49.760 | call | `tests/test_official_onnx_files.py::test_local_onnx_test_data_matches_testbench` |
| 2 | 7.160 | call | `tests/test_official_onnx_files.py::test_official_onnx_expected_errors` |
| 3 | 3.650 | call | `tests/test_endtoend_features.py::test_initializer_weights_emitted_as_static_arrays` |
| 4 | 2.430 | call | `tests/test_cli.py::test_cli_verify_reduce_model` |
| 5 | 1.840 | call | `tests/test_cli.py::test_cli_verify_operator_model` |
| 6 | 1.360 | call | `tests/test_golden_ops.py::test_op_eyelike_eye_like` |
| 7 | 1.330 | call | `tests/test_ops.py::test_gridsample_op_matches_onnxruntime` |
| 8 | 1.280 | call | `tests/test_ops.py::test_resize_op_matches_onnxruntime` |
| 9 | 1.260 | call | `tests/test_golden_ops.py::test_op_instancenormalization_instance_normalization` |
| 10 | 1.210 | call | `tests/test_golden_ops.py::test_op_reduce_reduce_mean` |
| 11 | 1.210 | call | `tests/test_endtoend_features.py::test_testbench_accepts_constant_inputs` |
| 12 | 1.200 | call | `tests/test_codegen_data_file.py::test_compile_with_data_file_emits_externs` |
| 13 | 1.190 | call | `tests/test_golden_ops.py::test_op_size_size` |
| 14 | 1.170 | call | `tests/test_golden_ops.py::test_op_lrn_lrn` |
| 15 | 1.120 | call | `tests/test_multi_output.py::test_multi_output_graph_compile_and_run` |
| 16 | 1.090 | call | `tests/test_golden_ops.py::test_op_transpose_transpose` |
| 17 | 1.070 | call | `tests/test_golden_ops.py::test_op_mish_mish` |
| 18 | 1.060 | call | `tests/test_golden_ops.py::test_op_conv_conv` |
| 19 | 1.000 | call | `tests/test_golden_ops.py::test_op_concat_concat` |
| 20 | 0.910 | call | `tests/test_golden.py::test_codegen_includes_testbench` |

Captured 765 duration entries (showing top 20).

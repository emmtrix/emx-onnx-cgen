# Pytest Speed Report

Generated: 2026-01-18T10:41:22.376763+00:00

## Run Summary
- Command: `/root/.pyenv/versions/3.12.12/bin/python -m pytest --durations=0 --durations-min=0`
- Exit code: 0

## Slowest Durations

| Rank | Duration (s) | Phase | Test |
| ---: | ---: | --- | --- |
| 1 | 25.370 | call | `tests/test_official_onnx_files.py::test_official_onnx_expected_errors[onnx-org/onnx/backend/test/data/light/light_densenet121.onnx]` |
| 2 | 13.240 | call | `tests/test_official_onnx_files.py::test_official_onnx_expected_errors[onnx-org/onnx/backend/test/data/light/light_inception_v2.onnx]` |
| 3 | 5.530 | call | `tests/test_official_onnx_files.py::test_official_onnx_expected_errors[onnx-org/onnx/backend/test/data/light/light_shufflenet.onnx]` |
| 4 | 5.280 | call | `tests/test_official_onnx_files.py::test_official_onnx_expected_errors[onnx-org/onnx/backend/test/data/light/light_resnet50.onnx]` |
| 5 | 4.290 | call | `tests/test_official_onnx_files.py::test_official_onnx_expected_errors[onnx-org/onnx/backend/test/data/light/light_inception_v1.onnx]` |
| 6 | 2.960 | call | `tests/test_official_onnx_files.py::test_official_onnx_expected_errors[onnx-org/onnx/backend/test/data/light/light_squeezenet.onnx]` |
| 7 | 2.620 | call | `tests/test_endtoend_features.py::test_initializer_weights_emitted_as_static_arrays` |
| 8 | 2.270 | call | `tests/test_official_onnx_files.py::test_official_onnx_expected_errors[onnx-org/onnx/backend/test/data/light/light_vgg19.onnx]` |
| 9 | 1.650 | call | `tests/test_cli.py::test_cli_verify_operator_model` |
| 10 | 1.590 | call | `tests/test_cli.py::test_cli_verify_reduce_model` |
| 11 | 1.410 | call | `tests/test_official_onnx_files.py::test_official_onnx_expected_errors[onnx-org/onnx/backend/test/data/node/test_attention_3d_attn_mask_expanded/model.onnx]` |
| 12 | 1.370 | call | `tests/test_official_onnx_files.py::test_official_onnx_expected_errors[onnx-org/onnx/backend/test/data/light/light_bvlc_alexnet.onnx]` |
| 13 | 1.350 | call | `tests/test_official_onnx_files.py::test_official_onnx_expected_errors[onnx-org/onnx/backend/test/data/light/light_zfnet512.onnx]` |
| 14 | 1.320 | call | `tests/test_official_onnx_files.py::test_official_onnx_expected_errors[onnx-org/onnx/backend/test/data/node/test_attention_3d_diff_heads_sizes_attn_mask_expanded/model.onnx]` |
| 15 | 1.060 | call | `tests/test_official_onnx_files.py::test_official_onnx_expected_errors[onnx-org/onnx/backend/test/data/node/test_attention_3d_attn_mask/model.onnx]` |
| 16 | 0.960 | call | `tests/test_official_onnx_files.py::test_official_onnx_expected_errors[onnx-org/onnx/backend/test/data/node/test_argmax_no_keepdims_random_select_last_index/model.onnx]` |
| 17 | 0.940 | call | `tests/test_official_onnx_files.py::test_official_onnx_expected_errors[onnx-org/onnx/backend/test/data/node/test_attention_3d_diff_heads_sizes_attn_mask/model.onnx]` |
| 19 | 0.930 | call | `tests/test_official_onnx_files.py::test_official_onnx_expected_errors[onnx-org/onnx/backend/test/data/node/test_attention_3d/model.onnx]` |
| 20 | 0.910 | call | `tests/test_official_onnx_files.py::test_official_onnx_expected_errors[onnx-org/onnx/backend/test/data/node/test_attention_3d_diff_heads_sizes_causal/model.onnx]` |
| 22 | 0.890 | call | `tests/test_official_onnx_files.py::test_official_onnx_expected_errors[onnx-org/onnx/backend/test/data/node/test_attention_3d_diff_heads_sizes/model.onnx]` |
| 24 | 0.880 | call | `tests/test_official_onnx_files.py::test_official_onnx_expected_errors[onnx-org/onnx/backend/test/data/node/test_abs/model.onnx]` |
| 27 | 0.870 | call | `tests/test_official_onnx_files.py::test_official_onnx_expected_errors[onnx-org/onnx/backend/test/data/node/test_argmin_default_axis_random/model.onnx]` |
| 30 | 0.860 | call | `tests/test_official_onnx_files.py::test_official_onnx_expected_errors[onnx-org/onnx/backend/test/data/node/test_argmax_default_axis_random_select_last_index/model.onnx]` |
| 32 | 0.860 | call | `tests/test_official_onnx_files.py::test_official_onnx_expected_errors[onnx-org/onnx/backend/test/data/node/test_argmax_negative_axis_keepdims_random_select_last_index/model.onnx]` |
| 33 | 0.850 | call | `tests/test_official_onnx_files.py::test_official_onnx_expected_errors[onnx-org/onnx/backend/test/data/node/test_argmin_negative_axis_keepdims_random_select_last_index/model.onnx]` |
| 34 | 0.850 | call | `tests/test_official_onnx_files.py::test_official_onnx_expected_errors[onnx-org/onnx/backend/test/data/node/test_attention_3d_causal/model.onnx]` |
| 35 | 0.850 | call | `tests/test_official_onnx_files.py::test_official_onnx_expected_errors[onnx-org/onnx/backend/test/data/node/test_add_uint16/model.onnx]` |
| 37 | 0.840 | call | `tests/test_official_onnx_files.py::test_official_onnx_expected_errors[onnx-org/onnx/backend/test/data/node/test_acosh/model.onnx]` |
| 38 | 0.840 | call | `tests/test_official_onnx_files.py::test_official_onnx_expected_errors[onnx-org/onnx/backend/test/data/node/test_argmin_default_axis_random_select_last_index/model.onnx]` |
| 40 | 0.830 | call | `tests/test_official_onnx_files.py::test_official_onnx_expected_errors[onnx-org/onnx/backend/test/data/node/test_add_uint32/model.onnx]` |
| 43 | 0.830 | call | `tests/test_official_onnx_files.py::test_official_onnx_expected_errors[onnx-org/onnx/backend/test/data/node/test_atan/model.onnx]` |
| 44 | 0.820 | call | `tests/test_official_onnx_files.py::test_official_onnx_expected_errors[onnx-org/onnx/backend/test/data/node/test_add_uint8/model.onnx]` |
| 46 | 0.820 | call | `tests/test_official_onnx_files.py::test_official_onnx_expected_errors[onnx-org/onnx/backend/test/data/node/test_add_int16/model.onnx]` |
| 49 | 0.820 | call | `tests/test_official_onnx_files.py::test_official_onnx_expected_errors[onnx-org/onnx/backend/test/data/node/test_argmax_negative_axis_keepdims_random/model.onnx]` |
| 50 | 0.820 | call | `tests/test_official_onnx_files.py::test_official_onnx_expected_errors[onnx-org/onnx/backend/test/data/node/test_argmin_keepdims_random/model.onnx]` |
| 51 | 0.820 | call | `tests/test_official_onnx_files.py::test_official_onnx_expected_errors[onnx-org/onnx/backend/test/data/node/test_asinh/model.onnx]` |
| 53 | 0.820 | call | `tests/test_official_onnx_files.py::test_official_onnx_expected_errors[onnx-org/onnx/backend/test/data/node/test_add/model.onnx]` |
| 57 | 0.820 | call | `tests/test_official_onnx_files.py::test_official_onnx_expected_errors[onnx-org/onnx/backend/test/data/node/test_add_uint64/model.onnx]` |
| 58 | 0.820 | call | `tests/test_official_onnx_files.py::test_official_onnx_expected_errors[onnx-org/onnx/backend/test/data/node/test_add_bcast/model.onnx]` |
| 59 | 0.810 | call | `tests/test_official_onnx_files.py::test_official_onnx_expected_errors[onnx-org/onnx/backend/test/data/node/test_atanh/model.onnx]` |
| 60 | 0.810 | call | `tests/test_official_onnx_files.py::test_official_onnx_expected_errors[onnx-org/onnx/backend/test/data/node/test_acos/model.onnx]` |
| 63 | 0.810 | call | `tests/test_official_onnx_files.py::test_official_onnx_expected_errors[onnx-org/onnx/backend/test/data/node/test_argmin_negative_axis_keepdims_random/model.onnx]` |
| 64 | 0.810 | call | `tests/test_ops.py::test_resize_op_matches_onnxruntime` |
| 65 | 0.810 | call | `tests/test_official_onnx_files.py::test_official_onnx_expected_errors[onnx-org/onnx/backend/test/data/node/test_asin/model.onnx]` |
| 67 | 0.810 | call | `tests/test_official_onnx_files.py::test_official_onnx_expected_errors[onnx-org/onnx/backend/test/data/node/test_argmin_no_keepdims_random/model.onnx]` |
| 68 | 0.810 | call | `tests/test_official_onnx_files.py::test_official_onnx_expected_errors[onnx-org/onnx/backend/test/data/node/test_and4d/model.onnx]` |
| 75 | 0.800 | call | `tests/test_official_onnx_files.py::test_official_onnx_expected_errors[onnx-org/onnx/backend/test/data/node/test_add_int8/model.onnx]` |
| 78 | 0.800 | call | `tests/test_official_onnx_files.py::test_official_onnx_expected_errors[onnx-org/onnx/backend/test/data/node/test_argmax_default_axis_example/model.onnx]` |
| 79 | 0.800 | call | `tests/test_ops.py::test_gridsample_op_matches_onnxruntime` |
| 80 | 0.800 | call | `tests/test_official_onnx_files.py::test_official_onnx_expected_errors[onnx-org/onnx/backend/test/data/node/test_argmax_keepdims_random_select_last_index/model.onnx]` |
| 82 | 0.800 | call | `tests/test_official_onnx_files.py::test_official_onnx_expected_errors[onnx-org/onnx/backend/test/data/node/test_asinh_example/model.onnx]` |
| 84 | 0.790 | call | `tests/test_official_onnx_files.py::test_official_onnx_expected_errors[onnx-org/onnx/backend/test/data/node/test_argmin_default_axis_example_select_last_index/model.onnx]` |
| 85 | 0.790 | call | `tests/test_official_onnx_files.py::test_official_onnx_expected_errors[onnx-org/onnx/backend/test/data/node/test_acos_example/model.onnx]` |
| 88 | 0.790 | call | `tests/test_official_onnx_files.py::test_official_onnx_expected_errors[onnx-org/onnx/backend/test/data/node/test_atanh_example/model.onnx]` |
| 89 | 0.790 | call | `tests/test_official_onnx_files.py::test_official_onnx_expected_errors[onnx-org/onnx/backend/test/data/node/test_asin_example/model.onnx]` |
| 90 | 0.790 | call | `tests/test_official_onnx_files.py::test_official_onnx_expected_errors[onnx-org/onnx/backend/test/data/node/test_argmax_negative_axis_keepdims_example_select_last_index/model.onnx]` |
| 91 | 0.790 | call | `tests/test_official_onnx_files.py::test_official_onnx_expected_errors[onnx-org/onnx/backend/test/data/node/test_argmax_no_keepdims_example/model.onnx]` |
| 94 | 0.790 | call | `tests/test_official_onnx_files.py::test_official_onnx_expected_errors[onnx-org/onnx/backend/test/data/node/test_and3d/model.onnx]` |
| 97 | 0.780 | call | `tests/test_official_onnx_files.py::test_official_onnx_expected_errors[onnx-org/onnx/backend/test/data/node/test_atan_example/model.onnx]` |
| 98 | 0.780 | call | `tests/test_official_onnx_files.py::test_official_onnx_expected_errors[onnx-org/onnx/backend/test/data/node/test_argmin_no_keepdims_example/model.onnx]` |
| 100 | 0.780 | call | `tests/test_official_onnx_files.py::test_official_onnx_expected_errors[onnx-org/onnx/backend/test/data/node/test_argmax_keepdims_random/model.onnx]` |

Captured 1275 duration entries (showing top 100).

## Slowest Test Note

The slowest recorded test in this run was `tests/test_official_onnx_files.py::test_official_onnx_expected_errors[onnx-org/onnx/backend/test/data/light/light_densenet121.onnx]` at 25.370 seconds.

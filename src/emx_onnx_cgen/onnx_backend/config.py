from __future__ import annotations

import os
from collections.abc import Iterable, Mapping

DEFAULT_TESTS: tuple[str, ...] = (
    "test_abs_cpu",
    "test_cast_FLOAT_to_FLOAT16_cpu",
    "test_cast_FLOAT_to_DOUBLE_cpu",
    "test_add_int8_cpu",
    "test_add_int16_cpu",
    "test_max_int32_cpu",
    "test_max_int64_cpu",
    "test_add_uint8_cpu",
    "test_add_uint16_cpu",
    "test_add_uint32_cpu",
    "test_add_uint64_cpu",
    "test_reduce_max_bool_inputs_cpu",
    "test_cast_FLOAT_to_BFLOAT16_cpu",
    "test_equal_string_cpu",
    "test_string_concat_utf8_cpu",
    "test_string_split_empty_tensor_cpu",
    "test_operator_conv_cpu",
    "test_optional_get_element_sequence_cpu",
    "test_optional_has_element_optional_input_cpu",
)

SELECTED_TESTS_ENV = "EMX_ONNX_BACKEND_TESTS"
RUN_ALL_TESTS_ENV = "EMX_ONNX_BACKEND_RUN_ALL"


def _split_test_names(raw_value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in raw_value.split(",") if part.strip())


def parse_selected_tests(
    names: Iterable[str] | None,
    *,
    default_tests: Iterable[str] = DEFAULT_TESTS,
) -> tuple[str, ...]:
    if names is None:
        return tuple(default_tests)
    parsed = tuple(name.strip() for name in names if name.strip())
    return parsed


def selected_tests_from_env(
    env: Mapping[str, str] | None = None,
    *,
    default_tests: Iterable[str] = DEFAULT_TESTS,
) -> tuple[str, ...] | None:
    active_env = os.environ if env is None else env
    if active_env.get(RUN_ALL_TESTS_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return None
    raw_names = active_env.get(SELECTED_TESTS_ENV)
    if raw_names is None:
        return tuple(default_tests)
    return parse_selected_tests(
        _split_test_names(raw_names), default_tests=default_tests
    )

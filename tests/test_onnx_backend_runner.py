from __future__ import annotations

from pathlib import Path

from emx_onnx_cgen.onnx_backend.config import DEFAULT_TESTS, selected_tests_from_env
from emx_onnx_cgen.onnx_backend.runner import build_pytest_args


def test_selected_tests_from_env_uses_defaults_when_not_configured() -> None:
    assert selected_tests_from_env(env={}) == DEFAULT_TESTS


def test_selected_tests_from_env_supports_run_all_flag() -> None:
    assert selected_tests_from_env(env={"EMX_ONNX_BACKEND_RUN_ALL": "1"}) is None


def test_default_backend_tests_cover_float8_castlike_no_saturate_regressions() -> None:
    assert "test_castlike_no_saturate_FLOAT16_to_FLOAT8E4M3FN_cpu" in DEFAULT_TESTS
    assert "test_castlike_no_saturate_FLOAT16_to_FLOAT8E4M3FNUZ_cpu" in DEFAULT_TESTS


def test_build_pytest_args_uses_explicit_node_ids() -> None:
    args = build_pytest_args(
        selected_tests=("test_abs_cpu", "test_add_cpu"),
        pytest_args=("-x",),
    )
    suite_path = Path(args[2])
    assert suite_path.name == "test_suite.py"
    assert args[3] == "-x"
    assert len(args) == 4

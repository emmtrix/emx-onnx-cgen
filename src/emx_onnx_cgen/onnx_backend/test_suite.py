from __future__ import annotations

import importlib
import os

import onnx.backend.test

from .config import selected_tests_from_env

pytest_plugins = ("emx_onnx_cgen.onnx_backend.pytest_plugin",)


def _import_backend(module_name: str):
    backend = importlib.import_module(module_name)
    if not hasattr(backend, "run_model") and not hasattr(backend, "run"):
        raise ValueError(f"{module_name} is not a valid ONNX backend")
    return backend


def _prune_suite_to_selection(suite: type, selected_tests: tuple[str, ...] | None) -> None:
    if selected_tests is None:
        return
    selected = set(selected_tests)
    for attr in dir(suite):
        if not attr.startswith("test_"):
            continue
        if attr in selected:
            continue
        delattr(suite, attr)


backend = _import_backend(
    os.getenv("ONNX_BACKEND_MODULE", "emx_onnx_cgen.onnx_backend")
)
backend.backend_name = "emx-onnx-cgen"

backend_test = onnx.backend.test.BackendTest(backend, __name__)
for suite_name, suite in backend_test.enable_report().test_cases.items():
    _prune_suite_to_selection(suite, selected_tests_from_env())
    globals()[suite_name] = suite

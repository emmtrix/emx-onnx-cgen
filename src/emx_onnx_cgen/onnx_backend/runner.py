from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

import pytest

from .config import DEFAULT_TESTS, RUN_ALL_TESTS_ENV, SELECTED_TESTS_ENV, parse_selected_tests


def build_pytest_args(
    *,
    selected_tests: Sequence[str] | None,
    pytest_args: Sequence[str],
) -> list[str]:
    suite_path = Path(__file__).with_name("test_suite.py")
    args = ["-q", "-ra", str(suite_path)]
    args.extend(pytest_args)
    return args


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="emx-onnx-cgen-backend-test",
        description="Run ONNX backend compliance smoke tests for emx-onnx-cgen.",
    )
    parser.add_argument(
        "--backend-module",
        default="emx_onnx_cgen.onnx_backend",
        help="Python module path of the ONNX backend to import.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all collected backend tests instead of the default smoke list.",
    )
    parser.add_argument(
        "--list-default-tests",
        action="store_true",
        help="Print the default smoke-test list and exit.",
    )
    parser.add_argument(
        "tests",
        nargs="*",
        help="Explicit backend test method names to run instead of the default list.",
    )
    parser.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="Additional pytest arguments. Prefix them with -- to separate them.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.list_default_tests:
        for name in DEFAULT_TESTS:
            print(name)
        return 0

    pytest_args = list(args.pytest_args)
    if pytest_args and pytest_args[0] == "--":
        pytest_args = pytest_args[1:]

    os.environ["ONNX_BACKEND_MODULE"] = args.backend_module
    if args.all:
        os.environ[RUN_ALL_TESTS_ENV] = "1"
        os.environ.pop(SELECTED_TESTS_ENV, None)
        selected_tests = None
    else:
        selected_tests = parse_selected_tests(args.tests or None)
        os.environ[RUN_ALL_TESTS_ENV] = "0"
        os.environ[SELECTED_TESTS_ENV] = ",".join(selected_tests)

    return pytest.main(
        build_pytest_args(selected_tests=selected_tests, pytest_args=pytest_args)
    )

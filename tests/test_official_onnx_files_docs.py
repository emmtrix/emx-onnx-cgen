from __future__ import annotations

import os
import re
from collections import Counter
from pathlib import Path

import pytest

from test_official_onnx_files import (
    LOCAL_ONNX_DATA_ROOT,
    LOCAL_REPO_ONNX_DATA_ROOT,
    MODEL_EXTRA_VERIFY_ARGS,
    OnnxFileExpectation,
    _load_expectation_for_repo_relative,
    _local_onnx_file_paths,
    _local_repo_onnx_file_paths,
    _maybe_init_onnx_org,
    _official_onnx_file_paths,
    _repo_root,
)

OFFICIAL_ONNX_FILE_SUPPORT_PATH = (
    Path(__file__).resolve().parents[1] / "ONNX_SUPPORT.md"
)
OFFICIAL_ONNX_FILE_SUPPORT_HISTOGRAM_PATH = (
    Path(__file__).resolve().parents[1] / "ONNX_ERRORS_HISTOGRAM.md"
)
SUPPORT_OPS_PATH = Path(__file__).resolve().parents[1] / "SUPPORT_OPS.md"
ONNX_VERSION_PATH = Path(__file__).resolve().parents[1] / "onnx-org" / "VERSION_NUMBER"


def _is_success_message(message: str) -> bool:
    return message == "" or message.startswith("OK")


def _render_onnx_file_support_table(
    expectations: list[OnnxFileExpectation],
) -> list[str]:
    lines = [
        "| File | Opset | Supported | Error |",
        "| --- | --- | --- | --- |",
    ]
    for expectation in sorted(expectations, key=lambda item: item.path):
        supported = "✅" if _is_success_message(expectation.error) else "❌"
        opset = (
            str(expectation.opset_version)
            if expectation.opset_version is not None
            else ""
        )
        message = expectation.error.replace("\n", " ").strip()
        extra_args = MODEL_EXTRA_VERIFY_ARGS.get(expectation.path)
        if extra_args:
            flag_text = " ".join(extra_args)
            message = f"{message} (flags: {flag_text})".strip()
        lines.append(
            f"| {expectation.path} | {opset} | {supported} | {message} |"
        )
    return lines


def _render_onnx_file_support_section(
    *,
    title: str,
    test_directory: str,
    expectations: list[OnnxFileExpectation],
) -> list[str]:
    supported_count = sum(
        1 for expectation in expectations if _is_success_message(expectation.error)
    )
    total_count = len(expectations)
    support_percent = (
        (supported_count / total_count) * 100.0 if total_count else 0.0
    )
    return [
        f"## {title}",
        "",
        f"Test directory: `{test_directory}`",
        "",
        f"Coverage {supported_count} / {total_count} ONNX files ({support_percent:.1f}%).",
        "",
        *_render_onnx_file_support_table(expectations),
    ]


def _render_onnx_file_support_markdown(
    official_expectations: list[OnnxFileExpectation],
    local_expectations: list[OnnxFileExpectation],
    local_repo_expectations: list[OnnxFileExpectation],
) -> str:
    onnx_version = ONNX_VERSION_PATH.read_text(encoding="utf-8").strip()
    official_supported_count = sum(
        1
        for expectation in official_expectations
        if _is_success_message(expectation.error)
    )
    official_total_count = len(official_expectations)
    local_supported_count = sum(
        1 for expectation in local_expectations if _is_success_message(expectation.error)
    )
    local_total_count = len(local_expectations)
    official_support_percent = (
        (official_supported_count / official_total_count) * 100.0
        if official_total_count
        else 0.0
    )
    local_support_percent = (
        (local_supported_count / local_total_count) * 100.0
        if local_total_count
        else 0.0
    )
    local_repo_supported_count = sum(
        1
        for expectation in local_repo_expectations
        if _is_success_message(expectation.error)
    )
    local_repo_total_count = len(local_repo_expectations)
    local_repo_support_percent = (
        (local_repo_supported_count / local_repo_total_count) * 100.0
        if local_repo_total_count
        else 0.0
    )
    lines = [
        "# ONNX test coverage",
        "",
        "Overview:",
        "",
        "| Test suite | Coverage | Version |",
        "| --- | --- | --- |",
        (
            "| [Official ONNX test coverage](#official-onnx-test-coverage) "
            f"| {official_supported_count} / {official_total_count}, "
            f"{official_support_percent:.1f}% | {onnx_version} |"
        ),
        (
            "| [ONNX2C test coverage](#onnx2c-test-coverage) "
            f"| {local_supported_count} / {local_total_count}, "
            f"{local_support_percent:.1f}% | n/a |"
        ),
        (
            "| [Local ONNX test coverage](#local-onnx-test-coverage) "
            f"| {local_repo_supported_count} / {local_repo_total_count}, "
            f"{local_repo_support_percent:.1f}% | n/a |"
        ),
        "",
        "See [`ONNX_ERRORS_HISTOGRAM.md`](ONNX_ERRORS_HISTOGRAM.md) for the error histogram.",
        "",
        (
            "Floating-point verification first ignores very small differences up to "
            "**1.0 × [machine epsilon](https://en.wikipedia.org/wiki/Machine_epsilon) "
            "of the evaluated floating-point type**, treating such values as equal. "
            "For values with a larger absolute difference, the ULP distance is "
            "computed, and the maximum ULP distance is reported."
        ),
        "",
        *_render_onnx_file_support_section(
            title="Official ONNX test coverage",
            test_directory="onnx-org/onnx/backend/test/data",
            expectations=official_expectations,
        ),
        "",
        *_render_onnx_file_support_section(
            title="ONNX2C test coverage",
            test_directory="onnx2c-org/test",
            expectations=local_expectations,
        ),
        "",
        *_render_onnx_file_support_section(
            title="Local ONNX test coverage",
            test_directory="tests/onnx",
            expectations=local_repo_expectations,
        ),
    ]
    return "\n".join(lines)


def _render_error_histogram_markdown(
    expectations: list[OnnxFileExpectation],
    title: str = "# Error frequency",
) -> str:
    def _next_heading(title_text: str, default_level: int = 2) -> str:
        match = re.match(r"(#+)\\s+", title_text)
        if match:
            next_level = len(match.group(1)) + 1
            return f"{'#' * next_level} Error frequency by opset"
        return f"{'#' * default_level} Error frequency by opset"

    def _sanitize_error(error: str) -> str:
        if error.startswith("Out of tolerance"):
            return "Out of tolerance"
        if error.startswith("ONNX Runtime failed to run"):
            return "ONNX Runtime failed to run"
        return re.sub(r"'[^']*'", "'*'", error)

    errors: list[str] = []
    error_opsets: dict[str, set[int]] = {}
    error_opset_pairs: list[tuple[str, int]] = []
    for expectation in expectations:
        if not expectation.error or _is_success_message(expectation.error):
            continue
        sanitized_error = _sanitize_error(expectation.error)
        errors.append(sanitized_error)
        if expectation.opset_version is None:
            continue
        error_opsets.setdefault(sanitized_error, set()).add(
            expectation.opset_version
        )
        error_opset_pairs.append(
            (sanitized_error, expectation.opset_version)
        )
    counts = Counter(errors)
    if not counts:
        return ""
    lines = [
        title,
        "",
        "| Error message | Count | Opset versions |",
        "| --- | --- | --- |",
    ]
    for error, count in sorted(
        counts.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        opset_versions = ", ".join(
            str(opset) for opset in sorted(error_opsets.get(error, set()))
        )
        lines.append(f"| {error} | {count} | {opset_versions} |")
    if error_opset_pairs:
        pair_counts = Counter(error_opset_pairs)
        lines.extend(
            [
                "",
                _next_heading(title),
                "",
                "| Error message | Opset | Count |",
                "| --- | --- | --- |",
            ]
        )
        for (error, opset), count in sorted(
            pair_counts.items(),
            key=lambda item: (item[0][1], -item[1], item[0][0]),
        ):
            lines.append(f"| {error} | {opset} | {count} |")
    lines.append("")
    return "\n".join(lines)


def _render_support_histogram_markdown(
    official_expectations: list[OnnxFileExpectation],
    local_expectations: list[OnnxFileExpectation],
    local_repo_expectations: list[OnnxFileExpectation],
) -> str:
    merged_expectations = [
        *official_expectations,
        *local_expectations,
        *local_repo_expectations,
    ]
    histogram_markdown = _render_error_histogram_markdown(
        merged_expectations,
        title="# Error frequency",
    )
    lines = histogram_markdown.splitlines()
    if lines and lines[0] == "# Error frequency":
        lines.insert(2, "This histogram is test-suite-overarching.")
        lines.insert(3, "")
    return "\n".join(lines) + "\n"


def _render_supported_ops_markdown(
    official_expectations: list[OnnxFileExpectation],
    local_expectations: list[OnnxFileExpectation],
    local_repo_expectations: list[OnnxFileExpectation],
) -> str:
    supported_ops: set[str] = set()
    all_ops: set[str] = set()
    for expectation in (
        *official_expectations,
        *local_expectations,
        *local_repo_expectations,
    ):
        if not expectation.operators:
            continue
        all_ops.update(expectation.operators)
        if _is_success_message(expectation.error):
            supported_ops.update(expectation.operators)
    sorted_ops = sorted(all_ops)
    lines = [
        "# Supported operators",
        "",
        (
            "Operators are marked supported when they appear in an ONNX file "
            "with a successful verify result."
        ),
        "",
        f"Supported operators: {len(supported_ops)} / {len(sorted_ops)}",
        "",
        "| Operator | Supported |",
        "| --- | --- |",
    ]
    for op in sorted_ops:
        marker = "✅" if op in supported_ops else "❌"
        lines.append(f"| {op} | {marker} |")
    lines.append("")
    return "\n".join(lines)


@pytest.mark.order(
    after="tests/test_official_onnx_files.py::test_local_repo_onnx_expected_errors"
)
def test_official_onnx_file_support_doc() -> None:
    if not ONNX_VERSION_PATH.exists():
        _maybe_init_onnx_org()
    if not ONNX_VERSION_PATH.exists():
        pytest.skip(
            "onnx-org version metadata is unavailable. Initialize the onnx-org "
            "submodule and fetch its data files or set ONNX_ORG_AUTO_INIT=0 to skip auto-init."
        )
    official_test_directory = "onnx-org/onnx/backend/test/data"
    official_prefix = f"{official_test_directory}/"
    official_expectations: list[OnnxFileExpectation] = []
    for path in _official_onnx_file_paths():
        expectation = _load_expectation_for_repo_relative(path)
        relative_path = (
            path[len(official_prefix) :] if path.startswith(official_prefix) else path
        )
        official_expectations.append(
            OnnxFileExpectation(
                path=relative_path,
                error=expectation.error,
                command_line=expectation.command_line,
                operators=expectation.operators,
                opset_version=expectation.opset_version,
            )
        )

    repo_root = _repo_root()
    local_prefix = LOCAL_ONNX_DATA_ROOT.relative_to(
        repo_root
    ).as_posix()
    local_expectations: list[OnnxFileExpectation] = []
    for local_path in _local_onnx_file_paths():
        repo_relative = f"{local_prefix}/{local_path}"
        expectation = _load_expectation_for_repo_relative(repo_relative)
        local_expectations.append(
            OnnxFileExpectation(
                path=local_path,
                error=expectation.error,
                command_line=expectation.command_line,
                operators=expectation.operators,
                opset_version=expectation.opset_version,
            )
        )

    local_repo_prefix = LOCAL_REPO_ONNX_DATA_ROOT.relative_to(
        repo_root
    ).as_posix()
    local_repo_expectations: list[OnnxFileExpectation] = []
    for local_path in _local_repo_onnx_file_paths():
        repo_relative = f"{local_repo_prefix}/{local_path}"
        expectation = _load_expectation_for_repo_relative(repo_relative)
        local_repo_expectations.append(
            OnnxFileExpectation(
                path=local_path,
                error=expectation.error,
                command_line=expectation.command_line,
                operators=expectation.operators,
                opset_version=expectation.opset_version,
            )
        )
    expected_markdown = _render_onnx_file_support_markdown(
        official_expectations,
        local_expectations,
        local_repo_expectations,
    )
    expected_histogram = _render_support_histogram_markdown(
        official_expectations,
        local_expectations,
        local_repo_expectations,
    )
    expected_support_ops = _render_supported_ops_markdown(
        official_expectations,
        local_expectations,
        local_repo_expectations,
    )
    if os.getenv("UPDATE_REFS"):
        OFFICIAL_ONNX_FILE_SUPPORT_PATH.write_text(
            expected_markdown,
            encoding="utf-8",
        )
        OFFICIAL_ONNX_FILE_SUPPORT_HISTOGRAM_PATH.write_text(
            expected_histogram,
            encoding="utf-8",
        )
        SUPPORT_OPS_PATH.write_text(
            expected_support_ops,
            encoding="utf-8",
        )
        return
    actual_markdown = OFFICIAL_ONNX_FILE_SUPPORT_PATH.read_text(encoding="utf-8")
    actual_histogram = OFFICIAL_ONNX_FILE_SUPPORT_HISTOGRAM_PATH.read_text(
        encoding="utf-8"
    )
    actual_support_ops = SUPPORT_OPS_PATH.read_text(encoding="utf-8")
    assert actual_markdown == expected_markdown
    assert actual_histogram == expected_histogram
    assert actual_support_ops == expected_support_ops

from __future__ import annotations

import os
import re
from collections import Counter
from pathlib import Path

import pytest

from test_official_onnx_files import (
    LOCAL_ONNX_DATA_ROOT,
    OnnxFileExpectation,
    _load_expectation_for_repo_relative,
    _local_onnx_file_paths,
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
        lines.append(
            f"| {expectation.path} | {opset} | {supported} | {message} |"
        )
    return lines


def _render_onnx_file_support_markdown(
    official_expectations: list[OnnxFileExpectation],
    local_expectations: list[OnnxFileExpectation],
) -> str:
    supported_count = sum(
        1
        for expectation in official_expectations
        if _is_success_message(expectation.error)
    )
    total_count = len(official_expectations)
    onnx_version = ONNX_VERSION_PATH.read_text(encoding="utf-8").strip()
    local_supported = sum(
        1
        for expectation in local_expectations
        if _is_success_message(expectation.error)
    )
    local_total = len(local_expectations)
    lines = [
        "# Official ONNX file support",
        "",
        f"Support {supported_count} / {total_count} official ONNX files.",
        "",
        f"ONNX version: {onnx_version}",
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
        *_render_onnx_file_support_table(official_expectations),
        "",
        "## Local ONNX file support",
        "",
        "Local tests: `onnx2c-org/test/local_ops`.",
        "",
        f"Support {local_supported} / {local_total} local ONNX files.",
        "",
        *_render_onnx_file_support_table(local_expectations),
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
) -> str:
    official_histogram = _render_error_histogram_markdown(official_expectations)
    local_histogram = _render_error_histogram_markdown(
        local_expectations,
        title="### Error frequency",
    )
    return "\n".join(
        [
            official_histogram,
            "## Local ONNX file support histogram",
            "",
            local_histogram,
        ]
    ).strip() + "\n"


def _render_supported_ops_markdown(
    official_expectations: list[OnnxFileExpectation],
    local_expectations: list[OnnxFileExpectation],
) -> str:
    supported_ops: set[str] = set()
    all_ops: set[str] = set()
    for expectation in (*official_expectations, *local_expectations):
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
    after="tests/test_official_onnx_files.py::test_local_onnx_expected_errors"
)
def test_official_onnx_file_support_doc() -> None:
    if not ONNX_VERSION_PATH.exists():
        _maybe_init_onnx_org()
    if not ONNX_VERSION_PATH.exists():
        pytest.skip(
            "onnx-org version metadata is unavailable. Initialize the onnx-org "
            "submodule and fetch its data files or set ONNX_ORG_AUTO_INIT=0 to skip auto-init."
        )
    official_expectations = [
        _load_expectation_for_repo_relative(path)
        for path in _official_onnx_file_paths()
    ]
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
    expected_markdown = _render_onnx_file_support_markdown(
        official_expectations,
        local_expectations,
    )
    expected_histogram = _render_support_histogram_markdown(
        official_expectations,
        local_expectations,
    )
    expected_support_ops = _render_supported_ops_markdown(
        official_expectations,
        local_expectations,
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

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
    Path(__file__).resolve().parents[1] / "OFFICIAL_ONNX_FILE_SUPPORT.md"
)
OFFICIAL_ONNX_FILE_SUPPORT_HISTOGRAM_PATH = (
    Path(__file__).resolve().parents[1] / "OFFICIAL_ONNX_FILE_SUPPORT_HISTOGRAM.md"
)
ONNX_VERSION_PATH = Path(__file__).resolve().parents[1] / "onnx-org" / "VERSION_NUMBER"


def _is_success_message(message: str) -> bool:
    return message == "" or message.startswith("OK")


def _render_onnx_file_support_table(
    expectations: list[OnnxFileExpectation],
) -> list[str]:
    lines = [
        "| File | Supported | Error |",
        "| --- | --- | --- |",
    ]
    for expectation in expectations:
        supported = "✅" if _is_success_message(expectation.error) else "❌"
        message = expectation.error.replace("\n", " ").strip()
        lines.append(f"| {expectation.path} | {supported} | {message} |")
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
        "See [`OFFICIAL_ONNX_FILE_SUPPORT_HISTOGRAM.md`](OFFICIAL_ONNX_FILE_SUPPORT_HISTOGRAM.md) for the error histogram.",
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
    def _sanitize_error(error: str) -> str:
        return re.sub(r"'[^']*'", "'*'", error)

    errors = [
        _sanitize_error(expectation.error)
        for expectation in expectations
        if expectation.error and not _is_success_message(expectation.error)
    ]
    counts = Counter(errors)
    if not counts:
        return ""
    max_count = max(counts.values())
    bar_width = 30

    def bar(count: int) -> str:
        if max_count == 0:
            return ""
        length = max(1, round(count / max_count * bar_width))
        return "█" * length

    lines = [
        title,
        "",
        "| Error message | Count | Histogram |",
        "| --- | --- | --- |",
    ]
    for error, count in counts.most_common():
        lines.append(f"| {error} | {count} | {bar(count)} |")
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


@pytest.mark.order(after="test_official_onnx_expected_errors")
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
    if os.getenv("UPDATE_REFS"):
        OFFICIAL_ONNX_FILE_SUPPORT_PATH.write_text(
            expected_markdown,
            encoding="utf-8",
        )
        OFFICIAL_ONNX_FILE_SUPPORT_HISTOGRAM_PATH.write_text(
            expected_histogram,
            encoding="utf-8",
        )
        return
    actual_markdown = OFFICIAL_ONNX_FILE_SUPPORT_PATH.read_text(encoding="utf-8")
    actual_histogram = OFFICIAL_ONNX_FILE_SUPPORT_HISTOGRAM_PATH.read_text(
        encoding="utf-8"
    )
    assert actual_markdown == expected_markdown
    assert actual_histogram == expected_histogram

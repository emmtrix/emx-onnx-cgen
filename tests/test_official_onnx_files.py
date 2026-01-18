from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import onnx
import numpy as np
import pytest

from onnx import numpy_helper

from emx_onnx_cgen import cli
from emx_onnx_cgen.testbench import decode_testbench_array
from emx_onnx_cgen.verification import format_success_message, max_ulp_diff

EXPECTED_ERRORS_ROOT = Path(__file__).resolve().parent / "expected_errors"
OFFICIAL_ONNX_PREFIX = "onnx-org/onnx/backend/test/data/"
LOCAL_ONNX_PREFIX = "onnx2c-org/test/local_ops/"
OFFICIAL_ONNX_FILE_SUPPORT_PATH = (
    Path(__file__).resolve().parents[1] / "OFFICIAL_ONNX_FILE_SUPPORT.md"
)
OFFICIAL_ONNX_FILE_SUPPORT_HISTOGRAM_PATH = (
    Path(__file__).resolve().parents[1] / "OFFICIAL_ONNX_FILE_SUPPORT_HISTOGRAM.md"
)
ONNX_VERSION_PATH = Path(__file__).resolve().parents[1] / "onnx-org" / "VERSION_NUMBER"
LOCAL_ONNX_DATA_ROOT = (
    Path(__file__).resolve().parents[1] / "onnx2c-org" / "test" / "local_ops"
)
ONNX_FILE_LIMIT = 100


@dataclass(frozen=True)
class OnnxFileExpectation:
    path: str
    error: str
    command_line: str = ""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _official_data_root() -> Path:
    return _repo_root() / "onnx-org" / "onnx" / "backend" / "test" / "data"


def _normalize_official_path(path: str) -> str:
    repo_root = _repo_root()
    candidate = repo_root / path
    if candidate.exists():
        return candidate.relative_to(repo_root).as_posix()
    return (_official_data_root() / path).relative_to(repo_root).as_posix()


def _load_official_onnx_file_expectations() -> list[OnnxFileExpectation]:
    return list(_OFFICIAL_ONNX_FILE_EXPECTATIONS)[:ONNX_FILE_LIMIT]


def _load_local_onnx_file_expectations() -> list[OnnxFileExpectation]:
    return list(_LOCAL_ONNX_FILE_EXPECTATIONS)[:ONNX_FILE_LIMIT]


def _official_onnx_file_paths() -> list[str]:
    return [
        _normalize_official_path(expectation.path)
        for expectation in _load_official_onnx_file_expectations()
    ]


def _encode_repo_relative_path(repo_relative_path: str) -> str:
    return repo_relative_path.replace("/", "__")


def _decode_repo_relative_path(encoded: str) -> str:
    return encoded.replace("__", "/")


def _expected_errors_path_for_repo_relative(repo_relative_path: str) -> Path:
    encoded = _encode_repo_relative_path(repo_relative_path)
    return EXPECTED_ERRORS_ROOT / f"{encoded}.json"


def _repo_relative_path_from_expectation_file(path: Path) -> str:
    encoded = path.relative_to(EXPECTED_ERRORS_ROOT).with_suffix("").as_posix()
    return _decode_repo_relative_path(encoded)


def _read_expectation_file(
    path: Path,
    *,
    fallback_path: str,
) -> OnnxFileExpectation:
    data = json.loads(path.read_text(encoding="utf-8"))
    error = ""
    command_line = ""
    if isinstance(data, dict):
        error = data.get("error", "")
        command_line = data.get("command_line", "")
    elif isinstance(data, list):
        if data and isinstance(data[0], str) and data[0].endswith(".onnx"):
            if len(data) >= 2:
                error = data[1]
            if len(data) >= 3:
                command_line = data[2]
        else:
            if len(data) >= 1:
                error = data[0]
            if len(data) >= 2:
                command_line = data[1]
    else:
        raise TypeError(f"Unsupported expectation data in {path}")
    return OnnxFileExpectation(
        path=fallback_path,
        error=error,
        command_line=command_line,
    )


def _set_official_onnx_file_expectations(
    expectations: list[OnnxFileExpectation],
) -> None:
    global _OFFICIAL_ONNX_FILE_EXPECTATIONS
    _OFFICIAL_ONNX_FILE_EXPECTATIONS = expectations


def _set_local_onnx_file_expectations(
    expectations: list[OnnxFileExpectation],
) -> None:
    global _LOCAL_ONNX_FILE_EXPECTATIONS
    _LOCAL_ONNX_FILE_EXPECTATIONS = expectations


def _write_expectation_file(
    expectation: OnnxFileExpectation,
    *,
    repo_relative_path: str,
) -> None:
    expectation_path = _expected_errors_path_for_repo_relative(
        repo_relative_path
    )
    expectation_path.parent.mkdir(parents=True, exist_ok=True)
    expectation_path.write_text(
        json.dumps(
            {
                "error": expectation.error,
                "command_line": expectation.command_line,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _load_expectations_from_root(
    root: Path,
    *,
    path_converter: Callable[[str], str],
    path_filter: Callable[[str], bool],
) -> list[OnnxFileExpectation]:
    if not root.exists():
        raise AssertionError(
            f"Expected errors directory {root} is missing."
        )
    expectations: list[OnnxFileExpectation] = []
    for expectation_file in sorted(root.glob("*.json")):
        repo_relative = _repo_relative_path_from_expectation_file(
            expectation_file
        )
        if not path_filter(repo_relative):
            continue
        expectation_path = path_converter(repo_relative)
        expectations.append(
            _read_expectation_file(
                expectation_file,
                fallback_path=expectation_path,
            )
        )
    return expectations


_OFFICIAL_ONNX_FILE_EXPECTATIONS = _load_expectations_from_root(
    EXPECTED_ERRORS_ROOT,
    path_converter=lambda repo_relative: repo_relative,
    path_filter=lambda repo_relative: repo_relative.startswith(OFFICIAL_ONNX_PREFIX),
)
_LOCAL_ONNX_FILE_EXPECTATIONS = _load_expectations_from_root(
    EXPECTED_ERRORS_ROOT,
    path_converter=lambda repo_relative: Path(repo_relative)
    .relative_to(LOCAL_ONNX_DATA_ROOT.relative_to(_repo_root()))
    .as_posix(),
    path_filter=lambda repo_relative: repo_relative.startswith(LOCAL_ONNX_PREFIX),
)


def _is_success_message(message: str) -> bool:
    return message == "" or message.startswith("OK")


def _format_missing_test_data_message() -> str:
    return "OK (max ULP 0; testbench unavailable)"


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


def _collect_onnx_files(data_root: Path) -> list[str]:
    return sorted(
        p.relative_to(data_root).as_posix()
        for p in data_root.rglob("*.onnx")
    )[:ONNX_FILE_LIMIT]


def _maybe_init_onnx_org() -> None:
    auto_init = os.getenv("ONNX_ORG_AUTO_INIT", "1").strip().lower()
    if auto_init in {"0", "false", "no", "off"}:
        return
    repo_root = Path(__file__).resolve().parents[1]
    if shutil.which("git") is None:
        return
    subprocess.run(
        ["git", "submodule", "update", "--init", "--recursive", "onnx-org"],
        cwd=repo_root,
        check=False,
    )
    lfs_probe = subprocess.run(
        ["git", "lfs", "version"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if lfs_probe.returncode != 0:
        return
    subprocess.run(
        ["git", "lfs", "pull", "--include", "onnx/backend/test/data/**"],
        cwd=repo_root / "onnx-org",
        check=False,
    )


def _maybe_init_onnx2c_org() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if shutil.which("git") is None:
        return
    subprocess.run(
        ["git", "submodule", "update", "--init", "--recursive", "onnx2c-org"],
        cwd=repo_root,
        check=False,
    )


def _ensure_official_onnx_files_present(data_root: Path) -> None:
    if not data_root.exists():
        _maybe_init_onnx_org()
    if not data_root.exists():
        pytest.skip(
            "onnx-org test data is unavailable. Initialize the onnx-org submodule "
            "and fetch its data files or set ONNX_ORG_AUTO_INIT=0 to skip auto-init."
        )
    missing = [
        path
        for path in _official_onnx_file_paths()
        if not (_repo_root() / path).exists()
    ]
    if missing:
        _maybe_init_onnx_org()
        missing = [
            path
            for path in _official_onnx_file_paths()
            if not (_repo_root() / path).exists()
        ]
        if not missing:
            return
        preview = ", ".join(missing[:5])
        suffix = "..." if len(missing) > 5 else ""
        pytest.skip(
            "onnx-org test data is incomplete; missing files include: "
            f"{preview}{suffix}. Initialize the submodule and fetch any LFS data or "
            "set ONNX_ORG_AUTO_INIT=0 to skip auto-init."
        )


def _ensure_local_onnx_files_present(data_root: Path) -> None:
    if not data_root.exists():
        _maybe_init_onnx2c_org()
    if not data_root.exists():
        pytest.skip(
            "onnx2c-org local test data is unavailable. Initialize the onnx2c-org "
            "submodule to run local operator tests."
        )


def _resolve_compiler() -> list[str] | None:
    compiler = os.environ.get("CC")
    if compiler:
        return [compiler]
    for candidate in ("cc", "gcc", "clang"):
        resolved = shutil.which(candidate)
        if resolved:
            return [resolved]
    return None


def _load_test_data_set(
    model: onnx.ModelProto,
    data_dir: Path,
    *,
    allow_missing: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]] | None:
    if not data_dir.exists():
        if allow_missing:
            return None
        pytest.skip(
            f"Missing test data directory {data_dir}. Ensure LFS data is available."
        )
    input_files = sorted(
        data_dir.glob("input_*.pb"),
        key=lambda path: int(path.stem.split("_")[-1]),
    )
    output_files = sorted(
        data_dir.glob("output_*.pb"),
        key=lambda path: int(path.stem.split("_")[-1]),
    )
    if not input_files or not output_files:
        if allow_missing:
            return None
        pytest.skip(
            f"Missing test data files in {data_dir}. Ensure LFS data is available."
        )
    if len(input_files) != len(model.graph.input):
        raise AssertionError(
            "Test data input count does not match model inputs: "
            f"{len(input_files)} vs {len(model.graph.input)}."
        )
    if len(output_files) != len(model.graph.output):
        raise AssertionError(
            "Test data output count does not match model outputs: "
            f"{len(output_files)} vs {len(model.graph.output)}."
        )
    inputs: dict[str, np.ndarray] = {}
    for index, path in enumerate(input_files):
        tensor = onnx.TensorProto()
        tensor.ParseFromString(path.read_bytes())
        inputs[model.graph.input[index].name] = numpy_helper.to_array(tensor)
    outputs: dict[str, np.ndarray] = {}
    for index, path in enumerate(output_files):
        tensor = onnx.TensorProto()
        tensor.ParseFromString(path.read_bytes())
        outputs[model.graph.output[index].name] = numpy_helper.to_array(tensor)
    return inputs, outputs


def _compile_and_run_testbench(
    model_path: Path,
    testbench_inputs: dict[str, np.ndarray],
) -> dict[str, object]:
    compiler_cmd = _resolve_compiler()
    if compiler_cmd is None:
        pytest.skip("C compiler not available (set CC or install gcc/clang)")
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        c_path = temp_path / "model.c"
        exe_path = temp_path / "model"
        repo_root = _repo_root()
        result = cli.run_cli_command(
            [
                "compile",
                str(model_path.relative_to(repo_root)),
                "--template-dir",
                "templates",
                "--emit-testbench",
            ],
            testbench_inputs=testbench_inputs,
        )
        if result.exit_code != 0:
            raise AssertionError(
                f"CLI compile failed for {model_path}: {result.error}"
            )
        generated = result.generated or ""
        c_path.write_text(generated, encoding="utf-8")
        subprocess.run(
            [
                *compiler_cmd,
                "-std=c99",
                "-O2",
                str(c_path),
                "-o",
                str(exe_path),
                "-lm",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        result = subprocess.run(
            [str(exe_path)],
            check=True,
            capture_output=True,
            text=True,
        )
    return json.loads(result.stdout)


def _assert_outputs_match(
    payload: dict[str, object],
    expected_outputs: dict[str, np.ndarray],
    *,
    require_allclose: bool = True,
) -> int:
    max_ulp = 0
    outputs = payload.get("outputs", {})
    for name, expected in expected_outputs.items():
        output_payload = outputs.get(name)
        if output_payload is None:
            raise AssertionError(f"Missing output {name} in testbench data")
        output_data = decode_testbench_array(output_payload["data"], expected.dtype)
        output_data = output_data.reshape(expected.shape)
        if np.issubdtype(expected.dtype, np.floating):
            if require_allclose:
                np.testing.assert_allclose(
                    output_data, expected, rtol=1e-4, atol=1e-5
                )
            max_ulp = max(max_ulp, max_ulp_diff(output_data, expected))
        else:
            if require_allclose:
                np.testing.assert_array_equal(output_data, expected)
    return max_ulp


def _errors_match(actual_error: str, expected_error: str) -> bool:
    if expected_error.startswith("OK"):
        return actual_error.startswith("OK")
    return actual_error == expected_error


def test_official_onnx_files() -> None:
    data_root = _official_data_root()
    _ensure_official_onnx_files_present(data_root)
    actual_files = _collect_onnx_files(data_root)
    repo_root = _repo_root()
    data_root_relative = data_root.relative_to(repo_root).as_posix()
    actual_files = [
        f"{data_root_relative}/{path}" for path in actual_files
    ]
    expected_files = sorted(_official_onnx_file_paths())
    actual_set = set(actual_files)
    expected_set = set(expected_files)
    missing = sorted(expected_set - actual_set)
    extra = sorted(actual_set - expected_set)
    assert not missing and not extra, (
        "Official ONNX file list mismatch. "
        f"Missing: {missing}. Extra: {extra}."
    )


def test_local_onnx_files() -> None:
    data_root = LOCAL_ONNX_DATA_ROOT
    _ensure_local_onnx_files_present(data_root)
    actual_files = _collect_onnx_files(data_root)
    expectations = _load_local_onnx_file_expectations()
    expected_files = sorted(expectation.path for expectation in expectations)
    actual_set = set(actual_files)
    expected_set = set(expected_files)
    missing = sorted(expected_set - actual_set)
    extra = sorted(actual_set - expected_set)
    assert not missing and not extra, (
        "Local ONNX file list mismatch. "
        f"Missing: {missing}. Extra: {extra}."
    )


@pytest.mark.order(1)
def test_official_onnx_expected_errors() -> None:
    data_root = _official_data_root()
    _ensure_official_onnx_files_present(data_root)
    expectations = _load_official_onnx_file_expectations()
    actual_expectations: list[OnnxFileExpectation] = []
    repo_root = _repo_root()
    compiler_cmd = _resolve_compiler()
    if compiler_cmd is None:
        pytest.skip("C compiler not available (set CC or install gcc/clang)")
    for expectation in expectations:
        rel_path = _normalize_official_path(expectation.path)
        expected_error = expectation.error
        model_path = repo_root / rel_path
        cli_result = cli.run_cli_command(
            [
                "emx-onnx-cgen",
                "verify",
                str(model_path.relative_to(repo_root)),
                "--template-dir",
                "templates",
                "--cc",
                compiler_cmd[0],
            ]
        )
        if cli_result.exit_code != 0:
            actual_error = cli_result.error or ""
        else:
            if os.getenv("UPDATE_REFS"):
                actual_error = (
                    cli_result.success_message
                    or _format_missing_test_data_message()
                )
            elif expected_error.startswith("OK"):
                actual_error = "OK"
            else:
                actual_error = ""
        actual_expectations.append(
            OnnxFileExpectation(
                path=rel_path,
                error=actual_error,
                command_line=cli_result.command_line,
            )
        )
        if os.getenv("UPDATE_REFS"):
            continue
        assert _errors_match(actual_error, expected_error), (
            f"Unexpected result for {rel_path}. Expected: {expected_error!r}. "
            f"Got: {actual_error!r}."
        )
    if os.getenv("UPDATE_REFS"):
        for item in actual_expectations:
            _write_expectation_file(
                item,
                repo_relative_path=item.path,
            )
        _set_official_onnx_file_expectations(actual_expectations)
        return


@pytest.mark.order(2)
def test_local_onnx_expected_errors() -> None:
    data_root = LOCAL_ONNX_DATA_ROOT
    _ensure_local_onnx_files_present(data_root)
    expectations = _load_local_onnx_file_expectations()
    expected_paths = [expectation.path for expectation in expectations]
    actual_paths = _collect_onnx_files(data_root)
    assert expected_paths == actual_paths
    actual_expectations: list[OnnxFileExpectation] = []
    repo_root = _repo_root()
    compiler_cmd = _resolve_compiler()
    if compiler_cmd is None:
        pytest.skip("C compiler not available (set CC or install gcc/clang)")
    for expectation in expectations:
        rel_path = expectation.path
        expected_error = expectation.error
        model_path = data_root / rel_path
        model = onnx.load_model(model_path)
        cli_result = cli.run_cli_command(
            [
                "emx-onnx-cgen",
                "verify",
                str(model_path.relative_to(repo_root)),
                "--template-dir",
                "templates",
                "--cc",
                compiler_cmd[0],
            ]
        )
        if cli_result.exit_code != 0:
            actual_error = cli_result.error or ""
        else:
            if os.getenv("UPDATE_REFS"):
                test_data_dir = model_path.parent / "test_data_set_0"
                test_data = _load_test_data_set(
                    model,
                    test_data_dir,
                    allow_missing=True,
                )
                if test_data is None:
                    actual_error = _format_missing_test_data_message()
                else:
                    inputs, expected_outputs = test_data
                    payload = _compile_and_run_testbench(model_path, inputs)
                    max_ulp = _assert_outputs_match(payload, expected_outputs)
                    actual_error = format_success_message(max_ulp)
            elif expected_error.startswith("OK"):
                actual_error = "OK"
            else:
                actual_error = ""
        actual_expectations.append(
            OnnxFileExpectation(
                path=rel_path,
                error=actual_error,
                command_line=cli_result.command_line,
            )
        )
        if os.getenv("UPDATE_REFS"):
            continue
        assert _errors_match(actual_error, expected_error), (
            f"Unexpected result for {rel_path}. Expected: {expected_error!r}. "
            f"Got: {actual_error!r}."
        )
    if os.getenv("UPDATE_REFS"):
        repo_root = _repo_root()
        for item in actual_expectations:
            repo_relative_path = (
                LOCAL_ONNX_DATA_ROOT / item.path
            ).relative_to(repo_root).as_posix()
            _write_expectation_file(
                item,
                repo_relative_path=repo_relative_path,
            )
        _set_local_onnx_file_expectations(actual_expectations)
        return


@pytest.mark.order(after="test_local_onnx_expected_errors")
def test_official_onnx_test_data_matches_testbench() -> None:
    data_root = _official_data_root()
    _ensure_official_onnx_files_present(data_root)
    expectations = _load_official_onnx_file_expectations()
    repo_root = _repo_root()
    for expectation in expectations:
        if not _is_success_message(expectation.error):
            continue
        model_path = repo_root / expectation.path
        model = onnx.load_model(model_path)
        test_data_dir = model_path.parent / "test_data_set_0"
        inputs, expected_outputs = _load_test_data_set(model, test_data_dir)
        payload = _compile_and_run_testbench(model_path, inputs)
        _assert_outputs_match(payload, expected_outputs)


@pytest.mark.order(after="test_official_onnx_test_data_matches_testbench")
def test_local_onnx_test_data_matches_testbench() -> None:
    data_root = LOCAL_ONNX_DATA_ROOT
    _ensure_local_onnx_files_present(data_root)
    expectations = _load_local_onnx_file_expectations()
    for expectation in expectations:
        if not _is_success_message(expectation.error):
            continue
        model_path = data_root / expectation.path
        model = onnx.load_model(model_path)
        test_data_dir = model_path.parent / "test_data_set_0"
        inputs, expected_outputs = _load_test_data_set(model, test_data_dir)
        payload = _compile_and_run_testbench(model_path, inputs)
        _assert_outputs_match(payload, expected_outputs)


@pytest.mark.order(after="test_official_onnx_expected_errors")
def test_official_onnx_file_support_doc() -> None:
    if not ONNX_VERSION_PATH.exists():
        _maybe_init_onnx_org()
    if not ONNX_VERSION_PATH.exists():
        pytest.skip(
            "onnx-org version metadata is unavailable. Initialize the onnx-org "
            "submodule and fetch its data files or set ONNX_ORG_AUTO_INIT=0 to skip auto-init."
        )
    official_expectations = _load_official_onnx_file_expectations()
    local_expectations = _load_local_onnx_file_expectations()
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
    actual_histogram = OFFICIAL_ONNX_FILE_SUPPORT_HISTOGRAM_PATH.read_text(encoding="utf-8")
    assert actual_markdown == expected_markdown
    assert actual_histogram == expected_histogram

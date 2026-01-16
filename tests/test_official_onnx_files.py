from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

import onnx
import numpy as np
import pytest

from onnx import numpy_helper

from emx_onnx_cgen.compiler import Compiler, CompilerOptions
from emx_onnx_cgen.testbench import decode_testbench_array

OFFICIAL_ONNX_FILE_EXPECTATIONS_PATH = (
    Path(__file__).resolve().parent / "official_onnx_expected_errors.json"
)
LOCAL_ONNX_FILE_EXPECTATIONS_PATH = (
    Path(__file__).resolve().parent / "local_onnx_expected_errors.json"
)
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


def _load_official_onnx_file_expectations() -> list[tuple[str, str]]:
    return list(_OFFICIAL_ONNX_FILE_EXPECTATIONS)


def _load_local_onnx_file_expectations() -> list[tuple[str, str]]:
    return list(_LOCAL_ONNX_FILE_EXPECTATIONS)


def _official_onnx_file_paths() -> list[str]:
    return [path for path, _ in _load_official_onnx_file_expectations()]


def _read_onnx_file_expectations(path: Path) -> list[tuple[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [(path, error) for path, error in data]


def _set_official_onnx_file_expectations(expectations: list[tuple[str, str]]) -> None:
    global _OFFICIAL_ONNX_FILE_EXPECTATIONS
    _OFFICIAL_ONNX_FILE_EXPECTATIONS = expectations


def _set_local_onnx_file_expectations(expectations: list[tuple[str, str]]) -> None:
    global _LOCAL_ONNX_FILE_EXPECTATIONS
    _LOCAL_ONNX_FILE_EXPECTATIONS = expectations


_OFFICIAL_ONNX_FILE_EXPECTATIONS = _read_onnx_file_expectations(
    OFFICIAL_ONNX_FILE_EXPECTATIONS_PATH
)
_LOCAL_ONNX_FILE_EXPECTATIONS = _read_onnx_file_expectations(
    LOCAL_ONNX_FILE_EXPECTATIONS_PATH
)


def _render_onnx_file_support_table(expectations: list[tuple[str, str]]) -> list[str]:
    lines = [
        "| File | Supported | Error |",
        "| --- | --- | --- |",
    ]
    for path, error in expectations:
        supported = "✅" if not error else "❌"
        message = error.replace("\n", " ").strip()
        lines.append(f"| {path} | {supported} | {message} |")
    return lines


def _render_onnx_file_support_markdown(
    official_expectations: list[tuple[str, str]],
    local_expectations: list[tuple[str, str]],
) -> str:
    supported_count = sum(1 for _, error in official_expectations if not error)
    total_count = len(official_expectations)
    onnx_version = ONNX_VERSION_PATH.read_text(encoding="utf-8").strip()
    local_supported = sum(1 for _, error in local_expectations if not error)
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
    expectations: list[tuple[str, str]],
    title: str = "# Error frequency",
) -> str:
    def _sanitize_error(error: str) -> str:
        return re.sub(r"'[^']*'", "'*'", error)

    errors = [_sanitize_error(error) for _, error in expectations if error]
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
    official_expectations: list[tuple[str, str]],
    local_expectations: list[tuple[str, str]],
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
    return sorted(p.relative_to(data_root).as_posix() for p in data_root.rglob("*.onnx"))


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
    missing = [path for path in _official_onnx_file_paths() if not (data_root / path).exists()]
    if missing:
        _maybe_init_onnx_org()
        missing = [path for path in _official_onnx_file_paths() if not (data_root / path).exists()]
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
    model: onnx.ModelProto, data_dir: Path
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    if not data_dir.exists():
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
    model: onnx.ModelProto,
    testbench_inputs: dict[str, np.ndarray],
) -> dict[str, object]:
    compiler_cmd = _resolve_compiler()
    if compiler_cmd is None:
        pytest.skip("C compiler not available (set CC or install gcc/clang)")
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        c_path = temp_path / "model.c"
        exe_path = temp_path / "model"
        options = CompilerOptions(
            template_dir=Path(__file__).resolve().parents[1] / "templates",
            emit_testbench=True,
            testbench_inputs=testbench_inputs,
        )
        compiler = Compiler(options)
        generated = compiler.compile(model)
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
    payload: dict[str, object], expected_outputs: dict[str, np.ndarray]
) -> None:
    outputs = payload.get("outputs", {})
    for name, expected in expected_outputs.items():
        output_payload = outputs.get(name)
        if output_payload is None:
            raise AssertionError(f"Missing output {name} in testbench data")
        output_data = decode_testbench_array(output_payload["data"], expected.dtype)
        output_data = output_data.reshape(expected.shape)
        if np.issubdtype(expected.dtype, np.floating):
            np.testing.assert_allclose(output_data, expected, rtol=1e-4, atol=1e-5)
        else:
            np.testing.assert_array_equal(output_data, expected)


def test_official_onnx_files() -> None:
    data_root = Path(__file__).resolve().parents[1] / "onnx-org" / "onnx" / "backend" / "test" / "data"
    _ensure_official_onnx_files_present(data_root)
    actual_files = _collect_onnx_files(data_root)
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
    expected_files = sorted(path for path, _ in expectations)
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
    data_root = Path(__file__).resolve().parents[1] / "onnx-org" / "onnx" / "backend" / "test" / "data"
    _ensure_official_onnx_files_present(data_root)
    expectations = _load_official_onnx_file_expectations()
    compiler = Compiler()
    actual_expectations: list[tuple[str, str]] = []
    for rel_path, expected_error in expectations:
        model_path = data_root / rel_path
        model = onnx.load_model(model_path)
        try:
            compiler.compile(model)
        except Exception as exc:
            actual_error = str(exc)
        else:
            actual_error = ""
        actual_expectations.append((rel_path, actual_error))
        if os.getenv("UPDATE_REFS"):
            continue
        assert actual_error == expected_error, (
            f"Unexpected result for {rel_path}. Expected: {expected_error!r}. "
            f"Got: {actual_error!r}."
        )
    if os.getenv("UPDATE_REFS"):
        OFFICIAL_ONNX_FILE_EXPECTATIONS_PATH.write_text(
            json.dumps(actual_expectations, indent=2) + "\n",
            encoding="utf-8",
        )
        _set_official_onnx_file_expectations(actual_expectations)
        return


@pytest.mark.order(2)
def test_local_onnx_expected_errors() -> None:
    data_root = LOCAL_ONNX_DATA_ROOT
    _ensure_local_onnx_files_present(data_root)
    expectations = _load_local_onnx_file_expectations()
    expected_paths = [path for path, _ in expectations]
    actual_paths = _collect_onnx_files(data_root)
    assert expected_paths == actual_paths
    compiler = Compiler()
    actual_expectations: list[tuple[str, str]] = []
    for rel_path, expected_error in expectations:
        model_path = data_root / rel_path
        model = onnx.load_model(model_path)
        try:
            compiler.compile(model)
        except Exception as exc:
            actual_error = str(exc)
        else:
            actual_error = ""
        actual_expectations.append((rel_path, actual_error))
        if os.getenv("UPDATE_REFS"):
            continue
        assert actual_error == expected_error, (
            f"Unexpected result for {rel_path}. Expected: {expected_error!r}. "
            f"Got: {actual_error!r}."
        )
    if os.getenv("UPDATE_REFS"):
        LOCAL_ONNX_FILE_EXPECTATIONS_PATH.write_text(
            json.dumps(actual_expectations, indent=2) + "\n",
            encoding="utf-8",
        )
        _set_local_onnx_file_expectations(actual_expectations)
        return


@pytest.mark.order(after="test_local_onnx_expected_errors")
def test_official_onnx_test_data_matches_testbench() -> None:
    data_root = Path(__file__).resolve().parents[1] / "onnx-org" / "onnx" / "backend" / "test" / "data"
    _ensure_official_onnx_files_present(data_root)
    expectations = _load_official_onnx_file_expectations()
    for rel_path, expected_error in expectations:
        if expected_error:
            continue
        model_path = data_root / rel_path
        model = onnx.load_model(model_path)
        test_data_dir = model_path.parent / "test_data_set_0"
        inputs, expected_outputs = _load_test_data_set(model, test_data_dir)
        payload = _compile_and_run_testbench(model, inputs)
        _assert_outputs_match(payload, expected_outputs)


@pytest.mark.order(after="test_official_onnx_test_data_matches_testbench")
def test_local_onnx_test_data_matches_testbench() -> None:
    data_root = LOCAL_ONNX_DATA_ROOT
    _ensure_local_onnx_files_present(data_root)
    expectations = _load_local_onnx_file_expectations()
    for rel_path, expected_error in expectations:
        if expected_error:
            continue
        model_path = data_root / rel_path
        model = onnx.load_model(model_path)
        test_data_dir = model_path.parent / "test_data_set_0"
        inputs, expected_outputs = _load_test_data_set(model, test_data_dir)
        payload = _compile_and_run_testbench(model, inputs)
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

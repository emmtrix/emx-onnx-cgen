from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Callable

import pytest

from emx_onnx_cgen import cli

EXPECTED_ERRORS_ROOT = Path(__file__).resolve().parent / "expected_errors"
OFFICIAL_ONNX_PREFIX = "onnx-org/onnx/backend/test/data/"
LOCAL_ONNX_PREFIX = "onnx2c-org/test/local_ops/"
LOCAL_ONNX_DATA_ROOT = (
    Path(__file__).resolve().parents[1] / "onnx2c-org" / "test" / "local_ops"
)
ONNX_FILE_LIMIT = 5000
_VERBOSE_FLAGS_REPORTED = False
MODEL_EXTRA_VERIFY_ARGS = {
    "onnx-org/onnx/backend/test/data/node/test_nllloss_NCd1d2d3d4d5_mean_weight/model.onnx": (
        "--fp32-accumulation-strategy",
        "fp64",
    ),
    "onnx-org/onnx/backend/test/data/node/test_sce_NCd1d2d3d4d5_mean_weight/model.onnx": (
        "--fp32-accumulation-strategy",
        "fp64",
    ),
    "onnx-org/onnx/backend/test/data/node/test_sce_NCd1d2d3d4d5_mean_weight_expanded/model.onnx": (
        "--fp32-accumulation-strategy",
        "fp64",
    ),
    "onnx-org/onnx/backend/test/data/node/test_sce_NCd1d2d3d4d5_mean_weight_log_prob/model.onnx": (
        "--fp32-accumulation-strategy",
        "fp64",
    ),
    "onnx-org/onnx/backend/test/data/node/test_sce_NCd1d2d3d4d5_mean_weight_log_prob_expanded/model.onnx": (
        "--fp32-accumulation-strategy",
        "fp64",
    ),
    "onnx-org/onnx/backend/test/data/pytorch-converted/test_Conv3d_dilated_strided/model.onnx": (
        "--fp32-accumulation-strategy",
        "fp64",
    ),
}


@dataclass(frozen=True)
class OnnxFileExpectation:
    path: str
    error: str
    command_line: str = ""
    operators: list[str] | None = None
    opset_version: int | None = None
    generated_checksum: str | None = None


@cache
def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@cache
def _official_data_root() -> Path:
    return _repo_root() / "onnx-org" / "onnx" / "backend" / "test" / "data"


@cache
def _official_onnx_repo_paths_set() -> frozenset[str]:
    return frozenset(_official_onnx_file_paths())


@cache
def _missing_official_onnx_paths() -> tuple[str, ...]:
    repo_root = _repo_root()
    missing = [
        path
        for path in _official_onnx_repo_paths_set()
        if not (repo_root / path).exists()
    ]
    return tuple(sorted(missing))


def _normalize_official_path(path: str) -> str:
    if path.startswith(OFFICIAL_ONNX_PREFIX):
        return path
    return f"{OFFICIAL_ONNX_PREFIX}{path}"


def _list_expectation_repo_paths(
    root: Path,
    *,
    path_filter: Callable[[str], bool],
) -> list[str]:
    if not root.exists():
        raise AssertionError(
            f"Expected errors directory {root} is missing."
        )
    repo_relative_paths: list[str] = []
    for expectation_file in sorted(root.glob("*.json")):
        repo_relative = _repo_relative_path_from_expectation_file(
            expectation_file
        )
        if not path_filter(repo_relative):
            continue
        repo_relative_paths.append(repo_relative)
    return repo_relative_paths[:ONNX_FILE_LIMIT]


@cache
def _official_onnx_file_paths() -> tuple[str, ...]:
    if os.getenv("UPDATE_REFS"):
        return tuple(
            _normalize_official_path(path)
            for path in _collect_onnx_files(_official_data_root())
        )
    return tuple(
        _normalize_official_path(path)
        for path in _list_expectation_repo_paths(
            EXPECTED_ERRORS_ROOT,
            path_filter=lambda repo_relative: repo_relative.startswith(
                OFFICIAL_ONNX_PREFIX
            ),
        )
    )


@cache
def _local_onnx_file_paths() -> tuple[str, ...]:
    if os.getenv("UPDATE_REFS"):
        return tuple(_collect_onnx_files(LOCAL_ONNX_DATA_ROOT))
    repo_relative_prefix = LOCAL_ONNX_DATA_ROOT.relative_to(
        _repo_root()
    ).as_posix()
    return tuple(
        Path(path).relative_to(repo_relative_prefix).as_posix()
        for path in _list_expectation_repo_paths(
            EXPECTED_ERRORS_ROOT,
            path_filter=lambda repo_relative: repo_relative.startswith(
                LOCAL_ONNX_PREFIX
            ),
        )
    )


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
    operators: list[str] | None = None
    opset_version: int | None = None
    generated_checksum: str | None = None
    if isinstance(data, dict):
        error = data.get("error", "")
        command_line = data.get("command_line", "")
        operators = data.get("operators")
        opset_version = data.get("opset_version")
        generated_checksum = data.get("generated_checksum")
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
        operators=operators,
        opset_version=opset_version,
        generated_checksum=generated_checksum,
    )


def _load_expectation_for_repo_relative(
    repo_relative_path: str,
) -> OnnxFileExpectation:
    expectation_path = _expected_errors_path_for_repo_relative(
        repo_relative_path
    )
    if not expectation_path.exists():
        if os.getenv("UPDATE_REFS"):
            return OnnxFileExpectation(
                path=repo_relative_path,
                error="",
                command_line="",
            )
        raise AssertionError(
            f"Missing expectation file for {repo_relative_path}"
        )
    return _read_expectation_file(
        expectation_path,
        fallback_path=repo_relative_path,
    )


_load_expectation_for_repo_relative = cache(_load_expectation_for_repo_relative)


def _write_expectation_file(
    expectation: OnnxFileExpectation,
    *,
    repo_relative_path: str,
) -> None:
    expectation_path = _expected_errors_path_for_repo_relative(
        repo_relative_path
    )
    expectation_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "error": expectation.error,
        "command_line": expectation.command_line,
    }
    if expectation.operators is not None:
        payload["operators"] = expectation.operators
    if expectation.opset_version is not None:
        payload["opset_version"] = expectation.opset_version
    if expectation.generated_checksum is not None:
        payload["generated_checksum"] = expectation.generated_checksum
    expectation_path.write_text(
        json.dumps(
            payload,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


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
        _missing_official_onnx_paths.cache_clear()
    if not data_root.exists():
        pytest.skip(
            "onnx-org test data is unavailable. Initialize the onnx-org submodule "
            "and fetch its data files or set ONNX_ORG_AUTO_INIT=0 to skip auto-init."
        )
    missing = _missing_official_onnx_paths()
    if missing:
        _maybe_init_onnx_org()
        _missing_official_onnx_paths.cache_clear()
        missing = _missing_official_onnx_paths()
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


def _find_test_data_dir(model_path: Path) -> Path | None:
    test_data_dir = model_path.parent / "test_data_set_0"
    if not test_data_dir.exists():
        return None
    if not list(test_data_dir.glob("input_*.pb")):
        return None
    return test_data_dir


_find_test_data_dir = cache(_find_test_data_dir)


@pytest.fixture(scope="module")
def official_repo_root_and_data_root() -> tuple[Path, Path]:
    data_root = _official_data_root()
    _ensure_official_onnx_files_present(data_root)
    return _repo_root(), data_root


@pytest.fixture(scope="module")
def local_repo_root_and_data_root() -> tuple[Path, Path]:
    data_root = LOCAL_ONNX_DATA_ROOT
    _ensure_local_onnx_files_present(data_root)
    return _repo_root(), data_root


def _errors_match(actual_error: str, expected_error: str) -> bool:
    return actual_error == expected_error


def _skip_expected_checksum() -> bool:
    value = os.getenv("DISABLE_CHECKSUM", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _update_refs_mode() -> int:
    value = os.getenv("UPDATE_REFS", "").strip()
    if not value:
        return 0
    if value.isdigit():
        return int(value)
    return 1


def _is_failure_expectation(expectation: OnnxFileExpectation) -> bool:
    if not expectation.error:
        return False
    return not expectation.error.startswith("OK")


def _should_use_expected_checksum(expectation: OnnxFileExpectation) -> bool:
    if expectation.generated_checksum is None:
        return False
    if _skip_expected_checksum():
        return False
    update_refs_mode = _update_refs_mode()
    if update_refs_mode >= 3:
        return False
    if update_refs_mode == 2 and _is_failure_expectation(expectation):
        return False
    return True


def _run_expected_error_test(
    *,
    repo_root: Path,
    repo_relative_path: str,
    model_path: Path,
    expectation: OnnxFileExpectation,
    expectation_path: str,
    request: Any | None = None,
) -> None:
    global _VERBOSE_FLAGS_REPORTED
    expected_error = expectation.error
    test_data_dir = _find_test_data_dir(model_path)
    base_dir = model_path.parent
    model_argument = model_path.name
    test_data_argument = None
    if test_data_dir is not None:
        try:
            test_data_argument = str(test_data_dir.relative_to(base_dir))
        except ValueError:
            test_data_argument = str(test_data_dir.relative_to(repo_root))
    verify_args = [
        "emx-onnx-cgen",
        "verify",
        "--model-base-dir",
        str(base_dir.relative_to(repo_root)),
        model_argument,
    ]
    if _should_use_expected_checksum(expectation):
        verify_args.extend(
            [
                "--expected-checksum",
                expectation.generated_checksum,
            ]
        )
    if test_data_argument is not None:
        verify_args.extend(
            [
                "--test-data-dir",
                test_data_argument,
            ]
        )
    extra_args = MODEL_EXTRA_VERIFY_ARGS.get(repo_relative_path)
    if extra_args:
        verify_args.extend(extra_args)

    cli_result = cli.run_cli_command(verify_args)

    if cli_result.exit_code != 0:
        actual_error = cli_result.result or "ERROR UNKNOWN"
    else:
        actual_error = cli_result.result or "OK UNKNOWN"

    if request is not None and request.config.getoption("verbose") > 0:
        reporter = request.config.pluginmanager.getplugin("terminalreporter")
        if reporter is not None:
            if not _VERBOSE_FLAGS_REPORTED:
                update_refs = os.getenv("UPDATE_REFS", "").strip() or "0"
                disable_checksum = (
                    os.getenv("DISABLE_CHECKSUM", "").strip() or "0"
                )
                reporter.write_line(
                    "env: UPDATE_REFS="
                    f"{update_refs} DISABLE_CHECKSUM={disable_checksum}"
                )
                _VERBOSE_FLAGS_REPORTED = True
            reporter.write_line(
                f"{expectation_path}: result={actual_error}"
            )

    if actual_error == "CHECKSUM":
        actual_error = expected_error

    if os.getenv("UPDATE_REFS"):
        actual_expectation = OnnxFileExpectation(
            path=expectation_path,
            error=actual_error,
            command_line=cli_result.command_line,
            operators=cli_result.operators,
            opset_version=cli_result.opset_version,
            generated_checksum=cli_result.generated_checksum,
        )

        _write_expectation_file(
            actual_expectation,
            repo_relative_path=repo_relative_path,
        )
        return
    else:
        assert _errors_match(actual_error, expected_error), (
            f"Unexpected result for {expectation_path}. Expected: {expected_error!r}. "
            f"Got: {actual_error!r}."
        )


@pytest.mark.order(1)
@pytest.mark.parametrize(
    "repo_relative_path",
    _official_onnx_file_paths(),
)
def test_official_onnx_expected_errors(
    repo_relative_path: str,
    request: Any,
    official_repo_root_and_data_root: tuple[Path, Path],
) -> None:
    repo_root, _ = official_repo_root_and_data_root
    rel_path = _normalize_official_path(repo_relative_path)
    expectation = _load_expectation_for_repo_relative(rel_path)
    model_path = repo_root / rel_path
    _run_expected_error_test(
        repo_root=repo_root,
        repo_relative_path=rel_path,
        model_path=model_path,
        expectation=expectation,
        expectation_path=Path(rel_path).as_posix(),
        request=request,
    )


@pytest.mark.order(2)
@pytest.mark.parametrize(
    "repo_relative_path",
    [
        f"{LOCAL_ONNX_DATA_ROOT.relative_to(_repo_root()).as_posix()}/{path}"
        for path in _local_onnx_file_paths()
    ],
)
def test_local_onnx_expected_errors(
    repo_relative_path: str,
    request: Any,
    local_repo_root_and_data_root: tuple[Path, Path],
) -> None:
    repo_root, data_root = local_repo_root_and_data_root
    expectation = _load_expectation_for_repo_relative(
        repo_relative_path
    )
    rel_path = Path(repo_relative_path).relative_to(
        data_root.relative_to(repo_root)
    )
    model_path = data_root / rel_path
    _run_expected_error_test(
        repo_root=repo_root,
        repo_relative_path=repo_relative_path,
        model_path=model_path,
        expectation=expectation,
        expectation_path=rel_path.as_posix(),
        request=request,
    )

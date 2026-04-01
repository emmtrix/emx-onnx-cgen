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
ORT_ARTIFACTS_ONNX_PREFIX = "emx-ort-test-artifacts-org/artifacts/onnxruntime/"
LOCAL_REPO_ONNX_PREFIX = "tests/onnx/"
ORT_ARTIFACTS_ONNX_DATA_ROOT = (
    Path(__file__).resolve().parents[1]
    / "emx-ort-test-artifacts-org"
    / "artifacts"
    / "onnxruntime"
)
LOCAL_REPO_ONNX_DATA_ROOT = Path(__file__).resolve().parent / "onnx"
ONNX_FILE_LIMIT = 5000
_VERBOSE_FLAGS_REPORTED = False
MODEL_EXTRA_VERIFY_ARGS = {
    "tests/onnx/micro_kws_m_qoperator_add_shape.onnx": ("--replicate-ort-bugs",),
    "tests/onnx/micro_kws_m_qoperator_avg_pool.onnx": ("--replicate-ort-bugs",),
    "tests/onnx/micro_kws_m_qoperator_softmax.onnx": ("--replicate-ort-bugs",),
    "tests/onnx/micro_kws_m_static_qoperator.onnx": ("--replicate-ort-bugs",),
    "onnx-org/onnx/backend/test/data/node/test_nllloss_NCd1d2d3d4d5_mean_weight/model.onnx": (
        "--fp32-accumulation-strategy",
        "fp64",
    ),
    "onnx-org/onnx/backend/test/data/node/test_nllloss_NCd1d2d3d4d5_mean_weight_expanded/model.onnx": (
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
    "onnx-org/onnx/backend/test/data/node/test_dft_inverse/model.onnx": (
        "--atol-eps",
        "2",
    ),
    "onnx-org/onnx/backend/test/data/node/test_dft_inverse_opset19/model.onnx": (
        "--atol-eps",
        "2",
    ),
    "onnx-org/onnx/backend/test/data/node/test_averagepool_2d_ceil_last_window_starts_on_pad/model.onnx": (
        "--runtime",
        "onnx-reference",
        "--test-data-inputs-only",
    ),
    "onnx-org/onnx/backend/test/data/node/test_roialign_aligned_false/model.onnx": (
        "--runtime",
        "onnx-reference",
        "--test-data-inputs-only",
    ),
    "onnx-org/onnx/backend/test/data/node/test_roialign_aligned_true/model.onnx": (
        "--runtime",
        "onnx-reference",
        "--test-data-inputs-only",
    ),
    "onnx-org/onnx/backend/test/data/node/test_constant/model.onnx": (
        "--runtime",
        "onnx-reference",
    ),
    "onnx-org/onnx/backend/test/data/node/test_sequence_insert_at_back/model.onnx": (
        "--sequence-element-shape",
        "sequence=[<=4]",
    ),
    "onnx-org/onnx/backend/test/data/node/test_sequence_insert_at_front/model.onnx": (
        "--sequence-element-shape",
        "sequence=[<=4]",
    ),
    "onnx-org/onnx/backend/test/data/node/test_identity_sequence/model.onnx": (
        "--sequence-element-shape",
        "x=[1,1,2,2]",
    ),
    "onnx-org/onnx/backend/test/data/node/test_sequence_map_add_1_sequence_1_tensor/model.onnx": (
        "--sequence-element-shape",
        "x0=[10]",
    ),
    "onnx-org/onnx/backend/test/data/node/test_sequence_map_add_1_sequence_1_tensor_expanded/model.onnx": (
        "--sequence-element-shape",
        "x0=[10]",
    ),
    "onnx-org/onnx/backend/test/data/node/test_sequence_map_add_2_sequences/model.onnx": (
        "--sequence-element-shape",
        "x0=[<=6]",
        "--sequence-element-shape",
        "x1=[<=6]",
    ),
    "onnx-org/onnx/backend/test/data/node/test_sequence_map_add_2_sequences_expanded/model.onnx": (
        "--sequence-element-shape",
        "x0=[<=6]",
        "--sequence-element-shape",
        "x1=[<=6]",
    ),
    "onnx-org/onnx/backend/test/data/node/test_sequence_map_extract_shapes/model.onnx": (
        "--sequence-element-shape",
        "in_seq=[<=40,<=30,3]",
    ),
    "onnx-org/onnx/backend/test/data/node/test_sequence_map_extract_shapes_expanded/model.onnx": (
        "--sequence-element-shape",
        "in_seq=[<=40,<=30,3]",
    ),
    "onnx-org/onnx/backend/test/data/node/test_sequence_map_identity_1_sequence/model.onnx": (
        "--sequence-element-shape",
        "x=[10]",
    ),
    "onnx-org/onnx/backend/test/data/node/test_sequence_map_identity_1_sequence_expanded/model.onnx": (
        "--sequence-element-shape",
        "x=[10]",
    ),
    "onnx-org/onnx/backend/test/data/node/test_sequence_map_identity_1_sequence_1_tensor/model.onnx": (
        "--sequence-element-shape",
        "x0=[<=9]",
    ),
    "onnx-org/onnx/backend/test/data/node/test_sequence_map_identity_1_sequence_1_tensor_expanded/model.onnx": (
        "--sequence-element-shape",
        "x0=[<=9]",
    ),
    "onnx-org/onnx/backend/test/data/node/test_sequence_map_identity_2_sequences/model.onnx": (
        "--sequence-element-shape",
        "x0=[<=9]",
        "--sequence-element-shape",
        "x1=[<=8]",
    ),
    "onnx-org/onnx/backend/test/data/node/test_sequence_map_identity_2_sequences_expanded/model.onnx": (
        "--sequence-element-shape",
        "x0=[<=9]",
        "--sequence-element-shape",
        "x1=[<=8]",
    ),
    "onnx-org/onnx/backend/test/data/node/test_gridsample_bicubic/model.onnx": (
        "--runtime",
        "onnx-reference",
        "--test-data-inputs-only",
    ),
    "emx-ort-test-artifacts-org/artifacts/onnxruntime/test/contrib_ops/gridsample_test/gridsample_mode_bicubic_run0/model.onnx": (
        "--max-ulp",
        "2000",
    ),
    "onnx-org/onnx/backend/test/data/node/test_affine_grid_3d/model.onnx": (
        "--fp32-accumulation-strategy",
        "fp64",
    ),
    "onnx-org/onnx/backend/test/data/node/test_affine_grid_3d_expanded/model.onnx": (
        "--fp32-accumulation-strategy",
        "fp64",
    ),
    **{
        f"emx-ort-test-artifacts-org/artifacts/onnxruntime/test/contrib_ops/dynamic_quantize_matmul_test/{d}/model.onnx": (
            "--max-ulp",
            "5000",
        )
        for d in (
            "Int8_run0",
            "Int8_run1",
            "Int8_run2",
            "Int8_run3",
            "Int8_run4",
            "Int8_run5",
            "Int8_run6",
            "Int8_run7",
            "Int8_run8",
            "Int8_run9",
            "Int8_run10",
            "Int8_run11",
            "Int8_run12",
            "Int8_run13",
            "Int8_run14",
            "Int8_run15",
            "UInt8_run0",
            "UInt8_run1",
            "UInt8_run2",
            "UInt8_run3",
            "UInt8_run4",
            "UInt8_run5",
            "UInt8_run6",
            "UInt8_run7",
            "UInt8_run8",
            "UInt8_run9",
            "UInt8_run10",
            "UInt8_run11",
            "UInt8_run12",
            "UInt8_run13",
            "UInt8_run14",
            "UInt8_run15",
            "WithConstantBInputs_run0",
            "WithConstantBInputs_run1",
            "WithConstantBInputs_run2",
            "WithConstantBInputs_run3",
            "WithConstantBInputs_run4",
            "WithConstantBInputs_run5",
        )
    },
    **{
        f"emx-ort-test-artifacts-org/artifacts/onnxruntime/test/contrib_ops/qembed_layer_norm_op_test/{d}/model.onnx": (
            "--max-ulp",
            "500000",
        )
        for d in (
            "EmbedLayerNormBatch1_run0",
            "EmbedLayerNormBatch1_run1",
            "EmbedLayerNormBatch1_Float16_run0",
            "EmbedLayerNormBatch1_Float16_run1",
            "EmbedLayerNormBatch2_run0",
            "EmbedLayerNormBatch2_run1",
            "EmbedLayerNormBatch2_NoMask_run0",
            "EmbedLayerNormBatch2_NoMask_run1",
            "EmbedLayerNormBatch_Distill_run0",
            "EmbedLayerNormBatch_Distill_run1",
            "EmbedLayerNormLargeBatchSmallHiddenSize_run0",
            "EmbedLayerNormLargeBatchSmallHiddenSize_run1",
        )
    },
}


@dataclass(frozen=True)
class OnnxFileExpectation:
    path: str
    error: str
    command_line: str = ""
    extra_cli_args: list[str] | None = None
    verification_mode: str | None = None
    operators: list[str] | None = None
    opset_version: int | None = None
    generated_checksum: str | None = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


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
    repo_root = _repo_root()
    candidate = repo_root / path
    if candidate.exists():
        return candidate.relative_to(repo_root).as_posix()
    return (_official_data_root() / path).relative_to(repo_root).as_posix()


def _list_expectation_repo_paths(
    root: Path,
    *,
    path_filter: Callable[[str], bool],
) -> list[str]:
    if not root.exists():
        raise AssertionError(f"Expected errors directory {root} is missing.")
    repo_relative_paths: list[str] = []
    for expectation_file in sorted(root.glob("*.json")):
        repo_relative = _repo_relative_path_from_expectation_file(expectation_file)
        if not path_filter(repo_relative):
            continue
        repo_relative_paths.append(repo_relative)
    return repo_relative_paths[:ONNX_FILE_LIMIT]


@cache
def _official_onnx_file_paths() -> tuple[str, ...]:
    return tuple(
        _normalize_official_path(path) for path in _collect_onnx_files(_official_data_root())
    )


@cache
def _ort_artifacts_onnx_file_paths() -> tuple[str, ...]:
    return tuple(_collect_onnx_files(ORT_ARTIFACTS_ONNX_DATA_ROOT))


@cache
def _local_repo_onnx_file_paths() -> tuple[str, ...]:
    return tuple(_collect_onnx_files(LOCAL_REPO_ONNX_DATA_ROOT))


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
    extra_cli_args: list[str] | None = None
    verification_mode: str | None = None
    operators: list[str] | None = None
    opset_version: int | None = None
    generated_checksum: str | None = None
    if isinstance(data, dict):
        error = data.get("error", "")
        command_line = data.get("command_line", "")
        extra_cli_args = data.get("extra_cli_args")
        if extra_cli_args is None:
            legacy_extra_args = MODEL_EXTRA_VERIFY_ARGS.get(fallback_path)
            if legacy_extra_args:
                extra_cli_args = list(legacy_extra_args)
        verification_mode = data.get("verification_mode")
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
        extra_cli_args=extra_cli_args,
        verification_mode=verification_mode,
        operators=operators,
        opset_version=opset_version,
        generated_checksum=generated_checksum,
    )


def _load_expectation_for_repo_relative(
    repo_relative_path: str,
) -> OnnxFileExpectation:
    expectation_path = _expected_errors_path_for_repo_relative(repo_relative_path)
    if not expectation_path.exists():
        if os.getenv("UPDATE_REFS"):
            return OnnxFileExpectation(
                path=repo_relative_path,
                error="",
                command_line="",
            )
        raise AssertionError(f"Missing expectation file for {repo_relative_path}")
    return _read_expectation_file(
        expectation_path,
        fallback_path=repo_relative_path,
    )


def _write_expectation_file(
    expectation: OnnxFileExpectation,
    *,
    repo_relative_path: str,
) -> None:
    expectation_path = _expected_errors_path_for_repo_relative(repo_relative_path)
    expectation_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "error": expectation.error,
        "command_line": expectation.command_line,
    }
    if expectation.extra_cli_args:
        payload["extra_cli_args"] = expectation.extra_cli_args
    if expectation.verification_mode is not None:
        payload["verification_mode"] = expectation.verification_mode
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
        p.relative_to(data_root).as_posix() for p in data_root.rglob("*.onnx")
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


def _maybe_init_emx_ort_test_artifacts_org() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if shutil.which("git") is None:
        return
    subprocess.run(
        [
            "git",
            "submodule",
            "update",
            "--init",
            "--recursive",
            "emx-ort-test-artifacts-org",
        ],
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


def _ensure_ort_artifacts_onnx_files_present(data_root: Path) -> None:
    if not data_root.exists():
        _maybe_init_emx_ort_test_artifacts_org()
    if not data_root.exists():
        pytest.skip(
            "emx-ort-test-artifacts-org ONNX data is unavailable. Initialize the "
            "emx-ort-test-artifacts-org submodule to run ORT artifact tests."
        )


def _ensure_local_repo_onnx_files_present(data_root: Path) -> None:
    if not data_root.exists():
        pytest.skip("tests/onnx local test data is unavailable.")


def _find_test_data_dir(model_path: Path) -> Path | None:
    test_data_dir = model_path.parent / "test_data_set_0"
    if not test_data_dir.exists():
        return None
    if not list(test_data_dir.glob("input_*.pb")):
        return None
    return test_data_dir


def _errors_match(actual_error: str, expected_error: str) -> bool:
    if expected_error == "Failed to build testbench.":
        return actual_error.startswith("Failed to build testbench")
    return actual_error == expected_error


def test_errors_match_accepts_build_failure_with_detail() -> None:
    assert _errors_match(
        "Failed to build testbench (cc1: error: unsupported _BitInt width).",
        "Failed to build testbench.",
    )


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
    extra_args = expectation.extra_cli_args
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
                disable_checksum = os.getenv("DISABLE_CHECKSUM", "").strip() or "0"
                reporter.write_line(
                    "env: UPDATE_REFS="
                    f"{update_refs} DISABLE_CHECKSUM={disable_checksum}"
                )
                _VERBOSE_FLAGS_REPORTED = True
            reporter.write_line(f"{expectation_path}: result={actual_error}")

    if actual_error == "CHECKSUM":
        actual_error = expected_error

    if os.getenv("UPDATE_REFS"):
        actual_expectation = OnnxFileExpectation(
            path=expectation_path,
            error=actual_error,
            command_line=cli_result.command_line,
            extra_cli_args=list(expectation.extra_cli_args or []),
            verification_mode=cli_result.verification_mode,
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


def test_run_expected_error_test_does_not_add_sanitize_flag(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    model_dir = repo_root / "tests" / "onnx"
    model_dir.mkdir(parents=True)
    model_path = model_dir / "model.onnx"
    model_path.write_bytes(b"")

    captured_args: list[list[str]] = []

    def _fake_run_cli_command(argv: list[str]) -> cli.CliResult:
        captured_args.append(argv)
        return cli.CliResult(
            exit_code=0,
            command_line="verify --model-base-dir tests/onnx model.onnx",
            result="OK",
        )

    monkeypatch.setattr(cli, "run_cli_command", _fake_run_cli_command)
    monkeypatch.delenv("UPDATE_REFS", raising=False)

    _run_expected_error_test(
        repo_root=repo_root,
        repo_relative_path="tests/onnx/model.onnx",
        model_path=model_path,
        expectation=OnnxFileExpectation(
            path="tests/onnx/model.onnx",
            error="OK",
        ),
        expectation_path="tests/onnx/model.onnx",
    )

    assert captured_args
    assert "--sanitize" not in captured_args[0]


def test_read_expectation_file_reads_extra_cli_args(tmp_path: Path) -> None:
    expectation_path = tmp_path / "expectation.json"
    expectation_path.write_text(
        json.dumps(
            {
                "error": "OK",
                "command_line": (
                    "verify --model-base-dir tests/onnx model.onnx --replicate-ort-bugs"
                ),
                "extra_cli_args": ["--replicate-ort-bugs"],
            }
        ),
        encoding="utf-8",
    )

    loaded = _read_expectation_file(
        expectation_path,
        fallback_path="tests/onnx/model.onnx",
    )

    assert loaded.extra_cli_args == ["--replicate-ort-bugs"]


@pytest.mark.order(1)
@pytest.mark.parametrize(
    "repo_relative_path",
    _official_onnx_file_paths(),
)
def test_official_onnx_expected_errors(
    repo_relative_path: str,
    request: Any,
) -> None:
    data_root = _official_data_root()
    _ensure_official_onnx_files_present(data_root)
    repo_root = _repo_root()
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
        f"{ORT_ARTIFACTS_ONNX_DATA_ROOT.relative_to(_repo_root()).as_posix()}/{path}"
        for path in _ort_artifacts_onnx_file_paths()
    ],
)
def test_ort_artifacts_onnx_expected_errors(
    repo_relative_path: str,
    request: Any,
) -> None:
    data_root = ORT_ARTIFACTS_ONNX_DATA_ROOT
    _ensure_ort_artifacts_onnx_files_present(data_root)
    repo_root = _repo_root()
    expectation = _load_expectation_for_repo_relative(repo_relative_path)
    rel_path = Path(repo_relative_path).relative_to(data_root.relative_to(repo_root))
    model_path = data_root / rel_path
    _run_expected_error_test(
        repo_root=repo_root,
        repo_relative_path=repo_relative_path,
        model_path=model_path,
        expectation=expectation,
        expectation_path=rel_path.as_posix(),
        request=request,
    )


@pytest.mark.order(3)
@pytest.mark.parametrize(
    "repo_relative_path",
    [
        f"{LOCAL_REPO_ONNX_DATA_ROOT.relative_to(_repo_root()).as_posix()}/{path}"
        for path in _local_repo_onnx_file_paths()
    ],
)
def test_local_repo_onnx_expected_errors(
    repo_relative_path: str,
    request: Any,
) -> None:
    data_root = LOCAL_REPO_ONNX_DATA_ROOT
    _ensure_local_repo_onnx_files_present(data_root)
    repo_root = _repo_root()
    expectation = _load_expectation_for_repo_relative(repo_relative_path)
    rel_path = Path(repo_relative_path).relative_to(data_root.relative_to(repo_root))
    model_path = data_root / rel_path
    _run_expected_error_test(
        repo_root=repo_root,
        repo_relative_path=repo_relative_path,
        model_path=model_path,
        expectation=expectation,
        expectation_path=rel_path.as_posix(),
        request=request,
    )

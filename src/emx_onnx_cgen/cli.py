from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shlex
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, TextIO

import numpy as np
import onnx
from onnx import numpy_helper
from shared.scalar_types import ScalarType
from shared.ulp import ulp_intdiff_float

from ._build_info import BUILD_DATE, GIT_VERSION
from .compiler import Compiler, CompilerOptions
from .errors import CodegenError, ShapeInferenceError, UnsupportedOpError
from .ir.model import TensorType
from .onnx_import import import_onnx
from .determinism import deterministic_reference_runtime
from .onnxruntime_utils import make_deterministic_session_options
from .testbench import decode_testbench_array
from .verification import worst_ulp_diff

LOGGER = logging.getLogger(__name__)
_NONDETERMINISTIC_OPERATORS = {"Bernoulli"}
_EMX_STRING_MAX_LEN = 256
_DISABLE_SANITIZE_ENV = "EMX_DISABLE_SANITIZE"


def _serialize_string_tensor(array: np.ndarray) -> bytes:
    encoded = bytearray()
    for item in array.reshape(-1):
        value = item.decode("utf-8") if isinstance(item, bytes) else str(item)
        chunk = value.encode("utf-8")
        if len(chunk) >= _EMX_STRING_MAX_LEN:
            chunk = chunk[: _EMX_STRING_MAX_LEN - 1]
        encoded.extend(chunk)
        encoded.extend(b"\0" * (_EMX_STRING_MAX_LEN - len(chunk)))
    return bytes(encoded)


def _env_flag(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


@dataclass(frozen=True)
class CliResult:
    exit_code: int
    command_line: str
    result: str | None = None
    generated: str | None = None
    data_source: str | None = None
    operators: list[str] | None = None
    opset_version: int | None = None
    generated_checksum: str | None = None


@dataclass(frozen=True)
class _WorstDiff:
    output_name: str
    node_name: str | None
    index: tuple[int, ...]
    got: float
    reference: float
    ulp: int


@dataclass(frozen=True)
class _WorstAbsDiff:
    output_name: str
    node_name: str | None
    index: tuple[int, ...]
    got: object
    reference: object
    abs_diff: float | int


@dataclass
class _ComparisonMetrics:
    has_abs_diff: bool = False
    max_abs_diff: float | int = 0
    has_ulp: bool = False
    max_ulp: int = 0
    worst_diff: _WorstDiff | None = None
    worst_abs_diff: _WorstAbsDiff | None = None

    def observe_ulp(
        self,
        *,
        output_name: str,
        node_name: str | None,
        index: tuple[int, ...],
        got: float,
        reference: float,
        ulp: int,
    ) -> None:
        self.has_ulp = True
        if ulp <= self.max_ulp:
            return
        self.max_ulp = ulp
        self.worst_diff = _WorstDiff(
            output_name=output_name,
            node_name=node_name,
            index=index,
            got=got,
            reference=reference,
            ulp=ulp,
        )

    def observe_abs(
        self,
        *,
        output_name: str,
        node_name: str | None,
        index: tuple[int, ...],
        got: object,
        reference: object,
        abs_diff: float | int,
    ) -> None:
        self.has_abs_diff = True
        if abs_diff <= self.max_abs_diff:
            return
        self.max_abs_diff = abs_diff
        self.worst_abs_diff = _WorstAbsDiff(
            output_name=output_name,
            node_name=node_name,
            index=index,
            got=got,
            reference=reference,
            abs_diff=abs_diff,
        )

    def format_metrics(self) -> str:
        parts: list[str] = []
        if self.has_abs_diff:
            parts.append(f"max abs diff {self.max_abs_diff}")
        if self.has_ulp:
            parts.append(f"max ULP {self.max_ulp}")
        if not parts:
            return "no numeric comparisons"
        return ", ".join(parts)

    def failing_metric_detail(self, *, max_ulp_limit: int) -> str | None:
        if self.has_abs_diff and self.max_abs_diff > 0:
            return f"max abs diff {self.max_abs_diff}"
        if self.has_ulp and self.max_ulp > max_ulp_limit:
            return f"max ULP {self.max_ulp}"
        return None

    def format_ok_with_metrics(self) -> str:
        return f"OK ({self.format_metrics()})"


@dataclass(frozen=True)
class _VerificationIssue:
    code: str
    message: str
    metrics: _ComparisonMetrics | None = None
    details: Mapping[str, object] | None = None

    def format(self) -> str:
        parts: list[str] = []
        if self.metrics is not None:
            parts.append(self.metrics.format_metrics())
        if self.details:
            parts.extend(
                f"{key}={value}" for key, value in sorted(self.details.items())
            )
        if not parts:
            return self.message
        return f"{self.message} ({', '.join(parts)})"

    @classmethod
    def missing_output(cls, output_name: str) -> "_VerificationIssue":
        return cls(
            code="missing_output",
            message=f"Missing output {output_name} in testbench data",
            details={"output": output_name},
        )

    @classmethod
    def shape_mismatch(
        cls,
        *,
        output_name: str,
        actual_shape: tuple[int, ...],
        expected_shape: tuple[int, ...],
        actual_size: int,
        expected_size: int,
    ) -> "_VerificationIssue":
        return cls(
            code="shape_mismatch",
            message=f"Output shape mismatch for {output_name}",
            details={
                "output": output_name,
                "actual_shape": actual_shape,
                "expected_shape": expected_shape,
                "actual_size": actual_size,
                "expected_size": expected_size,
            },
        )

    @classmethod
    def out_of_tolerance(
        cls, metrics: _ComparisonMetrics, *, max_ulp_limit: int
    ) -> "_VerificationIssue":
        assert metrics.failing_metric_detail(max_ulp_limit=max_ulp_limit) is not None
        return cls(
            code="out_of_tolerance",
            message="Out of tolerance",
            metrics=metrics,
        )


class _VerifyReporter:
    def __init__(
        self,
        stream: TextIO | None = None,
        *,
        color_mode: str = "auto",
    ) -> None:
        self._stream = stream or sys.stdout
        self._use_color = self._should_use_color(color_mode)
        self._deferred_actions: list[Callable[[], None]] = []

    def _should_use_color(self, color_mode: str) -> bool:
        if color_mode == "always":
            return True
        if color_mode == "never":
            return False
        if not hasattr(self._stream, "isatty"):
            return False
        return bool(self._stream.isatty())

    def _color(self, text: str, code: str) -> str:
        if not self._use_color:
            return text
        return f"\x1b[{code}m{text}\x1b[0m"

    def start_step(self, label: str) -> float:
        print(f"{label} ...", end=" ", file=self._stream, flush=True)
        return time.perf_counter()

    def step_ok(self, started_at: float) -> None:
        duration = time.perf_counter() - started_at
        ok = self._color("OK", "32")
        dim = self._color(f"({duration:.3f}s)", "90")
        print(f"{ok} {dim}", file=self._stream)

    def step_ok_simple(self) -> None:
        ok = self._color("OK", "32")
        print(ok, file=self._stream)

    def step_ok_detail(self, detail: str) -> None:
        ok = self._color("OK", "32")
        dim = self._color(f"({detail})", "90")
        print(f"{ok} {dim}", file=self._stream)

    def step_fail(self, reason: str) -> None:
        fail = self._color("FAIL", "31")
        print(f"{fail} ({reason})", file=self._stream)

    def note(self, message: str) -> None:
        label = self._color("Note:", "33")
        print(f"{label} {message}", file=self._stream)

    def info(self, message: str) -> None:
        print(message, file=self._stream)

    def result(self, message: str, *, ok: bool) -> None:
        colored = self._color(message, "32" if ok else "31")
        print(f"Result: {colored}", file=self._stream)

    def defer(self, action: Callable[[], None]) -> None:
        self._deferred_actions.append(action)

    def flush_deferred(self) -> None:
        if not self._deferred_actions:
            return
        print("", file=self._stream)
        while self._deferred_actions:
            action = self._deferred_actions.pop(0)
            action()


class _NullVerifyReporter(_VerifyReporter):
    def __init__(self) -> None:
        super().__init__(stream=sys.stdout, color_mode="never")

    def start_step(self, label: str) -> float:
        return time.perf_counter()

    def step_ok(self, started_at: float) -> None:
        return None

    def step_ok_simple(self) -> None:
        return None

    def step_ok_detail(self, detail: str) -> None:
        return None

    def step_fail(self, reason: str) -> None:
        return None

    def note(self, message: str) -> None:
        return None

    def info(self, message: str) -> None:
        return None

    def result(self, message: str, *, ok: bool) -> None:
        return None


def _format_artifact_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} bytes"
    return f"{size_bytes / 1024:.1f} KiB"


def _report_generated_artifacts(
    reporter: _VerifyReporter,
    *,
    artifacts: Sequence[tuple[str, int]],
) -> None:
    for name, size_bytes in artifacts:
        reporter.info(f"  {name} ({_format_artifact_size(size_bytes)})")


def _worst_ulp_diff(
    actual: "np.ndarray", expected: "np.ndarray"
) -> tuple[int, tuple[tuple[int, ...], float, float] | None]:
    if actual.shape != expected.shape:
        raise ValueError(
            f"Shape mismatch for ULP calculation: {actual.shape} vs {expected.shape}"
        )
    if not np.issubdtype(expected.dtype, np.floating):
        return 0, None
    if actual.size == 0:
        return 0, None
    dtype = expected.dtype
    actual_cast = actual.astype(dtype, copy=False)
    expected_cast = expected.astype(dtype, copy=False)
    max_diff = 0
    worst: tuple[tuple[int, ...], float, float] | None = None
    iterator = np.nditer([actual_cast, expected_cast], flags=["refs_ok", "multi_index"])
    for actual_value, expected_value in iterator:
        actual_scalar = float(actual_value[()])
        expected_scalar = float(expected_value[()])
        diff = ulp_intdiff_float(actual_value[()], expected_value[()])
        if diff > max_diff:
            max_diff = diff
            worst = (
                iterator.multi_index,
                actual_scalar,
                expected_scalar,
            )
    return max_diff, worst


def _worst_abs_diff(
    actual: "np.ndarray", expected: "np.ndarray"
) -> tuple[float | int, tuple[tuple[int, ...], object, object] | None]:
    if actual.shape != expected.shape:
        raise ValueError(
            f"Shape mismatch for diff calculation: {actual.shape} vs {expected.shape}"
        )
    if actual.size == 0:
        return 0, None
    dtype = expected.dtype
    actual_cast = actual.astype(dtype, copy=False)
    expected_cast = expected.astype(dtype, copy=False)
    max_diff: float | int = 0
    worst: tuple[tuple[int, ...], object, object] | None = None
    iterator = np.nditer([actual_cast, expected_cast], flags=["refs_ok", "multi_index"])
    for actual_value, expected_value in iterator:
        actual_scalar = actual_value[()]
        expected_scalar = expected_value[()]
        if actual_scalar == expected_scalar:
            continue
        try:
            if np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.bool_):
                diff: float | int = abs(int(actual_scalar) - int(expected_scalar))
            else:
                diff = float(abs(actual_scalar - expected_scalar))
        except Exception:
            diff = 1
        if diff > max_diff:
            max_diff = diff
            worst = (
                iterator.multi_index,
                actual_scalar,
                expected_scalar,
            )
    return max_diff, worst


def run_cli_command(
    argv: Sequence[str],
) -> CliResult:
    raw_argv = list(argv)
    parse_argv = raw_argv
    if raw_argv and raw_argv[0] == "emx-onnx-cgen":
        parse_argv = raw_argv[1:]
    parser = _build_parser()
    args = parser.parse_args(parse_argv)
    args.command_line = _format_command_line(raw_argv)
    _apply_base_dir(args, parser)

    try:
        if args.command != "compile":
            (
                success_message,
                error,
                operators,
                opset_version,
                generated_checksum,
            ) = _verify_model(
                args, include_build_details=False, reporter=_NullVerifyReporter()
            )
            return CliResult(
                exit_code=0 if error is None else 1,
                command_line=args.command_line,
                result=error or success_message,
                operators=operators,
                opset_version=opset_version,
                generated_checksum=generated_checksum,
            )
        generated, _testbench, data_source, _weight_data, error = _compile_model(args)
        if error:
            return CliResult(
                exit_code=1,
                command_line=args.command_line,
                result=error,
            )
        return CliResult(
            exit_code=0,
            command_line=args.command_line,
            result="",
            generated=generated,
            data_source=data_source,
        )
    except Exception as exc:  # pragma: no cover - defensive reporting
        LOGGER.exception("Unhandled exception while running CLI command.")
        return CliResult(
            exit_code=1,
            command_line=args.command_line,
            result=str(exc),
        )


def _build_parser() -> argparse.ArgumentParser:
    description = (
        "emmtrix ONNX-to-C Code Generator "
        f"(build date: {BUILD_DATE}, git: {GIT_VERSION})"
    )
    parser = argparse.ArgumentParser(prog="emx-onnx-cgen", description=description)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_color_flag(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "--color",
            choices=("auto", "always", "never"),
            default="auto",
            help=("Colorize CLI output (default: auto; options: auto, always, never)"),
        )

    def add_verbose_flag(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "--verbose",
            "-v",
            action="store_true",
            help="Enable verbose logging (includes codegen timing).",
        )

    def add_restrict_flags(subparser: argparse.ArgumentParser) -> None:
        restrict_group = subparser.add_mutually_exclusive_group()
        restrict_group.add_argument(
            "--restrict-arrays",
            dest="restrict_arrays",
            action="store_true",
            help="Enable restrict qualifiers on generated array parameters",
        )
        restrict_group.add_argument(
            "--no-restrict-arrays",
            dest="restrict_arrays",
            action="store_false",
            help="Disable restrict qualifiers on generated array parameters",
        )
        subparser.set_defaults(restrict_arrays=True)

    def add_fp32_accumulation_strategy_flag(
        subparser: argparse.ArgumentParser,
    ) -> None:
        subparser.add_argument(
            "--fp32-accumulation-strategy",
            choices=("simple", "fp64"),
            default="simple",
            help=(
                "Accumulation strategy for float32 inputs "
                "(simple uses float32, fp64 uses double; default: simple)"
            ),
        )

    def add_fp16_accumulation_strategy_flag(
        subparser: argparse.ArgumentParser,
    ) -> None:
        subparser.add_argument(
            "--fp16-accumulation-strategy",
            choices=("simple", "fp32"),
            default="fp32",
            help=(
                "Accumulation strategy for float16 inputs "
                "(simple uses float16, fp32 uses float; default: fp32)"
            ),
        )

    compile_parser = subparsers.add_parser(
        "compile", help="Compile an ONNX model into C source"
    )
    add_color_flag(compile_parser)
    add_verbose_flag(compile_parser)
    compile_parser.add_argument(
        "--model-base-dir",
        "-B",
        type=Path,
        default=None,
        help=(
            "Base directory for resolving the model path "
            "(example: tool --model-base-dir /data model.onnx)"
        ),
    )
    compile_parser.add_argument("model", type=Path, help="Path to the ONNX model")
    compile_parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=None,
        help=(
            "Output C file path (default: use model filename with .c suffix, "
            "e.g., model.onnx -> model.c)"
        ),
    )
    compile_parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Override the generated model name (default: output file stem)",
    )
    compile_parser.add_argument(
        "--emit-testbench",
        action="store_true",
        help="Emit a JSON-producing testbench main() for validation",
    )
    compile_parser.add_argument(
        "--testbench-file",
        type=str,
        default=None,
        help=(
            "If set, emit the testbench into a separate C file at this path "
            "(implies --emit-testbench)."
        ),
    )
    compile_parser.add_argument(
        "--emit-data-file",
        action="store_true",
        help=(
            "Emit constant data arrays to a separate C file "
            "named like the output with a _data suffix"
        ),
    )
    compile_parser.add_argument(
        "--truncate-weights-after",
        type=int,
        default=None,
        help=(
            "Truncate inline weight initializers after N values and insert "
            '"..." placeholders (default: no truncation)'
        ),
    )
    compile_parser.add_argument(
        "--large-temp-threshold",
        type=int,
        default=1024,
        dest="large_temp_threshold_bytes",
        help=(
            "Mark temporary buffers larger than this threshold as static "
            "(default: 1024)"
        ),
    )
    compile_parser.add_argument(
        "--large-weight-threshold",
        type=int,
        default=100 * 1024,
        help=(
            "Store weights in a binary file once the cumulative byte size "
            "exceeds this threshold (default: 102400; set to 0 to disable)"
        ),
    )
    add_restrict_flags(compile_parser)
    add_fp32_accumulation_strategy_flag(compile_parser)
    add_fp16_accumulation_strategy_flag(compile_parser)

    verify_parser = subparsers.add_parser(
        "verify",
        help="Compile an ONNX model and verify outputs against ONNX Runtime",
    )
    add_color_flag(verify_parser)
    add_verbose_flag(verify_parser)
    verify_parser.add_argument(
        "--model-base-dir",
        "-B",
        type=Path,
        default=None,
        help=(
            "Base directory for resolving the model and test data paths "
            "(example: tool --model-base-dir /data model.onnx --test-data-dir inputs)"
        ),
    )
    verify_parser.add_argument("model", type=Path, help="Path to the ONNX model")
    verify_parser.add_argument(
        "--cc",
        type=str,
        default=None,
        help="C compiler command to build the testbench binary",
    )
    verify_parser.add_argument(
        "--sanitize",
        action="store_true",
        help=(
            "Build the verification binary with sanitizers enabled "
            "(-fsanitize=address,undefined); set "
            f"{_DISABLE_SANITIZE_ENV}=1 to force-disable sanitizers"
        ),
    )
    verify_parser.add_argument(
        "--per-node-accuracy",
        action="store_true",
        help=(
            "Compare intermediate tensor outputs and print accuracy per node "
            "(runs verification with all tensor node outputs exposed)"
        ),
    )
    verify_parser.add_argument(
        "--truncate-weights-after",
        type=int,
        default=None,
        help=(
            "Truncate inline weight initializers after N values and insert "
            '"..." placeholders (default: no truncation)'
        ),
    )
    verify_parser.add_argument(
        "--large-temp-threshold",
        type=int,
        default=1024,
        dest="large_temp_threshold_bytes",
        help=(
            "Mark temporary buffers larger than this threshold as static "
            "(default: 1024)"
        ),
    )
    verify_parser.add_argument(
        "--large-weight-threshold",
        type=int,
        default=100 * 1024,
        help=(
            "Store weights in a binary file once the cumulative byte size "
            "exceeds this threshold (default: 102400)"
        ),
    )
    verify_parser.add_argument(
        "--test-data-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing input_*.pb files to seed verification inputs "
            "(default: use random testbench inputs)"
        ),
    )
    verify_parser.add_argument(
        "--temp-dir-root",
        type=Path,
        default=None,
        help=(
            "Root directory in which to create a temporary verification "
            "directory (default: system temp dir)"
        ),
    )
    verify_parser.add_argument(
        "--temp-dir",
        type=Path,
        default=None,
        help=(
            "Exact directory to use for temporary verification files "
            "(default: create a temporary directory)"
        ),
    )
    verify_parser.add_argument(
        "--keep-temp-dir",
        action="store_true",
        help="Keep the temporary verification directory (default: delete it)",
    )
    verify_parser.add_argument(
        "--max-ulp",
        type=int,
        default=100,
        help="Maximum allowed ULP difference for floating outputs (default: 100)",
    )
    verify_parser.add_argument(
        "--atol-eps",
        type=float,
        default=1.0,
        help=(
            "Absolute tolerance as a multiple of machine epsilon for ULP checks "
            "(default: 1.0)"
        ),
    )
    verify_parser.add_argument(
        "--runtime",
        choices=("onnxruntime", "onnx-reference"),
        default="onnxruntime",
        help=(
            "Runtime backend for verification (default: onnxruntime; "
            "options: onnxruntime, onnx-reference)"
        ),
    )
    verify_parser.add_argument(
        "--expected-checksum",
        type=str,
        default=None,
        help=(
            "Expected generated C checksum (sha256). When it matches the "
            "computed checksum, verification exits early with CHECKSUM."
        ),
    )
    add_restrict_flags(verify_parser)
    add_fp32_accumulation_strategy_flag(verify_parser)
    add_fp16_accumulation_strategy_flag(verify_parser)
    return parser


def _resolve_with_base_dir(base_dir: Path, path: Path) -> Path:
    if path.is_absolute():
        return path
    return Path(os.path.normpath(os.path.join(base_dir, path)))


def _apply_base_dir(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    model_base_dir: Path | None = args.model_base_dir
    if model_base_dir is None:
        return
    if not model_base_dir.exists() or not model_base_dir.is_dir():
        parser.error(
            f"--model-base-dir {model_base_dir} does not exist or is not a directory"
        )
    path_fields = ("model", "test_data_dir")
    for field in path_fields:
        value = getattr(args, field, None)
        if value is None:
            continue
        if not isinstance(value, Path):
            continue
        setattr(args, field, _resolve_with_base_dir(model_base_dir, value))


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO)
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.command_line = _format_command_line(argv)
    _apply_base_dir(args, parser)

    if args.command == "compile":
        return _handle_compile(args)
    if args.command == "verify":
        return _handle_verify(args)
    parser.error(f"Unknown command {args.command}")
    return 1


def _handle_compile(args: argparse.Namespace) -> int:
    reporter = _VerifyReporter(color_mode=args.color)
    model_path: Path = args.model
    output_path: Path = args.output or model_path.with_suffix(".c")
    model_name = args.model_name or "model"
    if args.testbench_file:
        args.emit_testbench = True
    generated, testbench, data_source, weight_data, error = _compile_model(
        args, reporter=reporter
    )
    if error:
        reporter.info("")
        reporter.result(error, ok=False)
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(generated or "", encoding="utf-8")
    if testbench is not None:
        testbench_path = _resolve_testbench_output_path(
            output_path, args.testbench_file
        )
        if args.testbench_file:
            testbench_decls = _compile_testbench_declarations(args, reporter=reporter)
            testbench_path.write_text(
                _wrap_separate_testbench_source(testbench_decls, testbench),
                encoding="utf-8",
            )
        else:
            # Embedded testbench: no separate file.
            pass
    if data_source is not None:
        data_path = output_path.with_name(
            f"{output_path.stem}_data{output_path.suffix}"
        )
        data_path.write_text(data_source, encoding="utf-8")
    if weight_data is not None:
        weights_path = output_path.with_name(f"{model_name}.bin")
        weights_path.write_bytes(weight_data)
    return 0


def _compile_model(
    args: argparse.Namespace,
    *,
    reporter: _VerifyReporter | None = None,
) -> tuple[str | None, str | None, str | None, bytes | None, str | None]:
    model_path: Path = args.model
    model_name = args.model_name or "model"
    active_reporter = reporter or _NullVerifyReporter()
    load_started = active_reporter.start_step(f"Loading model {model_path.name}")
    timings: dict[str, float] = {}
    try:
        model, model_checksum = _load_model_and_checksum(model_path)
        active_reporter.step_ok(load_started)
    except OSError as exc:
        active_reporter.step_fail(str(exc))
        return None, None, None, None, str(exc)
    operators = _collect_model_operators(model)
    opset_version = _model_opset_version(model)
    _report_model_details(
        active_reporter,
        model_path=model_path,
        model_checksum=model_checksum,
        operators=operators,
        opset_version=opset_version,
        node_count=len(model.graph.node),
        initializer_count=len(model.graph.initializer),
        input_count=len(model.graph.input),
        output_count=len(model.graph.output),
    )

    active_reporter.info("")
    codegen_started = active_reporter.start_step("Generating C code")
    try:
        separate_testbench = bool(args.testbench_file)
        options = CompilerOptions(
            model_name=model_name,
            emit_testbench=args.emit_testbench and not separate_testbench,
            command_line=args.command_line,
            model_checksum=model_checksum,
            restrict_arrays=args.restrict_arrays,
            fp32_accumulation_strategy=args.fp32_accumulation_strategy,
            fp16_accumulation_strategy=args.fp16_accumulation_strategy,
            truncate_weights_after=args.truncate_weights_after,
            large_temp_threshold_bytes=args.large_temp_threshold_bytes,
            large_weight_threshold=args.large_weight_threshold,
            timings=timings,
        )
        compiler = Compiler(options)
        if args.emit_data_file:
            generated, data_source, weight_data = (
                compiler.compile_with_data_file_and_weight_data(model)
            )
        else:
            generated, weight_data = compiler.compile_with_weight_data(model)
            data_source = None
        testbench = None
        if separate_testbench:
            testbench = compiler.compile_testbench(model)
        active_reporter.step_ok(codegen_started)
        if args.verbose:
            _report_codegen_timings(active_reporter, timings=timings)
    except (CodegenError, ShapeInferenceError, UnsupportedOpError) as exc:
        active_reporter.step_fail(str(exc))
        return None, None, None, None, str(exc)
    output_path: Path = args.output or model_path.with_suffix(".c")
    artifacts = [(str(output_path), len(generated.encode("utf-8")))]
    if testbench is not None:
        testbench_path = _resolve_testbench_output_path(
            output_path, args.testbench_file
        )
        if args.testbench_file:
            testbench_decls = _compile_testbench_declarations(args, reporter=reporter)
            wrapped_testbench = _wrap_separate_testbench_source(
                testbench_decls, testbench
            )
            artifacts.append(
                (str(testbench_path), len(wrapped_testbench.encode("utf-8")))
            )
    if data_source is not None:
        data_path = output_path.with_name(
            f"{output_path.stem}_data{output_path.suffix}"
        )
        artifacts.append((str(data_path), len(data_source.encode("utf-8"))))
    if weight_data is not None:
        weights_path = output_path.with_name(f"{model_name}.bin")
        artifacts.append((str(weights_path), len(weight_data)))
    _report_generated_artifacts(active_reporter, artifacts=artifacts)
    active_reporter.info(
        f"  Generated checksum (sha256): {_generated_checksum(generated)}"
    )
    return generated, testbench, data_source, weight_data, None


def _compile_testbench_declarations(
    args: argparse.Namespace,
    *,
    reporter: _VerifyReporter | None = None,
) -> str:
    model_path: Path = args.model
    model_name = args.model_name or "model"
    active_reporter = reporter or _NullVerifyReporter()
    try:
        model, _model_checksum = _load_model_and_checksum(model_path)
    except OSError as exc:
        raise CodegenError(str(exc)) from exc
    options = CompilerOptions(
        model_name=model_name,
        emit_testbench=False,
        command_line=args.command_line,
        model_checksum=None,
        restrict_arrays=args.restrict_arrays,
        fp32_accumulation_strategy=args.fp32_accumulation_strategy,
        fp16_accumulation_strategy=args.fp16_accumulation_strategy,
        truncate_weights_after=args.truncate_weights_after,
        large_temp_threshold_bytes=args.large_temp_threshold_bytes,
        large_weight_threshold=args.large_weight_threshold,
        testbench_optional_inputs=None,
        timings=None,
    )
    compiler = Compiler(options)
    try:
        return compiler.compile_testbench_declarations(model)
    except (CodegenError, ShapeInferenceError, UnsupportedOpError) as exc:
        active_reporter.step_fail(str(exc))
        raise


def _wrap_separate_testbench_source(declarations: str, testbench_source: str) -> str:
    preamble = "\n".join(
        [
            "/* Testbench (separate translation unit). */",
            "#include <stdint.h>",
            "#include <stdbool.h>",
            "#include <stdio.h>",
            "#include <math.h>",
            "#include <float.h>",
            "#include <string.h>",
            "",
            "#ifndef idx_t",
            "#define idx_t int32_t",
            "#endif",
            "#ifndef EMX_STRING_MAX_LEN",
            "#define EMX_STRING_MAX_LEN 256",
            "#endif",
            "#ifndef EMX_SEQUENCE_MAX_LEN",
            "#define EMX_SEQUENCE_MAX_LEN 32",
            "#endif",
            "",
            declarations.rstrip(),
            "",
        ]
    )
    return f"{preamble}\n{testbench_source}"


def _resolve_testbench_output_path(output_path: Path, testbench_file: str) -> Path:
    requested = Path(testbench_file)
    if not requested.suffix:
        requested = requested.with_suffix(output_path.suffix)
    if requested.is_absolute():
        return requested
    return output_path.with_name(str(requested))


def _resolve_compiler(cc: str | None, prefer_ccache: bool = False) -> list[str] | None:
    def maybe_prefix_ccache(tokens: list[str]) -> list[str]:
        if not prefer_ccache:
            return tokens
        ccache = shutil.which("ccache")
        if not ccache:
            return tokens
        return [ccache, *tokens]

    def resolve_tokens(tokens: list[str]) -> list[str] | None:
        if not tokens:
            return None
        if shutil.which(tokens[0]):
            return tokens
        for token in reversed(tokens):
            if shutil.which(token):
                return [token]
        return None

    if cc:
        return resolve_tokens(shlex.split(cc))
    env_cc = os.environ.get("CC")
    if env_cc:
        return resolve_tokens(shlex.split(env_cc))
    for candidate in ("cc", "gcc", "clang"):
        if shutil.which(candidate):
            return maybe_prefix_ccache([candidate])
    return None


def _handle_verify(args: argparse.Namespace) -> int:
    reporter = _VerifyReporter(color_mode=args.color)
    (
        success_message,
        error,
        _operators,
        _opset_version,
        generated_checksum,
    ) = _verify_model(args, include_build_details=True, reporter=reporter)
    if error is not None:
        reporter.flush_deferred()
        reporter.info("")
        reporter.result(error, ok=False)
        return 1
    reporter.flush_deferred()
    if success_message:
        reporter.info("")
        reporter.result(success_message, ok=True)
    return 0


def _augment_model_with_tensor_node_outputs(
    model: onnx.ModelProto,
    graph: Any,
) -> onnx.ModelProto:
    augmented = onnx.ModelProto()
    augmented.CopyFrom(model)
    existing_output_names = {output.name for output in augmented.graph.output}
    value_by_name = {
        value.name: value
        for value in (*graph.values, *graph.outputs)
        if isinstance(value.type, TensorType)
    }
    for node in graph.nodes:
        for output_name in node.outputs:
            if not output_name or output_name in existing_output_names:
                continue
            value = value_by_name.get(output_name)
            if value is None:
                continue
            dims: list[int | str | None] = []
            for index, dim in enumerate(value.type.shape):
                dim_param = None
                if index < len(value.type.dim_params):
                    dim_param = value.type.dim_params[index]
                if dim_param:
                    dims.append(dim_param)
                else:
                    dims.append(int(dim) if dim is not None else None)
            elem_type = onnx.helper.np_dtype_to_tensor_dtype(value.type.dtype.np_dtype)
            value_info = onnx.helper.make_tensor_value_info(
                output_name,
                elem_type,
                dims,
            )
            augmented.graph.output.append(value_info)
            existing_output_names.add(output_name)
    return augmented


def _report_per_node_accuracy(
    reporter: _VerifyReporter,
    *,
    graph: Any,
    decoded_outputs: Mapping[str, tuple[np.ndarray, np.ndarray]],
    output_dtypes: Mapping[str, ScalarType],
    atol_eps: float,
    max_ulp_limit: int,
) -> None:
    producer_by_output: dict[str, int] = {}
    for node_index, node in enumerate(graph.nodes):
        for output_name in node.outputs:
            if output_name:
                producer_by_output[output_name] = node_index

    node_dependencies: dict[int, set[int]] = {
        index: set() for index, _ in enumerate(graph.nodes)
    }
    for node_index, node in enumerate(graph.nodes):
        for input_name in node.inputs:
            if not input_name:
                continue
            producer_index = producer_by_output.get(input_name)
            if producer_index is None:
                continue
            if producer_index != node_index:
                node_dependencies[node_index].add(producer_index)

    reporter.info("Per-node accuracy:")
    compared_nodes = 0
    node_failed: dict[int, bool] = {}
    node_max_ulp: dict[int, int] = {}
    node_max_abs: dict[int, float | int] = {}
    for node_index, node in enumerate(graph.nodes):
        compared_output_names = [
            output_name
            for output_name in node.outputs
            if output_name in decoded_outputs
        ]
        if not compared_output_names:
            continue
        compared_nodes += 1
        node_has_failure = False
        node_peak_ulp = 0
        node_peak_abs: float | int = 0
        node_name = node.name or f"node_{node_index}"
        reporter.info(f"    {node_name} [{node.op_type}]")
        for output_name in compared_output_names:
            actual, reference = decoded_outputs[output_name]
            dtype = output_dtypes[output_name].np_dtype
            reporter.start_step(f"      {output_name}")
            if np.issubdtype(dtype, np.floating):
                output_max_ulp, _ = worst_ulp_diff(
                    actual,
                    reference,
                    atol_eps=atol_eps,
                )
                if output_max_ulp > node_peak_ulp:
                    node_peak_ulp = output_max_ulp
                if output_max_ulp > max_ulp_limit:
                    node_has_failure = True
                    reporter.step_fail(f"max ULP {output_max_ulp}")
                else:
                    reporter.step_ok_detail(f"max ULP {output_max_ulp}")
            else:
                output_max_abs, _ = _worst_abs_diff(actual, reference)
                if output_max_abs > node_peak_abs:
                    node_peak_abs = output_max_abs
                if output_max_abs > 0:
                    node_has_failure = True
                    reporter.step_fail(f"max abs diff {output_max_abs}")
                else:
                    reporter.step_ok_detail("max abs diff 0")
        node_failed[node_index] = node_has_failure
        node_max_ulp[node_index] = node_peak_ulp
        node_max_abs[node_index] = node_peak_abs
    if compared_nodes == 0:
        reporter.note("Per-node accuracy: no tensor node outputs were comparable.")
        return

    failing_nodes = {node_index for node_index, failed in node_failed.items() if failed}
    if not failing_nodes:
        reporter.info("Suspects: none (no failing nodes).")
        return

    suspects = [
        node_index
        for node_index in sorted(failing_nodes)
        if not any(parent in failing_nodes for parent in node_dependencies[node_index])
    ]
    if not suspects:
        reporter.info(
            "Suspects: none (every failing node depends on earlier failing nodes)."
        )
        return

    failing_children: dict[int, set[int]] = {
        node_index: set() for node_index in failing_nodes
    }
    for child_index in failing_nodes:
        for parent_index in node_dependencies[child_index]:
            if parent_index in failing_nodes:
                failing_children[parent_index].add(child_index)

    def impact_score(source_node_index: int) -> int:
        seen: set[int] = set()
        stack = [source_node_index]
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(failing_children.get(current, ()))
        return len(seen)

    ranked_suspects = sorted(
        suspects,
        key=lambda node_index: (-impact_score(node_index), node_index),
    )
    reporter.info("Suspects (likely root causes):")
    for node_index in ranked_suspects:
        node = graph.nodes[node_index]
        name = node.name or f"node_{node_index}"
        reporter.info(
            "    "
            f"{name} [{node.op_type}] "
            f"(impact={impact_score(node_index)}, "
            f"max_ulp={node_max_ulp.get(node_index, 0)}, "
            f"max_abs={node_max_abs.get(node_index, 0)})"
        )


def _verify_model(
    args: argparse.Namespace,
    *,
    include_build_details: bool,
    reporter: _VerifyReporter | None = None,
) -> tuple[str | None, str | None, list[str], int | None, str | None]:
    active_reporter = reporter or _NullVerifyReporter()
    operators: list[str] = []
    opset_version: int | None = None
    generated_checksum: str | None = None

    def describe_exit_code(returncode: int) -> str:
        if returncode >= 0:
            return f"exit code {returncode}"
        signal_id = -returncode
        try:
            signal_name = signal.Signals(signal_id).name
        except ValueError:
            signal_name = "unknown"
        return f"exit code {returncode} (signal {signal_id}: {signal_name})"

    model_path: Path = args.model
    model_name = "model"
    model, model_checksum = _load_model_and_checksum(model_path)
    compiler_cmd = _resolve_compiler(args.cc, prefer_ccache=False)
    if compiler_cmd is None:
        return (
            None,
            "No C compiler found (set --cc or CC environment variable).",
            [],
            None,
            None,
        )
    temp_dir_root: Path | None = args.temp_dir_root
    explicit_temp_dir: Path | None = args.temp_dir
    if temp_dir_root is not None and explicit_temp_dir is not None:
        return (
            None,
            "Cannot set both --temp-dir-root and --temp-dir.",
            operators,
            opset_version,
            generated_checksum,
        )
    if temp_dir_root is not None:
        if temp_dir_root.exists() and not temp_dir_root.is_dir():
            return (
                None,
                f"Verification temp dir root is not a directory: {temp_dir_root}",
                operators,
                opset_version,
                generated_checksum,
            )
        temp_dir_root.mkdir(parents=True, exist_ok=True)
    if explicit_temp_dir is not None:
        if explicit_temp_dir.exists() and not explicit_temp_dir.is_dir():
            return (
                None,
                f"Verification temp dir is not a directory: {explicit_temp_dir}",
                operators,
                opset_version,
                generated_checksum,
            )
    temp_dir: tempfile.TemporaryDirectory | None = None
    cleanup_created_dir = False
    if explicit_temp_dir is not None:
        temp_path = explicit_temp_dir
        if not temp_path.exists():
            temp_path.mkdir(parents=True, exist_ok=True)
            cleanup_created_dir = not args.keep_temp_dir
    elif args.keep_temp_dir:
        temp_path = Path(
            tempfile.mkdtemp(
                dir=str(temp_dir_root) if temp_dir_root is not None else None
            )
        )
    else:
        temp_dir = tempfile.TemporaryDirectory(
            dir=str(temp_dir_root) if temp_dir_root is not None else None
        )
        temp_path = Path(temp_dir.name)
    keep_label = (
        "--keep-temp-dir set" if args.keep_temp_dir else "--keep-temp-dir not set"
    )
    active_reporter.note(f"Using temporary folder [{keep_label}]: {temp_path}")
    active_reporter.info("")
    load_started = active_reporter.start_step(f"Loading model {model_path.name}")
    try:
        model, model_checksum = _load_model_and_checksum(model_path)
    except OSError as exc:
        active_reporter.step_fail(str(exc))
        return None, str(exc), [], None, None
    active_reporter.step_ok(load_started)

    operators = _collect_model_operators(model)
    opset_version = _model_opset_version(model)
    _report_model_details(
        active_reporter,
        model_path=model_path,
        model_checksum=model_checksum,
        operators=operators,
        opset_version=opset_version,
        node_count=len(model.graph.node),
        initializer_count=len(model.graph.initializer),
        input_count=len(model.graph.input),
        output_count=len(model.graph.output),
    )

    try:
        graph = import_onnx(model)
    except (KeyError, UnsupportedOpError, ShapeInferenceError) as exc:
        return (
            None,
            str(exc),
            operators,
            opset_version,
            None,
        )
    original_output_names = tuple(value.name for value in graph.outputs)
    if args.per_node_accuracy:
        original_output_count = len(model.graph.output)
        model = _augment_model_with_tensor_node_outputs(model, graph)
        added_outputs = len(model.graph.output) - original_output_count
        if added_outputs > 0:
            active_reporter.note(
                f"Per-node accuracy enabled: added {added_outputs} tensor node outputs."
            )
        try:
            graph = import_onnx(model)
        except (KeyError, UnsupportedOpError, ShapeInferenceError) as exc:
            return (
                None,
                str(exc),
                operators,
                opset_version,
                None,
            )
    output_compare_names = set(original_output_names)
    requires_non_tensor_reference = any(
        value.name in output_compare_names and not isinstance(value.type, TensorType)
        for value in graph.outputs
    )
    timings: dict[str, float] = {}
    lowered_sequence_input_shapes: dict[str, tuple[int, ...]] = {}
    try:
        active_reporter.info("")
        codegen_started = active_reporter.start_step("Generating C code")
        testbench_inputs, testbench_optional_inputs, reshaped_tensor_inputs = (
            _load_test_data_inputs(
            model, args.test_data_dir
            )
        )
        testbench_outputs = None
        if not args.per_node_accuracy:
            testbench_outputs = _load_test_data_outputs(model, args.test_data_dir)
            if reshaped_tensor_inputs and testbench_outputs is not None:
                active_reporter.note(
                    "Ignoring test-data outputs because input tensors were reshaped "
                    "to match model input signatures; using runtime reference instead."
                )
                testbench_outputs = None
        if args.per_node_accuracy:
            active_reporter.note(
                "Per-node accuracy: ignoring --test-data-dir reference outputs "
                "and running runtime backend for all node outputs."
            )
        options = CompilerOptions(
            model_name=model_name,
            emit_testbench=True,
            command_line=None,
            model_checksum=model_checksum,
            restrict_arrays=args.restrict_arrays,
            fp32_accumulation_strategy=args.fp32_accumulation_strategy,
            fp16_accumulation_strategy=args.fp16_accumulation_strategy,
            truncate_weights_after=args.truncate_weights_after,
            large_temp_threshold_bytes=args.large_temp_threshold_bytes,
            large_weight_threshold=args.large_weight_threshold,
            testbench_optional_inputs=None,
            shape_inference_inputs=testbench_inputs,
            timings=timings,
        )
        compiler = Compiler(options)
        generated, weight_data = compiler.compile_with_weight_data(model)
        if testbench_inputs:
            compile_ctx = compiler._build_compile_context(model)
            lowered_sequence_input_shapes = {
                name: tuple(shape)
                for name, shape, value_type in zip(
                    compile_ctx.lowered.input_names,
                    compile_ctx.lowered.input_shapes,
                    compile_ctx.lowered.input_types,
                )
                if not isinstance(value_type, TensorType)
            }
        active_reporter.step_ok(codegen_started)
        if args.verbose:
            _report_codegen_timings(active_reporter, timings=timings)
        artifacts = [("model.c", len(generated.encode("utf-8")))]
        if weight_data is not None:
            artifacts.append((f"{model_name}.bin", len(weight_data)))
        _report_generated_artifacts(active_reporter, artifacts=artifacts)
    except (CodegenError, ShapeInferenceError, UnsupportedOpError) as exc:
        active_reporter.step_fail(str(exc))
        return None, str(exc), operators, opset_version, None
    generated_checksum = _generated_checksum(generated)
    active_reporter.info(f"  Generated checksum (sha256): {generated_checksum}")
    expected_checksum = args.expected_checksum
    if expected_checksum and expected_checksum == generated_checksum:
        return "CHECKSUM", None, operators, opset_version, generated_checksum

    try:
        output_dtypes = {
            value.name: value.type.dtype
            for value in graph.outputs
            if isinstance(value.type, TensorType)
        }
        input_dtypes = {
            value.name: value.type.dtype
            for value in graph.inputs
            if isinstance(value.type, TensorType)
        }
    except (KeyError, UnsupportedOpError, ShapeInferenceError) as exc:
        return (
            None,
            f"Failed to resolve model dtype: {exc}",
            operators,
            opset_version,
            None,
        )

    def _cleanup_temp() -> None:
        if temp_dir is None and not cleanup_created_dir:
            return
        if temp_dir is None:
            shutil.rmtree(temp_path)
        else:
            temp_dir.cleanup()

    try:
        payload: dict[str, Any] | None = None
        testbench_input_path: Path | None = None
        if testbench_inputs:
            testbench_input_path = temp_path / "testbench_inputs.bin"
            with testbench_input_path.open("wb") as handle:
                for value in graph.inputs:
                    name = value.name
                    input_data = testbench_inputs.get(name)
                    if input_data is None:
                        return (
                            None,
                            f"Missing testbench input data for {name}.",
                            operators,
                            opset_version,
                            generated_checksum,
                        )
                    if isinstance(value.type, TensorType):
                        dtype = input_dtypes[name].np_dtype
                        if input_dtypes[name] == ScalarType.STRING:
                            blob = _serialize_string_tensor(input_data)
                        else:
                            blob = np.ascontiguousarray(
                                input_data.astype(dtype, copy=False)
                            ).tobytes(order="C")
                        if value.type.is_optional:
                            present = bool(
                                (testbench_optional_inputs or {}).get(name, True)
                            )
                            handle.write(struct.pack("<B", 1 if present else 0))
                            if not present:
                                continue
                        handle.write(blob)
                        continue
                    seq_type = value.type
                    seq_dtype = seq_type.elem.dtype.np_dtype
                    seq_data = np.asarray(input_data).astype(seq_dtype, copy=False)
                    if seq_data.ndim < 1:
                        return (
                            None,
                            f"Sequence input {name} must be at least 1D.",
                            operators,
                            opset_version,
                            generated_checksum,
                        )
                    present = bool((testbench_optional_inputs or {}).get(name, True))
                    if seq_type.is_optional:
                        handle.write(struct.pack("<B", 1 if present else 0))
                        if not present:
                            handle.write(struct.pack("<i", 0))
                            elem_shape = tuple(seq_type.elem.shape)
                            zero_payload = np.zeros((32, *elem_shape), dtype=seq_dtype)
                            handle.write(
                                np.ascontiguousarray(zero_payload).tobytes(order="C")
                            )
                            continue
                    seq_count = int(seq_data.shape[0])
                    handle.write(struct.pack("<i", seq_count))
                    elem_shape = lowered_sequence_input_shapes.get(
                        name, tuple(seq_type.elem.shape)
                    )
                    normalized = np.zeros((32, *elem_shape), dtype=seq_dtype)
                    limit = min(seq_count, 32)
                    if limit > 0:
                        for idx in range(limit):
                            item = np.asarray(seq_data[idx])
                            target = normalized[idx]
                            if target.ndim == item.ndim:
                                slices = tuple(
                                    slice(0, min(cur, tgt))
                                    for cur, tgt in zip(item.shape, target.shape)
                                )
                                if slices:
                                    target[slices] = item[slices]
                                elif item.ndim == 0:
                                    target[...] = item
                                else:
                                    flat_src = item.reshape(-1)
                                    flat_dst = target.reshape(-1)
                                    copy_count = min(flat_src.size, flat_dst.size)
                                    if copy_count:
                                        flat_dst[:copy_count] = flat_src[:copy_count]
                            else:
                                flat_src = item.reshape(-1)
                                flat_dst = target.reshape(-1)
                                copy_count = min(flat_src.size, flat_dst.size)
                                if copy_count:
                                    flat_dst[:copy_count] = flat_src[:copy_count]
                    handle.write(np.ascontiguousarray(normalized).tobytes(order="C"))
        c_path = temp_path / "model.c"
        weights_path = temp_path / f"{model_name}.bin"
        exe_path = temp_path / "model"
        c_path.write_text(generated, encoding="utf-8")
        if weight_data is not None:
            weights_path.write_bytes(weight_data)
        try:
            compile_cmd = [
                *compiler_cmd,
                "-std=c99",
                "-O1",
            ]
            sanitize_enabled = args.sanitize and not _env_flag(_DISABLE_SANITIZE_ENV)
            if args.sanitize and not sanitize_enabled:
                active_reporter.note(
                    f"--sanitize requested but disabled via {_DISABLE_SANITIZE_ENV}."
                )
            if sanitize_enabled:
                compile_cmd.append("-fsanitize=address,undefined")
            compile_cmd.extend(
                [
                    "-Wall",
                    "-Werror",
                    str(c_path.name),
                    "-o",
                    str(exe_path.name),
                    "-lm",
                ]
            )
            active_reporter.info("")
            compile_started = active_reporter.start_step("Compiling C code")
            subprocess.run(
                compile_cmd,
                check=True,
                capture_output=True,
                text=True,
                cwd=temp_path,
            )
            active_reporter.step_ok(compile_started)
            active_reporter.info(f"  Compile command: {shlex.join(compile_cmd)}")
            active_reporter.info("")
            if args.test_data_dir is not None:
                active_reporter.info(
                    f"Verifying using test data set: {args.test_data_dir.name}"
                )
            else:
                active_reporter.info("Verifying using generated random inputs")
        except subprocess.CalledProcessError as exc:
            message = "Failed to build testbench."
            if include_build_details:
                details = exc.stderr.strip()
                if details:
                    message = f"{message} {details}"
            active_reporter.step_fail(message)
            return None, message, operators, opset_version, generated_checksum
        try:
            run_started = active_reporter.start_step("  Running generated binary")
            run_cmd = [str(exe_path)]
            if testbench_input_path is not None:
                run_cmd.append(str(testbench_input_path))
            result = subprocess.run(
                run_cmd,
                check=True,
                capture_output=True,
                text=True,
                cwd=temp_path,
            )
            active_reporter.step_ok(run_started)
            result_json_path = temp_path / "testbench.json"
            result_json_path.write_text(result.stdout, encoding="utf-8")
            try:
                payload = json.loads(result_json_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                return (
                    None,
                    f"Failed to parse testbench JSON: {exc}",
                    operators,
                    opset_version,
                    generated_checksum,
                )
        except subprocess.CalledProcessError as exc:
            active_reporter.step_fail(describe_exit_code(exc.returncode))
            return (
                None,
                ("Testbench execution failed: " + describe_exit_code(exc.returncode)),
                operators,
                opset_version,
                generated_checksum,
            )
        if payload is None:
            return (
                None,
                "Failed to parse testbench JSON: missing output.",
                operators,
                opset_version,
                generated_checksum,
            )

        if requires_non_tensor_reference and testbench_outputs is None:
            return (
                None,
                "Verification for non-tensor outputs requires --test-data-dir.",
                operators,
                opset_version,
                generated_checksum,
            )

        if testbench_inputs:
            inputs = {}
            for value in graph.inputs:
                if not isinstance(value.type, TensorType):
                    continue
                values = testbench_inputs.get(value.name)
                if values is None:
                    return (
                        None,
                        f"Missing input {value.name} in testbench data.",
                        operators,
                        opset_version,
                        generated_checksum,
                    )
                inputs[value.name] = values.astype(
                    input_dtypes[value.name].np_dtype, copy=False
                )
        else:
            payload_inputs = payload.get("inputs", {})
            inputs = {}
            for value in graph.inputs:
                if not isinstance(value.type, TensorType):
                    continue
                input_payload = payload_inputs.get(value.name)
                if input_payload is None:
                    return (
                        None,
                        f"Missing input {value.name} in testbench data.",
                        operators,
                        opset_version,
                        generated_checksum,
                    )
                inputs[value.name] = decode_testbench_array(
                    input_payload["data"], input_dtypes[value.name].np_dtype
                )
        runtime_outputs: dict[str, np.ndarray] | None = None
        if testbench_outputs is not None:
            runtime_outputs = {}
            for value in graph.outputs:
                if not isinstance(value.type, TensorType):
                    continue
                output = testbench_outputs.get(value.name)
                if output is None:
                    continue
                runtime_outputs[value.name] = output.astype(
                    output_dtypes[value.name].np_dtype, copy=False
                )
        else:
            runtime_name = args.runtime
            custom_domains = sorted(
                {
                    opset.domain
                    for opset in model.opset_import
                    if opset.domain not in {"", "ai.onnx"}
                }
            )
            if runtime_name == "onnx-reference" and custom_domains:
                active_reporter.note(
                    "Runtime: switching to onnxruntime for custom domains "
                    f"{', '.join(custom_domains)}"
                )
                runtime_name = "onnxruntime"
            runtime_started = active_reporter.start_step(
                f"  Running {runtime_name} [--runtime={args.runtime}]"
            )
            try:
                if runtime_name == "onnxruntime":
                    import onnxruntime as ort

                    sess_options = make_deterministic_session_options(ort)
                    sess = ort.InferenceSession(
                        model.SerializeToString(),
                        sess_options=sess_options,
                        providers=["CPUExecutionProvider"],
                    )
                    runtime_outputs_list = sess.run(None, inputs)
                else:
                    from onnx.reference import ReferenceEvaluator

                    with deterministic_reference_runtime():
                        evaluator = ReferenceEvaluator(model)
                        runtime_outputs_list = evaluator.run(None, inputs)
            except Exception as exc:
                active_reporter.step_fail(str(exc))
                message = str(exc)
                if runtime_name == "onnxruntime" and "NOT_IMPLEMENTED" in message:
                    active_reporter.note(
                        f"Skipping verification for {model_path}: "
                        "ONNX Runtime does not support the model "
                        f"({message})"
                    )
                    return "", None, operators, opset_version, generated_checksum
                return (
                    None,
                    f"{runtime_name} failed to run {model_path}: {message}",
                    operators,
                    opset_version,
                    generated_checksum,
                )
            active_reporter.step_ok(runtime_started)
            runtime_outputs = {
                value.name: output
                for value, output in zip(graph.outputs, runtime_outputs_list)
            }
        nondeterministic_ops = sorted(
            set(operators).intersection(_NONDETERMINISTIC_OPERATORS)
        )
        if nondeterministic_ops:
            active_reporter.note(
                "Skipping output comparison for non-deterministic operator(s): "
                f"{', '.join(nondeterministic_ops)}"
            )
            return (
                "OK (non-deterministic output)",
                None,
                operators,
                opset_version,
                generated_checksum,
            )
        payload_outputs = payload.get("outputs", {})
        decoded_outputs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        output_nodes = {
            output_name: node for node in graph.nodes for output_name in node.outputs
        }
        metrics = _ComparisonMetrics()
        for value in graph.outputs:
            if not isinstance(value.type, TensorType):
                continue
            output_name = value.name
            runtime_out = runtime_outputs.get(output_name)
            output_payload = payload_outputs.get(output_name)
            if runtime_out is None or output_payload is None:
                continue
            info = output_dtypes.get(output_name)
            if info is None:
                continue
            output_data = decode_testbench_array(
                output_payload["data"], info.np_dtype
            ).astype(info.np_dtype, copy=False)
            runtime_cast = runtime_out.astype(info.np_dtype, copy=False)
            try:
                output_data = output_data.reshape(runtime_cast.shape)
            except ValueError:
                issue = _VerificationIssue.shape_mismatch(
                    output_name=output_name,
                    actual_shape=tuple(output_data.shape),
                    expected_shape=tuple(runtime_cast.shape),
                    actual_size=int(output_data.size),
                    expected_size=int(runtime_cast.size),
                )
                active_reporter.step_fail(issue.message)
                return (
                    None,
                    issue.format(),
                    operators,
                    opset_version,
                    generated_checksum,
                )
            decoded_outputs[output_name] = (output_data, runtime_cast)

        active_reporter.start_step(f"  Comparing outputs [--max-ulp={args.max_ulp}]")
        try:
            for value in graph.outputs:
                if value.name not in output_compare_names:
                    continue
                if isinstance(value.type, TensorType):
                    continue
                expected_sequence = (
                    None
                    if testbench_outputs is None
                    else testbench_outputs.get(value.name)
                )
                if not isinstance(expected_sequence, list):
                    raise AssertionError(
                        f"Missing sequence reference output for {value.name}."
                    )
                output_payload = payload_outputs.get(value.name)
                if output_payload is None:
                    issue = _VerificationIssue.missing_output(value.name)
                    raise AssertionError(issue.format())
                actual_data = decode_testbench_array(
                    output_payload["data"], value.type.elem.dtype.np_dtype
                )
                sequence_count = int(output_payload.get("sequence_count", 0))
                expected_count = len(expected_sequence)
                if sequence_count != expected_count:
                    active_reporter.note(
                        f"Sequence length differs for {value.name}: "
                        f"{sequence_count} vs {expected_count}."
                    )
                compare_count = min(sequence_count, expected_count)
                actual_sequence = [actual_data[index] for index in range(compare_count)]
                expected_trimmed_sequence = expected_sequence[:compare_count]
                for sequence_index, (actual_item, expected_item) in enumerate(
                    zip(actual_sequence, expected_trimmed_sequence)
                ):
                    if actual_item.ndim != expected_item.ndim:
                        active_reporter.note(
                            f"Skipping rank-mismatched sequence item {value.name}[{sequence_index}]: "
                            f"{actual_item.ndim} vs {expected_item.ndim}."
                        )
                        continue
                    overlap_shape = tuple(
                        min(actual_dim, expected_dim)
                        for actual_dim, expected_dim in zip(
                            actual_item.shape, expected_item.shape
                        )
                    )
                    if overlap_shape != expected_item.shape:
                        active_reporter.note(
                            f"Comparing overlapping region for {value.name}[{sequence_index}] "
                            f"with shapes {actual_item.shape} vs {expected_item.shape}."
                        )
                    slices = tuple(slice(0, dim) for dim in overlap_shape)
                    actual_trimmed = actual_item[slices]
                    expected_trimmed = expected_item[slices]
                    node = output_nodes.get(value.name)
                    node_name = node.name if node else None
                    if np.issubdtype(expected_item.dtype, np.floating):
                        output_max, output_worst = worst_ulp_diff(
                            actual_trimmed,
                            expected_trimmed,
                            atol_eps=args.atol_eps,
                        )
                        if output_worst is not None:
                            metrics.observe_ulp(
                                output_name=value.name,
                                node_name=node_name,
                                index=(sequence_index, *output_worst[0]),
                                got=float(output_worst[1]),
                                reference=float(output_worst[2]),
                                ulp=output_max,
                            )
                        else:
                            metrics.has_ulp = True
                    else:
                        output_max, output_worst = _worst_abs_diff(
                            actual_trimmed,
                            expected_trimmed,
                        )
                        if output_worst is not None:
                            metrics.observe_abs(
                                output_name=value.name,
                                node_name=node_name,
                                index=(sequence_index, *output_worst[0]),
                                got=output_worst[1],
                                reference=output_worst[2],
                                abs_diff=output_max,
                            )
                        else:
                            metrics.has_abs_diff = True

            for value in graph.outputs:
                if value.name not in output_compare_names:
                    continue
                if not isinstance(value.type, TensorType):
                    continue
                pair = decoded_outputs.get(value.name)
                if pair is None:
                    issue = _VerificationIssue.missing_output(value.name)
                    raise AssertionError(issue.format())
                output_data, runtime_out = pair
                node = output_nodes.get(value.name)
                node_name = node.name if node else None
                info = output_dtypes[value.name]
                if np.issubdtype(info.np_dtype, np.floating):
                    output_max, output_worst = worst_ulp_diff(
                        output_data,
                        runtime_out,
                        atol_eps=args.atol_eps,
                    )
                    if output_worst is not None:
                        metrics.observe_ulp(
                            output_name=value.name,
                            node_name=node_name,
                            index=output_worst[0],
                            got=float(output_worst[1]),
                            reference=float(output_worst[2]),
                            ulp=output_max,
                        )
                    else:
                        metrics.has_ulp = True
                else:
                    output_max, output_worst = _worst_abs_diff(output_data, runtime_out)
                    if output_worst is not None:
                        metrics.observe_abs(
                            output_name=value.name,
                            node_name=node_name,
                            index=output_worst[0],
                            got=output_worst[1],
                            reference=output_worst[2],
                            abs_diff=output_max,
                        )
                    else:
                        metrics.has_abs_diff = True
        except AssertionError as exc:
            active_reporter.step_fail(str(exc))
            return None, str(exc), operators, opset_version, generated_checksum
        failure_detail = metrics.failing_metric_detail(max_ulp_limit=args.max_ulp)
        if failure_detail is not None:
            active_reporter.step_fail(failure_detail)
        if metrics.has_abs_diff and metrics.max_abs_diff > 0:
            if metrics.worst_abs_diff is not None:
                node_label = metrics.worst_abs_diff.node_name or "(unknown)"
                index_display = ", ".join(
                    str(dim) for dim in metrics.worst_abs_diff.index
                )
                active_reporter.info(
                    "  Worst diff: output="
                    f"{metrics.worst_abs_diff.output_name} node={node_label} "
                    f"index=[{index_display}] "
                    f"got={metrics.worst_abs_diff.got} "
                    f"ref={metrics.worst_abs_diff.reference} "
                    f"abs_diff={metrics.worst_abs_diff.abs_diff}"
                )
            if args.per_node_accuracy:
                active_reporter.defer(
                    lambda: _report_per_node_accuracy(
                        active_reporter,
                        graph=graph,
                        decoded_outputs=decoded_outputs,
                        output_dtypes=output_dtypes,
                        atol_eps=args.atol_eps,
                        max_ulp_limit=args.max_ulp,
                    )
                )
            issue = _VerificationIssue.out_of_tolerance(
                metrics, max_ulp_limit=args.max_ulp
            )
            return (
                None,
                issue.format(),
                operators,
                opset_version,
                generated_checksum,
            )
        if metrics.has_ulp and metrics.max_ulp > args.max_ulp:
            if metrics.worst_diff is not None:
                node_label = metrics.worst_diff.node_name or "(unknown)"
                index_display = ", ".join(str(dim) for dim in metrics.worst_diff.index)
                active_reporter.info(
                    "  Worst diff: output="
                    f"{metrics.worst_diff.output_name} node={node_label} "
                    f"index=[{index_display}] "
                    f"got={metrics.worst_diff.got:.8g} "
                    f"ref={metrics.worst_diff.reference:.8g} "
                    f"ulp={metrics.worst_diff.ulp}"
                )
            if args.per_node_accuracy:
                active_reporter.defer(
                    lambda: _report_per_node_accuracy(
                        active_reporter,
                        graph=graph,
                        decoded_outputs=decoded_outputs,
                        output_dtypes=output_dtypes,
                        atol_eps=args.atol_eps,
                        max_ulp_limit=args.max_ulp,
                    )
                )
            issue = _VerificationIssue.out_of_tolerance(
                metrics, max_ulp_limit=args.max_ulp
            )
            return (
                None,
                issue.format(),
                operators,
                opset_version,
                generated_checksum,
            )
        active_reporter.step_ok_simple()
        active_reporter.info(f"    Output metrics: {metrics.format_metrics()}")
        if args.per_node_accuracy:
            active_reporter.defer(
                lambda: _report_per_node_accuracy(
                    active_reporter,
                    graph=graph,
                    decoded_outputs=decoded_outputs,
                    output_dtypes=output_dtypes,
                    atol_eps=args.atol_eps,
                    max_ulp_limit=args.max_ulp,
                )
            )
        return (
            metrics.format_ok_with_metrics(),
            None,
            operators,
            opset_version,
            generated_checksum,
        )
    finally:
        active_reporter.info("")
        _cleanup_temp()


def _load_test_data_inputs(
    model: onnx.ModelProto, data_dir: Path | None
) -> tuple[dict[str, "np.ndarray"] | None, dict[str, bool] | None, bool]:
    if data_dir is None:
        return None, None, False
    if not data_dir.exists():
        raise CodegenError(f"Test data directory not found: {data_dir}")
    input_files = sorted(
        data_dir.glob("input_*.pb"),
        key=lambda path: int(path.stem.split("_")[-1]),
    )
    if not input_files:
        raise CodegenError(f"No input_*.pb files found in {data_dir}")
    initializer_names = {init.name for init in model.graph.initializer}
    initializer_names.update(
        sparse_init.name for sparse_init in model.graph.sparse_initializer
    )
    model_inputs = [
        value_info
        for value_info in model.graph.input
        if value_info.name not in initializer_names
    ]
    if len(input_files) != len(model_inputs):
        raise CodegenError(
            "Test data input count does not match model inputs: "
            f"{len(input_files)} vs {len(model_inputs)}."
        )
    for value_info in model_inputs:
        value_kind = value_info.type.WhichOneof("value")
        if value_kind not in {"tensor_type", "optional_type", "sequence_type"}:
            LOGGER.warning(
                "Skipping test data load for unsupported input %s (type %s).",
                value_info.name,
                value_kind or "unknown",
            )
            return None, None, False
    inputs: dict[str, np.ndarray] = {}
    optional_flags: dict[str, bool] = {}
    reshaped_tensor_inputs = False

    def _sequence_tensor_values_to_array(
        *,
        name: str,
        tensors: list[np.ndarray],
        tensor_type: onnx.TypeProto.Tensor,
    ) -> np.ndarray:
        if not tensors:
            dtype_info = onnx._mapping.TENSOR_TYPE_MAP.get(tensor_type.elem_type)
            if dtype_info is None:
                raise CodegenError(f"Sequence input {name} has unsupported elem_type.")
            shape = [
                dim.dim_value if dim.HasField("dim_value") else 1
                for dim in tensor_type.shape.dim
            ]
            return np.zeros((0, *shape), dtype=dtype_info.np_dtype)
        first_shape = tensors[0].shape
        if any(tensor.shape != first_shape for tensor in tensors[1:]):
            if any(tensor.ndim != tensors[0].ndim for tensor in tensors[1:]):
                LOGGER.warning(
                    "Skipping sequence input %s with mixed tensor ranks in test data.",
                    name,
                )
                raise CodegenError(f"Sequence input {name} has mixed tensor ranks.")
            max_shape = tuple(
                max(tensor.shape[axis] for tensor in tensors)
                for axis in range(tensors[0].ndim)
            )
            target_shape: list[int] = []
            for axis, max_dim in enumerate(max_shape):
                if axis < len(tensor_type.shape.dim):
                    dim = tensor_type.shape.dim[axis]
                    if dim.HasField("dim_value") and dim.dim_value > 0:
                        target_shape.append(int(dim.dim_value))
                        continue
                target_shape.append(int(max_dim))
            LOGGER.warning(
                "Sequence test input %s has variable element shapes; "
                "normalizing to %s for testbench binary input.",
                name,
                tuple(target_shape),
            )
            normalized: list[np.ndarray] = []
            for tensor in tensors:
                item = np.zeros(tuple(target_shape), dtype=tensor.dtype)
                slices = tuple(
                    slice(0, min(cur_dim, tgt_dim))
                    for cur_dim, tgt_dim in zip(tensor.shape, target_shape)
                )
                item[slices] = tensor[slices]
                normalized.append(item)
            return np.stack(normalized, axis=0)
        return np.stack(tensors, axis=0)

    for index, path in enumerate(input_files):
        value_info = model_inputs[index]
        value_kind = value_info.type.WhichOneof("value")
        if value_kind == "tensor_type":
            tensor = onnx.TensorProto()
            tensor.ParseFromString(path.read_bytes())
            array = numpy_helper.to_array(tensor)
            tensor_type = value_info.type.tensor_type
            if tensor_type.HasField("shape"):
                target_shape: list[int] = []
                shape_known = True
                for dim in tensor_type.shape.dim:
                    if dim.HasField("dim_value"):
                        target_shape.append(int(dim.dim_value))
                    elif dim.HasField("dim_param"):
                        target_shape.append(1)
                    else:
                        shape_known = False
                        break
                if shape_known and target_shape:
                    expected = tuple(target_shape)
                    if array.shape != expected and array.size == int(
                        np.prod(target_shape, dtype=np.int64)
                    ):
                        LOGGER.warning(
                            "Reshaping test input %s from %s to %s to match model input.",
                            value_info.name,
                            array.shape,
                            expected,
                        )
                        array = array.reshape(expected)
                        reshaped_tensor_inputs = True
            inputs[value_info.name] = array
            continue
        if value_kind == "sequence_type":
            elem_type = value_info.type.sequence_type.elem_type
            if elem_type.WhichOneof("value") != "tensor_type":
                    return None, None, False
            seq = onnx.SequenceProto()
            seq.ParseFromString(path.read_bytes())
            tensors = [numpy_helper.to_array(tensor) for tensor in seq.tensor_values]
            inputs[value_info.name] = _sequence_tensor_values_to_array(
                name=value_info.name,
                tensors=tensors,
                tensor_type=elem_type.tensor_type,
            )
            continue
        optional = onnx.OptionalProto()
        optional.ParseFromString(path.read_bytes())
        elem_type = value_info.type.optional_type.elem_type
        elem_kind = elem_type.WhichOneof("value")
        if elem_kind == "tensor_type":
            tensor_type = elem_type.tensor_type
            if optional.HasField("tensor_value"):
                inputs[value_info.name] = numpy_helper.to_array(optional.tensor_value)
                optional_flags[value_info.name] = True
                continue
            if not tensor_type.HasField("elem_type"):
                raise CodegenError(
                    f"Optional input {value_info.name} is missing elem_type."
                )
            dtype_info = onnx._mapping.TENSOR_TYPE_MAP.get(tensor_type.elem_type)
            if dtype_info is None:
                raise CodegenError(
                    f"Optional input {value_info.name} has unsupported elem_type."
                )
            shape: list[int] = []
            for dim in tensor_type.shape.dim:
                if dim.HasField("dim_value"):
                    shape.append(dim.dim_value)
                elif dim.HasField("dim_param"):
                    shape.append(1)
                else:
                    raise CodegenError(
                        f"Optional input {value_info.name} has unknown shape."
                    )
            inputs[value_info.name] = np.zeros(tuple(shape), dtype=dtype_info.np_dtype)
            optional_flags[value_info.name] = False
            continue
        if elem_kind == "sequence_type":
            seq_elem = elem_type.sequence_type.elem_type
            if seq_elem.WhichOneof("value") != "tensor_type":
                LOGGER.warning(
                    "Skipping test data load for non-tensor optional sequence input %s.",
                    value_info.name,
                )
                return None, None, False
            if optional.HasField("sequence_value"):
                tensors = [
                    numpy_helper.to_array(tensor)
                    for tensor in optional.sequence_value.tensor_values
                ]
                inputs[value_info.name] = _sequence_tensor_values_to_array(
                    name=value_info.name,
                    tensors=tensors,
                    tensor_type=seq_elem.tensor_type,
                )
                optional_flags[value_info.name] = True
            else:
                inputs[value_info.name] = _sequence_tensor_values_to_array(
                    name=value_info.name,
                    tensors=[],
                    tensor_type=seq_elem.tensor_type,
                )
                optional_flags[value_info.name] = False
            continue
        LOGGER.warning(
            "Skipping test data load for unsupported optional input %s.",
            value_info.name,
        )
        return None, None, False
    return inputs, optional_flags, reshaped_tensor_inputs


def _load_test_data_outputs(
    model: onnx.ModelProto, data_dir: Path | None
) -> dict[str, "np.ndarray | list[np.ndarray]"] | None:
    if data_dir is None:
        return None
    if not data_dir.exists():
        raise CodegenError(f"Test data directory not found: {data_dir}")
    output_files = sorted(
        data_dir.glob("output_*.pb"),
        key=lambda path: int(path.stem.split("_")[-1]),
    )
    if not output_files:
        return None
    model_outputs = list(model.graph.output)
    if len(output_files) != len(model_outputs):
        raise CodegenError(
            "Test data output count does not match model outputs: "
            f"{len(output_files)} vs {len(model_outputs)}."
        )
    for value_info in model_outputs:
        value_kind = value_info.type.WhichOneof("value")
        if value_kind not in {"tensor_type", "sequence_type", "optional_type"}:
            LOGGER.warning(
                "Skipping test data load for unsupported output %s (type %s).",
                value_info.name,
                value_kind or "unknown",
            )
            return None
    outputs: dict[str, np.ndarray | list[np.ndarray]] = {}
    for index, path in enumerate(output_files):
        value_info = model_outputs[index]
        value_kind = value_info.type.WhichOneof("value")
        if value_kind == "tensor_type":
            tensor = onnx.TensorProto()
            tensor.ParseFromString(path.read_bytes())
            outputs[value_info.name] = numpy_helper.to_array(tensor)
            continue
        if value_kind == "sequence_type":
            seq = onnx.SequenceProto()
            seq.ParseFromString(path.read_bytes())
            outputs[value_info.name] = [
                numpy_helper.to_array(tensor) for tensor in seq.tensor_values
            ]
            continue
        optional = onnx.OptionalProto()
        optional.ParseFromString(path.read_bytes())
        elem_kind = value_info.type.optional_type.elem_type.WhichOneof("value")
        if elem_kind == "sequence_type":
            if optional.HasField("sequence_value"):
                outputs[value_info.name] = [
                    numpy_helper.to_array(tensor)
                    for tensor in optional.sequence_value.tensor_values
                ]
            else:
                outputs[value_info.name] = []
            continue
        if elem_kind == "tensor_type":
            if optional.HasField("tensor_value"):
                outputs[value_info.name] = numpy_helper.to_array(optional.tensor_value)
            else:
                tensor_type = value_info.type.optional_type.elem_type.tensor_type
                dtype_info = onnx._mapping.TENSOR_TYPE_MAP.get(tensor_type.elem_type)
                if dtype_info is None:
                    return None
                shape = [
                    dim.dim_value if dim.HasField("dim_value") else 1
                    for dim in tensor_type.shape.dim
                ]
                outputs[value_info.name] = np.zeros(shape, dtype=dtype_info.np_dtype)
            continue
        return None
    return outputs


def _format_command_line(argv: Sequence[str] | None) -> str:
    if argv is None:
        argv = sys.argv
    args = [str(arg) for arg in argv[1:]]
    if not args:
        return ""
    filtered: list[str] = []
    skip_next = False
    for arg in args:
        if skip_next:
            skip_next = False
            continue
        if arg == "--expected-checksum":
            skip_next = True
            continue
        if arg.startswith("--expected-checksum="):
            continue
        filtered.append(arg.replace("\\", "/"))
    if not filtered:
        return ""
    return shlex.join(filtered)


def _load_model_and_checksum(
    model_path: Path,
) -> tuple[onnx.ModelProto, str]:
    model_bytes = model_path.read_bytes()
    digest = hashlib.sha256()
    digest.update(model_bytes)
    model = onnx.load_model_from_string(model_bytes)
    return model, digest.hexdigest()


def _generated_checksum(generated: str) -> str:
    digest = hashlib.sha256()
    digest.update(generated.encode("utf-8"))
    return digest.hexdigest()


def _report_model_details(
    reporter: _VerifyReporter,
    *,
    model_path: Path,
    model_checksum: str,
    operators: Sequence[str],
    opset_version: int | None,
    node_count: int,
    initializer_count: int,
    input_count: int,
    output_count: int,
) -> None:
    operators_display = ", ".join(operators) if operators else "(none)"
    reporter.info(f"  Model operators ({len(operators)}): {operators_display}")
    reporter.info(
        f"  Model file size: {_format_artifact_size(model_path.stat().st_size)}"
    )
    reporter.info(f"  Model checksum (sha256): {model_checksum}")
    if opset_version is not None:
        reporter.info(f"  Opset version: {opset_version}")
    reporter.info(
        "  Counts: "
        f"nodes={node_count}, "
        f"initializers={initializer_count}, "
        f"inputs={input_count}, "
        f"outputs={output_count}"
    )


def _report_codegen_timings(
    reporter: _VerifyReporter, *, timings: Mapping[str, float]
) -> None:
    if not timings:
        return
    order = [
        ("import_onnx", "import"),
        ("concretize_shapes", "concretize"),
        ("collect_variable_dims", "var_dims"),
        ("lower_model", "lower"),
        ("emit_model", "emit"),
        ("emit_model_with_data_file", "emit_data"),
        ("collect_weight_data", "weights"),
    ]
    seen = set()
    parts: list[str] = []
    for key, label in order:
        if key not in timings:
            continue
        parts.append(f"{label}={timings[key]:.3f}s")
        seen.add(key)
    for key in sorted(k for k in timings if k not in seen):
        parts.append(f"{key}={timings[key]:.3f}s")
    reporter.info(f"  Codegen timing: {', '.join(parts)}")


def _collect_model_operators(model: onnx.ModelProto) -> list[str]:
    operators: list[str] = []
    seen: set[str] = set()
    for node in model.graph.node:
        op_name = f"{node.domain}::{node.op_type}" if node.domain else node.op_type
        if op_name in seen:
            continue
        seen.add(op_name)
        operators.append(op_name)
    return operators


def _model_opset_version(model: onnx.ModelProto, *, domain: str = "") -> int | None:
    if not model.opset_import:
        return None
    domains = (domain,) if domain else ("", "ai.onnx")
    for target_domain in domains:
        for opset in model.opset_import:
            if opset.domain == target_domain:
                return opset.version
    return None

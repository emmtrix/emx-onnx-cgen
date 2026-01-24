from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence, TextIO

import numpy as np
import onnx
from onnx import numpy_helper

from shared.ulp import ulp_intdiff_float

from ._build_info import BUILD_DATE, GIT_VERSION
from .compiler import Compiler, CompilerOptions
from .errors import CodegenError, ShapeInferenceError, UnsupportedOpError
from .onnx_import import import_onnx
from .onnxruntime_utils import make_deterministic_session_options
from .testbench import decode_testbench_array
from .verification import format_success_message

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CliResult:
    exit_code: int
    command_line: str
    error: str | None = None
    success_message: str | None = None
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


class _VerifyReporter:
    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream or sys.stdout

    def start_step(self, label: str) -> float:
        print(f"{label} ...", end=" ", file=self._stream, flush=True)
        return time.perf_counter()

    def step_ok(self, started_at: float) -> None:
        duration = time.perf_counter() - started_at
        print(f"OK ({duration:.3f}s)", file=self._stream)

    def step_ok_detail(self, detail: str) -> None:
        print(f"OK ({detail})", file=self._stream)

    def step_fail(self, reason: str) -> None:
        print(f"FAIL ({reason})", file=self._stream)

    def note(self, message: str) -> None:
        print(f"Note: {message}", file=self._stream)

    def info(self, message: str) -> None:
        print(message, file=self._stream)


class _NullVerifyReporter(_VerifyReporter):
    def __init__(self) -> None:
        super().__init__(stream=sys.stdout)

    def start_step(self, label: str) -> float:
        return time.perf_counter()

    def step_ok(self, started_at: float) -> None:
        return None

    def step_ok_detail(self, detail: str) -> None:
        return None

    def step_fail(self, reason: str) -> None:
        return None

    def note(self, message: str) -> None:
        return None

    def info(self, message: str) -> None:
        return None


def _worst_ulp_diff(
    actual: "np.ndarray", expected: "np.ndarray"
) -> tuple[int, tuple[tuple[int, ...], float, float] | None]:
    if actual.shape != expected.shape:
        raise ValueError(
            f"Shape mismatch for ULP calculation: {actual.shape} vs {expected.shape}"
        )
    if not np.issubdtype(expected.dtype, np.floating):
        return 0, None
    dtype = expected.dtype
    actual_cast = actual.astype(dtype, copy=False)
    expected_cast = expected.astype(dtype, copy=False)
    max_diff = 0
    worst: tuple[tuple[int, ...], float, float] | None = None
    iterator = np.nditer(
        [actual_cast, expected_cast], flags=["refs_ok", "multi_index"]
    )
    for actual_value, expected_value in iterator:
        diff = ulp_intdiff_float(actual_value[()], expected_value[()])
        if diff > max_diff:
            max_diff = diff
            worst = (
                iterator.multi_index,
                float(actual_value[()]),
                float(expected_value[()]),
            )
    return max_diff, worst


def run_cli_command(
    argv: Sequence[str],
    *,
    testbench_inputs: Mapping[str, "np.ndarray"] | None = None,
) -> CliResult:
    raw_argv = list(argv)
    parse_argv = raw_argv
    if raw_argv and raw_argv[0] == "emx-onnx-cgen":
        parse_argv = raw_argv[1:]
    parser = _build_parser()
    args = parser.parse_args(parse_argv)
    args.command_line = _format_command_line(raw_argv)

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
                error=error,
                success_message=success_message,
                operators=operators,
                opset_version=opset_version,
                generated_checksum=generated_checksum,
            )
        generated, data_source, error = _compile_model(
            args, testbench_inputs=testbench_inputs
        )
        if error:
            return CliResult(
                exit_code=1,
                command_line=args.command_line,
                error=error,
            )
        return CliResult(
            exit_code=0,
            command_line=args.command_line,
            success_message="",
            generated=generated,
            data_source=data_source,
        )
    except Exception as exc:  # pragma: no cover - defensive reporting
        LOGGER.exception("Unhandled exception while running CLI command.")
        return CliResult(
            exit_code=1,
            command_line=args.command_line,
            error=str(exc),
        )


def _build_parser() -> argparse.ArgumentParser:
    description = (
        "emmtrix ONNX-to-C Code Generator "
        f"(build date: {BUILD_DATE}, git: {GIT_VERSION})"
    )
    parser = argparse.ArgumentParser(prog="emx-onnx-cgen", description=description)
    subparsers = parser.add_subparsers(dest="command", required=True)

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

    compile_parser = subparsers.add_parser(
        "compile", help="Compile an ONNX model into C source"
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
            "\"...\" placeholders (default: no truncation)"
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

    verify_parser = subparsers.add_parser(
        "verify",
        help="Compile an ONNX model and verify outputs against ONNX Runtime",
    )
    verify_parser.add_argument("model", type=Path, help="Path to the ONNX model")
    verify_parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Override the generated model name (default: model file stem)",
    )
    verify_parser.add_argument(
        "--cc",
        type=str,
        default=None,
        help="C compiler command to build the testbench binary",
    )
    verify_parser.add_argument(
        "--truncate-weights-after",
        type=int,
        default=None,
        help=(
            "Truncate inline weight initializers after N values and insert "
            "\"...\" placeholders (default: no truncation)"
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
        "--temp-dir",
        type=Path,
        default=None,
        help=(
            "Directory in which to create temporary verification files "
            "(default: system temp dir)"
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
        "--runtime",
        choices=("onnxruntime", "onnx-reference"),
        default="onnx-reference",
        help=(
            "Runtime backend for verification (default: onnx-reference; "
            "options: onnxruntime, onnx-reference)"
        ),
    )
    verify_parser.add_argument(
        "--expected-checksum",
        type=str,
        default=None,
        help=(
            "Expected generated C checksum (sha256). When it matches the "
            "computed checksum, verification exits early with OK CHECKSUM."
        ),
    )
    add_restrict_flags(verify_parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO)
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.command_line = _format_command_line(argv)

    if args.command == "compile":
        return _handle_compile(args)
    if args.command == "verify":
        return _handle_verify(args)
    parser.error(f"Unknown command {args.command}")
    return 1


def _handle_compile(args: argparse.Namespace) -> int:
    model_path: Path = args.model
    output_path: Path = args.output or model_path.with_suffix(".c")
    model_name = args.model_name or "model"
    generated, data_source, weight_data, error = _compile_model(args)
    if error:
        LOGGER.error("Failed to compile %s: %s", model_path, error)
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(generated or "", encoding="utf-8")
    LOGGER.info("Wrote C source to %s", output_path)
    if data_source is not None:
        data_path = output_path.with_name(
            f"{output_path.stem}_data{output_path.suffix}"
        )
        data_path.write_text(data_source, encoding="utf-8")
        LOGGER.info("Wrote data source to %s", data_path)
    if weight_data is not None:
        weights_path = output_path.with_name(f"{model_name}.bin")
        weights_path.write_bytes(weight_data)
        LOGGER.info("Wrote weights binary to %s", weights_path)
    return 0


def _compile_model(
    args: argparse.Namespace,
    *,
    testbench_inputs: Mapping[str, "np.ndarray"] | None = None,
) -> tuple[str | None, str | None, bytes | None, str | None]:
    model_path: Path = args.model
    model_name = args.model_name or "model"
    try:
        model_checksum = _model_checksum(model_path)
        model = onnx.load_model(model_path)
        options = CompilerOptions(
            model_name=model_name,
            emit_testbench=args.emit_testbench,
            command_line=args.command_line,
            model_checksum=model_checksum,
            restrict_arrays=args.restrict_arrays,
            truncate_weights_after=args.truncate_weights_after,
            large_temp_threshold_bytes=args.large_temp_threshold_bytes,
            large_weight_threshold=args.large_weight_threshold,
            testbench_inputs=testbench_inputs,
        )
        compiler = Compiler(options)
        if args.emit_data_file:
            generated, data_source, weight_data = (
                compiler.compile_with_data_file_and_weight_data(model)
            )
        else:
            generated, weight_data = compiler.compile_with_weight_data(model)
            data_source = None
    except (OSError, CodegenError, ShapeInferenceError, UnsupportedOpError) as exc:
        return None, None, None, str(exc)
    return generated, data_source, weight_data, None


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
        resolved = shutil.which(candidate)
        if resolved:
            return maybe_prefix_ccache([resolved])
    return None


def _handle_verify(args: argparse.Namespace) -> int:
    reporter = _VerifyReporter()
    (
        success_message,
        error,
        _operators,
        _opset_version,
        generated_checksum,
    ) = _verify_model(args, include_build_details=True, reporter=reporter)
    if error is not None:
        reporter.info("")
        reporter.info(f"Result: {error}")
        return 1
    if success_message:
        reporter.info("")
        reporter.info(f"Result: {success_message}")
    if generated_checksum:
        reporter.note(f"Generated checksum (sha256): {generated_checksum}")
    return 0


def _verify_model(
    args: argparse.Namespace,
    *,
    include_build_details: bool,
    reporter: _VerifyReporter | None = None,
) -> tuple[str | None, str | None, list[str], int | None, str | None]:
    active_reporter = reporter or _NullVerifyReporter()

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
    model_name = args.model_name or "model"
    model_checksum = _model_checksum(model_path)
    compiler_cmd = _resolve_compiler(args.cc, prefer_ccache=False)
    if compiler_cmd is None:
        return (
            None,
            "No C compiler found (set --cc or CC environment variable).",
            [],
            None,
            None,
        )
    load_started = active_reporter.start_step(f"Loading model {model_path.name}")
    try:
        model = onnx.load_model(model_path)
    except OSError as exc:
        active_reporter.step_fail(str(exc))
        return None, str(exc), [], None, None
    active_reporter.step_ok(load_started)

    operators = _collect_model_operators(model)
    opset_version = _model_opset_version(model)
    operators_display = ", ".join(operators) if operators else "(none)"
    active_reporter.note(
        f"Model operators ({len(operators)}): {operators_display}"
    )

    try:
        codegen_started = active_reporter.start_step("Generating C code")
        testbench_inputs = _load_test_data_inputs(model, args.test_data_dir)
        options = CompilerOptions(
            model_name=model_name,
            emit_testbench=True,
            command_line=args.command_line,
            model_checksum=model_checksum,
            restrict_arrays=args.restrict_arrays,
            truncate_weights_after=args.truncate_weights_after,
            large_temp_threshold_bytes=args.large_temp_threshold_bytes,
            large_weight_threshold=args.large_weight_threshold,
            testbench_inputs=testbench_inputs,
        )
        compiler = Compiler(options)
        generated, weight_data = compiler.compile_with_weight_data(model)
        active_reporter.step_ok(codegen_started)
    except (CodegenError, ShapeInferenceError, UnsupportedOpError) as exc:
        active_reporter.step_fail(str(exc))
        return None, str(exc), operators, opset_version, None
    generated_checksum = _generated_checksum(generated)
    expected_checksum = args.expected_checksum
    if expected_checksum and expected_checksum == generated_checksum:
        return "OK CHECKSUM", None, operators, opset_version, generated_checksum

    try:
        graph = import_onnx(model)
        output_dtypes = {value.name: value.type.dtype for value in graph.outputs}
        input_dtypes = {value.name: value.type.dtype for value in graph.inputs}
    except (KeyError, UnsupportedOpError, ShapeInferenceError) as exc:
        return (
            None,
            f"Failed to resolve model dtype: {exc}",
            operators,
            opset_version,
            None,
        )

    temp_dir_root: Path | None = args.temp_dir
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
    temp_dir: tempfile.TemporaryDirectory | None = None
    if args.keep_temp_dir:
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
    active_reporter.info(f"Using temporary folder: {temp_path}")
    try:
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
                "-O2",
                "-Wall",
                "-Werror",
                str(c_path),
                "-o",
                str(exe_path),
                "-lm",
            ]
            active_reporter.note(f"Compile command: {shlex.join(compile_cmd)}")
            compile_started = active_reporter.start_step("Compiling C code")
            subprocess.run(
                compile_cmd,
                check=True,
                capture_output=True,
                text=True,
            )
            active_reporter.step_ok(compile_started)
        except subprocess.CalledProcessError as exc:
            message = "Failed to build testbench."
            if include_build_details:
                details = exc.stderr.strip()
                if details:
                    message = f"{message} {details}"
            active_reporter.step_fail(message)
            return None, message, operators, opset_version, generated_checksum
        try:
            run_started = active_reporter.start_step("Running generated binary")
            result = subprocess.run(
                [str(exe_path)],
                check=True,
                capture_output=True,
                text=True,
                cwd=temp_path,
            )
            active_reporter.step_ok(run_started)
        except subprocess.CalledProcessError as exc:
            active_reporter.step_fail(describe_exit_code(exc.returncode))
            return None, (
                "Testbench execution failed: " + describe_exit_code(exc.returncode)
            ), operators, opset_version, generated_checksum
    finally:
        if temp_dir is None:
            active_reporter.note(f"Keeping temporary folder: {temp_path}")
        else:
            delete_started = active_reporter.start_step(
                "Deleting temporary folder: "
                f"{temp_path} [--keep-temp-dir not set]"
            )
            temp_dir.cleanup()
            active_reporter.step_ok(delete_started)
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return (
            None,
            f"Failed to parse testbench JSON: {exc}",
            operators,
            opset_version,
            generated_checksum,
        )

    if testbench_inputs:
        inputs = {
            name: values.astype(input_dtypes[name].np_dtype, copy=False)
            for name, values in testbench_inputs.items()
        }
    else:
        inputs = {
            name: decode_testbench_array(
                value["data"], input_dtypes[name].np_dtype
            )
            for name, value in payload["inputs"].items()
        }
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
    runtime_started = active_reporter.start_step(f"Running {runtime_name}")
    try:
        if runtime_name == "onnxruntime":
            import onnxruntime as ort

            sess_options = make_deterministic_session_options(ort)
            sess = ort.InferenceSession(
                model.SerializeToString(),
                sess_options=sess_options,
                providers=["CPUExecutionProvider"],
            )
            runtime_outputs = sess.run(None, inputs)
        else:
            from onnx.reference import ReferenceEvaluator

            evaluator = ReferenceEvaluator(model)
            runtime_outputs = evaluator.run(None, inputs)
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
    payload_outputs = payload.get("outputs", {})
    max_ulp = 0
    worst_diff: _WorstDiff | None = None
    output_nodes = {
        output_name: node
        for node in graph.nodes
        for output_name in node.outputs
    }
    active_reporter.start_step(
        f"Verifying outputs [--max-ulp={args.max_ulp}]"
    )
    try:
        for value, runtime_out in zip(graph.outputs, runtime_outputs):
            output_payload = payload_outputs.get(value.name)
            if output_payload is None:
                raise AssertionError(f"Missing output {value.name} in testbench data")
            info = output_dtypes[value.name]
            output_data = decode_testbench_array(
                output_payload["data"], info.np_dtype
            ).astype(info.np_dtype, copy=False)
            runtime_out = runtime_out.astype(info.np_dtype, copy=False)
            output_data = output_data.reshape(runtime_out.shape)
            if np.issubdtype(info.np_dtype, np.floating):
                output_max, output_worst = _worst_ulp_diff(
                    output_data, runtime_out
                )
                if output_max > max_ulp:
                    max_ulp = output_max
                    if output_worst is not None:
                        node = output_nodes.get(value.name)
                        worst_diff = _WorstDiff(
                            output_name=value.name,
                            node_name=node.name if node else None,
                            index=output_worst[0],
                            got=float(output_worst[1]),
                            reference=float(output_worst[2]),
                            ulp=output_max,
                        )
            else:
                np.testing.assert_array_equal(output_data, runtime_out)
    except AssertionError as exc:
        active_reporter.step_fail(str(exc))
        return None, str(exc), operators, opset_version, generated_checksum
    if max_ulp > args.max_ulp:
        active_reporter.step_fail(f"max ULP {max_ulp}")
        if worst_diff is not None:
            node_label = worst_diff.node_name or "(unknown)"
            index_display = ", ".join(str(dim) for dim in worst_diff.index)
            active_reporter.info(
                "  Worst diff: output="
                f"{worst_diff.output_name} node={node_label} "
                f"index=[{index_display}] "
                f"got={worst_diff.got:.8g} "
                f"ref={worst_diff.reference:.8g} "
                f"ulp={worst_diff.ulp}"
            )
        return (
            None,
            f"Out of tolerance (max ULP {max_ulp})",
            operators,
            opset_version,
            generated_checksum,
        )
    active_reporter.step_ok_detail(f"max ULP {max_ulp}")
    return (
        format_success_message(max_ulp),
        None,
        operators,
        opset_version,
        generated_checksum,
    )


def _load_test_data_inputs(
    model: onnx.ModelProto, data_dir: Path | None
) -> dict[str, "np.ndarray"] | None:
    if data_dir is None:
        return None
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
        if value_kind != "tensor_type":
            LOGGER.warning(
                "Skipping test data load for non-tensor input %s (type %s).",
                value_info.name,
                value_kind or "unknown",
            )
            return None
    inputs: dict[str, np.ndarray] = {}
    for index, path in enumerate(input_files):
        tensor = onnx.TensorProto()
        tensor.ParseFromString(path.read_bytes())
        inputs[model_inputs[index].name] = numpy_helper.to_array(tensor)
    return inputs


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
        filtered.append(arg)
    if not filtered:
        return ""
    return shlex.join(filtered)


def _model_checksum(model_path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(model_path.read_bytes())
    return digest.hexdigest()


def _generated_checksum(generated: str) -> str:
    digest = hashlib.sha256()
    digest.update(generated.encode("utf-8"))
    return digest.hexdigest()


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

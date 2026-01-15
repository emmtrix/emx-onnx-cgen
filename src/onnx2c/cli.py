from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence

import onnx

from ._build_info import BUILD_DATE, GIT_VERSION
from .compiler import Compiler, CompilerOptions
from .errors import CodegenError, ShapeInferenceError, UnsupportedOpError
from .onnx_import import import_onnx

LOGGER = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    description = (
        "emmtrix ONNX to C Compiler "
        f"(build date: {BUILD_DATE}, git: {GIT_VERSION})"
    )
    parser = argparse.ArgumentParser(prog="onnx2c", description=description)
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
    compile_parser.add_argument("output", type=Path, help="Output C file path")
    compile_parser.add_argument(
        "--template-dir",
        type=Path,
        default=Path("templates"),
        help="Template directory (default: templates)",
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
    add_restrict_flags(compile_parser)

    verify_parser = subparsers.add_parser(
        "verify",
        help="Compile an ONNX model and verify outputs against ONNX Runtime",
    )
    verify_parser.add_argument("model", type=Path, help="Path to the ONNX model")
    verify_parser.add_argument(
        "--template-dir",
        type=Path,
        default=Path("templates"),
        help="Template directory (default: templates)",
    )
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
    output_path: Path = args.output
    model_name = args.model_name or output_path.stem
    try:
        model_checksum = _model_checksum(model_path)
        model = onnx.load_model(model_path)
        options = CompilerOptions(
            template_dir=args.template_dir,
            model_name=model_name,
            emit_testbench=args.emit_testbench,
            command_line=args.command_line,
            model_checksum=model_checksum,
            restrict_arrays=args.restrict_arrays,
        )
        compiler = Compiler(options)
        if args.emit_data_file:
            generated, data_source = compiler.compile_with_data_file(model)
        else:
            generated = compiler.compile(model)
            data_source = None
    except (OSError, CodegenError, ShapeInferenceError, UnsupportedOpError) as exc:
        LOGGER.error("Failed to compile %s: %s", model_path, exc)
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(generated, encoding="utf-8")
    LOGGER.info("Wrote C source to %s", output_path)
    if data_source is not None:
        data_path = output_path.with_name(
            f"{output_path.stem}_data{output_path.suffix}"
        )
        data_path.write_text(data_source, encoding="utf-8")
        LOGGER.info("Wrote data source to %s", data_path)
    return 0


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
    import numpy as np
    import onnxruntime as ort

    model_path: Path = args.model
    model_name = args.model_name or model_path.stem
    model_checksum = _model_checksum(model_path)
    compiler_cmd = _resolve_compiler(args.cc, prefer_ccache=False)
    if compiler_cmd is None:
        LOGGER.error("No C compiler found (set --cc or CC environment variable).")
        return 1
    try:
        model = onnx.load_model(model_path)
        options = CompilerOptions(
            template_dir=args.template_dir,
            model_name=model_name,
            emit_testbench=True,
            command_line=args.command_line,
            model_checksum=model_checksum,
            restrict_arrays=args.restrict_arrays,
        )
        compiler = Compiler(options)
        generated = compiler.compile(model)
    except (OSError, CodegenError, ShapeInferenceError, UnsupportedOpError) as exc:
        LOGGER.error("Failed to compile %s: %s", model_path, exc)
        return 1

    try:
        graph = import_onnx(model)
        output_dtypes = {value.name: value.type.dtype for value in graph.outputs}
        input_dtypes = {value.name: value.type.dtype for value in graph.inputs}
    except (KeyError, UnsupportedOpError, ShapeInferenceError) as exc:
        LOGGER.error("Failed to resolve model dtype: %s", exc)
        return 1

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        c_path = temp_path / "model.c"
        exe_path = temp_path / "model"
        c_path.write_text(generated, encoding="utf-8")
        try:
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
        except subprocess.CalledProcessError as exc:
            LOGGER.error("Failed to build testbench: %s", exc.stderr.strip())
            return 1
        try:
            result = subprocess.run(
                [str(exe_path)],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            LOGGER.error("Testbench execution failed: %s", exc.stderr.strip())
            return 1

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        LOGGER.error("Failed to parse testbench JSON: %s", exc)
        return 1

    inputs = {
        name: np.array(value["data"], dtype=input_dtypes[name].np_dtype)
        for name, value in payload["inputs"].items()
    }
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    try:
        ort_outputs = sess.run(None, inputs)
    except Exception as exc:
        message = str(exc)
        if "NOT_IMPLEMENTED" in message:
            LOGGER.warning(
                "Skipping verification for %s: ONNX Runtime does not support the model (%s)",
                model_path,
                message,
            )
            return 0
        LOGGER.error("ONNX Runtime failed to run %s: %s", model_path, message)
        return 1
    payload_outputs = payload.get("outputs", {})
    try:
        for value, ort_out in zip(graph.outputs, ort_outputs):
            output_payload = payload_outputs.get(value.name)
            if output_payload is None:
                raise AssertionError(f"Missing output {value.name} in testbench data")
            info = output_dtypes[value.name]
            output_data = np.array(output_payload["data"], dtype=info.np_dtype)
            if np.issubdtype(info.np_dtype, np.floating):
                np.testing.assert_allclose(
                    output_data, ort_out, rtol=1e-4, atol=1e-5
                )
            else:
                np.testing.assert_array_equal(output_data, ort_out)
    except AssertionError as exc:
        LOGGER.error("Verification failed: %s", exc)
        return 1
    LOGGER.info("Verification succeeded for %s", model_path)
    return 0


def _format_command_line(argv: Sequence[str] | None) -> str:
    if argv is None:
        argv = sys.argv
    return shlex.join([str(arg) for arg in argv])


def _model_checksum(model_path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(model_path.read_bytes())
    return digest.hexdigest()

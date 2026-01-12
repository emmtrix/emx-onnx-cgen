from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence

import onnx

from .compiler import Compiler, CompilerOptions
from .dtypes import dtype_info
from .errors import CodegenError, ShapeInferenceError, UnsupportedOpError
from .onnx_import import import_onnx

LOGGER = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="onnx2c", description="ONNX to C compiler")
    subparsers = parser.add_subparsers(dest="command", required=True)

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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO)
    parser = _build_parser()
    args = parser.parse_args(argv)

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
        model = onnx.load_model(model_path)
        options = CompilerOptions(
            template_dir=args.template_dir,
            model_name=model_name,
            emit_testbench=args.emit_testbench,
        )
        compiler = Compiler(options)
        generated = compiler.compile(model)
    except (OSError, CodegenError, ShapeInferenceError, UnsupportedOpError) as exc:
        LOGGER.error("Failed to compile %s: %s", model_path, exc)
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(generated, encoding="utf-8")
    LOGGER.info("Wrote C source to %s", output_path)
    return 0


def _resolve_compiler(cc: str | None) -> str | None:
    if cc:
        return cc
    env_cc = os.environ.get("CC")
    if env_cc:
        return env_cc
    return shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")


def _handle_verify(args: argparse.Namespace) -> int:
    import numpy as np
    import onnxruntime as ort

    model_path: Path = args.model
    model_name = args.model_name or model_path.stem
    compiler_cmd = _resolve_compiler(args.cc)
    if compiler_cmd is None:
        LOGGER.error("No C compiler found (set --cc or CC environment variable).")
        return 1
    try:
        model = onnx.load_model(model_path)
        options = CompilerOptions(
            template_dir=args.template_dir,
            model_name=model_name,
            emit_testbench=True,
        )
        compiler = Compiler(options)
        generated = compiler.compile(model)
    except (OSError, CodegenError, ShapeInferenceError, UnsupportedOpError) as exc:
        LOGGER.error("Failed to compile %s: %s", model_path, exc)
        return 1

    try:
        graph = import_onnx(model)
        output_dtypes = {
            value.name: dtype_info(value.type.dtype) for value in graph.outputs
        }
        input_dtypes = {
            value.name: dtype_info(value.type.dtype) for value in graph.inputs
        }
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
                    compiler_cmd,
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
    ort_outputs = sess.run(None, inputs)
    payload_outputs = payload.get("outputs", {})
    try:
        for value, ort_out in zip(graph.outputs, ort_outputs):
            output_payload = payload_outputs.get(value.name)
            if output_payload is None:
                raise AssertionError(f"Missing output {value.name} in testbench data")
            info = output_dtypes[value.name]
            output_data = np.array(output_payload["data"], dtype=info.np_dtype)
            if value.type.dtype == "float":
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

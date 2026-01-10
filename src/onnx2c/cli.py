from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import onnx

from .compiler import Compiler, CompilerOptions
from .errors import CodegenError, ShapeInferenceError, UnsupportedOpError

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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO)
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "compile":
        return _handle_compile(args)
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

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import onnx

from onnx import TensorProto

from test_ops import (
    _make_operator_model,
    _make_reduce_model,
    _reduce_output_shape,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"


def _run_cli_verify(model: onnx.ModelProto) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = Path(temp_dir) / "model.onnx"
        onnx.save_model(model, model_path)
        env = os.environ.copy()
        python_path = str(SRC_ROOT)
        if env.get("PYTHONPATH"):
            python_path = f"{python_path}{os.pathsep}{env['PYTHONPATH']}"
        env["PYTHONPATH"] = python_path
        subprocess.run(
            [
                sys.executable,
                "-m",
                "emx_onnx_cgen",
                "verify",
                str(model_path),
                "--temp-dir-root",
                str(temp_dir),
            ],
            check=True,
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            env=env,
        )


def test_cli_verify_operator_model() -> None:
    model = _make_operator_model(
        op_type="Add",
        input_shapes=[[2, 3], [2, 3]],
        output_shape=[2, 3],
        dtype=TensorProto.FLOAT,
        attrs={},
    )
    _run_cli_verify(model)


def test_cli_verify_reduce_model() -> None:
    output_shape = _reduce_output_shape([2, 3, 4], [1], 1)
    model = _make_reduce_model(
        op_type="ReduceSum",
        input_shape=[2, 3, 4],
        output_shape=output_shape,
        axes=[1],
        keepdims=1,
        dtype=TensorProto.FLOAT,
    )
    _run_cli_verify(model)

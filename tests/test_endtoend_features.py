from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
import pytest

from onnx import TensorProto, helper

from emx_onnx_cgen.compiler import Compiler, CompilerOptions
from emx_onnx_cgen.testbench import decode_testbench_array

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"


def _make_add_initializer_model() -> tuple[onnx.ModelProto, np.ndarray]:
    input_shape = [2, 3]
    input_info = helper.make_tensor_value_info("in0", TensorProto.FLOAT, input_shape)
    weight_values = np.linspace(0.1, 0.6, num=6, dtype=np.float32).reshape(input_shape)
    weight_initializer = helper.make_tensor(
        "weight",
        TensorProto.FLOAT,
        dims=input_shape,
        vals=weight_values.flatten().tolist(),
    )
    weight_info = helper.make_tensor_value_info(
        "weight", TensorProto.FLOAT, input_shape
    )
    output = helper.make_tensor_value_info("out", TensorProto.FLOAT, input_shape)
    node = helper.make_node("Add", inputs=["in0", "weight"], outputs=["out"])
    graph = helper.make_graph(
        [node],
        "add_init_graph",
        [input_info, weight_info],
        [output],
        initializer=[weight_initializer],
    )
    model = helper.make_model(
        graph,
        producer_name="onnx2c",
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model, weight_values


def _compile_and_run_testbench(
    model: onnx.ModelProto,
    *,
    compiler_options: CompilerOptions | None = None,
) -> tuple[dict[str, object], str]:
    compiler_cmd = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc")
    if compiler_cmd is None:
        pytest.skip("C compiler not available (set CC or install gcc/clang)")
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        c_path = temp_path / "model.c"
        exe_path = temp_path / "model"
        if compiler_options is None:
            compiler_options = CompilerOptions(emit_testbench=True)
        compiler = Compiler(compiler_options)
        generated = compiler.compile(model)
        c_path.write_text(generated, encoding="utf-8")
        subprocess.run(
            [compiler_cmd, "-std=c99", "-O2", str(c_path), "-o", str(exe_path), "-lm"],
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
        generated = c_path.read_text(encoding="utf-8")
    return json.loads(result.stdout), generated


def _run_cli_verify(model_path: Path) -> None:
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
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        env=env,
    )


def test_initializer_weights_emitted_as_static_arrays() -> None:
    model, weights = _make_add_initializer_model()
    payload, generated = _compile_and_run_testbench(model)
    assert "static const float weight" in generated
    output_data = decode_testbench_array(
        payload["outputs"]["out"]["data"], np.dtype(np.float32)
    )
    assert output_data.shape == weights.shape
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = Path(temp_dir) / "add_init.onnx"
        onnx.save_model(model, model_path)
        _run_cli_verify(model_path)


def test_testbench_accepts_constant_inputs() -> None:
    model, weights = _make_add_initializer_model()
    input_values = np.linspace(1.0, 6.0, num=6, dtype=np.float32).reshape(
        weights.shape
    )
    options = CompilerOptions(
        emit_testbench=True,
        testbench_inputs={"in0": input_values},
    )
    payload, generated = _compile_and_run_testbench(
        model, compiler_options=options
    )
    assert "static const float in0_testbench_data" in generated
    output_data = decode_testbench_array(
        payload["outputs"]["out"]["data"], np.dtype(np.float32)
    )
    np.testing.assert_allclose(output_data, input_values + weights)

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest

from onnx import TensorProto, helper

from onnx2c import Compiler
from onnx2c.compiler import CompilerOptions


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


def _compile_and_run_testbench(model: onnx.ModelProto) -> dict[str, object]:
    options = CompilerOptions(template_dir=Path("templates"), emit_testbench=True)
    compiler = Compiler(options)
    generated = compiler.compile(model)
    compiler_cmd = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc")
    if compiler_cmd is None:
        pytest.skip("C compiler not available (set CC or install gcc/clang)")
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        c_path = temp_path / "model.c"
        exe_path = temp_path / "model"
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
    return json.loads(result.stdout)


def test_initializer_weights_emitted_as_static_arrays() -> None:
    model, weights = _make_add_initializer_model()
    compiler = Compiler()
    generated = compiler.compile(model)
    assert "static const float weight" in generated
    payload = _compile_and_run_testbench(model)
    inputs = {
        name: np.array(value["data"], dtype=np.float32)
        for name, value in payload["inputs"].items()
    }
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    (ort_out,) = sess.run(None, {**inputs, "weight": weights})
    output_data = np.array(payload["output"]["data"], dtype=np.float32)
    np.testing.assert_allclose(output_data, ort_out, rtol=1e-4, atol=1e-5)

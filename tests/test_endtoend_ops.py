from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
import pytest

from onnx import TensorProto, helper

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"


def _make_operator_model(
    *,
    op_type: str,
    input_shapes: list[list[int]],
    output_shape: list[int],
    dtype: int,
    attrs: dict[str, object] | None = None,
    opset: int = 13,
) -> onnx.ModelProto:
    input_names = [f"in{idx}" for idx in range(len(input_shapes))]
    inputs = [
        helper.make_tensor_value_info(name, dtype, shape)
        for name, shape in zip(input_names, input_shapes)
    ]
    output = helper.make_tensor_value_info("out", dtype, output_shape)
    node = helper.make_node(
        op_type,
        inputs=input_names,
        outputs=[output.name],
        **(attrs or {}),
    )
    graph = helper.make_graph([node], f"{op_type.lower()}_graph", inputs, [output])
    model = helper.make_model(
        graph,
        producer_name="onnx2c",
        opset_imports=[helper.make_operatorsetid("", opset)],
    )
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model


def _make_constant_add_model() -> onnx.ModelProto:
    input_shape = [2, 3]
    input_info = helper.make_tensor_value_info("in0", TensorProto.FLOAT, input_shape)
    const_values = np.linspace(0.5, 1.0, num=6, dtype=np.float32).reshape(input_shape)
    const_tensor = helper.make_tensor(
        "const_tensor",
        TensorProto.FLOAT,
        dims=input_shape,
        vals=const_values.flatten().tolist(),
    )
    const_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["const_out"],
        value=const_tensor,
    )
    const_info = helper.make_tensor_value_info(
        "const_out", TensorProto.FLOAT, input_shape
    )
    output = helper.make_tensor_value_info("out", TensorProto.FLOAT, input_shape)
    add_node = helper.make_node(
        "Add", inputs=["in0", "const_out"], outputs=[output.name]
    )
    graph = helper.make_graph(
        [const_node, add_node],
        "constant_add_graph",
        [input_info],
        [output],
        value_info=[const_info],
    )
    model = helper.make_model(
        graph,
        producer_name="onnx2c",
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model


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
                "onnx2c",
                "verify",
                str(model_path),
                "--template-dir",
                str(PROJECT_ROOT / "templates"),
            ],
            check=True,
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            env=env,
        )


OPERATOR_CASES = [
    {
        "name": "Add",
        "op_type": "Add",
        "input_shapes": [[2, 3, 4], [2, 3, 4]],
        "output_shape": [2, 3, 4],
        "dtype": TensorProto.FLOAT,
        "attrs": {},
    },
    {
        "name": "AndBool",
        "op_type": "And",
        "input_shapes": [[2, 3], [2, 3]],
        "output_shape": [2, 3],
        "dtype": TensorProto.BOOL,
        "attrs": {},
    },
    {
        "name": "NotBool",
        "op_type": "Not",
        "input_shapes": [[2, 3]],
        "output_shape": [2, 3],
        "dtype": TensorProto.BOOL,
        "attrs": {},
    },
    {
        "name": "AddInt64",
        "op_type": "Add",
        "input_shapes": [[2, 3], [2, 3]],
        "output_shape": [2, 3],
        "dtype": TensorProto.INT64,
        "attrs": {},
    },
    {
        "name": "AddInt32",
        "op_type": "Add",
        "input_shapes": [[2, 3], [2, 3]],
        "output_shape": [2, 3],
        "dtype": TensorProto.INT32,
        "attrs": {},
    },
    {
        "name": "AddInt16",
        "op_type": "Add",
        "input_shapes": [[2, 3], [2, 3]],
        "output_shape": [2, 3],
        "dtype": TensorProto.INT16,
        "attrs": {},
        "opset": 14,
    },
    {
        "name": "AddInt8",
        "op_type": "Add",
        "input_shapes": [[2, 3], [2, 3]],
        "output_shape": [2, 3],
        "dtype": TensorProto.INT8,
        "attrs": {},
        "opset": 14,
    },
    {
        "name": "Mul",
        "op_type": "Mul",
        "input_shapes": [[2, 3], [2, 3]],
        "output_shape": [2, 3],
        "dtype": TensorProto.FLOAT,
        "attrs": {},
    },
    {
        "name": "Sub",
        "op_type": "Sub",
        "input_shapes": [[2, 3], [2, 3]],
        "output_shape": [2, 3],
        "dtype": TensorProto.FLOAT,
        "attrs": {},
    },
    {
        "name": "Div",
        "op_type": "Div",
        "input_shapes": [[2, 3], [2, 3]],
        "output_shape": [2, 3],
        "dtype": TensorProto.FLOAT,
        "attrs": {},
    },
    {
        "name": "Pow",
        "op_type": "Pow",
        "input_shapes": [[2, 3], [2, 3]],
        "output_shape": [2, 3],
        "dtype": TensorProto.FLOAT,
        "attrs": {},
        "opset": 13,
    },
    {
        "name": "Mod",
        "op_type": "Mod",
        "input_shapes": [[2, 3], [2, 3]],
        "output_shape": [2, 3],
        "dtype": TensorProto.FLOAT,
        "attrs": {"fmod": 1},
        "opset": 13,
    },
    {
        "name": "Min",
        "op_type": "Min",
        "input_shapes": [[2, 3], [2, 3]],
        "output_shape": [2, 3],
        "dtype": TensorProto.FLOAT,
        "attrs": {},
        "opset": 13,
    },
    {
        "name": "Max",
        "op_type": "Max",
        "input_shapes": [[2, 3], [2, 3]],
        "output_shape": [2, 3],
        "dtype": TensorProto.FLOAT,
        "attrs": {},
        "opset": 13,
    },
    {
        "name": "Mean",
        "op_type": "Mean",
        "input_shapes": [[2, 3], [2, 3]],
        "output_shape": [2, 3],
        "dtype": TensorProto.FLOAT,
        "attrs": {},
        "opset": 13,
    },
    {
        "name": "Sum",
        "op_type": "Sum",
        "input_shapes": [[2, 3], [2, 3]],
        "output_shape": [2, 3],
        "dtype": TensorProto.FLOAT,
        "attrs": {},
        "opset": 13,
    },
    {
        "name": "PRelu",
        "op_type": "PRelu",
        "input_shapes": [[2, 3], [2, 3]],
        "output_shape": [2, 3],
        "dtype": TensorProto.FLOAT,
        "attrs": {},
        "opset": 13,
    },
    {
        "name": "Tanh",
        "op_type": "Tanh",
        "input_shapes": [[2, 3]],
        "output_shape": [2, 3],
        "dtype": TensorProto.FLOAT,
        "attrs": {},
    },
    {
        "name": "Abs",
        "op_type": "Abs",
        "input_shapes": [[2, 3]],
        "output_shape": [2, 3],
        "dtype": TensorProto.FLOAT,
        "attrs": {},
    },
    {
        "name": "Ceil",
        "op_type": "Ceil",
        "input_shapes": [[2, 3]],
        "output_shape": [2, 3],
        "dtype": TensorProto.FLOAT,
        "attrs": {},
    },
    {
        "name": "Cos",
        "op_type": "Cos",
        "input_shapes": [[2, 3]],
        "output_shape": [2, 3],
        "dtype": TensorProto.FLOAT,
        "attrs": {},
    },
    {
        "name": "Exp",
        "op_type": "Exp",
        "input_shapes": [[2, 3]],
        "output_shape": [2, 3],
        "dtype": TensorProto.FLOAT,
        "attrs": {},
    },
    {
        "name": "Floor",
        "op_type": "Floor",
        "input_shapes": [[2, 3]],
        "output_shape": [2, 3],
        "dtype": TensorProto.FLOAT,
        "attrs": {},
    },
    {
        "name": "Log",
        "op_type": "Log",
        "input_shapes": [[2, 3]],
        "output_shape": [2, 3],
        "dtype": TensorProto.FLOAT,
        "attrs": {},
    },
    {
        "name": "Neg",
        "op_type": "Neg",
        "input_shapes": [[2, 3]],
        "output_shape": [2, 3],
        "dtype": TensorProto.FLOAT,
        "attrs": {},
    },
    {
        "name": "Relu",
        "op_type": "Relu",
        "input_shapes": [[2, 3]],
        "output_shape": [2, 3],
        "dtype": TensorProto.FLOAT,
        "attrs": {},
    },
    {
        "name": "MatMul",
        "op_type": "MatMul",
        "input_shapes": [[2, 3], [3, 4]],
        "output_shape": [2, 4],
        "dtype": TensorProto.FLOAT,
        "attrs": {},
    },
    {
        "name": "Gemm",
        "op_type": "Gemm",
        "input_shapes": [[2, 3], [3, 4]],
        "output_shape": [2, 4],
        "dtype": TensorProto.FLOAT,
        "attrs": {},
        "opset": 13,
    },
    {
        "name": "Attention",
        "op_type": "Attention",
        "input_shapes": [[1, 2, 3, 4], [1, 2, 5, 4], [1, 2, 5, 4]],
        "output_shape": [1, 2, 3, 4],
        "dtype": TensorProto.FLOAT,
        "attrs": {},
        "opset": 23,
    },
    {
        "name": "Sin",
        "op_type": "Sin",
        "input_shapes": [[2, 3]],
        "output_shape": [2, 3],
        "dtype": TensorProto.FLOAT,
        "attrs": {},
    },
    {
        "name": "Sqrt",
        "op_type": "Sqrt",
        "input_shapes": [[2, 3]],
        "output_shape": [2, 3],
        "dtype": TensorProto.FLOAT,
        "attrs": {},
    },
    {
        "name": "Tan",
        "op_type": "Tan",
        "input_shapes": [[2, 3]],
        "output_shape": [2, 3],
        "dtype": TensorProto.FLOAT,
        "attrs": {},
    },
]


@pytest.mark.parametrize("case", OPERATOR_CASES, ids=lambda case: case["name"])
def test_operator_c_testbench_matches_onnxruntime(case: dict[str, object]) -> None:
    model = _make_operator_model(
        op_type=case["op_type"],
        input_shapes=case["input_shapes"],
        output_shape=case["output_shape"],
        dtype=case["dtype"],
        attrs=case["attrs"],
        opset=case.get("opset", 13),
    )
    _run_cli_verify(model)


def test_constant_op_matches_onnxruntime() -> None:
    model = _make_constant_add_model()
    _run_cli_verify(model)

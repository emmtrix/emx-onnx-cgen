from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

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
        "name": "Mul",
        "op_type": "Mul",
        "input_shapes": [[2, 3], [2, 3]],
        "output_shape": [2, 3],
        "dtype": TensorProto.FLOAT,
        "attrs": {},
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
]


@pytest.mark.parametrize("case", OPERATOR_CASES, ids=lambda case: case["name"])
def test_operator_c_testbench_matches_onnxruntime(case: dict[str, object]) -> None:
    model = _make_operator_model(
        op_type=case["op_type"],
        input_shapes=case["input_shapes"],
        output_shape=case["output_shape"],
        dtype=case["dtype"],
        attrs=case["attrs"],
    )
    _run_cli_verify(model)

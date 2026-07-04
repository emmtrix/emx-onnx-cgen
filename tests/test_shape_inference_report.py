from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from golden_utils import assert_golden

from emx_onnx_cgen import cli
from emx_onnx_cgen.compiler import Compiler


def _make_matmul_relu_model(
    *, input_dims: list[object], output_dims: list[object] | None
) -> onnx.ModelProto:
    weight = numpy_helper.from_array(np.ones((4, 3), dtype=np.float32), name="w")
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, input_dims)
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, output_dims)
    nodes = [
        helper.make_node("MatMul", ["x", "w"], ["mid"]),
        helper.make_node("Relu", ["mid"], ["y"]),
    ]
    graph = helper.make_graph(nodes, "g", [x], [y], initializer=[weight])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 21)])


def test_shape_inference_report_contains_model_and_inferred_shapes() -> None:
    model = _make_matmul_relu_model(input_dims=[1, 4], output_dims=None)
    report = Compiler().shape_inference_report(model)

    assert report["format"] == "emx-onnx-cgen-shape-inference"
    tensors = report["tensors"]
    assert list(tensors) == ["x", "mid", "y"]
    assert "w" not in tensors

    assert tensors["x"]["kind"] == "input"
    assert tensors["x"]["model"] == {"dtype": "float", "dims": [1, 4]}
    assert tensors["x"]["inferred"] == {"dtype": "float", "dims": [1, 4]}

    # Internal node output without value_info: no declared, but inferred shape.
    assert tensors["mid"]["kind"] == "internal"
    assert tensors["mid"]["model"] is None
    assert tensors["mid"]["inferred"] == {"dtype": "float", "dims": [1, 3]}

    # Output declared without shape: dtype only on the model side.
    assert tensors["y"]["kind"] == "output"
    assert tensors["y"]["model"] == {"dtype": "float"}
    assert tensors["y"]["inferred"] == {"dtype": "float", "dims": [1, 3]}


def test_shape_inference_report_keeps_model_dim_params() -> None:
    model = _make_matmul_relu_model(input_dims=["N", 4], output_dims=["N", 3])
    report = Compiler().shape_inference_report(model)

    tensors = report["tensors"]
    assert tensors["x"]["model"]["dims"] == ["N", 4]
    assert tensors["x"]["inferred"]["dims"] == ["N", 4]
    assert tensors["mid"]["inferred"]["dims"] == ["N", 3]
    assert tensors["y"]["inferred"]["dims"] == ["N", 3]


def test_cli_compile_writes_shape_inference_json(tmp_path: Path) -> None:
    model = _make_matmul_relu_model(input_dims=[1, 4], output_dims=[1, 3])
    model_path = tmp_path / "model.onnx"
    onnx.save_model(model, model_path)
    json_path = tmp_path / "shapes.json"

    exit_code = cli.main(
        [
            "compile",
            str(model_path),
            str(tmp_path / "model.c"),
            "--shape-inference-json",
            str(json_path),
        ]
    )

    assert exit_code == 0
    report = json.loads(json_path.read_text(encoding="utf-8"))
    tensors = report["tensors"]
    assert tensors["mid"]["inferred"] == {"dtype": "float", "dims": [1, 3]}
    assert tensors["y"]["model"] == {"dtype": "float", "dims": [1, 3]}


# The golden file doubles as the documentation example referenced in README.md.
def test_shape_inference_json_matches_golden_reference(tmp_path: Path) -> None:
    model_path = Path(__file__).parent / "onnx" / "mixed_ops_dynamic_batch.onnx"
    json_path = tmp_path / "shapes.json"

    exit_code = cli.main(
        [
            "compile",
            str(model_path),
            str(tmp_path / "model.c"),
            "--shape-inference-json",
            str(json_path),
        ]
    )

    assert exit_code == 0
    golden_path = (
        Path(__file__).parent / "golden" / "mixed_ops_dynamic_batch_shapes.json"
    )
    assert_golden(json_path.read_text(encoding="utf-8"), golden_path)


def test_cli_compile_shape_inference_json_reflects_input_dim_pinning(
    tmp_path: Path,
) -> None:
    model = _make_matmul_relu_model(input_dims=["N", 4], output_dims=["N", 3])
    model_path = tmp_path / "model.onnx"
    onnx.save_model(model, model_path)
    json_path = tmp_path / "shapes.json"

    exit_code = cli.main(
        [
            "compile",
            str(model_path),
            str(tmp_path / "model.c"),
            "--input-dim",
            "N=2",
            "--shape-inference-json",
            str(json_path),
        ]
    )

    assert exit_code == 0
    report = json.loads(json_path.read_text(encoding="utf-8"))
    tensors = report["tensors"]
    assert tensors["x"]["model"]["dims"] == [2, 4]
    assert tensors["mid"]["inferred"]["dims"] == [2, 3]
    assert tensors["y"]["inferred"]["dims"] == [2, 3]

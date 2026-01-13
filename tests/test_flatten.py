from __future__ import annotations

import numpy as np
import onnx

from onnx import TensorProto, helper

from onnx2c.lowering.flatten import lower_flatten
from onnx2c.onnx_import import import_onnx


def _flatten_output_shape(
    input_shape: list[int], axis: int
) -> list[int]:
    rank = len(input_shape)
    if axis < 0:
        axis += rank
    first = int(np.prod(input_shape[:axis])) if axis else 1
    second = int(np.prod(input_shape[axis:])) if axis < rank else 1
    return [first, second]


def _make_flatten_model(
    input_shape: list[int], axis: int
) -> onnx.ModelProto:
    input_info = helper.make_tensor_value_info(
        "in0", TensorProto.FLOAT, input_shape
    )
    output_shape = _flatten_output_shape(input_shape, axis)
    output = helper.make_tensor_value_info(
        "out", TensorProto.FLOAT, output_shape
    )
    node = helper.make_node(
        "Flatten",
        inputs=["in0"],
        outputs=[output.name],
        axis=axis,
    )
    graph = helper.make_graph(
        [node],
        "flatten_graph",
        [input_info],
        [output],
    )
    model = helper.make_model(
        graph,
        producer_name="onnx2c",
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model


def test_lower_flatten_axis_default() -> None:
    model = _make_flatten_model([2, 3, 4], axis=1)
    graph = import_onnx(model)
    op = lower_flatten(graph, graph.nodes[0])
    assert op.input_shape == (2, 3, 4)
    assert op.output_shape == (2, 12)
    assert op.dtype == "float"


def test_lower_flatten_negative_axis() -> None:
    model = _make_flatten_model([2, 3, 4], axis=-1)
    graph = import_onnx(model)
    op = lower_flatten(graph, graph.nodes[0])
    assert op.output_shape == (6, 4)

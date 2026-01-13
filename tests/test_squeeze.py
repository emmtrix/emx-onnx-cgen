from __future__ import annotations

import numpy as np
import onnx

from onnx import TensorProto, helper

from onnx2c.lowering.squeeze import lower_squeeze
from onnx2c.onnx_import import import_onnx


def _make_squeeze_model(
    input_shape: list[int],
    output_shape: list[int],
    *,
    axes: list[int] | None = None,
    include_axes_input: bool = False,
    opset: int = 13,
) -> onnx.ModelProto:
    input_info = helper.make_tensor_value_info(
        "in0", TensorProto.FLOAT, input_shape
    )
    output = helper.make_tensor_value_info(
        "out", TensorProto.FLOAT, output_shape
    )
    inputs = ["in0"]
    initializers: list[onnx.TensorProto] = []
    attrs: dict[str, object] = {}
    if include_axes_input:
        if axes is None:
            raise ValueError("axes must be provided when axes input is included")
        axes_values = np.array(axes, dtype=np.int64)
        axes_tensor = helper.make_tensor(
            "axes",
            TensorProto.INT64,
            dims=axes_values.shape,
            vals=axes_values.tolist(),
        )
        initializers.append(axes_tensor)
        inputs.append("axes")
    elif axes is not None:
        attrs["axes"] = axes
    node = helper.make_node(
        "Squeeze",
        inputs=inputs,
        outputs=[output.name],
        **attrs,
    )
    graph = helper.make_graph(
        [node],
        "squeeze_graph",
        [input_info],
        [output],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="onnx2c",
        opset_imports=[helper.make_operatorsetid("", opset)],
    )
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model


def test_lower_squeeze_axes_input() -> None:
    model = _make_squeeze_model(
        [1, 3, 1, 5],
        [3, 5],
        axes=[0, 2],
        include_axes_input=True,
    )
    graph = import_onnx(model)
    op = lower_squeeze(graph, graph.nodes[0])
    assert op.input_shape == (1, 3, 1, 5)
    assert op.output_shape == (3, 5)
    assert op.dtype == "float"


def test_lower_squeeze_default_axes() -> None:
    model = _make_squeeze_model([1, 3, 1, 5], [3, 5])
    graph = import_onnx(model)
    op = lower_squeeze(graph, graph.nodes[0])
    assert op.output_shape == (3, 5)

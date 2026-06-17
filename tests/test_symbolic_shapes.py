from __future__ import annotations

import onnx
from onnx import TensorProto, helper, shape_inference

from emx_onnx_cgen.symbolic_shapes import infer_symbolic_shapes


def _dims(value_info: onnx.ValueInfoProto) -> list[object]:
    dims = []
    for dim in value_info.type.tensor_type.shape.dim:
        dims.append(dim.dim_value if dim.HasField("dim_value") else dim.dim_param)
    return dims


def _value_info(model: onnx.ModelProto, name: str) -> onnx.ValueInfoProto:
    for value_info in (*model.graph.value_info, *model.graph.output):
        if value_info.name == name:
            return value_info
    raise KeyError(name)


def test_reshape_target_resolved_from_static_output() -> None:
    # `r` is reshaped using the runtime `size` input, so ONNX leaves its shape
    # dynamic. The declared static output pins the symbols and the pass fills it.
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [12])
    size = helper.make_tensor_value_info("size", TensorProto.INT64, [2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [3, 4])
    nodes = [
        helper.make_node("Reshape", ["x", "size"], ["r"]),
        helper.make_node("Relu", ["r"], ["y"]),
    ]
    graph = helper.make_graph(nodes, "reshape_graph", [x, size], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 18)])
    model.ir_version = 9
    model = shape_inference.infer_shapes(model, data_prop=True)

    # Precondition: the intermediate is dynamic before the pass.
    assert any(isinstance(d, str) for d in _dims(_value_info(model, "r")))

    model, changed = infer_symbolic_shapes(model)
    assert changed
    assert _dims(_value_info(model, "r")) == [3, 4]


def test_dynamic_batch_is_left_untouched() -> None:
    # A genuinely dynamic dimension (no runtime integer input feeds it) must not
    # be pinned to a concrete value.
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", 4])
    graph = helper.make_graph(
        [helper.make_node("Relu", ["x"], ["y"])], "dyn_graph", [x], [y]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 18)])
    model.ir_version = 9
    model = shape_inference.infer_shapes(model, data_prop=True)

    model, changed = infer_symbolic_shapes(model)
    assert not changed
    assert _dims(model.graph.output[0]) == ["batch", 4]


def test_existing_static_dim_is_never_overwritten() -> None:
    # The pass only fills dynamic dims; concrete dims declared on outputs stay.
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [12])
    size = helper.make_tensor_value_info("size", TensorProto.INT64, [2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [3, 4])
    graph = helper.make_graph(
        [helper.make_node("Reshape", ["x", "size"], ["y"])],
        "reshape_out_graph",
        [x, size],
        [y],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 18)])
    model.ir_version = 9
    model = shape_inference.infer_shapes(model, data_prop=True)

    model, _ = infer_symbolic_shapes(model)
    assert _dims(model.graph.output[0]) == [3, 4]

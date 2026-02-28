from __future__ import annotations

from shared.scalar_types import ScalarType

from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from ..ir.ops import QLinearSoftmaxOp
from .common import onnx_opset_version
from .common import value_dtype as _value_dtype
from .common import value_shape as _value_shape
from .registry import register_lowering


def _ensure_scalar_input(
    graph: Graph, name: str, node: Node, label: str
) -> tuple[int, ...]:
    shape = _value_shape(graph, name, node)
    if shape not in {(), (1,)}:
        raise UnsupportedOpError(
            f"QLinearSoftmax {label} must be scalar, got shape {shape}"
        )
    return shape


@register_lowering("QLinearSoftmax")
def lower_qlinear_softmax(graph: Graph, node: Node) -> QLinearSoftmaxOp:
    if len(node.inputs) != 5 or len(node.outputs) != 1:
        raise UnsupportedOpError("QLinearSoftmax must have 5 inputs and 1 output")

    input_shape = _value_shape(graph, node.inputs[0], node)
    try:
        output_shape = _value_shape(graph, node.outputs[0], node)
    except ShapeInferenceError:
        output_shape = input_shape
    if output_shape != input_shape:
        raise ShapeInferenceError(
            f"QLinearSoftmax output shape must be {input_shape}, got {output_shape}"
        )

    input_dtype = _value_dtype(graph, node.inputs[0], node)
    try:
        output_dtype = _value_dtype(graph, node.outputs[0], node)
    except ShapeInferenceError:
        output_dtype = input_dtype
    if input_dtype not in {ScalarType.U8, ScalarType.I8}:
        raise UnsupportedOpError("QLinearSoftmax supports uint8/int8 inputs only")
    if output_dtype != input_dtype:
        raise UnsupportedOpError(
            "QLinearSoftmax expects output dtype to match input dtype"
        )

    input_scale_dtype = _value_dtype(graph, node.inputs[1], node)
    output_scale_dtype = _value_dtype(graph, node.inputs[3], node)
    if not input_scale_dtype.is_float:
        raise UnsupportedOpError("QLinearSoftmax x_scale must be float16/float/double")
    if not output_scale_dtype.is_float:
        raise UnsupportedOpError("QLinearSoftmax y_scale must be float16/float/double")

    input_zero_dtype = _value_dtype(graph, node.inputs[2], node)
    output_zero_dtype = _value_dtype(graph, node.inputs[4], node)
    if input_zero_dtype != input_dtype:
        raise UnsupportedOpError("QLinearSoftmax x_zero_point dtype must match x")
    if output_zero_dtype != output_dtype:
        raise UnsupportedOpError("QLinearSoftmax y_zero_point dtype must match y")

    input_scale_shape = _ensure_scalar_input(graph, node.inputs[1], node, "x_scale")
    output_scale_shape = _ensure_scalar_input(graph, node.inputs[3], node, "y_scale")
    input_zero_shape = _ensure_scalar_input(graph, node.inputs[2], node, "x_zero_point")
    output_zero_shape = _ensure_scalar_input(
        graph, node.inputs[4], node, "y_zero_point"
    )

    softmax_opset = int(node.attrs["opset"]) if "opset" in node.attrs else None
    if softmax_opset is None:
        softmax_opset = onnx_opset_version(graph)
    use_legacy_axis_semantics = softmax_opset is not None and softmax_opset < 13
    axis = int(node.attrs["axis"]) if "axis" in node.attrs else None
    if axis is None:
        axis = 1 if use_legacy_axis_semantics else -1
    if axis < 0:
        axis += len(input_shape)
    if axis < 0 or axis >= len(input_shape):
        raise ShapeInferenceError(
            f"QLinearSoftmax axis {node.attrs.get('axis', None)} "
            f"is out of bounds for shape {input_shape}"
        )

    if use_legacy_axis_semantics:
        outer = 1
        for dim in input_shape[:axis]:
            outer *= dim
        axis_size = 1
        for dim in input_shape[axis:]:
            axis_size *= dim
        inner = 1
    else:
        outer = 1
        for dim in input_shape[:axis]:
            outer *= dim
        axis_size = input_shape[axis]
        inner = 1
        for dim in input_shape[axis + 1 :]:
            inner *= dim

    return QLinearSoftmaxOp(
        input0=node.inputs[0],
        input_scale=node.inputs[1],
        input_zero_point=node.inputs[2],
        output_scale=node.inputs[3],
        output_zero_point=node.inputs[4],
        output=node.outputs[0],
        input_shape=input_shape,
        output_shape=output_shape,
        axis=axis,
        outer=outer,
        axis_size=axis_size,
        inner=inner,
        dtype=output_dtype,
        input_dtype=input_dtype,
        input_scale_dtype=input_scale_dtype,
        output_scale_dtype=output_scale_dtype,
        input_scale_shape=input_scale_shape,
        output_scale_shape=output_scale_shape,
        input_zero_shape=input_zero_shape,
        output_zero_shape=output_zero_shape,
    )

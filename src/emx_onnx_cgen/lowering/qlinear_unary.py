from __future__ import annotations

from shared.scalar_types import ScalarType

from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from ..ir.ops import QLinearUnaryOp
from .common import optional_name, value_dtype, value_shape
from .registry import register_lowering


def _lower_qlinear_unary(
    graph: Graph, node: Node, op_kind: str, alpha: float = 0.0
) -> QLinearUnaryOp:
    if len(node.inputs) < 4 or len(node.outputs) != 1:
        raise UnsupportedOpError(
            f"{node.op_type} must have at least 4 inputs and 1 output"
        )
    input_name = node.inputs[0]
    x_scale_name = node.inputs[1]
    x_zero_name = optional_name(node.inputs, 2)
    y_scale_name = node.inputs[3]
    y_zero_name = node.inputs[4]
    output_name = node.outputs[0]

    input_dtype = value_dtype(graph, input_name, node)
    if input_dtype not in {ScalarType.U8, ScalarType.I8}:
        raise UnsupportedOpError(f"{node.op_type} supports uint8/int8 inputs only")
    try:
        output_dtype = value_dtype(graph, output_name, node)
    except ShapeInferenceError:
        output_dtype = input_dtype
    if output_dtype != input_dtype:
        raise UnsupportedOpError(
            f"{node.op_type} expects output dtype to match input dtype"
        )

    input_shape = value_shape(graph, input_name, node)
    try:
        output_shape = value_shape(graph, output_name, node)
    except ShapeInferenceError:
        output_shape = input_shape
    if output_shape != input_shape:
        raise ShapeInferenceError(
            f"{node.op_type} output shape must be {input_shape}, got {output_shape}"
        )

    return QLinearUnaryOp(
        input0=input_name,
        x_scale=x_scale_name,
        x_zero_point=x_zero_name,
        y_scale=y_scale_name,
        y_zero_point=y_zero_name,
        output=output_name,
        shape=input_shape,
        op_kind=op_kind,
        alpha=alpha,
        dtype=input_dtype,
    )


@register_lowering("QLinearSigmoid")
def lower_qlinear_sigmoid(graph: Graph, node: Node) -> QLinearUnaryOp:
    return _lower_qlinear_unary(graph, node, op_kind="sigmoid")


@register_lowering("QLinearLeakyRelu")
def lower_qlinear_leaky_relu(graph: Graph, node: Node) -> QLinearUnaryOp:
    alpha = float(node.attrs.get("alpha", 0.01))
    return _lower_qlinear_unary(graph, node, op_kind="leaky_relu", alpha=alpha)

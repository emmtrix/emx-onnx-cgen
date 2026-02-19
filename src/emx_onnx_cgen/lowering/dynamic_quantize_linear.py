from __future__ import annotations

from shared.scalar_types import ScalarType

from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from ..ir.ops import DynamicQuantizeLinearOp
from .common import value_dtype as _value_dtype, value_shape as _value_shape
from .registry import register_lowering


@register_lowering("DynamicQuantizeLinear")
def lower_dynamic_quantize_linear(graph: Graph, node: Node) -> DynamicQuantizeLinearOp:
    if len(node.inputs) != 1 or len(node.outputs) != 3:
        raise UnsupportedOpError(
            "DynamicQuantizeLinear must have 1 input and 3 outputs"
        )
    if node.attrs:
        raise UnsupportedOpError("DynamicQuantizeLinear has unsupported attributes")

    input_dtype = _value_dtype(graph, node.inputs[0], node)
    if input_dtype != ScalarType.F32:
        raise UnsupportedOpError("DynamicQuantizeLinear supports float inputs only")

    output_dtype = _value_dtype(graph, node.outputs[0], node)
    if output_dtype != ScalarType.U8:
        raise UnsupportedOpError("DynamicQuantizeLinear output must be uint8")

    scale_dtype = _value_dtype(graph, node.outputs[1], node)
    if scale_dtype != ScalarType.F32:
        raise UnsupportedOpError("DynamicQuantizeLinear y_scale output must be float")

    zero_point_dtype = _value_dtype(graph, node.outputs[2], node)
    if zero_point_dtype != ScalarType.U8:
        raise UnsupportedOpError(
            "DynamicQuantizeLinear y_zero_point output must be uint8"
        )

    input_shape = _value_shape(graph, node.inputs[0], node)
    output_shape = _value_shape(graph, node.outputs[0], node)
    if output_shape != input_shape:
        raise ShapeInferenceError(
            "DynamicQuantizeLinear output shape must match input shape"
        )

    scale_shape = _value_shape(graph, node.outputs[1], node)
    if scale_shape not in {(), (1,)}:
        raise ShapeInferenceError("DynamicQuantizeLinear y_scale must be scalar")

    zero_point_shape = _value_shape(graph, node.outputs[2], node)
    if zero_point_shape not in {(), (1,)}:
        raise ShapeInferenceError("DynamicQuantizeLinear y_zero_point must be scalar")

    return DynamicQuantizeLinearOp(
        input0=node.inputs[0],
        output=node.outputs[0],
        scale=node.outputs[1],
        zero_point=node.outputs[2],
    )

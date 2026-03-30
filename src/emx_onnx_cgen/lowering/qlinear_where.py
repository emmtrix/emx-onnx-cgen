from __future__ import annotations

from shared.scalar_types import ScalarType

from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from ..ir.op_base import BroadcastingOpBase
from ..ir.ops import QLinearWhereOp
from .common import value_dtype as _value_dtype
from .common import value_shape as _value_shape
from .registry import register_lowering


@register_lowering("QLinearWhere")
def lower_qlinear_where(graph: Graph, node: Node) -> QLinearWhereOp:
    # Inputs: condition, X, x_scale, x_zero_point, Y, y_scale, y_zero_point, z_scale, z_zero_point
    if len(node.inputs) != 9 or len(node.outputs) != 1:
        raise UnsupportedOpError("QLinearWhere must have 9 inputs and 1 output")

    (
        condition_name,
        x_name,
        x_scale_name,
        x_zero_name,
        y_name,
        y_scale_name,
        y_zero_name,
        z_scale_name,
        z_zero_name,
    ) = node.inputs
    output_name = node.outputs[0]

    condition_dtype = _value_dtype(graph, condition_name, node)
    if condition_dtype != ScalarType.BOOL:
        raise UnsupportedOpError(
            f"QLinearWhere: condition must be bool, got {condition_dtype.onnx_name}"
        )

    x_dtype = _value_dtype(graph, x_name, node)
    y_dtype = _value_dtype(graph, y_name, node)
    output_dtype = _value_dtype(graph, output_name, node)

    if x_dtype not in {ScalarType.I8, ScalarType.U8}:
        raise UnsupportedOpError(
            f"QLinearWhere: X must be int8 or uint8, got {x_dtype.onnx_name}"
        )
    if x_dtype != y_dtype:
        raise UnsupportedOpError(
            f"QLinearWhere: X and Y must have the same dtype, got {x_dtype.onnx_name} and {y_dtype.onnx_name}"
        )
    if x_dtype != output_dtype:
        raise UnsupportedOpError(
            f"QLinearWhere: output dtype must match X/Y, got {output_dtype.onnx_name}"
        )

    for scale_name, label in [(x_scale_name, "x_scale"), (y_scale_name, "y_scale"), (z_scale_name, "z_scale")]:
        scale_dtype = _value_dtype(graph, scale_name, node)
        if not scale_dtype.is_float:
            raise UnsupportedOpError(f"QLinearWhere: {label} must be float")
        scale_shape = _value_shape(graph, scale_name, node)
        if scale_shape not in {(), (1,)}:
            raise UnsupportedOpError(f"QLinearWhere: {label} must be scalar")

    for zero_name, label in [(x_zero_name, "x_zero_point"), (y_zero_name, "y_zero_point"), (z_zero_name, "z_zero_point")]:
        zero_shape = _value_shape(graph, zero_name, node)
        if zero_shape not in {(), (1,)}:
            raise UnsupportedOpError(f"QLinearWhere: {label} must be scalar")

    condition_shape = _value_shape(graph, condition_name, node)
    x_shape = _value_shape(graph, x_name, node)
    y_shape = _value_shape(graph, y_name, node)
    output_shape = _value_shape(graph, output_name, node)

    broadcast_shape = BroadcastingOpBase.broadcast_shapes(
        BroadcastingOpBase.broadcast_shapes(condition_shape, x_shape), y_shape
    )
    if tuple(output_shape) != broadcast_shape:
        raise ShapeInferenceError(
            f"QLinearWhere: output shape mismatch, expected {broadcast_shape}, got {output_shape}"
        )

    return QLinearWhereOp(
        condition=condition_name,
        input_x=x_name,
        x_scale=x_scale_name,
        x_zero_point=x_zero_name,
        input_y=y_name,
        y_scale=y_scale_name,
        y_zero_point=y_zero_name,
        z_scale=z_scale_name,
        z_zero_point=z_zero_name,
        output=output_name,
        condition_shape=condition_shape,
        x_shape=x_shape,
        y_shape=y_shape,
        output_shape=broadcast_shape,
        dtype=x_dtype,
    )

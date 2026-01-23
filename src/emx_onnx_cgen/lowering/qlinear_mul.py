from __future__ import annotations

from shared.scalar_types import ScalarType

from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from ..ir.op_base import BroadcastingOpBase
from ..ir.ops import QLinearMulOp
from .common import value_dtype as _value_dtype
from .common import value_shape as _value_shape
from .registry import register_lowering


def _ensure_scalar_input(
    graph: Graph, name: str, node: Node, label: str
) -> tuple[int, ...]:
    shape = _value_shape(graph, name, node)
    if shape not in {(), (1,)}:
        raise UnsupportedOpError(
            f"QLinearMul {label} must be scalar, got shape {shape}"
        )
    return shape


def _ensure_scale_dtype(dtype: ScalarType, label: str) -> None:
    if not dtype.is_float:
        raise UnsupportedOpError(
            f"QLinearMul {label} must be float16/float/double"
        )


@register_lowering("QLinearMul")
def lower_qlinear_mul(graph: Graph, node: Node) -> QLinearMulOp:
    if len(node.inputs) != 8 or len(node.outputs) != 1:
        raise UnsupportedOpError("QLinearMul must have 8 inputs and 1 output")
    input0_shape = _value_shape(graph, node.inputs[0], node)
    input1_shape = _value_shape(graph, node.inputs[3], node)
    output_shape = BroadcastingOpBase.broadcast_shapes(
        input0_shape, input1_shape
    )
    expected_output_shape = _value_shape(graph, node.outputs[0], node)
    if expected_output_shape != output_shape:
        raise ShapeInferenceError(
            "QLinearMul output shape must be "
            f"{output_shape}, got {expected_output_shape}"
        )
    input0_dtype = _value_dtype(graph, node.inputs[0], node)
    input1_dtype = _value_dtype(graph, node.inputs[3], node)
    output_dtype = _value_dtype(graph, node.outputs[0], node)
    if input0_dtype not in {ScalarType.U8, ScalarType.I8}:
        raise UnsupportedOpError("QLinearMul supports uint8/int8 inputs only")
    if input1_dtype not in {ScalarType.U8, ScalarType.I8}:
        raise UnsupportedOpError("QLinearMul supports uint8/int8 inputs only")
    if output_dtype not in {ScalarType.U8, ScalarType.I8}:
        raise UnsupportedOpError(
            "QLinearMul supports uint8/int8 outputs only"
        )
    input0_scale_dtype = _value_dtype(graph, node.inputs[1], node)
    input1_scale_dtype = _value_dtype(graph, node.inputs[4], node)
    output_scale_dtype = _value_dtype(graph, node.inputs[6], node)
    _ensure_scale_dtype(input0_scale_dtype, "a_scale")
    _ensure_scale_dtype(input1_scale_dtype, "b_scale")
    _ensure_scale_dtype(output_scale_dtype, "y_scale")
    input0_zero_dtype = _value_dtype(graph, node.inputs[2], node)
    input1_zero_dtype = _value_dtype(graph, node.inputs[5], node)
    output_zero_dtype = _value_dtype(graph, node.inputs[7], node)
    if input0_zero_dtype != input0_dtype:
        raise UnsupportedOpError("QLinearMul a_zero_point dtype must match a")
    if input1_zero_dtype != input1_dtype:
        raise UnsupportedOpError("QLinearMul b_zero_point dtype must match b")
    if output_zero_dtype != output_dtype:
        raise UnsupportedOpError("QLinearMul y_zero_point dtype must match y")
    input0_scale_shape = _ensure_scalar_input(
        graph, node.inputs[1], node, "a_scale"
    )
    input1_scale_shape = _ensure_scalar_input(
        graph, node.inputs[4], node, "b_scale"
    )
    output_scale_shape = _ensure_scalar_input(
        graph, node.inputs[6], node, "y_scale"
    )
    input0_zero_shape = _ensure_scalar_input(
        graph, node.inputs[2], node, "a_zero_point"
    )
    input1_zero_shape = _ensure_scalar_input(
        graph, node.inputs[5], node, "b_zero_point"
    )
    output_zero_shape = _ensure_scalar_input(
        graph, node.inputs[7], node, "y_zero_point"
    )
    return QLinearMulOp(
        input0=node.inputs[0],
        input0_scale=node.inputs[1],
        input0_zero_point=node.inputs[2],
        input1=node.inputs[3],
        input1_scale=node.inputs[4],
        input1_zero_point=node.inputs[5],
        output_scale=node.inputs[6],
        output_zero_point=node.inputs[7],
        output=node.outputs[0],
        input0_shape=input0_shape,
        input1_shape=input1_shape,
        output_shape=output_shape,
        input0_dtype=input0_dtype,
        input1_dtype=input1_dtype,
        dtype=output_dtype,
        input0_scale_dtype=input0_scale_dtype,
        input1_scale_dtype=input1_scale_dtype,
        output_scale_dtype=output_scale_dtype,
        input0_scale_shape=input0_scale_shape,
        input1_scale_shape=input1_scale_shape,
        output_scale_shape=output_scale_shape,
        input0_zero_shape=input0_zero_shape,
        input1_zero_shape=input1_zero_shape,
        output_zero_shape=output_zero_shape,
    )

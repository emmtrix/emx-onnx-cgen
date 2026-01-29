from __future__ import annotations

from dataclasses import dataclass

from shared.scalar_types import ScalarType

from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from ..validation import normalize_axis
from .common import (
    optional_name,
    value_dtype as _value_dtype,
    value_shape as _value_shape,
)
from .registry import register_lowering
from ..ir.ops import DequantizeLinearOp


@dataclass(frozen=True)
class DequantizeSpec:
    input_shape: tuple[int, ...]
    scale_shape: tuple[int, ...]
    axis: int | None


def resolve_dequantize_spec(graph: Graph, node: Node) -> DequantizeSpec:
    if len(node.inputs) not in {2, 3} or len(node.outputs) != 1:
        raise UnsupportedOpError(
            "DequantizeLinear must have 2 or 3 inputs and 1 output"
        )
    supported_attrs = {"axis"}
    if set(node.attrs) - supported_attrs:
        raise UnsupportedOpError("DequantizeLinear has unsupported attributes")
    input_shape = _value_shape(graph, node.inputs[0], node)
    scale_shape = _value_shape(graph, node.inputs[1], node)
    zero_point_name = optional_name(node.inputs, 2)
    if zero_point_name is not None:
        zero_point_shape = _value_shape(graph, zero_point_name, node)
        if zero_point_shape != scale_shape:
            raise ShapeInferenceError(
                "DequantizeLinear zero_point shape must match scale shape"
            )
    if scale_shape not in {(), (1,)}:
        if len(scale_shape) != 1:
            raise UnsupportedOpError(
                "DequantizeLinear supports per-tensor and per-axis scales only"
            )
        axis = int(node.attrs.get("axis", 1))
        axis = normalize_axis(axis, input_shape, node)
        if scale_shape[0] != input_shape[axis]:
            raise ShapeInferenceError(
                "DequantizeLinear scale length must match input axis size"
            )
    else:
        axis = None
    return DequantizeSpec(
        input_shape=input_shape,
        scale_shape=scale_shape,
        axis=axis,
    )


@register_lowering("DequantizeLinear")
def lower_dequantize_linear(graph: Graph, node: Node) -> DequantizeLinearOp:
    input_dtype = _value_dtype(graph, node.inputs[0], node)
    scale_dtype = _value_dtype(graph, node.inputs[1], node)
    output_dtype = _value_dtype(graph, node.outputs[0], node)
    if input_dtype not in {
        ScalarType.U8,
        ScalarType.I8,
        ScalarType.U16,
        ScalarType.I16,
    }:
        raise UnsupportedOpError(
            "DequantizeLinear supports int8/uint8/int16/uint16 inputs only"
        )
    if not scale_dtype.is_float or not output_dtype.is_float:
        raise UnsupportedOpError(
            "DequantizeLinear supports float16/float/double scales and outputs only"
        )
    if output_dtype != scale_dtype:
        raise UnsupportedOpError(
            "DequantizeLinear output dtype must match scale dtype"
        )
    zero_point_name = optional_name(node.inputs, 2)
    if zero_point_name is not None:
        zero_point_dtype = _value_dtype(graph, zero_point_name, node)
        if zero_point_dtype != input_dtype:
            raise UnsupportedOpError(
                "DequantizeLinear zero_point dtype must match input dtype"
            )
    spec = resolve_dequantize_spec(graph, node)
    return DequantizeLinearOp(
        input0=node.inputs[0],
        scale=node.inputs[1],
        zero_point=zero_point_name,
        output=node.outputs[0],
        input_shape=spec.input_shape,
        axis=spec.axis,
        dtype=output_dtype,
        input_dtype=input_dtype,
        scale_dtype=scale_dtype,
    )

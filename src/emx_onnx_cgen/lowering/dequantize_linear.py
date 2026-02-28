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
    block_size: int | None


def _shapes_match_or_both_scalar_like(
    lhs: tuple[int, ...], rhs: tuple[int, ...]
) -> bool:
    if lhs == rhs:
        return True
    scalar_like_shapes = {(), (1,)}
    return lhs in scalar_like_shapes and rhs in scalar_like_shapes


def resolve_dequantize_spec(graph: Graph, node: Node) -> DequantizeSpec:
    if len(node.inputs) not in {2, 3} or len(node.outputs) != 1:
        raise UnsupportedOpError(
            "DequantizeLinear must have 2 or 3 inputs and 1 output"
        )
    supported_attrs = {"axis", "block_size"}
    if set(node.attrs) - supported_attrs:
        raise UnsupportedOpError("DequantizeLinear has unsupported attributes")
    block_size = int(node.attrs.get("block_size", 0))
    if block_size < 0:
        raise UnsupportedOpError("DequantizeLinear block_size must be >= 0")
    try:
        input_shape = _value_shape(graph, node.inputs[0], node)
    except ShapeInferenceError:
        input_shape = _infer_missing_input_shape(graph, node)
    scale_shape = _value_shape(graph, node.inputs[1], node)
    zero_point_name = optional_name(node.inputs, 2)
    if zero_point_name is not None:
        zero_point_shape = _value_shape(graph, zero_point_name, node)
        if not _shapes_match_or_both_scalar_like(zero_point_shape, scale_shape):
            raise ShapeInferenceError(
                "DequantizeLinear zero_point shape must match scale shape"
            )
    if scale_shape not in {(), (1,)}:
        axis = int(node.attrs.get("axis", 1))
        axis = normalize_axis(axis, input_shape, node)
        if block_size > 0:
            if len(scale_shape) != len(input_shape):
                raise UnsupportedOpError(
                    "DequantizeLinear blocked scales must match input rank"
                )
            if input_shape[axis] % block_size != 0:
                raise ShapeInferenceError(
                    "DequantizeLinear block_size must evenly divide axis length"
                )
            expected = list(input_shape)
            expected[axis] = input_shape[axis] // block_size
            if scale_shape != tuple(expected):
                raise ShapeInferenceError(
                    "DequantizeLinear blocked scale shape must match "
                    "input shape with a reduced axis"
                )
        else:
            if len(scale_shape) != 1:
                raise UnsupportedOpError(
                    "DequantizeLinear supports per-tensor, per-axis, "
                    "and blocked scales only"
                )
            if scale_shape[0] != input_shape[axis]:
                raise ShapeInferenceError(
                    "DequantizeLinear scale length must match input axis size"
                )
    else:
        axis = None
        block_size = 0
    return DequantizeSpec(
        input_shape=input_shape,
        scale_shape=scale_shape,
        axis=axis,
        block_size=block_size or None,
    )


def _producer_by_output(graph: Graph, output_name: str) -> Node | None:
    for producer in graph.nodes:
        if output_name in producer.outputs:
            return producer
    return None


def _infer_missing_input_shape(graph: Graph, node: Node) -> tuple[int, ...]:
    producer = _producer_by_output(graph, node.inputs[0])
    if (
        producer is None
        or producer.op_type != "QLinearSoftmax"
        or len(producer.inputs) < 1
    ):
        raise ShapeInferenceError(
            f"Missing shape for value '{node.inputs[0]}' in op {node.op_type}. "
            "Hint: run ONNX shape inference or export with static shapes."
        )
    return _value_shape(graph, producer.inputs[0], producer)


def _infer_missing_input_dtype(graph: Graph, node: Node) -> ScalarType:
    producer = _producer_by_output(graph, node.inputs[0])
    if (
        producer is None
        or producer.op_type != "QLinearSoftmax"
        or len(producer.inputs) < 3
    ):
        raise ShapeInferenceError(
            f"Missing dtype for value '{node.inputs[0]}' in op {node.op_type}. "
            "Hint: run ONNX shape inference or export with static shapes."
        )
    return _value_dtype(graph, producer.inputs[2], producer)


@register_lowering("DequantizeLinear")
def lower_dequantize_linear(graph: Graph, node: Node) -> DequantizeLinearOp:
    try:
        input_dtype = _value_dtype(graph, node.inputs[0], node)
    except ShapeInferenceError:
        input_dtype = _infer_missing_input_dtype(graph, node)
    scale_dtype = _value_dtype(graph, node.inputs[1], node)
    output_dtype = _value_dtype(graph, node.outputs[0], node)
    if input_dtype not in {
        ScalarType.U8,
        ScalarType.I8,
        ScalarType.U16,
        ScalarType.I16,
        ScalarType.I32,
        ScalarType.U32,
    }:
        raise UnsupportedOpError(
            "DequantizeLinear supports int8/uint8/int16/uint16/int32/uint32 inputs only"
        )
    if not scale_dtype.is_float or not output_dtype.is_float:
        raise UnsupportedOpError(
            "DequantizeLinear supports float16/float/double scales and outputs only"
        )
    if output_dtype != scale_dtype:
        raise UnsupportedOpError("DequantizeLinear output dtype must match scale dtype")
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
        axis=spec.axis,
        block_size=spec.block_size,
    )

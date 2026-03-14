from __future__ import annotations

from dataclasses import dataclass

from shared.scalar_types import ScalarType

from ..dtypes import scalar_type_from_onnx
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from ..validation import normalize_axis
from .common import (
    optional_name,
    value_dtype as _value_dtype,
    value_shape as _value_shape,
)
from .registry import register_lowering
from ..ir.ops import QuantizeLinearOp


@dataclass(frozen=True)
class QuantizeSpec:
    input_shape: tuple[int, ...]
    scale_shape: tuple[int, ...]
    axis: int | None
    block_size: int | None
    output_dtype: ScalarType


def _resolve_output_dtype(
    graph: Graph, node: Node, zero_point_name: str | None
) -> ScalarType:
    output_attr = int(node.attrs.get("output_dtype", 0))
    if output_attr:
        output_dtype = _value_dtype(graph, node.outputs[0], node)
        attr_dtype = scalar_type_from_onnx(output_attr)
        if attr_dtype is None:
            raise UnsupportedOpError(
                "QuantizeLinear output_dtype must map to a supported scalar type"
            )
        if output_dtype != attr_dtype:
            raise UnsupportedOpError(
                "QuantizeLinear output_dtype must match output tensor dtype"
            )
        return output_dtype
    if zero_point_name is None:
        return ScalarType.U8
    return _value_dtype(graph, zero_point_name, node)


def resolve_quantize_spec(graph: Graph, node: Node) -> QuantizeSpec:
    if len(node.inputs) not in {2, 3} or len(node.outputs) != 1:
        raise UnsupportedOpError("QuantizeLinear must have 2 or 3 inputs and 1 output")
    supported_attrs = {"axis", "block_size", "output_dtype", "precision", "saturate"}
    if set(node.attrs) - supported_attrs:
        raise UnsupportedOpError("QuantizeLinear has unsupported attributes")
    block_size = int(node.attrs.get("block_size", 0))
    if block_size < 0:
        raise UnsupportedOpError("QuantizeLinear block_size must be >= 0")
    precision = int(node.attrs.get("precision", 0))
    if precision != 0:
        raise UnsupportedOpError("QuantizeLinear precision is not supported")
    saturate = int(node.attrs.get("saturate", 1))
    if saturate != 1:
        raise UnsupportedOpError("QuantizeLinear saturate must be 1")
    input_shape = _value_shape(graph, node.inputs[0], node)
    scale_shape = _value_shape(graph, node.inputs[1], node)
    zero_point_name = optional_name(node.inputs, 2)
    output_dtype = _resolve_output_dtype(graph, node, zero_point_name)
    if output_dtype not in {
        ScalarType.U2,
        ScalarType.I2,
        ScalarType.U4,
        ScalarType.I4,
        ScalarType.U8,
        ScalarType.I8,
        ScalarType.U16,
        ScalarType.I16,
    } and not output_dtype.is_float8:
        raise UnsupportedOpError(
            "QuantizeLinear supports int/uint and float8 outputs only"
        )
    if zero_point_name is not None:
        zero_point_dtype = _value_dtype(graph, zero_point_name, node)
        if zero_point_dtype != output_dtype:
            raise UnsupportedOpError(
                "QuantizeLinear zero_point dtype must match output dtype"
            )
        zero_point_shape = _value_shape(graph, zero_point_name, node)
        if zero_point_shape != scale_shape and {
            zero_point_shape,
            scale_shape,
        } != {(), (1,)}:
            raise ShapeInferenceError(
                "QuantizeLinear zero_point shape must match scale shape"
            )
    if scale_shape not in {(), (1,)}:
        axis = int(node.attrs.get("axis", 1))
        axis = normalize_axis(axis, input_shape, node)
        if block_size > 0:
            if len(scale_shape) != len(input_shape):
                raise UnsupportedOpError(
                    "QuantizeLinear blocked scales must match input rank"
                )
            if input_shape[axis] % block_size != 0:
                raise ShapeInferenceError(
                    "QuantizeLinear block_size must evenly divide axis length"
                )
            expected = list(input_shape)
            expected[axis] = input_shape[axis] // block_size
            if scale_shape != tuple(expected):
                raise ShapeInferenceError(
                    "QuantizeLinear blocked scale shape must match "
                    "input shape with a reduced axis"
                )
        else:
            if len(scale_shape) != 1:
                raise UnsupportedOpError(
                    "QuantizeLinear supports per-tensor, per-axis, "
                    "and blocked scales only"
                )
            if scale_shape[0] != input_shape[axis]:
                raise ShapeInferenceError(
                    "QuantizeLinear scale length must match input axis size"
                )
    else:
        axis = None
        block_size = 0
    return QuantizeSpec(
        input_shape=input_shape,
        scale_shape=scale_shape,
        axis=axis,
        block_size=block_size or None,
        output_dtype=output_dtype,
    )


@register_lowering("QuantizeLinear")
def lower_quantize_linear(graph: Graph, node: Node) -> QuantizeLinearOp:
    op_dtype = _value_dtype(graph, node.inputs[0], node)
    scale_dtype = _value_dtype(graph, node.inputs[1], node)
    if not op_dtype.is_float or not scale_dtype.is_float:
        raise UnsupportedOpError(
            "QuantizeLinear supports float16/float/double inputs only"
        )
    spec = resolve_quantize_spec(graph, node)
    return QuantizeLinearOp(
        input0=node.inputs[0],
        scale=node.inputs[1],
        zero_point=optional_name(node.inputs, 2),
        output=node.outputs[0],
        axis=spec.axis,
        block_size=spec.block_size,
    )

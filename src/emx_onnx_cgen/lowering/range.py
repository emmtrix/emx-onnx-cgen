from __future__ import annotations

import math

from shared.scalar_types import ScalarType

from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from ..ir.ops import RangeOp
from ..lowering.common import (
    node_dtype,
    resolve_numeric_list_from_value,
    value_shape,
)
from .registry import register_lowering


_SUPPORTED_RANGE_DTYPES = {
    ScalarType.F32,
    ScalarType.F64,
    ScalarType.I16,
    ScalarType.I32,
    ScalarType.I64,
}


def _is_scalar_shape(shape: tuple[int, ...]) -> bool:
    return shape == () or shape == (1,)


@register_lowering("Range")
def lower_range(graph: Graph, node: Node) -> RangeOp:
    if len(node.inputs) != 3 or len(node.outputs) != 1:
        raise UnsupportedOpError("Range must have 3 inputs and 1 output")
    start_shape = value_shape(graph, node.inputs[0], node)
    limit_shape = value_shape(graph, node.inputs[1], node)
    delta_shape = value_shape(graph, node.inputs[2], node)
    if not (
        _is_scalar_shape(start_shape)
        and _is_scalar_shape(limit_shape)
        and _is_scalar_shape(delta_shape)
    ):
        raise UnsupportedOpError("Range inputs must be scalars")
    dtype = node_dtype(graph, node, *node.inputs, *node.outputs)
    if dtype not in _SUPPORTED_RANGE_DTYPES:
        raise UnsupportedOpError(f"Range does not support dtype {dtype.onnx_name}")
    output_shape = value_shape(graph, node.outputs[0], node)
    if len(output_shape) != 1:
        raise ShapeInferenceError("Range output must be 1D")
    start_values = resolve_numeric_list_from_value(graph, node.inputs[0], node)
    limit_values = resolve_numeric_list_from_value(graph, node.inputs[1], node)
    delta_values = resolve_numeric_list_from_value(graph, node.inputs[2], node)
    if (
        start_values is not None
        and limit_values is not None
        and delta_values is not None
        and len(start_values) == 1
        and len(limit_values) == 1
        and len(delta_values) == 1
    ):
        start_value = start_values[0]
        limit_value = limit_values[0]
        delta_value = delta_values[0]
        if float(delta_value) == 0.0:
            raise UnsupportedOpError("Range delta must be non-zero")
        raw_count = (float(limit_value) - float(start_value)) / float(delta_value)
        length = max(int(math.ceil(raw_count)), 0)
        if length < 0:
            raise ShapeInferenceError("Range output length must be non-negative")
        if output_shape[0] != length:
            raise ShapeInferenceError(
                f"Range output length must be {length}, got {output_shape[0]}"
            )
    else:
        length = output_shape[0]
        if length < 0:
            raise ShapeInferenceError("Range output length must be non-negative")
    return RangeOp(
        start=node.inputs[0],
        limit=node.inputs[1],
        delta=node.inputs[2],
        output=node.outputs[0],
    )

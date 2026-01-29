from __future__ import annotations

import math

import numpy as np

from shared.scalar_types import ScalarType

from ..ir.ops import RangeOp
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Initializer, Node
from ..lowering.common import node_dtype, value_shape, _shape_values_from_input
from .registry import register_lowering


_SUPPORTED_RANGE_DTYPES = {
    ScalarType.F32,
    ScalarType.F64,
    ScalarType.I16,
    ScalarType.I32,
    ScalarType.I64,
}


def _find_initializer(graph: Graph, name: str) -> Initializer | None:
    for initializer in graph.initializers:
        if initializer.name == name:
            return initializer
    return None


def _read_scalar_initializer(
    graph: Graph, name: str, node: Node, label: str
) -> float | int | None:
    initializer = _find_initializer(graph, name)
    if initializer is None:
        return None
    data = np.array(initializer.data)
    if data.size != 1:
        raise UnsupportedOpError(
            f"{node.op_type} {label} input must be a scalar"
        )
    return data.reshape(-1)[0].item()


def _read_scalar_value(
    graph: Graph, name: str, node: Node, label: str
) -> float | int | None:
    initializer_value = _read_scalar_initializer(graph, name, node, label)
    if initializer_value is not None:
        return initializer_value
    shape_values = _shape_values_from_input(graph, name, node)
    if shape_values is None:
        return None
    if len(shape_values) != 1:
        raise UnsupportedOpError(
            f"{node.op_type} {label} input must be a scalar"
        )
    return int(shape_values[0])


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
        raise UnsupportedOpError(
            f"Range does not support dtype {dtype.onnx_name}"
        )
    output_shape = value_shape(graph, node.outputs[0], node)
    if len(output_shape) != 1:
        raise ShapeInferenceError("Range output must be 1D")
    start_value = _read_scalar_value(graph, node.inputs[0], node, "start")
    limit_value = _read_scalar_value(graph, node.inputs[1], node, "limit")
    delta_value = _read_scalar_value(graph, node.inputs[2], node, "delta")
    if (
        start_value is not None
        and limit_value is not None
        and delta_value is not None
    ):
        if float(delta_value) == 0.0:
            raise UnsupportedOpError("Range delta must be non-zero")
        raw_count = (
            float(limit_value) - float(start_value)
        ) / float(delta_value)
        length = max(int(math.ceil(raw_count)), 0)
        if length < 0:
            raise ShapeInferenceError("Range output length must be non-negative")
        output_value = graph.find_value(node.outputs[0])
        output_dim_param = output_value.type.dim_params[0]
        if output_shape[0] != length and output_dim_param:
            output_shape = (length,)
        elif output_shape[0] != length:
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
        output_shape=output_shape,
        length=length,
        dtype=dtype,
        input_dtype=dtype,
    )

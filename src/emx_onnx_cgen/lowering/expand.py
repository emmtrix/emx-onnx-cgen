from __future__ import annotations

import numpy as np

from shared.scalar_types import ScalarType

from ..ir.ops import ExpandOp
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Initializer, Node
from ..lowering.common import value_dtype, value_shape
from .registry import register_lowering


def _find_initializer(graph: Graph, name: str) -> Initializer | None:
    for initializer in graph.initializers:
        if initializer.name == name:
            return initializer
    return None


def _read_shape_values(graph: Graph, name: str, node: Node) -> list[int] | None:
    initializer = _find_initializer(graph, name)
    if initializer is None:
        return None
    if initializer.type.dtype not in {ScalarType.I64, ScalarType.I32}:
        raise UnsupportedOpError(
            f"{node.op_type} shape input must be int64 or int32"
        )
    if len(initializer.type.shape) != 1:
        raise UnsupportedOpError(
            f"{node.op_type} shape input must be a 1D tensor"
        )
    values = np.array(initializer.data, dtype=np.int64).reshape(-1)
    if values.size == 0:
        raise ShapeInferenceError(
            f"{node.op_type} shape input cannot be empty"
        )
    return [int(value) for value in values]


def _validate_shape_input(graph: Graph, name: str, node: Node) -> None:
    dtype = value_dtype(graph, name, node)
    if dtype not in {ScalarType.I64, ScalarType.I32}:
        raise UnsupportedOpError(
            f"{node.op_type} shape input must be int64 or int32"
        )
    shape = value_shape(graph, name, node)
    if len(shape) != 1:
        raise UnsupportedOpError(
            f"{node.op_type} shape input must be a 1D tensor"
        )
    if shape[0] <= 0:
        raise ShapeInferenceError(
            f"{node.op_type} shape input cannot be empty"
        )


def _validate_static_dims(shape: tuple[int, ...], node: Node) -> None:
    if any(dim < 0 for dim in shape):
        raise ShapeInferenceError(
            f"{node.op_type} does not support dynamic dims"
        )


def _broadcast_shape(
    input_shape: tuple[int, ...], shape_values: list[int], node: Node
) -> tuple[int, ...]:
    _validate_static_dims(input_shape, node)
    for dim in shape_values:
        if dim < 0:
            raise ShapeInferenceError(
                f"{node.op_type} does not support dynamic dims"
            )
    output_rank = max(len(input_shape), len(shape_values))
    input_padded = (1,) * (output_rank - len(input_shape)) + input_shape
    shape_padded = (1,) * (output_rank - len(shape_values)) + tuple(shape_values)
    result: list[int] = []
    for input_dim, shape_dim in zip(input_padded, shape_padded):
        if input_dim == 1:
            result.append(shape_dim)
        elif shape_dim == 1:
            result.append(input_dim)
        elif input_dim == shape_dim:
            result.append(input_dim)
        else:
            raise ShapeInferenceError(
                f"{node.op_type} input shape {input_shape} is not "
                f"broadcastable to {shape_values}"
            )
    return tuple(result)


def _compute_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides: list[int] = []
    stride = 1
    for dim in reversed(shape):
        strides.append(stride)
        stride *= dim
    return tuple(reversed(strides))


@register_lowering("Expand")
def lower_expand(graph: Graph, node: Node) -> ExpandOp:
    if len(node.inputs) != 2 or len(node.outputs) != 1:
        raise UnsupportedOpError("Expand must have 2 inputs and 1 output")
    input_shape = value_shape(graph, node.inputs[0], node)
    output_shape = value_shape(graph, node.outputs[0], node)
    input_dtype = value_dtype(graph, node.inputs[0], node)
    output_dtype = value_dtype(graph, node.outputs[0], node)
    if input_dtype != output_dtype:
        raise UnsupportedOpError(
            f"{node.op_type} expects matching input/output dtypes, "
            f"got {input_dtype} and {output_dtype}"
        )
    shape_values = _read_shape_values(graph, node.inputs[1], node)
    if shape_values is not None:
        expected_output_shape = _broadcast_shape(input_shape, shape_values, node)
        _validate_static_dims(expected_output_shape, node)
        if output_shape and output_shape != expected_output_shape:
            raise ShapeInferenceError(
                f"{node.op_type} output shape must be {expected_output_shape}, "
                f"got {output_shape}"
            )
    else:
        _validate_shape_input(graph, node.inputs[1], node)
        if not output_shape:
            raise ShapeInferenceError(
                f"{node.op_type} output shape must be specified"
            )
        expected_output_shape = _broadcast_shape(
            input_shape, list(output_shape), node
        )
        if expected_output_shape != output_shape:
            raise ShapeInferenceError(
                f"{node.op_type} output shape must be {expected_output_shape}, "
                f"got {output_shape}"
            )
    input_shape_padded = (
        (1,) * (len(expected_output_shape) - len(input_shape)) + input_shape
    )
    input_strides = _compute_strides(input_shape_padded)
    return ExpandOp(
        input0=node.inputs[0],
        output=node.outputs[0],
        input_shape=input_shape,
        output_shape=expected_output_shape,
        input_shape_padded=input_shape_padded,
        input_strides=input_strides,
        dtype=input_dtype,
        input_dtype=input_dtype,
    )

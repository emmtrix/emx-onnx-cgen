from __future__ import annotations

import numpy as np

from shared.scalar_types import ScalarType

from ..ir.ops import TileOp
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Initializer, Node
from ..lowering.common import value_dtype, value_shape
from .registry import register_lowering


def _find_initializer(graph: Graph, name: str) -> Initializer | None:
    for initializer in graph.initializers:
        if initializer.name == name:
            return initializer
    return None


def _read_repeats(graph: Graph, name: str, node: Node) -> tuple[int, ...] | None:
    initializer = _find_initializer(graph, name)
    if initializer is None:
        return None
    if initializer.type.dtype not in {ScalarType.I64, ScalarType.I32}:
        raise UnsupportedOpError("Tile repeats input must be int64 or int32")
    if len(initializer.type.shape) != 1:
        raise UnsupportedOpError("Tile repeats input must be a 1D tensor")
    values = np.array(initializer.data, dtype=np.int64).reshape(-1)
    return tuple(int(value) for value in values)


def _infer_repeats_from_shapes(
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
) -> tuple[int, ...]:
    if len(input_shape) != len(output_shape):
        raise ShapeInferenceError(
            "Tile repeats must have the same rank as input shape"
        )
    repeats: list[int] = []
    for input_dim, output_dim in zip(input_shape, output_shape):
        if input_dim < 0 or output_dim < 0:
            raise ShapeInferenceError(
                "Tile repeats input must be constant when shapes are dynamic"
            )
        if input_dim == 0:
            if output_dim != 0:
                raise ShapeInferenceError(
                    "Tile output shape mismatch: "
                    f"expected 0 for dimension, got {output_dim}"
                )
            repeats.append(0)
            continue
        if output_dim % input_dim != 0:
            raise ShapeInferenceError(
                "Tile output shape mismatch: "
                f"expected multiple of {input_dim}, got {output_dim}"
            )
        repeats.append(int(output_dim // input_dim))
    return tuple(repeats)


def _compute_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides: list[int] = []
    stride = 1
    for dim in reversed(shape):
        strides.append(stride)
        stride *= dim
    return tuple(reversed(strides))


@register_lowering("Tile")
def lower_tile(graph: Graph, node: Node) -> TileOp:
    if len(node.inputs) != 2 or len(node.outputs) != 1:
        raise UnsupportedOpError("Tile must have 2 inputs and 1 output")
    input_shape = value_shape(graph, node.inputs[0], node)
    output_shape = value_shape(graph, node.outputs[0], node)
    input_dtype = value_dtype(graph, node.inputs[0], node)
    output_dtype = value_dtype(graph, node.outputs[0], node)
    if input_dtype != output_dtype:
        raise UnsupportedOpError(
            "Tile expects matching input/output dtypes, "
            f"got {input_dtype.onnx_name} and {output_dtype.onnx_name}"
        )
    repeats = _read_repeats(graph, node.inputs[1], node)
    if repeats is None:
        repeats_shape = value_shape(graph, node.inputs[1], node)
        repeats_dtype = value_dtype(graph, node.inputs[1], node)
        if repeats_dtype not in {ScalarType.I64, ScalarType.I32}:
            raise UnsupportedOpError("Tile repeats input must be int64 or int32")
        if len(repeats_shape) != 1:
            raise UnsupportedOpError("Tile repeats input must be a 1D tensor")
        repeats = _infer_repeats_from_shapes(input_shape, output_shape)
    if len(repeats) != len(input_shape):
        raise ShapeInferenceError(
            "Tile repeats must have the same rank as input shape"
        )
    if any(value < 0 for value in repeats):
        raise UnsupportedOpError("Tile repeats must be non-negative")
    expected_shape = tuple(
        int(dim) * int(repeat) for dim, repeat in zip(input_shape, repeats)
    )
    if output_shape and output_shape != expected_shape:
        raise ShapeInferenceError(
            "Tile output shape mismatch: "
            f"expected {expected_shape}, got {output_shape}"
        )
    return TileOp(
        input0=node.inputs[0],
        output=node.outputs[0],
        input_shape=input_shape,
        output_shape=expected_shape,
        repeats=repeats,
        input_strides=_compute_strides(input_shape),
        dtype=output_dtype,
        input_dtype=input_dtype,
    )

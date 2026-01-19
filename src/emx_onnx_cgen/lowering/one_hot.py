from __future__ import annotations

import numpy as np

from shared.scalar_types import ScalarType

from ..codegen.c_emitter import OneHotOp
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Initializer, Node
from ..lowering.common import value_dtype, value_shape
from .registry import register_lowering


def _find_initializer(graph: Graph, name: str) -> Initializer | None:
    for initializer in graph.initializers:
        if initializer.name == name:
            return initializer
    return None


def _read_scalar_initializer(
    graph: Graph, name: str, node: Node, label: str
) -> int | None:
    initializer = _find_initializer(graph, name)
    if initializer is None:
        return None
    data = np.array(initializer.data)
    if data.size != 1:
        raise UnsupportedOpError(
            f"{node.op_type} {label} input must be a scalar"
        )
    return int(data.reshape(-1)[0])


def _is_scalar_shape(shape: tuple[int, ...]) -> bool:
    return shape == () or shape == (1,)


def _normalize_onehot_axis(axis: int, rank: int, node: Node) -> int:
    if axis < 0:
        axis += rank + 1
    if axis < 0 or axis > rank:
        raise ShapeInferenceError(
            f"{node.op_type} axis {axis} is out of range for rank {rank}"
        )
    return axis


@register_lowering("OneHot")
def lower_onehot(graph: Graph, node: Node) -> OneHotOp:
    if len(node.inputs) != 3 or len(node.outputs) != 1:
        raise UnsupportedOpError("OneHot must have 3 inputs and 1 output")
    indices_name, depth_name, values_name = node.inputs
    indices_shape = value_shape(graph, indices_name, node)
    depth_shape = value_shape(graph, depth_name, node)
    values_shape = value_shape(graph, values_name, node)
    output_shape = value_shape(graph, node.outputs[0], node)
    if not _is_scalar_shape(depth_shape):
        raise UnsupportedOpError("OneHot depth input must be a scalar")
    if len(values_shape) != 1 or values_shape[0] != 2:
        raise UnsupportedOpError(
            "OneHot values input must be a 1D tensor of size 2"
        )
    output_rank = len(indices_shape) + 1
    if len(output_shape) != output_rank:
        raise ShapeInferenceError(
            f"OneHot output rank must be {output_rank}, got {len(output_shape)}"
        )
    axis = _normalize_onehot_axis(
        int(node.attrs.get("axis", -1)), len(indices_shape), node
    )
    depth_value = _read_scalar_initializer(graph, depth_name, node, "depth")
    if depth_value is not None:
        if depth_value < 0:
            raise ShapeInferenceError("OneHot depth must be non-negative")
        if output_shape[axis] != depth_value:
            raise ShapeInferenceError(
                "OneHot output depth must be "
                f"{depth_value}, got {output_shape[axis]}"
            )
        depth_dim = depth_value
    else:
        depth_dim = output_shape[axis]
        if depth_dim < 0:
            raise ShapeInferenceError("OneHot output depth must be non-negative")
    expected_output_shape = (
        indices_shape[:axis] + (depth_dim,) + indices_shape[axis:]
    )
    if output_shape != expected_output_shape:
        raise ShapeInferenceError(
            "OneHot output shape must be "
            f"{expected_output_shape}, got {output_shape}"
        )
    indices_dtype = value_dtype(graph, indices_name, node)
    depth_dtype = value_dtype(graph, depth_name, node)
    values_dtype = value_dtype(graph, values_name, node)
    output_dtype = value_dtype(graph, node.outputs[0], node)
    if indices_dtype.is_bool:
        raise UnsupportedOpError("OneHot indices must be numeric")
    if depth_dtype.is_bool:
        raise UnsupportedOpError("OneHot depth must be numeric")
    if values_dtype != output_dtype:
        raise UnsupportedOpError(
            "OneHot values dtype must match output dtype, "
            f"got {values_dtype.onnx_name} and {output_dtype.onnx_name}"
        )
    return OneHotOp(
        indices=indices_name,
        depth=depth_name,
        values=values_name,
        output=node.outputs[0],
        axis=axis,
        indices_shape=indices_shape,
        values_shape=values_shape,
        output_shape=output_shape,
        depth_dim=depth_dim,
        dtype=values_dtype,
        indices_dtype=indices_dtype,
        depth_dtype=depth_dtype,
    )

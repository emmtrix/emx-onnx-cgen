from __future__ import annotations

import numpy as np

from shared.scalar_types import ScalarType

from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.context import GraphContext
from ..ir.model import Graph, Initializer, Node
from ..ir.ops import ReshapeOp
from ..lowering.common import value_dtype, value_has_dim_params, value_shape
from .registry import register_lowering


def _find_initializer(graph: Graph, name: str) -> Initializer | None:
    for initializer in graph.initializers:
        if initializer.name == name:
            return initializer
    return None


def _resolve_axis(graph: Graph, node: Node) -> int | None:
    """Read the axis scalar from the initializer, or return None if dynamic."""
    axes_initializer = _find_initializer(graph, node.inputs[1])
    if axes_initializer is None:
        return None
    if axes_initializer.type.dtype not in {ScalarType.I64, ScalarType.I32}:
        raise UnsupportedOpError(
            "ExpandDims axis input must be int64 or int32, "
            f"got {axes_initializer.type.dtype.onnx_name}"
        )
    return int(np.array(axes_initializer.data).reshape(-1)[0])


def _normalize_axis(axis: int, output_rank: int) -> int | None:
    """Normalize the axis, returning None if it is out of range (ORT no-op)."""
    if axis < 0:
        axis += output_rank
    if axis < 0 or axis >= output_rank:
        return None
    return axis


@register_lowering("ExpandDims")
def lower_expand_dims(graph: Graph, node: Node) -> ReshapeOp:
    if len(node.inputs) != 2 or len(node.outputs) != 1:
        raise UnsupportedOpError("ExpandDims must have 2 inputs and 1 output")
    input_shape = value_shape(graph, node.inputs[0], node)
    output_shape = value_shape(graph, node.outputs[0], node)
    if value_has_dim_params(graph, node.outputs[0]):
        output_shape = ()
    for dim in input_shape:
        if dim < 0:
            raise ShapeInferenceError(
                f"{node.op_type} does not support dynamic dims in input"
            )
    input_dtype = value_dtype(graph, node.inputs[0], node)
    output_dtype = value_dtype(graph, node.outputs[0], node)
    if input_dtype != output_dtype:
        raise UnsupportedOpError(
            "ExpandDims expects matching input/output dtypes, "
            f"got {input_dtype.onnx_name} and {output_dtype.onnx_name}"
        )
    axis_dtype = value_dtype(graph, node.inputs[1], node)
    if axis_dtype not in {ScalarType.I64, ScalarType.I32}:
        raise UnsupportedOpError(
            f"ExpandDims axis input must be int64 or int32, got {axis_dtype.onnx_name}"
        )
    output_rank = len(input_shape) + 1
    axis = _resolve_axis(graph, node)
    if axis is not None:
        normalized_axis = _normalize_axis(axis, output_rank)
        if normalized_axis is None:
            # Out-of-range axis: ORT returns the input unchanged (identity).
            output_shape = input_shape
        else:
            expected_shape: list[int] = []
            input_index = 0
            for i in range(output_rank):
                if i == normalized_axis:
                    expected_shape.append(1)
                else:
                    expected_shape.append(input_shape[input_index])
                    input_index += 1
            if output_shape and tuple(expected_shape) != output_shape:
                raise ShapeInferenceError(
                    f"ExpandDims output shape must be {tuple(expected_shape)}, "
                    f"got {output_shape}"
                )
            output_shape = tuple(expected_shape)
    else:
        # Axis is a dynamic input; infer from the known output shape.
        # If the declared output shape matches the input shape (out-of-range axis
        # behavior from ORT), treat this as an identity op.
        if output_shape == input_shape:
            pass
        elif len(output_shape) != output_rank:
            raise ShapeInferenceError(
                f"ExpandDims output rank must be {output_rank}, got {len(output_shape)}"
            )
        else:
            for dim in output_shape:
                if dim < 0:
                    raise ShapeInferenceError(
                        f"{node.op_type} does not support dynamic dims in output"
                    )
            inserted = 0
            input_index = 0
            for dim in output_shape:
                if input_index < len(input_shape) and dim == input_shape[input_index]:
                    input_index += 1
                else:
                    if dim != 1:
                        raise ShapeInferenceError(
                            "ExpandDims output shape must insert exactly one 1-dim"
                        )
                    inserted += 1
            if inserted != 1 or input_index != len(input_shape):
                raise ShapeInferenceError(
                    "ExpandDims output shape must insert exactly one 1-dim into the input shape"
                )
    if isinstance(graph, GraphContext):
        graph.set_shape(node.outputs[0], output_shape)
    return ReshapeOp(
        input0=node.inputs[0],
        output=node.outputs[0],
    )

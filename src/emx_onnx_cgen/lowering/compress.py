from __future__ import annotations

from shared.scalar_types import ScalarType

from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.context import GraphContext
from ..ir.model import Graph, Node
from ..ir.ops import CompressOp
from ..lowering.common import value_dtype, value_shape
from ..validation import normalize_axis
from .registry import register_lowering


@register_lowering("Compress")
def lower_compress(graph: Graph, node: Node) -> CompressOp:
    if len(node.inputs) != 2 or len(node.outputs) != 1:
        raise UnsupportedOpError("Compress must have 2 inputs and 1 output")

    data_name, condition_name = node.inputs
    output_name = node.outputs[0]
    data_shape = value_shape(graph, data_name, node)
    condition_shape = value_shape(graph, condition_name, node)
    output_shape = value_shape(graph, output_name, node)

    if len(condition_shape) != 1:
        raise ShapeInferenceError(
            f"Compress condition must be rank 1, got shape {condition_shape}"
        )

    condition_dtype = value_dtype(graph, condition_name, node)
    if condition_dtype != ScalarType.BOOL:
        raise UnsupportedOpError(
            f"Compress condition must be bool, got {condition_dtype.onnx_name}"
        )

    axis_attr = node.attrs.get("axis")
    if axis_attr is None:
        axis: int | None = None
        data_element_count = 1
        for dim in data_shape:
            data_element_count *= dim
        if condition_shape[0] > data_element_count:
            raise ShapeInferenceError(
                "Compress condition length must be <= flattened data length, "
                f"got {condition_shape[0]} and {data_element_count}"
            )
        if len(output_shape) != 1:
            raise ShapeInferenceError(
                "Compress output must be rank 1 when axis is not provided, "
                f"got {output_shape}"
            )
    else:
        axis = normalize_axis(int(axis_attr), data_shape, node)
        if condition_shape[0] > data_shape[axis]:
            raise ShapeInferenceError(
                f"Compress condition length must be <= axis dimension, got {condition_shape[0]} and {data_shape[axis]}"
            )
        if len(output_shape) != len(data_shape):
            raise ShapeInferenceError(
                "Compress output rank must match data rank when axis is set, "
                f"got output shape {output_shape} and data shape {data_shape}"
            )

    if isinstance(graph, GraphContext):
        graph.set_shape(output_name, output_shape)

    return CompressOp(
        data=data_name,
        condition=condition_name,
        output=output_name,
        axis=axis,
    )

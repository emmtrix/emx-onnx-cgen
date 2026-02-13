from __future__ import annotations

from shared.scalar_types import ScalarType

from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from ..ir.ops import ScatterOp
from ..validation import normalize_axis
from .common import value_dtype, value_shape
from .registry import register_lowering


@register_lowering("Scatter")
def lower_scatter(graph: Graph, node: Node) -> ScatterOp:
    if len(node.inputs) != 3 or len(node.outputs) != 1:
        raise UnsupportedOpError("Scatter must have 3 inputs and 1 output")

    data_name, indices_name, updates_name = node.inputs
    output_name = node.outputs[0]

    data_shape = value_shape(graph, data_name, node)
    indices_shape = value_shape(graph, indices_name, node)
    updates_shape = value_shape(graph, updates_name, node)
    output_shape = value_shape(graph, output_name, node)

    if output_shape != data_shape:
        raise ShapeInferenceError(
            "Scatter output shape must match data shape, "
            f"got {output_shape} vs {data_shape}"
        )
    if indices_shape != updates_shape:
        raise ShapeInferenceError(
            "Scatter indices and updates shapes must match, "
            f"got {indices_shape} and {updates_shape}"
        )
    if len(indices_shape) != len(data_shape):
        raise ShapeInferenceError(
            "Scatter indices rank must match data rank, "
            f"got {len(indices_shape)} and {len(data_shape)}"
        )

    axis = normalize_axis(int(node.attrs.get("axis", 0)), data_shape, node)
    for dim_index, (data_dim, index_dim) in enumerate(zip(data_shape, indices_shape)):
        if dim_index == axis:
            continue
        if data_dim != index_dim:
            raise ShapeInferenceError(
                "Scatter data and indices must match on non-axis dimensions, "
                f"got {data_shape} and {indices_shape}"
            )

    data_dtype = value_dtype(graph, data_name, node)
    updates_dtype = value_dtype(graph, updates_name, node)
    if updates_dtype != data_dtype:
        raise UnsupportedOpError(
            "Scatter updates dtype must match data dtype, "
            f"got {updates_dtype.onnx_name} vs {data_dtype.onnx_name}"
        )

    indices_dtype = value_dtype(graph, indices_name, node)
    if indices_dtype not in {ScalarType.I32, ScalarType.I64}:
        raise UnsupportedOpError(
            "Scatter indices must be int32 or int64, " f"got {indices_dtype.onnx_name}"
        )

    return ScatterOp(
        data=data_name,
        indices=indices_name,
        updates=updates_name,
        output=output_name,
        axis=axis,
    )

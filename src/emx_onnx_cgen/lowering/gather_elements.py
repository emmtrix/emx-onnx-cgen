from __future__ import annotations

from shared.scalar_types import ScalarType

from ..ir.ops import GatherElementsOp
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from ..validation import normalize_axis
from .common import value_dtype as _value_dtype
from .common import value_shape as _value_shape
from .registry import register_lowering


@register_lowering("GatherElements")
def lower_gather_elements(graph: Graph, node: Node) -> GatherElementsOp:
    if len(node.inputs) != 2 or len(node.outputs) != 1:
        raise UnsupportedOpError("GatherElements must have 2 inputs and 1 output")
    data_name, indices_name = node.inputs
    data_shape = _value_shape(graph, data_name, node)
    indices_shape = _value_shape(graph, indices_name, node)
    output_shape = _value_shape(graph, node.outputs[0], node)
    if len(data_shape) != len(indices_shape):
        raise ShapeInferenceError(
            "GatherElements inputs must have matching ranks, "
            f"got {data_shape} and {indices_shape}"
        )
    if output_shape != indices_shape:
        raise ShapeInferenceError(
            "GatherElements output shape must match indices shape, "
            f"got {output_shape} and {indices_shape}"
        )
    axis = normalize_axis(int(node.attrs.get("axis", 0)), data_shape, node)
    for dim_index, (data_dim, index_dim) in enumerate(
        zip(data_shape, indices_shape)
    ):
        if dim_index != axis and data_dim != index_dim:
            raise ShapeInferenceError(
                "GatherElements inputs must match on non-axis dimensions, "
                f"got {data_shape} and {indices_shape}"
            )
    op_dtype = _value_dtype(graph, data_name, node)
    indices_dtype = _value_dtype(graph, indices_name, node)
    if indices_dtype not in {ScalarType.I64, ScalarType.I32}:
        raise UnsupportedOpError(
            "GatherElements indices must be int32 or int64, "
            f"got {indices_dtype.onnx_name}"
        )
    return GatherElementsOp(
        data=data_name,
        indices=indices_name,
        output=node.outputs[0],
        axis=axis,
        data_shape=data_shape,
        indices_shape=indices_shape,
        output_shape=output_shape,
        dtype=op_dtype,
        indices_dtype=indices_dtype,
    )

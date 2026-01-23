from __future__ import annotations

from shared.scalar_types import ScalarType

from ..ir.ops import GatherOp
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from ..validation import normalize_axis
from .common import value_dtype as _value_dtype
from .common import value_shape as _value_shape
from .registry import register_lowering


@register_lowering("Gather")
def lower_gather(graph: Graph, node: Node) -> GatherOp:
    if len(node.inputs) != 2 or len(node.outputs) != 1:
        raise UnsupportedOpError("Gather must have 2 inputs and 1 output")
    data_name, indices_name = node.inputs
    data_shape = _value_shape(graph, data_name, node)
    indices_shape = _value_shape(graph, indices_name, node)
    output_shape = _value_shape(graph, node.outputs[0], node)
    axis = normalize_axis(int(node.attrs.get("axis", 0)), data_shape, node)
    expected_output_shape = (
        data_shape[:axis] + indices_shape + data_shape[axis + 1 :]
    )
    if output_shape != expected_output_shape:
        raise ShapeInferenceError(
            "Gather output shape must be "
            f"{expected_output_shape}, got {output_shape}"
        )
    op_dtype = _value_dtype(graph, data_name, node)
    indices_dtype = _value_dtype(graph, indices_name, node)
    if indices_dtype not in {ScalarType.I64, ScalarType.I32}:
        raise UnsupportedOpError(
            "Gather indices must be int32 or int64, "
            f"got {indices_dtype.onnx_name}"
        )
    return GatherOp(
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

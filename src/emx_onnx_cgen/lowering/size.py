from __future__ import annotations

from shared.scalar_types import ScalarType

from ..ir.ops import SizeOp
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from .common import value_dtype, value_shape
from .registry import register_lowering


@register_lowering("Size")
def lower_size(graph: Graph, node: Node) -> SizeOp:
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        raise UnsupportedOpError("Size must have 1 input and 1 output")
    _ = value_shape(graph, node.inputs[0], node)
    output_shape = value_shape(graph, node.outputs[0], node)
    if len(output_shape) != 0:
        raise ShapeInferenceError("Size output must be a scalar")
    output_dtype = value_dtype(graph, node.outputs[0], node)
    if output_dtype != ScalarType.I64:
        raise UnsupportedOpError("Size output dtype must be int64")
    _ = value_dtype(graph, node.inputs[0], node)
    return SizeOp(
        input0=node.inputs[0],
        output=node.outputs[0],
    )

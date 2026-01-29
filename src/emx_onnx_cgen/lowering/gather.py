from __future__ import annotations

from ..errors import UnsupportedOpError
from ..ir.context import GraphContext
from ..ir.model import Graph, Node
from ..ir.ops import GatherOp
from ..lowering.common import value_shape
from ..validation import normalize_axis
from .registry import register_lowering


@register_lowering("Gather")
def lower_gather(graph: Graph, node: Node) -> GatherOp:
    if len(node.inputs) != 2 or len(node.outputs) != 1:
        raise UnsupportedOpError("Gather must have 2 inputs and 1 output")
    data_name, indices_name = node.inputs
    data_shape = value_shape(graph, data_name, node)
    indices_shape = value_shape(graph, indices_name, node)
    axis = normalize_axis(int(node.attrs.get("axis", 0)), data_shape, node)
    output_shape = data_shape[:axis] + indices_shape + data_shape[axis + 1 :]
    if isinstance(graph, GraphContext):
        graph.set_shape(node.outputs[0], output_shape)
    return GatherOp(
        data=data_name,
        indices=indices_name,
        output=node.outputs[0],
        axis=axis,
    )

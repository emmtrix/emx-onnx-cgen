from __future__ import annotations

from ..ir.ops import GatherOp
from ..errors import UnsupportedOpError
from ..ir.model import Graph, Node
from .registry import register_lowering


@register_lowering("Gather")
def lower_gather(graph: Graph, node: Node) -> GatherOp:
    if len(node.inputs) != 2 or len(node.outputs) != 1:
        raise UnsupportedOpError("Gather must have 2 inputs and 1 output")
    data_name, indices_name = node.inputs
    return GatherOp(
        data=data_name,
        indices=indices_name,
        output=node.outputs[0],
        axis=int(node.attrs.get("axis", 0)),
    )

from __future__ import annotations

from ..errors import UnsupportedOpError
from ..ir.model import Graph, Node
from ..ir.ops import ExpandOp
from .registry import register_lowering


@register_lowering("Expand")
def lower_expand(graph: Graph, node: Node) -> ExpandOp:
    if len(node.inputs) != 2 or len(node.outputs) != 1:
        raise UnsupportedOpError("Expand must have 2 inputs and 1 output")
    return ExpandOp(
        input0=node.inputs[0],
        input_shape=node.inputs[1],
        output=node.outputs[0],
    )

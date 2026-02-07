from __future__ import annotations

from ..ir.ops import NonZeroOp
from ..errors import UnsupportedOpError
from ..ir.model import Graph, Node
from .registry import register_lowering


@register_lowering("NonZero")
def lower_nonzero(graph: Graph, node: Node) -> NonZeroOp:
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        raise UnsupportedOpError("NonZero must have 1 input and 1 output")
    return NonZeroOp(
        input0=node.inputs[0],
        output=node.outputs[0],
    )

from __future__ import annotations

from ..ir.ops import MatMulOp
from ..errors import UnsupportedOpError
from ..ir.model import Graph, Node
from .registry import register_lowering


@register_lowering("MatMul")
def lower_matmul(graph: Graph, node: Node) -> MatMulOp:
    if len(node.inputs) != 2 or len(node.outputs) != 1:
        raise UnsupportedOpError("MatMul must have 2 inputs and 1 output")
    return MatMulOp(
        input0=node.inputs[0],
        input1=node.inputs[1],
        output=node.outputs[0],
    )

from __future__ import annotations

from ..ir.ops import GemmOp
from ..errors import UnsupportedOpError
from ..ir.model import Graph, Node
from .registry import register_lowering


@register_lowering("Gemm")
def lower_gemm(graph: Graph, node: Node) -> GemmOp:
    if len(node.inputs) not in {2, 3} or len(node.outputs) != 1:
        raise UnsupportedOpError("Gemm must have 2 or 3 inputs and 1 output")
    return GemmOp(
        input_a=node.inputs[0],
        input_b=node.inputs[1],
        input_c=node.inputs[2] if len(node.inputs) == 3 else None,
        output=node.outputs[0],
        alpha=float(node.attrs.get("alpha", 1.0)),
        beta=float(node.attrs.get("beta", 1.0)),
        trans_a=int(node.attrs.get("transA", 0)),
        trans_b=int(node.attrs.get("transB", 0)),
    )

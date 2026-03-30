from __future__ import annotations

from ..errors import UnsupportedOpError
from ..ir.model import Graph, Node
from ..ir.ops import SampleOp
from .common import node_dtype, value_shape
from .registry import register_lowering


@register_lowering("SampleOp")
def lower_sample_op(graph: Graph, node: Node) -> SampleOp:
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        raise UnsupportedOpError("SampleOp must have exactly 1 input and 1 output")

    op_dtype = node_dtype(graph, node, *node.inputs, *node.outputs)
    shape = value_shape(graph, node.inputs[0], node)

    return SampleOp(
        input0=node.inputs[0],
        output=node.outputs[0],
        shape=shape,
        dtype=op_dtype,
    )

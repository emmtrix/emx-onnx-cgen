from __future__ import annotations

from ..errors import UnsupportedOpError
from ..ir.model import Graph, Node, SequenceType
from ..ir.ops import SequenceEmptyOp
from .registry import register_lowering


@register_lowering("SequenceEmpty")
def lower_sequence_empty(graph: Graph, node: Node) -> SequenceEmptyOp:
    if node.inputs or len(node.outputs) != 1:
        raise UnsupportedOpError("SequenceEmpty must have 0 inputs and 1 output")

    output_sequence = node.outputs[0]
    if not output_sequence:
        raise UnsupportedOpError("SequenceEmpty output must be provided")

    output_value = graph.find_value(output_sequence)
    if not isinstance(output_value.type, SequenceType):
        raise UnsupportedOpError("SequenceEmpty output must be a sequence<tensor<...>>")

    return SequenceEmptyOp(output_sequence=output_sequence)

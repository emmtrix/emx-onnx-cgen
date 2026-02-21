from __future__ import annotations

from shared.scalar_types import ScalarType

from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from ..ir.ops import SequenceLengthOp
from .common import value_dtype, value_shape
from .registry import register_lowering
from .sequence_insert import _sequence_type


@register_lowering("SequenceLength")
def lower_sequence_length(graph: Graph, node: Node) -> SequenceLengthOp:
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        raise UnsupportedOpError("SequenceLength must have 1 input and 1 output")
    input_sequence = node.inputs[0]
    if not input_sequence:
        raise UnsupportedOpError("SequenceLength input sequence must be provided")
    output = node.outputs[0]
    if not output:
        raise UnsupportedOpError("SequenceLength output must be provided")

    _sequence_type(graph, input_sequence, node)

    output_dtype = value_dtype(graph, output, node)
    if output_dtype != ScalarType.I64:
        raise UnsupportedOpError("SequenceLength output dtype must be int64")
    output_shape = value_shape(graph, output, node)
    if output_shape != ():
        raise ShapeInferenceError("SequenceLength output must be a scalar")

    return SequenceLengthOp(input_sequence=input_sequence, output=output)

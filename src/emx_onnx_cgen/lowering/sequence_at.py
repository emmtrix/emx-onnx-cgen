from __future__ import annotations

from shared.scalar_types import ScalarType

from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node, TensorType
from ..ir.ops import SequenceAtOp
from .common import value_dtype, value_shape
from .registry import register_lowering
from .sequence_insert import _sequence_type


@register_lowering("SequenceAt")
def lower_sequence_at(graph: Graph, node: Node) -> SequenceAtOp:
    if len(node.inputs) != 2 or len(node.outputs) != 1:
        raise UnsupportedOpError("SequenceAt must have 2 inputs and 1 output")
    input_sequence, position_name = node.inputs
    if not input_sequence or not position_name:
        raise UnsupportedOpError("SequenceAt inputs must be provided")
    output = node.outputs[0]
    if not output:
        raise UnsupportedOpError("SequenceAt output must be provided")

    input_sequence_type = _sequence_type(graph, input_sequence, node)
    output_value = graph.find_value(output)
    if not isinstance(output_value.type, TensorType):
        raise UnsupportedOpError("SequenceAt output must be a tensor")
    if output_value.type.dtype != input_sequence_type.elem.dtype:
        raise UnsupportedOpError(
            "SequenceAt output dtype must match sequence element dtype"
        )

    pos_dtype = value_dtype(graph, position_name, node)
    if pos_dtype not in {ScalarType.I32, ScalarType.I64}:
        raise UnsupportedOpError("SequenceAt position must be int32 or int64")
    pos_shape = value_shape(graph, position_name, node)
    if pos_shape not in {(), (1,)}:
        raise ShapeInferenceError(
            "SequenceAt position must be a scalar or size-1 tensor"
        )

    return SequenceAtOp(
        input_sequence=input_sequence,
        position=position_name,
        output=output,
    )

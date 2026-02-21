from __future__ import annotations

from shared.scalar_types import ScalarType

from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from ..ir.ops import SequenceEraseOp
from .common import optional_name, resolve_int_list_from_value, value_dtype, value_shape
from .registry import register_lowering
from .sequence_insert import _sequence_type


@register_lowering("SequenceErase")
def lower_sequence_erase(graph: Graph, node: Node) -> SequenceEraseOp:
    if len(node.inputs) not in {1, 2} or len(node.outputs) != 1:
        raise UnsupportedOpError("SequenceErase must have 1 or 2 inputs and 1 output")
    input_sequence = optional_name(node.inputs, 0)
    if input_sequence is None:
        raise UnsupportedOpError("SequenceErase input sequence must be provided")
    position_name = optional_name(node.inputs, 1)
    output_sequence = optional_name(node.outputs, 0)
    if output_sequence is None:
        raise UnsupportedOpError("SequenceErase output must be provided")

    input_sequence_type = _sequence_type(graph, input_sequence, node)
    output_sequence_type = _sequence_type(graph, output_sequence, node)
    if output_sequence_type.elem.dtype != input_sequence_type.elem.dtype:
        raise UnsupportedOpError(
            "SequenceErase output sequence dtype must match input sequence dtype"
        )

    if position_name is not None:
        pos_dtype = value_dtype(graph, position_name, node)
        if pos_dtype not in {ScalarType.I32, ScalarType.I64}:
            raise UnsupportedOpError("SequenceErase position must be int32 or int64")
        pos_shape = value_shape(graph, position_name, node)
        if pos_shape not in {(), (1,)}:
            raise ShapeInferenceError("SequenceErase position must be a scalar or size-1 tensor")
        resolve_int_list_from_value(graph, position_name, node)

    return SequenceEraseOp(
        input_sequence=input_sequence,
        position=position_name,
        output_sequence=output_sequence,
    )

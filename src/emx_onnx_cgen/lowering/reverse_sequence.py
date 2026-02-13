from __future__ import annotations

from shared.scalar_types import ScalarType

from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.context import GraphContext
from ..ir.model import Graph, Node
from ..ir.ops import ReverseSequenceOp
from .common import resolve_int_list_from_value, value_dtype, value_shape
from .registry import register_lowering
from ..validation import normalize_axis


@register_lowering("ReverseSequence")
def lower_reverse_sequence(graph: Graph, node: Node) -> ReverseSequenceOp:
    if len(node.inputs) != 2 or len(node.outputs) != 1:
        raise UnsupportedOpError("ReverseSequence must have 2 inputs and 1 output")
    input_name, sequence_lens_name = node.inputs
    if not input_name or not sequence_lens_name:
        raise UnsupportedOpError("ReverseSequence inputs must be provided")
    input_shape = value_shape(graph, input_name, node)
    if len(input_shape) < 2:
        raise ShapeInferenceError("ReverseSequence input rank must be at least 2")
    output_shape = value_shape(graph, node.outputs[0], node)
    if output_shape and output_shape != input_shape:
        raise ShapeInferenceError("ReverseSequence output shape must match input shape")
    batch_axis = normalize_axis(int(node.attrs.get("batch_axis", 1)), input_shape, node)
    time_axis = normalize_axis(int(node.attrs.get("time_axis", 0)), input_shape, node)
    if batch_axis == time_axis:
        raise UnsupportedOpError("ReverseSequence batch_axis and time_axis must differ")

    seq_dtype = value_dtype(graph, sequence_lens_name, node)
    if seq_dtype not in {ScalarType.I32, ScalarType.I64}:
        raise UnsupportedOpError("ReverseSequence sequence_lens must be int32 or int64")
    seq_shape = value_shape(graph, sequence_lens_name, node)
    if len(seq_shape) != 1:
        raise ShapeInferenceError("ReverseSequence sequence_lens must be a 1D tensor")
    batch_size = input_shape[batch_axis]
    if seq_shape[0] != batch_size:
        raise ShapeInferenceError(
            "ReverseSequence sequence_lens length must match input batch axis"
        )

    seq_values = resolve_int_list_from_value(graph, sequence_lens_name, node)
    if seq_values is not None:
        time_size = input_shape[time_axis]
        for seq_len in seq_values:
            if seq_len < 0 or seq_len > time_size:
                raise ShapeInferenceError(
                    "ReverseSequence sequence_lens values must be in [0, time_axis dim]"
                )

    if isinstance(graph, GraphContext):
        graph.set_shape(node.outputs[0], input_shape)

    return ReverseSequenceOp(
        input0=input_name,
        sequence_lens=sequence_lens_name,
        output=node.outputs[0],
        batch_axis=batch_axis,
        time_axis=time_axis,
    )

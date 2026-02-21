from __future__ import annotations

from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node, SequenceType, TensorType
from ..ir.ops import SequenceConstructOp
from .registry import register_lowering


@register_lowering("SequenceConstruct")
def lower_sequence_construct(graph: Graph, node: Node) -> SequenceConstructOp:
    if len(node.outputs) != 1 or not node.outputs[0]:
        raise UnsupportedOpError("SequenceConstruct must have exactly one output")
    if not node.inputs:
        raise UnsupportedOpError("SequenceConstruct requires at least one input tensor")
    if any(not name for name in node.inputs):
        raise UnsupportedOpError("SequenceConstruct inputs must be provided")

    output_sequence_name = node.outputs[0]
    output_value = graph.find_value(output_sequence_name)
    if not isinstance(output_value.type, SequenceType):
        raise UnsupportedOpError(
            "SequenceConstruct output must be a sequence<tensor<...>> value"
        )
    output_elem_type = output_value.type.elem

    input_names = tuple(node.inputs)
    first_input_type: TensorType | None = None
    for input_name in input_names:
        value = graph.find_value(input_name)
        if not isinstance(value.type, TensorType):
            raise UnsupportedOpError(
                f"SequenceConstruct expects tensor inputs; '{input_name}' is a sequence"
            )
        tensor_type = value.type
        if first_input_type is None:
            first_input_type = tensor_type
        elif tensor_type.dtype != first_input_type.dtype:
            raise UnsupportedOpError(
                "SequenceConstruct requires all input tensors to have the same dtype"
            )
        elif tensor_type.shape != first_input_type.shape:
            raise ShapeInferenceError(
                "SequenceConstruct requires all input tensors to have the same shape"
            )

    if first_input_type is None:
        raise UnsupportedOpError("SequenceConstruct requires at least one input tensor")

    if output_elem_type.dtype != first_input_type.dtype:
        raise UnsupportedOpError(
            "SequenceConstruct output sequence dtype must match input tensor dtype"
        )
    if output_elem_type.shape != first_input_type.shape:
        raise ShapeInferenceError(
            "SequenceConstruct output sequence element shape must match input tensor shape"
        )

    return SequenceConstructOp(inputs=input_names, output_sequence=output_sequence_name)

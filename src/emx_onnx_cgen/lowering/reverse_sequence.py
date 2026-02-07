from __future__ import annotations

from shared.scalar_types import ScalarType

from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.context import GraphContext
from ..ir.model import Graph, Node
from ..ir.ops import ReverseSequenceOp
from ..lowering.common import value_dtype, value_shape
from ..validation import normalize_axis
from .registry import register_lowering


@register_lowering("ReverseSequence")
def lower_reverse_sequence(graph: Graph, node: Node) -> ReverseSequenceOp:
    if len(node.inputs) != 2 or len(node.outputs) != 1:
        raise UnsupportedOpError("ReverseSequence must have 2 inputs and 1 output")
    input_name, sequence_lens_name = node.inputs
    output_name = node.outputs[0]
    if not input_name or not sequence_lens_name or not output_name:
        raise UnsupportedOpError("ReverseSequence requires all inputs and outputs")

    input_shape = value_shape(graph, input_name, node)
    output_shape = value_shape(graph, output_name, node)
    if len(input_shape) < 2:
        raise ShapeInferenceError(
            f"{node.op_type} input rank must be >= 2, got {len(input_shape)}"
        )
    if any(dim < 0 for dim in input_shape):
        raise ShapeInferenceError(f"{node.op_type} does not support dynamic dims")

    batch_axis = normalize_axis(int(node.attrs.get("batch_axis", 1)), input_shape, node)
    time_axis = normalize_axis(int(node.attrs.get("time_axis", 0)), input_shape, node)
    if batch_axis == time_axis:
        raise UnsupportedOpError(
            f"{node.op_type} batch_axis and time_axis must differ, got {batch_axis}"
        )

    if output_shape and output_shape != input_shape:
        raise ShapeInferenceError(
            f"{node.op_type} output shape must be {input_shape}, got {output_shape}"
        )

    sequence_lens_shape = value_shape(graph, sequence_lens_name, node)
    if len(sequence_lens_shape) != 1:
        raise UnsupportedOpError(
            f"{node.op_type} sequence_lens must be 1D, got shape {sequence_lens_shape}"
        )
    expected_batch = input_shape[batch_axis]
    if sequence_lens_shape[0] != expected_batch:
        raise ShapeInferenceError(
            f"{node.op_type} sequence_lens length must match input batch axis "
            f"dimension {expected_batch}, got {sequence_lens_shape[0]}"
        )

    sequence_lens_dtype = value_dtype(graph, sequence_lens_name, node)
    if sequence_lens_dtype not in {ScalarType.I32, ScalarType.I64}:
        raise UnsupportedOpError(
            f"{node.op_type} sequence_lens dtype must be int32 or int64, "
            f"got {sequence_lens_dtype.onnx_name}"
        )

    dtype = value_dtype(graph, input_name, node)
    output_dtype = value_dtype(graph, output_name, node)
    if output_dtype != dtype:
        raise UnsupportedOpError(
            f"{node.op_type} input/output dtypes must match, got "
            f"{dtype.onnx_name} and {output_dtype.onnx_name}"
        )

    if isinstance(graph, GraphContext):
        graph.set_shape(output_name, input_shape)

    return ReverseSequenceOp(
        input0=input_name,
        sequence_lens=sequence_lens_name,
        output=output_name,
        input_shape=input_shape,
        output_shape=input_shape,
        batch_axis=batch_axis,
        time_axis=time_axis,
        dtype=dtype,
        input_dtype=dtype,
        sequence_lens_dtype=sequence_lens_dtype,
    )

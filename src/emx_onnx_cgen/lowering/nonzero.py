from __future__ import annotations

from shared.scalar_types import ScalarType

from ..ir.ops import NonZeroOp
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from .common import value_dtype, value_shape
from .registry import register_lowering


@register_lowering("NonZero")
def lower_nonzero(graph: Graph, node: Node) -> NonZeroOp:
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        raise UnsupportedOpError("NonZero must have 1 input and 1 output")
    input_shape = value_shape(graph, node.inputs[0], node)
    if len(input_shape) == 0:
        raise UnsupportedOpError("NonZero does not support scalar inputs")
    output_shape = value_shape(graph, node.outputs[0], node)
    if len(output_shape) != 2:
        raise ShapeInferenceError("NonZero output must be 2D")
    if output_shape[0] != len(input_shape):
        raise ShapeInferenceError(
            "NonZero output shape must be "
            f"({len(input_shape)}, N), got {output_shape}"
        )
    if output_shape[0] < 0 or output_shape[1] < 0:
        raise ShapeInferenceError(
            "NonZero output shape must be non-negative"
        )
    output_dtype = value_dtype(graph, node.outputs[0], node)
    if output_dtype != ScalarType.I64:
        raise UnsupportedOpError("NonZero output dtype must be int64")
    input_dtype = value_dtype(graph, node.inputs[0], node)
    return NonZeroOp(
        input0=node.inputs[0],
        output=node.outputs[0],
        input_shape=input_shape,
        output_shape=output_shape,
        dtype=output_dtype,
        input_dtype=input_dtype,
    )

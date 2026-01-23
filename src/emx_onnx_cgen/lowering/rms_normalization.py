from __future__ import annotations

from ..ir.ops import RMSNormalizationOp
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from ..validation import ensure_output_shape_matches_input
from ..validation import normalize_axis
from .common import node_dtype, shape_product, value_shape
from .registry import register_lowering


def _ensure_broadcastable(
    name: str,
    shape: tuple[int, ...],
    normalized_shape: tuple[int, ...],
) -> None:
    if len(shape) != len(normalized_shape):
        raise ShapeInferenceError(
            f"RMSNormalization {name} rank must match normalized rank"
        )
    for dim, expected in zip(shape, normalized_shape):
        if dim not in {1, expected}:
            raise ShapeInferenceError(
                f"RMSNormalization {name} shape {shape} must be broadcastable "
                f"to {normalized_shape}"
            )


@register_lowering("RMSNormalization")
def lower_rms_normalization(
    graph: Graph, node: Node
) -> RMSNormalizationOp:
    if len(node.inputs) != 2 or len(node.outputs) != 1:
        raise UnsupportedOpError("RMSNormalization must have 2 inputs and 1 output")
    op_dtype = node_dtype(graph, node, *node.inputs, *node.outputs)
    if not op_dtype.is_float:
        raise UnsupportedOpError(
            "RMSNormalization supports float16, float, and double inputs only"
        )
    input_shape = value_shape(graph, node.inputs[0], node)
    output_shape = value_shape(graph, node.outputs[0], node)
    ensure_output_shape_matches_input(node, input_shape, output_shape)
    axis = normalize_axis(int(node.attrs.get("axis", -1)), input_shape, node)
    normalized_shape = input_shape[axis:]
    scale_shape = value_shape(graph, node.inputs[1], node)
    _ensure_broadcastable("scale", scale_shape, normalized_shape)
    epsilon = float(node.attrs.get("epsilon", 1e-5))
    stash_type = int(node.attrs.get("stash_type", 1))
    if stash_type != 1:
        raise UnsupportedOpError(
            "RMSNormalization supports stash_type=1 only"
        )
    outer = shape_product(input_shape[:axis]) if axis > 0 else 1
    inner = shape_product(normalized_shape)
    return RMSNormalizationOp(
        input0=node.inputs[0],
        scale=node.inputs[1],
        output=node.outputs[0],
        shape=input_shape,
        normalized_shape=normalized_shape,
        scale_shape=scale_shape,
        outer=outer,
        inner=inner,
        axis=axis,
        epsilon=epsilon,
        dtype=op_dtype,
    )

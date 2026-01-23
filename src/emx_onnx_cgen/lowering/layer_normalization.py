from __future__ import annotations

from ..ir.ops import LayerNormalizationOp
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from ..validation import ensure_output_shape_matches_input
from ..validation import normalize_axis
from .common import node_dtype, shape_product, value_dtype, value_shape
from .registry import register_lowering


def _ensure_broadcastable(
    name: str,
    shape: tuple[int, ...],
    normalized_shape: tuple[int, ...],
) -> None:
    if len(shape) != len(normalized_shape):
        raise ShapeInferenceError(
            f"LayerNormalization {name} rank must match normalized rank"
        )
    for dim, expected in zip(shape, normalized_shape):
        if dim not in {1, expected}:
            raise ShapeInferenceError(
                f"LayerNormalization {name} shape {shape} must be broadcastable "
                f"to {normalized_shape}"
            )


@register_lowering("LayerNormalization")
def lower_layer_normalization(
    graph: Graph, node: Node
) -> LayerNormalizationOp:
    if len(node.inputs) < 2 or len(node.inputs) > 3:
        raise UnsupportedOpError(
            "LayerNormalization must have 2 or 3 inputs"
        )
    if len(node.outputs) < 1 or len(node.outputs) > 3:
        raise UnsupportedOpError(
            "LayerNormalization must have 1 to 3 outputs"
        )
    op_dtype = node_dtype(graph, node, *node.inputs, node.outputs[0])
    if not op_dtype.is_float:
        raise UnsupportedOpError(
            "LayerNormalization supports float16, float, and double inputs only"
        )
    input_shape = value_shape(graph, node.inputs[0], node)
    output_shape = value_shape(graph, node.outputs[0], node)
    ensure_output_shape_matches_input(node, input_shape, output_shape)
    axis = normalize_axis(int(node.attrs.get("axis", -1)), input_shape, node)
    normalized_shape = input_shape[axis:]
    scale_shape = value_shape(graph, node.inputs[1], node)
    _ensure_broadcastable("scale", scale_shape, normalized_shape)
    bias_input = node.inputs[2] if len(node.inputs) > 2 and node.inputs[2] else None
    bias_shape = None
    if bias_input is not None:
        bias_shape = value_shape(graph, bias_input, node)
        _ensure_broadcastable("bias", bias_shape, normalized_shape)
    epsilon = float(node.attrs.get("epsilon", 1e-5))
    stash_type = int(node.attrs.get("stash_type", 1))
    if stash_type != 1:
        raise UnsupportedOpError(
            "LayerNormalization supports stash_type=1 only"
        )
    mean_output = node.outputs[1] if len(node.outputs) > 1 else None
    invstd_output = node.outputs[2] if len(node.outputs) > 2 else None
    if mean_output is not None:
        mean_dtype = value_dtype(graph, mean_output, node)
        if mean_dtype != op_dtype:
            raise UnsupportedOpError(
                "LayerNormalization expects mean output dtype to match input"
            )
        expected_mean_shape = input_shape[:axis] + (1,) * len(normalized_shape)
        mean_shape = value_shape(graph, mean_output, node)
        if mean_shape != expected_mean_shape:
            raise ShapeInferenceError(
                "LayerNormalization mean output shape must be "
                f"{expected_mean_shape}, got {mean_shape}"
            )
    if invstd_output is not None:
        invstd_dtype = value_dtype(graph, invstd_output, node)
        if invstd_dtype != op_dtype:
            raise UnsupportedOpError(
                "LayerNormalization expects invstd output dtype to match input"
            )
        expected_invstd_shape = input_shape[:axis] + (1,) * len(normalized_shape)
        invstd_shape = value_shape(graph, invstd_output, node)
        if invstd_shape != expected_invstd_shape:
            raise ShapeInferenceError(
                "LayerNormalization invstd output shape must be "
                f"{expected_invstd_shape}, got {invstd_shape}"
            )
    outer = shape_product(input_shape[:axis]) if axis > 0 else 1
    inner = shape_product(normalized_shape)
    return LayerNormalizationOp(
        input0=node.inputs[0],
        scale=node.inputs[1],
        bias=bias_input,
        output=node.outputs[0],
        mean_output=mean_output,
        invstd_output=invstd_output,
        shape=input_shape,
        normalized_shape=normalized_shape,
        scale_shape=scale_shape,
        bias_shape=bias_shape,
        outer=outer,
        inner=inner,
        axis=axis,
        epsilon=epsilon,
        dtype=op_dtype,
    )

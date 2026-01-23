from __future__ import annotations

from ..ir.ops import GroupNormalizationOp
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from ..validation import ensure_output_shape_matches_input
from .common import node_dtype, shape_product, value_shape
from .registry import register_lowering


@register_lowering("GroupNormalization")
def lower_group_normalization(
    graph: Graph, node: Node
) -> GroupNormalizationOp:
    if len(node.inputs) != 3 or len(node.outputs) != 1:
        raise UnsupportedOpError(
            "GroupNormalization must have 3 inputs and 1 output"
        )
    op_dtype = node_dtype(graph, node, *node.inputs, *node.outputs)
    if not op_dtype.is_float:
        raise UnsupportedOpError(
            "GroupNormalization supports float16, float, and double inputs only"
        )
    input_shape = value_shape(graph, node.inputs[0], node)
    output_shape = value_shape(graph, node.outputs[0], node)
    ensure_output_shape_matches_input(node, input_shape, output_shape)
    if len(input_shape) < 3:
        raise ShapeInferenceError(
            "GroupNormalization expects input rank of at least 3"
        )
    channels = input_shape[1]
    num_groups_attr = node.attrs.get("num_groups")
    if num_groups_attr is None:
        raise UnsupportedOpError("GroupNormalization requires num_groups")
    num_groups = int(num_groups_attr)
    if num_groups <= 0:
        raise ShapeInferenceError("GroupNormalization num_groups must be > 0")
    if channels % num_groups != 0:
        raise ShapeInferenceError(
            "GroupNormalization num_groups must divide the channel dimension"
        )
    scale_shape = value_shape(graph, node.inputs[1], node)
    bias_shape = value_shape(graph, node.inputs[2], node)
    if scale_shape != (channels,) or bias_shape != (channels,):
        raise ShapeInferenceError(
            "GroupNormalization scale and bias must be 1D with length C"
        )
    spatial_size = shape_product(input_shape[2:])
    group_size = channels // num_groups
    epsilon = float(node.attrs.get("epsilon", 1e-5))
    stash_type = int(node.attrs.get("stash_type", 1))
    if stash_type != 1:
        raise UnsupportedOpError(
            "GroupNormalization supports stash_type=1 only"
        )
    return GroupNormalizationOp(
        input0=node.inputs[0],
        scale=node.inputs[1],
        bias=node.inputs[2],
        output=node.outputs[0],
        shape=input_shape,
        channels=channels,
        num_groups=num_groups,
        group_size=group_size,
        spatial_size=spatial_size,
        epsilon=epsilon,
        dtype=op_dtype,
    )

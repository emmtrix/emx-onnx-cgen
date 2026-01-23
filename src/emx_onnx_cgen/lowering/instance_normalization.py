from __future__ import annotations

from ..ir.ops import InstanceNormalizationOp
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from ..validation import ensure_output_shape_matches_input
from .common import node_dtype, shape_product, value_shape
from .registry import register_lowering


@register_lowering("InstanceNormalization")
def lower_instance_normalization(
    graph: Graph, node: Node
) -> InstanceNormalizationOp:
    if len(node.inputs) != 3 or len(node.outputs) != 1:
        raise UnsupportedOpError(
            "InstanceNormalization must have 3 inputs and 1 output"
        )
    op_dtype = node_dtype(graph, node, *node.inputs, *node.outputs)
    if not op_dtype.is_float:
        raise UnsupportedOpError(
            "InstanceNormalization supports float16, float, and double inputs only"
        )
    input_shape = value_shape(graph, node.inputs[0], node)
    output_shape = value_shape(graph, node.outputs[0], node)
    ensure_output_shape_matches_input(node, input_shape, output_shape)
    if len(input_shape) < 3:
        raise ShapeInferenceError(
            "InstanceNormalization expects input rank of at least 3"
        )
    channels = input_shape[1]
    scale_shape = value_shape(graph, node.inputs[1], node)
    bias_shape = value_shape(graph, node.inputs[2], node)
    if scale_shape != (channels,) or bias_shape != (channels,):
        raise ShapeInferenceError(
            "InstanceNormalization scale and bias must be 1D with length C"
        )
    spatial_size = shape_product(input_shape[2:])
    epsilon = float(node.attrs.get("epsilon", 1e-5))
    return InstanceNormalizationOp(
        input0=node.inputs[0],
        scale=node.inputs[1],
        bias=node.inputs[2],
        output=node.outputs[0],
        shape=input_shape,
        channels=channels,
        spatial_size=spatial_size,
        epsilon=epsilon,
        dtype=op_dtype,
    )

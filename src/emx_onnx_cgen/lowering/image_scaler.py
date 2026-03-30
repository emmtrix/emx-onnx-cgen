from __future__ import annotations

from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from ..ir.ops import ImageScalerOp
from .common import node_dtype, value_shape
from .registry import register_lowering


@register_lowering("ImageScaler")
def lower_image_scaler(graph: Graph, node: Node) -> ImageScalerOp:
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        raise UnsupportedOpError("ImageScaler must have exactly 1 input and 1 output")

    op_dtype = node_dtype(graph, node, *node.inputs, *node.outputs)
    if not op_dtype.is_float:
        raise UnsupportedOpError("ImageScaler supports float inputs only")

    input_shape = value_shape(graph, node.inputs[0], node)
    if len(input_shape) != 4:
        raise UnsupportedOpError(
            f"ImageScaler expects rank-4 input (NCHW), got rank {len(input_shape)}"
        )

    output_shape = value_shape(graph, node.outputs[0], node)
    if output_shape != input_shape:
        raise ShapeInferenceError(
            f"ImageScaler output shape {output_shape} does not match input {input_shape}"
        )

    scale = float(node.attrs.get("scale", 1.0))
    bias_raw = node.attrs.get("bias", [])
    bias = tuple(float(b) for b in bias_raw)

    channels = input_shape[1]
    if len(bias) != channels:
        raise UnsupportedOpError(
            f"ImageScaler bias length {len(bias)} must match channels {channels}"
        )

    return ImageScalerOp(
        input0=node.inputs[0],
        output=node.outputs[0],
        shape=input_shape,
        channels=channels,
        scale=scale,
        bias=bias,
        dtype=op_dtype,
    )

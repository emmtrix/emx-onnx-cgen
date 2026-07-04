from __future__ import annotations

from shared.scalar_types import ScalarType

from ..errors import UnsupportedOpError
from ..ir.model import Graph, Node
from ..ir.ops import ImageDecoderOp
from .common import node_dtype, value_shape
from .registry import register_lowering

_PIXEL_FORMAT_CHANNELS = {"RGB": 3, "BGR": 3, "Grayscale": 1}


@register_lowering("ImageDecoder")
def lower_image_decoder(graph: Graph, node: Node) -> ImageDecoderOp:
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        raise UnsupportedOpError("ImageDecoder must have exactly 1 input and 1 output")

    op_dtype = node_dtype(graph, node, *node.inputs, *node.outputs)
    if op_dtype is not ScalarType.U8:
        raise UnsupportedOpError(
            f"ImageDecoder supports uint8 input/output only, got {op_dtype.onnx_name}"
        )

    input_shape = value_shape(graph, node.inputs[0], node)
    if len(input_shape) != 1:
        raise UnsupportedOpError(
            f"ImageDecoder expects a rank-1 encoded byte stream input, "
            f"got rank {len(input_shape)}"
        )

    pixel_format_attr = node.attrs.get("pixel_format", "RGB")
    if isinstance(pixel_format_attr, bytes):
        pixel_format_attr = pixel_format_attr.decode()
    pixel_format = str(pixel_format_attr)
    channels = _PIXEL_FORMAT_CHANNELS.get(pixel_format)
    if channels is None:
        raise UnsupportedOpError(
            f"ImageDecoder pixel_format must be one of "
            f"{sorted(_PIXEL_FORMAT_CHANNELS)}, got {pixel_format!r}"
        )

    output_shape = value_shape(graph, node.outputs[0], node)
    if len(output_shape) != 3:
        raise UnsupportedOpError(
            f"ImageDecoder expects a rank-3 (H, W, C) output with static shape, "
            f"got rank {len(output_shape)}. "
            "Hint: export the model with a static decoded image shape."
        )
    if output_shape[2] != channels:
        raise UnsupportedOpError(
            f"ImageDecoder output channel count {output_shape[2]} does not match "
            f"pixel_format {pixel_format!r} (expected {channels})"
        )

    return ImageDecoderOp(
        input0=node.inputs[0],
        output=node.outputs[0],
        input_length=input_shape[0],
        output_shape=output_shape,
        pixel_format=pixel_format,
        dtype=op_dtype,
    )

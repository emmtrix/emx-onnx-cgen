from __future__ import annotations

from shared.scalar_types import ScalarType

from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from ..ir.ops import ImageDecoderOp
from .common import node_dtype, value_shape
from .registry import register_lowering

# Maps the ONNX ``pixel_format`` attribute to the source channel index in the
# RGB(A) buffer that stb_image produces for each emitted output channel.
_CHANNEL_MAPS: dict[str, tuple[int, ...]] = {
    "RGB": (0, 1, 2),
    "BGR": (2, 1, 0),
    "Grayscale": (0,),
}


def _decode_attr(value: object, default: str) -> str:
    if value is None:
        return default
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, str):
        return value
    return str(value)


@register_lowering("ImageDecoder")
def lower_image_decoder(graph: Graph, node: Node) -> ImageDecoderOp:
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        raise UnsupportedOpError("ImageDecoder expects exactly 1 input and 1 output")

    op_dtype = node_dtype(graph, node, node.inputs[0], node.outputs[0])
    if op_dtype != ScalarType.U8:
        raise UnsupportedOpError(
            f"ImageDecoder only supports uint8 tensors, got {op_dtype}"
        )

    input_shape = value_shape(graph, node.inputs[0], node)
    if len(input_shape) != 1:
        raise UnsupportedOpError(
            f"ImageDecoder expects a rank-1 encoded byte stream, got rank "
            f"{len(input_shape)}"
        )

    output_shape = value_shape(graph, node.outputs[0], node)
    if len(output_shape) != 3:
        raise UnsupportedOpError(
            f"ImageDecoder expects a rank-3 (H, W, C) output, got rank "
            f"{len(output_shape)}"
        )

    pixel_format = _decode_attr(node.attrs.get("pixel_format"), "RGB")
    channel_map = _CHANNEL_MAPS.get(pixel_format)
    if channel_map is None:
        raise UnsupportedOpError(
            f"ImageDecoder pixel_format '{pixel_format}' is not supported "
            "(expected RGB, BGR or Grayscale)"
        )

    channels = output_shape[2]
    if channels != len(channel_map):
        raise ShapeInferenceError(
            f"ImageDecoder output channel count {channels} does not match "
            f"pixel_format '{pixel_format}' ({len(channel_map)} channels)"
        )

    return ImageDecoderOp(
        input0=node.inputs[0],
        output=node.outputs[0],
        input_shape=input_shape,
        shape=output_shape,
        channel_map=channel_map,
        grayscale=pixel_format == "Grayscale",
        dtype=op_dtype,
    )

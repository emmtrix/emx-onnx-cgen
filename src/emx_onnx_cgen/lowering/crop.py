from __future__ import annotations

from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.context import GraphContext
from ..ir.model import Graph, Node
from ..ir.ops import SliceOp
from ..lowering.common import value_dtype, value_shape
from .registry import register_lowering


@register_lowering("Crop")
def lower_crop(graph: Graph, node: Node) -> SliceOp:
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        raise UnsupportedOpError("Crop must have 1 input and 1 output")
    input_shape = value_shape(graph, node.inputs[0], node)
    if len(input_shape) != 4:
        raise UnsupportedOpError(
            f"Crop only supports 4D (NCHW) inputs, got rank {len(input_shape)}"
        )
    if any(dim < 0 for dim in input_shape):
        raise ShapeInferenceError("Crop does not support dynamic dims")
    input_dtype = value_dtype(graph, node.inputs[0], node)
    output_dtype = value_dtype(graph, node.outputs[0], node)
    if input_dtype != output_dtype:
        raise UnsupportedOpError(
            f"Crop expects matching input/output dtypes, "
            f"got {input_dtype.onnx_name} and {output_dtype.onnx_name}"
        )
    N, C, H, W = input_shape
    border_raw = node.attrs.get("border", [0, 0, 0, 0])
    border = [int(v) for v in border_raw]
    if len(border) != 4:
        raise UnsupportedOpError("Crop border attribute must have exactly 4 values")
    top, bottom, left, right = border[0], border[1], border[2], border[3]
    scale_raw = node.attrs.get("scale", [])
    scale = [int(v) for v in scale_raw]
    if scale:
        if len(scale) != 2:
            raise UnsupportedOpError(
                "Crop scale attribute must have exactly 2 values"
            )
        out_H = scale[0]
        out_W = scale[1]
    else:
        out_H = H - top - bottom
        out_W = W - left - right
    if out_H <= 0 or out_W <= 0:
        raise ShapeInferenceError("Crop produces empty spatial dimensions")
    starts = (0, 0, top, left)
    steps = (1, 1, 1, 1)
    if isinstance(graph, GraphContext):
        graph.set_shape(node.outputs[0], (N, C, out_H, out_W))
    return SliceOp(
        input0=node.inputs[0],
        output=node.outputs[0],
        starts=starts,
        steps=steps,
        axes=None,
        starts_input=None,
        ends_input=None,
        axes_input=None,
        steps_input=None,
    )

from __future__ import annotations

from ..ir.ops import CenterCropPadOp
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from ..lowering.common import value_dtype, value_shape
from ..validation import normalize_axis
from .registry import register_lowering


@register_lowering("CenterCropPad")
def lower_center_crop_pad(graph: Graph, node: Node) -> CenterCropPadOp:
    if len(node.inputs) != 2 or len(node.outputs) != 1:
        raise UnsupportedOpError("CenterCropPad must have 2 inputs and 1 output")
    input_name = node.inputs[0]
    shape_name = node.inputs[1]
    input_shape = value_shape(graph, input_name, node)
    output_shape = value_shape(graph, node.outputs[0], node)
    if any(d < 0 for d in input_shape):
        raise ShapeInferenceError("CenterCropPad: dynamic input dims not supported")
    if any(d < 0 for d in output_shape):
        raise ShapeInferenceError("CenterCropPad: dynamic output dims not supported")
    input_dtype = value_dtype(graph, input_name, node)
    output_dtype = value_dtype(graph, node.outputs[0], node)
    if input_dtype != output_dtype:
        raise UnsupportedOpError(
            "CenterCropPad expects matching input/output dtypes, "
            f"got {input_dtype.onnx_name} and {output_dtype.onnx_name}"
        )
    axes_attr = node.attrs.get("axes")
    if axes_attr is not None:
        axes = tuple(
            normalize_axis(int(a), input_shape, node) for a in axes_attr
        )
    else:
        axes = None  # means all axes
    return CenterCropPadOp(
        input0=input_name,
        shape_input=shape_name,
        output=node.outputs[0],
        axes=axes,
        input_shape=input_shape,
        output_shape=output_shape,
    )

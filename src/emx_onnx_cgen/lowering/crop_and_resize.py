from __future__ import annotations

from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from ..ir.ops import CropAndResizeOp
from .common import node_dtype
from .common import value_dtype
from .common import value_shape
from .registry import register_lowering


@register_lowering("CropAndResize")
def lower_crop_and_resize(graph: Graph, node: Node) -> CropAndResizeOp:
    if len(node.inputs) != 4 or len(node.outputs) != 1:
        raise UnsupportedOpError("CropAndResize expects 4 inputs and 1 output")

    x_name, rois_name, box_ind_name, _crop_size_name = node.inputs
    output_name = node.outputs[0]

    x_shape = value_shape(graph, x_name, node)
    rois_shape = value_shape(graph, rois_name, node)
    box_ind_shape = value_shape(graph, box_ind_name, node)
    output_shape = value_shape(graph, output_name, node)

    if len(x_shape) != 4:
        raise UnsupportedOpError(
            f"CropAndResize expects 4D input (N,C,H,W), got rank {len(x_shape)}"
        )
    if len(rois_shape) != 2 or rois_shape[1] != 4:
        raise ShapeInferenceError(
            f"CropAndResize rois must have shape (num_rois, 4), got {rois_shape}"
        )
    if len(output_shape) != 4:
        raise ShapeInferenceError(
            f"CropAndResize output must be 4D, got rank {len(output_shape)}"
        )

    num_rois = rois_shape[0]
    channels = x_shape[1]
    input_height = x_shape[2]
    input_width = x_shape[3]
    output_height = output_shape[2]
    output_width = output_shape[3]

    extrapolation_value = 0.0
    method = "bilinear"
    for name, val in node.attrs.items():
        if name == "extrapolation_value":
            extrapolation_value = float(val)
        elif name == "mode":
            method = val.decode("utf-8") if isinstance(val, bytes) else str(val)

    if method != "bilinear":
        raise UnsupportedOpError(f"CropAndResize method {method!r} is not supported")

    input_dtype = node_dtype(graph, node, x_name, output_name)
    rois_dtype = value_dtype(graph, rois_name, node)
    if not input_dtype.is_float:
        raise UnsupportedOpError(
            f"CropAndResize input dtype must be float, got {input_dtype.onnx_name}"
        )
    if rois_dtype != input_dtype:
        raise UnsupportedOpError(
            "CropAndResize rois dtype must match input dtype, got "
            f"{rois_dtype.onnx_name} and {input_dtype.onnx_name}"
        )

    return CropAndResizeOp(
        x=x_name,
        rois=rois_name,
        box_ind=box_ind_name,
        output=output_name,
        num_rois=num_rois,
        channels=channels,
        input_height=input_height,
        input_width=input_width,
        output_height=output_height,
        output_width=output_width,
        extrapolation_value=extrapolation_value,
    )

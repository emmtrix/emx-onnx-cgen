from __future__ import annotations

import math

from shared.scalar_types import ScalarType

from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from ..ir.ops import RoiAlignOp
from .common import node_dtype
from .common import value_dtype
from .common import value_shape
from .registry import register_lowering

_SUPPORTED_MODES = {"avg", "max"}
_SUPPORTED_COORDINATE_TRANSFORMATION_MODES = {"half_pixel", "output_half_pixel"}


def _decode_attr(value: object, default: str) -> str:
    if value is None:
        return default
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, str):
        return value
    return str(value)


@register_lowering("RoiAlign")
def lower_roi_align(graph: Graph, node: Node) -> RoiAlignOp:
    if len(node.inputs) != 3 or len(node.outputs) != 1:
        raise UnsupportedOpError("RoiAlign expects 3 inputs and 1 output")
    supported_attrs = {
        "coordinate_transformation_mode",
        "mode",
        "output_height",
        "output_width",
        "sampling_ratio",
        "spatial_scale",
    }
    if set(node.attrs) - supported_attrs:
        raise UnsupportedOpError("RoiAlign has unsupported attributes")

    input0, rois, batch_indices = node.inputs
    output = node.outputs[0]

    input_shape = value_shape(graph, input0, node)
    rois_shape = value_shape(graph, rois, node)
    batch_indices_shape = value_shape(graph, batch_indices, node)
    output_shape = value_shape(graph, output, node)

    if len(input_shape) != 4:
        raise UnsupportedOpError(
            f"RoiAlign expects 4D input tensor (N,C,H,W), got rank {len(input_shape)}"
        )
    if len(rois_shape) != 2 or rois_shape[1] != 4:
        raise ShapeInferenceError(
            f"RoiAlign rois input must have shape (num_rois, 4), got {rois_shape}"
        )
    if len(batch_indices_shape) != 1:
        raise ShapeInferenceError(
            f"RoiAlign batch_indices input must have shape (num_rois,), got {batch_indices_shape}"
        )

    num_rois = rois_shape[0]
    if batch_indices_shape[0] != num_rois:
        raise ShapeInferenceError(
            "RoiAlign batch_indices length must match rois count, got "
            f"{batch_indices_shape[0]} and {num_rois}"
        )

    output_height = int(node.attrs.get("output_height", 1))
    output_width = int(node.attrs.get("output_width", 1))
    sampling_ratio = int(node.attrs.get("sampling_ratio", 0))
    spatial_scale = float(node.attrs.get("spatial_scale", 1.0))
    mode = _decode_attr(node.attrs.get("mode"), "avg")
    coordinate_transformation_mode = _decode_attr(
        node.attrs.get("coordinate_transformation_mode"), "half_pixel"
    )

    if output_height <= 0 or output_width <= 0:
        raise UnsupportedOpError("RoiAlign output_height/output_width must be positive")
    if sampling_ratio < 0:
        raise UnsupportedOpError("RoiAlign sampling_ratio must be non-negative")
    if mode not in _SUPPORTED_MODES:
        raise UnsupportedOpError(f"RoiAlign mode {mode!r} is not supported")
    if coordinate_transformation_mode not in _SUPPORTED_COORDINATE_TRANSFORMATION_MODES:
        raise UnsupportedOpError(
            "RoiAlign coordinate_transformation_mode "
            f"{coordinate_transformation_mode!r} is not supported"
        )

    expected_output_shape = (
        num_rois,
        input_shape[1],
        output_height,
        output_width,
    )
    if output_shape != expected_output_shape:
        raise ShapeInferenceError(
            f"RoiAlign output shape must be {expected_output_shape}, got {output_shape}"
        )

    input_dtype = node_dtype(graph, node, input0, output)
    rois_dtype = value_dtype(graph, rois, node)
    batch_indices_dtype = value_dtype(graph, batch_indices, node)
    if not input_dtype.is_float:
        raise UnsupportedOpError(
            f"RoiAlign input/output dtype must be floating-point, got {input_dtype.onnx_name}"
        )
    if rois_dtype != input_dtype:
        raise UnsupportedOpError(
            "RoiAlign rois dtype must match input dtype, got "
            f"{rois_dtype.onnx_name} and {input_dtype.onnx_name}"
        )
    if batch_indices_dtype != ScalarType.I64:
        raise UnsupportedOpError(
            "RoiAlign batch_indices dtype must be int64, got "
            f"{batch_indices_dtype.onnx_name}"
        )

    roi_max_height = (input_shape[2] * spatial_scale) + 1.0
    roi_max_width = (input_shape[3] * spatial_scale) + 1.0
    max_roi_bin = max(roi_max_height / output_height, roi_max_width / output_width)
    if sampling_ratio == 0:
        max_sampling_points = int(math.ceil(max_roi_bin))
    else:
        max_sampling_points = sampling_ratio

    return RoiAlignOp(
        input0=input0,
        rois=rois,
        batch_indices=batch_indices,
        output=output,
        num_rois=num_rois,
        channels=input_shape[1],
        input_height=input_shape[2],
        input_width=input_shape[3],
        output_height=output_height,
        output_width=output_width,
        sampling_ratio=sampling_ratio,
        max_sampling_points=max(1, max_sampling_points),
        spatial_scale=spatial_scale,
        mode=mode,
        coordinate_transformation_mode=coordinate_transformation_mode,
        dtype=input_dtype,
    )

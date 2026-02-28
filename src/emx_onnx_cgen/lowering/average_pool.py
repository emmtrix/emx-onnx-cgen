from __future__ import annotations

import math
from dataclasses import dataclass

from shared.scalar_types import ScalarType

from ..ir.ops import AveragePoolOp, QLinearAveragePoolOp
from ..ir.context import GraphContext
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from .registry import register_lowering


@dataclass(frozen=True)
class _AveragePoolSpec:
    batch: int
    channels: int
    spatial_rank: int
    in_d: int
    in_h: int
    in_w: int
    out_d: int
    out_h: int
    out_w: int
    kernel_d: int
    kernel_h: int
    kernel_w: int
    dilation_d: int
    dilation_h: int
    dilation_w: int
    stride_d: int
    stride_h: int
    stride_w: int
    pad_front: int
    pad_top: int
    pad_left: int
    pad_back: int
    pad_bottom: int
    pad_right: int
    count_include_pad: bool


def _value_shape(graph: Graph, name: str, node: Node) -> tuple[int, ...]:
    if isinstance(graph, GraphContext):
        if graph.has_shape(name):
            return graph.shape(name, node)
        return graph.shape(name, node)
    try:
        return graph.find_value(name).type.shape
    except KeyError as exc:
        raise ShapeInferenceError(
            f"Missing shape for value '{name}' in op {node.op_type}. "
            "Hint: run ONNX shape inference or export with static shapes."
        ) from exc


def _value_dtype(graph: Graph, name: str, node: Node) -> str:
    if isinstance(graph, GraphContext):
        return graph.dtype(name, node)
    try:
        return graph.find_value(name).type.dtype
    except KeyError as exc:
        raise ShapeInferenceError(
            f"Missing dtype for value '{name}' in op {node.op_type}. "
            "Hint: run ONNX shape inference or export with static shapes."
        ) from exc


def _resolve_average_pool_spec(
    graph: Graph,
    node: Node,
    *,
    input_name: str,
    output_name: str,
    require_output_shape: bool = True,
) -> _AveragePoolSpec:
    supported_attrs = {
        "auto_pad",
        "ceil_mode",
        "count_include_pad",
        "dilations",
        "kernel_shape",
        "pads",
        "strides",
    }
    if set(node.attrs) - supported_attrs:
        raise UnsupportedOpError("AveragePool has unsupported attributes")
    auto_pad = node.attrs.get("auto_pad", b"NOTSET")
    if isinstance(auto_pad, bytes):
        auto_pad = auto_pad.decode("utf-8", errors="ignore")
    ceil_mode = int(node.attrs.get("ceil_mode", 0))
    if ceil_mode not in (0, 1):
        raise UnsupportedOpError("AveragePool supports ceil_mode=0 or 1 only")
    count_include_pad = int(node.attrs.get("count_include_pad", 0))
    if count_include_pad not in (0, 1):
        raise UnsupportedOpError("AveragePool supports count_include_pad 0 or 1")
    kernel_shape = node.attrs.get("kernel_shape")
    if kernel_shape is None:
        raise UnsupportedOpError("AveragePool requires kernel_shape")
    kernel_shape = tuple(int(value) for value in kernel_shape)
    input_shape = _value_shape(graph, input_name, node)
    if len(input_shape) < 3:
        raise UnsupportedOpError("AveragePool expects NCHW inputs with spatial dims")
    spatial_rank = len(input_shape) - 2
    if spatial_rank not in {1, 2, 3}:
        raise UnsupportedOpError("AveragePool supports 1D/2D/3D inputs only")
    if len(kernel_shape) != spatial_rank:
        raise ShapeInferenceError(
            "AveragePool kernel_shape must have "
            f"{spatial_rank} dims, got {kernel_shape}"
        )
    strides = tuple(
        int(value) for value in node.attrs.get("strides", (1,) * spatial_rank)
    )
    if len(strides) != spatial_rank:
        raise UnsupportedOpError("AveragePool stride rank mismatch")
    dilations = tuple(
        int(value) for value in node.attrs.get("dilations", (1,) * spatial_rank)
    )
    if len(dilations) != spatial_rank:
        raise UnsupportedOpError("AveragePool dilation rank mismatch")
    pads = tuple(
        int(value) for value in node.attrs.get("pads", (0,) * (2 * spatial_rank))
    )
    if len(pads) != 2 * spatial_rank:
        raise UnsupportedOpError("AveragePool pads rank mismatch")
    if auto_pad in ("", "NOTSET"):
        pad_begin = pads[:spatial_rank]
        pad_end = pads[spatial_rank:]
    elif auto_pad == "VALID":
        pad_begin = (0,) * spatial_rank
        pad_end = (0,) * spatial_rank
    elif auto_pad in {"SAME_UPPER", "SAME_LOWER"}:
        pad_begin = []
        pad_end = []
        for dim, stride, dilation, kernel in zip(
            input_shape[2:], strides, dilations, kernel_shape
        ):
            effective_kernel = dilation * (kernel - 1) + 1
            out_dim = math.ceil(dim / stride)
            pad_needed = max(0, (out_dim - 1) * stride + effective_kernel - dim)
            if auto_pad == "SAME_UPPER":
                pad_start = pad_needed // 2
            else:
                pad_start = (pad_needed + 1) // 2
            pad_begin.append(pad_start)
            pad_end.append(pad_needed - pad_start)
        pad_begin = tuple(pad_begin)
        pad_end = tuple(pad_end)
    else:
        raise UnsupportedOpError("AveragePool has unsupported auto_pad mode")
    batch, channels = input_shape[:2]
    in_spatial = input_shape[2:]
    out_spatial = []
    for dim, stride, dilation, kernel, pad_start, pad_finish in zip(
        in_spatial, strides, dilations, kernel_shape, pad_begin, pad_end
    ):
        effective_kernel = dilation * (kernel - 1) + 1
        numerator = dim + pad_start + pad_finish - effective_kernel
        if ceil_mode:
            out_dim = (numerator + stride - 1) // stride + 1
            if (out_dim - 1) * stride >= dim + pad_start:
                out_dim -= 1
        else:
            out_dim = numerator // stride + 1
        if out_dim < 0:
            raise ShapeInferenceError("AveragePool output shape must be non-negative")
        out_spatial.append(out_dim)
    expected_output_shape = (batch, channels, *out_spatial)
    try:
        output_shape = _value_shape(graph, output_name, node)
    except ShapeInferenceError:
        output_shape = None
    if output_shape is not None and output_shape != expected_output_shape:
        raise ShapeInferenceError(
            "AveragePool output shape must be "
            f"{expected_output_shape}, got {output_shape}"
        )
    if output_shape is None and require_output_shape:
        raise ShapeInferenceError(
            f"Missing shape for value '{output_name}' in op {node.op_type}. "
            "Hint: run ONNX shape inference or export with static shapes."
        )
    in_d = in_spatial[0] if spatial_rank == 3 else 1
    in_h = in_spatial[-2] if spatial_rank >= 2 else 1
    in_w = in_spatial[-1]
    out_d = out_spatial[0] if spatial_rank == 3 else 1
    out_h = out_spatial[-2] if spatial_rank >= 2 else 1
    out_w = out_spatial[-1]
    kernel_d = kernel_shape[0] if spatial_rank == 3 else 1
    kernel_h = kernel_shape[-2] if spatial_rank >= 2 else 1
    kernel_w = kernel_shape[-1]
    dilation_d = dilations[0] if spatial_rank == 3 else 1
    dilation_h = dilations[-2] if spatial_rank >= 2 else 1
    dilation_w = dilations[-1]
    stride_d = strides[0] if spatial_rank == 3 else 1
    stride_h = strides[-2] if spatial_rank >= 2 else 1
    stride_w = strides[-1]
    pad_front = pad_begin[0] if spatial_rank == 3 else 0
    pad_top = pad_begin[-2] if spatial_rank >= 2 else 0
    pad_left = pad_begin[-1]
    pad_back = pad_end[0] if spatial_rank == 3 else 0
    pad_bottom = pad_end[-2] if spatial_rank >= 2 else 0
    pad_right = pad_end[-1]
    return _AveragePoolSpec(
        batch=batch,
        channels=channels,
        spatial_rank=spatial_rank,
        in_d=in_d,
        in_h=in_h,
        in_w=in_w,
        out_d=out_d,
        out_h=out_h,
        out_w=out_w,
        kernel_d=kernel_d,
        kernel_h=kernel_h,
        kernel_w=kernel_w,
        dilation_d=dilation_d,
        dilation_h=dilation_h,
        dilation_w=dilation_w,
        stride_d=stride_d,
        stride_h=stride_h,
        stride_w=stride_w,
        pad_front=pad_front,
        pad_top=pad_top,
        pad_left=pad_left,
        pad_back=pad_back,
        pad_bottom=pad_bottom,
        pad_right=pad_right,
        count_include_pad=bool(count_include_pad),
    )


def _resolve_global_average_pool_spec(graph: Graph, node: Node) -> _AveragePoolSpec:
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        raise UnsupportedOpError("GlobalAveragePool must have 1 input and 1 output")
    if node.attrs:
        raise UnsupportedOpError("GlobalAveragePool has unsupported attributes")
    input_shape = _value_shape(graph, node.inputs[0], node)
    if len(input_shape) < 3:
        raise UnsupportedOpError(
            "GlobalAveragePool expects NCHW inputs with spatial dims"
        )
    spatial_rank = len(input_shape) - 2
    if spatial_rank not in {1, 2, 3}:
        raise UnsupportedOpError("GlobalAveragePool supports 1D/2D/3D inputs only")
    batch, channels = input_shape[:2]
    in_spatial = input_shape[2:]
    output_shape = _value_shape(graph, node.outputs[0], node)
    expected_output_shape = (batch, channels, *([1] * spatial_rank))
    if output_shape != expected_output_shape:
        raise ShapeInferenceError(
            "GlobalAveragePool output shape must be "
            f"{expected_output_shape}, got {output_shape}"
        )
    in_d = in_spatial[0] if spatial_rank == 3 else 1
    in_h = in_spatial[-2] if spatial_rank >= 2 else 1
    in_w = in_spatial[-1]
    return _AveragePoolSpec(
        batch=batch,
        channels=channels,
        spatial_rank=spatial_rank,
        in_d=in_d,
        in_h=in_h,
        in_w=in_w,
        out_d=1,
        out_h=1,
        out_w=1,
        kernel_d=in_d,
        kernel_h=in_h,
        kernel_w=in_w,
        dilation_d=1,
        dilation_h=1,
        dilation_w=1,
        stride_d=1,
        stride_h=1,
        stride_w=1,
        pad_front=0,
        pad_top=0,
        pad_left=0,
        pad_back=0,
        pad_bottom=0,
        pad_right=0,
        count_include_pad=False,
    )


@register_lowering("AveragePool")
def lower_average_pool(graph: Graph, node: Node) -> AveragePoolOp:
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        raise UnsupportedOpError("AveragePool must have 1 input and 1 output")
    op_dtype = _value_dtype(graph, node.inputs[0], node)
    output_dtype = _value_dtype(graph, node.outputs[0], node)
    if op_dtype != output_dtype:
        raise UnsupportedOpError(
            "AveragePool expects matching input/output dtypes, "
            f"got {op_dtype.onnx_name} and {output_dtype.onnx_name}"
        )
    if not op_dtype.is_float:
        raise UnsupportedOpError(
            "AveragePool supports float16, float, and double inputs only"
        )
    spec = _resolve_average_pool_spec(
        graph,
        node,
        input_name=node.inputs[0],
        output_name=node.outputs[0],
    )
    return AveragePoolOp(
        input0=node.inputs[0],
        output=node.outputs[0],
        batch=spec.batch,
        channels=spec.channels,
        spatial_rank=spec.spatial_rank,
        in_d=spec.in_d,
        in_h=spec.in_h,
        in_w=spec.in_w,
        out_d=spec.out_d,
        out_h=spec.out_h,
        out_w=spec.out_w,
        kernel_d=spec.kernel_d,
        kernel_h=spec.kernel_h,
        kernel_w=spec.kernel_w,
        dilation_d=spec.dilation_d,
        dilation_h=spec.dilation_h,
        dilation_w=spec.dilation_w,
        stride_d=spec.stride_d,
        stride_h=spec.stride_h,
        stride_w=spec.stride_w,
        pad_front=spec.pad_front,
        pad_top=spec.pad_top,
        pad_left=spec.pad_left,
        pad_back=spec.pad_back,
        pad_bottom=spec.pad_bottom,
        pad_right=spec.pad_right,
        count_include_pad=spec.count_include_pad,
        dtype=op_dtype,
    )


def _ensure_scalar_shape(shape: tuple[int, ...], label: str) -> None:
    if shape not in {(), (1,)}:
        raise UnsupportedOpError(
            f"QLinearAveragePool {label} must be scalar, got shape {shape}"
        )


def _ensure_scale_dtype(dtype: ScalarType, label: str) -> None:
    if not dtype.is_float:
        raise UnsupportedOpError(
            f"QLinearAveragePool {label} must be float16/float/double"
        )


@register_lowering("QLinearAveragePool")
def lower_qlinear_average_pool(graph: Graph, node: Node) -> QLinearAveragePoolOp:
    if len(node.inputs) != 5 or len(node.outputs) != 1:
        raise UnsupportedOpError("QLinearAveragePool must have 5 inputs and 1 output")
    input_name = node.inputs[0]
    input_scale_name = node.inputs[1]
    input_zero_name = node.inputs[2]
    output_scale_name = node.inputs[3]
    output_zero_name = node.inputs[4]
    output_name = node.outputs[0]

    input_dtype = _value_dtype(graph, input_name, node)
    try:
        output_dtype = _value_dtype(graph, output_name, node)
    except ShapeInferenceError:
        output_dtype = input_dtype
    if input_dtype not in {ScalarType.U8, ScalarType.I8}:
        raise UnsupportedOpError("QLinearAveragePool supports uint8/int8 inputs only")
    if output_dtype not in {ScalarType.U8, ScalarType.I8}:
        raise UnsupportedOpError("QLinearAveragePool supports uint8/int8 outputs only")

    input_scale_dtype = _value_dtype(graph, input_scale_name, node)
    output_scale_dtype = _value_dtype(graph, output_scale_name, node)
    _ensure_scale_dtype(input_scale_dtype, "x_scale")
    _ensure_scale_dtype(output_scale_dtype, "y_scale")

    input_zero_dtype = _value_dtype(graph, input_zero_name, node)
    output_zero_dtype = _value_dtype(graph, output_zero_name, node)
    if input_zero_dtype != input_dtype:
        raise UnsupportedOpError(
            "QLinearAveragePool x_zero_point dtype must match input dtype"
        )
    if output_zero_dtype != output_dtype:
        raise UnsupportedOpError(
            "QLinearAveragePool y_zero_point dtype must match output dtype"
        )

    input_scale_shape = _value_shape(graph, input_scale_name, node)
    output_scale_shape = _value_shape(graph, output_scale_name, node)
    input_zero_shape = _value_shape(graph, input_zero_name, node)
    output_zero_shape = _value_shape(graph, output_zero_name, node)
    _ensure_scalar_shape(input_scale_shape, "x_scale")
    _ensure_scalar_shape(output_scale_shape, "y_scale")
    _ensure_scalar_shape(input_zero_shape, "x_zero_point")
    _ensure_scalar_shape(output_zero_shape, "y_zero_point")

    spec = _resolve_average_pool_spec(
        graph,
        node,
        input_name=input_name,
        output_name=output_name,
        require_output_shape=False,
    )
    if isinstance(graph, GraphContext):
        if spec.spatial_rank == 3:
            graph.set_shape(
                output_name,
                (spec.batch, spec.channels, spec.out_d, spec.out_h, spec.out_w),
            )
        elif spec.spatial_rank == 1:
            graph.set_shape(output_name, (spec.batch, spec.channels, spec.out_w))
        else:
            graph.set_shape(
                output_name, (spec.batch, spec.channels, spec.out_h, spec.out_w)
            )
        graph.set_dtype(output_name, output_dtype)
    return QLinearAveragePoolOp(
        input0=input_name,
        input_scale=input_scale_name,
        input_zero_point=input_zero_name,
        output_scale=output_scale_name,
        output_zero_point=output_zero_name,
        output=output_name,
        batch=spec.batch,
        channels=spec.channels,
        spatial_rank=spec.spatial_rank,
        in_d=spec.in_d,
        in_h=spec.in_h,
        in_w=spec.in_w,
        out_d=spec.out_d,
        out_h=spec.out_h,
        out_w=spec.out_w,
        kernel_d=spec.kernel_d,
        kernel_h=spec.kernel_h,
        kernel_w=spec.kernel_w,
        dilation_d=spec.dilation_d,
        dilation_h=spec.dilation_h,
        dilation_w=spec.dilation_w,
        stride_d=spec.stride_d,
        stride_h=spec.stride_h,
        stride_w=spec.stride_w,
        pad_front=spec.pad_front,
        pad_top=spec.pad_top,
        pad_left=spec.pad_left,
        pad_back=spec.pad_back,
        pad_bottom=spec.pad_bottom,
        pad_right=spec.pad_right,
        count_include_pad=spec.count_include_pad,
        input_dtype=input_dtype,
        dtype=output_dtype,
        input_scale_dtype=input_scale_dtype,
        output_scale_dtype=output_scale_dtype,
        input_scale_shape=input_scale_shape,
        output_scale_shape=output_scale_shape,
        input_zero_shape=input_zero_shape,
        output_zero_shape=output_zero_shape,
    )


@register_lowering("GlobalAveragePool")
def lower_global_average_pool(graph: Graph, node: Node) -> AveragePoolOp:
    op_dtype = _value_dtype(graph, node.inputs[0], node)
    output_dtype = _value_dtype(graph, node.outputs[0], node)
    if op_dtype != output_dtype:
        raise UnsupportedOpError(
            "GlobalAveragePool expects matching input/output dtypes, "
            f"got {op_dtype.onnx_name} and {output_dtype.onnx_name}"
        )
    if not op_dtype.is_float:
        raise UnsupportedOpError(
            "GlobalAveragePool supports float16, float, and double inputs only"
        )
    spec = _resolve_global_average_pool_spec(graph, node)
    return AveragePoolOp(
        input0=node.inputs[0],
        output=node.outputs[0],
        batch=spec.batch,
        channels=spec.channels,
        spatial_rank=spec.spatial_rank,
        in_d=spec.in_d,
        in_h=spec.in_h,
        in_w=spec.in_w,
        out_d=spec.out_d,
        out_h=spec.out_h,
        out_w=spec.out_w,
        kernel_d=spec.kernel_d,
        kernel_h=spec.kernel_h,
        kernel_w=spec.kernel_w,
        dilation_d=spec.dilation_d,
        dilation_h=spec.dilation_h,
        dilation_w=spec.dilation_w,
        stride_d=spec.stride_d,
        stride_h=spec.stride_h,
        stride_w=spec.stride_w,
        pad_front=spec.pad_front,
        pad_top=spec.pad_top,
        pad_left=spec.pad_left,
        pad_back=spec.pad_back,
        pad_bottom=spec.pad_bottom,
        pad_right=spec.pad_right,
        count_include_pad=spec.count_include_pad,
        dtype=op_dtype,
    )

from __future__ import annotations

from dataclasses import dataclass

from ..ir.ops import LpPoolOp
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from .registry import register_lowering
from .common import value_dtype as _value_dtype, value_shape as _value_shape


@dataclass(frozen=True)
class LpPoolSpec:
    batch: int
    channels: int
    in_h: int
    in_w: int
    out_h: int
    out_w: int
    kernel_h: int
    kernel_w: int
    dilation_h: int
    dilation_w: int
    stride_h: int
    stride_w: int
    pad_top: int
    pad_left: int
    pad_bottom: int
    pad_right: int
    p: int


def _resolve_lp_pool_spec(graph: Graph, node: Node) -> LpPoolSpec:
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        raise UnsupportedOpError("LpPool must have 1 input and 1 output")
    supported_attrs = {
        "auto_pad",
        "ceil_mode",
        "dilations",
        "kernel_shape",
        "pads",
        "p",
        "strides",
    }
    if set(node.attrs) - supported_attrs:
        raise UnsupportedOpError("LpPool has unsupported attributes")
    auto_pad = node.attrs.get("auto_pad", b"NOTSET")
    if isinstance(auto_pad, bytes):
        auto_pad = auto_pad.decode("utf-8", errors="ignore")
    if auto_pad not in ("", "NOTSET"):
        raise UnsupportedOpError("LpPool supports auto_pad=NOTSET only")
    ceil_mode = int(node.attrs.get("ceil_mode", 0))
    if ceil_mode != 0:
        raise UnsupportedOpError("LpPool supports ceil_mode=0 only")
    dilations = tuple(int(value) for value in node.attrs.get("dilations", (1, 1)))
    if len(dilations) != 2:
        raise UnsupportedOpError("LpPool expects 2D dilations")
    if any(value < 1 for value in dilations):
        raise UnsupportedOpError("LpPool requires dilations >= 1")
    kernel_shape = node.attrs.get("kernel_shape")
    if kernel_shape is None:
        raise UnsupportedOpError("LpPool requires kernel_shape")
    kernel_shape = tuple(int(value) for value in kernel_shape)
    if len(kernel_shape) != 2:
        raise UnsupportedOpError("LpPool expects 2D kernel_shape")
    kernel_h, kernel_w = kernel_shape
    strides = tuple(int(value) for value in node.attrs.get("strides", (1, 1)))
    if len(strides) != 2:
        raise UnsupportedOpError("LpPool expects 2D strides")
    pads = tuple(int(value) for value in node.attrs.get("pads", (0, 0, 0, 0)))
    if len(pads) != 4:
        raise UnsupportedOpError("LpPool expects 4D pads")
    pad_top, pad_left, pad_bottom, pad_right = pads
    p = int(node.attrs.get("p", 2))
    if p < 1:
        raise UnsupportedOpError("LpPool p must be >= 1")
    input_shape = _value_shape(graph, node.inputs[0], node)
    if len(input_shape) != 4:
        raise UnsupportedOpError("LpPool supports NCHW 2D inputs only")
    batch, channels, in_h, in_w = input_shape
    stride_h, stride_w = strides
    dilation_h, dilation_w = dilations
    effective_kernel_h = dilation_h * (kernel_h - 1) + 1
    effective_kernel_w = dilation_w * (kernel_w - 1) + 1
    out_h = (in_h + pad_top + pad_bottom - effective_kernel_h) // stride_h + 1
    out_w = (in_w + pad_left + pad_right - effective_kernel_w) // stride_w + 1
    if out_h < 0 or out_w < 0:
        raise ShapeInferenceError("LpPool output shape must be non-negative")
    output_shape = _value_shape(graph, node.outputs[0], node)
    expected_output_shape = (batch, channels, out_h, out_w)
    if output_shape != expected_output_shape:
        raise ShapeInferenceError(
            "LpPool output shape must be "
            f"{expected_output_shape}, got {output_shape}"
        )
    return LpPoolSpec(
        batch=batch,
        channels=channels,
        in_h=in_h,
        in_w=in_w,
        out_h=out_h,
        out_w=out_w,
        kernel_h=kernel_h,
        kernel_w=kernel_w,
        dilation_h=dilation_h,
        dilation_w=dilation_w,
        stride_h=stride_h,
        stride_w=stride_w,
        pad_top=pad_top,
        pad_left=pad_left,
        pad_bottom=pad_bottom,
        pad_right=pad_right,
        p=p,
    )


@register_lowering("LpPool")
def lower_lp_pool(graph: Graph, node: Node) -> LpPoolOp:
    op_dtype = _value_dtype(graph, node.inputs[0], node)
    output_dtype = _value_dtype(graph, node.outputs[0], node)
    if op_dtype != output_dtype:
        raise UnsupportedOpError(
            "LpPool expects matching input/output dtypes, "
            f"got {op_dtype.onnx_name} and {output_dtype.onnx_name}"
        )
    if not op_dtype.is_float:
        raise UnsupportedOpError(
            "LpPool supports float16, float, and double inputs only"
        )
    spec = _resolve_lp_pool_spec(graph, node)
    return LpPoolOp(
        input0=node.inputs[0],
        output=node.outputs[0],
        batch=spec.batch,
        channels=spec.channels,
        in_h=spec.in_h,
        in_w=spec.in_w,
        out_h=spec.out_h,
        out_w=spec.out_w,
        kernel_h=spec.kernel_h,
        kernel_w=spec.kernel_w,
        dilation_h=spec.dilation_h,
        dilation_w=spec.dilation_w,
        stride_h=spec.stride_h,
        stride_w=spec.stride_w,
        pad_top=spec.pad_top,
        pad_left=spec.pad_left,
        pad_bottom=spec.pad_bottom,
        pad_right=spec.pad_right,
        p=spec.p,
        dtype=op_dtype,
    )

from __future__ import annotations

from dataclasses import dataclass

from ..codegen.c_emitter import AveragePoolOp
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from .registry import register_lowering


@dataclass(frozen=True)
class _AveragePoolSpec:
    batch: int
    channels: int
    in_h: int
    in_w: int
    out_h: int
    out_w: int
    kernel_h: int
    kernel_w: int
    stride_h: int
    stride_w: int
    pad_top: int
    pad_left: int
    pad_bottom: int
    pad_right: int
    count_include_pad: bool


def _value_shape(graph: Graph, name: str, node: Node) -> tuple[int, ...]:
    try:
        return graph.find_value(name).type.shape
    except KeyError as exc:
        raise ShapeInferenceError(
            f"Missing shape for value '{name}' in op {node.op_type}. "
            "Hint: run ONNX shape inference or export with static shapes."
        ) from exc


def _value_dtype(graph: Graph, name: str, node: Node) -> str:
    try:
        return graph.find_value(name).type.dtype
    except KeyError as exc:
        raise ShapeInferenceError(
            f"Missing dtype for value '{name}' in op {node.op_type}. "
            "Hint: run ONNX shape inference or export with static shapes."
        ) from exc


def _resolve_average_pool_spec(graph: Graph, node: Node) -> _AveragePoolSpec:
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        raise UnsupportedOpError("AveragePool must have 1 input and 1 output")
    supported_attrs = {
        "auto_pad",
        "ceil_mode",
        "count_include_pad",
        "kernel_shape",
        "pads",
        "strides",
    }
    if set(node.attrs) - supported_attrs:
        raise UnsupportedOpError("AveragePool has unsupported attributes")
    auto_pad = node.attrs.get("auto_pad", b"NOTSET")
    if isinstance(auto_pad, bytes):
        auto_pad = auto_pad.decode("utf-8", errors="ignore")
    if auto_pad not in ("", "NOTSET"):
        raise UnsupportedOpError("AveragePool supports auto_pad=NOTSET only")
    ceil_mode = int(node.attrs.get("ceil_mode", 0))
    if ceil_mode != 0:
        raise UnsupportedOpError("AveragePool supports ceil_mode=0 only")
    count_include_pad = int(node.attrs.get("count_include_pad", 0))
    if count_include_pad not in (0, 1):
        raise UnsupportedOpError("AveragePool supports count_include_pad 0 or 1")
    kernel_shape = node.attrs.get("kernel_shape")
    if kernel_shape is None:
        raise UnsupportedOpError("AveragePool requires kernel_shape")
    kernel_shape = tuple(int(value) for value in kernel_shape)
    if len(kernel_shape) != 2:
        raise UnsupportedOpError("AveragePool expects 2D kernel_shape")
    kernel_h, kernel_w = kernel_shape
    strides = tuple(int(value) for value in node.attrs.get("strides", (1, 1)))
    if len(strides) != 2:
        raise UnsupportedOpError("AveragePool expects 2D strides")
    pads = tuple(int(value) for value in node.attrs.get("pads", (0, 0, 0, 0)))
    if len(pads) != 4:
        raise UnsupportedOpError("AveragePool expects 4D pads")
    pad_top, pad_left, pad_bottom, pad_right = pads
    input_shape = _value_shape(graph, node.inputs[0], node)
    if len(input_shape) != 4:
        raise UnsupportedOpError("AveragePool supports NCHW 2D inputs only")
    batch, channels, in_h, in_w = input_shape
    stride_h, stride_w = strides
    out_h = (in_h + pad_top + pad_bottom - kernel_h) // stride_h + 1
    out_w = (in_w + pad_left + pad_right - kernel_w) // stride_w + 1
    if out_h < 0 or out_w < 0:
        raise ShapeInferenceError(
            "AveragePool output shape must be non-negative"
        )
    output_shape = _value_shape(graph, node.outputs[0], node)
    expected_output_shape = (batch, channels, out_h, out_w)
    if output_shape != expected_output_shape:
        raise ShapeInferenceError(
            "AveragePool output shape must be "
            f"{expected_output_shape}, got {output_shape}"
        )
    return _AveragePoolSpec(
        batch=batch,
        channels=channels,
        in_h=in_h,
        in_w=in_w,
        out_h=out_h,
        out_w=out_w,
        kernel_h=kernel_h,
        kernel_w=kernel_w,
        stride_h=stride_h,
        stride_w=stride_w,
        pad_top=pad_top,
        pad_left=pad_left,
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
    if len(input_shape) != 4:
        raise UnsupportedOpError("GlobalAveragePool supports NCHW 2D inputs only")
    batch, channels, in_h, in_w = input_shape
    output_shape = _value_shape(graph, node.outputs[0], node)
    expected_output_shape = (batch, channels, 1, 1)
    if output_shape != expected_output_shape:
        raise ShapeInferenceError(
            "GlobalAveragePool output shape must be "
            f"{expected_output_shape}, got {output_shape}"
        )
    return _AveragePoolSpec(
        batch=batch,
        channels=channels,
        in_h=in_h,
        in_w=in_w,
        out_h=1,
        out_w=1,
        kernel_h=in_h,
        kernel_w=in_w,
        stride_h=1,
        stride_w=1,
        pad_top=0,
        pad_left=0,
        pad_bottom=0,
        pad_right=0,
        count_include_pad=False,
    )


@register_lowering("AveragePool")
def lower_average_pool(graph: Graph, node: Node) -> AveragePoolOp:
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
    spec = _resolve_average_pool_spec(graph, node)
    return AveragePoolOp(
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
        stride_h=spec.stride_h,
        stride_w=spec.stride_w,
        pad_top=spec.pad_top,
        pad_left=spec.pad_left,
        pad_bottom=spec.pad_bottom,
        pad_right=spec.pad_right,
        count_include_pad=spec.count_include_pad,
        dtype=op_dtype,
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
        in_h=spec.in_h,
        in_w=spec.in_w,
        out_h=spec.out_h,
        out_w=spec.out_w,
        kernel_h=spec.kernel_h,
        kernel_w=spec.kernel_w,
        stride_h=spec.stride_h,
        stride_w=spec.stride_w,
        pad_top=spec.pad_top,
        pad_left=spec.pad_left,
        pad_bottom=spec.pad_bottom,
        pad_right=spec.pad_right,
        count_include_pad=spec.count_include_pad,
        dtype=op_dtype,
    )

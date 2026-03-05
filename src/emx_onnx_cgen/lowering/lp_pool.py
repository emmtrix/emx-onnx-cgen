from __future__ import annotations

import math
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
    spatial_rank: int
    in_spatial: tuple[int, ...]
    out_spatial: tuple[int, ...]
    kernel_shape: tuple[int, ...]
    dilations: tuple[int, ...]
    strides: tuple[int, ...]
    pads: tuple[int, ...]
    p: float


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
    if auto_pad not in ("", "NOTSET", "VALID", "SAME_UPPER", "SAME_LOWER"):
        raise UnsupportedOpError("LpPool has unsupported auto_pad mode")
    ceil_mode = int(node.attrs.get("ceil_mode", 0))
    if ceil_mode not in (0, 1):
        raise UnsupportedOpError("LpPool supports ceil_mode=0 or 1 only")
    kernel_shape = node.attrs.get("kernel_shape")
    if kernel_shape is None:
        raise UnsupportedOpError("LpPool requires kernel_shape")
    kernel_shape = tuple(int(value) for value in kernel_shape)
    input_shape = _value_shape(graph, node.inputs[0], node)
    if len(input_shape) < 3:
        raise UnsupportedOpError("LpPool expects NCHW inputs with spatial dims")
    spatial_rank = len(input_shape) - 2
    if spatial_rank not in {1, 2, 3}:
        raise UnsupportedOpError("LpPool supports 1D/2D/3D inputs only")
    if len(kernel_shape) != spatial_rank:
        raise UnsupportedOpError(
            f"LpPool kernel_shape must have {spatial_rank} dims, got {kernel_shape}"
        )
    strides = tuple(
        int(value) for value in node.attrs.get("strides", (1,) * spatial_rank)
    )
    if len(strides) != spatial_rank:
        raise UnsupportedOpError("LpPool stride rank mismatch")
    dilations = tuple(
        int(value) for value in node.attrs.get("dilations", (1,) * spatial_rank)
    )
    if len(dilations) != spatial_rank:
        raise UnsupportedOpError("LpPool dilation rank mismatch")
    if any(value < 1 for value in dilations):
        raise UnsupportedOpError("LpPool requires dilations >= 1")
    pads = tuple(
        int(value) for value in node.attrs.get("pads", (0,) * (2 * spatial_rank))
    )
    if len(pads) != 2 * spatial_rank:
        raise UnsupportedOpError("LpPool pads rank mismatch")
    if auto_pad in ("", "NOTSET"):
        pad_begin = pads[:spatial_rank]
        pad_end = pads[spatial_rank:]
    elif auto_pad == "VALID":
        pad_begin = (0,) * spatial_rank
        pad_end = (0,) * spatial_rank
    else:
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
    p = float(node.attrs.get("p", 2))
    if p < 1.0:
        raise UnsupportedOpError("LpPool p must be >= 1")
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
            raise ShapeInferenceError("LpPool output shape must be non-negative")
        out_spatial.append(out_dim)
    output_shape = _value_shape(graph, node.outputs[0], node)
    expected_output_shape = (batch, channels, *out_spatial)
    if output_shape != expected_output_shape:
        raise ShapeInferenceError(
            f"LpPool output shape must be {expected_output_shape}, got {output_shape}"
        )
    return LpPoolSpec(
        batch=batch,
        channels=channels,
        spatial_rank=spatial_rank,
        in_spatial=in_spatial,
        out_spatial=tuple(out_spatial),
        kernel_shape=kernel_shape,
        dilations=dilations,
        strides=strides,
        pads=(*pad_begin, *pad_end),
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
        spatial_rank=spec.spatial_rank,
        in_spatial=spec.in_spatial,
        out_spatial=spec.out_spatial,
        kernel_shape=spec.kernel_shape,
        dilations=spec.dilations,
        strides=spec.strides,
        pads=spec.pads,
        p=spec.p,
        dtype=op_dtype,
    )

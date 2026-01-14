from __future__ import annotations

import math
from dataclasses import dataclass

from ..codegen.c_emitter import ConvOp
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from .common import node_dtype as _node_dtype
from .common import value_shape as _value_shape
from .registry import register_lowering


@dataclass(frozen=True)
class ConvSpec:
    batch: int
    in_channels: int
    out_channels: int
    spatial_rank: int
    in_spatial: tuple[int, ...]
    out_spatial: tuple[int, ...]
    kernel_shape: tuple[int, ...]
    strides: tuple[int, ...]
    pads: tuple[int, ...]
    dilations: tuple[int, ...]
    group: int


def resolve_conv_spec(graph: Graph, node: Node) -> ConvSpec:
    if len(node.inputs) not in {2, 3} or len(node.outputs) != 1:
        raise UnsupportedOpError("Conv must have 2 or 3 inputs and 1 output")
    supported_attrs = {
        "auto_pad",
        "dilations",
        "group",
        "kernel_shape",
        "pads",
        "strides",
    }
    if set(node.attrs) - supported_attrs:
        raise UnsupportedOpError("Conv has unsupported attributes")
    input_shape = _value_shape(graph, node.inputs[0], node)
    weight_shape = _value_shape(graph, node.inputs[1], node)
    if len(input_shape) < 3:
        raise UnsupportedOpError("Conv expects NCHW inputs with spatial dims")
    spatial_rank = len(input_shape) - 2
    if spatial_rank not in {1, 2, 3}:
        raise UnsupportedOpError("Conv supports 1D/2D/3D inputs only")
    if len(weight_shape) != spatial_rank + 2:
        raise UnsupportedOpError("Conv weight rank must match spatial rank")
    batch, in_channels = input_shape[0], input_shape[1]
    in_spatial = input_shape[2:]
    out_channels, weight_in_channels, *kernel_shape = weight_shape
    kernel_shape = node.attrs.get("kernel_shape")
    if kernel_shape is not None:
        kernel_shape = tuple(int(value) for value in kernel_shape)
        if len(kernel_shape) != spatial_rank:
            raise UnsupportedOpError(
                "Conv kernel_shape rank must match input spatial rank"
            )
        if kernel_shape != tuple(weight_shape[2:]):
            raise ShapeInferenceError(
                "Conv kernel_shape must match weights, "
                f"got {kernel_shape} and {tuple(weight_shape[2:])}"
            )
    else:
        kernel_shape = tuple(weight_shape[2:])
    group = int(node.attrs.get("group", 1))
    if group <= 0:
        raise UnsupportedOpError("Conv expects group >= 1")
    if in_channels % group != 0 or out_channels % group != 0:
        raise ShapeInferenceError(
            "Conv expects group to evenly divide in/out channels, "
            f"got group={group}, in_channels={in_channels}, "
            f"out_channels={out_channels}"
        )
    if weight_in_channels != in_channels // group:
        raise ShapeInferenceError(
            "Conv input channels must match weight channels, "
            f"got {in_channels} and {weight_in_channels * group}"
        )
    if len(node.inputs) == 3:
        bias_shape = _value_shape(graph, node.inputs[2], node)
        if bias_shape != (out_channels,):
            raise ShapeInferenceError(
                f"Conv bias shape must be {(out_channels,)}, got {bias_shape}"
            )
    strides = tuple(
        int(value) for value in node.attrs.get("strides", (1,) * spatial_rank)
    )
    if len(strides) != spatial_rank:
        raise UnsupportedOpError("Conv stride rank mismatch")
    dilations = tuple(
        int(value) for value in node.attrs.get("dilations", (1,) * spatial_rank)
    )
    if len(dilations) != spatial_rank:
        raise UnsupportedOpError("Conv dilation rank mismatch")
    pads = tuple(
        int(value)
        for value in node.attrs.get("pads", (0,) * (2 * spatial_rank))
    )
    if len(pads) != 2 * spatial_rank:
        raise UnsupportedOpError("Conv pads rank mismatch")
    auto_pad = node.attrs.get("auto_pad", b"NOTSET")
    if isinstance(auto_pad, bytes):
        auto_pad = auto_pad.decode("utf-8", errors="ignore")
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
            in_spatial, strides, dilations, kernel_shape
        ):
            effective_kernel = dilation * (kernel - 1) + 1
            out_dim = math.ceil(dim / stride)
            pad_needed = max(
                0, (out_dim - 1) * stride + effective_kernel - dim
            )
            if auto_pad == "SAME_UPPER":
                pad_start = pad_needed // 2
            else:
                pad_start = (pad_needed + 1) // 2
            pad_begin.append(pad_start)
            pad_end.append(pad_needed - pad_start)
        pad_begin = tuple(pad_begin)
        pad_end = tuple(pad_end)
    else:
        raise UnsupportedOpError("Conv has unsupported auto_pad mode")
    out_spatial = []
    for dim, stride, dilation, kernel, pad_start, pad_finish in zip(
        in_spatial, strides, dilations, kernel_shape, pad_begin, pad_end
    ):
        effective_kernel = dilation * (kernel - 1) + 1
        out_dim = (dim + pad_start + pad_finish - effective_kernel) // stride + 1
        if out_dim < 0:
            raise ShapeInferenceError("Conv output shape must be non-negative")
        out_spatial.append(out_dim)
    output_shape = _value_shape(graph, node.outputs[0], node)
    expected_output_shape = (batch, out_channels, *out_spatial)
    if output_shape != expected_output_shape:
        raise ShapeInferenceError(
            "Conv output shape must be "
            f"{expected_output_shape}, got {output_shape}"
        )
    return ConvSpec(
        batch=batch,
        in_channels=in_channels,
        out_channels=out_channels,
        spatial_rank=spatial_rank,
        in_spatial=in_spatial,
        out_spatial=tuple(out_spatial),
        kernel_shape=kernel_shape,
        strides=strides,
        pads=(*pad_begin, *pad_end),
        dilations=dilations,
        group=group,
    )


@register_lowering("Conv")
def lower_conv(graph: Graph, node: Node) -> ConvOp:
    if len(node.inputs) not in {2, 3} or len(node.outputs) != 1:
        raise UnsupportedOpError("Conv must have 2 or 3 inputs and 1 output")
    op_dtype = _node_dtype(graph, node, *node.inputs, *node.outputs)
    if not op_dtype.is_float:
        raise UnsupportedOpError(
            "Conv supports float16, float, and double inputs only"
        )
    spec = resolve_conv_spec(graph, node)
    return ConvOp(
        input0=node.inputs[0],
        weights=node.inputs[1],
        bias=node.inputs[2] if len(node.inputs) == 3 else None,
        output=node.outputs[0],
        batch=spec.batch,
        in_channels=spec.in_channels,
        out_channels=spec.out_channels,
        spatial_rank=spec.spatial_rank,
        in_spatial=spec.in_spatial,
        out_spatial=spec.out_spatial,
        kernel_shape=spec.kernel_shape,
        strides=spec.strides,
        pads=spec.pads,
        dilations=spec.dilations,
        group=spec.group,
        dtype=op_dtype,
    )

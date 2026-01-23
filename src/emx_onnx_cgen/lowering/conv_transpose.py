from __future__ import annotations

import math
from dataclasses import dataclass

from ..ir.ops import ConvTransposeOp
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from .common import node_dtype as _node_dtype
from .common import value_shape as _value_shape
from .registry import register_lowering


@dataclass(frozen=True)
class ConvTransposeSpec:
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
    output_padding: tuple[int, ...]
    group: int


def _split_padding(
    total_padding: int, auto_pad: str, *, dim: int
) -> tuple[int, int]:
    if total_padding < 0:
        raise ShapeInferenceError(
            "ConvTranspose output shape must be fully defined and non-negative"
        )
    pad_end = total_padding // 2
    pad_begin = total_padding - pad_end
    if auto_pad == "SAME_UPPER":
        pad_begin, pad_end = pad_end, pad_begin
    elif auto_pad not in {"SAME_LOWER", "NOTSET", ""}:
        raise UnsupportedOpError(
            f"ConvTranspose has unsupported auto_pad mode '{auto_pad}'"
        )
    if pad_begin < 0 or pad_end < 0:
        raise ShapeInferenceError(
            f"ConvTranspose pads must be non-negative for dim {dim}"
        )
    return pad_begin, pad_end


def resolve_conv_transpose_spec(graph: Graph, node: Node) -> ConvTransposeSpec:
    if len(node.inputs) not in {2, 3} or len(node.outputs) != 1:
        raise UnsupportedOpError(
            "ConvTranspose must have 2 or 3 inputs and 1 output"
        )
    supported_attrs = {
        "auto_pad",
        "dilations",
        "group",
        "kernel_shape",
        "output_padding",
        "output_shape",
        "pads",
        "strides",
    }
    if set(node.attrs) - supported_attrs:
        raise UnsupportedOpError("ConvTranspose has unsupported attributes")
    input_shape = _value_shape(graph, node.inputs[0], node)
    weight_shape = _value_shape(graph, node.inputs[1], node)
    if len(input_shape) < 3:
        raise UnsupportedOpError("ConvTranspose expects NCHW inputs with spatial dims")
    spatial_rank = len(input_shape) - 2
    if spatial_rank not in {1, 2, 3}:
        raise UnsupportedOpError("ConvTranspose supports 1D/2D/3D inputs only")
    if len(weight_shape) != spatial_rank + 2:
        raise UnsupportedOpError(
            "ConvTranspose weight rank must match spatial rank"
        )
    batch, in_channels = input_shape[0], input_shape[1]
    in_spatial = input_shape[2:]
    weight_in_channels, weight_out_channels, *kernel_shape = weight_shape
    kernel_attr = node.attrs.get("kernel_shape")
    if kernel_attr is not None:
        kernel_attr = tuple(int(value) for value in kernel_attr)
        if len(kernel_attr) != spatial_rank:
            raise UnsupportedOpError(
                "ConvTranspose kernel_shape rank must match input spatial rank"
            )
        if kernel_attr != tuple(kernel_shape):
            raise ShapeInferenceError(
                "ConvTranspose kernel_shape must match weights, "
                f"got {kernel_attr} and {tuple(kernel_shape)}"
            )
        kernel_shape = list(kernel_attr)
    else:
        kernel_shape = list(kernel_shape)
    group = int(node.attrs.get("group", 1))
    if group <= 0:
        raise UnsupportedOpError("ConvTranspose expects group >= 1")
    if in_channels % group != 0:
        raise ShapeInferenceError(
            "ConvTranspose expects group to evenly divide in channels, "
            f"got group={group}, in_channels={in_channels}"
        )
    if weight_in_channels != in_channels:
        raise ShapeInferenceError(
            "ConvTranspose input channels must match weight channels, "
            f"got {in_channels} and {weight_in_channels}"
        )
    out_channels = weight_out_channels * group
    if out_channels % group != 0:
        raise ShapeInferenceError(
            "ConvTranspose expects group to evenly divide out channels, "
            f"got group={group}, out_channels={out_channels}"
        )
    if len(node.inputs) == 3:
        bias_shape = _value_shape(graph, node.inputs[2], node)
        if bias_shape != (out_channels,):
            raise ShapeInferenceError(
                f"ConvTranspose bias shape must be {(out_channels,)}, got {bias_shape}"
            )
    strides = tuple(
        int(value) for value in node.attrs.get("strides", (1,) * spatial_rank)
    )
    if len(strides) != spatial_rank:
        raise UnsupportedOpError("ConvTranspose stride rank mismatch")
    dilations = tuple(
        int(value) for value in node.attrs.get("dilations", (1,) * spatial_rank)
    )
    if len(dilations) != spatial_rank:
        raise UnsupportedOpError("ConvTranspose dilation rank mismatch")
    output_padding = tuple(
        int(value)
        for value in node.attrs.get("output_padding", (0,) * spatial_rank)
    )
    if len(output_padding) != spatial_rank:
        raise UnsupportedOpError("ConvTranspose output_padding rank mismatch")
    for dim, (padding, stride) in enumerate(zip(output_padding, strides)):
        if padding < 0:
            raise UnsupportedOpError(
                "ConvTranspose output_padding must be non-negative"
            )
        if padding >= stride:
            raise UnsupportedOpError(
                "ConvTranspose output_padding must be smaller than stride"
            )
    pads = tuple(
        int(value)
        for value in node.attrs.get("pads", (0,) * (2 * spatial_rank))
    )
    if len(pads) != 2 * spatial_rank:
        raise UnsupportedOpError("ConvTranspose pads rank mismatch")
    auto_pad = node.attrs.get("auto_pad", b"NOTSET")
    if isinstance(auto_pad, bytes):
        auto_pad = auto_pad.decode("utf-8", errors="ignore")
    if auto_pad == "":
        auto_pad = "NOTSET"
    output_shape_attr = node.attrs.get("output_shape")
    output_shape: list[int] | None = None
    if output_shape_attr is not None:
        output_shape = [int(value) for value in output_shape_attr]
        if len(output_shape) != spatial_rank:
            raise UnsupportedOpError("ConvTranspose output_shape rank mismatch")
    if output_shape is not None:
        if auto_pad == "VALID":
            auto_pad = "NOTSET"
        pad_begin = []
        pad_end = []
        for dim, (in_dim, stride, dilation, kernel, out_dim, out_pad) in enumerate(
            zip(
                in_spatial,
                strides,
                dilations,
                kernel_shape,
                output_shape,
                output_padding,
            )
        ):
            effective_kernel = dilation * (kernel - 1) + 1
            total_padding = (
                stride * (in_dim - 1)
                + out_pad
                + effective_kernel
                - out_dim
            )
            pad_start, pad_finish = _split_padding(
                total_padding, auto_pad, dim=dim
            )
            pad_begin.append(pad_start)
            pad_end.append(pad_finish)
        out_spatial = output_shape
    else:
        if auto_pad == "VALID":
            pad_begin = [0] * spatial_rank
            pad_end = [0] * spatial_rank
        elif auto_pad in {"SAME_UPPER", "SAME_LOWER"}:
            pad_begin = []
            pad_end = []
            for dim, (in_dim, stride, dilation, kernel, out_pad) in enumerate(
                zip(in_spatial, strides, dilations, kernel_shape, output_padding)
            ):
                effective_kernel = dilation * (kernel - 1) + 1
                out_dim = in_dim * stride
                total_padding = (
                    stride * (in_dim - 1)
                    + out_pad
                    + effective_kernel
                    - out_dim
                )
                pad_start, pad_finish = _split_padding(
                    total_padding, auto_pad, dim=dim
                )
                pad_begin.append(pad_start)
                pad_end.append(pad_finish)
        elif auto_pad in {"NOTSET"}:
            pad_begin = list(pads[:spatial_rank])
            pad_end = list(pads[spatial_rank:])
        else:
            raise UnsupportedOpError(
                f"ConvTranspose has unsupported auto_pad mode '{auto_pad}'"
            )
        out_spatial = []
        for dim, (in_dim, stride, dilation, kernel, pad_start, pad_finish, out_pad) in enumerate(
            zip(
                in_spatial,
                strides,
                dilations,
                kernel_shape,
                pad_begin,
                pad_end,
                output_padding,
            )
        ):
            effective_kernel = dilation * (kernel - 1) + 1
            out_dim = (
                stride * (in_dim - 1)
                + out_pad
                + effective_kernel
                - pad_start
                - pad_finish
            )
            if out_dim < 0:
                raise ShapeInferenceError(
                    "ConvTranspose output shape must be non-negative"
                )
            out_spatial.append(out_dim)
    output_shape = _value_shape(graph, node.outputs[0], node)
    expected_output_shape = (batch, out_channels, *out_spatial)
    if output_shape != expected_output_shape:
        raise ShapeInferenceError(
            "ConvTranspose output shape must be "
            f"{expected_output_shape}, got {output_shape}"
        )
    return ConvTransposeSpec(
        batch=batch,
        in_channels=in_channels,
        out_channels=out_channels,
        spatial_rank=spatial_rank,
        in_spatial=in_spatial,
        out_spatial=tuple(out_spatial),
        kernel_shape=tuple(kernel_shape),
        strides=strides,
        pads=(*pad_begin, *pad_end),
        dilations=dilations,
        output_padding=output_padding,
        group=group,
    )


@register_lowering("ConvTranspose")
def lower_conv_transpose(graph: Graph, node: Node) -> ConvTransposeOp:
    if len(node.inputs) not in {2, 3} or len(node.outputs) != 1:
        raise UnsupportedOpError(
            "ConvTranspose must have 2 or 3 inputs and 1 output"
        )
    op_dtype = _node_dtype(graph, node, *node.inputs, *node.outputs)
    if not op_dtype.is_float:
        raise UnsupportedOpError(
            "ConvTranspose supports float16, float, and double inputs only"
        )
    spec = resolve_conv_transpose_spec(graph, node)
    return ConvTransposeOp(
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
        output_padding=spec.output_padding,
        group=spec.group,
        dtype=op_dtype,
    )

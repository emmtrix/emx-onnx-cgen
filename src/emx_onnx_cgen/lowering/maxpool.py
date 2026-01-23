from __future__ import annotations

import math
from dataclasses import dataclass

from shared.scalar_types import ScalarType

from ..ir.ops import MaxPoolOp
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from .common import node_dtype as _node_dtype
from .common import value_dtype as _value_dtype
from .common import value_shape as _value_shape
from .registry import register_lowering


@dataclass(frozen=True)
class MaxPoolSpec:
    batch: int
    channels: int
    spatial_rank: int
    in_spatial: tuple[int, ...]
    out_spatial: tuple[int, ...]
    kernel_shape: tuple[int, ...]
    strides: tuple[int, ...]
    pads: tuple[int, ...]
    dilations: tuple[int, ...]
    ceil_mode: bool
    storage_order: int


def resolve_maxpool_spec(graph: Graph, node: Node) -> MaxPoolSpec:
    if len(node.inputs) != 1 or len(node.outputs) not in {1, 2}:
        raise UnsupportedOpError("MaxPool must have 1 input and 1 or 2 outputs")
    supported_attrs = {
        "auto_pad",
        "ceil_mode",
        "dilations",
        "kernel_shape",
        "pads",
        "storage_order",
        "strides",
    }
    if set(node.attrs) - supported_attrs:
        raise UnsupportedOpError("MaxPool has unsupported attributes")
    storage_order = int(node.attrs.get("storage_order", 0))
    if storage_order not in (0, 1):
        raise UnsupportedOpError("MaxPool supports storage_order=0 or 1 only")
    kernel_shape = node.attrs.get("kernel_shape")
    if kernel_shape is None:
        raise UnsupportedOpError("MaxPool requires kernel_shape")
    kernel_shape = tuple(int(value) for value in kernel_shape)
    input_shape = _value_shape(graph, node.inputs[0], node)
    if len(input_shape) < 3:
        raise UnsupportedOpError("MaxPool expects NCHW inputs with spatial dims")
    spatial_rank = len(input_shape) - 2
    if spatial_rank not in {1, 2, 3}:
        raise UnsupportedOpError("MaxPool supports 1D/2D/3D inputs only")
    if len(kernel_shape) != spatial_rank:
        raise ShapeInferenceError(
            f"MaxPool kernel_shape must have {spatial_rank} dims, got {kernel_shape}"
        )
    strides = tuple(
        int(value) for value in node.attrs.get("strides", (1,) * spatial_rank)
    )
    if len(strides) != spatial_rank:
        raise UnsupportedOpError("MaxPool stride rank mismatch")
    dilations = tuple(
        int(value) for value in node.attrs.get("dilations", (1,) * spatial_rank)
    )
    if len(dilations) != spatial_rank:
        raise UnsupportedOpError("MaxPool dilation rank mismatch")
    pads = tuple(
        int(value)
        for value in node.attrs.get("pads", (0,) * (2 * spatial_rank))
    )
    if len(pads) != 2 * spatial_rank:
        raise UnsupportedOpError("MaxPool pads rank mismatch")
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
            input_shape[2:], strides, dilations, kernel_shape
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
        raise UnsupportedOpError("MaxPool has unsupported auto_pad mode")
    ceil_mode = int(node.attrs.get("ceil_mode", 0))
    if ceil_mode not in (0, 1):
        raise UnsupportedOpError("MaxPool supports ceil_mode=0 or 1 only")
    batch, channels = input_shape[0], input_shape[1]
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
            raise ShapeInferenceError(
                "MaxPool output shape must be non-negative"
            )
        out_spatial.append(out_dim)
    expected_output_shape = (batch, channels, *out_spatial)
    output_shape = _value_shape(graph, node.outputs[0], node)
    if output_shape != expected_output_shape:
        raise ShapeInferenceError(
            "MaxPool output shape must be "
            f"{expected_output_shape}, got {output_shape}"
        )
    if len(node.outputs) == 2:
        indices_shape = _value_shape(graph, node.outputs[1], node)
        if indices_shape != expected_output_shape:
            raise ShapeInferenceError(
                "MaxPool indices output shape must be "
                f"{expected_output_shape}, got {indices_shape}"
            )
        indices_dtype = _value_dtype(graph, node.outputs[1], node)
        if indices_dtype != ScalarType.I64:
            raise UnsupportedOpError("MaxPool indices output must be int64")
    pads = (*pad_begin, *pad_end)
    return MaxPoolSpec(
        batch=batch,
        channels=channels,
        spatial_rank=spatial_rank,
        in_spatial=in_spatial,
        out_spatial=tuple(out_spatial),
        kernel_shape=kernel_shape,
        strides=strides,
        pads=pads,
        dilations=dilations,
        ceil_mode=bool(ceil_mode),
        storage_order=storage_order,
    )


@register_lowering("MaxPool")
def lower_maxpool(graph: Graph, node: Node) -> MaxPoolOp:
    if len(node.inputs) != 1 or len(node.outputs) not in {1, 2}:
        raise UnsupportedOpError("MaxPool must have 1 input and 1 or 2 outputs")
    op_dtype = _node_dtype(graph, node, node.inputs[0], node.outputs[0])
    if op_dtype == ScalarType.BOOL:
        raise UnsupportedOpError("MaxPool supports numeric inputs only")
    spec = resolve_maxpool_spec(graph, node)
    indices = node.outputs[1] if len(node.outputs) == 2 else None
    indices_dtype = (
        _value_dtype(graph, indices, node) if indices is not None else None
    )
    if indices_dtype is not None and indices_dtype != ScalarType.I64:
        raise UnsupportedOpError("MaxPool indices output must be int64")
    return MaxPoolOp(
        input0=node.inputs[0],
        output=node.outputs[0],
        indices=indices,
        batch=spec.batch,
        channels=spec.channels,
        spatial_rank=spec.spatial_rank,
        in_spatial=spec.in_spatial,
        out_spatial=spec.out_spatial,
        kernel_shape=spec.kernel_shape,
        strides=spec.strides,
        pads=spec.pads,
        dilations=spec.dilations,
        ceil_mode=spec.ceil_mode,
        storage_order=spec.storage_order,
        dtype=op_dtype,
        indices_dtype=indices_dtype,
    )

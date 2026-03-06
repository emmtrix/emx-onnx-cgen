from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shared.scalar_types import ScalarType

from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Initializer, Node
from ..ir.ops import MaxUnpoolOp
from .common import node_dtype as _node_dtype
from .common import value_dtype as _value_dtype
from .common import value_shape as _value_shape
from .registry import register_lowering


def _find_initializer(graph: Graph, name: str) -> Initializer | None:
    for initializer in graph.initializers:
        if initializer.name == name:
            return initializer
    return None


def _shape_from_output_shape_input(
    graph: Graph, name: str, node: Node
) -> tuple[int, ...] | None:
    initializer = _find_initializer(graph, name)
    if initializer is None:
        return None
    if initializer.type.dtype not in {ScalarType.I64, ScalarType.I32}:
        raise UnsupportedOpError(
            "MaxUnpool output_shape input must be int64/int32 when constant"
        )
    values = np.array(initializer.data, dtype=np.int64).reshape(-1)
    if values.size == 0:
        raise ShapeInferenceError("MaxUnpool output_shape must not be empty")
    return tuple(int(value) for value in values)


@dataclass(frozen=True)
class MaxUnpoolSpec:
    batch: int
    channels: int
    spatial_rank: int
    in_spatial: tuple[int, ...]
    out_spatial: tuple[int, ...]
    inferred_out_spatial: tuple[int, ...]


def _resolve_maxunpool_spec(graph: Graph, node: Node) -> MaxUnpoolSpec:
    if len(node.inputs) not in {2, 3} or len(node.outputs) != 1:
        raise UnsupportedOpError("MaxUnpool must have 2 or 3 inputs and 1 output")
    supported_attrs = {"kernel_shape", "pads", "strides"}
    if set(node.attrs) - supported_attrs:
        raise UnsupportedOpError("MaxUnpool has unsupported attributes")
    kernel_shape = node.attrs.get("kernel_shape")
    if kernel_shape is None:
        raise UnsupportedOpError("MaxUnpool requires kernel_shape")
    kernel_shape = tuple(int(value) for value in kernel_shape)
    x_shape = _value_shape(graph, node.inputs[0], node)
    if len(x_shape) < 3:
        raise UnsupportedOpError("MaxUnpool expects NCHW inputs with spatial dims")
    spatial_rank = len(x_shape) - 2
    if spatial_rank not in {1, 2, 3}:
        raise UnsupportedOpError("MaxUnpool supports 1D/2D/3D inputs only")
    if len(kernel_shape) != spatial_rank:
        raise ShapeInferenceError(
            f"MaxUnpool kernel_shape must have {spatial_rank} dims, got {kernel_shape}"
        )
    strides = tuple(
        int(value) for value in node.attrs.get("strides", (1,) * spatial_rank)
    )
    if len(strides) != spatial_rank:
        raise UnsupportedOpError("MaxUnpool stride rank mismatch")
    pads = tuple(
        int(value) for value in node.attrs.get("pads", (0,) * (2 * spatial_rank))
    )
    if len(pads) != 2 * spatial_rank:
        raise UnsupportedOpError("MaxUnpool pads rank mismatch")
    in_spatial = x_shape[2:]
    pad_begin = pads[:spatial_rank]
    pad_end = pads[spatial_rank:]
    inferred_out_spatial = tuple(
        (dim - 1) * stride - (pad_start + pad_finish) + kernel
        for dim, stride, pad_start, pad_finish, kernel in zip(
            in_spatial, strides, pad_begin, pad_end, kernel_shape
        )
    )
    if any(dim < 0 for dim in inferred_out_spatial):
        raise ShapeInferenceError(
            f"MaxUnpool inferred output spatial dims must be non-negative, got {inferred_out_spatial}"
        )

    output_shape = _value_shape(graph, node.outputs[0], node)
    if len(output_shape) != len(x_shape):
        raise ShapeInferenceError(
            f"MaxUnpool output rank must be {len(x_shape)}, got {len(output_shape)}"
        )
    if output_shape[:2] != x_shape[:2]:
        raise ShapeInferenceError("MaxUnpool output must preserve N and C dimensions")

    if len(node.inputs) == 3:
        explicit_shape = _shape_from_output_shape_input(graph, node.inputs[2], node)
        if explicit_shape is not None and tuple(explicit_shape) != output_shape:
            raise ShapeInferenceError(
                "MaxUnpool output_shape input must match output tensor shape"
            )

    out_spatial = output_shape[2:]
    for out_dim, inferred_dim in zip(out_spatial, inferred_out_spatial):
        if out_dim < inferred_dim:
            raise ShapeInferenceError(
                "MaxUnpool output spatial dims must be >= inferred shape "
                f"{inferred_out_spatial}, got {out_spatial}"
            )

    indices_shape = _value_shape(graph, node.inputs[1], node)
    if indices_shape != x_shape:
        raise ShapeInferenceError(
            f"MaxUnpool indices shape must be {x_shape}, got {indices_shape}"
        )

    return MaxUnpoolSpec(
        batch=x_shape[0],
        channels=x_shape[1],
        spatial_rank=spatial_rank,
        in_spatial=in_spatial,
        out_spatial=out_spatial,
        inferred_out_spatial=inferred_out_spatial,
    )


@register_lowering("MaxUnpool")
def lower_maxunpool(graph: Graph, node: Node) -> MaxUnpoolOp:
    spec = _resolve_maxunpool_spec(graph, node)
    op_dtype = _node_dtype(graph, node, node.inputs[0], node.outputs[0])
    if op_dtype == ScalarType.BOOL:
        raise UnsupportedOpError("MaxUnpool supports numeric inputs only")
    indices_dtype = _value_dtype(graph, node.inputs[1], node)
    if indices_dtype != ScalarType.I64:
        raise UnsupportedOpError("MaxUnpool indices input must be int64")
    output_shape_name = node.inputs[2] if len(node.inputs) == 3 else None
    return MaxUnpoolOp(
        input0=node.inputs[0],
        indices=node.inputs[1],
        output_shape=output_shape_name,
        output=node.outputs[0],
        batch=spec.batch,
        channels=spec.channels,
        spatial_rank=spec.spatial_rank,
        in_spatial=spec.in_spatial,
        out_spatial=spec.out_spatial,
        inferred_out_spatial=spec.inferred_out_spatial,
        dtype=op_dtype,
        indices_dtype=indices_dtype,
    )

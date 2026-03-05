from __future__ import annotations

import itertools
from dataclasses import dataclass

from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from ..ir.ops import Col2ImOp
from .common import (
    _find_initializer,
    _shape_values_from_input,
    node_dtype as _node_dtype,
    value_shape as _value_shape,
)
from .registry import register_lowering


def _product(values: tuple[int, ...] | list[int]) -> int:
    result = 1
    for v in values:
        result *= v
    return result


def _read_const_int_values(
    graph: Graph, name: str, node: Node,
) -> tuple[int, ...] | None:
    values = _shape_values_from_input(graph, name, node)
    if values is not None:
        return tuple(int(v) for v in values)
    initializer = _find_initializer(graph, name)
    if initializer is not None:
        return tuple(int(v) for v in initializer.data.reshape(-1))
    return None


def _compute_col_dims(
    image_shape: tuple[int, ...],
    block_shape: tuple[int, ...],
    strides: tuple[int, ...],
    dilations: tuple[int, ...],
    pads: tuple[int, ...],
    spatial_rank: int,
) -> list[int]:
    col_dims: list[int] = []
    for i in range(spatial_rank):
        effective_block = dilations[i] * (block_shape[i] - 1) + 1
        pad_total = pads[i] + pads[i + spatial_rank]
        col_dim = (image_shape[i] + pad_total - effective_block) // strides[i] + 1
        if col_dim <= 0:
            raise ShapeInferenceError(
                f"Col2Im column dimension {i} is non-positive ({col_dim})"
            )
        col_dims.append(col_dim)
    return col_dims


def _infer_block_shape(
    kernel_total: int,
    L: int,
    image_shape: tuple[int, ...],
    strides: tuple[int, ...],
    dilations: tuple[int, ...],
    pads: tuple[int, ...],
    spatial_rank: int,
) -> tuple[int, ...] | None:
    """Infer block_shape by enumerating factorizations of kernel_total."""
    ranges = [
        range(1, min(kernel_total, image_shape[i] + dilations[i]) + 1)
        for i in range(spatial_rank)
    ]
    for combo in itertools.product(*ranges):
        if _product(combo) != kernel_total:
            continue
        try:
            col_dims = _compute_col_dims(
                image_shape, combo, strides, dilations, pads, spatial_rank
            )
        except ShapeInferenceError:
            continue
        if _product(col_dims) == L:
            return combo
    return None


@dataclass(frozen=True)
class Col2ImSpec:
    batch: int
    channels: int
    spatial_rank: int
    image_shape: tuple[int, ...]
    block_shape: tuple[int, ...]
    col_dims: tuple[int, ...]
    strides: tuple[int, ...]
    pads: tuple[int, ...]
    dilations: tuple[int, ...]


def resolve_col2im_spec(graph: Graph, node: Node) -> Col2ImSpec:
    if len(node.inputs) != 3 or len(node.outputs) != 1:
        raise UnsupportedOpError("Col2Im must have 3 inputs and 1 output")
    supported_attrs = {"dilations", "pads", "strides"}
    if set(node.attrs) - supported_attrs:
        raise UnsupportedOpError("Col2Im has unsupported attributes")

    input_shape = _value_shape(graph, node.inputs[0], node)
    if len(input_shape) != 3:
        raise UnsupportedOpError(
            f"Col2Im expects 3D input (N, C*prod(block_shape), L), "
            f"got rank {len(input_shape)}"
        )
    output_shape = _value_shape(graph, node.outputs[0], node)
    if len(output_shape) < 3:
        raise ShapeInferenceError(
            f"Col2Im output must be at least 3D, got rank {len(output_shape)}"
        )

    batch = input_shape[0]
    spatial_rank = len(output_shape) - 2

    image_shape_values = _read_const_int_values(graph, node.inputs[1], node)
    if image_shape_values is None:
        image_shape_values = output_shape[2:]
    image_shape = tuple(image_shape_values)
    if len(image_shape) != spatial_rank:
        raise ShapeInferenceError(
            f"Col2Im image_shape rank ({len(image_shape)}) must match "
            f"spatial rank ({spatial_rank})"
        )

    strides = tuple(
        int(v) for v in node.attrs.get("strides", (1,) * spatial_rank)
    )
    if len(strides) != spatial_rank:
        raise UnsupportedOpError(
            f"Col2Im strides rank ({len(strides)}) must match "
            f"spatial rank ({spatial_rank})"
        )

    dilations = tuple(
        int(v) for v in node.attrs.get("dilations", (1,) * spatial_rank)
    )
    if len(dilations) != spatial_rank:
        raise UnsupportedOpError(
            f"Col2Im dilations rank ({len(dilations)}) must match "
            f"spatial rank ({spatial_rank})"
        )

    pads = tuple(
        int(v) for v in node.attrs.get("pads", (0,) * (2 * spatial_rank))
    )
    if len(pads) != 2 * spatial_rank:
        raise UnsupportedOpError(
            f"Col2Im pads rank ({len(pads)}) must be "
            f"2 * spatial rank ({2 * spatial_rank})"
        )

    block_shape_values = _read_const_int_values(graph, node.inputs[2], node)
    if block_shape_values is not None:
        block_shape = tuple(block_shape_values)
    else:
        channels = output_shape[1]
        if channels <= 0:
            raise ShapeInferenceError("Col2Im output channels must be positive")
        if input_shape[1] % channels != 0:
            raise ShapeInferenceError(
                f"Col2Im input dim 1 ({input_shape[1]}) must be divisible by "
                f"output channels ({channels})"
            )
        kernel_total = input_shape[1] // channels
        L = input_shape[2]
        inferred = _infer_block_shape(
            kernel_total, L, image_shape, strides, dilations, pads, spatial_rank
        )
        if inferred is None:
            raise UnsupportedOpError(
                "Col2Im block_shape must be a compile-time constant; "
                "could not infer from model shapes"
            )
        block_shape = inferred

    if len(block_shape) != spatial_rank:
        raise ShapeInferenceError(
            f"Col2Im block_shape rank ({len(block_shape)}) must match "
            f"spatial rank ({spatial_rank})"
        )

    kernel_total = _product(block_shape)
    if input_shape[1] % kernel_total != 0:
        raise ShapeInferenceError(
            f"Col2Im input dim 1 ({input_shape[1]}) must be divisible by "
            f"prod(block_shape)={kernel_total}"
        )
    channels = input_shape[1] // kernel_total

    col_dims = _compute_col_dims(
        image_shape, block_shape, strides, dilations, pads, spatial_rank
    )

    expected_L = _product(col_dims)
    if input_shape[2] != expected_L:
        raise ShapeInferenceError(
            f"Col2Im input dim 2 ({input_shape[2]}) must equal "
            f"prod(col_dims)={expected_L} (col_dims={col_dims})"
        )

    expected_output_shape = (batch, channels, *image_shape)
    if output_shape != expected_output_shape:
        raise ShapeInferenceError(
            f"Col2Im output shape must be {expected_output_shape}, "
            f"got {output_shape}"
        )

    return Col2ImSpec(
        batch=batch,
        channels=channels,
        spatial_rank=spatial_rank,
        image_shape=image_shape,
        block_shape=block_shape,
        col_dims=tuple(col_dims),
        strides=strides,
        pads=pads,
        dilations=dilations,
    )


@register_lowering("Col2Im")
def lower_col2im(graph: Graph, node: Node) -> Col2ImOp:
    if len(node.inputs) != 3 or len(node.outputs) != 1:
        raise UnsupportedOpError("Col2Im must have 3 inputs and 1 output")
    op_dtype = _node_dtype(graph, node, node.inputs[0], node.outputs[0])
    if not op_dtype.is_float:
        raise UnsupportedOpError(
            "Col2Im supports float16, float, and double inputs only"
        )
    spec = resolve_col2im_spec(graph, node)
    return Col2ImOp(
        input0=node.inputs[0],
        output=node.outputs[0],
        batch=spec.batch,
        channels=spec.channels,
        spatial_rank=spec.spatial_rank,
        image_shape=spec.image_shape,
        block_shape=spec.block_shape,
        col_dims=spec.col_dims,
        strides=spec.strides,
        pads=spec.pads,
        dilations=spec.dilations,
        dtype=op_dtype,
    )

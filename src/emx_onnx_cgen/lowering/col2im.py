from __future__ import annotations

from dataclasses import dataclass
from math import prod

from ..ir.ops import Col2ImOp
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from .common import node_dtype as _node_dtype
from .common import value_shape as _value_shape
from .common import resolve_int_list_from_value
from .registry import register_lowering


def _compute_num_blocks(
    image_shape: tuple[int, ...],
    block_shape: tuple[int, ...],
    strides: tuple[int, ...],
    dilations: tuple[int, ...],
    pads: tuple[int, ...],
) -> tuple[int, ...]:
    spatial_rank = len(image_shape)
    pads_begin = pads[:spatial_rank]
    pads_end = pads[spatial_rank:]
    num_blocks = []
    for i in range(spatial_rank):
        s = image_shape[i]
        b = block_shape[i]
        d = dilations[i]
        stride = strides[i]
        pb = pads_begin[i]
        pe = pads_end[i]
        n = (s + pb + pe - d * (b - 1) - 1) // stride + 1
        if n <= 0:
            raise ShapeInferenceError(
                f"Col2Im: num_blocks[{i}] must be positive, got {n}"
            )
        num_blocks.append(n)
    return tuple(num_blocks)


@register_lowering("Col2Im")
def lower_col2im(graph: Graph, node: Node) -> Col2ImOp:
    if len(node.inputs) != 3 or len(node.outputs) != 1:
        raise UnsupportedOpError("Col2Im must have 3 inputs and 1 output")
    supported_attrs = {"dilations", "pads", "strides"}
    if set(node.attrs) - supported_attrs:
        raise UnsupportedOpError("Col2Im has unsupported attributes")

    input_name = node.inputs[0]
    image_shape_name = node.inputs[1]
    block_shape_name = node.inputs[2]
    output_name = node.outputs[0]

    input_shape = _value_shape(graph, input_name, node)
    if len(input_shape) != 3:
        raise UnsupportedOpError(
            f"Col2Im input must be 3-dimensional [N, C*M, L], got rank {len(input_shape)}"
        )
    batch = input_shape[0]
    cm = input_shape[1]

    image_shape_values = resolve_int_list_from_value(graph, image_shape_name, node)
    if image_shape_values is None:
        raise UnsupportedOpError(
            "Col2Im requires image_shape to be a compile-time constant"
        )
    block_shape_values = resolve_int_list_from_value(graph, block_shape_name, node)
    if block_shape_values is None:
        raise UnsupportedOpError(
            "Col2Im requires block_shape to be a compile-time constant"
        )
    spatial_rank = len(image_shape_values)
    if spatial_rank < 1:
        raise UnsupportedOpError("Col2Im requires at least 1 spatial dimension")
    if len(block_shape_values) != spatial_rank:
        raise UnsupportedOpError(
            "Col2Im block_shape and image_shape must have the same length"
        )

    image_shape = tuple(int(v) for v in image_shape_values)
    block_shape = tuple(int(v) for v in block_shape_values)

    strides = tuple(
        int(v) for v in node.attrs.get("strides", (1,) * spatial_rank)
    )
    if len(strides) != spatial_rank:
        raise UnsupportedOpError("Col2Im strides rank must match spatial rank")
    dilations = tuple(
        int(v) for v in node.attrs.get("dilations", (1,) * spatial_rank)
    )
    if len(dilations) != spatial_rank:
        raise UnsupportedOpError("Col2Im dilations rank must match spatial rank")
    pads = tuple(
        int(v) for v in node.attrs.get("pads", (0,) * (2 * spatial_rank))
    )
    if len(pads) != 2 * spatial_rank:
        raise UnsupportedOpError("Col2Im pads must have length 2 * spatial_rank")

    block_flat_size = prod(block_shape)
    if cm % block_flat_size != 0:
        raise ShapeInferenceError(
            f"Col2Im input dim 1 ({cm}) must be divisible by product(block_shape) ({block_flat_size})"
        )
    channels = cm // block_flat_size

    num_blocks = _compute_num_blocks(image_shape, block_shape, strides, dilations, pads)

    expected_l = prod(num_blocks)
    if input_shape[2] != expected_l:
        raise ShapeInferenceError(
            f"Col2Im input L dimension must be {expected_l}, got {input_shape[2]}"
        )

    op_dtype = _node_dtype(graph, node, input_name, output_name)

    output_shape = _value_shape(graph, output_name, node)
    expected_output_shape = (batch, channels, *image_shape)
    if output_shape != expected_output_shape:
        raise ShapeInferenceError(
            f"Col2Im output shape must be {expected_output_shape}, got {output_shape}"
        )

    return Col2ImOp(
        input0=input_name,
        output=output_name,
        batch=batch,
        channels=channels,
        spatial_rank=spatial_rank,
        image_shape=image_shape,
        block_shape=block_shape,
        num_blocks=num_blocks,
        strides=strides,
        dilations=dilations,
        pads=pads,
        dtype=op_dtype,
    )

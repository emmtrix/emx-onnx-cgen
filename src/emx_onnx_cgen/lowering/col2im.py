from __future__ import annotations

from math import prod

from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from .common import node_dtype as _node_dtype
from .common import value_shape as _value_shape
from .common import resolve_int_list_from_value
from ..ir.ops import Col2ImOp
from .registry import register_lowering


@register_lowering("Col2Im")
def lower_col2im(graph: Graph, node: Node) -> Col2ImOp:
    if len(node.inputs) != 3 or len(node.outputs) != 1:
        raise UnsupportedOpError("Col2Im must have 3 inputs and 1 output")
    op_dtype = _node_dtype(graph, node, node.inputs[0], node.outputs[0])
    input_shape = _value_shape(graph, node.inputs[0], node)
    if len(input_shape) != 3:
        raise UnsupportedOpError(
            f"Col2Im input must be 3D [N, C*prod(block_shape), L], got shape {input_shape}"
        )
    image_shape_vals = resolve_int_list_from_value(graph, node.inputs[1], node)
    if image_shape_vals is None:
        raise UnsupportedOpError(
            "Col2Im requires image_shape to be a static initializer"
        )
    block_shape_vals = resolve_int_list_from_value(graph, node.inputs[2], node)
    if block_shape_vals is None:
        raise UnsupportedOpError(
            "Col2Im requires block_shape to be a static initializer"
        )
    image_shape = tuple(image_shape_vals)
    block_shape = tuple(block_shape_vals)
    spatial_rank = len(image_shape)
    if spatial_rank < 2:
        raise UnsupportedOpError("Col2Im requires at least 2 spatial dimensions")
    if len(block_shape) != spatial_rank:
        raise UnsupportedOpError(
            "Col2Im block_shape must have the same length as image_shape"
        )
    supported_attrs = {"dilations", "pads", "strides"}
    if set(node.attrs) - supported_attrs:
        raise UnsupportedOpError("Col2Im has unsupported attributes")
    strides = tuple(
        int(v) for v in node.attrs.get("strides", (1,) * spatial_rank)
    )
    if len(strides) != spatial_rank:
        raise UnsupportedOpError("Col2Im strides rank mismatch")
    dilations = tuple(
        int(v) for v in node.attrs.get("dilations", (1,) * spatial_rank)
    )
    if len(dilations) != spatial_rank:
        raise UnsupportedOpError("Col2Im dilations rank mismatch")
    pads = tuple(
        int(v) for v in node.attrs.get("pads", (0,) * (2 * spatial_rank))
    )
    if len(pads) != 2 * spatial_rank:
        raise UnsupportedOpError("Col2Im pads rank mismatch")
    pad_begin = pads[:spatial_rank]
    pad_end = pads[spatial_rank:]
    out_blocks = []
    for d in range(spatial_rank):
        effective_kernel = dilations[d] * (block_shape[d] - 1) + 1
        n_blocks = (
            image_shape[d] + pad_begin[d] + pad_end[d] - effective_kernel
        ) // strides[d] + 1
        if n_blocks <= 0:
            raise ShapeInferenceError(
                f"Col2Im out_blocks[{d}] must be positive, got {n_blocks}"
            )
        out_blocks.append(n_blocks)
    out_blocks_tuple = tuple(out_blocks)
    L = prod(out_blocks_tuple)
    prod_block_shape = prod(block_shape)
    if input_shape[2] != L:
        raise ShapeInferenceError(
            f"Col2Im input L dimension must be {L}, got {input_shape[2]}"
        )
    if input_shape[1] % prod_block_shape != 0:
        raise ShapeInferenceError(
            f"Col2Im input channel dimension {input_shape[1]} must be divisible by "
            f"prod(block_shape)={prod_block_shape}"
        )
    channels = input_shape[1] // prod_block_shape
    batch = input_shape[0]
    expected_output_shape = (batch, channels, *image_shape)
    try:
        output_shape = _value_shape(graph, node.outputs[0], node)
        if output_shape != expected_output_shape:
            raise ShapeInferenceError(
                f"Col2Im output shape must be {expected_output_shape}, got {output_shape}"
            )
    except ShapeInferenceError as exc:
        if "Missing shape" not in str(exc):
            raise
    return Col2ImOp(
        input0=node.inputs[0],
        output=node.outputs[0],
        batch=batch,
        channels=channels,
        spatial_rank=spatial_rank,
        image_shape=image_shape,
        block_shape=block_shape,
        strides=strides,
        dilations=dilations,
        pads=pads,
        out_blocks=out_blocks_tuple,
        dtype=op_dtype,
    )

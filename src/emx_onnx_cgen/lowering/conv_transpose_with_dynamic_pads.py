from __future__ import annotations

from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from ..ir.ops import ConvTransposeWithDynamicPadsOp
from .common import node_dtype as _node_dtype
from .common import value_shape as _value_shape
from .registry import register_lowering


@register_lowering("ConvTransposeWithDynamicPads")
def lower_conv_transpose_with_dynamic_pads(
    graph: Graph, node: Node
) -> ConvTransposeWithDynamicPadsOp:
    # Inputs: X, W, Pads (no bias in this variant)
    if len(node.inputs) != 3 or len(node.outputs) != 1:
        raise UnsupportedOpError(
            "ConvTransposeWithDynamicPads must have exactly 3 inputs and 1 output"
        )

    op_dtype = _node_dtype(graph, node, node.inputs[0], node.inputs[1], node.outputs[0])
    if not op_dtype.is_float:
        raise UnsupportedOpError(
            "ConvTransposeWithDynamicPads supports float inputs only"
        )

    input_shape = _value_shape(graph, node.inputs[0], node)
    weight_shape = _value_shape(graph, node.inputs[1], node)

    if len(input_shape) < 3:
        raise UnsupportedOpError(
            "ConvTransposeWithDynamicPads expects inputs with spatial dims"
        )
    spatial_rank = len(input_shape) - 2
    if spatial_rank not in {1, 2, 3}:
        raise UnsupportedOpError(
            "ConvTransposeWithDynamicPads supports 1D/2D/3D inputs only"
        )
    if len(weight_shape) != spatial_rank + 2:
        raise UnsupportedOpError(
            "ConvTransposeWithDynamicPads weight rank must match spatial rank"
        )

    batch, in_channels = input_shape[0], input_shape[1]
    in_spatial = input_shape[2:]
    weight_in_channels, weight_out_channels = weight_shape[0], weight_shape[1]
    kernel_shape_from_weight = weight_shape[2:]

    kernel_attr = node.attrs.get("kernel_shape")
    if kernel_attr is not None:
        kernel_shape = tuple(int(v) for v in kernel_attr)
        if kernel_shape != tuple(kernel_shape_from_weight):
            raise ShapeInferenceError(
                "ConvTransposeWithDynamicPads kernel_shape mismatch with weights"
            )
    else:
        kernel_shape = tuple(kernel_shape_from_weight)

    group = int(node.attrs.get("group", 1))
    if group <= 0:
        raise UnsupportedOpError("ConvTransposeWithDynamicPads expects group >= 1")
    if in_channels % group != 0:
        raise ShapeInferenceError(
            "ConvTransposeWithDynamicPads in_channels must be divisible by group"
        )
    if weight_in_channels != in_channels:
        raise ShapeInferenceError(
            "ConvTransposeWithDynamicPads input channels must match weight channels"
        )

    out_channels = weight_out_channels * group

    strides = tuple(
        int(v) for v in node.attrs.get("strides", (1,) * spatial_rank)
    )
    dilations = tuple(
        int(v) for v in node.attrs.get("dilations", (1,) * spatial_rank)
    )
    output_padding = tuple(
        int(v) for v in node.attrs.get("output_padding", (0,) * spatial_rank)
    )

    # Validate pads tensor shape
    pads_shape = _value_shape(graph, node.inputs[2], node)
    if pads_shape != (2 * spatial_rank,):
        raise ShapeInferenceError(
            f"ConvTransposeWithDynamicPads pads tensor must have shape "
            f"[{2 * spatial_rank}], got {pads_shape}"
        )

    # Read output shape from graph (since pads are dynamic we trust the declared shape)
    output_shape = _value_shape(graph, node.outputs[0], node)
    if len(output_shape) != spatial_rank + 2:
        raise ShapeInferenceError(
            "ConvTransposeWithDynamicPads output rank mismatch"
        )
    if output_shape[0] != batch or output_shape[1] != out_channels:
        raise ShapeInferenceError(
            "ConvTransposeWithDynamicPads output batch/channels mismatch"
        )
    out_spatial = output_shape[2:]

    return ConvTransposeWithDynamicPadsOp(
        input0=node.inputs[0],
        weights=node.inputs[1],
        pads_tensor=node.inputs[2],
        output=node.outputs[0],
        batch=batch,
        in_channels=in_channels,
        out_channels=out_channels,
        spatial_rank=spatial_rank,
        in_spatial=in_spatial,
        out_spatial=out_spatial,
        kernel_shape=kernel_shape,
        strides=strides,
        dilations=dilations,
        output_padding=output_padding,
        group=group,
        dtype=op_dtype,
    )

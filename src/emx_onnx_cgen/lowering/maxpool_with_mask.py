from __future__ import annotations

from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from ..ir.ops import MaxpoolWithMaskOp
from .common import node_dtype, value_shape
from .maxpool import resolve_maxpool_spec
from .registry import register_lowering


@register_lowering("MaxpoolWithMask")
def lower_maxpool_with_mask(graph: Graph, node: Node) -> MaxpoolWithMaskOp:
    if len(node.inputs) != 2 or len(node.outputs) != 1:
        raise UnsupportedOpError(
            "MaxpoolWithMask must have exactly 2 inputs and 1 output"
        )

    op_dtype = node_dtype(graph, node, node.inputs[0], node.outputs[0])
    if not op_dtype.is_float:
        raise UnsupportedOpError("MaxpoolWithMask supports float inputs only")

    # Temporarily create a proxy node with only the X input so we can reuse
    # resolve_maxpool_spec from the standard MaxPool lowering.
    class _ProxyNode:
        def __init__(self, original: Node) -> None:
            self.inputs = (original.inputs[0],)
            self.outputs = original.outputs
            self.attrs = original.attrs
            self.op_type = original.op_type
            self.name = original.name
            self.domain = original.domain

    proxy = _ProxyNode(node)
    spec = resolve_maxpool_spec(graph, proxy)  # type: ignore[arg-type]

    if spec.spatial_rank != 2:
        raise UnsupportedOpError(
            "MaxpoolWithMask currently supports only 2D spatial inputs"
        )

    mask_shape = value_shape(graph, node.inputs[1], node)
    input_shape = (spec.batch, spec.channels, *spec.in_spatial)
    if mask_shape != input_shape:
        raise ShapeInferenceError(
            f"MaxpoolWithMask mask shape {mask_shape} must match input shape {input_shape}"
        )

    return MaxpoolWithMaskOp(
        input0=node.inputs[0],
        mask=node.inputs[1],
        output=node.outputs[0],
        batch=spec.batch,
        channels=spec.channels,
        in_h=spec.in_spatial[0],
        in_w=spec.in_spatial[1],
        out_h=spec.out_spatial[0],
        out_w=spec.out_spatial[1],
        kernel_shape=(spec.kernel_shape[0], spec.kernel_shape[1]),
        strides=(spec.strides[0], spec.strides[1]),
        pads=(spec.pads[0], spec.pads[1], spec.pads[2], spec.pads[3]),
        dilations=(spec.dilations[0], spec.dilations[1]),
        dtype=op_dtype,
    )

from __future__ import annotations

from ..codegen.c_emitter import MeanVarianceNormalizationOp
from ..errors import UnsupportedOpError
from ..ir.model import Graph, Node
from ..validation import ensure_output_shape_matches_input
from .common import node_dtype, shape_product, value_shape
from .reduce import normalize_reduce_axes
from .registry import register_lowering


@register_lowering("MeanVarianceNormalization")
def lower_mean_variance_normalization(
    graph: Graph, node: Node
) -> MeanVarianceNormalizationOp:
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        raise UnsupportedOpError(
            "MeanVarianceNormalization must have 1 input and 1 output"
        )
    op_dtype = node_dtype(graph, node, *node.inputs, *node.outputs)
    if not op_dtype.is_float:
        raise UnsupportedOpError(
            "MeanVarianceNormalization supports float16, float, and double inputs only"
        )
    input_shape = value_shape(graph, node.inputs[0], node)
    output_shape = value_shape(graph, node.outputs[0], node)
    ensure_output_shape_matches_input(node, input_shape, output_shape)
    axes_attr = node.attrs.get("axes")
    if axes_attr is None:
        axes = (0, 2, 3)
    else:
        axes = tuple(int(axis) for axis in axes_attr)
    axes = normalize_reduce_axes(axes, input_shape, node)
    if not axes:
        raise UnsupportedOpError(
            "MeanVarianceNormalization requires non-empty reduction axes"
        )
    non_axes = tuple(i for i in range(len(input_shape)) if i not in axes)
    reduce_count = shape_product(tuple(input_shape[axis] for axis in axes))
    return MeanVarianceNormalizationOp(
        input0=node.inputs[0],
        output=node.outputs[0],
        shape=input_shape,
        axes=axes,
        non_axes=non_axes,
        reduce_count=reduce_count,
        epsilon=1e-9,
        dtype=op_dtype,
    )

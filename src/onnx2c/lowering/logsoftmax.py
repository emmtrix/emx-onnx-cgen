from __future__ import annotations

from ..codegen.c_emitter import LogSoftmaxOp
from ..errors import UnsupportedOpError
from ..ir.model import Graph, Node
from .common import node_dtype as _node_dtype
from .common import shape_product as _shape_product
from .common import value_shape as _value_shape
from .registry import register_lowering
from ..validation import ensure_output_shape_matches_input
from ..validation import normalize_axis as _normalize_axis


@register_lowering("LogSoftmax")
def lower_logsoftmax(graph: Graph, node: Node) -> LogSoftmaxOp:
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        raise UnsupportedOpError("LogSoftmax must have 1 input and 1 output")
    op_dtype = _node_dtype(graph, node, *node.inputs, *node.outputs)
    if op_dtype not in {"float", "double"}:
        raise UnsupportedOpError(
            "LogSoftmax supports float and double inputs only"
        )
    input_shape = _value_shape(graph, node.inputs[0], node)
    output_shape = _value_shape(graph, node.outputs[0], node)
    ensure_output_shape_matches_input(node, input_shape, output_shape)
    axis = _normalize_axis(
        int(node.attrs.get("axis", -1)),
        input_shape,
        node,
    )
    outer = _shape_product(input_shape[:axis]) if axis > 0 else 1
    axis_size = input_shape[axis]
    inner = (
        _shape_product(input_shape[axis + 1 :])
        if axis + 1 < len(input_shape)
        else 1
    )
    return LogSoftmaxOp(
        input0=node.inputs[0],
        output=node.outputs[0],
        outer=outer,
        axis_size=axis_size,
        inner=inner,
        axis=axis,
        shape=input_shape,
        dtype=op_dtype,
    )

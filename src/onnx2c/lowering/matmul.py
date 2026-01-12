from __future__ import annotations

from ..codegen.c_emitter import MatMulOp
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from .common import node_dtype as _node_dtype
from .common import value_shape as _value_shape
from .registry import register_lowering


@register_lowering("MatMul")
def lower_matmul(graph: Graph, node: Node) -> MatMulOp:
    if len(node.inputs) != 2 or len(node.outputs) != 1:
        raise UnsupportedOpError("MatMul must have 2 inputs and 1 output")
    op_dtype = _node_dtype(graph, node, *node.inputs, *node.outputs)
    input0_shape = _value_shape(graph, node.inputs[0], node)
    input1_shape = _value_shape(graph, node.inputs[1], node)
    if len(input0_shape) != 2 or len(input1_shape) != 2:
        raise UnsupportedOpError(
            "MatMul supports 2D inputs only, "
            f"got {input0_shape} x {input1_shape}"
        )
    m, k_left = input0_shape
    k_right, n = input1_shape
    if k_left != k_right:
        raise ShapeInferenceError(
            f"MatMul inner dimensions must match, got {k_left} and {k_right}"
        )
    output_shape = _value_shape(graph, node.outputs[0], node)
    if output_shape != (m, n):
        raise ShapeInferenceError(
            f"MatMul output shape must be {(m, n)}, got {output_shape}"
        )
    return MatMulOp(
        input0=node.inputs[0],
        input1=node.inputs[1],
        output=node.outputs[0],
        m=m,
        n=n,
        k=k_left,
        dtype=op_dtype,
    )

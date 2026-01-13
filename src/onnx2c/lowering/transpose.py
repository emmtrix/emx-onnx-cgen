from __future__ import annotations

from ..codegen.c_emitter import TransposeOp
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from .common import node_dtype as _node_dtype
from .common import value_shape as _value_shape
from .registry import register_lowering


@register_lowering("Transpose")
def lower_transpose(graph: Graph, node: Node) -> TransposeOp:
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        raise UnsupportedOpError("Transpose must have 1 input and 1 output")
    input_shape = _value_shape(graph, node.inputs[0], node)
    output_shape = _value_shape(graph, node.outputs[0], node)
    perm = node.attrs.get("perm")
    if perm is None:
        perm = tuple(reversed(range(len(input_shape))))
    else:
        perm = tuple(int(axis) for axis in perm)
    if len(perm) != len(input_shape):
        raise ShapeInferenceError(
            "Transpose perm must match input rank, "
            f"got perm {perm} for shape {input_shape}"
        )
    if set(perm) != set(range(len(input_shape))):
        raise UnsupportedOpError(
            f"Transpose perm must be a permutation, got {perm}"
        )
    expected_shape = tuple(input_shape[axis] for axis in perm)
    if output_shape != expected_shape:
        raise ShapeInferenceError(
            "Transpose output shape must match permuted input shape, "
            f"expected {expected_shape}, got {output_shape}"
        )
    op_dtype = _node_dtype(graph, node, *node.inputs, *node.outputs)
    return TransposeOp(
        input0=node.inputs[0],
        output=node.outputs[0],
        perm=perm,
        input_shape=input_shape,
        output_shape=output_shape,
        dtype=op_dtype,
        input_dtype=op_dtype,
    )

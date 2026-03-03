from __future__ import annotations

from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.context import GraphContext
from ..ir.model import Graph, Node
from ..ir.ops import TransposeOp
from .common import value_has_dim_params as _value_has_dim_params
from .common import value_shape as _value_shape
from .registry import register_lowering


@register_lowering("Transpose")
def lower_transpose(graph: Graph, node: Node) -> TransposeOp:
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        raise UnsupportedOpError("Transpose must have 1 input and 1 output")
    input_shape = _value_shape(graph, node.inputs[0], node)
    output_shape = _value_shape(graph, node.outputs[0], node)
    if _value_has_dim_params(graph, node.outputs[0]) or not output_shape:
        output_shape = ()
    perm = node.attrs.get("perm")
    if perm is None:
        perm = tuple(reversed(range(len(input_shape))))
    else:
        perm = tuple(int(axis) for axis in perm)
    if set(perm) != set(range(len(perm))):
        raise UnsupportedOpError(f"Transpose perm must be a permutation, got {perm}")
    input_rank_unknown = input_shape == () and len(perm) > 0
    if input_rank_unknown:
        if output_shape and len(output_shape) == len(perm):
            inverse_perm = [0] * len(perm)
            for out_axis, in_axis in enumerate(perm):
                inverse_perm[in_axis] = out_axis
            inferred_input_shape = tuple(output_shape[idx] for idx in inverse_perm)
            if isinstance(graph, GraphContext):
                graph.set_shape(node.inputs[0], inferred_input_shape)
                graph.set_shape(node.outputs[0], output_shape)
        return TransposeOp(
            input0=node.inputs[0],
            output=node.outputs[0],
            perm=perm,
        )
    if len(perm) != len(input_shape):
        raise ShapeInferenceError(
            "Transpose perm must match input rank, "
            f"got perm {perm} for shape {input_shape}"
        )
    expected_shape = tuple(input_shape[axis] for axis in perm)
    if output_shape and output_shape != expected_shape:
        raise ShapeInferenceError(
            "Transpose output shape must match permuted input shape, "
            f"expected {expected_shape}, got {output_shape}"
        )
    if isinstance(graph, GraphContext):
        graph.set_shape(node.outputs[0], expected_shape)
    return TransposeOp(
        input0=node.inputs[0],
        output=node.outputs[0],
        perm=perm,
    )

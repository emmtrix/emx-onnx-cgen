from __future__ import annotations

from ..codegen.c_emitter import EinsumKind, EinsumOp
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from .common import node_dtype as _node_dtype
from .common import value_shape as _value_shape
from .registry import register_lowering


def _normalize_equation(equation: str) -> str:
    return equation.replace(" ", "")


@register_lowering("Einsum")
def lower_einsum(graph: Graph, node: Node) -> EinsumOp:
    if not node.inputs or len(node.outputs) != 1:
        raise UnsupportedOpError("Einsum must have 1 output and at least 1 input")
    equation_value = node.attrs.get("equation")
    if equation_value is None:
        raise UnsupportedOpError("Einsum equation attribute is required")
    equation = (
        equation_value.decode()
        if isinstance(equation_value, (bytes, bytearray))
        else str(equation_value)
    )
    normalized = _normalize_equation(equation)
    input_shapes = tuple(
        _value_shape(graph, name, node) for name in node.inputs
    )
    output_shape = _value_shape(graph, node.outputs[0], node)
    op_dtype = _node_dtype(graph, node, *node.inputs, *node.outputs)
    if normalized == "->":
        if len(node.inputs) != 1:
            raise UnsupportedOpError("Einsum '->' must have 1 input")
        if output_shape:
            raise ShapeInferenceError(
                "Einsum '->' output must be scalar, "
                f"got shape {output_shape}"
            )
        kind = EinsumKind.REDUCE_ALL
    elif normalized == "ij->i":
        if len(node.inputs) != 1:
            raise UnsupportedOpError("Einsum 'ij->i' must have 1 input")
        input_shape = input_shapes[0]
        if len(input_shape) != 2:
            raise ShapeInferenceError(
                "Einsum 'ij->i' input must be 2D, "
                f"got shape {input_shape}"
            )
        expected = (input_shape[0],)
        if output_shape != expected:
            raise ShapeInferenceError(
                f"Einsum 'ij->i' output must match shape {expected}, "
                f"got {output_shape}"
            )
        kind = EinsumKind.SUM_J
    elif normalized == "ij->ji":
        if len(node.inputs) != 1:
            raise UnsupportedOpError("Einsum 'ij->ji' must have 1 input")
        input_shape = input_shapes[0]
        if len(input_shape) != 2:
            raise ShapeInferenceError(
                "Einsum 'ij->ji' input must be 2D, "
                f"got shape {input_shape}"
            )
        expected = (input_shape[1], input_shape[0])
        if output_shape != expected:
            raise ShapeInferenceError(
                f"Einsum 'ij->ji' output must match shape {expected}, "
                f"got {output_shape}"
            )
        kind = EinsumKind.TRANSPOSE
    elif normalized in {"i,i", "i,i->"}:
        if len(node.inputs) != 2:
            raise UnsupportedOpError("Einsum 'i,i' must have 2 inputs")
        left_shape, right_shape = input_shapes
        if len(left_shape) != 1 or len(right_shape) != 1:
            raise ShapeInferenceError(
                "Einsum 'i,i' inputs must be vectors, "
                f"got shapes {left_shape} and {right_shape}"
            )
        if left_shape[0] != right_shape[0]:
            raise ShapeInferenceError(
                "Einsum 'i,i' inputs must have the same length, "
                f"got shapes {left_shape} and {right_shape}"
            )
        if output_shape:
            raise ShapeInferenceError(
                "Einsum 'i,i' output must be scalar, "
                f"got shape {output_shape}"
            )
        kind = EinsumKind.DOT
    elif normalized == "bij,bjk->bik":
        if len(node.inputs) != 2:
            raise UnsupportedOpError("Einsum 'bij,bjk->bik' must have 2 inputs")
        left_shape, right_shape = input_shapes
        if len(left_shape) != 3 or len(right_shape) != 3:
            raise ShapeInferenceError(
                "Einsum 'bij,bjk->bik' inputs must be 3D, "
                f"got shapes {left_shape} and {right_shape}"
            )
        if left_shape[0] != right_shape[0]:
            raise ShapeInferenceError(
                "Einsum 'bij,bjk->bik' batch dimensions must match, "
                f"got shapes {left_shape} and {right_shape}"
            )
        if left_shape[2] != right_shape[1]:
            raise ShapeInferenceError(
                "Einsum 'bij,bjk->bik' contraction dimensions must match, "
                f"got shapes {left_shape} and {right_shape}"
            )
        expected = (left_shape[0], left_shape[1], right_shape[2])
        if output_shape != expected:
            raise ShapeInferenceError(
                f"Einsum 'bij,bjk->bik' output must match shape {expected}, "
                f"got {output_shape}"
            )
        kind = EinsumKind.BATCH_MATMUL
    elif normalized == "...ii->...i":
        if len(node.inputs) != 1:
            raise UnsupportedOpError("Einsum '...ii->...i' must have 1 input")
        input_shape = input_shapes[0]
        if len(input_shape) < 2:
            raise ShapeInferenceError(
                "Einsum '...ii->...i' input must be at least 2D, "
                f"got shape {input_shape}"
            )
        if input_shape[-1] != input_shape[-2]:
            raise ShapeInferenceError(
                "Einsum '...ii->...i' requires last two dims to match, "
                f"got shape {input_shape}"
            )
        expected = (*input_shape[:-2], input_shape[-1])
        if output_shape != expected:
            raise ShapeInferenceError(
                f"Einsum '...ii->...i' output must match shape {expected}, "
                f"got {output_shape}"
            )
        kind = EinsumKind.BATCH_DIAGONAL
    else:
        raise UnsupportedOpError(
            f"Unsupported Einsum equation '{equation}'"
        )
    return EinsumOp(
        inputs=tuple(node.inputs),
        output=node.outputs[0],
        kind=kind,
        input_shapes=input_shapes,
        output_shape=output_shape,
        dtype=op_dtype,
        input_dtype=op_dtype,
    )

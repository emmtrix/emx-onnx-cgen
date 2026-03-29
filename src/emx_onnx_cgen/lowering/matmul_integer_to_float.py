from __future__ import annotations

from shared.scalar_types import ScalarType

from ..ir.ops import MatMulIntegerToFloatOp
from ..ir.context import GraphContext
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from .common import value_dtype as _value_dtype
from .common import value_shape as _value_shape
from .registry import register_lowering


def _ensure_scalar_or_per_column(
    graph: Graph, name: str, node: Node, n: int, label: str
) -> tuple[tuple[int, ...], bool]:
    """Return (shape, per_column).  Accepts scalar () / (1,) or per-column (N,)."""
    shape = _value_shape(graph, name, node)
    if shape in {(), (1,)}:
        return shape, False
    if len(shape) == 1 and shape[0] == n:
        return shape, True
    raise UnsupportedOpError(
        f"MatMulIntegerToFloat {label} must be scalar or shape [{n}], got {shape}"
    )


def _ensure_scalar_input(
    graph: Graph, name: str, node: Node, label: str
) -> tuple[int, ...]:
    shape = _value_shape(graph, name, node)
    if shape not in {(), (1,)}:
        raise UnsupportedOpError(
            f"MatMulIntegerToFloat {label} must be scalar, got shape {shape}"
        )
    return shape


@register_lowering("MatMulIntegerToFloat")
def lower_matmul_integer_to_float(graph: Graph, node: Node) -> MatMulIntegerToFloatOp:
    # Inputs: A, B, a_scale, b_scale, [a_zero_point, b_zero_point, bias]
    if len(node.inputs) != 7 or len(node.outputs) != 1:
        raise UnsupportedOpError("MatMulIntegerToFloat must have 7 inputs and 1 output")

    input_a_name = node.inputs[0]
    input_b_name = node.inputs[1]
    a_scale_name = node.inputs[2]
    b_scale_name = node.inputs[3]
    a_zero_name: str | None = node.inputs[4] if node.inputs[4] else None
    b_zero_name: str | None = node.inputs[5] if node.inputs[5] else None
    bias_name: str | None = node.inputs[6] if node.inputs[6] else None

    # A and B must be quantized integers.
    input_a_dtype = _value_dtype(graph, input_a_name, node)
    input_b_dtype = _value_dtype(graph, input_b_name, node)
    if input_a_dtype not in {ScalarType.U8, ScalarType.I8}:
        raise UnsupportedOpError(
            "MatMulIntegerToFloat supports uint8/int8 A input only"
        )
    if input_b_dtype not in {ScalarType.U8, ScalarType.I8}:
        raise UnsupportedOpError(
            "MatMulIntegerToFloat supports uint8/int8 B input only"
        )

    # Scales must be float.
    a_scale_dtype = _value_dtype(graph, a_scale_name, node)
    b_scale_dtype = _value_dtype(graph, b_scale_name, node)
    if not a_scale_dtype.is_float:
        raise UnsupportedOpError(
            "MatMulIntegerToFloat a_scale must be float16/float/double"
        )
    if not b_scale_dtype.is_float:
        raise UnsupportedOpError(
            "MatMulIntegerToFloat b_scale must be float16/float/double"
        )

    # Resolve matrix shapes and output dimensions.
    input_a_shape = _value_shape(graph, input_a_name, node)
    input_b_shape = _value_shape(graph, input_b_name, node)
    if len(input_a_shape) != 2 or len(input_b_shape) != 2:
        raise UnsupportedOpError(
            "MatMulIntegerToFloat supports 2D inputs only, "
            f"got {input_a_shape} x {input_b_shape}"
        )
    m, k_left = input_a_shape
    k_right, n = input_b_shape
    if k_left != k_right:
        raise ShapeInferenceError(
            "MatMulIntegerToFloat inner dimensions must match, "
            f"got {k_left} and {k_right}"
        )

    # a_scale must be scalar.
    a_scale_shape = _ensure_scalar_input(graph, a_scale_name, node, "a_scale")

    # b_scale may be scalar or per-column [N].
    b_scale_shape, b_scale_per_column = _ensure_scalar_or_per_column(
        graph, b_scale_name, node, n, "b_scale"
    )

    # Optional zero points.
    a_zero_shape: tuple[int, ...] | None = None
    b_zero_shape: tuple[int, ...] | None = None
    if a_zero_name is not None:
        a_zero_dtype = _value_dtype(graph, a_zero_name, node)
        if a_zero_dtype != input_a_dtype:
            raise UnsupportedOpError(
                "MatMulIntegerToFloat a_zero_point dtype must match A"
            )
        a_zero_shape = _ensure_scalar_input(graph, a_zero_name, node, "a_zero_point")
    if b_zero_name is not None:
        b_zero_dtype = _value_dtype(graph, b_zero_name, node)
        if b_zero_dtype != input_b_dtype:
            raise UnsupportedOpError(
                "MatMulIntegerToFloat b_zero_point dtype must match B"
            )
        b_zero_shape, _ = _ensure_scalar_or_per_column(
            graph, b_zero_name, node, n, "b_zero_point"
        )

    # Optional bias: must be 1D [N].
    bias_dtype: ScalarType | None = None
    bias_shape: tuple[int, ...] | None = None
    if bias_name is not None:
        bias_dtype = _value_dtype(graph, bias_name, node)
        if not bias_dtype.is_float:
            raise UnsupportedOpError(
                "MatMulIntegerToFloat bias must be float16/float/double"
            )
        raw_bias_shape = _value_shape(graph, bias_name, node)
        if raw_bias_shape not in {(n,), (1,)}:
            raise UnsupportedOpError(
                f"MatMulIntegerToFloat bias must be shape [{n}] or [1], "
                f"got {raw_bias_shape}"
            )
        bias_shape = raw_bias_shape

    output_shape = (m, n)
    output_dtype = ScalarType.F32

    lowered = MatMulIntegerToFloatOp(
        input_a=input_a_name,
        input_b=input_b_name,
        a_scale=a_scale_name,
        b_scale=b_scale_name,
        a_zero_point=a_zero_name,
        b_zero_point=b_zero_name,
        bias=bias_name,
        output=node.outputs[0],
        input_a_dtype=input_a_dtype,
        input_b_dtype=input_b_dtype,
        dtype=output_dtype,
        a_scale_dtype=a_scale_dtype,
        b_scale_dtype=b_scale_dtype,
        bias_dtype=bias_dtype,
        a_scale_shape=a_scale_shape,
        b_scale_shape=b_scale_shape,
        a_zero_shape=a_zero_shape,
        b_zero_shape=b_zero_shape,
        bias_shape=bias_shape,
        b_scale_per_column=b_scale_per_column,
        m=m,
        n=n,
        k=k_left,
    )

    if isinstance(graph, GraphContext):
        graph.set_shape(node.outputs[0], output_shape)
        graph.set_dtype(node.outputs[0], output_dtype)

    return lowered

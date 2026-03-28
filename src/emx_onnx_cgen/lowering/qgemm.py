from __future__ import annotations

from shared.scalar_types import ScalarType

from ..ir.ops import QGemmOp
from ..ir.context import GraphContext
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from .common import value_dtype as _value_dtype
from .common import value_shape as _value_shape
from .registry import register_lowering


def _ensure_scalar_input(
    graph: Graph, name: str, node: Node, label: str
) -> tuple[int, ...]:
    shape = _value_shape(graph, name, node)
    if shape not in {(), (1,)}:
        raise UnsupportedOpError(f"QGemm {label} must be scalar, got shape {shape}")
    return shape


def _ensure_scale_dtype(dtype: ScalarType, label: str) -> None:
    if not dtype.is_float:
        raise UnsupportedOpError(f"QGemm {label} must be float16/float/double")


@register_lowering("QGemm")
def lower_qgemm(graph: Graph, node: Node) -> QGemmOp:
    if len(node.inputs) != 9 or len(node.outputs) != 1:
        raise UnsupportedOpError("QGemm must have 9 inputs and 1 output")

    # Parse optional inputs (empty string means absent).
    input_c_name: str | None = node.inputs[6] if node.inputs[6] else None
    y_scale_name: str | None = node.inputs[7] if node.inputs[7] else None
    y_zero_name: str | None = node.inputs[8] if node.inputs[8] else None

    # A (input 0) and B (input 3) must be quantized integers.
    input_a_dtype = _value_dtype(graph, node.inputs[0], node)
    input_b_dtype = _value_dtype(graph, node.inputs[3], node)
    if input_a_dtype not in {ScalarType.U8, ScalarType.I8}:
        raise UnsupportedOpError("QGemm supports uint8/int8 A input only")
    if input_b_dtype not in {ScalarType.U8, ScalarType.I8}:
        raise UnsupportedOpError("QGemm supports uint8/int8 B input only")

    # Scale dtypes must be float.
    a_scale_dtype = _value_dtype(graph, node.inputs[1], node)
    b_scale_dtype = _value_dtype(graph, node.inputs[4], node)
    _ensure_scale_dtype(a_scale_dtype, "a_scale")
    _ensure_scale_dtype(b_scale_dtype, "b_scale")

    # Zero point dtypes must match their respective matrix dtypes.
    a_zero_dtype = _value_dtype(graph, node.inputs[2], node)
    b_zero_dtype = _value_dtype(graph, node.inputs[5], node)
    if a_zero_dtype != input_a_dtype:
        raise UnsupportedOpError("QGemm a_zero_point dtype must match A")
    if b_zero_dtype != input_b_dtype:
        raise UnsupportedOpError("QGemm b_zero_point dtype must match B")

    # A scale/zero must be scalar.
    a_scale_shape = _ensure_scalar_input(graph, node.inputs[1], node, "a_scale")
    a_zero_shape = _ensure_scalar_input(graph, node.inputs[2], node, "a_zero_point")

    # B scale/zero may be scalar or per-column (shape [N]).
    b_scale_shape = _value_shape(graph, node.inputs[4], node)
    b_zero_shape = _value_shape(graph, node.inputs[5], node)

    # Determine transpose attributes.
    trans_a = int(node.attrs.get("transA", 0))
    trans_b = int(node.attrs.get("transB", 0))
    alpha = float(node.attrs.get("alpha", 1.0))

    # Resolve matrix shapes and compute output dimensions.
    input_a_shape = _value_shape(graph, node.inputs[0], node)
    input_b_shape = _value_shape(graph, node.inputs[3], node)
    if len(input_a_shape) != 2 or len(input_b_shape) != 2:
        raise UnsupportedOpError(
            f"QGemm supports 2D inputs only, got {input_a_shape} x {input_b_shape}"
        )
    if trans_a:
        m, k_left = input_a_shape[1], input_a_shape[0]
    else:
        m, k_left = input_a_shape
    if trans_b:
        n, k_right = input_b_shape[0], input_b_shape[1]
    else:
        k_right, n = input_b_shape
    if k_left != k_right:
        raise ShapeInferenceError(
            f"QGemm inner dimensions must match, got {k_left} and {k_right}"
        )
    output_shape = (m, n)

    # Validate B scale/zero shapes (scalar [1] or per-column [N]).
    b_scale_per_column = False
    if b_scale_shape in {(), (1,)}:
        pass  # scalar
    elif len(b_scale_shape) == 1 and b_scale_shape[0] == n:
        b_scale_per_column = True
    else:
        raise UnsupportedOpError(
            f"QGemm b_scale must be scalar or shape [{n}], got {b_scale_shape}"
        )
    if b_zero_shape not in {(), (1,)}:
        if not (len(b_zero_shape) == 1 and b_zero_shape[0] == n):
            raise UnsupportedOpError(
                f"QGemm b_zero_point must be scalar or shape [{n}], got {b_zero_shape}"
            )

    # Determine output dtype.
    y_scale_dtype: ScalarType | None = None
    y_scale_shape_resolved: tuple[int, ...] | None = None
    y_zero_shape_resolved: tuple[int, ...] | None = None
    c_dtype: ScalarType | None = None

    if y_scale_name is not None and y_zero_name is not None:
        y_scale_dtype = _value_dtype(graph, y_scale_name, node)
        _ensure_scale_dtype(y_scale_dtype, "y_scale")
        y_zero_dtype = _value_dtype(graph, y_zero_name, node)
        output_dtype = y_zero_dtype
        y_scale_shape_resolved = _ensure_scalar_input(
            graph, y_scale_name, node, "y_scale"
        )
        y_zero_shape_resolved = _ensure_scalar_input(
            graph, y_zero_name, node, "y_zero_point"
        )
    else:
        # No requantization: output is float.
        y_scale_name = None
        y_zero_name = None
        output_dtype = ScalarType.F32

    if input_c_name is not None:
        c_dtype = _value_dtype(graph, input_c_name, node)

    try:
        expected_output_dtype = _value_dtype(graph, node.outputs[0], node)
    except ShapeInferenceError:
        expected_output_dtype = output_dtype
    if expected_output_dtype != output_dtype:
        output_dtype = expected_output_dtype

    lowered = QGemmOp(
        input_a=node.inputs[0],
        a_scale=node.inputs[1],
        a_zero_point=node.inputs[2],
        input_b=node.inputs[3],
        b_scale=node.inputs[4],
        b_zero_point=node.inputs[5],
        input_c=input_c_name,
        y_scale=y_scale_name,
        y_zero_point=y_zero_name,
        output=node.outputs[0],
        trans_a=trans_a,
        trans_b=trans_b,
        alpha=alpha,
        input_a_dtype=input_a_dtype,
        input_b_dtype=input_b_dtype,
        dtype=output_dtype,
        a_scale_dtype=a_scale_dtype,
        b_scale_dtype=b_scale_dtype,
        y_scale_dtype=y_scale_dtype,
        c_dtype=c_dtype,
        a_scale_shape=a_scale_shape,
        a_zero_shape=a_zero_shape,
        b_scale_shape=b_scale_shape,
        b_zero_shape=b_zero_shape,
        y_scale_shape=y_scale_shape_resolved,
        y_zero_shape=y_zero_shape_resolved,
        b_scale_per_column=b_scale_per_column,
    )

    if isinstance(graph, GraphContext):
        graph.set_shape(node.outputs[0], output_shape)
        graph.set_dtype(node.outputs[0], output_dtype)

    return lowered

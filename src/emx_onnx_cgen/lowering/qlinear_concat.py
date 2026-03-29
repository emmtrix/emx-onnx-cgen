from __future__ import annotations

from shared.scalar_types import ScalarType

from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.context import GraphContext
from ..ir.model import Graph, Node
from ..ir.ops import QLinearConcatOp
from ..validation import normalize_concat_axis
from .common import value_dtype as _value_dtype
from .common import value_shape as _value_shape
from .registry import register_lowering

_SUPPORTED_DTYPES = {ScalarType.U8, ScalarType.I8}


def _ensure_scalar(graph: Graph, name: str, node: Node, label: str) -> tuple[int, ...]:
    shape = _value_shape(graph, name, node)
    if shape not in {(), (1,)}:
        raise UnsupportedOpError(
            f"QLinearConcat {label} must be scalar, got shape {shape}"
        )
    return shape


def _ensure_scale_dtype(dtype: ScalarType, label: str) -> None:
    if not (dtype.is_float or dtype.is_integer):
        raise UnsupportedOpError(f"QLinearConcat {label} must be a numeric type")


@register_lowering("QLinearConcat")
def lower_qlinear_concat(graph: Graph, node: Node) -> QLinearConcatOp:
    n_inputs = len(node.inputs)
    if n_inputs < 5 or (n_inputs - 2) % 3 != 0 or len(node.outputs) != 1:
        raise UnsupportedOpError(
            "QLinearConcat must have 2 + N*3 inputs (N >= 1) and 1 output"
        )

    n = (n_inputs - 2) // 3
    axis = int(node.attrs.get("axis", 0))

    output_scale_name = node.inputs[0]
    output_zero_name = node.inputs[1]

    input_names = tuple(node.inputs[2 + i * 3] for i in range(n))
    input_scale_names = tuple(node.inputs[2 + i * 3 + 1] for i in range(n))
    input_zero_names = tuple(node.inputs[2 + i * 3 + 2] for i in range(n))

    input_shapes = tuple(_value_shape(graph, name, node) for name in input_names)
    input_dtypes = tuple(_value_dtype(graph, name, node) for name in input_names)

    for i, dtype in enumerate(input_dtypes):
        if dtype not in _SUPPORTED_DTYPES:
            raise UnsupportedOpError(
                f"QLinearConcat supports uint8/int8 inputs only, "
                f"got {dtype} for input {i}"
            )

    # Determine output dtype; fall back to the first input dtype when unknown.
    try:
        output_dtype = _value_dtype(graph, node.outputs[0], node)
    except ShapeInferenceError:
        output_dtype = input_dtypes[0]
    if output_dtype not in _SUPPORTED_DTYPES:
        raise UnsupportedOpError(
            f"QLinearConcat supports uint8/int8 output only, got {output_dtype}"
        )

    output_scale_dtype = _value_dtype(graph, output_scale_name, node)
    _ensure_scale_dtype(output_scale_dtype, "y_scale")

    input_scale_dtypes = tuple(
        _value_dtype(graph, name, node) for name in input_scale_names
    )
    for i, dtype in enumerate(input_scale_dtypes):
        _ensure_scale_dtype(dtype, f"x_scale{i}")

    output_zero_dtype = _value_dtype(graph, output_zero_name, node)
    output_zero_dtype_match = output_zero_dtype == output_dtype

    # When the zero-point dtype doesn't match the input/output dtype, treat it as 0
    # (mirrors ORT behaviour for type-mismatched zero-points).
    input_zero_dtypes = tuple(
        _value_dtype(graph, input_zero_names[i], node) for i in range(n)
    )
    input_zero_dtype_matches = tuple(
        input_zero_dtypes[i] == input_dtypes[i] for i in range(n)
    )

    ranks = {len(shape) for shape in input_shapes}
    if len(ranks) != 1:
        raise UnsupportedOpError(
            f"QLinearConcat inputs must have matching ranks, got {input_shapes}"
        )
    rank = ranks.pop()
    axis = normalize_concat_axis(axis, rank)

    base_shape = list(input_shapes[0])
    axis_dim = 0
    for shape in input_shapes:
        for dim_index, dim in enumerate(shape):
            if dim_index == axis:
                continue
            if dim != base_shape[dim_index]:
                raise UnsupportedOpError(
                    "QLinearConcat inputs must match on non-axis dimensions, "
                    f"got {input_shapes}"
                )
        axis_dim += shape[axis]
    base_shape[axis] = axis_dim
    output_shape = tuple(base_shape)

    output_scale_shape = _ensure_scalar(graph, output_scale_name, node, "y_scale")
    output_zero_shape = _ensure_scalar(graph, output_zero_name, node, "y_zero_point")
    input_scale_shapes = tuple(
        _ensure_scalar(graph, name, node, f"x_scale{i}")
        for i, name in enumerate(input_scale_names)
    )
    input_zero_shapes = tuple(
        _ensure_scalar(graph, name, node, f"x_zero_point{i}")
        for i, name in enumerate(input_zero_names)
    )

    if isinstance(graph, GraphContext):
        graph.set_shape(node.outputs[0], output_shape)
        graph.set_dtype(node.outputs[0], output_dtype)

    return QLinearConcatOp(
        output_scale=output_scale_name,
        output_zero_point=output_zero_name,
        inputs=input_names,
        input_scales=input_scale_names,
        input_zero_points=input_zero_names,
        output=node.outputs[0],
        axis=axis,
        input_shapes=input_shapes,
        output_shape=output_shape,
        dtype=output_dtype,
        input_dtypes=input_dtypes,
        output_scale_dtype=output_scale_dtype,
        input_scale_dtypes=input_scale_dtypes,
        output_scale_shape=output_scale_shape,
        output_zero_shape=output_zero_shape,
        input_scale_shapes=input_scale_shapes,
        input_zero_shapes=input_zero_shapes,
        output_zero_dtype_match=output_zero_dtype_match,
        output_zero_dtype=output_zero_dtype,
        input_zero_dtype_matches=input_zero_dtype_matches,
        input_zero_dtypes=input_zero_dtypes,
    )

from __future__ import annotations

import numpy as np

from shared.scalar_types import ScalarType

from ..ir.ops import CumProdOp, CumSumOp
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Initializer, Node
from ..lowering.common import value_dtype, value_shape
from ..validation import ensure_output_shape_matches_input, normalize_axis
from .registry import register_lowering


_SUPPORTED_CUMULATIVE_DTYPES = {
    ScalarType.F16,
    ScalarType.F32,
    ScalarType.F64,
    ScalarType.I32,
    ScalarType.I64,
    ScalarType.U32,
    ScalarType.U64,
}


def _find_initializer(graph: Graph, name: str) -> Initializer | None:
    for initializer in graph.initializers:
        if initializer.name == name:
            return initializer
    return None


def _is_scalar_shape(shape: tuple[int, ...]) -> bool:
    return shape == () or shape == (1,)


def _validate_static_shape(shape: tuple[int, ...], node: Node) -> None:
    for dim in shape:
        if dim < 0:
            raise ShapeInferenceError(f"{node.op_type} does not support dynamic dims")


def _read_axis_initializer(initializer: Initializer, node: Node) -> int:
    if initializer.type.dtype not in {ScalarType.I64, ScalarType.I32}:
        raise UnsupportedOpError(f"{node.op_type} axis input must be int64 or int32")
    axis_data = np.array(initializer.data, dtype=np.int64).reshape(-1)
    if axis_data.size != 1:
        raise UnsupportedOpError(f"{node.op_type} axis input must be scalar")
    return int(axis_data[0])


def _lower_cumulative(
    graph: Graph, node: Node, *, op_type: str
) -> CumSumOp | CumProdOp:
    if len(node.inputs) != 2 or len(node.outputs) != 1:
        raise UnsupportedOpError(f"{op_type} must have 2 inputs and 1 output")
    input_name = node.inputs[0]
    axis_name = node.inputs[1]
    if not input_name or not axis_name:
        raise UnsupportedOpError(f"{op_type} requires input and axis values")
    input_shape = value_shape(graph, input_name, node)
    output_shape = value_shape(graph, node.outputs[0], node)
    _validate_static_shape(input_shape, node)
    ensure_output_shape_matches_input(node, input_shape, output_shape)
    input_dtype = value_dtype(graph, input_name, node)
    output_dtype = value_dtype(graph, node.outputs[0], node)
    if input_dtype != output_dtype:
        raise UnsupportedOpError(
            f"{op_type} expects matching input/output dtypes, "
            f"got {input_dtype.onnx_name} and {output_dtype.onnx_name}"
        )
    if input_dtype not in _SUPPORTED_CUMULATIVE_DTYPES:
        raise UnsupportedOpError(
            f"{op_type} does not support dtype {input_dtype.onnx_name}"
        )
    axis_initializer = _find_initializer(graph, axis_name)
    axis_value = None
    axis_input = None
    if axis_initializer is not None:
        axis_value = normalize_axis(
            _read_axis_initializer(axis_initializer, node),
            input_shape,
            node,
        )
    else:
        axis_shape = value_shape(graph, axis_name, node)
        if not _is_scalar_shape(axis_shape):
            raise UnsupportedOpError(f"{op_type} axis input must be scalar")
        axis_input_dtype = value_dtype(graph, axis_name, node)
        if axis_input_dtype not in {ScalarType.I64, ScalarType.I32}:
            raise UnsupportedOpError(f"{op_type} axis input must be int64 or int32")
        axis_input = axis_name
    exclusive = int(node.attrs.get("exclusive", 0))
    reverse = int(node.attrs.get("reverse", 0))
    if exclusive not in {0, 1}:
        raise UnsupportedOpError(f"{op_type} exclusive must be 0 or 1")
    if reverse not in {0, 1}:
        raise UnsupportedOpError(f"{op_type} reverse must be 0 or 1")
    op_cls = CumSumOp if op_type == "CumSum" else CumProdOp
    return op_cls(
        input0=input_name,
        axis_input=axis_input,
        axis=axis_value,
        output=node.outputs[0],
        exclusive=bool(exclusive),
        reverse=bool(reverse),
    )


@register_lowering("CumSum")
def lower_cumsum(graph: Graph, node: Node) -> CumSumOp:
    return _lower_cumulative(graph, node, op_type="CumSum")


@register_lowering("CumProd")
def lower_cumprod(graph: Graph, node: Node) -> CumProdOp:
    return _lower_cumulative(graph, node, op_type="CumProd")

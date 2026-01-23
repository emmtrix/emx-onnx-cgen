from __future__ import annotations

from shared.scalar_types import ScalarType

from ..ir.ops import ArgReduceOp
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from .common import shape_product, value_dtype, value_shape
from .registry import register_lowering

ARG_REDUCE_KIND_BY_OP = {"ArgMax": "max", "ArgMin": "min"}


def _normalize_axis(axis: int, rank: int, node: Node) -> int:
    if rank <= 0:
        raise ShapeInferenceError(
            f"{node.op_type} requires input rank >= 1, got {rank}"
        )
    if axis < 0:
        axis += rank
    if axis < 0 or axis >= rank:
        raise ShapeInferenceError(
            f"{node.op_type} axis {axis} is out of range for rank {rank}"
        )
    return axis


def _output_shape(
    input_shape: tuple[int, ...], axis: int, keepdims: bool
) -> tuple[int, ...]:
    if keepdims:
        return tuple(1 if idx == axis else dim for idx, dim in enumerate(input_shape))
    return tuple(dim for idx, dim in enumerate(input_shape) if idx != axis)


def _arg_reduce_dtype_supported(dtype: ScalarType) -> bool:
    return dtype in {
        ScalarType.F16,
        ScalarType.F32,
        ScalarType.F64,
        ScalarType.I64,
        ScalarType.I32,
        ScalarType.I16,
        ScalarType.I8,
        ScalarType.U64,
        ScalarType.U32,
        ScalarType.U16,
        ScalarType.U8,
    }


def lower_arg_reduce(graph: Graph, node: Node) -> ArgReduceOp:
    if node.op_type not in ARG_REDUCE_KIND_BY_OP:
        raise UnsupportedOpError(f"Unsupported op {node.op_type}")
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        raise UnsupportedOpError(
            f"{node.op_type} must have 1 input and 1 output"
        )
    input_name = node.inputs[0]
    output_name = node.outputs[0]
    input_shape = value_shape(graph, input_name, node)
    shape_product(input_shape)
    rank = len(input_shape)
    axis = int(node.attrs.get("axis", 0))
    axis = _normalize_axis(axis, rank, node)
    keepdims = bool(int(node.attrs.get("keepdims", 1)))
    select_last_index = bool(int(node.attrs.get("select_last_index", 0)))
    expected_output_shape = _output_shape(input_shape, axis, keepdims)
    output_shape = value_shape(graph, output_name, node)
    if output_shape != expected_output_shape:
        raise ShapeInferenceError(
            f"{node.op_type} output shape must be {expected_output_shape}, got {output_shape}"
        )
    input_dtype = value_dtype(graph, input_name, node)
    if not _arg_reduce_dtype_supported(input_dtype):
        raise UnsupportedOpError(
            f"{node.op_type} does not support dtype {input_dtype.onnx_name}"
        )
    output_dtype = value_dtype(graph, output_name, node)
    if output_dtype != ScalarType.I64:
        raise UnsupportedOpError(
            f"{node.op_type} expects output dtype int64, got {output_dtype.onnx_name}"
        )
    return ArgReduceOp(
        input0=input_name,
        output=output_name,
        input_shape=input_shape,
        output_shape=output_shape,
        axis=axis,
        keepdims=keepdims,
        select_last_index=select_last_index,
        reduce_kind=ARG_REDUCE_KIND_BY_OP[node.op_type],
        input_dtype=input_dtype,
        output_dtype=output_dtype,
    )


register_lowering("ArgMax")(lower_arg_reduce)
register_lowering("ArgMin")(lower_arg_reduce)

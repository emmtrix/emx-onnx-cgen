from __future__ import annotations

from shared.scalar_types import ScalarType

from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from ..ir.ops import SparseToDenseMatMulOp
from .common import node_dtype, value_dtype, value_shape
from .registry import register_lowering

_SUPPORTED_DTYPES = {
    ScalarType.F32,
    ScalarType.F64,
    ScalarType.I32,
    ScalarType.I64,
    ScalarType.U32,
    ScalarType.U64,
}


@register_lowering("SparseToDenseMatMul")
def lower_sparse_to_dense_matmul(graph: Graph, node: Node) -> SparseToDenseMatMulOp:
    # Sparse graph inputs are expanded before import (see sparse_inputs), so A
    # arrives as COO values [nnz] and indices [nnz, 2].
    if len(node.inputs) != 3 or len(node.outputs) != 1:
        raise UnsupportedOpError(
            "SparseToDenseMatMul supports A only as a sparse tensor graph input "
            f"(node {node.name or '<unnamed>'})."
        )
    values, indices, input_b = node.inputs
    output = node.outputs[0]
    dtype = node_dtype(graph, node, values, input_b, output)
    if dtype not in _SUPPORTED_DTYPES:
        raise UnsupportedOpError(
            f"SparseToDenseMatMul does not support dtype {dtype.onnx_name}"
        )
    if value_dtype(graph, indices, node) != ScalarType.I64:
        raise UnsupportedOpError("SparseToDenseMatMul expects int64 COO indices")
    indices_shape = value_shape(graph, indices, node)
    if len(indices_shape) != 2 or indices_shape[1] != 2:
        raise UnsupportedOpError(
            "SparseToDenseMatMul supports 2-D sparse matrices only, "
            f"got COO indices of shape {indices_shape}"
        )
    b_shape = value_shape(graph, input_b, node)
    output_shape = value_shape(graph, output, node)
    if len(b_shape) != 2 or len(output_shape) != 2:
        raise UnsupportedOpError(
            "SparseToDenseMatMul supports 2-D matrices only, "
            f"got B {b_shape} and output {output_shape}"
        )
    trans_a = int(node.attrs.get("transA", 0)) != 0
    trans_b = int(node.attrs.get("transB", 0)) != 0
    k, n = (b_shape[1], b_shape[0]) if trans_b else (b_shape[0], b_shape[1])
    m = output_shape[0]
    if m < 0 or k < 0 or n < 0 or output_shape[1] != n:
        raise ShapeInferenceError(
            "SparseToDenseMatMul needs static shapes with output [M, N] matching "
            f"op(B) of shape [{k}, {n}], got output {output_shape}. "
            "Hint: export the model with static shapes."
        )
    alpha = float(node.attrs.get("alpha", 1.0))
    if alpha != 1.0 and not dtype.is_float:
        raise UnsupportedOpError(
            "SparseToDenseMatMul supports alpha != 1 for float types only"
        )
    return SparseToDenseMatMulOp(
        values=values,
        indices=indices,
        input_b=input_b,
        output=output,
        m=m,
        k=k,
        n=n,
        trans_a=trans_a,
        trans_b=trans_b,
        alpha=alpha,
        dtype=dtype,
    )

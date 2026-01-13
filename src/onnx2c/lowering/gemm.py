from __future__ import annotations

from dataclasses import dataclass

from ..codegen.c_emitter import GemmOp
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from .common import node_dtype as _node_dtype
from .common import value_shape as _value_shape
from .registry import register_lowering


@dataclass(frozen=True)
class GemmSpec:
    m: int
    n: int
    k: int
    alpha: float | int
    beta: float | int
    trans_a: bool
    trans_b: bool
    c_shape: tuple[int, ...] | None


def resolve_gemm_spec(graph: Graph, node: Node, dtype: str) -> GemmSpec:
    if len(node.inputs) not in {2, 3} or len(node.outputs) != 1:
        raise UnsupportedOpError("Gemm must have 2 or 3 inputs and 1 output")
    alpha, beta, trans_a, trans_b = _resolve_gemm_attrs(node, dtype)
    input0_shape = _value_shape(graph, node.inputs[0], node)
    input1_shape = _value_shape(graph, node.inputs[1], node)
    if len(input0_shape) != 2 or len(input1_shape) != 2:
        raise UnsupportedOpError(
            "Gemm supports 2D inputs only, "
            f"got {input0_shape} x {input1_shape}"
        )
    if trans_a:
        m, k_left = input0_shape[1], input0_shape[0]
    else:
        m, k_left = input0_shape
    if trans_b:
        n, k_right = input1_shape[0], input1_shape[1]
    else:
        k_right, n = input1_shape
    if k_left != k_right:
        raise ShapeInferenceError(
            f"Gemm inner dimensions must match, got {k_left} and {k_right}"
        )
    output_shape = _value_shape(graph, node.outputs[0], node)
    if output_shape != (m, n):
        raise ShapeInferenceError(
            f"Gemm output shape must be {(m, n)}, got {output_shape}"
        )
    c_shape = None
    if len(node.inputs) == 3:
        bias_shape = _value_shape(graph, node.inputs[2], node)
        c_shape = validate_gemm_bias_shape((m, n), bias_shape, node)
    return GemmSpec(
        m=m,
        n=n,
        k=k_left,
        alpha=alpha,
        beta=beta,
        trans_a=trans_a,
        trans_b=trans_b,
        c_shape=c_shape,
    )


def _resolve_gemm_attrs(
    node: Node, dtype: str
) -> tuple[float | int, float | int, bool, bool]:
    alpha = float(node.attrs.get("alpha", 1.0))
    beta = float(node.attrs.get("beta", 1.0))
    trans_a = int(node.attrs.get("transA", 0))
    trans_b = int(node.attrs.get("transB", 0))
    if trans_a not in {0, 1} or trans_b not in {0, 1}:
        raise UnsupportedOpError(
            "Gemm only supports transA/transB values of 0 or 1"
        )
    if dtype == "bool":
        raise UnsupportedOpError("Gemm supports numeric inputs only")
    if dtype not in {"float", "double", "float16"}:
        alpha_int = int(alpha)
        beta_int = int(beta)
        if alpha != alpha_int or beta != beta_int:
            raise UnsupportedOpError(
                "Gemm alpha and beta must be integers for non-float inputs"
            )
        alpha = alpha_int
        beta = beta_int
    return alpha, beta, bool(trans_a), bool(trans_b)


def validate_gemm_bias_shape(
    output_shape: tuple[int, int], bias_shape: tuple[int, ...], node: Node
) -> tuple[int, ...]:
    if len(bias_shape) == 1:
        if bias_shape[0] != output_shape[1]:
            raise ShapeInferenceError(
                "Gemm bias input must be broadcastable to output shape, "
                f"got {bias_shape} vs {output_shape}"
            )
        return bias_shape
    if len(bias_shape) == 2:
        m, n = output_shape
        if bias_shape[0] not in {1, m} or bias_shape[1] not in {1, n}:
            raise ShapeInferenceError(
                "Gemm bias input must be broadcastable to output shape, "
                f"got {bias_shape} vs {output_shape}"
            )
        return bias_shape
    raise ShapeInferenceError(
        f"Gemm bias input must be rank 1 or 2, got {bias_shape}"
    )


@register_lowering("Gemm")
def lower_gemm(graph: Graph, node: Node) -> GemmOp:
    op_dtype = _node_dtype(graph, node, *node.inputs, *node.outputs)
    spec = resolve_gemm_spec(graph, node, op_dtype)
    return GemmOp(
        input_a=node.inputs[0],
        input_b=node.inputs[1],
        input_c=node.inputs[2] if len(node.inputs) == 3 else None,
        output=node.outputs[0],
        m=spec.m,
        n=spec.n,
        k=spec.k,
        trans_a=spec.trans_a,
        trans_b=spec.trans_b,
        alpha=spec.alpha,
        beta=spec.beta,
        c_shape=spec.c_shape,
        dtype=op_dtype,
    )

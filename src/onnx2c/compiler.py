from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import onnx

from .codegen.c_emitter import (
    AttentionOp,
    BinaryOp,
    CEmitter,
    ConstTensor,
    LoweredModel,
    MatMulOp,
    UnaryOp,
)
from .dtypes import dtype_info
from .errors import CodegenError, ShapeInferenceError, UnsupportedOpError
from .ir.model import Graph, Node
from .onnx_import import import_onnx


@dataclass(frozen=True)
class CompilerOptions:
    template_dir: Path
    model_name: str = "model"
    emit_testbench: bool = False


class Compiler:
    def __init__(self, options: CompilerOptions | None = None) -> None:
        if options is None:
            options = CompilerOptions(template_dir=Path("templates"))
        self._options = options
        self._emitter = CEmitter(options.template_dir)

    def compile(self, model: onnx.ModelProto) -> str:
        graph = import_onnx(model)
        lowered = self._lower_model(graph)
        return self._emitter.emit_model(
            lowered, emit_testbench=self._options.emit_testbench
        )

    def _lower_model(self, graph: Graph) -> LoweredModel:
        dtype = _model_dtype(graph)
        if len(graph.nodes) != 1:
            if len(graph.nodes) == 2:
                return self._lower_binary_chain_model(graph, dtype)
            raise UnsupportedOpError(
                f"Only one- or two-node graphs are supported, got {len(graph.nodes)}"
            )
        node = graph.nodes[0]
        if node.op_type in {"MatMul", "Gemm"}:
            if node.op_type == "Gemm":
                _validate_gemm(node)
            return self._lower_matmul(graph, node, dtype)
        if node.op_type == "Attention":
            return self._lower_attention(graph, node, dtype)
        op_spec = _binary_op_symbol(node.op_type, node.attrs, dtype=dtype)
        unary_symbol = _unary_op_symbol(node.op_type, dtype=dtype)
        if op_spec is None and unary_symbol is None:
            raise UnsupportedOpError(f"Unsupported op {node.op_type}")
        if op_spec is not None:
            if len(node.inputs) != 2 or len(node.outputs) != 1:
                raise UnsupportedOpError(
                    f"{node.op_type} must have 2 inputs and 1 output"
                )
        else:
            if len(node.inputs) != 1 or len(node.outputs) != 1:
                raise UnsupportedOpError(
                    f"{node.op_type} must have 1 input and 1 output"
                )
        output_value = graph.outputs[0]
        element_count = _element_count(output_value.type.shape)
        if element_count <= 0:
            raise ShapeInferenceError("Output shape must be fully defined")
        constants = _lowered_constants(graph, dtype)
        input_names = tuple(value.name for value in graph.inputs)
        input_shapes = tuple(value.type.shape for value in graph.inputs)
        if unary_symbol is not None:
            return LoweredModel(
                name=self._options.model_name,
                dtype=dtype,
                input_names=input_names,
                input_shapes=input_shapes,
                output_name=node.outputs[0],
                element_count=element_count,
                output_shape=output_value.type.shape,
                constants=constants,
                ops=(
                    UnaryOp(
                        input0=node.inputs[0],
                        output=node.outputs[0],
                        operator=unary_symbol,
                    ),
                ),
            )
        return LoweredModel(
            name=self._options.model_name,
            dtype=dtype,
            input_names=input_names,
            input_shapes=input_shapes,
            output_name=node.outputs[0],
            element_count=element_count,
            output_shape=output_value.type.shape,
            constants=constants,
            ops=(
                BinaryOp(
                    input0=node.inputs[0],
                    input1=node.inputs[1],
                    output=node.outputs[0],
                    operator=op_spec.operator,
                    operator_kind=op_spec.kind,
                ),
            ),
        )

    def _lower_binary_chain_model(self, graph: Graph, dtype: str) -> LoweredModel:
        node1, node2 = graph.nodes
        op1_spec = _binary_op_symbol(node1.op_type, node1.attrs, dtype=dtype)
        if op1_spec is None:
            raise UnsupportedOpError(f"Unsupported op {node1.op_type}")
        op2_spec = _binary_op_symbol(node2.op_type, node2.attrs, dtype=dtype)
        if op2_spec is None:
            raise UnsupportedOpError(f"Unsupported op {node2.op_type}")
        if len(node1.inputs) != 2 or len(node1.outputs) != 1:
            raise UnsupportedOpError(
                f"{node1.op_type} must have 2 inputs and 1 output"
            )
        if len(node2.inputs) != 2 or len(node2.outputs) != 1:
            raise UnsupportedOpError(
                f"{node2.op_type} must have 2 inputs and 1 output"
            )
        intermediate = node1.outputs[0]
        if intermediate not in node2.inputs:
            raise UnsupportedOpError(
                "Second node must consume the first node output"
            )
        output_value = graph.outputs[0]
        element_count = _element_count(output_value.type.shape)
        if element_count <= 0:
            raise ShapeInferenceError("Output shape must be fully defined")
        input_names = tuple(value.name for value in graph.inputs)
        input_shapes = tuple(value.type.shape for value in graph.inputs)
        return LoweredModel(
            name=self._options.model_name,
            dtype=dtype,
            input_names=input_names,
            input_shapes=input_shapes,
            output_name=node2.outputs[0],
            element_count=element_count,
            output_shape=output_value.type.shape,
            constants=_lowered_constants(graph, dtype),
            ops=(
                BinaryOp(
                    input0=node1.inputs[0],
                    input1=node1.inputs[1],
                    output=intermediate,
                    operator=op1_spec.operator,
                    operator_kind=op1_spec.kind,
                ),
                BinaryOp(
                    input0=node2.inputs[0],
                    input1=node2.inputs[1],
                    output=node2.outputs[0],
                    operator=op2_spec.operator,
                    operator_kind=op2_spec.kind,
                ),
            ),
        )

    def run(
        self, model: onnx.ModelProto, feeds: Mapping[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        graph = import_onnx(model)
        dtype = _model_dtype(graph)
        constants = {
            initializer.name: initializer.data for initializer in graph.initializers
        }
        resolved_feeds = _ResolvedFeeds(feeds=feeds, constants=constants)
        if len(graph.nodes) != 1:
            if len(graph.nodes) == 2:
                return self._run_binary_chain(graph, resolved_feeds, dtype)
            raise UnsupportedOpError("Only one- or two-node graphs are supported")
        node = graph.nodes[0]
        if node.op_type in {"MatMul", "Gemm"}:
            if node.op_type == "Gemm":
                _validate_gemm(node)
            left = resolved_feeds.fetch(node.inputs[0])
            right = resolved_feeds.fetch(node.inputs[1])
            result = _apply_matmul(left, right)
            return {node.outputs[0]: result}
        if node.op_type == "Attention":
            spec = _resolve_attention_spec(graph, node, dtype)
            query = resolved_feeds.fetch(node.inputs[0])
            key = resolved_feeds.fetch(node.inputs[1])
            value = resolved_feeds.fetch(node.inputs[2])
            result = _apply_attention(spec, query, key, value)
            return {node.outputs[0]: result}
        op_spec = _binary_op_symbol(node.op_type, node.attrs, dtype=dtype)
        unary_symbol = _unary_op_symbol(node.op_type, dtype=dtype)
        if op_spec is None and unary_symbol is None:
            raise UnsupportedOpError(f"Unsupported op {node.op_type}")
        if op_spec is not None:
            left = resolved_feeds.fetch(node.inputs[0])
            right = resolved_feeds.fetch(node.inputs[1])
            result = _apply_binary_op(op_spec, left, right)
        else:
            value = resolved_feeds.fetch(node.inputs[0])
            result = _apply_unary_op(unary_symbol, value)
        return {node.outputs[0]: result}

    def _run_binary_chain(
        self, graph: Graph, feeds: _ResolvedFeeds, dtype: str
    ) -> dict[str, np.ndarray]:
        node1, node2 = graph.nodes
        op1_spec = _binary_op_symbol(node1.op_type, node1.attrs, dtype=dtype)
        if op1_spec is None:
            raise UnsupportedOpError(f"Unsupported op {node1.op_type}")
        op2_spec = _binary_op_symbol(node2.op_type, node2.attrs, dtype=dtype)
        if op2_spec is None:
            raise UnsupportedOpError(f"Unsupported op {node2.op_type}")
        intermediate = node1.outputs[0]
        if intermediate not in node2.inputs:
            raise UnsupportedOpError(
                "Second node must consume the first node output"
            )
        left = feeds.fetch(node1.inputs[0])
        right = feeds.fetch(node1.inputs[1])
        tmp = _apply_binary_op(op1_spec, left, right)
        if node2.inputs[0] == intermediate:
            left = tmp
            right = feeds.fetch(node2.inputs[1])
        else:
            left = feeds.fetch(node2.inputs[0])
            right = tmp
        result = _apply_binary_op(op2_spec, left, right)
        return {node2.outputs[0]: result}

    def _lower_matmul(
        self, graph: Graph, node: Node | None, dtype: str
    ) -> LoweredModel:
        if node is None:
            raise UnsupportedOpError("MatMul node is missing")
        if len(node.inputs) != 2 or len(node.outputs) != 1:
            raise UnsupportedOpError("MatMul must have 2 inputs and 1 output")
        output_value = graph.outputs[0]
        input0_shape = graph.find_value(node.inputs[0]).type.shape
        input1_shape = graph.find_value(node.inputs[1]).type.shape
        if len(input0_shape) != 2 or len(input1_shape) != 2:
            raise UnsupportedOpError(
                f"MatMul supports 2D inputs only, got {input0_shape} x {input1_shape}"
            )
        m, k_left = input0_shape
        k_right, n = input1_shape
        if k_left != k_right:
            raise ShapeInferenceError(
                f"MatMul inner dimensions must match, got {k_left} and {k_right}"
            )
        if output_value.type.shape != (m, n):
            raise ShapeInferenceError(
                f"MatMul output shape must be {(m, n)}, got {output_value.type.shape}"
            )
        element_count = _element_count(output_value.type.shape)
        return LoweredModel(
            name=self._options.model_name,
            dtype=dtype,
            input_names=tuple(value.name for value in graph.inputs),
            input_shapes=tuple(value.type.shape for value in graph.inputs),
            output_name=node.outputs[0],
            element_count=element_count,
            output_shape=output_value.type.shape,
            constants=_lowered_constants(graph, dtype),
            ops=(
                MatMulOp(
                    input0=node.inputs[0],
                    input1=node.inputs[1],
                    output=node.outputs[0],
                    m=m,
                    n=n,
                    k=k_left,
                ),
            ),
        )

    def _lower_attention(
        self, graph: Graph, node: Node | None, dtype: str
    ) -> LoweredModel:
        if node is None:
            raise UnsupportedOpError("Attention node is missing")
        spec = _resolve_attention_spec(graph, node, dtype)
        output_value = graph.outputs[0]
        element_count = _element_count(output_value.type.shape)
        return LoweredModel(
            name=self._options.model_name,
            dtype=dtype,
            input_names=tuple(value.name for value in graph.inputs),
            input_shapes=tuple(value.type.shape for value in graph.inputs),
            output_name=node.outputs[0],
            element_count=element_count,
            output_shape=output_value.type.shape,
            constants=_lowered_constants(graph, dtype),
            ops=(
                AttentionOp(
                    input_q=node.inputs[0],
                    input_k=node.inputs[1],
                    input_v=node.inputs[2],
                    output=node.outputs[0],
                    batch=spec.batch,
                    heads=spec.heads,
                    q_seq=spec.q_seq,
                    kv_seq=spec.kv_seq,
                    qk_head_size=spec.qk_head_size,
                    v_head_size=spec.v_head_size,
                    scale=spec.scale,
                    is_causal=spec.is_causal,
                ),
            ),
        )


@dataclass(frozen=True)
class _ResolvedFeeds:
    feeds: Mapping[str, np.ndarray]
    constants: Mapping[str, np.ndarray]

    def fetch(self, name: str) -> np.ndarray:
        if name in self.constants:
            return self.constants[name]
        return self.feeds[name]


@dataclass(frozen=True)
class _BinaryOpSpec:
    operator: str
    kind: str
    apply: Callable[[np.ndarray, np.ndarray], np.ndarray]


def _lowered_constants(graph: Graph, dtype: str) -> tuple[ConstTensor, ...]:
    info = dtype_info(dtype)
    return tuple(
        ConstTensor(
            name=initializer.name,
            shape=initializer.type.shape,
            data=tuple(
                info.np_dtype.type(value) for value in initializer.data.ravel()
            ),
        )
        for initializer in graph.initializers
    )


def _model_dtype(graph: Graph) -> str:
    dtypes = {value.type.dtype for value in graph.inputs + graph.outputs}
    dtypes.update(initializer.type.dtype for initializer in graph.initializers)
    if len(dtypes) != 1:
        raise UnsupportedOpError(
            f"Mixed dtypes are not supported, got {', '.join(sorted(dtypes))}"
        )
    dtype = next(iter(dtypes))
    if dtype not in {"float", "int64", "int32", "int16", "int8"}:
        raise UnsupportedOpError(f"Unsupported dtype {dtype}")
    return dtype


def _element_count(shape: tuple[int, ...]) -> int:
    if not shape:
        raise ShapeInferenceError("Scalar outputs are not supported")
    count = 1
    for dim in shape:
        if dim <= 0:
            raise ShapeInferenceError("Dynamic or zero dims are not supported")
        count *= dim
    return count


def _binary_op_symbol(
    op_type: str, attrs: Mapping[str, object] | None = None, *, dtype: str
) -> _BinaryOpSpec | None:
    if dtype in {"int64", "int32", "int16", "int8"}:
        if op_type in {"Add", "Sum"}:
            return _BinaryOpSpec("+", "infix", lambda left, right: left + right)
        if op_type == "Sub":
            return _BinaryOpSpec("-", "infix", lambda left, right: left - right)
        if op_type == "Mul":
            return _BinaryOpSpec("*", "infix", lambda left, right: left * right)
        return None
    if op_type == "Add":
        return _BinaryOpSpec("+", "infix", lambda left, right: left + right)
    if op_type == "Div":
        return _BinaryOpSpec("/", "infix", lambda left, right: left / right)
    if op_type == "Max":
        return _BinaryOpSpec("fmaxf", "func", np.maximum)
    if op_type == "Mean":
        return _BinaryOpSpec(
            "({left} + {right}) * 0.5f",
            "expr",
            lambda left, right: (left + right) * 0.5,
        )
    if op_type == "Min":
        return _BinaryOpSpec("fminf", "func", np.minimum)
    if op_type == "Mod":
        fmod = 0
        if attrs is not None:
            fmod = int(attrs.get("fmod", 0))
        if fmod != 1:
            raise UnsupportedOpError(
                "Mod only supports fmod=1 for floating point types"
            )
        return _BinaryOpSpec("fmodf", "func", np.fmod)
    if op_type == "Mul":
        return _BinaryOpSpec("*", "infix", lambda left, right: left * right)
    if op_type == "Pow":
        return _BinaryOpSpec("powf", "func", np.power)
    if op_type == "PRelu":
        return _BinaryOpSpec(
            "({left} > 0.0f ? {left} : {right} * {left})",
            "expr",
            lambda left, right: np.where(left > 0.0, left, right * left),
        )
    if op_type == "Sub":
        return _BinaryOpSpec("-", "infix", lambda left, right: left - right)
    if op_type == "Sum":
        return _BinaryOpSpec("+", "infix", lambda left, right: left + right)
    return None


def _validate_gemm(node: Node) -> None:
    if len(node.inputs) != 2 or len(node.outputs) != 1:
        raise UnsupportedOpError("Gemm must have 2 inputs and 1 output")
    alpha = float(node.attrs.get("alpha", 1.0))
    beta = float(node.attrs.get("beta", 1.0))
    trans_a = int(node.attrs.get("transA", 0))
    trans_b = int(node.attrs.get("transB", 0))
    if alpha != 1.0 or beta != 1.0 or trans_a != 0 or trans_b != 0:
        raise UnsupportedOpError(
            "Gemm only supports alpha=1, beta=1, transA=0, transB=0"
        )


def _unary_op_symbol(op_type: str, *, dtype: str) -> str | None:
    if dtype in {"int64", "int32", "int16", "int8"}:
        if op_type == "Abs":
            return "llabs" if dtype == "int64" else "abs"
        if op_type == "Neg":
            return "neg"
        return None
    if op_type == "Abs":
        return "fabsf"
    if op_type == "Ceil":
        return "ceilf"
    if op_type == "Cos":
        return "cosf"
    if op_type == "Exp":
        return "expf"
    if op_type == "Floor":
        return "floorf"
    if op_type == "Log":
        return "logf"
    if op_type == "Neg":
        return "neg"
    if op_type == "Relu":
        return "relu"
    if op_type == "Sin":
        return "sinf"
    if op_type == "Sqrt":
        return "sqrtf"
    if op_type == "Tan":
        return "tanf"
    if op_type == "Tanh":
        return "tanhf"
    if op_type == "Atanh":
        return "atanhf"
    return None


def _apply_binary_op(
    op_spec: _BinaryOpSpec, left: np.ndarray, right: np.ndarray
) -> np.ndarray:
    return op_spec.apply(left, right)


def _apply_unary_op(op_symbol: str, value: np.ndarray) -> np.ndarray:
    if op_symbol == "fabsf":
        return np.abs(value)
    if op_symbol == "abs":
        return np.abs(value)
    if op_symbol == "llabs":
        return np.abs(value)
    if op_symbol == "ceilf":
        return np.ceil(value)
    if op_symbol == "cosf":
        return np.cos(value)
    if op_symbol == "expf":
        return np.exp(value)
    if op_symbol == "floorf":
        return np.floor(value)
    if op_symbol == "logf":
        return np.log(value)
    if op_symbol == "neg":
        return -value
    if op_symbol == "relu":
        return np.maximum(value, 0)
    if op_symbol == "sinf":
        return np.sin(value)
    if op_symbol == "sqrtf":
        return np.sqrt(value)
    if op_symbol == "tanf":
        return np.tan(value)
    if op_symbol == "tanhf":
        return np.tanh(value)
    if op_symbol == "atanhf":
        return np.arctanh(value)
    raise UnsupportedOpError(f"Unsupported unary op {op_symbol}")


def _apply_matmul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.ndim != 2 or right.ndim != 2:
        raise UnsupportedOpError(
            f"MatMul supports 2D inputs only, got {left.shape} x {right.shape}"
        )
    if left.shape[1] != right.shape[0]:
        raise ShapeInferenceError(
            f"MatMul inner dimensions must match, got {left.shape[1]} and {right.shape[0]}"
        )
    return np.matmul(left, right)


@dataclass(frozen=True)
class _AttentionSpec:
    batch: int
    heads: int
    q_seq: int
    kv_seq: int
    qk_head_size: int
    v_head_size: int
    scale: float
    is_causal: bool


def _resolve_attention_spec(
    graph: Graph, node: Node, dtype: str
) -> _AttentionSpec:
    if dtype != "float":
        raise UnsupportedOpError("Unsupported op Attention")
    if len(node.inputs) != 3 or len(node.outputs) != 1:
        raise UnsupportedOpError("Unsupported op Attention")
    if set(node.attrs) - {"scale", "is_causal"}:
        raise UnsupportedOpError("Unsupported op Attention")
    q_shape = graph.find_value(node.inputs[0]).type.shape
    k_shape = graph.find_value(node.inputs[1]).type.shape
    v_shape = graph.find_value(node.inputs[2]).type.shape
    if len(q_shape) != 4 or len(k_shape) != 4 or len(v_shape) != 4:
        raise UnsupportedOpError("Unsupported op Attention")
    batch, q_heads, q_seq, qk_head_size = q_shape
    batch_k, kv_heads, kv_seq, k_head_size = k_shape
    batch_v, v_heads, kv_seq_v, v_head_size = v_shape
    if batch != batch_k or batch != batch_v:
        raise ShapeInferenceError("Attention batch sizes must match")
    if kv_seq != kv_seq_v:
        raise ShapeInferenceError("Attention key/value sequence lengths must match")
    if qk_head_size != k_head_size:
        raise ShapeInferenceError("Attention Q/K head sizes must match")
    if q_heads != kv_heads or kv_heads != v_heads:
        raise UnsupportedOpError("Unsupported op Attention")
    output_value = graph.outputs[0]
    expected_output_shape = (batch, q_heads, q_seq, v_head_size)
    if output_value.type.shape != expected_output_shape:
        raise ShapeInferenceError(
            "Attention output shape must be "
            f"{expected_output_shape}, got {output_value.type.shape}"
        )
    scale = float(node.attrs.get("scale", 1.0 / math.sqrt(qk_head_size)))
    is_causal = int(node.attrs.get("is_causal", 0))
    if is_causal not in (0, 1):
        raise UnsupportedOpError("Unsupported op Attention")
    return _AttentionSpec(
        batch=batch,
        heads=q_heads,
        q_seq=q_seq,
        kv_seq=kv_seq,
        qk_head_size=qk_head_size,
        v_head_size=v_head_size,
        scale=scale,
        is_causal=bool(is_causal),
    )


def _apply_attention(
    spec: _AttentionSpec,
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
) -> np.ndarray:
    k_transpose = np.transpose(key, (0, 1, 3, 2))
    scores = np.matmul(query, k_transpose) * spec.scale
    if spec.is_causal:
        mask = np.triu(np.ones((spec.q_seq, spec.kv_seq), dtype=bool), k=1)
        scores = scores.copy()
        scores[..., mask] = -np.inf
    max_scores = np.max(scores, axis=-1, keepdims=True)
    weights = np.exp(scores - max_scores)
    weights /= np.sum(weights, axis=-1, keepdims=True)
    return np.matmul(weights, value)

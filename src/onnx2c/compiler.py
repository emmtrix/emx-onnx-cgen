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
        if len(graph.outputs) != 1:
            raise UnsupportedOpError("Only single-output graphs are supported")
        if not graph.nodes:
            raise UnsupportedOpError("Graph must contain at least one node")
        output_value = graph.outputs[0]
        output_dtype = _value_dtype(graph, output_value.name)
        element_count = _element_count(output_value.type.shape)
        if element_count <= 0:
            raise ShapeInferenceError("Output shape must be fully defined")
        constants = _lowered_constants(graph)
        input_names = tuple(value.name for value in graph.inputs)
        input_shapes = tuple(value.type.shape for value in graph.inputs)
        input_dtypes = tuple(
            _value_dtype(graph, value.name) for value in graph.inputs
        )
        ops: list[BinaryOp | UnaryOp | MatMulOp | AttentionOp] = []
        for node in graph.nodes:
            if node.op_type in {"MatMul", "Gemm"}:
                if node.op_type == "Gemm":
                    _validate_gemm(node)
                ops.append(self._lower_matmul_op(graph, node))
                continue
            if node.op_type == "Attention":
                ops.append(self._lower_attention_op(graph, node))
                continue
            op_dtype = _node_dtype(graph, node, *node.inputs, *node.outputs)
            op_spec = _binary_op_symbol(
                node.op_type, node.attrs, dtype=op_dtype
            )
            unary_symbol = _unary_op_symbol(node.op_type, dtype=op_dtype)
            if op_spec is None and unary_symbol is None:
                raise UnsupportedOpError(f"Unsupported op {node.op_type}")
            if op_spec is not None:
                if len(node.inputs) != 2 or len(node.outputs) != 1:
                    raise UnsupportedOpError(
                        f"{node.op_type} must have 2 inputs and 1 output"
                    )
                output_shape = _value_shape(graph, node.outputs[0], node)
                ops.append(
                    BinaryOp(
                        input0=node.inputs[0],
                        input1=node.inputs[1],
                        output=node.outputs[0],
                        operator=op_spec.operator,
                        operator_kind=op_spec.kind,
                        shape=output_shape,
                        dtype=op_dtype,
                    )
                )
                continue
            if len(node.inputs) != 1 or len(node.outputs) != 1:
                raise UnsupportedOpError(
                    f"{node.op_type} must have 1 input and 1 output"
                )
            output_shape = _value_shape(graph, node.outputs[0], node)
            ops.append(
                UnaryOp(
                    input0=node.inputs[0],
                    output=node.outputs[0],
                    operator=unary_symbol,
                    shape=output_shape,
                    dtype=op_dtype,
                )
            )
        return LoweredModel(
            name=self._options.model_name,
            input_names=input_names,
            input_shapes=input_shapes,
            input_dtypes=input_dtypes,
            output_name=output_value.name,
            output_dtype=output_dtype,
            element_count=element_count,
            output_shape=output_value.type.shape,
            constants=constants,
            ops=tuple(ops),
        )

    def run(
        self, model: onnx.ModelProto, feeds: Mapping[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        graph = import_onnx(model)
        constants = {
            initializer.name: initializer.data for initializer in graph.initializers
        }
        values: dict[str, np.ndarray] = dict(constants)
        values.update(feeds)
        for node in graph.nodes:
            if node.op_type in {"MatMul", "Gemm"}:
                if node.op_type == "Gemm":
                    _validate_gemm(node)
                left = values[node.inputs[0]]
                right = values[node.inputs[1]]
                values[node.outputs[0]] = _apply_matmul(left, right)
                continue
            if node.op_type == "Attention":
                op_dtype = _node_dtype(graph, node, *node.inputs, *node.outputs)
                spec = _resolve_attention_spec(graph, node, op_dtype)
                query = values[node.inputs[0]]
                key = values[node.inputs[1]]
                value = values[node.inputs[2]]
                values[node.outputs[0]] = _apply_attention(spec, query, key, value)
                continue
            op_dtype = _node_dtype(graph, node, *node.inputs, *node.outputs)
            op_spec = _binary_op_symbol(node.op_type, node.attrs, dtype=op_dtype)
            unary_symbol = _unary_op_symbol(node.op_type, dtype=op_dtype)
            if op_spec is None and unary_symbol is None:
                raise UnsupportedOpError(f"Unsupported op {node.op_type}")
            if op_spec is not None:
                left = values[node.inputs[0]]
                right = values[node.inputs[1]]
                values[node.outputs[0]] = _apply_binary_op(op_spec, left, right)
                continue
            value = values[node.inputs[0]]
            values[node.outputs[0]] = _apply_unary_op(unary_symbol, value)
        return {output.name: values[output.name] for output in graph.outputs}

    def _lower_matmul_op(self, graph: Graph, node: Node) -> MatMulOp:
        if len(node.inputs) != 2 or len(node.outputs) != 1:
            raise UnsupportedOpError("MatMul must have 2 inputs and 1 output")
        op_dtype = _node_dtype(graph, node, *node.inputs, *node.outputs)
        input0_shape = _value_shape(graph, node.inputs[0], node)
        input1_shape = _value_shape(graph, node.inputs[1], node)
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
        output_shape = _value_shape(graph, node.outputs[0], node)
        if output_shape != (m, n):
            raise ShapeInferenceError(
                f"MatMul output shape must be {(m, n)}, got {output_shape}"
            )
        return MatMulOp(
            input0=node.inputs[0],
            input1=node.inputs[1],
            output=node.outputs[0],
            m=m,
            n=n,
            k=k_left,
            dtype=op_dtype,
        )

    def _lower_attention_op(self, graph: Graph, node: Node) -> AttentionOp:
        op_dtype = _node_dtype(graph, node, *node.inputs, *node.outputs)
        spec = _resolve_attention_spec(graph, node, op_dtype)
        return AttentionOp(
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
            dtype=op_dtype,
        )


@dataclass(frozen=True)
class _BinaryOpSpec:
    operator: str
    kind: str
    apply: Callable[[np.ndarray, np.ndarray], np.ndarray]


def _lowered_constants(graph: Graph) -> tuple[ConstTensor, ...]:
    constants: list[ConstTensor] = []
    for initializer in graph.initializers:
        dtype = _ensure_supported_dtype(initializer.type.dtype)
        info = dtype_info(dtype)
        constants.append(
            ConstTensor(
                name=initializer.name,
                shape=initializer.type.shape,
                data=tuple(
                    info.np_dtype.type(value)
                    for value in initializer.data.ravel()
                ),
                dtype=dtype,
            )
        )
    return tuple(constants)


def _ensure_supported_dtype(dtype: str) -> str:
    if dtype not in {"float", "bool", "int64", "int32", "int16", "int8"}:
        raise UnsupportedOpError(f"Unsupported dtype {dtype}")
    return dtype


def _value_dtype(graph: Graph, name: str, node: Node | None = None) -> str:
    try:
        value = graph.find_value(name)
    except KeyError as exc:
        op_type = node.op_type if node is not None else "unknown"
        raise ShapeInferenceError(
            f"Missing dtype for value '{name}' in op {op_type}. "
            "Hint: run ONNX shape inference or export with static shapes."
        ) from exc
    return _ensure_supported_dtype(value.type.dtype)


def _node_dtype(graph: Graph, node: Node, *names: str) -> str:
    dtypes = {_value_dtype(graph, name, node) for name in names}
    if len(dtypes) != 1:
        raise UnsupportedOpError(
            f"{node.op_type} expects matching dtypes, got {', '.join(sorted(dtypes))}"
        )
    return next(iter(dtypes))


def _element_count(shape: tuple[int, ...]) -> int:
    if not shape:
        raise ShapeInferenceError("Scalar outputs are not supported")
    count = 1
    for dim in shape:
        if dim <= 0:
            raise ShapeInferenceError("Dynamic or zero dims are not supported")
        count *= dim
    return count


def _value_shape(graph: Graph, name: str, node: Node | None = None) -> tuple[int, ...]:
    try:
        return graph.find_value(name).type.shape
    except KeyError as exc:
        op_type = node.op_type if node is not None else "unknown"
        raise ShapeInferenceError(
            f"Missing shape for value '{name}' in op {op_type}. "
            "Hint: run ONNX shape inference or export with static shapes."
        ) from exc


def _binary_op_symbol(
    op_type: str, attrs: Mapping[str, object] | None = None, *, dtype: str
) -> _BinaryOpSpec | None:
    if dtype == "bool":
        if op_type == "And":
            return _BinaryOpSpec(
                "&&", "infix", lambda left, right: np.logical_and(left, right)
            )
        if op_type == "Or":
            return _BinaryOpSpec(
                "||", "infix", lambda left, right: np.logical_or(left, right)
            )
        if op_type == "Xor":
            return _BinaryOpSpec(
                "!=", "infix", lambda left, right: np.logical_xor(left, right)
            )
        return None
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
    if dtype == "bool":
        if op_type == "Not":
            return "!"
        return None
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
    if op_symbol == "!":
        return np.logical_not(value)
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
    q_shape = _value_shape(graph, node.inputs[0], node)
    k_shape = _value_shape(graph, node.inputs[1], node)
    v_shape = _value_shape(graph, node.inputs[2], node)
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
    output_shape = _value_shape(graph, node.outputs[0], node)
    expected_output_shape = (batch, q_heads, q_seq, v_head_size)
    if output_shape != expected_output_shape:
        raise ShapeInferenceError(
            "Attention output shape must be "
            f"{expected_output_shape}, got {output_shape}"
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

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import onnx

from .codegen.c_emitter import (
    BinaryOp,
    CEmitter,
    ConstTensor,
    LoweredModel,
    MatMulOp,
    UnaryOp,
)
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
        if len(graph.nodes) != 1:
            if len(graph.nodes) == 2:
                return self._lower_binary_chain_model(graph)
            raise UnsupportedOpError(
                f"Only one- or two-node graphs are supported, got {len(graph.nodes)}"
            )
        node = graph.nodes[0]
        if node.op_type == "MatMul":
            return self._lower_matmul(graph, node)
        op_symbol = _binary_op_symbol(node.op_type)
        unary_symbol = _unary_op_symbol(node.op_type)
        if op_symbol is None and unary_symbol is None:
            raise UnsupportedOpError(f"Unsupported op {node.op_type}")
        if op_symbol is not None:
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
        if output_value.type.dtype != "float":
            raise UnsupportedOpError(
                f"Unsupported dtype {output_value.type.dtype} for {output_value.name}"
            )
        element_count = _element_count(output_value.type.shape)
        if element_count <= 0:
            raise ShapeInferenceError("Output shape must be fully defined")
        constants = _lowered_constants(graph)
        input_names = tuple(value.name for value in graph.inputs)
        input_shapes = tuple(value.type.shape for value in graph.inputs)
        if unary_symbol is not None:
            return LoweredModel(
                name=self._options.model_name,
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
                    operator=op_symbol,
                ),
            ),
        )

    def _lower_binary_chain_model(self, graph: Graph) -> LoweredModel:
        node1, node2 = graph.nodes
        op1_symbol = _binary_op_symbol(node1.op_type)
        if op1_symbol is None:
            raise UnsupportedOpError(f"Unsupported op {node1.op_type}")
        op2_symbol = _binary_op_symbol(node2.op_type)
        if op2_symbol is None:
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
        if output_value.type.dtype != "float":
            raise UnsupportedOpError(
                f"Unsupported dtype {output_value.type.dtype} for {output_value.name}"
            )
        element_count = _element_count(output_value.type.shape)
        if element_count <= 0:
            raise ShapeInferenceError("Output shape must be fully defined")
        input_names = tuple(value.name for value in graph.inputs)
        input_shapes = tuple(value.type.shape for value in graph.inputs)
        return LoweredModel(
            name=self._options.model_name,
            input_names=input_names,
            input_shapes=input_shapes,
            output_name=node2.outputs[0],
            element_count=element_count,
            output_shape=output_value.type.shape,
            constants=_lowered_constants(graph),
            ops=(
                BinaryOp(
                    input0=node1.inputs[0],
                    input1=node1.inputs[1],
                    output=intermediate,
                    operator=op1_symbol,
                ),
                BinaryOp(
                    input0=node2.inputs[0],
                    input1=node2.inputs[1],
                    output=node2.outputs[0],
                    operator=op2_symbol,
                ),
            ),
        )

    def run(
        self, model: onnx.ModelProto, feeds: Mapping[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        graph = import_onnx(model)
        constants = {
            initializer.name: initializer.data for initializer in graph.initializers
        }
        resolved_feeds = _ResolvedFeeds(feeds=feeds, constants=constants)
        if len(graph.nodes) != 1:
            if len(graph.nodes) == 2:
                return self._run_binary_chain(graph, resolved_feeds)
            raise UnsupportedOpError("Only one- or two-node graphs are supported")
        node = graph.nodes[0]
        if node.op_type == "MatMul":
            left = resolved_feeds.fetch(node.inputs[0])
            right = resolved_feeds.fetch(node.inputs[1])
            result = _apply_matmul(left, right)
            return {node.outputs[0]: result}
        op_symbol = _binary_op_symbol(node.op_type)
        unary_symbol = _unary_op_symbol(node.op_type)
        if op_symbol is None and unary_symbol is None:
            raise UnsupportedOpError(f"Unsupported op {node.op_type}")
        if op_symbol is not None:
            left = resolved_feeds.fetch(node.inputs[0])
            right = resolved_feeds.fetch(node.inputs[1])
            result = _apply_binary_op(op_symbol, left, right)
        else:
            value = resolved_feeds.fetch(node.inputs[0])
            result = _apply_unary_op(unary_symbol, value)
        return {node.outputs[0]: result}

    def _run_binary_chain(
        self, graph: Graph, feeds: _ResolvedFeeds
    ) -> dict[str, np.ndarray]:
        node1, node2 = graph.nodes
        op1_symbol = _binary_op_symbol(node1.op_type)
        if op1_symbol is None:
            raise UnsupportedOpError(f"Unsupported op {node1.op_type}")
        op2_symbol = _binary_op_symbol(node2.op_type)
        if op2_symbol is None:
            raise UnsupportedOpError(f"Unsupported op {node2.op_type}")
        intermediate = node1.outputs[0]
        if intermediate not in node2.inputs:
            raise UnsupportedOpError(
                "Second node must consume the first node output"
            )
        left = feeds.fetch(node1.inputs[0])
        right = feeds.fetch(node1.inputs[1])
        tmp = _apply_binary_op(op1_symbol, left, right)
        if node2.inputs[0] == intermediate:
            left = tmp
            right = feeds.fetch(node2.inputs[1])
        else:
            left = feeds.fetch(node2.inputs[0])
            right = tmp
        result = _apply_binary_op(op2_symbol, left, right)
        return {node2.outputs[0]: result}

    def _lower_matmul(self, graph: Graph, node: Node | None = None) -> LoweredModel:
        if node is None:
            raise UnsupportedOpError("MatMul node is missing")
        if len(node.inputs) != 2 or len(node.outputs) != 1:
            raise UnsupportedOpError("MatMul must have 2 inputs and 1 output")
        output_value = graph.outputs[0]
        if output_value.type.dtype != "float":
            raise UnsupportedOpError(
                f"Unsupported dtype {output_value.type.dtype} for {output_value.name}"
            )
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
            input_names=tuple(value.name for value in graph.inputs),
            input_shapes=tuple(value.type.shape for value in graph.inputs),
            output_name=node.outputs[0],
            element_count=element_count,
            output_shape=output_value.type.shape,
            constants=_lowered_constants(graph),
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


@dataclass(frozen=True)
class _ResolvedFeeds:
    feeds: Mapping[str, np.ndarray]
    constants: Mapping[str, np.ndarray]

    def fetch(self, name: str) -> np.ndarray:
        if name in self.constants:
            return self.constants[name]
        return self.feeds[name]


def _lowered_constants(graph: Graph) -> tuple[ConstTensor, ...]:
    return tuple(
        ConstTensor(
            name=initializer.name,
            shape=initializer.type.shape,
            data=tuple(float(value) for value in initializer.data.ravel()),
        )
        for initializer in graph.initializers
    )


def _element_count(shape: tuple[int, ...]) -> int:
    if not shape:
        raise ShapeInferenceError("Scalar outputs are not supported")
    count = 1
    for dim in shape:
        if dim <= 0:
            raise ShapeInferenceError("Dynamic or zero dims are not supported")
        count *= dim
    return count


def _binary_op_symbol(op_type: str) -> str | None:
    if op_type == "Add":
        return "+"
    if op_type == "Mul":
        return "*"
    return None


def _unary_op_symbol(op_type: str) -> str | None:
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


def _apply_binary_op(op_symbol: str, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if op_symbol == "+":
        return left + right
    return left * right


def _apply_unary_op(op_symbol: str, value: np.ndarray) -> np.ndarray:
    if op_symbol == "fabsf":
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

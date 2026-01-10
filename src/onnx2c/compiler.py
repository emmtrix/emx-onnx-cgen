from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import onnx

from .codegen.c_emitter import BinaryOp, CEmitter, LoweredModel, UnaryOp
from .errors import CodegenError, ShapeInferenceError, UnsupportedOpError
from .ir.model import Graph
from .onnx_import import import_onnx


@dataclass(frozen=True)
class CompilerOptions:
    template_dir: Path
    model_name: str = "model"


class Compiler:
    def __init__(self, options: CompilerOptions | None = None) -> None:
        if options is None:
            options = CompilerOptions(template_dir=Path("templates"))
        self._options = options
        self._emitter = CEmitter(options.template_dir)

    def compile(self, model: onnx.ModelProto) -> str:
        graph = import_onnx(model)
        lowered = self._lower_model(graph)
        return self._emitter.emit_model(lowered)

    def _lower_model(self, graph: Graph) -> LoweredModel:
        if len(graph.nodes) != 1:
            if len(graph.nodes) == 2:
                return self._lower_binary_chain_model(graph)
            raise UnsupportedOpError(
                f"Only one- or two-node graphs are supported, got {len(graph.nodes)}"
            )
        node = graph.nodes[0]
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
        if unary_symbol is not None:
            return LoweredModel(
                name=self._options.model_name,
                input_names=(node.inputs[0],),
                output_name=node.outputs[0],
                element_count=element_count,
                output_shape=output_value.type.shape,
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
            input_names=(node.inputs[0], node.inputs[1]),
            output_name=node.outputs[0],
            element_count=element_count,
            output_shape=output_value.type.shape,
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
        return LoweredModel(
            name=self._options.model_name,
            input_names=input_names,
            output_name=node2.outputs[0],
            element_count=element_count,
            output_shape=output_value.type.shape,
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

    def run(self, model: onnx.ModelProto, feeds: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        graph = import_onnx(model)
        if len(graph.nodes) != 1:
            if len(graph.nodes) == 2:
                return self._run_binary_chain(graph, feeds)
            raise UnsupportedOpError("Only one- or two-node graphs are supported")
        node = graph.nodes[0]
        op_symbol = _binary_op_symbol(node.op_type)
        unary_symbol = _unary_op_symbol(node.op_type)
        if op_symbol is None and unary_symbol is None:
            raise UnsupportedOpError(f"Unsupported op {node.op_type}")
        if op_symbol is not None:
            left = feeds[node.inputs[0]]
            right = feeds[node.inputs[1]]
            result = _apply_binary_op(op_symbol, left, right)
        else:
            value = feeds[node.inputs[0]]
            result = _apply_unary_op(unary_symbol, value)
        return {node.outputs[0]: result}

    def _run_binary_chain(
        self, graph: Graph, feeds: Mapping[str, np.ndarray]
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
        left = feeds[node1.inputs[0]]
        right = feeds[node1.inputs[1]]
        tmp = _apply_binary_op(op1_symbol, left, right)
        if node2.inputs[0] == intermediate:
            left = tmp
            right = feeds[node2.inputs[1]]
        else:
            left = feeds[node2.inputs[0]]
            right = tmp
        result = _apply_binary_op(op2_symbol, left, right)
        return {node2.outputs[0]: result}


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
    if op_type == "Tanh":
        return "tanhf"
    return None


def _apply_binary_op(op_symbol: str, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if op_symbol == "+":
        return left + right
    return left * right


def _apply_unary_op(op_symbol: str, value: np.ndarray) -> np.ndarray:
    if op_symbol == "tanhf":
        return np.tanh(value)
    raise UnsupportedOpError(f"Unsupported unary op {op_symbol}")

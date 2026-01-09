from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import onnx

from .codegen.c_emitter import BinaryModel, CEmitter
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
        binary_model = self._lower_binary_model(graph)
        return self._emitter.emit_binary_model(binary_model)

    def _lower_binary_model(self, graph: Graph) -> BinaryModel:
        if len(graph.nodes) != 1:
            raise UnsupportedOpError(
                f"Only single-node graphs are supported, got {len(graph.nodes)}"
            )
        node = graph.nodes[0]
        op_symbol = _binary_op_symbol(node.op_type)
        if op_symbol is None:
            raise UnsupportedOpError(f"Unsupported op {node.op_type}")
        if len(node.inputs) != 2 or len(node.outputs) != 1:
            raise UnsupportedOpError(
                f"{node.op_type} must have 2 inputs and 1 output"
            )
        output_value = graph.outputs[0]
        if output_value.type.dtype != "float":
            raise UnsupportedOpError(
                f"Unsupported dtype {output_value.type.dtype} for {output_value.name}"
            )
        element_count = _element_count(output_value.type.shape)
        if element_count <= 0:
            raise ShapeInferenceError("Output shape must be fully defined")
        return BinaryModel(
            name=self._options.model_name,
            input_names=(node.inputs[0], node.inputs[1]),
            output_name=node.outputs[0],
            element_count=element_count,
            operator=op_symbol,
        )

    def run(self, model: onnx.ModelProto, feeds: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        graph = import_onnx(model)
        if len(graph.nodes) != 1:
            raise UnsupportedOpError("Only single-node graphs are supported")
        node = graph.nodes[0]
        op_symbol = _binary_op_symbol(node.op_type)
        if op_symbol is None:
            raise UnsupportedOpError(f"Unsupported op {node.op_type}")
        left = feeds[node.inputs[0]]
        right = feeds[node.inputs[1]]
        if op_symbol == "+":
            result = left + right
        else:
            result = left * right
        return {node.outputs[0]: result}


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

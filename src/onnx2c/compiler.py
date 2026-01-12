from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import onnx

from .codegen.c_emitter import (
    AttentionOp,
    AveragePoolOp,
    BatchNormOp,
    BinaryOp,
    CEmitter,
    ConstTensor,
    ConvOp,
    ConcatOp,
    ConstantOfShapeOp,
    GemmOp,
    LrnOp,
    LstmOp,
    LogSoftmaxOp,
    LoweredModel,
    MatMulOp,
    MaxPoolOp,
    ReduceOp,
    ReshapeOp,
    ResizeOp,
    SoftmaxOp,
    ShapeOp,
    TransposeOp,
    UnaryOp,
)
from .dtypes import dtype_info
from .errors import ShapeInferenceError, UnsupportedOpError
from .ir.model import Graph
from .lowering.attention import AttentionSpec, resolve_attention_spec
from .lowering.average_pool import (
    lower_average_pool,
    lower_global_average_pool,
)
from .lowering.batch_normalization import lower_batch_normalization
from .lowering.concat import lower_concat
from .lowering.common import (
    ensure_supported_dtype,
    node_dtype,
    shape_product,
    value_dtype,
    value_shape,
)
from .lowering.conv import ConvSpec, resolve_conv_spec
from .lowering.constant_of_shape import lower_constant_of_shape
from .lowering.dropout import lower_dropout
from .lowering.gemm import resolve_gemm_spec, validate_gemm_bias_shape
from .lowering.lrn import LrnSpec, resolve_lrn_spec
from .lowering.logsoftmax import lower_logsoftmax
from .lowering.matmul import lower_matmul
from .lowering.maxpool import MaxPoolSpec, resolve_maxpool_spec
from .lowering.reduce import (
    REDUCE_KIND_BY_OP,
    REDUCE_OUTPUTS_FLOAT_ONLY,
    resolve_reduce_axes,
)
from .lowering.reshape import lower_reshape
from .lowering.resize import lower_resize
from .lowering.shape import lower_shape
from .lowering.softmax import lower_softmax
from .lowering.transpose import lower_transpose
from .lowering.unsqueeze import lower_unsqueeze
from .lowering.registry import get_lowering_registry, resolve_dispatch
from .onnx_import import import_onnx
from .ops import (
    BINARY_OP_TYPES,
    COMPARE_OP_TYPES,
    UNARY_OP_TYPES,
    binary_op_symbol,
    unary_op_symbol,
)
from .runtime.evaluator import Evaluator


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

    def compile_with_data_file(self, model: onnx.ModelProto) -> tuple[str, str]:
        graph = import_onnx(model)
        lowered = self._lower_model(graph)
        return self._emitter.emit_model_with_data_file(
            lowered, emit_testbench=self._options.emit_testbench
        )

    def _lower_model(self, graph: Graph) -> LoweredModel:
        if not graph.outputs:
            raise UnsupportedOpError("Graph must have at least one output")
        if not graph.nodes:
            raise UnsupportedOpError("Graph must contain at least one node")
        output_names = tuple(value.name for value in graph.outputs)
        output_shapes = tuple(value.type.shape for value in graph.outputs)
        output_dtypes = tuple(
            value_dtype(graph, value.name) for value in graph.outputs
        )
        for shape in output_shapes:
            element_count = shape_product(shape)
            if element_count <= 0:
                raise ShapeInferenceError("Output shape must be fully defined")
        constants = _lowered_constants(graph)
        input_names = tuple(value.name for value in graph.inputs)
        input_shapes = tuple(value.type.shape for value in graph.inputs)
        input_dtypes = tuple(
            value_dtype(graph, value.name) for value in graph.inputs
        )
        ops: list[
            BinaryOp
            | UnaryOp
            | MatMulOp
            | GemmOp
            | AttentionOp
            | ConvOp
            | AveragePoolOp
            | BatchNormOp
            | LrnOp
            | LstmOp
            | SoftmaxOp
            | LogSoftmaxOp
            | MaxPoolOp
            | ConcatOp
            | TransposeOp
            | ConstantOfShapeOp
            | ReshapeOp
            | ResizeOp
            | ReduceOp
            | ShapeOp
        ] = []
        for node in graph.nodes:
            lowering = resolve_dispatch(
                node.op_type,
                get_lowering_registry(),
                binary_types=BINARY_OP_TYPES,
                unary_types=UNARY_OP_TYPES,
                binary_fallback=lambda: _lower_binary_unary,
                unary_fallback=lambda: _lower_binary_unary,
            )
            ops.append(lowering(graph, node))
        return LoweredModel(
            name=self._options.model_name,
            input_names=input_names,
            input_shapes=input_shapes,
            input_dtypes=input_dtypes,
            output_names=output_names,
            output_shapes=output_shapes,
            output_dtypes=output_dtypes,
            constants=constants,
            ops=tuple(ops),
        )

    def run(
        self, model: onnx.ModelProto, feeds: Mapping[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        graph = import_onnx(model)
        evaluator = Evaluator(graph)
        return evaluator.run(feeds)


def _lowered_constants(graph: Graph) -> tuple[ConstTensor, ...]:
    constants: list[ConstTensor] = []
    for initializer in graph.initializers:
        dtype = ensure_supported_dtype(initializer.type.dtype)
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


def _lower_binary_unary(graph: Graph, node: Node) -> BinaryOp | UnaryOp:
    if node.op_type in COMPARE_OP_TYPES:
        input_dtype = node_dtype(graph, node, *node.inputs)
        output_dtype = value_dtype(graph, node.outputs[0], node)
        op_spec = binary_op_symbol(node.op_type, node.attrs, dtype=input_dtype)
        if op_spec is None:
            raise UnsupportedOpError(f"Unsupported op {node.op_type}")
        if len(node.inputs) != 2 or len(node.outputs) != 1:
            raise UnsupportedOpError(
                f"{node.op_type} must have 2 inputs and 1 output"
            )
        if output_dtype != "bool":
            raise UnsupportedOpError(
                f"{node.op_type} expects bool output, got {output_dtype}"
            )
        output_shape = value_shape(graph, node.outputs[0], node)
        return BinaryOp(
            input0=node.inputs[0],
            input1=node.inputs[1],
            output=node.outputs[0],
            operator=op_spec.operator,
            operator_kind=op_spec.kind,
            shape=output_shape,
            dtype=output_dtype,
            input_dtype=input_dtype,
        )
    op_dtype = node_dtype(graph, node, *node.inputs, *node.outputs)
    op_spec = binary_op_symbol(node.op_type, node.attrs, dtype=op_dtype)
    unary_symbol = unary_op_symbol(node.op_type, dtype=op_dtype)
    if op_spec is None and unary_symbol is None:
        raise UnsupportedOpError(f"Unsupported op {node.op_type}")
    if op_spec is not None:
        if len(node.inputs) != 2 or len(node.outputs) != 1:
            raise UnsupportedOpError(
                f"{node.op_type} must have 2 inputs and 1 output"
            )
        output_shape = value_shape(graph, node.outputs[0], node)
        return BinaryOp(
            input0=node.inputs[0],
            input1=node.inputs[1],
            output=node.outputs[0],
            operator=op_spec.operator,
            operator_kind=op_spec.kind,
            shape=output_shape,
            dtype=op_dtype,
            input_dtype=op_dtype,
        )
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        raise UnsupportedOpError(f"{node.op_type} must have 1 input and 1 output")
    output_shape = value_shape(graph, node.outputs[0], node)
    return UnaryOp(
        input0=node.inputs[0],
        output=node.outputs[0],
        operator=unary_symbol,
        shape=output_shape,
        dtype=op_dtype,
    )

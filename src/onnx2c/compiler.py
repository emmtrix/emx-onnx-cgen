from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Mapping

import numpy as np
import onnx

from .codegen.c_emitter import (
    AttentionOp,
    AveragePoolOp,
    BatchNormOp,
    BinaryOp,
    CastOp,
    CEmitter,
    ConstTensor,
    ConvOp,
    ConcatOp,
    ConstantOfShapeOp,
    GemmOp,
    GatherElementsOp,
    LrnOp,
    LstmOp,
    LogSoftmaxOp,
    NegativeLogLikelihoodLossOp,
    NodeInfo,
    SoftmaxCrossEntropyLossOp,
    LoweredModel,
    ModelHeader,
    MatMulOp,
    MaxPoolOp,
    ReduceOp,
    ReshapeOp,
    ResizeOp,
    SoftmaxOp,
    ShapeOp,
    TransposeOp,
    UnaryOp,
    WhereOp,
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
from .lowering.cast import lower_cast
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
from .lowering.gather_elements import lower_gather_elements
from .lowering.gemm import resolve_gemm_spec, validate_gemm_bias_shape
from .lowering.lrn import LrnSpec, resolve_lrn_spec
from .lowering.logsoftmax import lower_logsoftmax
from .lowering.negative_log_likelihood_loss import (
    lower_negative_log_likelihood_loss,
)
from .lowering.softmax_cross_entropy_loss import (
    lower_softmax_cross_entropy_loss,
)
from .lowering.matmul import lower_matmul
from .lowering.maxpool import MaxPoolSpec, resolve_maxpool_spec
from .lowering.reduce import (
    REDUCE_KIND_BY_OP,
    REDUCE_OUTPUTS_FLOAT_ONLY,
)
from .lowering.reshape import lower_reshape
from .lowering.resize import lower_resize
from .lowering.shape import lower_shape
from .lowering.softmax import lower_softmax
from .lowering.transpose import lower_transpose
from .lowering.unsqueeze import lower_unsqueeze
from .lowering.where import lower_where
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
    command_line: str | None = None
    model_checksum: str | None = None
    restrict_arrays: bool = True


class Compiler:
    def __init__(self, options: CompilerOptions | None = None) -> None:
        if options is None:
            options = CompilerOptions(template_dir=Path("templates"))
        self._options = options
        self._emitter = CEmitter(
            options.template_dir, restrict_arrays=options.restrict_arrays
        )

    def compile(self, model: onnx.ModelProto) -> str:
        graph = import_onnx(model)
        lowered = self._lower_model(model, graph)
        return self._emitter.emit_model(
            lowered, emit_testbench=self._options.emit_testbench
        )

    def compile_with_data_file(self, model: onnx.ModelProto) -> tuple[str, str]:
        graph = import_onnx(model)
        lowered = self._lower_model(model, graph)
        return self._emitter.emit_model_with_data_file(
            lowered, emit_testbench=self._options.emit_testbench
        )

    def _lower_model(self, model: onnx.ModelProto, graph: Graph) -> LoweredModel:
        constants = _lowered_constants(graph)
        self._validate_graph(graph)
        (
            input_names,
            input_shapes,
            input_dtypes,
            output_names,
            output_shapes,
            output_dtypes,
        ) = self._collect_io_specs(graph)
        ops, node_infos = self._lower_nodes(graph)
        header = self._build_header(model, graph)
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
            node_infos=tuple(node_infos),
            header=header,
        )

    def _validate_graph(self, graph: Graph) -> None:
        if not graph.outputs:
            raise UnsupportedOpError("Graph must have at least one output")
        if not graph.nodes:
            raise UnsupportedOpError("Graph must contain at least one node")
        for value in graph.outputs:
            element_count = shape_product(value.type.shape)
            if element_count <= 0:
                raise ShapeInferenceError("Output shape must be fully defined")

    def _collect_io_specs(
        self, graph: Graph
    ) -> tuple[
        tuple[str, ...],
        tuple[tuple[int, ...], ...],
        tuple[str, ...],
        tuple[str, ...],
        tuple[tuple[int, ...], ...],
        tuple[str, ...],
    ]:
        input_names = tuple(value.name for value in graph.inputs)
        input_shapes = tuple(value.type.shape for value in graph.inputs)
        input_dtypes = tuple(
            value_dtype(graph, value.name) for value in graph.inputs
        )
        output_names = tuple(value.name for value in graph.outputs)
        output_shapes = tuple(value.type.shape for value in graph.outputs)
        output_dtypes = tuple(
            value_dtype(graph, value.name) for value in graph.outputs
        )
        return (
            input_names,
            input_shapes,
            input_dtypes,
            output_names,
            output_shapes,
            output_dtypes,
        )

    def _lower_nodes(
        self, graph: Graph
    ) -> tuple[
        list[
            BinaryOp
            | UnaryOp
            | CastOp
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
            | NegativeLogLikelihoodLossOp
            | SoftmaxCrossEntropyLossOp
            | MaxPoolOp
            | ConcatOp
            | GatherElementsOp
            | TransposeOp
            | ConstantOfShapeOp
            | ReshapeOp
            | ResizeOp
            | ReduceOp
            | ShapeOp
        ],
        list[NodeInfo],
    ]:
        ops: list[
            BinaryOp
            | UnaryOp
            | CastOp
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
            | NegativeLogLikelihoodLossOp
            | SoftmaxCrossEntropyLossOp
            | MaxPoolOp
            | ConcatOp
            | GatherElementsOp
            | TransposeOp
            | ConstantOfShapeOp
            | ReshapeOp
            | ResizeOp
            | ReduceOp
            | ShapeOp
            | WhereOp
        ] = []
        node_infos: list[NodeInfo] = []
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
            node_infos.append(
                NodeInfo(
                    op_type=node.op_type,
                    inputs=tuple(node.inputs),
                    outputs=tuple(node.outputs),
                    attrs=dict(node.attrs),
                )
            )
        return ops, node_infos

    def _build_header(self, model: onnx.ModelProto, graph: Graph) -> ModelHeader:
        metadata_props = tuple(
            (prop.key, prop.value) for prop in model.metadata_props
        )
        opset_imports = tuple(
            (opset.domain, opset.version) for opset in model.opset_import
        )
        checksum = self._options.model_checksum
        if checksum is None:
            checksum = hashlib.sha256(model.SerializeToString()).hexdigest()
        return ModelHeader(
            generator="Generated by emmtrix ONNX to C Compiler (emx-onnx2c)",
            command_line=self._options.command_line,
            model_checksum=checksum,
            model_name=self._options.model_name,
            graph_name=model.graph.name or None,
            description=model.doc_string or None,
            graph_description=model.graph.doc_string or None,
            producer_name=model.producer_name or None,
            producer_version=model.producer_version or None,
            domain=model.domain or None,
            model_version=model.model_version or None,
            ir_version=model.ir_version or None,
            opset_imports=opset_imports,
            metadata_props=metadata_props,
            input_count=len(graph.inputs),
            output_count=len(graph.outputs),
            node_count=len(graph.nodes),
            initializer_count=len(graph.initializers),
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

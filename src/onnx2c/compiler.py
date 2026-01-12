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
from .ir.model import Graph, Node
from .lowering import get_lowering
from .lowering.attention import AttentionSpec, resolve_attention_spec
from .lowering.average_pool import (
    lower_average_pool,
    lower_global_average_pool,
)
from .lowering.batch_normalization import lower_batch_normalization
from .lowering.concat import lower_concat
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
from .onnx_import import import_onnx
from .ops import (
    BINARY_OP_TYPES,
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

    def _lower_model(self, graph: Graph) -> LoweredModel:
        if not graph.outputs:
            raise UnsupportedOpError("Graph must have at least one output")
        if not graph.nodes:
            raise UnsupportedOpError("Graph must contain at least one node")
        output_names = tuple(value.name for value in graph.outputs)
        output_shapes = tuple(value.type.shape for value in graph.outputs)
        output_dtypes = tuple(
            _value_dtype(graph, value.name) for value in graph.outputs
        )
        for shape in output_shapes:
            element_count = _element_count(shape)
            if element_count <= 0:
                raise ShapeInferenceError("Output shape must be fully defined")
        constants = _lowered_constants(graph)
        input_names = tuple(value.name for value in graph.inputs)
        input_shapes = tuple(value.type.shape for value in graph.inputs)
        input_dtypes = tuple(
            _value_dtype(graph, value.name) for value in graph.inputs
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
            lowering = get_lowering(node.op_type)
            if lowering is not None:
                ops.append(lowering(graph, node))
                continue
            if node.op_type not in BINARY_OP_TYPES | UNARY_OP_TYPES:
                raise UnsupportedOpError(f"Unsupported op {node.op_type}")
            op_dtype = _node_dtype(graph, node, *node.inputs, *node.outputs)
            op_spec = binary_op_symbol(node.op_type, node.attrs, dtype=op_dtype)
            unary_symbol = unary_op_symbol(node.op_type, dtype=op_dtype)
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
    if dtype not in {
        "float",
        "double",
        "bool",
        "int64",
        "int32",
        "int16",
        "int8",
        "uint64",
        "uint32",
        "uint16",
        "uint8",
    }:
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
    filtered = [name for name in names if name]
    if not filtered:
        raise UnsupportedOpError(
            f"{node.op_type} expects at least one typed input or output"
        )
    dtypes = {_value_dtype(graph, name, node) for name in filtered}
    if len(dtypes) != 1:
        raise UnsupportedOpError(
            f"{node.op_type} expects matching dtypes, got {', '.join(sorted(dtypes))}"
        )
    return next(iter(dtypes))


def _element_count(shape: tuple[int, ...]) -> int:
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

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import onnx
from onnx import numpy_helper

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
from .dtypes import ONNX_TO_DTYPE, dtype_info
from .errors import CodegenError, ShapeInferenceError, UnsupportedOpError
from .ir.model import Graph, Initializer, Node
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
            if node.op_type not in _BINARY_OP_TYPES | _UNARY_OP_TYPES:
                raise UnsupportedOpError(f"Unsupported op {node.op_type}")
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
        constants = {
            initializer.name: initializer.data for initializer in graph.initializers
        }
        values: dict[str, np.ndarray] = dict(constants)
        values.update(feeds)
        for node in graph.nodes:
            if node.op_type in {"MatMul", "Gemm"}:
                if node.op_type == "Gemm":
                    op_dtype = _node_dtype(graph, node, *node.inputs, *node.outputs)
                    spec = resolve_gemm_spec(graph, node, op_dtype)
                    left = values[node.inputs[0]]
                    right = values[node.inputs[1]]
                    if spec.trans_a:
                        left = left.T
                    if spec.trans_b:
                        right = right.T
                    result = _apply_matmul(left, right)
                    if op_dtype in {"float", "double"}:
                        alpha = float(spec.alpha)
                        beta = float(spec.beta)
                    else:
                        alpha = int(spec.alpha)
                        beta = int(spec.beta)
                    if alpha != 1:
                        result = result * alpha
                    if len(node.inputs) == 3:
                        bias = values[node.inputs[2]]
                        validate_gemm_bias_shape(
                            (spec.m, spec.n), bias.shape, node
                        )
                        if beta != 1:
                            bias = bias * beta
                        result = result + bias
                    values[node.outputs[0]] = result
                else:
                    left = values[node.inputs[0]]
                    right = values[node.inputs[1]]
                    result = _apply_matmul(left, right)
                    values[node.outputs[0]] = result
                continue
            if node.op_type == "Attention":
                input_q = node.inputs[0]
                input_k = node.inputs[1]
                input_v = node.inputs[2]
                output_y = node.outputs[0]
                op_dtype = _node_dtype(
                    graph, node, input_q, input_k, input_v, output_y
                )
                spec = resolve_attention_spec(graph, node, op_dtype)
                attn_mask_name = _optional_name(node.inputs, 3)
                past_key_name = _optional_name(node.inputs, 4)
                past_value_name = _optional_name(node.inputs, 5)
                nonpad_name = _optional_name(node.inputs, 6)
                present_key_name = _optional_name(node.outputs, 1)
                present_value_name = _optional_name(node.outputs, 2)
                qk_matmul_output_name = _optional_name(node.outputs, 3)
                output, present_key, present_value, qk_output = _apply_attention(
                    spec,
                    values[input_q],
                    values[input_k],
                    values[input_v],
                    values[attn_mask_name] if attn_mask_name else None,
                    values[past_key_name] if past_key_name else None,
                    values[past_value_name] if past_value_name else None,
                    values[nonpad_name] if nonpad_name else None,
                )
                values[output_y] = output
                if present_key_name is not None:
                    values[present_key_name] = present_key
                if present_value_name is not None:
                    values[present_value_name] = present_value
                if qk_matmul_output_name is not None:
                    values[qk_matmul_output_name] = qk_output
                continue
            if node.op_type == "Conv":
                op_dtype = _node_dtype(graph, node, *node.inputs, *node.outputs)
                if op_dtype not in {"float", "double"}:
                    raise UnsupportedOpError(
                        "Conv supports float and double inputs only"
                    )
                spec = resolve_conv_spec(graph, node)
                data = values[node.inputs[0]]
                weights = values[node.inputs[1]]
                bias = values[node.inputs[2]] if len(node.inputs) > 2 else None
                values[node.outputs[0]] = _apply_conv(
                    spec, data, weights, bias
                )
                continue
            if node.op_type == "BatchNormalization":
                if len(node.inputs) != 5 or len(node.outputs) != 1:
                    raise UnsupportedOpError(
                        "BatchNormalization must have 5 inputs and 1 output"
                    )
                op_dtype = _node_dtype(graph, node, *node.inputs, *node.outputs)
                if op_dtype not in {"float", "double"}:
                    raise UnsupportedOpError(
                        "BatchNormalization supports float and double inputs only"
                    )
                is_test = int(node.attrs.get("is_test", 1))
                if is_test != 1:
                    raise UnsupportedOpError(
                        "BatchNormalization supports is_test=1 only"
                    )
                training_mode = int(node.attrs.get("training_mode", 0))
                if training_mode != 0:
                    raise UnsupportedOpError(
                        "BatchNormalization supports training_mode=0 only"
                    )
                spatial = int(node.attrs.get("spatial", 1))
                if spatial != 1:
                    raise UnsupportedOpError(
                        "BatchNormalization supports spatial=1 only"
                    )
                epsilon = float(node.attrs.get("epsilon", 1e-5))
                input_shape = _value_shape(graph, node.inputs[0], node)
                if len(input_shape) < 2:
                    raise UnsupportedOpError(
                        "BatchNormalization expects input rank of at least 2"
                    )
                channels = input_shape[1]
                for name in node.inputs[1:]:
                    shape = _value_shape(graph, name, node)
                    if shape != (channels,):
                        raise ShapeInferenceError(
                            "BatchNormalization parameter shape must be "
                            f"({channels},), got {shape}"
                        )
                data = values[node.inputs[0]]
                scale = values[node.inputs[1]].reshape(
                    (1, channels) + (1,) * (data.ndim - 2)
                )
                bias = values[node.inputs[2]].reshape(
                    (1, channels) + (1,) * (data.ndim - 2)
                )
                mean = values[node.inputs[3]].reshape(
                    (1, channels) + (1,) * (data.ndim - 2)
                )
                variance = values[node.inputs[4]].reshape(
                    (1, channels) + (1,) * (data.ndim - 2)
                )
                values[node.outputs[0]] = (
                    (data - mean) / np.sqrt(variance + epsilon) * scale + bias
                )
                continue
            if node.op_type == "LRN":
                op_dtype = _node_dtype(graph, node, *node.inputs, *node.outputs)
                if op_dtype not in {"float", "double"}:
                    raise UnsupportedOpError("LRN supports float and double inputs only")
                spec = resolve_lrn_spec(graph, node)
                data = values[node.inputs[0]]
                values[node.outputs[0]] = _apply_lrn(spec, data)
                continue
            if node.op_type == "AveragePool":
                op = lower_average_pool(graph, node)
                data = values[node.inputs[0]]
                values[node.outputs[0]] = _apply_average_pool(op, data)
                continue
            if node.op_type == "GlobalAveragePool":
                op = lower_global_average_pool(graph, node)
                data = values[node.inputs[0]]
                values[node.outputs[0]] = _apply_average_pool(op, data)
                continue
            if node.op_type == "MaxPool":
                op_dtype = _node_dtype(graph, node, *node.inputs, *node.outputs)
                if op_dtype == "bool":
                    raise UnsupportedOpError("MaxPool supports numeric inputs only")
                spec = resolve_maxpool_spec(graph, node)
                data = values[node.inputs[0]]
                values[node.outputs[0]] = _apply_maxpool(spec, data)
                continue
            if node.op_type == "Softmax":
                op_dtype = _node_dtype(graph, node, *node.inputs, *node.outputs)
                if op_dtype not in {"float", "double"}:
                    raise UnsupportedOpError(
                        "Softmax supports float and double inputs only"
                    )
                axis = _normalize_axis(
                    int(node.attrs.get("axis", -1)),
                    _value_shape(graph, node.inputs[0], node),
                    node,
                )
                value = values[node.inputs[0]]
                values[node.outputs[0]] = _apply_softmax(value, axis)
                continue
            if node.op_type == "LogSoftmax":
                op_dtype = _node_dtype(graph, node, *node.inputs, *node.outputs)
                if op_dtype not in {"float", "double"}:
                    raise UnsupportedOpError(
                        "LogSoftmax supports float and double inputs only"
                    )
                axis = _normalize_axis(
                    int(node.attrs.get("axis", -1)),
                    _value_shape(graph, node.inputs[0], node),
                    node,
                )
                value = values[node.inputs[0]]
                values[node.outputs[0]] = _apply_logsoftmax(value, axis)
                continue
            if node.op_type in REDUCE_KIND_BY_OP:
                if len(node.inputs) not in {1, 2} or len(node.outputs) != 1:
                    raise UnsupportedOpError(
                        f"{node.op_type} must have 1 or 2 inputs and 1 output"
                    )
                op_dtype = _node_dtype(graph, node, *node.inputs, *node.outputs)
                if (
                    node.op_type in REDUCE_OUTPUTS_FLOAT_ONLY
                    and op_dtype not in {"float", "double"}
                ):
                    raise UnsupportedOpError(
                        f"{node.op_type} supports float and double inputs only"
                    )
                value = values[node.inputs[0]]
                input_shape = _value_shape(graph, node.inputs[0], node)
                axes, noop = resolve_reduce_axes(graph, node, input_shape)
                if noop:
                    values[node.outputs[0]] = value.copy()
                    continue
                keepdims = bool(int(node.attrs.get("keepdims", 1)))
                reduce_kind = REDUCE_KIND_BY_OP[node.op_type]
                if reduce_kind == "sum":
                    result = np.sum(value, axis=axes, keepdims=keepdims)
                elif reduce_kind == "mean":
                    result = np.mean(value, axis=axes, keepdims=keepdims)
                elif reduce_kind == "max":
                    result = np.max(value, axis=axes, keepdims=keepdims)
                elif reduce_kind == "min":
                    result = np.min(value, axis=axes, keepdims=keepdims)
                elif reduce_kind == "prod":
                    result = np.prod(value, axis=axes, keepdims=keepdims)
                elif reduce_kind == "l1":
                    result = np.sum(
                        np.abs(value), axis=axes, keepdims=keepdims
                    )
                elif reduce_kind == "l2":
                    result = np.sqrt(
                        np.sum(value * value, axis=axes, keepdims=keepdims)
                    )
                elif reduce_kind == "logsum":
                    result = np.log(
                        np.sum(value, axis=axes, keepdims=keepdims)
                    )
                elif reduce_kind == "logsumexp":
                    result = np.log(
                        np.sum(np.exp(value), axis=axes, keepdims=keepdims)
                    )
                elif reduce_kind == "sumsquare":
                    result = np.sum(
                        value * value, axis=axes, keepdims=keepdims
                    )
                else:
                    raise UnsupportedOpError(
                        f"Unsupported reduce kind {reduce_kind}"
                    )
                values[node.outputs[0]] = result
                continue
            if node.op_type == "Dropout":
                if len(node.outputs) not in {1, 2} or len(node.inputs) != 1:
                    raise UnsupportedOpError(
                        "Dropout supports only the data input and 1 or 2 outputs"
                    )
                if len(node.outputs) == 2 and _is_value_used(graph, node.outputs[1]):
                    raise UnsupportedOpError(
                        "Dropout mask output is not supported"
                    )
                values[node.outputs[0]] = values[node.inputs[0]].copy()
                continue
            if node.op_type == "Concat":
                axis = int(node.attrs.get("axis", 0))
                tensors = [values[name] for name in node.inputs]
                values[node.outputs[0]] = np.concatenate(tensors, axis=axis)
                continue
            if node.op_type == "Transpose":
                perm = node.attrs.get("perm")
                if perm is None:
                    perm = tuple(reversed(range(values[node.inputs[0]].ndim)))
                values[node.outputs[0]] = np.transpose(
                    values[node.inputs[0]], axes=tuple(perm)
                )
                continue
            if node.op_type == "Unsqueeze":
                if len(node.outputs) != 1 or len(node.inputs) not in {1, 2}:
                    raise UnsupportedOpError(
                        "Unsqueeze must have 1 or 2 inputs and 1 output"
                    )
                input_shape = _value_shape(graph, node.inputs[0], node)
                output_shape = _value_shape(graph, node.outputs[0], node)
                axes = _resolve_unsqueeze_axes(graph, node)
                if axes is None:
                    if len(node.inputs) == 2:
                        axes_dtype = _value_dtype(graph, node.inputs[1], node)
                        if axes_dtype not in {"int64", "int32"}:
                            raise UnsupportedOpError(
                                "Unsqueeze axes input must be int64 or int32, "
                                f"got {axes_dtype}"
                            )
                    _validate_unsqueeze_shape_without_axes(
                        input_shape, output_shape, node
                    )
                else:
                    expected_shape = _expected_unsqueeze_shape(
                        input_shape, axes, node
                    )
                    if expected_shape != output_shape:
                        raise ShapeInferenceError(
                            "Unsqueeze output shape must be "
                            f"{expected_shape}, got {output_shape}"
                        )
                values[node.outputs[0]] = values[node.inputs[0]].reshape(
                    output_shape
                )
                continue
            if node.op_type == "Reshape":
                output_shape = _value_shape(graph, node.outputs[0], node)
                values[node.outputs[0]] = values[node.inputs[0]].reshape(
                    output_shape
                )
                continue
            if node.op_type == "ConstantOfShape":
                output_shape = _value_shape(graph, node.outputs[0], node)
                output_dtype = _value_dtype(graph, node.outputs[0], node)
                value_attr = node.attrs.get("value")
                if value_attr is None:
                    if output_dtype != "float":
                        raise UnsupportedOpError(
                            "ConstantOfShape output dtype must be float when value is omitted"
                        )
                    fill_value = 0.0
                else:
                    value_dtype = ONNX_TO_DTYPE.get(value_attr.data_type)
                    if value_dtype is None:
                        raise UnsupportedOpError(
                            f"Unsupported dtype {value_attr.data_type}"
                        )
                    if value_dtype != output_dtype:
                        raise UnsupportedOpError(
                            "ConstantOfShape output dtype must match value dtype"
                        )
                    value_data = numpy_helper.to_array(value_attr)
                    if value_data.size != 1:
                        raise UnsupportedOpError(
                            "ConstantOfShape value must be a scalar"
                        )
                    fill_value = value_data.reshape(-1)[0].item()
                info = dtype_info(output_dtype)
                values[node.outputs[0]] = np.full(
                    output_shape, fill_value, dtype=info.np_dtype
                )
                continue
            if node.op_type == "Shape":
                if len(node.inputs) != 1 or len(node.outputs) != 1:
                    raise UnsupportedOpError("Shape must have 1 input and 1 output")
                input_value = values[node.inputs[0]]
                rank = input_value.ndim
                start_index, end_index = _normalize_shape_slice(
                    rank,
                    start=node.attrs.get("start"),
                    end=node.attrs.get("end"),
                )
                if end_index <= start_index:
                    raise ShapeInferenceError(
                        "Shape start must be less than end"
                    )
                output_dtype = _value_dtype(graph, node.outputs[0], node)
                if output_dtype != "int64":
                    raise UnsupportedOpError("Shape output dtype must be int64")
                output_values = np.array(
                    input_value.shape[start_index:end_index], dtype=np.int64
                )
                values[node.outputs[0]] = output_values
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


_BINARY_OP_TYPES = {
    "Add",
    "And",
    "Div",
    "Max",
    "Mean",
    "Min",
    "Mod",
    "Mul",
    "Or",
    "PRelu",
    "Pow",
    "Sub",
    "Sum",
    "Xor",
}

_UNARY_OP_TYPES = {
    "Abs",
    "Atanh",
    "Ceil",
    "Cos",
    "Exp",
    "Floor",
    "Log",
    "Neg",
    "Not",
    "Relu",
    "Sin",
    "Sqrt",
    "Tan",
    "Tanh",
}


def _format_float_literal(value: float, dtype: str) -> str:
    formatted = f"{value:.9g}"
    if "e" not in formatted and "E" not in formatted and "." not in formatted:
        formatted = f"{formatted}.0"
    if dtype == "float":
        return f"{formatted}f"
    return formatted


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


def _shape_product(shape: tuple[int, ...]) -> int:
    if not shape:
        return 1
    product = 1
    for dim in shape:
        if dim <= 0:
            raise ShapeInferenceError("Dynamic or zero dims are not supported")
        product *= dim
    return product


def _normalize_axis(axis: int, shape: tuple[int, ...], node: Node) -> int:
    if not shape:
        raise ShapeInferenceError(
            f"{node.op_type} does not support scalar inputs"
        )
    rank = len(shape)
    if axis < 0:
        axis += rank
    if axis < 0 or axis >= rank:
        raise ShapeInferenceError(
            f"{node.op_type} axis {axis} is out of range for rank {rank}"
        )
    return axis


def _normalize_shape_slice(
    rank: int, *, start: int | None, end: int | None
) -> tuple[int, int]:
    normalized_start = 0 if start is None else int(start)
    normalized_end = rank if end is None else int(end)
    if normalized_start < 0:
        normalized_start += rank
    if normalized_end < 0:
        normalized_end += rank
    normalized_start = max(0, min(normalized_start, rank))
    normalized_end = max(0, min(normalized_end, rank))
    return normalized_start, normalized_end


def _find_initializer(graph: Graph, name: str) -> Initializer | None:
    for initializer in graph.initializers:
        if initializer.name == name:
            return initializer
    return None


def _normalize_unsqueeze_axes(
    axes: list[int], output_rank: int, node: Node
) -> tuple[int, ...]:
    normalized: list[int] = []
    for axis in axes:
        if axis < 0:
            axis += output_rank
        if axis < 0 or axis >= output_rank:
            raise ShapeInferenceError(
                f"{node.op_type} axis {axis} is out of range for rank {output_rank}"
            )
        normalized.append(axis)
    if len(set(normalized)) != len(normalized):
        raise ShapeInferenceError(f"{node.op_type} axes must be unique")
    return tuple(sorted(normalized))


def _resolve_unsqueeze_axes(graph: Graph, node: Node) -> tuple[int, ...] | None:
    axes_attr = node.attrs.get("axes")
    axes_values: list[int] | None = None
    if len(node.inputs) == 2:
        axes_initializer = _find_initializer(graph, node.inputs[1])
        if axes_initializer is not None:
            if axes_initializer.type.dtype not in {"int64", "int32"}:
                raise UnsupportedOpError(
                    "Unsqueeze axes input must be int64 or int32, "
                    f"got {axes_initializer.type.dtype}"
                )
            axes_values = [
                int(value) for value in axes_initializer.data.reshape(-1)
            ]
    elif axes_attr is not None:
        axes_values = [int(value) for value in axes_attr]
    if axes_values is None and axes_attr is None and len(node.inputs) != 2:
        raise UnsupportedOpError("Unsqueeze requires axes")
    if axes_values is None:
        return None
    if not axes_values:
        raise UnsupportedOpError("Unsqueeze requires non-empty axes")
    return tuple(axes_values)


def _expected_unsqueeze_shape(
    input_shape: tuple[int, ...], axes: tuple[int, ...], node: Node
) -> tuple[int, ...]:
    output_rank = len(input_shape) + len(axes)
    normalized_axes = _normalize_unsqueeze_axes(list(axes), output_rank, node)
    output_dims: list[int] = []
    input_index = 0
    for axis in range(output_rank):
        if axis in normalized_axes:
            output_dims.append(1)
        else:
            output_dims.append(input_shape[input_index])
            input_index += 1
    return tuple(output_dims)


def _validate_unsqueeze_shape_without_axes(
    input_shape: tuple[int, ...], output_shape: tuple[int, ...], node: Node
) -> None:
    if len(output_shape) <= len(input_shape):
        raise ShapeInferenceError(
            "Unsqueeze output rank must exceed input rank"
        )
    input_index = 0
    for dim in output_shape:
        if input_index < len(input_shape) and dim == input_shape[input_index]:
            input_index += 1
            continue
        if dim != 1:
            raise ShapeInferenceError(
                "Unsqueeze output shape must insert ones only"
            )
    if input_index != len(input_shape):
        raise ShapeInferenceError(
            "Unsqueeze output shape must contain input shape in order"
        )


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
    if dtype in {"int64", "int32", "int16", "int8", "uint64", "uint32", "uint16", "uint8"}:
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
        func = "fmaxf" if dtype == "float" else "fmax"
        return _BinaryOpSpec(func, "func", np.maximum)
    if op_type == "Mean":
        mean_literal = _format_float_literal(0.5, dtype)
        return _BinaryOpSpec(
            f"({{left}} + {{right}}) * {mean_literal}",
            "expr",
            lambda left, right: (left + right) * 0.5,
        )
    if op_type == "Min":
        func = "fminf" if dtype == "float" else "fmin"
        return _BinaryOpSpec(func, "func", np.minimum)
    if op_type == "Mod":
        fmod = 0
        if attrs is not None:
            fmod = int(attrs.get("fmod", 0))
        if fmod != 1:
            raise UnsupportedOpError(
                "Mod only supports fmod=1 for floating point types"
            )
        func = "fmodf" if dtype == "float" else "fmod"
        return _BinaryOpSpec(func, "func", np.fmod)
    if op_type == "Mul":
        return _BinaryOpSpec("*", "infix", lambda left, right: left * right)
    if op_type == "Pow":
        func = "powf" if dtype == "float" else "pow"
        return _BinaryOpSpec(func, "func", np.power)
    if op_type == "PRelu":
        zero_literal = _format_float_literal(0.0, dtype)
        return _BinaryOpSpec(
            f"({{left}} > {zero_literal} ? {{left}} : {{right}} * {{left}})",
            "expr",
            lambda left, right: np.where(left > 0.0, left, right * left),
        )
    if op_type == "Sub":
        return _BinaryOpSpec("-", "infix", lambda left, right: left - right)
    if op_type == "Sum":
        return _BinaryOpSpec("+", "infix", lambda left, right: left + right)
    return None


def _unique_value_name(graph: Graph, base: str) -> str:
    existing = {value.name for value in graph.inputs + graph.outputs + graph.values}
    existing.update(initializer.name for initializer in graph.initializers)
    candidate = base
    index = 1
    while candidate in existing:
        candidate = f"{base}_{index}"
        index += 1
    return candidate


def _is_value_used(graph: Graph, name: str) -> bool:
    if any(value.name == name for value in graph.outputs):
        return True
    return any(name in node.inputs for node in graph.nodes)


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
    if dtype == "double":
        if op_type == "Abs":
            return "fabs"
        if op_type == "Ceil":
            return "ceil"
        if op_type == "Cos":
            return "cos"
        if op_type == "Exp":
            return "exp"
        if op_type == "Floor":
            return "floor"
        if op_type == "Log":
            return "log"
        if op_type == "Neg":
            return "neg"
        if op_type == "Relu":
            return "relu"
        if op_type == "Sin":
            return "sin"
        if op_type == "Sqrt":
            return "sqrt"
        if op_type == "Tan":
            return "tan"
        if op_type == "Tanh":
            return "tanh"
        if op_type == "Atanh":
            return "atanh"
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
    if op_symbol in {"fabsf", "fabs"}:
        return np.abs(value)
    if op_symbol == "abs":
        return np.abs(value)
    if op_symbol == "llabs":
        return np.abs(value)
    if op_symbol == "!":
        return np.logical_not(value)
    if op_symbol in {"ceilf", "ceil"}:
        return np.ceil(value)
    if op_symbol in {"cosf", "cos"}:
        return np.cos(value)
    if op_symbol in {"expf", "exp"}:
        return np.exp(value)
    if op_symbol in {"floorf", "floor"}:
        return np.floor(value)
    if op_symbol in {"logf", "log"}:
        return np.log(value)
    if op_symbol == "neg":
        return -value
    if op_symbol == "relu":
        return np.maximum(value, 0)
    if op_symbol in {"sinf", "sin"}:
        return np.sin(value)
    if op_symbol in {"sqrtf", "sqrt"}:
        return np.sqrt(value)
    if op_symbol in {"tanf", "tan"}:
        return np.tan(value)
    if op_symbol in {"tanhf", "tanh"}:
        return np.tanh(value)
    if op_symbol in {"atanhf", "atanh"}:
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


def _apply_softmax(values: np.ndarray, axis: int) -> np.ndarray:
    max_values = np.max(values, axis=axis, keepdims=True)
    exp_values = np.exp(values - max_values)
    sum_values = np.sum(exp_values, axis=axis, keepdims=True)
    return exp_values / sum_values


def _apply_logsoftmax(values: np.ndarray, axis: int) -> np.ndarray:
    max_values = np.max(values, axis=axis, keepdims=True)
    shifted = values - max_values
    logsum = np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
    return shifted - logsum


def _optional_name(names: Sequence[str], index: int) -> str | None:
    if index >= len(names):
        return None
    name = names[index]
    return name or None


def _apply_attention(
    spec: AttentionSpec,
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    attn_mask: np.ndarray | None,
    past_key: np.ndarray | None,
    past_value: np.ndarray | None,
    nonpad_kv_seqlen: np.ndarray | None,
) -> tuple[
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    if spec.q_rank == 3:
        query_4d = query.reshape(
            spec.batch, spec.q_seq, spec.q_heads, spec.qk_head_size
        ).transpose(0, 2, 1, 3)
        key_4d = key.reshape(
            spec.batch, spec.kv_seq, spec.kv_heads, spec.qk_head_size
        ).transpose(0, 2, 1, 3)
        value_4d = value.reshape(
            spec.batch, spec.kv_seq, spec.kv_heads, spec.v_head_size
        ).transpose(0, 2, 1, 3)
    else:
        query_4d = query
        key_4d = key
        value_4d = value
    if past_key is not None and past_value is not None:
        key_total = np.concatenate([past_key, key_4d], axis=2)
        value_total = np.concatenate([past_value, value_4d], axis=2)
    else:
        key_total = key_4d
        value_total = value_4d
    if spec.head_group_size > 1:
        key_total_expanded = np.repeat(key_total, spec.head_group_size, axis=1)
        value_total_expanded = np.repeat(
            value_total, spec.head_group_size, axis=1
        )
    else:
        key_total_expanded = key_total
        value_total_expanded = value_total
    k_transpose = np.transpose(key_total_expanded, (0, 1, 3, 2))
    scores = np.matmul(query_4d, k_transpose) * spec.scale
    bias = np.zeros_like(scores)
    if spec.has_attn_mask and attn_mask is not None:
        if spec.mask_is_bool:
            bias_mask = np.where(attn_mask, 0.0, -np.inf)
        else:
            bias_mask = attn_mask.astype(scores.dtype)
        if spec.mask_rank == 2:
            bias_mask = bias_mask[None, None, ...]
        elif spec.mask_rank == 3:
            bias_mask = bias_mask[:, None, ...]
        bias_mask = np.broadcast_to(
            bias_mask, (spec.batch, spec.q_heads, spec.q_seq, spec.mask_kv_seq)
        )
        if spec.mask_kv_seq < spec.total_seq:
            pad_width = spec.total_seq - spec.mask_kv_seq
            bias_mask = np.pad(
                bias_mask,
                ((0, 0), (0, 0), (0, 0), (0, pad_width)),
                constant_values=-np.inf,
            )
        bias = bias + bias_mask
    if spec.has_nonpad and nonpad_kv_seqlen is not None:
        kv_range = np.arange(spec.total_seq)[None, None, None, :]
        valid = kv_range < nonpad_kv_seqlen[:, None, None, None]
        bias = bias + np.where(valid, 0.0, -np.inf)
    if spec.is_causal:
        kv_range = np.arange(spec.total_seq)[None, :]
        q_range = np.arange(spec.q_seq)[:, None] + spec.past_seq
        causal_mask = kv_range > q_range
        bias = bias + np.where(causal_mask, -np.inf, 0.0)[
            None, None, :, :
        ]
    scores_with_bias = scores + bias
    if spec.softcap != 0.0:
        scores_softcap = spec.softcap * np.tanh(scores_with_bias / spec.softcap)
    else:
        scores_softcap = scores_with_bias
    max_scores = np.max(scores_softcap, axis=-1, keepdims=True)
    weights = np.exp(scores_softcap - max_scores)
    weights /= np.sum(weights, axis=-1, keepdims=True)
    output = np.matmul(weights, value_total_expanded)
    if spec.q_rank == 3:
        output = output.transpose(0, 2, 1, 3).reshape(
            spec.batch, spec.q_seq, spec.q_heads * spec.v_head_size
        )
    qk_output = None
    if spec.qk_matmul_output_mode == 0:
        qk_output = scores
    elif spec.qk_matmul_output_mode == 1:
        qk_output = scores_with_bias
    elif spec.qk_matmul_output_mode == 2:
        qk_output = scores_softcap
    else:
        qk_output = weights
    return output, key_total, value_total, qk_output


def _apply_conv(
    spec: ConvSpec,
    data: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray | None,
) -> np.ndarray:
    output = np.zeros(
        (spec.batch, spec.out_channels, *spec.out_spatial),
        dtype=data.dtype,
    )
    pad_begin = spec.pads[: spec.spatial_rank]
    group_in_channels = spec.in_channels // spec.group
    group_out_channels = spec.out_channels // spec.group
    for n in range(spec.batch):
        for g in range(spec.group):
            oc_base = g * group_out_channels
            ic_base = g * group_in_channels
            for oc in range(group_out_channels):
                oc_global = oc_base + oc
                base = bias[oc_global] if bias is not None else 0.0
                for out_index in np.ndindex(*spec.out_spatial):
                    acc = base
                    for ic in range(group_in_channels):
                        ic_global = ic_base + ic
                        for kernel_index in np.ndindex(*spec.kernel_shape):
                            in_index = []
                            valid = True
                            for (
                                out_dim,
                                kernel_dim,
                                stride,
                                dilation,
                                pad,
                                in_size,
                            ) in zip(
                                out_index,
                                kernel_index,
                                spec.strides,
                                spec.dilations,
                                pad_begin,
                                spec.in_spatial,
                            ):
                                in_dim = out_dim * stride + kernel_dim * dilation - pad
                                if in_dim < 0 or in_dim >= in_size:
                                    valid = False
                                    break
                                in_index.append(in_dim)
                            if not valid:
                                continue
                            acc += data[
                                (n, ic_global, *in_index)
                            ] * weights[(oc_global, ic, *kernel_index)]
                    output[(n, oc_global, *out_index)] = acc
    return output


def _apply_lrn(spec: LrnSpec, data: np.ndarray) -> np.ndarray:
    output = np.empty_like(data)
    spatial_shape = spec.shape[2:]
    spatial_indices = [()]
    if spatial_shape:
        spatial_indices = list(np.ndindex(*spatial_shape))
    for n in range(spec.shape[0]):
        for c in range(spec.channels):
            start = max(0, c - spec.half)
            end = min(spec.channels - 1, c + spec.half)
            for index in spatial_indices:
                sum_val = 0.0
                for i in range(start, end + 1):
                    value = data[(n, i, *index)]
                    sum_val += value * value
                scale = spec.bias + (spec.alpha / spec.size) * sum_val
                output[(n, c, *index)] = data[(n, c, *index)] / math.pow(
                    scale, spec.beta
                )
    return output


def _apply_average_pool(op: AveragePoolOp, data: np.ndarray) -> np.ndarray:
    output = np.zeros((op.batch, op.channels, op.out_h, op.out_w), dtype=data.dtype)
    for n in range(op.batch):
        for c in range(op.channels):
            for oh in range(op.out_h):
                for ow in range(op.out_w):
                    acc = 0.0
                    count = 0
                    for kh in range(op.kernel_h):
                        ih = oh * op.stride_h + kh - op.pad_top
                        if ih < 0 or ih >= op.in_h:
                            if op.count_include_pad:
                                count += op.kernel_w
                            continue
                        for kw in range(op.kernel_w):
                            iw = ow * op.stride_w + kw - op.pad_left
                            if iw < 0 or iw >= op.in_w:
                                if op.count_include_pad:
                                    count += 1
                                continue
                            acc += data[n, c, ih, iw]
                            count += 1
                    output[n, c, oh, ow] = (
                        0.0 if count == 0 else acc / float(count)
                    )
    return output


def _maxpool_min_value(dtype: np.dtype) -> float | int:
    if np.issubdtype(dtype, np.floating):
        return -np.inf
    if np.issubdtype(dtype, np.integer):
        return np.iinfo(dtype).min
    raise UnsupportedOpError("MaxPool supports numeric inputs only")


def _apply_maxpool(spec: MaxPoolSpec, data: np.ndarray) -> np.ndarray:
    min_value = _maxpool_min_value(data.dtype)
    output = np.full(
        (spec.batch, spec.channels, *spec.out_spatial),
        min_value,
        dtype=data.dtype,
    )
    pad_begin = spec.pads[: spec.spatial_rank]
    for n in range(spec.batch):
        for c in range(spec.channels):
            for out_index in np.ndindex(*spec.out_spatial):
                max_value = min_value
                for kernel_index in np.ndindex(*spec.kernel_shape):
                    in_index = []
                    valid = True
                    for out_dim, kernel_dim, stride, dilation, pad in zip(
                        out_index,
                        kernel_index,
                        spec.strides,
                        spec.dilations,
                        pad_begin,
                    ):
                        idx = out_dim * stride + kernel_dim * dilation - pad
                        if idx < 0 or idx >= spec.in_spatial[len(in_index)]:
                            valid = False
                            break
                        in_index.append(idx)
                    if not valid:
                        continue
                    value = data[(n, c, *in_index)]
                    if value > max_value:
                        max_value = value
                output[(n, c, *out_index)] = max_value
    return output

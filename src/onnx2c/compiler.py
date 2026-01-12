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
from .lowering.average_pool import (
    lower_average_pool,
    lower_global_average_pool,
)
from .lowering.batch_normalization import lower_batch_normalization
from .lowering.constant_of_shape import lower_constant_of_shape
from .lowering.lrn import LrnSpec, resolve_lrn_spec
from .lowering.reduce import (
    REDUCE_KIND_BY_OP,
    REDUCE_OUTPUTS_FLOAT_ONLY,
    resolve_reduce_axes,
)
from .lowering.reshape import lower_reshape
from .lowering.resize import lower_resize
from .lowering.shape import lower_shape
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
            if node.op_type in {"MatMul", "Gemm"}:
                if node.op_type == "Gemm":
                    ops.append(self._lower_gemm_op(graph, node))
                else:
                    ops.append(self._lower_matmul_op(graph, node))
                continue
            if node.op_type == "Concat":
                ops.append(self._lower_concat_op(graph, node))
                continue
            if node.op_type == "Attention":
                ops.append(self._lower_attention_op(graph, node))
                continue
            if node.op_type == "Conv":
                ops.append(self._lower_conv_op(graph, node))
                continue
            if node.op_type == "MaxPool":
                ops.append(self._lower_maxpool_op(graph, node))
                continue
            if node.op_type == "Transpose":
                ops.append(self._lower_transpose_op(graph, node))
                continue
            if node.op_type == "Softmax":
                ops.append(self._lower_softmax_op(graph, node))
                continue
            if node.op_type == "LogSoftmax":
                ops.append(self._lower_logsoftmax_op(graph, node))
                continue
            if node.op_type == "Dropout":
                ops.append(self._lower_dropout_op(graph, node))
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
                    spec = _resolve_gemm_spec(graph, node, op_dtype)
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
                        _validate_gemm_bias_shape(
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
                spec = _resolve_attention_spec(graph, node, op_dtype)
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
                spec = _resolve_conv_spec(graph, node)
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
                spec = _resolve_maxpool_spec(graph, node)
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

    def _lower_gemm_op(self, graph: Graph, node: Node) -> GemmOp:
        op_dtype = _node_dtype(graph, node, *node.inputs, *node.outputs)
        spec = _resolve_gemm_spec(graph, node, op_dtype)
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

    def _lower_transpose_op(self, graph: Graph, node: Node) -> TransposeOp:
        if len(node.inputs) != 1 or len(node.outputs) != 1:
            raise UnsupportedOpError("Transpose must have 1 input and 1 output")
        input_shape = _value_shape(graph, node.inputs[0], node)
        output_shape = _value_shape(graph, node.outputs[0], node)
        perm = node.attrs.get("perm")
        if perm is None:
            perm = tuple(reversed(range(len(input_shape))))
        else:
            perm = tuple(int(axis) for axis in perm)
        if len(perm) != len(input_shape):
            raise ShapeInferenceError(
                "Transpose perm must match input rank, "
                f"got perm {perm} for shape {input_shape}"
            )
        if set(perm) != set(range(len(input_shape))):
            raise UnsupportedOpError(
                f"Transpose perm must be a permutation, got {perm}"
            )
        expected_shape = tuple(input_shape[axis] for axis in perm)
        if output_shape != expected_shape:
            raise ShapeInferenceError(
                "Transpose output shape must match permuted input shape, "
                f"expected {expected_shape}, got {output_shape}"
            )
        op_dtype = _node_dtype(graph, node, *node.inputs, *node.outputs)
        return TransposeOp(
            input0=node.inputs[0],
            output=node.outputs[0],
            perm=perm,
            input_shape=input_shape,
            output_shape=output_shape,
            dtype=op_dtype,
        )

    def _lower_softmax_op(self, graph: Graph, node: Node) -> SoftmaxOp:
        if len(node.inputs) != 1 or len(node.outputs) != 1:
            raise UnsupportedOpError("Softmax must have 1 input and 1 output")
        op_dtype = _node_dtype(graph, node, *node.inputs, *node.outputs)
        if op_dtype not in {"float", "double"}:
            raise UnsupportedOpError("Softmax supports float and double inputs only")
        input_shape = _value_shape(graph, node.inputs[0], node)
        output_shape = _value_shape(graph, node.outputs[0], node)
        if input_shape != output_shape:
            raise ShapeInferenceError(
                f"Softmax output shape must be {input_shape}, got {output_shape}"
            )
        axis = _normalize_axis(
            int(node.attrs.get("axis", -1)),
            input_shape,
            node,
        )
        outer = _shape_product(input_shape[:axis]) if axis > 0 else 1
        axis_size = input_shape[axis]
        inner = (
            _shape_product(input_shape[axis + 1 :])
            if axis + 1 < len(input_shape)
            else 1
        )
        return SoftmaxOp(
            input0=node.inputs[0],
            output=node.outputs[0],
            outer=outer,
            axis_size=axis_size,
            inner=inner,
            axis=axis,
            shape=input_shape,
            dtype=op_dtype,
        )

    def _lower_logsoftmax_op(self, graph: Graph, node: Node) -> LogSoftmaxOp:
        if len(node.inputs) != 1 or len(node.outputs) != 1:
            raise UnsupportedOpError("LogSoftmax must have 1 input and 1 output")
        op_dtype = _node_dtype(graph, node, *node.inputs, *node.outputs)
        if op_dtype not in {"float", "double"}:
            raise UnsupportedOpError(
                "LogSoftmax supports float and double inputs only"
            )
        input_shape = _value_shape(graph, node.inputs[0], node)
        output_shape = _value_shape(graph, node.outputs[0], node)
        if input_shape != output_shape:
            raise ShapeInferenceError(
                f"LogSoftmax output shape must be {input_shape}, got {output_shape}"
            )
        axis = _normalize_axis(
            int(node.attrs.get("axis", -1)),
            input_shape,
            node,
        )
        outer = _shape_product(input_shape[:axis]) if axis > 0 else 1
        axis_size = input_shape[axis]
        inner = (
            _shape_product(input_shape[axis + 1 :])
            if axis + 1 < len(input_shape)
            else 1
        )
        return LogSoftmaxOp(
            input0=node.inputs[0],
            output=node.outputs[0],
            outer=outer,
            axis_size=axis_size,
            inner=inner,
            axis=axis,
            shape=input_shape,
            dtype=op_dtype,
        )

    def _lower_dropout_op(self, graph: Graph, node: Node) -> ReshapeOp:
        if len(node.outputs) not in {1, 2} or len(node.inputs) != 1:
            raise UnsupportedOpError(
                "Dropout supports only the data input and 1 or 2 outputs"
            )
        if len(node.outputs) == 2 and _is_value_used(graph, node.outputs[1]):
            raise UnsupportedOpError("Dropout mask output is not supported")
        input_shape = _value_shape(graph, node.inputs[0], node)
        output_shape = _value_shape(graph, node.outputs[0], node)
        if input_shape != output_shape:
            raise ShapeInferenceError(
                "Dropout output shape must match input shape, "
                f"got {output_shape} for input {input_shape}"
            )
        input_dtype = _value_dtype(graph, node.inputs[0], node)
        output_dtype = _value_dtype(graph, node.outputs[0], node)
        if input_dtype != output_dtype:
            raise UnsupportedOpError(
                "Dropout expects matching input/output dtypes, "
                f"got {input_dtype} and {output_dtype}"
            )
        return ReshapeOp(
            input0=node.inputs[0],
            output=node.outputs[0],
            input_shape=input_shape,
            output_shape=output_shape,
            dtype=input_dtype,
        )

    def _lower_attention_op(self, graph: Graph, node: Node) -> AttentionOp:
        input_q = node.inputs[0]
        input_k = node.inputs[1]
        input_v = node.inputs[2]
        output_y = node.outputs[0]
        op_dtype = _node_dtype(graph, node, input_q, input_k, input_v, output_y)
        spec = _resolve_attention_spec(graph, node, op_dtype)
        input_mask = _optional_name(node.inputs, 3)
        input_past_key = _optional_name(node.inputs, 4)
        input_past_value = _optional_name(node.inputs, 5)
        input_nonpad = _optional_name(node.inputs, 6)
        output_present_key = _optional_name(node.outputs, 1)
        output_present_value = _optional_name(node.outputs, 2)
        output_qk_matmul = _optional_name(node.outputs, 3)
        return AttentionOp(
            input_q=input_q,
            input_k=input_k,
            input_v=input_v,
            input_attn_mask=input_mask,
            input_past_key=input_past_key,
            input_past_value=input_past_value,
            input_nonpad_kv_seqlen=input_nonpad,
            output=output_y,
            output_present_key=output_present_key,
            output_present_value=output_present_value,
            output_qk_matmul=output_qk_matmul,
            batch=spec.batch,
            q_heads=spec.q_heads,
            kv_heads=spec.kv_heads,
            q_seq=spec.q_seq,
            kv_seq=spec.kv_seq,
            total_seq=spec.total_seq,
            past_seq=spec.past_seq,
            qk_head_size=spec.qk_head_size,
            v_head_size=spec.v_head_size,
            q_hidden_size=spec.q_hidden_size,
            k_hidden_size=spec.k_hidden_size,
            v_hidden_size=spec.v_hidden_size,
            scale=spec.scale,
            is_causal=spec.is_causal,
            softcap=spec.softcap,
            qk_matmul_output_mode=spec.qk_matmul_output_mode,
            q_rank=spec.q_rank,
            k_rank=spec.k_rank,
            v_rank=spec.v_rank,
            output_rank=spec.output_rank,
            mask_shape=spec.mask_shape,
            mask_is_bool=spec.mask_is_bool,
            mask_rank=spec.mask_rank,
            mask_broadcast_batch=spec.mask_broadcast_batch,
            mask_broadcast_heads=spec.mask_broadcast_heads,
            mask_broadcast_q_seq=spec.mask_broadcast_q_seq,
            mask_q_seq=spec.mask_q_seq,
            mask_kv_seq=spec.mask_kv_seq,
            head_group_size=spec.head_group_size,
            dtype=op_dtype,
        )

    def _lower_conv_op(self, graph: Graph, node: Node) -> ConvOp:
        if len(node.inputs) not in {2, 3} or len(node.outputs) != 1:
            raise UnsupportedOpError("Conv must have 2 or 3 inputs and 1 output")
        op_dtype = _node_dtype(graph, node, *node.inputs, *node.outputs)
        if op_dtype not in {"float", "double"}:
            raise UnsupportedOpError("Conv supports float and double inputs only")
        spec = _resolve_conv_spec(graph, node)
        return ConvOp(
            input0=node.inputs[0],
            weights=node.inputs[1],
            bias=node.inputs[2] if len(node.inputs) == 3 else None,
            output=node.outputs[0],
            batch=spec.batch,
            in_channels=spec.in_channels,
            out_channels=spec.out_channels,
            spatial_rank=spec.spatial_rank,
            in_spatial=spec.in_spatial,
            out_spatial=spec.out_spatial,
            kernel_shape=spec.kernel_shape,
            strides=spec.strides,
            pads=spec.pads,
            dilations=spec.dilations,
            group=spec.group,
            dtype=op_dtype,
        )

    def _lower_maxpool_op(self, graph: Graph, node: Node) -> MaxPoolOp:
        if len(node.inputs) != 1 or len(node.outputs) != 1:
            raise UnsupportedOpError("MaxPool must have 1 input and 1 output")
        op_dtype = _node_dtype(graph, node, *node.inputs, *node.outputs)
        if op_dtype == "bool":
            raise UnsupportedOpError("MaxPool supports numeric inputs only")
        spec = _resolve_maxpool_spec(graph, node)
        return MaxPoolOp(
            input0=node.inputs[0],
            output=node.outputs[0],
            batch=spec.batch,
            channels=spec.channels,
            spatial_rank=spec.spatial_rank,
            in_spatial=spec.in_spatial,
            out_spatial=spec.out_spatial,
            kernel_shape=spec.kernel_shape,
            strides=spec.strides,
            pads=spec.pads,
            dilations=spec.dilations,
            ceil_mode=spec.ceil_mode,
            dtype=op_dtype,
        )

    def _lower_concat_op(self, graph: Graph, node: Node) -> ConcatOp:
        if len(node.inputs) < 1 or len(node.outputs) != 1:
            raise UnsupportedOpError("Concat must have at least 1 input and 1 output")
        op_dtype = _node_dtype(graph, node, *node.inputs, *node.outputs)
        output_shape = _value_shape(graph, node.outputs[0], node)
        input_shapes = tuple(
            _value_shape(graph, name, node) for name in node.inputs
        )
        ranks = {len(shape) for shape in input_shapes}
        if len(ranks) != 1:
            raise ShapeInferenceError(
                f"Concat inputs must have matching ranks, got {input_shapes}"
            )
        rank = ranks.pop()
        axis = int(node.attrs.get("axis", 0))
        if axis < 0:
            axis += rank
        if axis < 0 or axis >= rank:
            raise ShapeInferenceError(
                f"Concat axis out of range for rank {rank}: {axis}"
            )
        base_shape = list(input_shapes[0])
        axis_dim = 0
        for shape in input_shapes:
            if len(shape) != rank:
                raise ShapeInferenceError(
                    f"Concat inputs must have matching ranks, got {input_shapes}"
                )
            for dim_index, dim in enumerate(shape):
                if dim_index == axis:
                    continue
                if dim != base_shape[dim_index]:
                    raise ShapeInferenceError(
                        "Concat inputs must match on non-axis dimensions, "
                        f"got {input_shapes}"
                    )
            axis_dim += shape[axis]
        base_shape[axis] = axis_dim
        expected_output_shape = tuple(base_shape)
        if output_shape != expected_output_shape:
            raise ShapeInferenceError(
                "Concat output shape must be "
                f"{expected_output_shape}, got {output_shape}"
            )
        return ConcatOp(
            inputs=node.inputs,
            output=node.outputs[0],
            axis=axis,
            input_shapes=input_shapes,
            output_shape=output_shape,
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


@dataclass(frozen=True)
class _GemmSpec:
    m: int
    n: int
    k: int
    alpha: float | int
    beta: float | int
    trans_a: bool
    trans_b: bool
    c_shape: tuple[int, ...] | None


def _resolve_gemm_spec(graph: Graph, node: Node, dtype: str) -> _GemmSpec:
    if len(node.inputs) not in {2, 3} or len(node.outputs) != 1:
        raise UnsupportedOpError("Gemm must have 2 or 3 inputs and 1 output")
    alpha, beta, trans_a, trans_b = _resolve_gemm_attrs(node, dtype)
    input0_shape = _value_shape(graph, node.inputs[0], node)
    input1_shape = _value_shape(graph, node.inputs[1], node)
    if len(input0_shape) != 2 or len(input1_shape) != 2:
        raise UnsupportedOpError(
            f"Gemm supports 2D inputs only, got {input0_shape} x {input1_shape}"
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
        c_shape = _validate_gemm_bias_shape((m, n), bias_shape, node)
    return _GemmSpec(
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
    if dtype not in {"float", "double"}:
        alpha_int = int(alpha)
        beta_int = int(beta)
        if alpha != alpha_int or beta != beta_int:
            raise UnsupportedOpError(
                "Gemm alpha and beta must be integers for non-float inputs"
            )
        alpha = alpha_int
        beta = beta_int
    return alpha, beta, bool(trans_a), bool(trans_b)


def _validate_gemm_bias_shape(
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


@dataclass(frozen=True)
class _AttentionSpec:
    batch: int
    q_heads: int
    kv_heads: int
    q_seq: int
    kv_seq: int
    total_seq: int
    past_seq: int
    qk_head_size: int
    v_head_size: int
    q_hidden_size: int | None
    k_hidden_size: int | None
    v_hidden_size: int | None
    scale: float
    is_causal: bool
    softcap: float
    qk_matmul_output_mode: int
    q_rank: int
    k_rank: int
    v_rank: int
    output_rank: int
    mask_shape: tuple[int, ...] | None
    mask_is_bool: bool
    mask_rank: int | None
    mask_broadcast_batch: bool
    mask_broadcast_heads: bool
    mask_broadcast_q_seq: bool
    mask_q_seq: int | None
    mask_kv_seq: int | None
    head_group_size: int
    has_attn_mask: bool
    has_past: bool
    has_present: bool
    has_nonpad: bool


def _optional_name(names: Sequence[str], index: int) -> str | None:
    if index >= len(names):
        return None
    name = names[index]
    return name or None


def _resolve_attention_spec(
    graph: Graph, node: Node, dtype: str
) -> _AttentionSpec:
    if dtype not in {"float", "double"}:
        raise UnsupportedOpError("Unsupported op Attention")
    if len(node.inputs) < 3 or len(node.outputs) < 1:
        raise UnsupportedOpError("Unsupported op Attention")
    supported_attrs = {
        "scale",
        "is_causal",
        "q_num_heads",
        "kv_num_heads",
        "softmax_precision",
        "softcap",
        "qk_matmul_output_mode",
    }
    if set(node.attrs) - supported_attrs:
        raise UnsupportedOpError("Unsupported op Attention")
    q_shape = _value_shape(graph, node.inputs[0], node)
    k_shape = _value_shape(graph, node.inputs[1], node)
    v_shape = _value_shape(graph, node.inputs[2], node)
    q_rank = len(q_shape)
    k_rank = len(k_shape)
    v_rank = len(v_shape)
    if q_rank not in {3, 4} or k_rank not in {3, 4} or v_rank not in {3, 4}:
        raise UnsupportedOpError("Unsupported op Attention")
    if q_rank != k_rank or q_rank != v_rank:
        raise UnsupportedOpError("Unsupported op Attention")
    batch = q_shape[0]
    if batch != k_shape[0] or batch != v_shape[0]:
        raise ShapeInferenceError("Attention batch sizes must match")
    q_hidden_size = None
    k_hidden_size = None
    v_hidden_size = None
    if q_rank == 3:
        q_heads = node.attrs.get("q_num_heads")
        kv_heads = node.attrs.get("kv_num_heads")
        if q_heads is None or kv_heads is None:
            raise UnsupportedOpError("Unsupported op Attention")
        q_heads = int(q_heads)
        kv_heads = int(kv_heads)
        q_seq = q_shape[1]
        kv_seq = k_shape[1]
        if kv_seq != v_shape[1]:
            raise ShapeInferenceError(
                "Attention key/value sequence lengths must match"
            )
        q_hidden_size = q_shape[2]
        k_hidden_size = k_shape[2]
        v_hidden_size = v_shape[2]
        if q_hidden_size % q_heads != 0:
            raise ShapeInferenceError(
                "Attention query hidden size must be divisible by q_num_heads"
            )
        if k_hidden_size % kv_heads != 0:
            raise ShapeInferenceError(
                "Attention key hidden size must be divisible by kv_num_heads"
            )
        if v_hidden_size % kv_heads != 0:
            raise ShapeInferenceError(
                "Attention value hidden size must be divisible by kv_num_heads"
            )
        qk_head_size = q_hidden_size // q_heads
        k_head_size = k_hidden_size // kv_heads
        v_head_size = v_hidden_size // kv_heads
        if qk_head_size != k_head_size:
            raise ShapeInferenceError("Attention Q/K head sizes must match")
    else:
        q_heads = q_shape[1]
        kv_heads = k_shape[1]
        if kv_heads != v_shape[1]:
            raise ShapeInferenceError("Attention key/value head counts must match")
        q_seq = q_shape[2]
        kv_seq = k_shape[2]
        if kv_seq != v_shape[2]:
            raise ShapeInferenceError(
                "Attention key/value sequence lengths must match"
            )
        qk_head_size = q_shape[3]
        k_head_size = k_shape[3]
        v_head_size = v_shape[3]
        if qk_head_size != k_head_size:
            raise ShapeInferenceError("Attention Q/K head sizes must match")
        attr_q_heads = node.attrs.get("q_num_heads")
        attr_kv_heads = node.attrs.get("kv_num_heads")
        if attr_q_heads is not None and int(attr_q_heads) != q_heads:
            raise ShapeInferenceError(
                "Attention q_num_heads must match query head dimension"
            )
        if attr_kv_heads is not None and int(attr_kv_heads) != kv_heads:
            raise ShapeInferenceError(
                "Attention kv_num_heads must match key/value head dimension"
            )
    if q_heads < kv_heads or q_heads % kv_heads != 0:
        raise ShapeInferenceError(
            "Attention requires q_num_heads to be a multiple of kv_num_heads"
        )
    head_group_size = q_heads // kv_heads
    past_key_name = _optional_name(node.inputs, 4)
    past_value_name = _optional_name(node.inputs, 5)
    has_past = past_key_name is not None or past_value_name is not None
    if has_past and (past_key_name is None or past_value_name is None):
        raise UnsupportedOpError(
            "Attention expects both past_key and past_value if either is provided"
        )
    past_seq = 0
    if has_past:
        past_key_shape = _value_shape(graph, past_key_name, node)
        past_value_shape = _value_shape(graph, past_value_name, node)
        if len(past_key_shape) != 4 or len(past_value_shape) != 4:
            raise ShapeInferenceError("Attention past key/value must be 4D")
        if (
            past_key_shape[0] != batch
            or past_value_shape[0] != batch
            or past_key_shape[1] != kv_heads
            or past_value_shape[1] != kv_heads
        ):
            raise ShapeInferenceError(
                "Attention past key/value batch/head sizes must match"
            )
        if past_key_shape[3] != qk_head_size:
            raise ShapeInferenceError(
                "Attention past key head size must match key head size"
            )
        if past_value_shape[3] != v_head_size:
            raise ShapeInferenceError(
                "Attention past value head size must match value head size"
            )
        past_seq = past_key_shape[2]
    total_seq = kv_seq + past_seq
    output_shape = _value_shape(graph, node.outputs[0], node)
    output_rank = len(output_shape)
    if q_rank == 3:
        expected_output_shape = (
            batch,
            q_seq,
            q_heads * v_head_size,
        )
    else:
        expected_output_shape = (batch, q_heads, q_seq, v_head_size)
    if output_shape != expected_output_shape:
        raise ShapeInferenceError(
            "Attention output shape must be "
            f"{expected_output_shape}, got {output_shape}"
        )
    present_key_name = _optional_name(node.outputs, 1)
    present_value_name = _optional_name(node.outputs, 2)
    has_present = present_key_name is not None or present_value_name is not None
    if has_present and (present_key_name is None or present_value_name is None):
        raise UnsupportedOpError(
            "Attention expects both present_key and present_value if either is provided"
        )
    if has_present and not has_past:
        raise UnsupportedOpError(
            "Attention present outputs require past key/value inputs"
        )
    if has_present:
        present_key_shape = _value_shape(graph, present_key_name, node)
        present_value_shape = _value_shape(graph, present_value_name, node)
        expected_present_key = (batch, kv_heads, total_seq, qk_head_size)
        expected_present_value = (batch, kv_heads, total_seq, v_head_size)
        if present_key_shape != expected_present_key:
            raise ShapeInferenceError(
                "Attention present key shape must be "
                f"{expected_present_key}, got {present_key_shape}"
            )
        if present_value_shape != expected_present_value:
            raise ShapeInferenceError(
                "Attention present value shape must be "
                f"{expected_present_value}, got {present_value_shape}"
            )
    qk_matmul_output_name = _optional_name(node.outputs, 3)
    if qk_matmul_output_name is not None:
        qk_shape = _value_shape(graph, qk_matmul_output_name, node)
        expected_qk_shape = (batch, q_heads, q_seq, total_seq)
        if qk_shape != expected_qk_shape:
            raise ShapeInferenceError(
                "Attention qk_matmul_output shape must be "
                f"{expected_qk_shape}, got {qk_shape}"
            )
    attn_mask_name = _optional_name(node.inputs, 3)
    mask_shape = None
    mask_rank = None
    mask_is_bool = False
    mask_broadcast_batch = False
    mask_broadcast_heads = True
    mask_broadcast_q_seq = False
    mask_q_seq = None
    mask_kv_seq = None
    has_attn_mask = attn_mask_name is not None
    if has_attn_mask:
        mask_shape = _value_shape(graph, attn_mask_name, node)
        mask_rank = len(mask_shape)
        if mask_rank not in {2, 3, 4}:
            raise ShapeInferenceError("Attention mask must be 2D/3D/4D")
        mask_dtype = _value_dtype(graph, attn_mask_name, node)
        if mask_dtype == "bool":
            mask_is_bool = True
        elif mask_dtype != dtype:
            raise UnsupportedOpError(
                "Attention mask must be bool or match attention dtype"
            )
        if mask_rank == 2:
            mask_q_seq, mask_kv_seq = mask_shape
            mask_broadcast_batch = True
            mask_broadcast_heads = True
            mask_broadcast_q_seq = mask_q_seq == 1
            if mask_q_seq not in {1, q_seq}:
                raise ShapeInferenceError(
                    "Attention mask sequence length must match query length"
                )
        elif mask_rank == 3:
            mask_batch, mask_q_seq, mask_kv_seq = mask_shape
            mask_broadcast_batch = mask_batch == 1
            mask_broadcast_heads = True
            mask_broadcast_q_seq = mask_q_seq == 1
            if mask_batch not in {1, batch}:
                raise ShapeInferenceError(
                    "Attention mask batch dimension must match batch size"
                )
            if mask_q_seq not in {1, q_seq}:
                raise ShapeInferenceError(
                    "Attention mask sequence length must match query length"
                )
        else:
            mask_batch, mask_heads, mask_q_seq, mask_kv_seq = mask_shape
            mask_broadcast_batch = mask_batch == 1
            mask_broadcast_heads = mask_heads == 1
            mask_broadcast_q_seq = mask_q_seq == 1
            if mask_batch not in {1, batch}:
                raise ShapeInferenceError(
                    "Attention mask batch dimension must match batch size"
                )
            if mask_heads not in {1, q_heads}:
                raise ShapeInferenceError(
                    "Attention mask head dimension must match q_num_heads"
                )
            if mask_q_seq not in {1, q_seq}:
                raise ShapeInferenceError(
                    "Attention mask sequence length must match query length"
                )
        if mask_kv_seq is None:
            raise ShapeInferenceError("Attention mask must include kv sequence")
        if mask_kv_seq > total_seq:
            raise ShapeInferenceError(
                "Attention mask kv sequence length exceeds total sequence length"
            )
    nonpad_name = _optional_name(node.inputs, 6)
    has_nonpad = nonpad_name is not None
    if has_nonpad:
        if has_past or has_present:
            raise UnsupportedOpError(
                "Attention nonpad_kv_seqlen is not supported with KV cache"
            )
        nonpad_shape = _value_shape(graph, nonpad_name, node)
        if nonpad_shape != (batch,):
            raise ShapeInferenceError(
                "Attention nonpad_kv_seqlen must have shape (batch,)"
            )
        nonpad_dtype = _value_dtype(graph, nonpad_name, node)
        if nonpad_dtype != "int64":
            raise UnsupportedOpError(
                "Attention nonpad_kv_seqlen must be int64"
            )
    scale = float(node.attrs.get("scale", 1.0 / math.sqrt(qk_head_size)))
    softcap = float(node.attrs.get("softcap", 0.0))
    is_causal = int(node.attrs.get("is_causal", 0))
    if is_causal not in (0, 1):
        raise UnsupportedOpError("Unsupported op Attention")
    qk_matmul_output_mode = int(node.attrs.get("qk_matmul_output_mode", 0))
    if qk_matmul_output_mode not in {0, 1, 2, 3}:
        raise UnsupportedOpError("Unsupported op Attention")
    return _AttentionSpec(
        batch=batch,
        q_heads=q_heads,
        kv_heads=kv_heads,
        q_seq=q_seq,
        kv_seq=kv_seq,
        total_seq=total_seq,
        past_seq=past_seq,
        qk_head_size=qk_head_size,
        v_head_size=v_head_size,
        q_hidden_size=q_hidden_size,
        k_hidden_size=k_hidden_size,
        v_hidden_size=v_hidden_size,
        scale=scale,
        is_causal=bool(is_causal),
        softcap=softcap,
        qk_matmul_output_mode=qk_matmul_output_mode,
        q_rank=q_rank,
        k_rank=k_rank,
        v_rank=v_rank,
        output_rank=output_rank,
        mask_shape=mask_shape,
        mask_is_bool=mask_is_bool,
        mask_rank=mask_rank,
        mask_broadcast_batch=mask_broadcast_batch,
        mask_broadcast_heads=mask_broadcast_heads,
        mask_broadcast_q_seq=mask_broadcast_q_seq,
        mask_q_seq=mask_q_seq,
        mask_kv_seq=mask_kv_seq,
        head_group_size=head_group_size,
        has_attn_mask=has_attn_mask,
        has_past=has_past,
        has_present=has_present,
        has_nonpad=has_nonpad,
    )


def _apply_attention(
    spec: _AttentionSpec,
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


@dataclass(frozen=True)
class _ConvSpec:
    batch: int
    in_channels: int
    out_channels: int
    spatial_rank: int
    in_spatial: tuple[int, ...]
    out_spatial: tuple[int, ...]
    kernel_shape: tuple[int, ...]
    strides: tuple[int, ...]
    pads: tuple[int, ...]
    dilations: tuple[int, ...]
    group: int


def _resolve_conv_spec(graph: Graph, node: Node) -> _ConvSpec:
    if len(node.inputs) not in {2, 3} or len(node.outputs) != 1:
        raise UnsupportedOpError("Conv must have 2 or 3 inputs and 1 output")
    supported_attrs = {
        "auto_pad",
        "dilations",
        "group",
        "kernel_shape",
        "pads",
        "strides",
    }
    if set(node.attrs) - supported_attrs:
        raise UnsupportedOpError("Conv has unsupported attributes")
    input_shape = _value_shape(graph, node.inputs[0], node)
    weight_shape = _value_shape(graph, node.inputs[1], node)
    if len(input_shape) < 3:
        raise UnsupportedOpError("Conv expects NCHW inputs with spatial dims")
    spatial_rank = len(input_shape) - 2
    if spatial_rank not in {1, 2, 3}:
        raise UnsupportedOpError("Conv supports 1D/2D/3D inputs only")
    if len(weight_shape) != spatial_rank + 2:
        raise UnsupportedOpError("Conv weight rank must match spatial rank")
    batch, in_channels = input_shape[0], input_shape[1]
    in_spatial = input_shape[2:]
    out_channels, weight_in_channels, *kernel_shape = weight_shape
    kernel_shape = node.attrs.get("kernel_shape")
    if kernel_shape is not None:
        kernel_shape = tuple(int(value) for value in kernel_shape)
        if len(kernel_shape) != spatial_rank:
            raise UnsupportedOpError(
                "Conv kernel_shape rank must match input spatial rank"
            )
        if kernel_shape != tuple(weight_shape[2:]):
            raise ShapeInferenceError(
                "Conv kernel_shape must match weights, "
                f"got {kernel_shape} and {tuple(weight_shape[2:])}"
            )
    else:
        kernel_shape = tuple(weight_shape[2:])
    group = int(node.attrs.get("group", 1))
    if group <= 0:
        raise UnsupportedOpError("Conv expects group >= 1")
    if in_channels % group != 0 or out_channels % group != 0:
        raise ShapeInferenceError(
            "Conv expects group to evenly divide in/out channels, "
            f"got group={group}, in_channels={in_channels}, "
            f"out_channels={out_channels}"
        )
    if weight_in_channels != in_channels // group:
        raise ShapeInferenceError(
            "Conv input channels must match weight channels, "
            f"got {in_channels} and {weight_in_channels * group}"
        )
    if len(node.inputs) == 3:
        bias_shape = _value_shape(graph, node.inputs[2], node)
        if bias_shape != (out_channels,):
            raise ShapeInferenceError(
                f"Conv bias shape must be {(out_channels,)}, got {bias_shape}"
            )
    strides = tuple(
        int(value) for value in node.attrs.get("strides", (1,) * spatial_rank)
    )
    if len(strides) != spatial_rank:
        raise UnsupportedOpError("Conv stride rank mismatch")
    dilations = tuple(
        int(value) for value in node.attrs.get("dilations", (1,) * spatial_rank)
    )
    if len(dilations) != spatial_rank:
        raise UnsupportedOpError("Conv dilation rank mismatch")
    pads = tuple(
        int(value)
        for value in node.attrs.get("pads", (0,) * (2 * spatial_rank))
    )
    if len(pads) != 2 * spatial_rank:
        raise UnsupportedOpError("Conv pads rank mismatch")
    auto_pad = node.attrs.get("auto_pad", b"NOTSET")
    if isinstance(auto_pad, bytes):
        auto_pad = auto_pad.decode("utf-8", errors="ignore")
    if auto_pad in ("", "NOTSET"):
        pad_begin = pads[:spatial_rank]
        pad_end = pads[spatial_rank:]
    elif auto_pad == "VALID":
        pad_begin = (0,) * spatial_rank
        pad_end = (0,) * spatial_rank
    elif auto_pad in {"SAME_UPPER", "SAME_LOWER"}:
        pad_begin = []
        pad_end = []
        for dim, stride, dilation, kernel in zip(
            in_spatial, strides, dilations, kernel_shape
        ):
            effective_kernel = dilation * (kernel - 1) + 1
            out_dim = math.ceil(dim / stride)
            pad_needed = max(
                0, (out_dim - 1) * stride + effective_kernel - dim
            )
            if auto_pad == "SAME_UPPER":
                pad_start = pad_needed // 2
            else:
                pad_start = (pad_needed + 1) // 2
            pad_begin.append(pad_start)
            pad_end.append(pad_needed - pad_start)
        pad_begin = tuple(pad_begin)
        pad_end = tuple(pad_end)
    else:
        raise UnsupportedOpError("Conv has unsupported auto_pad mode")
    out_spatial = []
    for dim, stride, dilation, kernel, pad_start, pad_finish in zip(
        in_spatial, strides, dilations, kernel_shape, pad_begin, pad_end
    ):
        effective_kernel = dilation * (kernel - 1) + 1
        out_dim = (dim + pad_start + pad_finish - effective_kernel) // stride + 1
        if out_dim <= 0:
            raise ShapeInferenceError("Conv output shape must be positive")
        out_spatial.append(out_dim)
    output_shape = _value_shape(graph, node.outputs[0], node)
    expected_output_shape = (batch, out_channels, *out_spatial)
    if output_shape != expected_output_shape:
        raise ShapeInferenceError(
            "Conv output shape must be "
            f"{expected_output_shape}, got {output_shape}"
        )
    return _ConvSpec(
        batch=batch,
        in_channels=in_channels,
        out_channels=out_channels,
        spatial_rank=spatial_rank,
        in_spatial=in_spatial,
        out_spatial=tuple(out_spatial),
        kernel_shape=kernel_shape,
        strides=strides,
        pads=(*pad_begin, *pad_end),
        dilations=dilations,
        group=group,
    )


def _apply_conv(
    spec: _ConvSpec,
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


@dataclass(frozen=True)
class _MaxPoolSpec:
    batch: int
    channels: int
    spatial_rank: int
    in_spatial: tuple[int, ...]
    out_spatial: tuple[int, ...]
    kernel_shape: tuple[int, ...]
    strides: tuple[int, ...]
    pads: tuple[int, ...]
    dilations: tuple[int, ...]
    ceil_mode: bool


def _resolve_maxpool_spec(graph: Graph, node: Node) -> _MaxPoolSpec:
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        raise UnsupportedOpError("MaxPool must have 1 input and 1 output")
    supported_attrs = {
        "auto_pad",
        "ceil_mode",
        "dilations",
        "kernel_shape",
        "pads",
        "storage_order",
        "strides",
    }
    if set(node.attrs) - supported_attrs:
        raise UnsupportedOpError("MaxPool has unsupported attributes")
    storage_order = int(node.attrs.get("storage_order", 0))
    if storage_order != 0:
        raise UnsupportedOpError("MaxPool supports storage_order=0 only")
    kernel_shape = node.attrs.get("kernel_shape")
    if kernel_shape is None:
        raise UnsupportedOpError("MaxPool requires kernel_shape")
    kernel_shape = tuple(int(value) for value in kernel_shape)
    input_shape = _value_shape(graph, node.inputs[0], node)
    if len(input_shape) < 3:
        raise UnsupportedOpError("MaxPool expects NCHW inputs with spatial dims")
    spatial_rank = len(input_shape) - 2
    if spatial_rank not in {1, 2, 3}:
        raise UnsupportedOpError("MaxPool supports 1D/2D/3D inputs only")
    if len(kernel_shape) != spatial_rank:
        raise ShapeInferenceError(
            f"MaxPool kernel_shape must have {spatial_rank} dims, got {kernel_shape}"
        )
    strides = tuple(
        int(value) for value in node.attrs.get("strides", (1,) * spatial_rank)
    )
    if len(strides) != spatial_rank:
        raise UnsupportedOpError("MaxPool stride rank mismatch")
    dilations = tuple(
        int(value) for value in node.attrs.get("dilations", (1,) * spatial_rank)
    )
    if len(dilations) != spatial_rank:
        raise UnsupportedOpError("MaxPool dilation rank mismatch")
    pads = tuple(
        int(value)
        for value in node.attrs.get("pads", (0,) * (2 * spatial_rank))
    )
    if len(pads) != 2 * spatial_rank:
        raise UnsupportedOpError("MaxPool pads rank mismatch")
    auto_pad = node.attrs.get("auto_pad", b"NOTSET")
    if isinstance(auto_pad, bytes):
        auto_pad = auto_pad.decode("utf-8", errors="ignore")
    if auto_pad in ("", "NOTSET"):
        pad_begin = pads[:spatial_rank]
        pad_end = pads[spatial_rank:]
    elif auto_pad == "VALID":
        pad_begin = (0,) * spatial_rank
        pad_end = (0,) * spatial_rank
    elif auto_pad in {"SAME_UPPER", "SAME_LOWER"}:
        pad_begin = []
        pad_end = []
        for dim, stride, dilation, kernel in zip(
            input_shape[2:], strides, dilations, kernel_shape
        ):
            effective_kernel = dilation * (kernel - 1) + 1
            out_dim = math.ceil(dim / stride)
            pad_needed = max(
                0, (out_dim - 1) * stride + effective_kernel - dim
            )
            if auto_pad == "SAME_UPPER":
                pad_start = pad_needed // 2
            else:
                pad_start = (pad_needed + 1) // 2
            pad_begin.append(pad_start)
            pad_end.append(pad_needed - pad_start)
        pad_begin = tuple(pad_begin)
        pad_end = tuple(pad_end)
    else:
        raise UnsupportedOpError("MaxPool has unsupported auto_pad mode")
    ceil_mode = int(node.attrs.get("ceil_mode", 0))
    if ceil_mode not in (0, 1):
        raise UnsupportedOpError("MaxPool supports ceil_mode=0 or 1 only")
    batch, channels = input_shape[0], input_shape[1]
    in_spatial = input_shape[2:]
    out_spatial = []
    for dim, stride, dilation, kernel, pad_start, pad_finish in zip(
        in_spatial, strides, dilations, kernel_shape, pad_begin, pad_end
    ):
        effective_kernel = dilation * (kernel - 1) + 1
        numerator = dim + pad_start + pad_finish - effective_kernel
        if ceil_mode:
            out_dim = (numerator + stride - 1) // stride + 1
            if (out_dim - 1) * stride >= dim + pad_start:
                out_dim -= 1
        else:
            out_dim = numerator // stride + 1
        if out_dim <= 0:
            raise ShapeInferenceError("MaxPool output shape must be positive")
        out_spatial.append(out_dim)
    expected_output_shape = (batch, channels, *out_spatial)
    output_shape = _value_shape(graph, node.outputs[0], node)
    if output_shape != expected_output_shape:
        raise ShapeInferenceError(
            "MaxPool output shape must be "
            f"{expected_output_shape}, got {output_shape}"
        )
    pads = (*pad_begin, *pad_end)
    return _MaxPoolSpec(
        batch=batch,
        channels=channels,
        spatial_rank=spatial_rank,
        in_spatial=in_spatial,
        out_spatial=tuple(out_spatial),
        kernel_shape=kernel_shape,
        strides=strides,
        pads=pads,
        dilations=dilations,
        ceil_mode=bool(ceil_mode),
    )


def _maxpool_min_value(dtype: np.dtype) -> float | int:
    if np.issubdtype(dtype, np.floating):
        return -np.inf
    if np.issubdtype(dtype, np.integer):
        return np.iinfo(dtype).min
    raise UnsupportedOpError("MaxPool supports numeric inputs only")


def _apply_maxpool(spec: _MaxPoolSpec, data: np.ndarray) -> np.ndarray:
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

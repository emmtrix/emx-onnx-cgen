from __future__ import annotations

from collections.abc import Callable, Mapping
import math

import numpy as np

from ..dtypes import dtype_info
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from ..lowering.attention import resolve_attention_spec
from ..lowering.average_pool import lower_average_pool, lower_global_average_pool
from ..lowering.batch_normalization import lower_batch_normalization
from ..lowering.concat import lower_concat
from ..lowering.constant_of_shape import lower_constant_of_shape
from ..lowering.conv import resolve_conv_spec
from ..lowering.dropout import lower_dropout
from ..lowering.gemm import resolve_gemm_spec
from ..lowering.logsoftmax import lower_logsoftmax
from ..lowering.negative_log_likelihood_loss import (
    lower_negative_log_likelihood_loss,
)
from ..lowering.softmax_cross_entropy_loss import (
    lower_softmax_cross_entropy_loss,
)
from ..lowering.lstm import ACTIVATION_KIND_BY_NAME, resolve_lstm_spec
from ..lowering.lrn import resolve_lrn_spec
from ..lowering.matmul import lower_matmul
from ..lowering.maxpool import resolve_maxpool_spec
from ..lowering.reduce import (
    REDUCE_KIND_BY_OP,
    REDUCE_OUTPUTS_FLOAT_ONLY,
    resolve_reduce_axes,
)
from ..lowering.reshape import lower_reshape
from ..lowering.shape import lower_shape
from ..lowering.softmax import lower_softmax
from ..lowering.transpose import lower_transpose
from ..lowering.unsqueeze import lower_unsqueeze
from ..lowering.where import lower_where
from ..lowering.registry import resolve_dispatch
from ..lowering.common import node_dtype, optional_name, value_dtype
from ..ops import (
    BINARY_OP_TYPES,
    COMPARE_OP_TYPES,
    UNARY_OP_TYPES,
    apply_binary_op,
    apply_unary_op,
    binary_op_symbol,
    unary_op_symbol,
)
from ..validation import normalize_axis

Handler = Callable[["Evaluator", Node], None]
_EVAL_REGISTRY: dict[str, Handler] = {}


def register_evaluator(op_type: str) -> Callable[[Handler], Handler]:
    def decorator(func: Handler) -> Handler:
        _EVAL_REGISTRY[op_type] = func
        return func

    return decorator


class Evaluator:
    def __init__(self, graph: Graph) -> None:
        self._graph = graph
        self._values: dict[str, np.ndarray] = {}

    @property
    def graph(self) -> Graph:
        return self._graph

    @property
    def values(self) -> dict[str, np.ndarray]:
        return self._values

    def run(self, feeds: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        values = {
            initializer.name: initializer.data
            for initializer in self._graph.initializers
        }
        values.update(feeds)
        self._values = values
        for node in self._graph.nodes:
            self._dispatch(node)
        return {
            output.name: self._values[output.name]
            for output in self._graph.outputs
        }

    def _dispatch(self, node: Node) -> None:
        handler = resolve_dispatch(
            node.op_type,
            _EVAL_REGISTRY,
            binary_types=BINARY_OP_TYPES,
            unary_types=UNARY_OP_TYPES,
            binary_fallback=lambda: _eval_binary_unary,
            unary_fallback=lambda: _eval_binary_unary,
        )
        handler(self, node)


@register_evaluator("MatMul")
def _eval_matmul(evaluator: Evaluator, node: Node) -> None:
    lower_matmul(evaluator.graph, node)
    left = evaluator.values[node.inputs[0]]
    right = evaluator.values[node.inputs[1]]
    evaluator.values[node.outputs[0]] = _apply_matmul(left, right)


@register_evaluator("Gemm")
def _eval_gemm(evaluator: Evaluator, node: Node) -> None:
    op_dtype = node_dtype(evaluator.graph, node, *node.inputs, *node.outputs)
    spec = resolve_gemm_spec(evaluator.graph, node, op_dtype)
    left = evaluator.values[node.inputs[0]]
    right = evaluator.values[node.inputs[1]]
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
        bias = evaluator.values[node.inputs[2]]
        if beta != 1:
            bias = bias * beta
        result = result + bias
    evaluator.values[node.outputs[0]] = result


@register_evaluator("Cast")
def _eval_cast(evaluator: Evaluator, node: Node) -> None:
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        raise UnsupportedOpError("Cast must have 1 input and 1 output")
    output_dtype = value_dtype(evaluator.graph, node.outputs[0], node)
    target_info = dtype_info(output_dtype)
    input_value = evaluator.values[node.inputs[0]]
    evaluator.values[node.outputs[0]] = input_value.astype(
        target_info.np_dtype, copy=False
    )


@register_evaluator("CastLike")
def _eval_castlike(evaluator: Evaluator, node: Node) -> None:
    if len(node.inputs) != 2 or len(node.outputs) != 1:
        raise UnsupportedOpError("CastLike must have 2 inputs and 1 output")
    like_dtype = value_dtype(evaluator.graph, node.inputs[1], node)
    output_dtype = value_dtype(evaluator.graph, node.outputs[0], node)
    if output_dtype != like_dtype:
        raise UnsupportedOpError(
            "CastLike output dtype must match like input dtype, "
            f"got {output_dtype} and {like_dtype}"
        )
    target_info = dtype_info(output_dtype)
    input_value = evaluator.values[node.inputs[0]]
    evaluator.values[node.outputs[0]] = input_value.astype(
        target_info.np_dtype, copy=False
    )


@register_evaluator("Where")
def _eval_where(evaluator: Evaluator, node: Node) -> None:
    lower_where(evaluator.graph, node)
    condition = evaluator.values[node.inputs[0]]
    x_value = evaluator.values[node.inputs[1]]
    y_value = evaluator.values[node.inputs[2]]
    evaluator.values[node.outputs[0]] = np.where(condition, x_value, y_value)


@register_evaluator("GatherElements")
def _eval_gather_elements(evaluator: Evaluator, node: Node) -> None:
    if len(node.inputs) != 2 or len(node.outputs) != 1:
        raise UnsupportedOpError("GatherElements must have 2 inputs and 1 output")
    data = evaluator.values[node.inputs[0]]
    indices = evaluator.values[node.inputs[1]]
    if indices.dtype.type not in {np.int32, np.int64}:
        raise UnsupportedOpError(
            f"GatherElements indices must be int32 or int64, got {indices.dtype}"
        )
    axis = normalize_axis(int(node.attrs.get("axis", 0)), data.shape, node)
    evaluator.values[node.outputs[0]] = np.take_along_axis(
        data, indices, axis=axis
    )


@register_evaluator("Attention")
def _eval_attention(evaluator: Evaluator, node: Node) -> None:
    input_q = node.inputs[0]
    input_k = node.inputs[1]
    input_v = node.inputs[2]
    output_y = node.outputs[0]
    op_dtype = node_dtype(evaluator.graph, node, input_q, input_k, input_v, output_y)
    spec = resolve_attention_spec(evaluator.graph, node, op_dtype)
    attn_mask_name = optional_name(node.inputs, 3)
    past_key_name = optional_name(node.inputs, 4)
    past_value_name = optional_name(node.inputs, 5)
    nonpad_name = optional_name(node.inputs, 6)
    present_key_name = optional_name(node.outputs, 1)
    present_value_name = optional_name(node.outputs, 2)
    qk_matmul_output_name = optional_name(node.outputs, 3)
    output, present_key, present_value, qk_output = _apply_attention(
        spec,
        evaluator.values[input_q],
        evaluator.values[input_k],
        evaluator.values[input_v],
        evaluator.values[attn_mask_name] if attn_mask_name else None,
        evaluator.values[past_key_name] if past_key_name else None,
        evaluator.values[past_value_name] if past_value_name else None,
        evaluator.values[nonpad_name] if nonpad_name else None,
    )
    evaluator.values[output_y] = output
    if present_key_name is not None:
        evaluator.values[present_key_name] = present_key
    if present_value_name is not None:
        evaluator.values[present_value_name] = present_value
    if qk_matmul_output_name is not None:
        evaluator.values[qk_matmul_output_name] = qk_output


def _apply_lstm_activation(
    kind: int, value: np.ndarray, alpha: float, beta: float
) -> np.ndarray:
    if kind == ACTIVATION_KIND_BY_NAME["Relu"]:
        return np.maximum(value, 0)
    if kind == ACTIVATION_KIND_BY_NAME["Tanh"]:
        return np.tanh(value)
    if kind == ACTIVATION_KIND_BY_NAME["Sigmoid"]:
        return 1 / (1 + np.exp(-value))
    if kind == ACTIVATION_KIND_BY_NAME["Affine"]:
        return alpha * value + beta
    if kind == ACTIVATION_KIND_BY_NAME["LeakyRelu"]:
        return np.where(value < 0, alpha * value, value)
    if kind == ACTIVATION_KIND_BY_NAME["ThresholdedRelu"]:
        return np.where(value > alpha, value, 0)
    if kind == ACTIVATION_KIND_BY_NAME["ScaledTanh"]:
        return alpha * np.tanh(beta * value)
    if kind == ACTIVATION_KIND_BY_NAME["HardSigmoid"]:
        return np.clip(alpha * value + beta, 0, 1)
    if kind == ACTIVATION_KIND_BY_NAME["Elu"]:
        return np.where(value >= 0, value, alpha * (np.exp(value) - 1))
    if kind == ACTIVATION_KIND_BY_NAME["Softsign"]:
        return value / (1 + np.abs(value))
    if kind == ACTIVATION_KIND_BY_NAME["Softplus"]:
        return np.log1p(np.exp(value))
    raise UnsupportedOpError(f"Unsupported LSTM activation kind {kind}")


@register_evaluator("LSTM")
def _eval_lstm(evaluator: Evaluator, node: Node) -> None:
    spec = resolve_lstm_spec(evaluator.graph, node)
    inputs = evaluator.values
    x = inputs[spec.input_x]
    w = inputs[spec.input_w]
    r = inputs[spec.input_r]
    b = inputs[spec.input_b] if spec.input_b is not None else None
    sequence_lens = (
        inputs[spec.input_sequence_lens]
        if spec.input_sequence_lens is not None
        else None
    )
    initial_h = (
        inputs[spec.input_initial_h]
        if spec.input_initial_h is not None
        else None
    )
    initial_c = (
        inputs[spec.input_initial_c]
        if spec.input_initial_c is not None
        else None
    )
    p = inputs[spec.input_p] if spec.input_p is not None else None
    output_y, output_y_h, output_y_c = _apply_lstm(
        spec,
        x,
        w,
        r,
        b,
        sequence_lens,
        initial_h,
        initial_c,
        p,
    )
    if spec.output_y is not None:
        evaluator.values[spec.output_y] = output_y
    if spec.output_y_h is not None:
        evaluator.values[spec.output_y_h] = output_y_h
    if spec.output_y_c is not None:
        evaluator.values[spec.output_y_c] = output_y_c


@register_evaluator("Conv")
def _eval_conv(evaluator: Evaluator, node: Node) -> None:
    op_dtype = node_dtype(evaluator.graph, node, *node.inputs, *node.outputs)
    if op_dtype not in {"float", "double"}:
        raise UnsupportedOpError("Conv supports float and double inputs only")
    spec = resolve_conv_spec(evaluator.graph, node)
    data = evaluator.values[node.inputs[0]]
    weights = evaluator.values[node.inputs[1]]
    bias = evaluator.values[node.inputs[2]] if len(node.inputs) > 2 else None
    evaluator.values[node.outputs[0]] = _apply_conv(spec, data, weights, bias)


@register_evaluator("BatchNormalization")
def _eval_batch_norm(evaluator: Evaluator, node: Node) -> None:
    op = lower_batch_normalization(evaluator.graph, node)
    data = evaluator.values[op.input0]
    scale = evaluator.values[op.scale].reshape(
        (1, op.channels) + (1,) * (data.ndim - 2)
    )
    bias = evaluator.values[op.bias].reshape(
        (1, op.channels) + (1,) * (data.ndim - 2)
    )
    mean = evaluator.values[op.mean].reshape(
        (1, op.channels) + (1,) * (data.ndim - 2)
    )
    variance = evaluator.values[op.variance].reshape(
        (1, op.channels) + (1,) * (data.ndim - 2)
    )
    evaluator.values[op.output] = (
        (data - mean) / np.sqrt(variance + op.epsilon) * scale + bias
    )


@register_evaluator("LRN")
def _eval_lrn(evaluator: Evaluator, node: Node) -> None:
    op_dtype = node_dtype(evaluator.graph, node, *node.inputs, *node.outputs)
    if op_dtype not in {"float", "double"}:
        raise UnsupportedOpError("LRN supports float and double inputs only")
    spec = resolve_lrn_spec(evaluator.graph, node)
    data = evaluator.values[node.inputs[0]]
    evaluator.values[node.outputs[0]] = _apply_lrn(spec, data)


@register_evaluator("AveragePool")
def _eval_average_pool(evaluator: Evaluator, node: Node) -> None:
    op = lower_average_pool(evaluator.graph, node)
    data = evaluator.values[node.inputs[0]]
    evaluator.values[node.outputs[0]] = _apply_average_pool(op, data)


@register_evaluator("GlobalAveragePool")
def _eval_global_average_pool(evaluator: Evaluator, node: Node) -> None:
    op = lower_global_average_pool(evaluator.graph, node)
    data = evaluator.values[node.inputs[0]]
    evaluator.values[node.outputs[0]] = _apply_average_pool(op, data)


@register_evaluator("MaxPool")
def _eval_maxpool(evaluator: Evaluator, node: Node) -> None:
    op_dtype = node_dtype(evaluator.graph, node, *node.inputs, *node.outputs)
    if op_dtype == "bool":
        raise UnsupportedOpError("MaxPool supports numeric inputs only")
    spec = resolve_maxpool_spec(evaluator.graph, node)
    data = evaluator.values[node.inputs[0]]
    evaluator.values[node.outputs[0]] = _apply_maxpool(spec, data)


@register_evaluator("Softmax")
def _eval_softmax(evaluator: Evaluator, node: Node) -> None:
    op = lower_softmax(evaluator.graph, node)
    value = evaluator.values[node.inputs[0]]
    evaluator.values[node.outputs[0]] = _apply_softmax(value, op.axis)


@register_evaluator("LogSoftmax")
def _eval_logsoftmax(evaluator: Evaluator, node: Node) -> None:
    op = lower_logsoftmax(evaluator.graph, node)
    value = evaluator.values[node.inputs[0]]
    evaluator.values[node.outputs[0]] = _apply_logsoftmax(value, op.axis)


@register_evaluator("NegativeLogLikelihoodLoss")
def _eval_negative_log_likelihood_loss(
    evaluator: Evaluator, node: Node
) -> None:
    op = lower_negative_log_likelihood_loss(evaluator.graph, node)
    input_value = evaluator.values[op.input0]
    target_value = evaluator.values[op.target]
    weight_value = evaluator.values[op.weight] if op.weight is not None else None
    evaluator.values[op.output] = _apply_negative_log_likelihood_loss(
        input_value,
        target_value,
        weight_value,
        reduction=op.reduction,
        ignore_index=op.ignore_index,
    )


@register_evaluator("SoftmaxCrossEntropyLoss")
def _eval_softmax_cross_entropy_loss(
    evaluator: Evaluator, node: Node
) -> None:
    op = lower_softmax_cross_entropy_loss(evaluator.graph, node)
    input_value = evaluator.values[op.input0]
    target_value = evaluator.values[op.target]
    weight_value = evaluator.values[op.weight] if op.weight is not None else None
    loss, log_prob = _apply_softmax_cross_entropy_loss(
        input_value,
        target_value,
        weight_value,
        reduction=op.reduction,
        ignore_index=op.ignore_index,
        return_log_prob=op.log_prob is not None,
    )
    evaluator.values[op.output] = loss
    if op.log_prob is not None and log_prob is not None:
        evaluator.values[op.log_prob] = log_prob


@register_evaluator("Dropout")
def _eval_dropout(evaluator: Evaluator, node: Node) -> None:
    op = lower_dropout(evaluator.graph, node)
    evaluator.values[op.output] = evaluator.values[op.input0].copy()


@register_evaluator("Concat")
def _eval_concat(evaluator: Evaluator, node: Node) -> None:
    op = lower_concat(evaluator.graph, node)
    tensors = [evaluator.values[name] for name in node.inputs]
    evaluator.values[op.output] = np.concatenate(tensors, axis=op.axis)


@register_evaluator("Transpose")
def _eval_transpose(evaluator: Evaluator, node: Node) -> None:
    op = lower_transpose(evaluator.graph, node)
    evaluator.values[op.output] = np.transpose(
        evaluator.values[op.input0], axes=tuple(op.perm)
    )


@register_evaluator("Unsqueeze")
def _eval_unsqueeze(evaluator: Evaluator, node: Node) -> None:
    op = lower_unsqueeze(evaluator.graph, node)
    evaluator.values[op.output] = evaluator.values[op.input0].reshape(
        op.output_shape
    )


@register_evaluator("Reshape")
def _eval_reshape(evaluator: Evaluator, node: Node) -> None:
    op = lower_reshape(evaluator.graph, node)
    evaluator.values[op.output] = evaluator.values[op.input0].reshape(
        op.output_shape
    )


@register_evaluator("ConstantOfShape")
def _eval_constant_of_shape(evaluator: Evaluator, node: Node) -> None:
    op = lower_constant_of_shape(evaluator.graph, node)
    info = dtype_info(op.dtype)
    evaluator.values[op.output] = np.full(
        op.shape, op.value, dtype=info.np_dtype
    )


@register_evaluator("Shape")
def _eval_shape(evaluator: Evaluator, node: Node) -> None:
    op = lower_shape(evaluator.graph, node)
    evaluator.values[op.output] = np.array(op.values, dtype=np.int64)


@register_evaluator("ReduceMean")
@register_evaluator("ReduceSum")
@register_evaluator("ReduceMax")
@register_evaluator("ReduceMin")
@register_evaluator("ReduceProd")
@register_evaluator("ReduceL1")
@register_evaluator("ReduceL2")
@register_evaluator("ReduceLogSum")
@register_evaluator("ReduceLogSumExp")
@register_evaluator("ReduceSumSquare")
def _eval_reduce(evaluator: Evaluator, node: Node) -> None:
    if len(node.inputs) not in {1, 2} or len(node.outputs) != 1:
        raise UnsupportedOpError(
            f"{node.op_type} must have 1 or 2 inputs and 1 output"
        )
    op_dtype = node_dtype(evaluator.graph, node, *node.inputs, *node.outputs)
    if (
        node.op_type in REDUCE_OUTPUTS_FLOAT_ONLY
        and op_dtype not in {"float", "double"}
    ):
        raise UnsupportedOpError(
            f"{node.op_type} supports float and double inputs only"
        )
    value = evaluator.values[node.inputs[0]]
    input_shape = value.shape
    axes, noop = resolve_reduce_axes(evaluator.graph, node, input_shape)
    if noop:
        evaluator.values[node.outputs[0]] = value.copy()
        return
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
        result = np.sum(np.abs(value), axis=axes, keepdims=keepdims)
    elif reduce_kind == "l2":
        result = np.sqrt(np.sum(value * value, axis=axes, keepdims=keepdims))
    elif reduce_kind == "logsum":
        result = np.log(np.sum(value, axis=axes, keepdims=keepdims))
    elif reduce_kind == "logsumexp":
        result = np.log(np.sum(np.exp(value), axis=axes, keepdims=keepdims))
    elif reduce_kind == "sumsquare":
        result = np.sum(value * value, axis=axes, keepdims=keepdims)
    else:
        raise UnsupportedOpError(f"Unsupported reduce kind {reduce_kind}")
    evaluator.values[node.outputs[0]] = result


def _eval_binary_unary(evaluator: Evaluator, node: Node) -> None:
    if node.op_type in COMPARE_OP_TYPES:
        input_dtype = node_dtype(evaluator.graph, node, *node.inputs)
        output_dtype = value_dtype(evaluator.graph, node.outputs[0], node)
        if output_dtype != "bool":
            raise UnsupportedOpError(
                f"{node.op_type} expects bool output, got {output_dtype}"
            )
        op_spec = binary_op_symbol(node.op_type, node.attrs, dtype=input_dtype)
        if op_spec is None:
            raise UnsupportedOpError(f"Unsupported op {node.op_type}")
        if len(node.inputs) != 2 or len(node.outputs) != 1:
            raise UnsupportedOpError(
                f"{node.op_type} must have 2 inputs and 1 output"
            )
        left = evaluator.values[node.inputs[0]]
        right = evaluator.values[node.inputs[1]]
        evaluator.values[node.outputs[0]] = apply_binary_op(
            op_spec, left, right
        )
        return
    op_dtype = node_dtype(evaluator.graph, node, *node.inputs, *node.outputs)
    op_spec = binary_op_symbol(node.op_type, node.attrs, dtype=op_dtype)
    unary_symbol = unary_op_symbol(node.op_type, dtype=op_dtype)
    if op_spec is None and unary_symbol is None:
        raise UnsupportedOpError(f"Unsupported op {node.op_type}")
    if op_spec is not None:
        if len(node.inputs) != 2 or len(node.outputs) != 1:
            raise UnsupportedOpError(
                f"{node.op_type} must have 2 inputs and 1 output"
            )
        left = evaluator.values[node.inputs[0]]
        right = evaluator.values[node.inputs[1]]
        evaluator.values[node.outputs[0]] = apply_binary_op(
            op_spec, left, right
        )
        return
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        raise UnsupportedOpError(
            f"{node.op_type} must have 1 input and 1 output"
        )
    value = evaluator.values[node.inputs[0]]
    evaluator.values[node.outputs[0]] = apply_unary_op(unary_symbol, value)


def _apply_matmul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.ndim < 1 or right.ndim < 1:
        raise UnsupportedOpError(
            "MatMul inputs must be at least 1D, "
            f"got {left.shape} x {right.shape}"
        )
    left_dim = left.shape[-1]
    right_dim = right.shape[0] if right.ndim == 1 else right.shape[-2]
    if left_dim != right_dim:
        raise ShapeInferenceError(
            "MatMul inner dimensions must match, "
            f"got {left_dim} and {right_dim}"
        )
    left_batch = left.shape[:-2] if left.ndim > 1 else ()
    right_batch = right.shape[:-2] if right.ndim > 1 else ()
    if not _matmul_batch_broadcastable(left_batch, right_batch):
        raise ShapeInferenceError(
            "MatMul batch dimensions must be broadcastable, "
            f"got {left_batch} x {right_batch}"
        )
    return np.matmul(left, right)


def _matmul_batch_broadcastable(
    left: tuple[int, ...], right: tuple[int, ...]
) -> bool:
    max_rank = max(len(left), len(right))
    left_padded = (1,) * (max_rank - len(left)) + left
    right_padded = (1,) * (max_rank - len(right)) + right
    for left_dim, right_dim in zip(left_padded, right_padded):
        if left_dim == right_dim or left_dim == 1 or right_dim == 1:
            continue
        return False
    return True


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


def _apply_negative_log_likelihood_loss(
    values: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray | None,
    *,
    reduction: str,
    ignore_index: int,
) -> np.ndarray:
    input_shape = values.shape
    if len(input_shape) < 2:
        raise UnsupportedOpError(
            "NegativeLogLikelihoodLoss input must be at least 2D"
        )
    target_shape = target.shape
    if input_shape[0] != target_shape[0]:
        raise ShapeInferenceError(
            "NegativeLogLikelihoodLoss target batch dimension must match input"
        )
    if input_shape[2:] != target_shape[1:]:
        raise ShapeInferenceError(
            "NegativeLogLikelihoodLoss target spatial dimensions must match input"
        )
    n = input_shape[0]
    c = input_shape[1]
    if weight is not None:
        gather_weight = np.take(weight, target.astype(np.int32), mode="clip")
        if ignore_index is not None:
            gather_weight = np.where(target == ignore_index, 0, gather_weight).astype(
                dtype=values.dtype
            )
    elif ignore_index != -1:
        gather_weight = np.where(target == ignore_index, 0, 1).astype(
            dtype=values.dtype
        )
    else:
        gather_weight = None
    if len(input_shape) != 3:
        values = values.reshape((n, c, -1))
        target = target.reshape((n, -1))
    d = values.shape[2]
    loss = np.zeros((n, d), dtype=values.dtype)
    for i in range(n):
        for d_index in range(d):
            if target[i][d_index] != ignore_index:
                loss[i][d_index] = -values[i][target[i][d_index]][d_index]
    if len(input_shape) != 3:
        loss = loss.reshape(target_shape)
    if gather_weight is not None:
        loss = gather_weight * loss
        if reduction == "mean":
            loss = loss.sum() / gather_weight.sum()
            return loss.astype(values.dtype)
    if reduction == "mean":
        loss = np.mean(loss)
    elif reduction == "sum":
        loss = np.sum(loss)
    return loss.astype(values.dtype)


def _apply_softmax_cross_entropy_loss(
    values: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray | None,
    *,
    reduction: str,
    ignore_index: int | None,
    return_log_prob: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    input_shape = values.shape
    if len(input_shape) < 2:
        raise UnsupportedOpError(
            "SoftmaxCrossEntropyLoss input must be at least 2D"
        )
    target_shape = target.shape
    if input_shape[0] != target_shape[0]:
        raise ShapeInferenceError(
            "SoftmaxCrossEntropyLoss target batch dimension must match input"
        )
    if input_shape[2:] != target_shape[1:]:
        raise ShapeInferenceError(
            "SoftmaxCrossEntropyLoss target spatial dimensions must match input"
        )
    log_prob = _apply_logsoftmax(values, axis=1)
    log_prob_output = log_prob if return_log_prob else None
    if weight is not None:
        gather_weight = np.take(weight, target.astype(np.int32), mode="clip")
        if ignore_index is not None:
            gather_weight = np.where(target == ignore_index, 0, gather_weight).astype(
                dtype=values.dtype
            )
    elif ignore_index is not None:
        gather_weight = np.where(target == ignore_index, 0, 1).astype(
            dtype=values.dtype
        )
    else:
        gather_weight = None
    n = input_shape[0]
    c = input_shape[1]
    if len(input_shape) != 3:
        log_prob = log_prob.reshape((n, c, -1))
        target = target.reshape((n, -1))
    d = log_prob.shape[2]
    loss = np.zeros((n, d), dtype=values.dtype)
    for i in range(n):
        for d_index in range(d):
            if ignore_index is None or target[i][d_index] != ignore_index:
                loss[i][d_index] = -log_prob[i][target[i][d_index]][d_index]
    if len(input_shape) != 3:
        loss = loss.reshape(target_shape)
    if gather_weight is not None:
        loss = gather_weight * loss
        if reduction == "mean":
            loss = loss.sum() / gather_weight.sum()
            loss = loss.astype(values.dtype)
            if return_log_prob:
                return loss, log_prob.astype(values.dtype)
            return loss, None
    if reduction == "mean":
        loss = np.mean(loss)
    elif reduction == "sum":
        loss = np.sum(loss)
    loss = loss.astype(values.dtype)
    if return_log_prob and log_prob_output is not None:
        return loss, log_prob_output.astype(values.dtype)
    return loss, None


def _apply_attention(
    spec,
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    attn_mask: np.ndarray | None,
    past_key: np.ndarray | None,
    past_value: np.ndarray | None,
    nonpad_kv_seqlen: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
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
        bias = bias + np.where(causal_mask, -np.inf, 0.0)[None, None, :, :]
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


def _apply_conv(spec, data: np.ndarray, weights: np.ndarray, bias: np.ndarray | None) -> np.ndarray:
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
                            acc += data[(n, ic_global, *in_index)] * weights[
                                (oc_global, ic, *kernel_index)
                            ]
                    output[(n, oc_global, *out_index)] = acc
    return output


def _apply_lrn(spec, data: np.ndarray) -> np.ndarray:
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


def _apply_average_pool(op, data: np.ndarray) -> np.ndarray:
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
                    output[n, c, oh, ow] = 0.0 if count == 0 else acc / float(count)
    return output


def _maxpool_min_value(dtype: np.dtype) -> float | int:
    if np.issubdtype(dtype, np.floating):
        return -np.inf
    if np.issubdtype(dtype, np.integer):
        return np.iinfo(dtype).min
    raise UnsupportedOpError("MaxPool supports numeric inputs only")


def _apply_maxpool(spec, data: np.ndarray) -> np.ndarray:
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


def _apply_lstm(
    spec,
    x: np.ndarray,
    w: np.ndarray,
    r: np.ndarray,
    b: np.ndarray | None,
    sequence_lens: np.ndarray | None,
    initial_h: np.ndarray | None,
    initial_c: np.ndarray | None,
    p: np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    if spec.layout == 1:
        x = np.swapaxes(x, 0, 1)
    seq_length = spec.seq_length
    batch_size = spec.batch_size
    hidden_size = spec.hidden_size
    num_directions = spec.num_directions
    if sequence_lens is None:
        sequence_lens = np.full((batch_size,), seq_length, dtype=np.int64)
    else:
        sequence_lens = sequence_lens.astype(np.int64, copy=False)
    if b is None:
        b = np.zeros((num_directions, 8 * hidden_size), dtype=x.dtype)
    if p is None:
        p = np.zeros((num_directions, 3 * hidden_size), dtype=x.dtype)
    if initial_h is None:
        initial_h = np.zeros((num_directions, batch_size, hidden_size), dtype=x.dtype)
    if initial_c is None:
        initial_c = np.zeros((num_directions, batch_size, hidden_size), dtype=x.dtype)
    output_y = None
    if spec.output_y is not None:
        output_y = np.zeros(
            (seq_length, num_directions, batch_size, hidden_size), dtype=x.dtype
        )
    output_y_h = (
        np.zeros((num_directions, batch_size, hidden_size), dtype=x.dtype)
        if spec.output_y_h is not None
        else None
    )
    output_y_c = (
        np.zeros((num_directions, batch_size, hidden_size), dtype=x.dtype)
        if spec.output_y_c is not None
        else None
    )
    directions = (
        ("forward", "reverse")
        if spec.direction == "bidirectional"
        else (spec.direction,)
    )
    for dir_index, dir_kind in enumerate(directions):
        w_dir = w[dir_index]
        r_dir = r[dir_index]
        b_dir = b[dir_index]
        bias = b_dir[: 4 * hidden_size] + b_dir[4 * hidden_size :]
        p_dir = p[dir_index]
        p_i = p_dir[:hidden_size]
        p_o = p_dir[hidden_size : 2 * hidden_size]
        p_f = p_dir[2 * hidden_size :]
        h_prev = initial_h[dir_index].copy()
        c_prev = initial_c[dir_index].copy()
        act_offset = dir_index * 3
        act_f = spec.activation_kinds[act_offset]
        act_g = spec.activation_kinds[act_offset + 1]
        act_h = spec.activation_kinds[act_offset + 2]
        alpha_f = spec.activation_alphas[act_offset]
        alpha_g = spec.activation_alphas[act_offset + 1]
        alpha_h = spec.activation_alphas[act_offset + 2]
        beta_f = spec.activation_betas[act_offset]
        beta_g = spec.activation_betas[act_offset + 1]
        beta_h = spec.activation_betas[act_offset + 2]
        for step in range(seq_length):
            t_index = step if dir_kind == "forward" else seq_length - 1 - step
            x_t = x[t_index]
            gates = x_t @ w_dir.T + h_prev @ r_dir.T + bias
            if spec.clip is not None and spec.clip > 0:
                gates = np.clip(gates, -spec.clip, spec.clip)
            i, o, f, c = np.split(gates, 4, axis=1)
            i = _apply_lstm_activation(act_f, i + p_i * c_prev, alpha_f, beta_f)
            if spec.input_forget:
                f = 1 - i
            else:
                f = _apply_lstm_activation(
                    act_f, f + p_f * c_prev, alpha_f, beta_f
                )
            c_tilde = _apply_lstm_activation(act_g, c, alpha_g, beta_g)
            c_new = f * c_prev + i * c_tilde
            o = _apply_lstm_activation(act_f, o + p_o * c_new, alpha_f, beta_f)
            h_new = o * _apply_lstm_activation(act_h, c_new, alpha_h, beta_h)
            active_mask = step < sequence_lens
            if not np.all(active_mask):
                h_new = np.where(active_mask[:, None], h_new, h_prev)
                c_new = np.where(active_mask[:, None], c_new, c_prev)
                if output_y is not None:
                    output_y[step, dir_index] = np.where(
                        active_mask[:, None], h_new, 0
                    )
            else:
                if output_y is not None:
                    output_y[step, dir_index] = h_new
            h_prev = h_new
            c_prev = c_new
        if output_y_h is not None:
            output_y_h[dir_index] = h_prev
        if output_y_c is not None:
            output_y_c[dir_index] = c_prev
    if output_y is not None and spec.layout == 1:
        output_y = np.transpose(output_y, (2, 0, 1, 3))
    return output_y, output_y_h, output_y_c

from __future__ import annotations

from shared.scalar_types import ScalarType

from ..codegen.c_emitter import SoftmaxCrossEntropyLossOp
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from .common import shape_product as _shape_product
from .common import value_dtype as _value_dtype
from .common import value_shape as _value_shape
from .registry import register_lowering


@register_lowering("SoftmaxCrossEntropyLoss")
def lower_softmax_cross_entropy_loss(
    graph: Graph, node: Node
) -> SoftmaxCrossEntropyLossOp:
    if len(node.inputs) not in {2, 3} or len(node.outputs) not in {1, 2}:
        raise UnsupportedOpError(
            "SoftmaxCrossEntropyLoss must have 2 or 3 inputs and 1 or 2 outputs"
        )
    input_name = node.inputs[0]
    target_name = node.inputs[1]
    weight_name = node.inputs[2] if len(node.inputs) > 2 else None
    input_dtype = _value_dtype(graph, input_name, node)
    if not input_dtype.is_float:
        raise UnsupportedOpError(
            "SoftmaxCrossEntropyLoss supports float16, float, and double inputs only"
        )
    output_name = node.outputs[0]
    output_dtype = _value_dtype(graph, output_name, node)
    if output_dtype != input_dtype:
        raise UnsupportedOpError(
            "SoftmaxCrossEntropyLoss output dtype must match input dtype"
        )
    log_prob_name = node.outputs[1] if len(node.outputs) > 1 else None
    if log_prob_name is not None:
        log_prob_dtype = _value_dtype(graph, log_prob_name, node)
        if log_prob_dtype != input_dtype:
            raise UnsupportedOpError(
                "SoftmaxCrossEntropyLoss log_prob dtype must match input dtype"
            )
    target_dtype = _value_dtype(graph, target_name, node)
    if target_dtype not in {ScalarType.I32, ScalarType.I64}:
        raise UnsupportedOpError(
            "SoftmaxCrossEntropyLoss target must be int32 or int64"
        )
    if weight_name is not None:
        weight_dtype = _value_dtype(graph, weight_name, node)
        if weight_dtype != input_dtype:
            raise UnsupportedOpError(
                "SoftmaxCrossEntropyLoss weight dtype must match input dtype"
            )
    input_shape = _value_shape(graph, input_name, node)
    target_shape = _value_shape(graph, target_name, node)
    output_shape = _value_shape(graph, output_name, node)
    if len(input_shape) < 2:
        raise ShapeInferenceError("SoftmaxCrossEntropyLoss input must be at least 2D")
    if len(target_shape) != len(input_shape) - 1:
        raise ShapeInferenceError(
            "SoftmaxCrossEntropyLoss target rank must be input rank - 1"
        )
    if input_shape[0] != target_shape[0]:
        raise ShapeInferenceError(
            "SoftmaxCrossEntropyLoss target batch dimension must match input"
        )
    if input_shape[2:] != target_shape[1:]:
        raise ShapeInferenceError(
            "SoftmaxCrossEntropyLoss target spatial dimensions must match input"
        )
    if weight_name is not None:
        weight_shape = _value_shape(graph, weight_name, node)
        if len(weight_shape) != 1 or weight_shape[0] != input_shape[1]:
            raise ShapeInferenceError(
                "SoftmaxCrossEntropyLoss weight must have shape (C,)"
            )
    if log_prob_name is not None:
        log_prob_shape = _value_shape(graph, log_prob_name, node)
        if log_prob_shape != input_shape:
            raise ShapeInferenceError(
                "SoftmaxCrossEntropyLoss log_prob output must match input shape"
            )
    reduction = node.attrs.get("reduction", "mean")
    if isinstance(reduction, bytes):
        reduction = reduction.decode("utf-8")
    if reduction not in {"none", "mean", "sum"}:
        raise UnsupportedOpError(
            "SoftmaxCrossEntropyLoss reduction must be none, mean, or sum"
        )
    if reduction == "none":
        if output_shape != target_shape:
            raise ShapeInferenceError(
                "SoftmaxCrossEntropyLoss output must match target shape "
                "when reduction is none"
            )
    else:
        if output_shape not in {(), (1,)}:
            raise ShapeInferenceError(
                "SoftmaxCrossEntropyLoss output must be scalar when reduced"
            )
    n = input_shape[0]
    c = input_shape[1]
    d = _shape_product(input_shape[2:]) if len(input_shape) > 2 else 1
    ignore_index = node.attrs.get("ignore_index")
    if ignore_index is not None:
        ignore_index = int(ignore_index)
    return SoftmaxCrossEntropyLossOp(
        input0=input_name,
        target=target_name,
        weight=weight_name,
        output=output_name,
        log_prob=log_prob_name,
        input_shape=input_shape,
        target_shape=target_shape,
        output_shape=output_shape,
        log_prob_shape=input_shape if log_prob_name is not None else None,
        n=n,
        c=c,
        d=d,
        reduction=reduction,
        ignore_index=ignore_index,
        dtype=input_dtype,
        target_dtype=target_dtype,
    )

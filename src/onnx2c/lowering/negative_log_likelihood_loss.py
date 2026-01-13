from __future__ import annotations

from ..codegen.c_emitter import NegativeLogLikelihoodLossOp
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from .common import shape_product as _shape_product
from .common import value_dtype as _value_dtype
from .common import value_shape as _value_shape
from .registry import register_lowering


@register_lowering("NegativeLogLikelihoodLoss")
def lower_negative_log_likelihood_loss(
    graph: Graph, node: Node
) -> NegativeLogLikelihoodLossOp:
    if len(node.inputs) not in {2, 3} or len(node.outputs) != 1:
        raise UnsupportedOpError(
            "NegativeLogLikelihoodLoss must have 2 or 3 inputs and 1 output"
        )
    input_name = node.inputs[0]
    target_name = node.inputs[1]
    weight_name = node.inputs[2] if len(node.inputs) > 2 else None
    input_dtype = _value_dtype(graph, input_name, node)
    if input_dtype not in {"float", "double", "float16"}:
        raise UnsupportedOpError(
            "NegativeLogLikelihoodLoss supports float16, float, and double inputs only"
        )
    output_dtype = _value_dtype(graph, node.outputs[0], node)
    if output_dtype != input_dtype:
        raise UnsupportedOpError(
            "NegativeLogLikelihoodLoss output dtype must match input dtype"
        )
    target_dtype = _value_dtype(graph, target_name, node)
    if target_dtype not in {"int32", "int64"}:
        raise UnsupportedOpError(
            "NegativeLogLikelihoodLoss target must be int32 or int64"
        )
    if weight_name is not None:
        weight_dtype = _value_dtype(graph, weight_name, node)
        if weight_dtype != input_dtype:
            raise UnsupportedOpError(
                "NegativeLogLikelihoodLoss weight dtype must match input dtype"
            )
    input_shape = _value_shape(graph, input_name, node)
    target_shape = _value_shape(graph, target_name, node)
    output_shape = _value_shape(graph, node.outputs[0], node)
    if len(input_shape) < 2:
        raise ShapeInferenceError(
            "NegativeLogLikelihoodLoss input must be at least 2D"
        )
    if len(target_shape) != len(input_shape) - 1:
        raise ShapeInferenceError(
            "NegativeLogLikelihoodLoss target rank must be input rank - 1"
        )
    if input_shape[0] != target_shape[0]:
        raise ShapeInferenceError(
            "NegativeLogLikelihoodLoss target batch dimension must match input"
        )
    if input_shape[2:] != target_shape[1:]:
        raise ShapeInferenceError(
            "NegativeLogLikelihoodLoss target spatial dimensions must match input"
        )
    if weight_name is not None:
        weight_shape = _value_shape(graph, weight_name, node)
        if len(weight_shape) != 1 or weight_shape[0] != input_shape[1]:
            raise ShapeInferenceError(
                "NegativeLogLikelihoodLoss weight must have shape (C,)"
            )
    reduction = node.attrs.get("reduction", "mean")
    if isinstance(reduction, bytes):
        reduction = reduction.decode("utf-8")
    if reduction not in {"none", "mean", "sum"}:
        raise UnsupportedOpError(
            "NegativeLogLikelihoodLoss reduction must be none, mean, or sum"
        )
    if reduction == "none":
        if output_shape != target_shape:
            raise ShapeInferenceError(
                "NegativeLogLikelihoodLoss output must match target shape "
                "when reduction is none"
            )
    else:
        if output_shape not in {(), (1,)}:
            raise ShapeInferenceError(
                "NegativeLogLikelihoodLoss output must be scalar when reduced"
            )
    n = input_shape[0]
    c = input_shape[1]
    d = _shape_product(input_shape[2:]) if len(input_shape) > 2 else 1
    ignore_index = int(node.attrs.get("ignore_index", -1))
    return NegativeLogLikelihoodLossOp(
        input0=input_name,
        target=target_name,
        weight=weight_name,
        output=node.outputs[0],
        input_shape=input_shape,
        target_shape=target_shape,
        output_shape=output_shape,
        n=n,
        c=c,
        d=d,
        reduction=reduction,
        ignore_index=ignore_index,
        dtype=input_dtype,
        target_dtype=target_dtype,
    )

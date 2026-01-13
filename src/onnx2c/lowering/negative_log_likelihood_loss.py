from __future__ import annotations

from shared.scalar_types import ScalarType

from ..codegen.c_emitter import NegativeLogLikelihoodLossOp
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Initializer, Node
from .common import shape_product as _shape_product
from .common import value_dtype as _value_dtype
from .common import value_shape as _value_shape
from .registry import register_lowering


def _find_node_by_output(graph: Graph, name: str) -> Node | None:
    for node in graph.nodes:
        if name in node.outputs:
            return node
    return None


def _find_initializer(graph: Graph, name: str) -> Initializer | None:
    for initializer in graph.initializers:
        if initializer.name == name:
            return initializer
    return None


def _resolve_target_shape(
    input_shape: tuple[int, ...],
    shape_values: list[int],
    *,
    allowzero: int,
    node: Node,
) -> tuple[int, ...]:
    if allowzero not in (0, 1):
        raise UnsupportedOpError("Reshape allowzero must be 0 or 1")
    output_dims: list[int] = []
    unknown_index: int | None = None
    known_product = 1
    for index, dim in enumerate(shape_values):
        if dim == -1:
            if unknown_index is not None:
                raise ShapeInferenceError("Reshape allows only one -1 dimension")
            unknown_index = index
            output_dims.append(-1)
            continue
        if dim == 0:
            if allowzero == 0:
                if index >= len(input_shape):
                    raise ShapeInferenceError(
                        "Reshape zero dim must index into input shape"
                    )
                dim = input_shape[index]
        if dim < 0:
            raise ShapeInferenceError("Reshape dims must be >= -1")
        output_dims.append(dim)
        known_product *= dim
    input_product = _shape_product(input_shape)
    if unknown_index is not None:
        if known_product == 0 or input_product % known_product != 0:
            raise ShapeInferenceError(
                "Reshape cannot infer dimension from input shape"
            )
        output_dims[unknown_index] = input_product // known_product
    output_shape = tuple(output_dims)
    if _shape_product(output_shape) != input_product:
        raise ShapeInferenceError(
            "Reshape input and output element counts must match"
        )
    return output_shape


def _shape_values_from_shape_node(
    graph: Graph, name: str, node: Node
) -> list[int] | None:
    shape_node = _find_node_by_output(graph, name)
    if shape_node is None or shape_node.op_type != "Shape":
        return None
    if len(shape_node.inputs) != 1 or len(shape_node.outputs) != 1:
        raise UnsupportedOpError("Shape must have 1 input and 1 output")
    source_shape = _value_shape(graph, shape_node.inputs[0], node)
    return list(source_shape)


def _resolve_shape_from_reshape(
    graph: Graph, name: str, node: Node
) -> tuple[int, ...] | None:
    reshape_node = _find_node_by_output(graph, name)
    if reshape_node is None or reshape_node.op_type != "Reshape":
        return None
    if len(reshape_node.inputs) != 2 or len(reshape_node.outputs) != 1:
        raise UnsupportedOpError("Reshape must have 2 inputs and 1 output")
    input_shape = _value_shape(graph, reshape_node.inputs[0], node)
    if not input_shape:
        return None
    allowzero = int(reshape_node.attrs.get("allowzero", 0))
    shape_initializer = _find_initializer(graph, reshape_node.inputs[1])
    if shape_initializer is not None:
        if shape_initializer.type.dtype not in {ScalarType.I64, ScalarType.I32}:
            raise UnsupportedOpError(
                "Reshape expects int64 or int32 shape input, "
                f"got {shape_initializer.type.dtype.onnx_name}"
            )
        if len(shape_initializer.type.shape) != 1:
            raise UnsupportedOpError("Reshape expects a 1D shape input")
        shape_values = [int(value) for value in shape_initializer.data.reshape(-1)]
        return _resolve_target_shape(
            input_shape,
            shape_values,
            allowzero=allowzero,
            node=node,
        )
    shape_values = _shape_values_from_shape_node(
        graph, reshape_node.inputs[1], node
    )
    if shape_values is None:
        return None
    return _resolve_target_shape(
        input_shape,
        shape_values,
        allowzero=allowzero,
        node=node,
    )


def _resolve_input_shape(
    graph: Graph,
    input_name: str,
    target_shape: tuple[int, ...],
    weight_name: str | None,
    node: Node,
) -> tuple[int, ...]:
    input_shape = _value_shape(graph, input_name, node)
    if input_shape:
        return input_shape
    reshaped = _resolve_shape_from_reshape(graph, input_name, node)
    if reshaped is not None:
        return reshaped
    if weight_name is not None and target_shape:
        weight_shape = _value_shape(graph, weight_name, node)
        if len(weight_shape) != 1:
            return input_shape
        return (target_shape[0], weight_shape[0], *target_shape[1:])
    return input_shape


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
    if not input_dtype.is_float:
        raise UnsupportedOpError(
            "NegativeLogLikelihoodLoss supports float16, float, and double inputs only"
        )
    output_dtype = _value_dtype(graph, node.outputs[0], node)
    if output_dtype != input_dtype:
        raise UnsupportedOpError(
            "NegativeLogLikelihoodLoss output dtype must match input dtype"
        )
    target_dtype = _value_dtype(graph, target_name, node)
    if target_dtype not in {ScalarType.I32, ScalarType.I64}:
        raise UnsupportedOpError(
            "NegativeLogLikelihoodLoss target must be int32 or int64"
        )
    if weight_name is not None:
        weight_dtype = _value_dtype(graph, weight_name, node)
        if weight_dtype != input_dtype:
            raise UnsupportedOpError(
                "NegativeLogLikelihoodLoss weight dtype must match input dtype"
            )
    target_shape = _value_shape(graph, target_name, node)
    output_shape = _value_shape(graph, node.outputs[0], node)
    input_shape = _resolve_input_shape(
        graph, input_name, target_shape, weight_name, node
    )
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
        if not output_shape:
            output_shape = target_shape
        if output_shape != target_shape:
            raise ShapeInferenceError(
                "NegativeLogLikelihoodLoss output must match target shape "
                "when reduction is none"
            )
    else:
        if output_shape and output_shape not in {(), (1,)}:
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

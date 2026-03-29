from __future__ import annotations

from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from ..ir.ops import BiasGeluOp, FastGeluOp
from .common import node_dtype, optional_name, value_shape
from .registry import register_lowering


@register_lowering("FastGelu")
def lower_fast_gelu(graph: Graph, node: Node) -> FastGeluOp:
    if len(node.inputs) < 1 or len(node.outputs) != 1:
        raise UnsupportedOpError("FastGelu must have at least 1 input and 1 output")
    input_name = node.inputs[0]
    bias_name = optional_name(node.inputs, 1)
    output_name = node.outputs[0]
    dtype = node_dtype(graph, node, input_name, output_name)
    if not dtype.is_float:
        raise UnsupportedOpError("FastGelu supports float inputs only")
    input_shape = value_shape(graph, input_name, node)
    output_shape = value_shape(graph, output_name, node)
    if input_shape != output_shape:
        raise ShapeInferenceError(
            f"FastGelu input shape {input_shape} must match output shape {output_shape}"
        )
    if bias_name is not None:
        bias_shape = value_shape(graph, bias_name, node)
        if len(input_shape) == 0 or bias_shape != (input_shape[-1],):
            raise UnsupportedOpError(
                f"FastGelu bias shape {bias_shape} must be (last_dim,) = ({input_shape[-1] if input_shape else '?'},)"
            )
    last_dim = input_shape[-1] if input_shape else 1
    return FastGeluOp(
        input0=input_name,
        bias=bias_name,
        output=output_name,
        shape=input_shape,
        last_dim=last_dim,
        dtype=dtype,
    )


@register_lowering("BiasGelu")
def lower_bias_gelu(graph: Graph, node: Node) -> BiasGeluOp:
    if len(node.inputs) != 2 or len(node.outputs) != 1:
        raise UnsupportedOpError("BiasGelu must have 2 inputs and 1 output")
    input_name = node.inputs[0]
    bias_name = node.inputs[1]
    output_name = node.outputs[0]
    dtype = node_dtype(graph, node, input_name, output_name)
    if not dtype.is_float:
        raise UnsupportedOpError("BiasGelu supports float inputs only")
    input_shape = value_shape(graph, input_name, node)
    output_shape = value_shape(graph, output_name, node)
    if input_shape != output_shape:
        raise ShapeInferenceError(
            f"BiasGelu input shape {input_shape} must match output shape {output_shape}"
        )
    bias_shape = value_shape(graph, bias_name, node)
    if len(input_shape) == 0 or bias_shape != (input_shape[-1],):
        raise UnsupportedOpError(
            f"BiasGelu bias shape {bias_shape} must be (last_dim,) = ({input_shape[-1] if input_shape else '?'},)"
        )
    last_dim = input_shape[-1]
    return BiasGeluOp(
        input0=input_name,
        bias=bias_name,
        output=output_name,
        shape=input_shape,
        last_dim=last_dim,
        dtype=dtype,
    )

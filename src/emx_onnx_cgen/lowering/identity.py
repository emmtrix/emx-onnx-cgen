from __future__ import annotations

from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.context import GraphContext
from ..ir.model import Graph, Node, SequenceType
from ..ir.ops import IdentityOp, SequenceIdentityOp
from .common import value_dtype, value_has_dim_params, value_shape
from .registry import register_lowering


@register_lowering("Identity")
def lower_identity(graph: Graph, node: Node) -> IdentityOp:
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        raise UnsupportedOpError("Identity must have 1 input and 1 output")
    input_value_type = graph.find_value(node.inputs[0]).type
    output_value_type = graph.find_value(node.outputs[0]).type
    if isinstance(input_value_type, SequenceType):
        if not isinstance(output_value_type, SequenceType):
            raise UnsupportedOpError(
                "Identity expects matching input/output types for sequence values"
            )
        input_optional = input_value_type.is_optional
        output_optional = output_value_type.is_optional
        if input_optional != output_optional:
            raise UnsupportedOpError(
                "Identity expects matching optionality for sequence values"
            )
        input_present = f"{node.inputs[0]}_present" if input_optional else None
        output_present = f"{node.outputs[0]}_present" if output_optional else None
        return SequenceIdentityOp(
            input_sequence=node.inputs[0],
            output_sequence=node.outputs[0],
            input_present=input_present,
            output_present=output_present,
        )
    input_shape = value_shape(graph, node.inputs[0], node)
    output_shape = value_shape(graph, node.outputs[0], node)
    if value_has_dim_params(graph, node.outputs[0]) or not output_shape:
        output_shape = ()
    input_dim_params = graph.find_value(node.inputs[0]).type.dim_params
    output_dim_params = graph.find_value(node.outputs[0]).type.dim_params
    if input_shape and output_shape:
        if len(input_shape) != len(output_shape):
            raise ShapeInferenceError("Identity input and output shapes must match")
        for index, (input_dim, output_dim) in enumerate(zip(input_shape, output_shape)):
            if input_dim != output_dim and not (
                input_dim_params[index] or output_dim_params[index]
            ):
                raise ShapeInferenceError("Identity input and output shapes must match")
    input_dtype = value_dtype(graph, node.inputs[0], node)
    output_dtype = value_dtype(graph, node.outputs[0], node)
    if input_dtype != output_dtype:
        raise UnsupportedOpError(
            "Identity expects matching input/output dtypes, "
            f"got {input_dtype.onnx_name} and {output_dtype.onnx_name}"
        )
    if isinstance(graph, GraphContext):
        graph.set_shape(node.outputs[0], input_shape)
    return IdentityOp(
        input0=node.inputs[0],
        output=node.outputs[0],
    )

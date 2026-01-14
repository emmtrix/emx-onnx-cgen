from __future__ import annotations

from ..codegen.c_emitter import IdentityOp
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from .common import value_dtype, value_shape
from .registry import register_lowering


@register_lowering("Identity")
def lower_identity(graph: Graph, node: Node) -> IdentityOp:
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        raise UnsupportedOpError("Identity must have 1 input and 1 output")
    input_shape = value_shape(graph, node.inputs[0], node)
    output_shape = value_shape(graph, node.outputs[0], node)
    if input_shape != output_shape:
        raise ShapeInferenceError("Identity input and output shapes must match")
    input_dtype = value_dtype(graph, node.inputs[0], node)
    output_dtype = value_dtype(graph, node.outputs[0], node)
    if input_dtype != output_dtype:
        raise UnsupportedOpError(
            "Identity expects matching input/output dtypes, "
            f"got {input_dtype.onnx_name} and {output_dtype.onnx_name}"
        )
    return IdentityOp(
        input0=node.inputs[0],
        output=node.outputs[0],
        shape=output_shape,
        dtype=output_dtype,
        input_dtype=input_dtype,
    )

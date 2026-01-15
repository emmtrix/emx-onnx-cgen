from __future__ import annotations

from ..codegen.c_emitter import ReshapeOp
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from .common import value_dtype as _value_dtype
from .common import value_shape as _value_shape
from .registry import register_lowering


def _is_value_used(graph: Graph, name: str) -> bool:
    if any(value.name == name for value in graph.outputs):
        return True
    return any(name in node.inputs for node in graph.nodes)


@register_lowering("Dropout")
def lower_dropout(graph: Graph, node: Node) -> ReshapeOp:
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
        input_dtype=input_dtype,
    )

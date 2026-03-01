from __future__ import annotations

from ..errors import UnsupportedOpError
from ..ir.model import Graph, Node
from ..ir.ops import StringConcatOp
from .common import value_dtype, value_shape
from .registry import register_lowering


@register_lowering("StringConcat")
def lower_string_concat(graph: Graph, node: Node) -> StringConcatOp:
    if len(node.inputs) != 2 or len(node.outputs) != 1:
        raise UnsupportedOpError("StringConcat must have 2 inputs and 1 output")

    input0_name = node.inputs[0]
    input1_name = node.inputs[1]
    output_name = node.outputs[0]

    for name in (input0_name, input1_name, output_name):
        dtype = value_dtype(graph, name, node)
        if dtype.onnx_name != "string":
            raise UnsupportedOpError("StringConcat supports only string tensors")

    input0_shape = value_shape(graph, input0_name, node)
    input1_shape = value_shape(graph, input1_name, node)
    output_shape = value_shape(graph, output_name, node)

    rank0, rank1 = len(input0_shape), len(input1_shape)
    rank_out = len(output_shape)
    if rank_out != max(rank0, rank1):
        raise UnsupportedOpError(
            "StringConcat output rank does not match expected broadcast rank"
        )

    return StringConcatOp(
        input0=input0_name,
        input1=input1_name,
        output=output_name,
    )

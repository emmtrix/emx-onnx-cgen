from __future__ import annotations

from ..errors import UnsupportedOpError
from ..ir.context import GraphContext
from ..ir.model import Node, SequenceType
from ..ir.ops import OptionalGetElementOp, SequenceIdentityOp
from .registry import register_lowering


@register_lowering("OptionalGetElement")
def lower_optional_get_element(
    ctx: GraphContext, node: Node
) -> OptionalGetElementOp | SequenceIdentityOp:
    if len(node.inputs) != 1 or not node.inputs[0]:
        raise UnsupportedOpError(
            "OptionalGetElement expects exactly one non-empty input."
        )
    if len(node.outputs) != 1 or not node.outputs[0]:
        raise UnsupportedOpError("OptionalGetElement expects exactly one output.")
    input_name = node.inputs[0]
    value = ctx.find_value(input_name)
    if isinstance(value.type, SequenceType):
        return SequenceIdentityOp(
            input_sequence=input_name, output_sequence=node.outputs[0]
        )
    return OptionalGetElementOp(input0=input_name, output=node.outputs[0])

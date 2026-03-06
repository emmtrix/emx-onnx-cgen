from __future__ import annotations

from ..errors import UnsupportedOpError
from ..ir.context import GraphContext
from ..ir.model import Node
from ..ir.ops import OptionalHasElementAbsentOp, OptionalHasElementOp
from .registry import register_lowering


@register_lowering("OptionalHasElement")
def lower_optional_has_element(
    ctx: GraphContext, node: Node
) -> OptionalHasElementOp | OptionalHasElementAbsentOp:
    if len(node.inputs) > 1:
        raise UnsupportedOpError("OptionalHasElement expects at most one input.")
    if len(node.outputs) != 1 or not node.outputs[0]:
        raise UnsupportedOpError("OptionalHasElement expects exactly one output.")
    if not node.inputs or not node.inputs[0]:
        return OptionalHasElementAbsentOp(output=node.outputs[0])
    input_name = node.inputs[0]
    value = ctx.find_value(input_name)
    return OptionalHasElementOp(
        input0=input_name,
        output=node.outputs[0],
        input_is_optional=value.type.is_optional,
    )

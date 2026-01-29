from __future__ import annotations

from ..errors import UnsupportedOpError
from ..ir.context import GraphContext
from ..ir.model import Node
from ..ir.ops import OptionalHasElementOp
from .registry import register_lowering


@register_lowering("OptionalHasElement")
def lower_optional_has_element(
    ctx: GraphContext, node: Node
) -> OptionalHasElementOp:
    if len(node.inputs) != 1 or not node.inputs[0]:
        raise UnsupportedOpError(
            "OptionalHasElement expects exactly one non-empty input."
        )
    if len(node.outputs) != 1 or not node.outputs[0]:
        raise UnsupportedOpError(
            "OptionalHasElement expects exactly one output."
        )
    input_name = node.inputs[0]
    value = ctx.find_value(input_name)
    if not value.type.is_optional:
        raise UnsupportedOpError(
            "OptionalHasElement expects an optional input."
        )
    return OptionalHasElementOp(input0=input_name, output=node.outputs[0])

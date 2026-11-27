from __future__ import annotations

from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from ..ir.ops import ExpandOp
from .common import value_dim_params, value_shape
from .registry import register_lowering


def _validate_dynamic_dims(graph: Graph, node: Node) -> None:
    """A dynamic dim must be named and keep its name through the Expand.

    The kernel indexes the input with the output loop variable on such an
    axis, so it must be the same runtime dimension; a dynamic input dim that
    is 1 at runtime (a valid ONNX broadcast) is not supported.
    """

    input_shape = value_shape(graph, node.inputs[0], node)
    output_shape = value_shape(graph, node.outputs[0], node)
    input_params = value_dim_params(graph, node.inputs[0])
    output_params = value_dim_params(graph, node.outputs[0])
    offset = len(output_shape) - len(input_shape)
    for axis, dim in enumerate(input_shape):
        if dim >= 0:
            continue
        input_param = input_params[axis] if axis < len(input_params) else None
        output_axis = axis + offset
        output_param = (
            output_params[output_axis]
            if 0 <= output_axis < len(output_params)
            else None
        )
        if not input_param or input_param != output_param:
            raise ShapeInferenceError(
                f"Expand {node.name or node.outputs[0]}: dynamic input axis {axis} "
                f"({input_param or '?'}) must keep its dim_param in the output "
                f"(got {output_param or '?'}); input shape {input_shape}, output "
                f"shape {output_shape}. Hint: pin it with --input-dim."
            )
    for axis, dim in enumerate(output_shape):
        if dim < 0 and not (axis < len(output_params) and output_params[axis]):
            raise ShapeInferenceError(
                f"Expand {node.name or node.outputs[0]}: dynamic output axis {axis} "
                f"has no dim_param; output shape {output_shape}. "
                "Hint: pin it with --input-dim or export with static shapes."
            )


@register_lowering("Expand")
def lower_expand(graph: Graph, node: Node) -> ExpandOp:
    if len(node.inputs) != 2 or len(node.outputs) != 1:
        raise UnsupportedOpError("Expand must have 2 inputs and 1 output")
    _validate_dynamic_dims(graph, node)
    return ExpandOp(
        input0=node.inputs[0],
        input_shape=node.inputs[1],
        output=node.outputs[0],
    )

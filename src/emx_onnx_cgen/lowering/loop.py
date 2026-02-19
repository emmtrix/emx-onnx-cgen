from __future__ import annotations

from onnx import GraphProto, NodeProto

from ..errors import UnsupportedOpError
from ..ir.model import Graph, Node
from ..ir.ops import LoopRangeOp
from ..lowering.common import node_dtype, value_shape
from .registry import register_lowering


def _find_body(node: Node) -> GraphProto:
    body = node.attrs.get("body")
    if not isinstance(body, GraphProto):
        raise UnsupportedOpError("Loop requires a body graph")
    return body


def _match_single_node(
    nodes: list[NodeProto],
    op_type: str,
    inputs: tuple[str, ...],
    outputs: tuple[str, ...],
) -> NodeProto:
    for candidate in nodes:
        if (
            candidate.op_type == op_type
            and tuple(candidate.input) == inputs
            and tuple(candidate.output) == outputs
        ):
            return candidate
    raise UnsupportedOpError("Unsupported op Loop")


@register_lowering("Loop")
def lower_loop(graph: Graph, node: Node) -> LoopRangeOp:
    if len(node.inputs) != 3 or len(node.outputs) != 2:
        raise UnsupportedOpError("Unsupported op Loop")
    body = _find_body(node)
    if len(body.input) != 3 or len(body.output) != 3:
        raise UnsupportedOpError("Unsupported op Loop")

    iter_name = body.input[0].name
    cond_in_name = body.input[1].name
    prev_name = body.input[2].name
    cond_out_name = body.output[0].name
    current_name = body.output[1].name
    range_name = body.output[2].name

    body_nodes = list(body.node)
    _match_single_node(body_nodes, "Identity", (cond_in_name,), (cond_out_name,))
    add_nodes = [candidate for candidate in body_nodes if candidate.op_type == "Add" and tuple(candidate.output) == (current_name,)]
    if len(add_nodes) != 1:
        raise UnsupportedOpError("Unsupported op Loop")
    add_node = add_nodes[0]
    _match_single_node(body_nodes, "Identity", (prev_name,), (range_name,))

    if len(add_node.input) != 2 or add_node.input[0] != prev_name:
        raise UnsupportedOpError("Unsupported op Loop")
    delta_name = add_node.input[1]
    if not delta_name:
        raise UnsupportedOpError("Unsupported op Loop")

    trip_count_shape = value_shape(graph, node.inputs[0], node)
    cond_shape = value_shape(graph, node.inputs[1], node)
    start_shape = value_shape(graph, node.inputs[2], node)
    delta_shape = value_shape(graph, delta_name, node)
    final_shape = value_shape(graph, node.outputs[0], node)
    output_shape = value_shape(graph, node.outputs[1], node)
    if trip_count_shape not in {(), (1,)}:
        raise UnsupportedOpError("Unsupported op Loop")
    if cond_shape not in {(), (1,)}:
        raise UnsupportedOpError("Unsupported op Loop")
    if start_shape not in {(), (1,)}:
        raise UnsupportedOpError("Unsupported op Loop")
    if delta_shape not in {(), (1,)}:
        raise UnsupportedOpError("Unsupported op Loop")
    if final_shape not in {(), (1,)}:
        raise UnsupportedOpError("Unsupported op Loop")
    if len(output_shape) != 1:
        raise UnsupportedOpError("Unsupported op Loop")

    dtype = node_dtype(graph, node, node.inputs[2], node.outputs[0], node.outputs[1])
    if dtype.name not in {"F32", "F64", "I32", "I64", "I16"}:
        raise UnsupportedOpError("Unsupported op Loop")

    if body_nodes and any(candidate.op_type not in {"Identity", "Add"} for candidate in body_nodes):
        raise UnsupportedOpError("Unsupported op Loop")

    _ = iter_name

    return LoopRangeOp(
        trip_count=node.inputs[0],
        cond=node.inputs[1],
        start=node.inputs[2],
        delta=delta_name,
        final=node.outputs[0],
        output=node.outputs[1],
    )

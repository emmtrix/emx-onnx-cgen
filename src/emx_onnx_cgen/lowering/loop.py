from __future__ import annotations

from onnx import GraphProto, NodeProto

from ..errors import UnsupportedOpError
from ..ir.model import Graph, Node
from ..ir.ops import RangeOp
from .registry import register_lowering


def _body_node_by_output(body: GraphProto) -> dict[str, NodeProto]:
    mapping: dict[str, NodeProto] = {}
    for node in body.node:
        for output in node.output:
            if output:
                mapping[output] = node
    return mapping


def _match_range_loop(node: Node) -> tuple[str, str] | None:
    if len(node.inputs) != 3 or len(node.outputs) != 2:
        return None
    body = node.attrs.get("body")
    if not isinstance(body, GraphProto):
        return None
    if len(body.input) != 3 or len(body.output) != 3:
        return None

    iter_name = body.input[0].name
    cond_name = body.input[1].name
    prev_name = body.input[2].name
    cond_out_name = body.output[0].name
    carry_out_name = body.output[1].name
    scan_out_name = body.output[2].name

    producers = _body_node_by_output(body)

    cond_producer = producers.get(cond_out_name)
    if (
        cond_producer is None
        or cond_producer.op_type != "Identity"
        or tuple(cond_producer.input) != (cond_name,)
    ):
        return None

    carry_producer = producers.get(carry_out_name)
    if (
        carry_producer is None
        or carry_producer.op_type != "Add"
        or len(carry_producer.input) != 2
    ):
        return None
    add_lhs, add_rhs = tuple(carry_producer.input)
    if add_lhs == prev_name:
        delta_name = add_rhs
    elif add_rhs == prev_name:
        delta_name = add_lhs
    else:
        return None
    if not delta_name or delta_name == iter_name:
        return None

    scan_producer = producers.get(scan_out_name)
    if (
        scan_producer is None
        or scan_producer.op_type != "Identity"
        or tuple(scan_producer.input) != (prev_name,)
    ):
        return None

    if len(body.node) != 3:
        return None

    return node.inputs[2], delta_name


@register_lowering("Loop")
def lower_loop(graph: Graph, node: Node) -> RangeOp:
    matched = _match_range_loop(node)
    if matched is None:
        raise UnsupportedOpError("Unsupported op Loop")
    start_name, delta_name = matched
    return RangeOp(
        start=start_name,
        # This Loop form represents a trip-count-driven range where the
        # generated Range kernel only needs start/delta and fixed output length.
        # Reusing start for limit keeps parameter dtypes consistent.
        limit=start_name,
        delta=delta_name,
        output=node.outputs[1],
    )

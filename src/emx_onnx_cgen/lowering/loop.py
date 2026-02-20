from __future__ import annotations

import onnx
from onnx import GraphProto, NodeProto

from ..errors import UnsupportedOpError
from ..ir.model import Graph, Node, TensorType
from ..ir.ops import LoopOp, LoopRangeOp
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


def _try_lower_loop_range(graph: Graph, node: Node) -> LoopRangeOp | None:
    try:
        if len(node.inputs) != 3 or len(node.outputs) != 2:
            return None
        body = _find_body(node)
        if len(body.input) != 3 or len(body.output) != 3:
            return None

        iter_name = body.input[0].name
        cond_in_name = body.input[1].name
        prev_name = body.input[2].name
        cond_out_name = body.output[0].name
        current_name = body.output[1].name
        range_name = body.output[2].name

        body_nodes = list(body.node)
        _match_single_node(body_nodes, "Identity", (cond_in_name,), (cond_out_name,))
        add_nodes = [
            candidate
            for candidate in body_nodes
            if candidate.op_type == "Add" and tuple(candidate.output) == (current_name,)
        ]
        if len(add_nodes) != 1:
            return None
        add_node = add_nodes[0]
        _match_single_node(body_nodes, "Identity", (prev_name,), (range_name,))

        if len(add_node.input) != 2 or add_node.input[0] != prev_name:
            return None
        delta_name = add_node.input[1]
        if not delta_name:
            return None

        trip_count_shape = value_shape(graph, node.inputs[0], node)
        cond_shape = value_shape(graph, node.inputs[1], node)
        start_shape = value_shape(graph, node.inputs[2], node)
        delta_shape = value_shape(graph, delta_name, node)
        final_shape = value_shape(graph, node.outputs[0], node)
        output_shape = value_shape(graph, node.outputs[1], node)
        if (
            trip_count_shape not in {(), (1,)}
            or cond_shape not in {(), (1,)}
            or start_shape not in {(), (1,)}
            or delta_shape not in {(), (1,)}
            or final_shape not in {(), (1,)}
            or len(output_shape) != 1
        ):
            return None

        dtype = node_dtype(
            graph, node, node.inputs[2], node.outputs[0], node.outputs[1]
        )
        if dtype.name not in {"F32", "F64", "I32", "I64", "I16"}:
            return None

        if body_nodes and any(
            candidate.op_type not in {"Identity", "Add"} for candidate in body_nodes
        ):
            return None

        _ = iter_name
        return LoopRangeOp(
            trip_count=node.inputs[0],
            cond=node.inputs[1],
            start=node.inputs[2],
            delta=delta_name,
            final=node.outputs[0],
            output=node.outputs[1],
        )
    except UnsupportedOpError:
        return None


@register_lowering("Loop")
def lower_loop(graph: Graph, node: Node) -> LoopRangeOp | LoopOp:
    # Prefer a small specialized lowering when possible.
    loop_range = _try_lower_loop_range(graph, node)
    if loop_range is not None:
        return loop_range

    if len(node.inputs) < 2:
        raise UnsupportedOpError("Loop requires trip_count and cond inputs")
    if not node.outputs:
        raise UnsupportedOpError("Loop requires outputs")

    trip_count = node.inputs[0]
    cond = node.inputs[1]
    if not trip_count:
        raise UnsupportedOpError("Loop requires trip_count input")
    if not cond:
        raise UnsupportedOpError("Loop requires cond input")

    carried_inputs = tuple(node.inputs[2:])
    if len(node.outputs) < len(carried_inputs):
        raise UnsupportedOpError("Loop outputs must include final loop-carried values")
    carried_outputs = tuple(node.outputs[: len(carried_inputs)])
    scan_outputs = tuple(node.outputs[len(carried_inputs) :])

    body = _find_body(node)
    expected_body_inputs = 2 + len(carried_inputs)
    if len(body.input) != expected_body_inputs:
        raise UnsupportedOpError("Loop body input count does not match Loop inputs")
    expected_body_outputs = 1 + len(carried_inputs) + len(scan_outputs)
    if len(body.output) != expected_body_outputs:
        raise UnsupportedOpError("Loop body output count does not match Loop outputs")

    # Codegen currently supports tensor loop-carried values and scan outputs only.
    for name in carried_inputs + carried_outputs + scan_outputs:
        if not name:
            raise UnsupportedOpError("Loop inputs/outputs must be provided")
        try:
            value = graph.find_value(name)
        except KeyError:
            continue
        if not isinstance(value.type, TensorType):
            raise UnsupportedOpError("Loop currently supports tensor values only")

    body_bytes = body.SerializeToString()
    # Ensure we can deserialize the body later during codegen.
    _ = onnx.GraphProto.FromString(body_bytes)

    return LoopOp(
        trip_count=trip_count,
        cond=cond,
        inputs=carried_inputs,
        outputs=carried_outputs,
        scan_outputs=scan_outputs,
        body=body_bytes,
    )

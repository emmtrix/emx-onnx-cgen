from __future__ import annotations

from onnx import GraphProto, NodeProto, numpy_helper

from ..errors import UnsupportedOpError
from ..ir.model import Graph, Node, SequenceType
from ..ir.ops import LoopRangeOp, LoopSequenceInsertOp
from ..lowering.common import node_dtype, shape_product, value_shape
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


def _lower_loop_range(graph: Graph, node: Node, body: GraphProto) -> LoopRangeOp:
    if len(node.inputs) != 3 or len(node.outputs) != 2:
        raise UnsupportedOpError("Unsupported op Loop")
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
    add_nodes = [
        candidate
        for candidate in body_nodes
        if candidate.op_type == "Add" and tuple(candidate.output) == (current_name,)
    ]
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

    if body_nodes and any(
        candidate.op_type not in {"Identity", "Add"} for candidate in body_nodes
    ):
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


def _const_value_map(body: GraphProto) -> dict[str, object]:
    constants: dict[str, object] = {}
    for body_node in body.node:
        if body_node.op_type != "Constant" or len(body_node.output) != 1:
            continue
        for attr in body_node.attribute:
            if attr.name == "value":
                constants[body_node.output[0]] = numpy_helper.to_array(attr.t)
                break
    return constants


def _lower_loop_tensor_scan_add(
    graph: Graph, node: Node, body: GraphProto
) -> LoopRangeOp:
    if len(node.inputs) != 3 or len(node.outputs) != 2:
        raise UnsupportedOpError("Unsupported op Loop")
    if len(body.input) != 3 or len(body.output) != 3:
        raise UnsupportedOpError("Unsupported op Loop")

    iter_name = body.input[0].name
    cond_in_name = body.input[1].name
    state_in = body.input[2].name
    cond_out = body.output[0].name
    state_out = body.output[1].name
    scan_out = body.output[2].name
    _match_single_node(list(body.node), "Identity", (cond_in_name,), (cond_out,))
    _match_single_node(list(body.node), "Identity", (state_out,), (scan_out,))

    add_state = _match_single_node(
        list(body.node), "Add", (state_in, "slice_out"), (state_out,)
    )
    _ = add_state
    _match_single_node(list(body.node), "Unsqueeze", (iter_name,), ("slice_start",))
    _match_single_node(
        list(body.node), "Slice", ("x", "slice_start", "slice_end"), ("slice_out",)
    )
    _match_single_node(list(body.node), "Add", (iter_name, "one"), ("end",))
    _match_single_node(list(body.node), "Unsqueeze", ("end",), ("slice_end",))

    constants = _const_value_map(body)
    table = constants.get("x")
    if table is None:
        raise UnsupportedOpError("Unsupported op Loop")

    trip_count_shape = value_shape(graph, node.inputs[0], node)
    cond_shape = value_shape(graph, node.inputs[1], node)
    start_shape = value_shape(graph, node.inputs[2], node)
    final_shape = value_shape(graph, node.outputs[0], node)
    output_shape = value_shape(graph, node.outputs[1], node)
    if trip_count_shape not in {(), (1,)} or cond_shape not in {(), (1,)}:
        raise UnsupportedOpError("Unsupported op Loop")
    if not start_shape or final_shape != start_shape:
        raise UnsupportedOpError("Unsupported op Loop")
    if len(start_shape) != 1:
        raise UnsupportedOpError("Unsupported op Loop")
    if len(output_shape) != 2:
        raise UnsupportedOpError("Unsupported op Loop")
    if output_shape[1] != start_shape[0]:
        raise UnsupportedOpError("Unsupported op Loop")
    state_size = shape_product(start_shape)
    if tuple(table.shape[1:]) == start_shape:
        table_data = table.reshape(-1).tolist()
        table_shape = (int(table.shape[0]), state_size)
    elif len(table.shape) == 1:
        table_data = []
        for value in table.reshape(-1).tolist():
            table_data.extend([value] * state_size)
        table_shape = (int(table.shape[0]), state_size)
    else:
        raise UnsupportedOpError("Unsupported op Loop")

    dtype = node_dtype(graph, node, node.inputs[2], node.outputs[0], node.outputs[1])
    if dtype.name not in {"F32", "F64", "I32", "I64", "I16"}:
        raise UnsupportedOpError("Unsupported op Loop")

    return LoopRangeOp(
        trip_count=node.inputs[0],
        cond=node.inputs[1],
        start=node.inputs[2],
        delta=node.inputs[2],
        final=node.outputs[0],
        output=node.outputs[1],
        add_table_data=tuple(table_data),
        add_table_shape=table_shape,
    )


def _lower_loop_sequence_insert(
    graph: Graph, node: Node, body: GraphProto
) -> LoopSequenceInsertOp:
    if len(node.inputs) != 3 or len(node.outputs) != 1:
        raise UnsupportedOpError("Unsupported op Loop")
    if len(body.input) != 3 or len(body.output) != 2:
        raise UnsupportedOpError("Unsupported op Loop")

    iter_name = body.input[0].name
    cond_in_name = body.input[1].name
    seq_in = body.input[2].name
    cond_out = body.output[0].name
    seq_out = body.output[1].name
    body_nodes = list(body.node)
    _match_single_node(body_nodes, "Identity", (cond_in_name,), (cond_out,))
    _match_single_node(body_nodes, "Add", (iter_name, "one"), ("end",))
    _match_single_node(body_nodes, "Unsqueeze", ("end", "axes"), ("slice_end",))
    _match_single_node(
        body_nodes, "Slice", ("x", "slice_start", "slice_end"), ("slice_out",)
    )
    _match_single_node(body_nodes, "SequenceInsert", (seq_in, "slice_out"), (seq_out,))

    constants = _const_value_map(body)
    table = constants.get("x")
    if table is None or len(table.shape) != 1:
        raise UnsupportedOpError("Unsupported op Loop")

    trip_count_shape = value_shape(graph, node.inputs[0], node)
    cond_shape = value_shape(graph, node.inputs[1], node)
    if trip_count_shape not in {(), (1,)} or cond_shape not in {(), (1,)}:
        raise UnsupportedOpError("Unsupported op Loop")

    input_value = graph.find_value(node.inputs[2])
    output_value = graph.find_value(node.outputs[0])
    if not isinstance(input_value.type, SequenceType) or not isinstance(
        output_value.type, SequenceType
    ):
        raise UnsupportedOpError("Unsupported op Loop")
    if input_value.type.elem.shape != output_value.type.elem.shape:
        raise UnsupportedOpError("Unsupported op Loop")
    if input_value.type.elem.shape != ():
        raise UnsupportedOpError("Unsupported op Loop")

    elem_dtype = input_value.type.elem.dtype
    if elem_dtype.name not in {"F32", "F64", "I32", "I64", "I16"}:
        raise UnsupportedOpError("Unsupported op Loop")

    return LoopSequenceInsertOp(
        trip_count=node.inputs[0],
        cond=node.inputs[1],
        input_sequence=node.inputs[2],
        output_sequence=node.outputs[0],
        table_data=tuple(table.reshape(-1).tolist()),
        table_shape=(int(table.shape[0]),),
        elem_shape=input_value.type.elem.shape,
        elem_dtype=input_value.type.elem.dtype,
    )


@register_lowering("Loop")
def lower_loop(graph: Graph, node: Node) -> LoopRangeOp | LoopSequenceInsertOp:
    body = _find_body(node)
    try:
        return _lower_loop_range(graph, node, body)
    except UnsupportedOpError:
        try:
            return _lower_loop_tensor_scan_add(graph, node, body)
        except UnsupportedOpError:
            return _lower_loop_sequence_insert(graph, node, body)

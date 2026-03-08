from __future__ import annotations

from onnx import GraphProto, NodeProto, numpy_helper

from ..errors import UnsupportedOpError
from ..ir.model import Graph, Node, SequenceType
from ..ir.ops import LoopRangeOp, LoopSequenceInsertOp, LoopSequenceMapOp
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


def _lower_loop_sequence_map(
    graph: Graph, node: Node, body: GraphProto
) -> LoopSequenceMapOp:
    if len(node.inputs) < 3 or len(node.outputs) < 1:
        raise UnsupportedOpError("Unsupported op Loop")
    if len(body.input) != len(node.inputs) or len(body.output) != len(node.outputs) + 1:
        raise UnsupportedOpError("Unsupported op Loop")

    iter_name = body.input[0].name
    cond_in_name = body.input[1].name
    cond_out_name = body.output[0].name
    _match_single_node(list(body.node), "Identity", (cond_in_name,), (cond_out_name,))

    state_inputs = tuple(inp.name for inp in body.input[2:])
    state_outputs = tuple(out.name for out in body.output[1:])
    if len(state_inputs) != len(state_outputs):
        raise UnsupportedOpError("Unsupported op Loop")

    sequence_inputs: set[str] = set()
    tensor_inputs: set[str] = set()
    produced_by: dict[str, tuple[str, tuple[str, ...]]] = {}
    for body_node in body.node:
        if body_node.op_type == "Identity" and tuple(body_node.input) == (
            cond_in_name,
        ):
            continue
        if body_node.op_type == "SequenceAt":
            if len(body_node.input) != 2 or body_node.input[1] != iter_name:
                raise UnsupportedOpError("Unsupported op Loop")
            produced_by[body_node.output[0]] = ("sequence_elem", (body_node.input[0],))
            sequence_inputs.add(body_node.input[0])
            continue
        if body_node.op_type == "Identity":
            if len(body_node.input) != 1 or len(body_node.output) != 1:
                raise UnsupportedOpError("Unsupported op Loop")
            source = body_node.input[0]
            if source in state_inputs:
                continue
            if source in produced_by:
                produced_by[body_node.output[0]] = produced_by[source]
            else:
                produced_by[body_node.output[0]] = ("tensor", (source,))
                tensor_inputs.add(source)
            continue
        if body_node.op_type == "Add":
            if len(body_node.input) != 2 or len(body_node.output) != 1:
                raise UnsupportedOpError("Unsupported op Loop")
            produced_by[body_node.output[0]] = (
                "add",
                (body_node.input[0], body_node.input[1]),
            )
            continue
        if body_node.op_type == "Shape":
            if len(body_node.input) != 1 or len(body_node.output) != 1:
                raise UnsupportedOpError("Unsupported op Loop")
            produced_by[body_node.output[0]] = ("shape", (body_node.input[0],))
            continue
        if body_node.op_type == "SequenceInsert":
            continue
        raise UnsupportedOpError("Unsupported op Loop")

    output_kinds: list[str] = []
    output_input0: list[str] = []
    output_input1: list[str | None] = []
    output_input0_is_sequence: list[bool] = []
    output_input1_is_sequence: list[bool] = []
    output_elem_shapes: list[tuple[int, ...]] = []
    output_elem_dtypes = []

    def _resolve_external_source(name: str) -> tuple[str, bool]:
        spec = produced_by.get(name)
        if spec is None:
            return name, name in sequence_inputs
        kind, args = spec
        if kind == "sequence_elem":
            return args[0], True
        if kind == "tensor":
            return args[0], False
        raise UnsupportedOpError("Unsupported op Loop")

    for state_in, state_out, output_name in zip(
        state_inputs, state_outputs, node.outputs
    ):
        insert_nodes = [
            body_node
            for body_node in body.node
            if body_node.op_type == "SequenceInsert"
            and tuple(body_node.input[:1]) == (state_in,)
            and tuple(body_node.output) == (state_out,)
            and len(body_node.input) == 2
        ]
        if len(insert_nodes) != 1:
            raise UnsupportedOpError("Unsupported op Loop")
        inserted_value = insert_nodes[0].input[1]
        spec = produced_by.get(inserted_value)
        if spec is None:
            raise UnsupportedOpError("Unsupported op Loop")
        kind, args = spec
        if kind == "shape":
            src_name, src_is_sequence = _resolve_external_source(args[0])
            src_value = graph.find_value(src_name)
            if src_is_sequence:
                if not isinstance(src_value.type, SequenceType):
                    raise UnsupportedOpError("Unsupported op Loop")
                src_shape = src_value.type.elem.shape
            else:
                src_shape = src_value.type.shape
            output_kinds.append("shape")
            output_input0.append(src_name)
            output_input1.append(None)
            output_input0_is_sequence.append(src_is_sequence)
            output_input1_is_sequence.append(False)
            output_elem_shapes.append((len(src_shape),))
        elif kind in {"sequence_elem", "tensor"}:
            src_name = args[0]
            output_kinds.append("identity")
            output_input0.append(src_name)
            output_input1.append(None)
            output_input0_is_sequence.append(kind == "sequence_elem")
            output_input1_is_sequence.append(False)
            if kind == "sequence_elem":
                src_shape = graph.find_value(src_name).type.elem.shape
            else:
                src_shape = graph.find_value(src_name).type.shape
            output_elem_shapes.append(src_shape)
        elif kind == "add":
            a_name, b_name = args
            a_spec = produced_by.get(a_name)
            b_spec = produced_by.get(b_name)
            if a_spec is None or b_spec is None:
                raise UnsupportedOpError("Unsupported op Loop")
            a_kind, a_args = a_spec
            b_kind, b_args = b_spec
            if a_kind not in {"sequence_elem", "tensor"} or b_kind not in {
                "sequence_elem",
                "tensor",
            }:
                raise UnsupportedOpError("Unsupported op Loop")
            output_kinds.append("add")
            output_input0.append(a_args[0])
            output_input1.append(b_args[0])
            output_input0_is_sequence.append(a_kind == "sequence_elem")
            output_input1_is_sequence.append(b_kind == "sequence_elem")
            if a_kind == "sequence_elem":
                src_shape = graph.find_value(a_args[0]).type.elem.shape
            else:
                src_shape = graph.find_value(a_args[0]).type.shape
            output_elem_shapes.append(src_shape)
        else:
            raise UnsupportedOpError("Unsupported op Loop")

        output_value = graph.find_value(output_name)
        if not isinstance(output_value.type, SequenceType):
            raise UnsupportedOpError("Unsupported op Loop")
        output_elem_dtypes.append(output_value.type.elem.dtype)

    trip_count_shape = value_shape(graph, node.inputs[0], node)
    cond_shape = value_shape(graph, node.inputs[1], node)
    if trip_count_shape not in {(), (1,)} or cond_shape not in {(), (1,)}:
        raise UnsupportedOpError("Unsupported op Loop")

    return LoopSequenceMapOp(
        trip_count=node.inputs[0],
        cond=node.inputs[1],
        input_sequences=tuple(sorted(sequence_inputs)),
        input_tensors=tuple(sorted(tensor_inputs)),
        output_sequences=tuple(node.outputs),
        output_kinds=tuple(output_kinds),
        output_input0=tuple(output_input0),
        output_input1=tuple(output_input1),
        output_input0_is_sequence=tuple(output_input0_is_sequence),
        output_input1_is_sequence=tuple(output_input1_is_sequence),
        output_elem_shapes=tuple(output_elem_shapes),
        output_elem_dtypes=tuple(output_elem_dtypes),
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
    table_values = table.reshape(-1).tolist()
    if not table_values:
        raise UnsupportedOpError("Unsupported op Loop")
    slice_start = constants.get("slice_start")
    if slice_start is not None:
        start_values = slice_start.reshape(-1).tolist()
        if len(start_values) != 1:
            raise UnsupportedOpError("Unsupported op Loop")
        start_index = int(start_values[0])
        if start_index < 0:
            start_index += len(table_values)
        if start_index < 0 or start_index >= len(table_values):
            raise UnsupportedOpError("Unsupported op Loop")
        table_values = [table_values[start_index]] * len(table_values)

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
        table_data=tuple(table_values),
        table_shape=(int(table.shape[0]),),
        elem_shape=input_value.type.elem.shape,
        elem_dtype=input_value.type.elem.dtype,
    )


def _extract_if_default_sequence(
    body: GraphProto,
    if_node: NodeProto,
    opt_state_name: str,
) -> tuple[str, tuple[float | int, ...]] | None:
    """Extract the If node output name and then-branch default sequence values.

    Returns (if_output_name, default_values) when the If node implements the
    optional-sequence pattern: condition=optional_is_none, then-branch creates
    a SequenceConstruct from a Constant, else-branch calls OptionalGetElement
    on the loop state. Returns None if the pattern does not match.
    """
    if len(if_node.input) != 1 or len(if_node.output) != 1:
        return None
    if_output = if_node.output[0]

    then_branch = next(
        (a.g for a in if_node.attribute if a.name == "then_branch"), None
    )
    else_branch = next(
        (a.g for a in if_node.attribute if a.name == "else_branch"), None
    )
    if then_branch is None or else_branch is None:
        return None

    # else-branch must be exactly: OptionalGetElement(opt_state) → output
    if len(else_branch.node) != 1:
        return None
    else_node = else_branch.node[0]
    if (
        else_node.op_type != "OptionalGetElement"
        or list(else_node.input) != [opt_state_name]
        or len(else_node.output) != 1
    ):
        return None

    # then-branch must produce a SequenceConstruct from a single Constant
    const_nodes = {n.output[0]: n for n in then_branch.node if n.op_type == "Constant"}
    seq_nodes = [n for n in then_branch.node if n.op_type == "SequenceConstruct"]
    if len(seq_nodes) != 1:
        return None
    seq_node = seq_nodes[0]
    if len(seq_node.input) != 1 or len(seq_node.output) != 1:
        return None
    const_name = seq_node.input[0]
    const_node = const_nodes.get(const_name)
    if const_node is None:
        return None
    for attr in const_node.attribute:
        if attr.name == "value":
            arr = numpy_helper.to_array(attr.t)
            return if_output, tuple(arr.reshape(-1).tolist())
    return None


def _lower_loop_optional_sequence_insert(
    graph: Graph, node: Node, body: GraphProto
) -> LoopSequenceInsertOp:
    """Lower a Loop with Optional[Sequence] state to LoopSequenceInsertOp.

    Handles the pattern where the loop state is Optional[Sequence] and the body
    uses OptionalHasElement + Not + If to obtain an initial sequence, then calls
    SequenceInsert to append a table element each iteration.
    """
    if len(node.inputs) != 3 or len(node.outputs) != 1:
        raise UnsupportedOpError("Unsupported op Loop")
    if len(body.input) != 3 or len(body.output) != 2:
        raise UnsupportedOpError("Unsupported op Loop")

    iter_name = body.input[0].name
    cond_in_name = body.input[1].name
    opt_state_name = body.input[2].name
    cond_out = body.output[0].name
    seq_out = body.output[1].name
    body_nodes = list(body.node)

    _match_single_node(body_nodes, "Identity", (cond_in_name,), (cond_out,))

    # Detect: OptionalHasElement(opt_state) → has_elem_name
    has_elem_nodes = [
        n
        for n in body_nodes
        if n.op_type == "OptionalHasElement"
        and list(n.input) == [opt_state_name]
        and len(n.output) == 1
    ]
    if len(has_elem_nodes) != 1:
        raise UnsupportedOpError("Unsupported op Loop")
    has_elem_name = has_elem_nodes[0].output[0]

    # Detect: Not(has_elem_name) → is_none_name
    not_nodes = [
        n
        for n in body_nodes
        if n.op_type == "Not"
        and list(n.input) == [has_elem_name]
        and len(n.output) == 1
    ]
    if len(not_nodes) != 1:
        raise UnsupportedOpError("Unsupported op Loop")
    is_none_name = not_nodes[0].output[0]

    # Detect: If(is_none_name) → sequence_name
    if_nodes = [
        n
        for n in body_nodes
        if n.op_type == "If"
        and list(n.input) == [is_none_name]
        and len(n.output) == 1
    ]
    if len(if_nodes) != 1:
        raise UnsupportedOpError("Unsupported op Loop")
    if_node = if_nodes[0]
    extracted = _extract_if_default_sequence(body, if_node, opt_state_name)
    if extracted is None:
        raise UnsupportedOpError("Unsupported op Loop")
    sequence_name, default_values = extracted

    # Validate the rest of the body matches the standard insert pattern
    _match_single_node(body_nodes, "Add", (iter_name, "one"), ("end",))
    _match_single_node(body_nodes, "Unsqueeze", ("end", "axes"), ("slice_end",))
    _match_single_node(
        body_nodes, "Slice", ("x", "slice_start", "slice_end"), ("slice_out",)
    )
    _match_single_node(
        body_nodes, "SequenceInsert", (sequence_name, "slice_out"), (seq_out,)
    )

    constants = _const_value_map(body)
    table = constants.get("x")
    if table is None or len(table.shape) != 1:
        raise UnsupportedOpError("Unsupported op Loop")
    table_values = table.reshape(-1).tolist()
    if not table_values:
        raise UnsupportedOpError("Unsupported op Loop")
    slice_start = constants.get("slice_start")
    if slice_start is not None:
        start_values = slice_start.reshape(-1).tolist()
        if len(start_values) != 1:
            raise UnsupportedOpError("Unsupported op Loop")
        start_index = int(start_values[0])
        if start_index < 0:
            start_index += len(table_values)
        if start_index < 0 or start_index >= len(table_values):
            raise UnsupportedOpError("Unsupported op Loop")
        table_values = [table_values[start_index]] * len(table_values)

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
    if not input_value.type.is_optional:
        raise UnsupportedOpError("Unsupported op Loop")
    if input_value.type.elem.shape != output_value.type.elem.shape:
        raise UnsupportedOpError("Unsupported op Loop")
    if input_value.type.elem.shape != ():
        raise UnsupportedOpError("Unsupported op Loop")

    elem_dtype = input_value.type.elem.dtype
    if elem_dtype.name not in {"F32", "F64", "I32", "I64", "I16"}:
        raise UnsupportedOpError("Unsupported op Loop")

    input_sequence_present = f"{node.inputs[2]}_present"
    return LoopSequenceInsertOp(
        trip_count=node.inputs[0],
        cond=node.inputs[1],
        input_sequence=node.inputs[2],
        output_sequence=node.outputs[0],
        table_data=tuple(table_values),
        table_shape=(int(table.shape[0]),),
        elem_shape=input_value.type.elem.shape,
        elem_dtype=elem_dtype,
        input_sequence_present=input_sequence_present,
        default_sequence_data=default_values,
    )


@register_lowering("Loop")
def lower_loop(
    graph: Graph, node: Node
) -> LoopRangeOp | LoopSequenceInsertOp | LoopSequenceMapOp:
    body = _find_body(node)
    try:
        return _lower_loop_range(graph, node, body)
    except UnsupportedOpError:
        try:
            return _lower_loop_tensor_scan_add(graph, node, body)
        except UnsupportedOpError:
            try:
                return _lower_loop_sequence_insert(graph, node, body)
            except UnsupportedOpError:
                try:
                    return _lower_loop_optional_sequence_insert(graph, node, body)
                except UnsupportedOpError:
                    return _lower_loop_sequence_map(graph, node, body)

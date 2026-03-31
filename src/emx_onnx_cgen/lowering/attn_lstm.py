from __future__ import annotations

from shared.scalar_types import ScalarType  # noqa: F401 - kept for type annotations

from ..errors import UnsupportedOpError
from ..ir.model import Graph, Node
from ..ir.ops import AttnLSTMOp
from .common import node_dtype, optional_name, value_dtype, value_shape
from .lstm import (
    DEFAULT_ACTIVATIONS,
    _normalize_activation_names,
    _normalize_direction,
    _resolve_activation_params,
)
from .registry import register_lowering


@register_lowering("AttnLSTM")
def lower_attn_lstm(graph: Graph, node: Node) -> AttnLSTMOp:
    if len(node.inputs) < 14 or len(node.outputs) < 1:
        raise UnsupportedOpError(
            "AttnLSTM expects 14 inputs and 1-3 outputs"
        )

    input_x = node.inputs[0]
    input_w = node.inputs[1]
    input_r = node.inputs[2]
    input_b = optional_name(node.inputs, 3)
    input_sequence_lens = optional_name(node.inputs, 4)
    input_initial_h = optional_name(node.inputs, 5)
    input_initial_c = optional_name(node.inputs, 6)
    # index 7 = P (peephole, optional) - not used
    input_qw = node.inputs[8]
    input_mw = node.inputs[9]
    input_v = node.inputs[10]
    input_m = node.inputs[11]
    input_memory_seq_lens = optional_name(node.inputs, 12)
    input_aw = optional_name(node.inputs, 13)

    output_y = optional_name(node.outputs, 0)
    output_y_h = optional_name(node.outputs, 1)
    output_y_c = optional_name(node.outputs, 2)

    if output_y is None and output_y_h is None and output_y_c is None:
        raise UnsupportedOpError("AttnLSTM expects at least one output")

    op_dtype = node_dtype(
        graph,
        node,
        input_x,
        input_w,
        input_r,
        input_qw,
        input_mw,
        input_v,
        input_m,
        *(n for n in (input_b, input_initial_h, input_initial_c) if n),
        *(n for n in (output_y, output_y_h, output_y_c) if n),
    )
    if not op_dtype.is_float:
        raise UnsupportedOpError("AttnLSTM supports float inputs only")

    x_shape = value_shape(graph, input_x, node)
    if len(x_shape) != 3:
        raise UnsupportedOpError("AttnLSTM X must be rank 3 [seq, batch, input]")
    seq_length, batch_size, _ = x_shape

    w_shape = value_shape(graph, input_w, node)
    if len(w_shape) != 3:
        raise UnsupportedOpError("AttnLSTM W must be rank 3")
    num_directions = w_shape[0]
    if w_shape[1] % 4 != 0:
        raise UnsupportedOpError("AttnLSTM W second dim must be divisible by 4")
    hidden_size_attr = node.attrs.get("hidden_size")
    if hidden_size_attr is None:
        hidden_size = w_shape[1] // 4
    else:
        hidden_size = int(hidden_size_attr)

    total_input_size = w_shape[2]  # input_only_size + attn_ctx_size

    # QW: [num_dirs, hidden_size, attn_dim]
    qw_shape = value_shape(graph, input_qw, node)
    if len(qw_shape) != 3 or qw_shape[0] != num_directions or qw_shape[1] != hidden_size:
        raise UnsupportedOpError(
            f"AttnLSTM QW shape must be [{num_directions}, {hidden_size}, attn_dim], got {qw_shape}"
        )
    attn_dim = qw_shape[2]

    # MW: [num_dirs, memory_depth, attn_dim]
    mw_shape = value_shape(graph, input_mw, node)
    if len(mw_shape) != 3 or mw_shape[0] != num_directions or mw_shape[2] != attn_dim:
        raise UnsupportedOpError(
            f"AttnLSTM MW shape must be [{num_directions}, memory_depth, {attn_dim}], got {mw_shape}"
        )
    memory_depth = mw_shape[1]

    # V: [num_dirs, attn_dim]
    v_shape = value_shape(graph, input_v, node)
    if v_shape != (num_directions, attn_dim):
        raise UnsupportedOpError(
            f"AttnLSTM V shape must be [{num_directions}, {attn_dim}], got {v_shape}"
        )

    # M: [batch_size, memory_seq, memory_depth]
    m_shape = value_shape(graph, input_m, node)
    if len(m_shape) != 3 or m_shape[0] != batch_size or m_shape[2] != memory_depth:
        raise UnsupportedOpError(
            f"AttnLSTM M shape must be [{batch_size}, memory_seq, {memory_depth}], got {m_shape}"
        )
    memory_seq_length = m_shape[1]

    # AW (optional): [num_dirs, hidden_size + memory_depth, attn_ctx_size]
    attn_ctx_size: int
    if input_aw is not None:
        aw_shape = value_shape(graph, input_aw, node)
        if (
            len(aw_shape) != 3
            or aw_shape[0] != num_directions
            or aw_shape[1] != hidden_size + memory_depth
        ):
            raise UnsupportedOpError(
                f"AttnLSTM AW shape must be [{num_directions}, {hidden_size + memory_depth}, ctx_size], got {aw_shape}"
            )
        attn_ctx_size = aw_shape[2]
    else:
        # Without AW, context is raw memory context (memory_depth)
        attn_ctx_size = memory_depth

    # total_input_size = input_only_size + attn_ctx_size
    input_only_size = total_input_size - attn_ctx_size
    if input_only_size < 0:
        raise UnsupportedOpError(
            f"AttnLSTM input size {total_input_size} < attn_ctx_size {attn_ctx_size}"
        )

    direction = _normalize_direction(node.attrs.get("direction", "forward"))
    if direction not in {"forward", "reverse", "bidirectional"}:
        raise UnsupportedOpError(f"AttnLSTM unsupported direction: {direction}")
    if direction == "bidirectional" and num_directions != 2:
        raise UnsupportedOpError("AttnLSTM bidirectional requires num_directions=2")
    if direction in {"forward", "reverse"} and num_directions != 1:
        raise UnsupportedOpError("AttnLSTM forward/reverse requires num_directions=1")

    # Activations
    activations_attr = node.attrs.get("activations")
    if activations_attr is None:
        activations = list(DEFAULT_ACTIVATIONS)
    else:
        activations = _normalize_activation_names(activations_attr)
    if num_directions == 1:
        if len(activations) != 3:
            raise UnsupportedOpError("AttnLSTM needs 3 activations for single-direction")
    else:
        if len(activations) == 3:
            activations = activations * 2
        elif len(activations) != 6:
            raise UnsupportedOpError("AttnLSTM needs 6 activations for bidirectional")

    activation_alpha = node.attrs.get("activation_alpha")
    activation_beta = node.attrs.get("activation_beta")
    activation_kinds, activation_alphas, activation_betas = _resolve_activation_params(
        activations, activation_alpha, activation_beta
    )

    input_forget = int(node.attrs.get("input_forget", 0))

    # sequence_lens dtype
    sequence_lens_dtype = None
    if input_sequence_lens is not None:
        sequence_lens_dtype = value_dtype(graph, input_sequence_lens, node)

    memory_seq_lens_dtype = None
    if input_memory_seq_lens is not None:
        memory_seq_lens_dtype = value_dtype(graph, input_memory_seq_lens, node)

    return AttnLSTMOp(
        input_x=input_x,
        input_w=input_w,
        input_r=input_r,
        input_b=input_b,
        input_sequence_lens=input_sequence_lens,
        input_initial_h=input_initial_h,
        input_initial_c=input_initial_c,
        input_qw=input_qw,
        input_mw=input_mw,
        input_v=input_v,
        input_m=input_m,
        input_memory_seq_lens=input_memory_seq_lens,
        input_aw=input_aw,
        output_y=output_y,
        output_y_h=output_y_h,
        output_y_c=output_y_c,
        seq_length=seq_length,
        batch_size=batch_size,
        input_only_size=input_only_size,
        hidden_size=hidden_size,
        attn_ctx_size=attn_ctx_size,
        attn_dim=attn_dim,
        memory_depth=memory_depth,
        memory_seq_length=memory_seq_length,
        num_directions=num_directions,
        direction=direction,
        input_forget=input_forget,
        clip=None,
        activation_kinds=activation_kinds,
        activation_alphas=activation_alphas,
        activation_betas=activation_betas,
        dtype=op_dtype,
        sequence_lens_dtype=sequence_lens_dtype,
        memory_seq_lens_dtype=memory_seq_lens_dtype,
    )

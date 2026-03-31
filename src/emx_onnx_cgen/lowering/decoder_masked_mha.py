from __future__ import annotations

import math

from shared.scalar_types import ScalarType as _ScalarType  # noqa: F401

from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from ..ir.ops import DecoderMaskedMHAOp
from .common import node_dtype, optional_name, value_shape
from .registry import register_lowering


@register_lowering("DecoderMaskedMultiHeadAttention")
def lower_decoder_masked_mha(graph: Graph, node: Node) -> DecoderMaskedMHAOp:
    # Inputs:
    # 0: query  [batch, 1, hidden]
    # 1: key    self=[batch,1,hidden], cross=[batch,heads,kv_seq,head_size]
    # 2: value  self=[batch,1,hidden], cross=[batch,heads,kv_seq,head_size]
    # 3: mask_index [batch, total_seq] or [batch, kv_seq]
    # 4: attention_bias (optional) [1,1,1,total_seq]
    # 5: past_key (optional) [batch,heads,max_seq,head_size]
    # 6: past_value (optional) [batch,heads,max_seq,head_size]
    # 7: past_sequence_length (optional) [1]
    # Outputs:
    # 0: output [batch,1,hidden]
    # 1: present_key [batch,heads,max_seq,head_size]
    # 2: present_value [batch,heads,max_seq,head_size]
    # 3: qk [batch,heads,1,total_seq] (optional, output_qk=1)

    if len(node.inputs) < 4:
        raise UnsupportedOpError("DecoderMaskedMultiHeadAttention needs at least 4 inputs")

    query_name = node.inputs[0]
    key_name = node.inputs[1]
    value_name = node.inputs[2]
    mask_index_name = node.inputs[3]
    attn_bias_name = optional_name(node.inputs, 4)
    past_key_name = optional_name(node.inputs, 5)
    past_value_name = optional_name(node.inputs, 6)
    past_seq_len_name = optional_name(node.inputs, 7)

    output_name = optional_name(node.outputs, 0)
    present_key_name = optional_name(node.outputs, 1)
    present_value_name = optional_name(node.outputs, 2)
    qk_output_name = optional_name(node.outputs, 3)

    if output_name is None:
        raise UnsupportedOpError("DecoderMaskedMultiHeadAttention requires output[0]")

    op_dtype = node_dtype(graph, node, query_name, key_name, value_name, output_name)
    if not op_dtype.is_float:
        raise UnsupportedOpError("DecoderMaskedMultiHeadAttention supports float only")

    num_heads_attr = node.attrs.get("num_heads")
    if num_heads_attr is None:
        raise UnsupportedOpError("DecoderMaskedMultiHeadAttention requires num_heads attribute")
    num_heads = int(num_heads_attr)

    mask_filter_value = float(node.attrs.get("mask_filter_value", -10000.0))
    scale_attr = float(node.attrs.get("scale", 0.0))
    output_qk = bool(int(node.attrs.get("output_qk", 0)))

    q_shape = value_shape(graph, query_name, node)
    if len(q_shape) != 3 or q_shape[1] != 1:
        raise UnsupportedOpError(
            f"DecoderMaskedMultiHeadAttention: query must be [batch,1,hidden], got {q_shape}"
        )
    batch = q_shape[0]
    hidden_size = q_shape[2]
    if hidden_size % num_heads != 0:
        raise UnsupportedOpError(
            f"DecoderMaskedMultiHeadAttention: hidden_size {hidden_size} not divisible by num_heads {num_heads}"
        )
    head_size = hidden_size // num_heads

    scale_value = scale_attr if scale_attr != 0.0 else 1.0 / math.sqrt(head_size)

    key_shape = value_shape(graph, key_name, node)

    # Determine self-attention vs cross-attention
    is_self_attn: bool
    kv_seq: int
    total_seq: int

    if len(key_shape) == 3:
        # Self-attention: key = [batch, 1, hidden]
        is_self_attn = True
        if past_key_name is None or past_seq_len_name is None:
            raise UnsupportedOpError(
                "DecoderMaskedMultiHeadAttention self-attention requires past_key and past_sequence_length"
            )
        pk_shape = value_shape(graph, past_key_name, node)
        if len(pk_shape) != 4 or pk_shape[0] != batch or pk_shape[1] != num_heads or pk_shape[3] != head_size:
            raise ShapeInferenceError(
                f"DecoderMaskedMultiHeadAttention: past_key shape mismatch, got {pk_shape}"
            )
        max_seq = pk_shape[2]
        # total_seq = max_seq (shared buffer), but kv_seq is dynamic (past_seq_len+1)
        # For C codegen we need static total_seq for cache output size
        total_seq = max_seq
        kv_seq = max_seq  # used for present_key/value shape
    elif len(key_shape) == 4:
        # Cross-attention: key = [batch, heads, kv_seq, head_size]
        is_self_attn = False
        if key_shape[0] != batch or key_shape[1] != num_heads or key_shape[3] != head_size:
            raise ShapeInferenceError(
                f"DecoderMaskedMultiHeadAttention: cross-attn key shape mismatch, got {key_shape}"
            )
        kv_seq = key_shape[2]
        total_seq = kv_seq
    else:
        raise UnsupportedOpError(
            f"DecoderMaskedMultiHeadAttention: unsupported key rank {len(key_shape)}"
        )

    return DecoderMaskedMHAOp(
        query=query_name,
        key=key_name,
        value=value_name,
        mask_index=mask_index_name,
        attn_bias=attn_bias_name,
        past_key=past_key_name,
        past_value=past_value_name,
        past_seq_len_input=past_seq_len_name,
        output=output_name,
        present_key=present_key_name,
        present_value=present_value_name,
        qk_output=qk_output_name,
        batch=batch,
        num_heads=num_heads,
        head_size=head_size,
        hidden_size=hidden_size,
        is_self_attn=is_self_attn,
        kv_seq=kv_seq,
        total_seq=total_seq,
        mask_filter_value=mask_filter_value,
        scale_value=scale_value,
        output_qk=output_qk,
        dtype=op_dtype,
    )

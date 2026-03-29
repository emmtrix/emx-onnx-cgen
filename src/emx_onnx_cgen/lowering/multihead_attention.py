from __future__ import annotations

import math
from dataclasses import dataclass

from shared.scalar_types import ScalarType

from ..ir.ops import MultiHeadAttentionOp
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from .common import node_dtype as _node_dtype
from .common import optional_name as _optional_name
from .common import value_dtype as _value_dtype
from .common import value_shape as _value_shape
from .registry import register_lowering


@dataclass(frozen=True)
class MultiHeadAttentionSpec:
    batch: int
    q_seq: int
    num_heads: int
    qk_head_size: int
    v_head_size: int
    q_hidden_size: int
    k_hidden_size: int
    v_hidden_size: int
    kv_3d: bool
    kv_seq: int
    past_seq: int
    total_seq: int
    has_bias: bool
    has_past: bool
    has_present_key: bool
    has_present_value: bool
    has_key_padding_mask: bool
    has_attention_bias: bool
    unidirectional: bool
    mask_filter_value: float
    scale: float


def resolve_multihead_attention_spec(
    graph: Graph, node: Node, dtype: ScalarType
) -> MultiHeadAttentionSpec:
    """Resolve and validate com.microsoft::MultiHeadAttention parameters."""
    if not dtype.is_float:
        raise UnsupportedOpError("Unsupported op MultiHeadAttention")
    if len(node.inputs) < 3 or len(node.outputs) < 1:
        raise UnsupportedOpError("Unsupported op MultiHeadAttention")

    supported_attrs = {
        "num_heads",
        "unidirectional",
        "qkv_hidden_sizes",
        "scale",
        "mask_filter_value",
        "past_present_share_buffer",
        "do_rotary",
        "rotary_embedding_dim",
    }
    if set(node.attrs) - supported_attrs:
        raise UnsupportedOpError("Unsupported op MultiHeadAttention")
    if int(node.attrs.get("do_rotary", 0)) != 0:
        raise UnsupportedOpError("Unsupported op MultiHeadAttention")

    num_heads = node.attrs.get("num_heads")
    if num_heads is None:
        raise UnsupportedOpError("Unsupported op MultiHeadAttention")
    num_heads = int(num_heads)

    query_shape = _value_shape(graph, node.inputs[0], node)
    key_shape = _value_shape(graph, node.inputs[1], node)
    value_shape = _value_shape(graph, node.inputs[2], node)

    if len(query_shape) != 3:
        raise ShapeInferenceError(
            "MultiHeadAttention query must be 3D (batch, q_seq, q_hidden)"
        )
    batch, q_seq, q_hidden_size = query_shape

    key_rank = len(key_shape)
    value_rank = len(value_shape)
    if key_rank not in (3, 4):
        raise ShapeInferenceError("MultiHeadAttention key must be 3D or 4D")
    if value_rank not in (3, 4):
        raise ShapeInferenceError("MultiHeadAttention value must be 3D or 4D")
    if key_rank != value_rank:
        raise ShapeInferenceError(
            "MultiHeadAttention key and value must have the same rank"
        )

    kv_3d = key_rank == 3

    if kv_3d:
        kv_seq = key_shape[1]
        k_hidden_size = key_shape[2]
        v_hidden_size = value_shape[2]
        if key_shape[0] != batch:
            raise ShapeInferenceError(
                "MultiHeadAttention key batch dimension must match query"
            )
        if value_shape[0] != batch or value_shape[1] != kv_seq:
            raise ShapeInferenceError("MultiHeadAttention value shape must match key")
    else:
        # 4D: [batch, num_heads, kv_seq, head_size]
        if key_shape[0] != batch or key_shape[1] != num_heads:
            raise ShapeInferenceError(
                "MultiHeadAttention 4D key must be [batch, num_heads, kv_seq, head_size]"
            )
        if value_shape[0] != batch or value_shape[1] != num_heads:
            raise ShapeInferenceError(
                "MultiHeadAttention 4D value must be [batch, num_heads, kv_seq, v_head_size]"
            )
        if key_shape[2] != value_shape[2]:
            raise ShapeInferenceError(
                "MultiHeadAttention 4D key and value kv_seq must match"
            )
        kv_seq = key_shape[2]
        k_hidden_size = num_heads * key_shape[3]
        v_hidden_size = num_heads * value_shape[3]

    qkv_hidden_sizes = node.attrs.get("qkv_hidden_sizes")
    if qkv_hidden_sizes is not None:
        qkv_hidden_sizes = [int(x) for x in qkv_hidden_sizes]
        if len(qkv_hidden_sizes) != 3:
            raise UnsupportedOpError("Unsupported op MultiHeadAttention")
        expected_q, expected_k, expected_v = qkv_hidden_sizes
        if expected_q != q_hidden_size:
            raise ShapeInferenceError(
                "MultiHeadAttention qkv_hidden_sizes[0] must match query hidden size"
            )
        if expected_k != k_hidden_size:
            raise ShapeInferenceError(
                "MultiHeadAttention qkv_hidden_sizes[1] must match key hidden size"
            )
        if expected_v != v_hidden_size:
            raise ShapeInferenceError(
                "MultiHeadAttention qkv_hidden_sizes[2] must match value hidden size"
            )

    if q_hidden_size % num_heads != 0:
        raise ShapeInferenceError(
            "MultiHeadAttention Q hidden size must be divisible by num_heads"
        )
    if k_hidden_size % num_heads != 0:
        raise ShapeInferenceError(
            "MultiHeadAttention K hidden size must be divisible by num_heads"
        )
    if v_hidden_size % num_heads != 0:
        raise ShapeInferenceError(
            "MultiHeadAttention V hidden size must be divisible by num_heads"
        )

    qk_head_size = q_hidden_size // num_heads
    k_head_size = k_hidden_size // num_heads
    v_head_size = v_hidden_size // num_heads
    if qk_head_size != k_head_size:
        raise ShapeInferenceError("MultiHeadAttention Q/K head sizes must match")
    if not kv_3d and key_shape[3] != qk_head_size:
        raise ShapeInferenceError(
            "MultiHeadAttention 4D key head_size must equal q_hidden / num_heads"
        )

    bias_name = _optional_name(node.inputs, 3)
    has_bias = bias_name is not None
    if has_bias:
        bias_shape = _value_shape(graph, bias_name, node)
        if len(bias_shape) != 1:
            raise ShapeInferenceError("MultiHeadAttention bias must be 1D")
        expected_bias = q_hidden_size + k_hidden_size + v_hidden_size
        if bias_shape[0] != expected_bias:
            raise ShapeInferenceError(
                f"MultiHeadAttention bias size must be {expected_bias}, "
                f"got {bias_shape[0]}"
            )

    mask_name = _optional_name(node.inputs, 4)
    has_key_padding_mask = mask_name is not None
    if has_key_padding_mask:
        mask_shape = _value_shape(graph, mask_name, node)
        mask_dtype = _value_dtype(graph, mask_name, node)
        if mask_dtype != ScalarType.I32:
            raise UnsupportedOpError(
                "MultiHeadAttention key_padding_mask must be int32"
            )
        if len(mask_shape) != 2 or mask_shape[0] != batch:
            raise ShapeInferenceError(
                "MultiHeadAttention key_padding_mask must be 2D [batch, kv_seq]"
            )

    attn_bias_name = _optional_name(node.inputs, 5)
    has_attention_bias = attn_bias_name is not None

    past_key_name = _optional_name(node.inputs, 6)
    past_value_name = _optional_name(node.inputs, 7)
    has_past = past_key_name is not None or past_value_name is not None
    if (past_key_name is None) != (past_value_name is None):
        raise UnsupportedOpError(
            "MultiHeadAttention past_key and past_value must both be present or absent"
        )

    past_seq = 0
    if has_past:
        if not kv_3d:
            raise UnsupportedOpError(
                "MultiHeadAttention past is only supported with 3D key/value"
            )
        past_key_shape = _value_shape(graph, past_key_name, node)
        past_value_shape = _value_shape(graph, past_value_name, node)
        if len(past_key_shape) != 4:
            raise ShapeInferenceError(
                "MultiHeadAttention past_key must be 4D [batch, num_heads, past_seq, head_size]"
            )
        if past_key_shape[0] != batch or past_key_shape[1] != num_heads:
            raise ShapeInferenceError(
                "MultiHeadAttention past_key batch/heads must match"
            )
        if past_key_shape[3] != qk_head_size:
            raise ShapeInferenceError(
                "MultiHeadAttention past_key head_size must match"
            )
        past_seq = past_key_shape[2]
        if past_value_shape != (batch, num_heads, past_seq, v_head_size):
            raise ShapeInferenceError(
                "MultiHeadAttention past_value shape must match past_key"
            )

    total_seq = kv_seq + past_seq

    if has_attention_bias:
        attn_bias_shape = _value_shape(graph, attn_bias_name, node)
        expected_attn_bias = (batch, num_heads, q_seq, total_seq)
        if attn_bias_shape != expected_attn_bias:
            raise ShapeInferenceError(
                f"MultiHeadAttention attention_bias shape must be {expected_attn_bias}, "
                f"got {attn_bias_shape}"
            )

    present_key_name = _optional_name(node.outputs, 1)
    present_value_name = _optional_name(node.outputs, 2)
    has_present_key = present_key_name is not None
    has_present_value = present_value_name is not None
    if has_present_key:
        present_key_shape = _value_shape(graph, present_key_name, node)
        expected_pk = (batch, num_heads, total_seq, qk_head_size)
        if kv_3d and present_key_shape != expected_pk:
            raise ShapeInferenceError(
                f"MultiHeadAttention present_key shape must be {expected_pk}, "
                f"got {present_key_shape}"
            )
    if has_present_value:
        present_value_shape = _value_shape(graph, present_value_name, node)
        expected_pv = (batch, num_heads, total_seq, v_head_size)
        if kv_3d and present_value_shape != expected_pv:
            raise ShapeInferenceError(
                f"MultiHeadAttention present_value shape must be {expected_pv}, "
                f"got {present_value_shape}"
            )

    output_shape = _value_shape(graph, node.outputs[0], node)
    expected_output = (batch, q_seq, v_hidden_size)
    if output_shape != expected_output:
        raise ShapeInferenceError(
            f"MultiHeadAttention output shape must be {expected_output}, "
            f"got {output_shape}"
        )

    scale = float(node.attrs.get("scale", 1.0 / math.sqrt(qk_head_size)))
    unidirectional = bool(int(node.attrs.get("unidirectional", 0)))
    mask_filter_value = float(node.attrs.get("mask_filter_value", -10000.0))

    return MultiHeadAttentionSpec(
        batch=batch,
        q_seq=q_seq,
        num_heads=num_heads,
        qk_head_size=qk_head_size,
        v_head_size=v_head_size,
        q_hidden_size=q_hidden_size,
        k_hidden_size=k_hidden_size,
        v_hidden_size=v_hidden_size,
        kv_3d=kv_3d,
        kv_seq=kv_seq,
        past_seq=past_seq,
        total_seq=total_seq,
        has_bias=has_bias,
        has_past=has_past,
        has_present_key=has_present_key,
        has_present_value=has_present_value,
        has_key_padding_mask=has_key_padding_mask,
        has_attention_bias=has_attention_bias,
        unidirectional=unidirectional,
        mask_filter_value=mask_filter_value,
        scale=scale,
    )


@register_lowering("MultiHeadAttention")
def lower_multihead_attention(graph: Graph, node: Node) -> MultiHeadAttentionOp:
    """Lower com.microsoft::MultiHeadAttention contrib op."""
    query = node.inputs[0]
    key = node.inputs[1]
    value = node.inputs[2]
    output = node.outputs[0]
    op_dtype = _node_dtype(graph, node, query, output)
    spec = resolve_multihead_attention_spec(graph, node, op_dtype)
    bias_name = _optional_name(node.inputs, 3)
    mask_name = _optional_name(node.inputs, 4)
    attn_bias_name = _optional_name(node.inputs, 5)
    past_key_name = _optional_name(node.inputs, 6)
    past_value_name = _optional_name(node.inputs, 7)
    present_key_name = _optional_name(node.outputs, 1)
    present_value_name = _optional_name(node.outputs, 2)
    return MultiHeadAttentionOp(
        query=query,
        key=key,
        value=value,
        bias=bias_name,
        key_padding_mask=mask_name,
        attention_bias=attn_bias_name,
        past_key=past_key_name,
        past_value=past_value_name,
        output=output,
        present_key=present_key_name,
        present_value=present_value_name,
        batch=spec.batch,
        q_seq=spec.q_seq,
        num_heads=spec.num_heads,
        qk_head_size=spec.qk_head_size,
        v_head_size=spec.v_head_size,
        q_hidden_size=spec.q_hidden_size,
        k_hidden_size=spec.k_hidden_size,
        v_hidden_size=spec.v_hidden_size,
        kv_3d=spec.kv_3d,
        kv_seq=spec.kv_seq,
        past_seq=spec.past_seq,
        total_seq=spec.total_seq,
        scale=spec.scale,
        unidirectional=spec.unidirectional,
        mask_filter_value=spec.mask_filter_value,
        dtype=op_dtype,
    )

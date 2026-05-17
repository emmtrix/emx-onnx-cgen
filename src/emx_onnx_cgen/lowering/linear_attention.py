from __future__ import annotations

import math

from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from ..ir.ops import LinearAttentionOp
from .common import node_dtype as _node_dtype
from .common import optional_name as _optional_name
from .common import value_shape as _value_shape
from .registry import register_lowering

_SUPPORTED_UPDATE_RULES = {"linear", "gated", "delta", "gated_delta"}


def _decode_update_rule(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    raise UnsupportedOpError("Unsupported op LinearAttention")


@register_lowering("LinearAttention")
def lower_linear_attention(graph: Graph, node: Node) -> LinearAttentionOp:
    if not 3 <= len(node.inputs) <= 6:
        raise UnsupportedOpError("Unsupported op LinearAttention")
    if len(node.outputs) < 2:
        raise UnsupportedOpError("LinearAttention requires present_state output")

    query = node.inputs[0]
    key = node.inputs[1]
    value = node.inputs[2]
    past_state = _optional_name(node.inputs, 3)
    decay = _optional_name(node.inputs, 4)
    beta = _optional_name(node.inputs, 5)
    output = node.outputs[0]
    present_state = _optional_name(node.outputs, 1)
    if present_state is None:
        raise UnsupportedOpError("LinearAttention requires present_state output")

    supported_attrs = {
        "chunk_size",
        "kv_num_heads",
        "q_num_heads",
        "scale",
        "update_rule",
    }
    if set(node.attrs) - supported_attrs:
        raise UnsupportedOpError("Unsupported op LinearAttention")

    op_dtype = _node_dtype(
        graph, node, query, key, value, output, present_state, past_state or ""
    )
    if not op_dtype.is_float:
        raise UnsupportedOpError("Unsupported op LinearAttention")

    q_heads = node.attrs.get("q_num_heads")
    kv_heads = node.attrs.get("kv_num_heads")
    if q_heads is None or kv_heads is None:
        raise UnsupportedOpError("Unsupported op LinearAttention")
    q_heads = int(q_heads)
    kv_heads = int(kv_heads)
    if q_heads <= 0 or kv_heads <= 0:
        raise ShapeInferenceError(
            "LinearAttention q_num_heads/kv_num_heads must be positive"
        )

    chunk_size = int(node.attrs.get("chunk_size", 64))
    if chunk_size <= 0:
        raise ShapeInferenceError("LinearAttention chunk_size must be positive")

    update_rule = _decode_update_rule(node.attrs.get("update_rule", "gated_delta"))
    if update_rule not in _SUPPORTED_UPDATE_RULES:
        raise UnsupportedOpError("Unsupported op LinearAttention")

    query_shape = _value_shape(graph, query, node)
    key_shape = _value_shape(graph, key, node)
    value_shape = _value_shape(graph, value, node)
    if len(query_shape) != 3 or len(key_shape) != 3 or len(value_shape) != 3:
        raise ShapeInferenceError("LinearAttention query/key/value must be 3D")
    batch, seq_len, q_hidden_size = query_shape
    if (
        key_shape[0] != batch
        or value_shape[0] != batch
        or key_shape[1] != seq_len
        or value_shape[1] != seq_len
    ):
        raise ShapeInferenceError(
            "LinearAttention query/key/value batch and sequence dims must match"
        )

    k_hidden_size = key_shape[2]
    v_hidden_size = value_shape[2]
    if q_hidden_size % q_heads != 0:
        raise ShapeInferenceError(
            "LinearAttention query hidden size must be divisible by q_num_heads"
        )
    qk_head_size = q_hidden_size // q_heads
    if k_hidden_size % qk_head_size != 0:
        raise ShapeInferenceError(
            "LinearAttention key hidden size must be divisible by query head size"
        )
    if v_hidden_size % kv_heads != 0:
        raise ShapeInferenceError(
            "LinearAttention value hidden size must be divisible by kv_num_heads"
        )

    n_k_heads = k_hidden_size // qk_head_size
    if kv_heads % n_k_heads != 0:
        raise ShapeInferenceError(
            "LinearAttention kv_num_heads must be divisible by key head count"
        )
    v_head_size = v_hidden_size // kv_heads
    if q_heads >= kv_heads:
        if q_heads % kv_heads != 0:
            raise ShapeInferenceError(
                "LinearAttention q_num_heads must be divisible by kv_num_heads"
            )
        head_group_size = q_heads // kv_heads
    else:
        if kv_heads % q_heads != 0:
            raise ShapeInferenceError(
                "LinearAttention kv_num_heads must be divisible by q_num_heads"
            )
        head_group_size = 0
    kv_per_k_head = kv_heads // n_k_heads
    output_hidden_size = max(q_heads, kv_heads) * v_head_size

    expected_state_shape = (batch, kv_heads, qk_head_size, v_head_size)
    if (
        past_state is not None
        and _value_shape(graph, past_state, node) != expected_state_shape
    ):
        raise ShapeInferenceError(
            f"LinearAttention past_state shape must be {expected_state_shape}"
        )
    if _value_shape(graph, present_state, node) != expected_state_shape:
        raise ShapeInferenceError(
            f"LinearAttention present_state shape must be {expected_state_shape}"
        )

    expected_output_shape = (batch, seq_len, output_hidden_size)
    if _value_shape(graph, output, node) != expected_output_shape:
        raise ShapeInferenceError(
            f"LinearAttention output shape must be {expected_output_shape}"
        )

    has_decay = update_rule in {"gated", "gated_delta"}
    has_beta = update_rule in {"delta", "gated_delta"}
    if has_decay != (decay is not None):
        raise UnsupportedOpError(
            "LinearAttention decay input does not match update_rule"
        )
    if has_beta != (beta is not None):
        raise UnsupportedOpError(
            "LinearAttention beta input does not match update_rule"
        )

    decay_last_dim = 0
    if decay is not None:
        decay_shape = _value_shape(graph, decay, node)
        if len(decay_shape) != 3 or decay_shape[:2] != (batch, seq_len):
            raise ShapeInferenceError(
                "LinearAttention decay must have shape [batch, seq_len, ...]"
            )
        decay_last_dim = decay_shape[2]
        if decay_last_dim not in {1, kv_heads, k_hidden_size}:
            raise ShapeInferenceError(
                "LinearAttention decay last dimension must be 1, kv_num_heads, "
                "or kv_num_heads * head_size"
            )

    beta_last_dim = 0
    if beta is not None:
        beta_shape = _value_shape(graph, beta, node)
        if len(beta_shape) != 3 or beta_shape[:2] != (batch, seq_len):
            raise ShapeInferenceError(
                "LinearAttention beta must have shape [batch, seq_len, ...]"
            )
        beta_last_dim = beta_shape[2]
        if beta_last_dim not in {1, kv_heads}:
            raise ShapeInferenceError(
                "LinearAttention beta last dimension must be 1 or kv_num_heads"
            )

    scale = float(node.attrs.get("scale", 0.0))
    if scale == 0.0:
        scale = 1.0 / math.sqrt(qk_head_size)

    return LinearAttentionOp(
        query=query,
        key=key,
        value=value,
        past_state=past_state,
        decay=decay,
        beta=beta,
        output=output,
        present_state=present_state,
        batch=batch,
        seq_len=seq_len,
        q_heads=q_heads,
        kv_heads=kv_heads,
        qk_head_size=qk_head_size,
        v_head_size=v_head_size,
        q_hidden_size=q_hidden_size,
        n_k_heads=n_k_heads,
        k_hidden_size=k_hidden_size,
        v_hidden_size=v_hidden_size,
        kv_per_k_head=kv_per_k_head,
        head_group_size=head_group_size,
        output_hidden_size=output_hidden_size,
        scale=scale,
        update_rule=update_rule,
        decay_last_dim=decay_last_dim,
        beta_last_dim=beta_last_dim,
        dtype=op_dtype,
    )

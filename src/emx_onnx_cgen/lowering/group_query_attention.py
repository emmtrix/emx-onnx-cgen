from __future__ import annotations

import math
from dataclasses import dataclass

from shared.scalar_types import ScalarType

from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from ..ir.ops import GroupQueryAttentionOp
from .common import node_dtype as _node_dtype
from .common import optional_name as _optional_name
from .common import value_dtype as _value_dtype
from .common import value_shape as _value_shape
from .registry import register_lowering


@dataclass(frozen=True)
class GroupQueryAttentionSpec:
    batch: int
    q_seq: int
    num_heads: int
    kv_num_heads: int
    qk_head_size: int
    v_head_size: int
    q_hidden_size: int
    k_hidden_size: int
    v_hidden_size: int
    kv_seq: int
    max_seq_len: int
    scale: float
    head_group_size: int
    has_past: bool
    has_present: bool
    seqlens_rank: int
    seqlens_rows: int
    seqlens_cols: int


def _normalize_quant_attr(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").upper()
    return str(value).upper()


def resolve_group_query_attention_spec(
    graph: Graph,
    node: Node,
    dtype: ScalarType,
) -> GroupQueryAttentionSpec:
    if not dtype.is_float:
        raise UnsupportedOpError("Unsupported op GroupQueryAttention")
    if len(node.inputs) < 7 or len(node.outputs) < 1:
        raise UnsupportedOpError("Unsupported op GroupQueryAttention")

    supported_attrs = {
        "num_heads",
        "kv_num_heads",
        "scale",
        "local_window_size",
        "do_rotary",
        "rotary_interleaved",
        "smooth_softmax",
        "softcap",
        "qk_output",
        "k_quant_type",
        "v_quant_type",
    }
    if set(node.attrs) - supported_attrs:
        raise UnsupportedOpError("Unsupported op GroupQueryAttention")

    if int(node.attrs.get("do_rotary", 0)) != 0:
        raise UnsupportedOpError("Unsupported op GroupQueryAttention")
    if int(node.attrs.get("qk_output", 0)) != 0:
        raise UnsupportedOpError("Unsupported op GroupQueryAttention")
    if int(node.attrs.get("local_window_size", -1)) != -1:
        raise UnsupportedOpError("Unsupported op GroupQueryAttention")
    if int(node.attrs.get("smooth_softmax", -1)) != -1:
        raise UnsupportedOpError("Unsupported op GroupQueryAttention")

    k_quant_type = _normalize_quant_attr(node.attrs.get("k_quant_type", "NONE"))
    v_quant_type = _normalize_quant_attr(node.attrs.get("v_quant_type", "NONE"))
    if k_quant_type != "NONE" or v_quant_type != "NONE":
        raise UnsupportedOpError("Unsupported op GroupQueryAttention")

    num_heads = node.attrs.get("num_heads")
    kv_num_heads = node.attrs.get("kv_num_heads")
    if num_heads is None or kv_num_heads is None:
        raise UnsupportedOpError("Unsupported op GroupQueryAttention")
    num_heads = int(num_heads)
    kv_num_heads = int(kv_num_heads)
    if num_heads <= 0 or kv_num_heads <= 0:
        raise ShapeInferenceError(
            "GroupQueryAttention num_heads and kv_num_heads must be > 0"
        )
    if num_heads < kv_num_heads or num_heads % kv_num_heads != 0:
        raise ShapeInferenceError(
            "GroupQueryAttention requires num_heads to be a multiple of kv_num_heads"
        )

    query_shape = _value_shape(graph, node.inputs[0], node)
    key_shape = _value_shape(graph, node.inputs[1], node)
    value_shape = _value_shape(graph, node.inputs[2], node)
    if len(query_shape) != 3 or len(key_shape) != 3 or len(value_shape) != 3:
        raise ShapeInferenceError("GroupQueryAttention expects 3D query/key/value")

    batch, q_seq, q_hidden_size = query_shape
    key_batch, kv_seq, k_hidden_size = key_shape
    value_batch, value_seq, v_hidden_size = value_shape
    if key_batch != batch or value_batch != batch:
        raise ShapeInferenceError("GroupQueryAttention batch sizes must match")
    if value_seq != kv_seq:
        raise ShapeInferenceError(
            "GroupQueryAttention key/value sequence lengths must match"
        )
    # kv_seq == 1 is the incremental decode step (one new token). kv_seq == 0 is
    # the shared/empty-KV case where no new token is supplied and the query
    # attends only into the existing cache. Longer prefill runs (kv_seq > 1) are
    # not supported yet.
    if kv_seq not in (0, 1):
        raise UnsupportedOpError("Unsupported op GroupQueryAttention")

    if q_hidden_size % num_heads != 0:
        raise ShapeInferenceError(
            "GroupQueryAttention query hidden size must be divisible by num_heads"
        )
    if k_hidden_size % kv_num_heads != 0:
        raise ShapeInferenceError(
            "GroupQueryAttention key hidden size must be divisible by kv_num_heads"
        )
    if v_hidden_size % kv_num_heads != 0:
        raise ShapeInferenceError(
            "GroupQueryAttention value hidden size must be divisible by kv_num_heads"
        )

    qk_head_size = q_hidden_size // num_heads
    k_head_size = k_hidden_size // kv_num_heads
    v_head_size = v_hidden_size // kv_num_heads
    if qk_head_size != k_head_size:
        raise ShapeInferenceError("GroupQueryAttention query/key head sizes must match")

    output_shape = _value_shape(graph, node.outputs[0], node)
    expected_output = (batch, q_seq, num_heads * v_head_size)
    if output_shape != expected_output:
        raise ShapeInferenceError(
            f"GroupQueryAttention output shape must be {expected_output}, got {output_shape}"
        )

    seqlens_name = _optional_name(node.inputs, 5)
    total_seq_name = _optional_name(node.inputs, 6)
    if seqlens_name is None or total_seq_name is None:
        raise UnsupportedOpError("Unsupported op GroupQueryAttention")

    seqlens_dtype = _value_dtype(graph, seqlens_name, node)
    total_seq_dtype = _value_dtype(graph, total_seq_name, node)
    if seqlens_dtype != ScalarType.I32 or total_seq_dtype != ScalarType.I32:
        raise UnsupportedOpError(
            "GroupQueryAttention seqlens_k and total_sequence_length must be int32"
        )

    seqlens_shape = _value_shape(graph, seqlens_name, node)
    seqlens_rank = len(seqlens_shape)
    if seqlens_rank == 1:
        if seqlens_shape[0] != batch:
            raise ShapeInferenceError(
                "GroupQueryAttention seqlens_k 1D shape must be [batch]"
            )
        seqlens_rows = seqlens_shape[0]
        seqlens_cols = 1
    elif seqlens_rank == 2:
        seqlens_rows, seqlens_cols = seqlens_shape
        if seqlens_rows * seqlens_cols != batch:
            raise ShapeInferenceError(
                "GroupQueryAttention seqlens_k 2D shape must contain batch elements"
            )
    else:
        raise ShapeInferenceError("GroupQueryAttention seqlens_k must be 1D or 2D")

    total_seq_shape = _value_shape(graph, total_seq_name, node)
    if total_seq_shape != (1,):
        raise ShapeInferenceError(
            "GroupQueryAttention total_sequence_length must have shape [1]"
        )

    past_key_name = _optional_name(node.inputs, 3)
    past_value_name = _optional_name(node.inputs, 4)
    has_past = past_key_name is not None or past_value_name is not None
    if has_past and (past_key_name is None or past_value_name is None):
        raise UnsupportedOpError(
            "GroupQueryAttention expects both past_key and past_value if either is provided"
        )
    # With no new key/value tokens there is nothing to attend to unless a cache
    # is provided, so the empty-KV case requires a past cache.
    if kv_seq == 0 and not has_past:
        raise UnsupportedOpError("Unsupported op GroupQueryAttention")

    present_key_name = _optional_name(node.outputs, 1)
    present_value_name = _optional_name(node.outputs, 2)
    has_present = present_key_name is not None or present_value_name is not None
    if has_present and (present_key_name is None or present_value_name is None):
        raise UnsupportedOpError(
            "GroupQueryAttention expects both present_key and present_value if either is provided"
        )
    if not has_present:
        raise UnsupportedOpError("Unsupported op GroupQueryAttention")

    max_seq_len: int | None = None
    if has_past:
        past_key_shape = _value_shape(graph, past_key_name, node)
        past_value_shape = _value_shape(graph, past_value_name, node)
        if len(past_key_shape) != 4 or len(past_value_shape) != 4:
            raise ShapeInferenceError("GroupQueryAttention past key/value must be 4D")
        if past_key_shape[0] != batch or past_key_shape[1] != kv_num_heads:
            raise ShapeInferenceError(
                "GroupQueryAttention past_key batch/kv_heads must match query"
            )
        if past_key_shape[3] != qk_head_size:
            raise ShapeInferenceError(
                "GroupQueryAttention past_key head_size must match query/key head size"
            )
        if past_value_shape[0] != batch or past_value_shape[1] != kv_num_heads:
            raise ShapeInferenceError(
                "GroupQueryAttention past_value batch/kv_heads must match query"
            )
        if past_value_shape[3] != v_head_size:
            raise ShapeInferenceError(
                "GroupQueryAttention past_value head_size must match value head size"
            )
        if past_key_shape[2] != past_value_shape[2]:
            raise ShapeInferenceError(
                "GroupQueryAttention past_key and past_value sequence lengths must match"
            )
        max_seq_len = past_key_shape[2]

    present_key_shape = _value_shape(graph, present_key_name, node)
    present_value_shape = _value_shape(graph, present_value_name, node)
    if len(present_key_shape) != 4 or len(present_value_shape) != 4:
        raise ShapeInferenceError("GroupQueryAttention present key/value must be 4D")
    if present_key_shape[0] != batch or present_key_shape[1] != kv_num_heads:
        raise ShapeInferenceError(
            "GroupQueryAttention present_key batch/kv_heads must match query"
        )
    if present_key_shape[3] != qk_head_size:
        raise ShapeInferenceError(
            "GroupQueryAttention present_key head_size must match query/key head size"
        )
    if present_value_shape[0] != batch or present_value_shape[1] != kv_num_heads:
        raise ShapeInferenceError(
            "GroupQueryAttention present_value batch/kv_heads must match query"
        )
    if present_value_shape[3] != v_head_size:
        raise ShapeInferenceError(
            "GroupQueryAttention present_value head_size must match value head size"
        )
    if present_key_shape[2] != present_value_shape[2]:
        raise ShapeInferenceError(
            "GroupQueryAttention present_key and present_value sequence lengths must match"
        )

    if max_seq_len is None:
        max_seq_len = present_key_shape[2]
    if present_key_shape[2] != max_seq_len or present_value_shape[2] != max_seq_len:
        raise ShapeInferenceError(
            "GroupQueryAttention present key/value max sequence length must match past cache"
        )

    scale = float(node.attrs.get("scale", 1.0 / math.sqrt(qk_head_size)))

    return GroupQueryAttentionSpec(
        batch=batch,
        q_seq=q_seq,
        num_heads=num_heads,
        kv_num_heads=kv_num_heads,
        qk_head_size=qk_head_size,
        v_head_size=v_head_size,
        q_hidden_size=q_hidden_size,
        k_hidden_size=k_hidden_size,
        v_hidden_size=v_hidden_size,
        kv_seq=kv_seq,
        max_seq_len=max_seq_len,
        scale=scale,
        head_group_size=num_heads // kv_num_heads,
        has_past=has_past,
        has_present=has_present,
        seqlens_rank=seqlens_rank,
        seqlens_rows=seqlens_rows,
        seqlens_cols=seqlens_cols,
    )


@register_lowering("GroupQueryAttention")
def lower_group_query_attention(graph: Graph, node: Node) -> GroupQueryAttentionOp:
    query = node.inputs[0]
    key = node.inputs[1]
    value = node.inputs[2]
    output = node.outputs[0]
    op_dtype = _node_dtype(graph, node, query, key, value, output)
    spec = resolve_group_query_attention_spec(graph, node, op_dtype)

    return GroupQueryAttentionOp(
        query=query,
        key=key,
        value=value,
        past_key=_optional_name(node.inputs, 3),
        past_value=_optional_name(node.inputs, 4),
        seqlens_k=node.inputs[5],
        total_sequence_length=node.inputs[6],
        output=output,
        present_key=_optional_name(node.outputs, 1),
        present_value=_optional_name(node.outputs, 2),
        batch=spec.batch,
        q_seq=spec.q_seq,
        num_heads=spec.num_heads,
        kv_num_heads=spec.kv_num_heads,
        qk_head_size=spec.qk_head_size,
        v_head_size=spec.v_head_size,
        q_hidden_size=spec.q_hidden_size,
        k_hidden_size=spec.k_hidden_size,
        v_hidden_size=spec.v_hidden_size,
        kv_seq=spec.kv_seq,
        max_seq_len=spec.max_seq_len,
        scale=spec.scale,
        head_group_size=spec.head_group_size,
        seqlens_rank=spec.seqlens_rank,
        seqlens_rows=spec.seqlens_rows,
        seqlens_cols=spec.seqlens_cols,
        dtype=op_dtype,
    )

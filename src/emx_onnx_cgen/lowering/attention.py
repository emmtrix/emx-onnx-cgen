from __future__ import annotations

import math
from dataclasses import dataclass

from shared.scalar_types import ScalarType

from ..ir.ops import AttentionOp
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from .common import node_dtype as _node_dtype
from .common import optional_name as _optional_name
from .common import value_dtype as _value_dtype
from .common import value_shape as _value_shape
from .registry import register_lowering


@dataclass(frozen=True)
class AttentionSpec:
    batch: int
    q_heads: int
    kv_heads: int
    q_seq: int
    kv_seq: int
    total_seq: int
    past_seq: int
    qk_head_size: int
    v_head_size: int
    q_hidden_size: int | None
    k_hidden_size: int | None
    v_hidden_size: int | None
    scale: float
    is_causal: bool
    softcap: float
    qk_matmul_output_mode: int
    q_rank: int
    k_rank: int
    v_rank: int
    output_rank: int
    mask_shape: tuple[int, ...] | None
    mask_is_bool: bool
    mask_rank: int | None
    mask_broadcast_batch: bool
    mask_broadcast_heads: bool
    mask_broadcast_q_seq: bool
    mask_q_seq: int | None
    mask_kv_seq: int | None
    head_group_size: int
    has_attn_mask: bool
    has_past: bool
    has_present: bool
    has_nonpad: bool


def resolve_attention_spec(
    graph: Graph, node: Node, dtype: ScalarType
) -> AttentionSpec:
    if not dtype.is_float:
        raise UnsupportedOpError("Unsupported op Attention")
    if len(node.inputs) < 3 or len(node.outputs) < 1:
        raise UnsupportedOpError("Unsupported op Attention")
    supported_attrs = {
        "scale",
        "is_causal",
        "q_num_heads",
        "kv_num_heads",
        "softmax_precision",
        "softcap",
        "qk_matmul_output_mode",
    }
    if set(node.attrs) - supported_attrs:
        raise UnsupportedOpError("Unsupported op Attention")
    q_shape = _value_shape(graph, node.inputs[0], node)
    k_shape = _value_shape(graph, node.inputs[1], node)
    v_shape = _value_shape(graph, node.inputs[2], node)
    q_rank = len(q_shape)
    k_rank = len(k_shape)
    v_rank = len(v_shape)
    if q_rank not in {3, 4} or k_rank not in {3, 4} or v_rank not in {3, 4}:
        raise UnsupportedOpError("Unsupported op Attention")
    if q_rank != k_rank or q_rank != v_rank:
        raise UnsupportedOpError("Unsupported op Attention")
    batch = q_shape[0]
    if batch != k_shape[0] or batch != v_shape[0]:
        raise ShapeInferenceError("Attention batch sizes must match")
    q_hidden_size = None
    k_hidden_size = None
    v_hidden_size = None
    if q_rank == 3:
        q_heads = node.attrs.get("q_num_heads")
        kv_heads = node.attrs.get("kv_num_heads")
        if q_heads is None or kv_heads is None:
            raise UnsupportedOpError("Unsupported op Attention")
        q_heads = int(q_heads)
        kv_heads = int(kv_heads)
        q_seq = q_shape[1]
        kv_seq = k_shape[1]
        if kv_seq != v_shape[1]:
            raise ShapeInferenceError(
                "Attention key/value sequence lengths must match"
            )
        q_hidden_size = q_shape[2]
        k_hidden_size = k_shape[2]
        v_hidden_size = v_shape[2]
        if q_hidden_size % q_heads != 0:
            raise ShapeInferenceError(
                "Attention query hidden size must be divisible by q_num_heads"
            )
        if k_hidden_size % kv_heads != 0:
            raise ShapeInferenceError(
                "Attention key hidden size must be divisible by kv_num_heads"
            )
        if v_hidden_size % kv_heads != 0:
            raise ShapeInferenceError(
                "Attention value hidden size must be divisible by kv_num_heads"
            )
        qk_head_size = q_hidden_size // q_heads
        k_head_size = k_hidden_size // kv_heads
        v_head_size = v_hidden_size // kv_heads
        if qk_head_size != k_head_size:
            raise ShapeInferenceError("Attention Q/K head sizes must match")
    else:
        q_heads = q_shape[1]
        kv_heads = k_shape[1]
        if kv_heads != v_shape[1]:
            raise ShapeInferenceError("Attention key/value head counts must match")
        q_seq = q_shape[2]
        kv_seq = k_shape[2]
        if kv_seq != v_shape[2]:
            raise ShapeInferenceError(
                "Attention key/value sequence lengths must match"
            )
        qk_head_size = q_shape[3]
        k_head_size = k_shape[3]
        v_head_size = v_shape[3]
        if qk_head_size != k_head_size:
            raise ShapeInferenceError("Attention Q/K head sizes must match")
        attr_q_heads = node.attrs.get("q_num_heads")
        attr_kv_heads = node.attrs.get("kv_num_heads")
        if attr_q_heads is not None and int(attr_q_heads) != q_heads:
            raise ShapeInferenceError(
                "Attention q_num_heads must match query head dimension"
            )
        if attr_kv_heads is not None and int(attr_kv_heads) != kv_heads:
            raise ShapeInferenceError(
                "Attention kv_num_heads must match key/value head dimension"
            )
    if q_heads < kv_heads or q_heads % kv_heads != 0:
        raise ShapeInferenceError(
            "Attention requires q_num_heads to be a multiple of kv_num_heads"
        )
    head_group_size = q_heads // kv_heads
    past_key_name = _optional_name(node.inputs, 4)
    past_value_name = _optional_name(node.inputs, 5)
    has_past = past_key_name is not None or past_value_name is not None
    if has_past and (past_key_name is None or past_value_name is None):
        raise UnsupportedOpError(
            "Attention expects both past_key and past_value if either is provided"
        )
    past_seq = 0
    if has_past:
        past_key_shape = _value_shape(graph, past_key_name, node)
        past_value_shape = _value_shape(graph, past_value_name, node)
        if len(past_key_shape) != 4 or len(past_value_shape) != 4:
            raise ShapeInferenceError("Attention past key/value must be 4D")
        if (
            past_key_shape[0] != batch
            or past_value_shape[0] != batch
            or past_key_shape[1] != kv_heads
            or past_value_shape[1] != kv_heads
        ):
            raise ShapeInferenceError(
                "Attention past key/value batch/head sizes must match"
            )
        if past_key_shape[3] != qk_head_size:
            raise ShapeInferenceError(
                "Attention past key head size must match key head size"
            )
        if past_value_shape[3] != v_head_size:
            raise ShapeInferenceError(
                "Attention past value head size must match value head size"
            )
        past_seq = past_key_shape[2]
    total_seq = kv_seq + past_seq
    output_shape = _value_shape(graph, node.outputs[0], node)
    output_rank = len(output_shape)
    if q_rank == 3:
        expected_output_shape = (
            batch,
            q_seq,
            q_heads * v_head_size,
        )
    else:
        expected_output_shape = (batch, q_heads, q_seq, v_head_size)
    if output_shape != expected_output_shape:
        raise ShapeInferenceError(
            "Attention output shape must be "
            f"{expected_output_shape}, got {output_shape}"
        )
    present_key_name = _optional_name(node.outputs, 1)
    present_value_name = _optional_name(node.outputs, 2)
    has_present = present_key_name is not None or present_value_name is not None
    if has_present and (present_key_name is None or present_value_name is None):
        raise UnsupportedOpError(
            "Attention expects both present_key and present_value if either is provided"
        )
    if has_present and not has_past:
        raise UnsupportedOpError(
            "Attention present outputs require past key/value inputs"
        )
    if has_present:
        present_key_shape = _value_shape(graph, present_key_name, node)
        present_value_shape = _value_shape(graph, present_value_name, node)
        expected_present_key = (batch, kv_heads, total_seq, qk_head_size)
        expected_present_value = (batch, kv_heads, total_seq, v_head_size)
        if present_key_shape != expected_present_key:
            raise ShapeInferenceError(
                "Attention present key shape must be "
                f"{expected_present_key}, got {present_key_shape}"
            )
        if present_value_shape != expected_present_value:
            raise ShapeInferenceError(
                "Attention present value shape must be "
                f"{expected_present_value}, got {present_value_shape}"
            )
    qk_matmul_output_name = _optional_name(node.outputs, 3)
    if qk_matmul_output_name is not None:
        qk_shape = _value_shape(graph, qk_matmul_output_name, node)
        expected_qk_shape = (batch, q_heads, q_seq, total_seq)
        if qk_shape != expected_qk_shape:
            raise ShapeInferenceError(
                "Attention qk_matmul_output shape must be "
                f"{expected_qk_shape}, got {qk_shape}"
            )
    attn_mask_name = _optional_name(node.inputs, 3)
    mask_shape = None
    mask_rank = None
    mask_q_seq = None
    mask_kv_seq = None
    mask_is_bool = False
    mask_broadcast_batch = False
    mask_broadcast_heads = True
    mask_broadcast_q_seq = False
    has_attn_mask = attn_mask_name is not None
    if has_attn_mask:
        mask_shape = _value_shape(graph, attn_mask_name, node)
        mask_rank = len(mask_shape)
        if mask_rank not in {2, 3, 4}:
            raise ShapeInferenceError("Attention mask must be 2D/3D/4D")
        mask_dtype = _value_dtype(graph, attn_mask_name, node)
        if mask_dtype == ScalarType.BOOL:
            mask_is_bool = True
        elif mask_dtype != dtype:
            raise UnsupportedOpError(
                "Attention mask must be bool or match attention dtype"
            )
        if mask_rank == 2:
            mask_q_seq, mask_kv_seq = mask_shape
            mask_broadcast_batch = True
            mask_broadcast_heads = True
            mask_broadcast_q_seq = mask_q_seq == 1
            if mask_q_seq not in {1, q_seq}:
                raise ShapeInferenceError(
                    "Attention mask sequence length must match query length"
                )
        elif mask_rank == 3:
            mask_batch, mask_q_seq, mask_kv_seq = mask_shape
            mask_broadcast_batch = mask_batch == 1
            mask_broadcast_heads = True
            mask_broadcast_q_seq = mask_q_seq == 1
            if mask_batch not in {1, batch}:
                raise ShapeInferenceError(
                    "Attention mask batch dimension must match batch size"
                )
            if mask_q_seq not in {1, q_seq}:
                raise ShapeInferenceError(
                    "Attention mask sequence length must match query length"
                )
        else:
            mask_batch, mask_heads, mask_q_seq, mask_kv_seq = mask_shape
            mask_broadcast_batch = mask_batch == 1
            mask_broadcast_heads = mask_heads == 1
            mask_broadcast_q_seq = mask_q_seq == 1
            if mask_batch not in {1, batch}:
                raise ShapeInferenceError(
                    "Attention mask batch dimension must match batch size"
                )
            if mask_heads not in {1, q_heads}:
                raise ShapeInferenceError(
                    "Attention mask head dimension must match q_num_heads"
                )
            if mask_q_seq not in {1, q_seq}:
                raise ShapeInferenceError(
                    "Attention mask sequence length must match query length"
                )
        if mask_kv_seq is None:
            raise ShapeInferenceError("Attention mask must include kv sequence")
        if mask_kv_seq > total_seq:
            raise ShapeInferenceError(
                "Attention mask kv sequence length exceeds total sequence length"
            )
    nonpad_name = _optional_name(node.inputs, 6)
    has_nonpad = nonpad_name is not None
    if has_nonpad:
        if has_past or has_present:
            raise UnsupportedOpError(
                "Attention nonpad_kv_seqlen is not supported with KV cache"
            )
        nonpad_shape = _value_shape(graph, nonpad_name, node)
        if nonpad_shape != (batch,):
            raise ShapeInferenceError(
                "Attention nonpad_kv_seqlen must have shape (batch,)"
            )
        nonpad_dtype = _value_dtype(graph, nonpad_name, node)
        if nonpad_dtype != ScalarType.I64:
            raise UnsupportedOpError(
                "Attention nonpad_kv_seqlen must be int64"
            )
    scale = float(node.attrs.get("scale", 1.0 / math.sqrt(qk_head_size)))
    softcap = float(node.attrs.get("softcap", 0.0))
    is_causal = int(node.attrs.get("is_causal", 0))
    if is_causal not in (0, 1):
        raise UnsupportedOpError("Unsupported op Attention")
    qk_matmul_output_mode = int(node.attrs.get("qk_matmul_output_mode", 0))
    if qk_matmul_output_mode not in {0, 1, 2, 3}:
        raise UnsupportedOpError("Unsupported op Attention")
    return AttentionSpec(
        batch=batch,
        q_heads=q_heads,
        kv_heads=kv_heads,
        q_seq=q_seq,
        kv_seq=kv_seq,
        total_seq=total_seq,
        past_seq=past_seq,
        qk_head_size=qk_head_size,
        v_head_size=v_head_size,
        q_hidden_size=q_hidden_size,
        k_hidden_size=k_hidden_size,
        v_hidden_size=v_hidden_size,
        scale=scale,
        is_causal=bool(is_causal),
        softcap=softcap,
        qk_matmul_output_mode=qk_matmul_output_mode,
        q_rank=q_rank,
        k_rank=k_rank,
        v_rank=v_rank,
        output_rank=output_rank,
        mask_shape=mask_shape,
        mask_is_bool=mask_is_bool,
        mask_rank=mask_rank,
        mask_broadcast_batch=mask_broadcast_batch,
        mask_broadcast_heads=mask_broadcast_heads,
        mask_broadcast_q_seq=mask_broadcast_q_seq,
        mask_q_seq=mask_q_seq,
        mask_kv_seq=mask_kv_seq,
        head_group_size=head_group_size,
        has_attn_mask=has_attn_mask,
        has_past=has_past,
        has_present=has_present,
        has_nonpad=has_nonpad,
    )


@register_lowering("Attention")
def lower_attention(graph: Graph, node: Node) -> AttentionOp:
    input_q = node.inputs[0]
    input_k = node.inputs[1]
    input_v = node.inputs[2]
    output_y = node.outputs[0]
    op_dtype = _node_dtype(graph, node, input_q, input_k, input_v, output_y)
    spec = resolve_attention_spec(graph, node, op_dtype)
    input_mask = _optional_name(node.inputs, 3)
    input_past_key = _optional_name(node.inputs, 4)
    input_past_value = _optional_name(node.inputs, 5)
    input_nonpad = _optional_name(node.inputs, 6)
    output_present_key = _optional_name(node.outputs, 1)
    output_present_value = _optional_name(node.outputs, 2)
    output_qk_matmul = _optional_name(node.outputs, 3)
    return AttentionOp(
        input_q=input_q,
        input_k=input_k,
        input_v=input_v,
        input_attn_mask=input_mask,
        input_past_key=input_past_key,
        input_past_value=input_past_value,
        input_nonpad_kv_seqlen=input_nonpad,
        output=output_y,
        output_present_key=output_present_key,
        output_present_value=output_present_value,
        output_qk_matmul=output_qk_matmul,
        batch=spec.batch,
        q_heads=spec.q_heads,
        kv_heads=spec.kv_heads,
        q_seq=spec.q_seq,
        kv_seq=spec.kv_seq,
        total_seq=spec.total_seq,
        past_seq=spec.past_seq,
        qk_head_size=spec.qk_head_size,
        v_head_size=spec.v_head_size,
        q_hidden_size=spec.q_hidden_size,
        k_hidden_size=spec.k_hidden_size,
        v_hidden_size=spec.v_hidden_size,
        scale=spec.scale,
        is_causal=spec.is_causal,
        softcap=spec.softcap,
        qk_matmul_output_mode=spec.qk_matmul_output_mode,
        q_rank=spec.q_rank,
        k_rank=spec.k_rank,
        v_rank=spec.v_rank,
        output_rank=spec.output_rank,
        mask_shape=spec.mask_shape,
        mask_is_bool=spec.mask_is_bool,
        mask_rank=spec.mask_rank,
        mask_broadcast_batch=spec.mask_broadcast_batch,
        mask_broadcast_heads=spec.mask_broadcast_heads,
        mask_broadcast_q_seq=spec.mask_broadcast_q_seq,
        mask_q_seq=spec.mask_q_seq,
        mask_kv_seq=spec.mask_kv_seq,
        head_group_size=spec.head_group_size,
        dtype=op_dtype,
    )

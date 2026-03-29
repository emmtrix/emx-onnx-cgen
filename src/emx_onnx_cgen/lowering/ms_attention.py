from __future__ import annotations

import math
from dataclasses import dataclass

from shared.scalar_types import ScalarType

from ..ir.ops import MsAttentionOp
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from .common import node_dtype as _node_dtype
from .common import optional_name as _optional_name
from .common import value_dtype as _value_dtype
from .common import value_shape as _value_shape

# Mask type constants
MASK_NONE = 0
MASK_1D_END = 1
MASK_1D_END_START = 2
MASK_2D = 3
MASK_3D = 4


@dataclass(frozen=True)
class MsAttentionSpec:
    batch: int
    seq_len: int
    num_heads: int
    qk_head_size: int
    v_head_size: int
    q_hidden_size: int
    k_hidden_size: int
    v_hidden_size: int
    input_hidden_size: int
    past_seq: int
    total_seq: int
    scale: float
    unidirectional: bool
    mask_filter_value: float
    mask_type: int
    mask_shape: tuple[int, ...] | None
    has_past: bool
    has_present: bool
    has_extra_add_qk: bool


def resolve_ms_attention_spec(
    graph: Graph, node: Node, dtype: ScalarType
) -> MsAttentionSpec:
    """Resolve and validate com.microsoft::Attention parameters."""
    if not dtype.is_float:
        raise UnsupportedOpError("Unsupported op Attention")
    if len(node.inputs) < 3 or len(node.outputs) < 1:
        raise UnsupportedOpError("Unsupported op Attention")
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
        raise UnsupportedOpError("Unsupported op Attention")
    if int(node.attrs.get("do_rotary", 0)) != 0:
        raise UnsupportedOpError("Unsupported op Attention")
    num_heads = node.attrs.get("num_heads")
    if num_heads is None:
        raise UnsupportedOpError("Unsupported op Attention")
    num_heads = int(num_heads)

    input_shape = _value_shape(graph, node.inputs[0], node)
    weight_shape = _value_shape(graph, node.inputs[1], node)
    bias_shape = _value_shape(graph, node.inputs[2], node)

    if len(input_shape) != 3:
        raise ShapeInferenceError("Attention input must be 3D (batch, seq, hidden)")
    if len(weight_shape) != 2:
        raise ShapeInferenceError("Attention weight must be 2D")
    if len(bias_shape) != 1:
        raise ShapeInferenceError("Attention bias must be 1D")

    batch, seq_len, input_hidden_size = input_shape
    if weight_shape[0] != input_hidden_size:
        raise ShapeInferenceError(
            "Attention weight first dimension must match input hidden size"
        )
    total_qkv = weight_shape[1]
    if bias_shape[0] != total_qkv:
        raise ShapeInferenceError(
            "Attention bias size must match weight second dimension"
        )

    qkv_hidden_sizes = node.attrs.get("qkv_hidden_sizes")
    if qkv_hidden_sizes is not None:
        qkv_hidden_sizes = [int(x) for x in qkv_hidden_sizes]
        if len(qkv_hidden_sizes) != 3:
            raise UnsupportedOpError("Unsupported op Attention")
        q_hidden_size, k_hidden_size, v_hidden_size = qkv_hidden_sizes
        if q_hidden_size + k_hidden_size + v_hidden_size != total_qkv:
            raise ShapeInferenceError(
                "Attention qkv_hidden_sizes must sum to weight second dimension"
            )
    else:
        if total_qkv % 3 != 0:
            raise ShapeInferenceError(
                "Attention weight dimension must be divisible by 3"
            )
        q_hidden_size = k_hidden_size = v_hidden_size = total_qkv // 3

    if q_hidden_size % num_heads != 0:
        raise ShapeInferenceError(
            "Attention Q hidden size must be divisible by num_heads"
        )
    if k_hidden_size % num_heads != 0:
        raise ShapeInferenceError(
            "Attention K hidden size must be divisible by num_heads"
        )
    if v_hidden_size % num_heads != 0:
        raise ShapeInferenceError(
            "Attention V hidden size must be divisible by num_heads"
        )

    qk_head_size = q_hidden_size // num_heads
    k_head_size = k_hidden_size // num_heads
    v_head_size = v_hidden_size // num_heads
    if qk_head_size != k_head_size:
        raise ShapeInferenceError("Attention Q/K head sizes must match")

    past_name = _optional_name(node.inputs, 4)
    has_past = past_name is not None
    past_seq = 0
    if has_past:
        past_shape = _value_shape(graph, past_name, node)
        if len(past_shape) != 5:
            raise ShapeInferenceError("Attention past must be 5D")
        if past_shape[0] != 2:
            raise ShapeInferenceError(
                "Attention past first dimension must be 2 (key, value)"
            )
        if past_shape[1] != batch or past_shape[2] != num_heads:
            raise ShapeInferenceError("Attention past batch/heads must match")
        if past_shape[4] != qk_head_size:
            raise ShapeInferenceError("Attention past head size must match")
        past_seq = past_shape[3]

    total_seq = seq_len + past_seq

    present_name = _optional_name(node.outputs, 1)
    has_present = present_name is not None
    if has_present:
        present_shape = _value_shape(graph, present_name, node)
        expected_present = (2, batch, num_heads, total_seq, qk_head_size)
        if present_shape != expected_present:
            raise ShapeInferenceError(
                f"Attention present shape must be {expected_present}, "
                f"got {present_shape}"
            )

    output_shape = _value_shape(graph, node.outputs[0], node)
    expected_output = (batch, seq_len, v_hidden_size)
    if output_shape != expected_output:
        raise ShapeInferenceError(
            f"Attention output shape must be {expected_output}, got {output_shape}"
        )

    mask_name = _optional_name(node.inputs, 3)
    mask_type = MASK_NONE
    mask_shape = None
    if mask_name is not None:
        mask_shape = _value_shape(graph, mask_name, node)
        mask_rank = len(mask_shape)
        mask_dtype = _value_dtype(graph, mask_name, node)
        if mask_dtype != ScalarType.I32:
            raise UnsupportedOpError("Attention mask_index must be int32")
        if mask_rank == 1:
            if mask_shape[0] == batch:
                mask_type = MASK_1D_END
            elif mask_shape[0] == 2 * batch:
                mask_type = MASK_1D_END_START
            else:
                raise ShapeInferenceError(
                    "Attention 1D mask must have batch or 2*batch elements"
                )
        elif mask_rank == 2:
            if mask_shape[0] != batch:
                raise ShapeInferenceError(
                    "Attention 2D mask batch dimension must match"
                )
            mask_type = MASK_2D
        elif mask_rank == 3:
            if mask_shape[0] != batch:
                raise ShapeInferenceError(
                    "Attention 3D mask batch dimension must match"
                )
            if mask_shape[1] != seq_len:
                raise ShapeInferenceError(
                    "Attention 3D mask sequence dimension must match"
                )
            mask_type = MASK_3D
        else:
            raise UnsupportedOpError("Attention mask_index must be 1D/2D/3D")

    extra_name = _optional_name(node.inputs, 5)
    has_extra_add_qk = extra_name is not None
    if has_extra_add_qk:
        extra_shape = _value_shape(graph, extra_name, node)
        expected_extra = (batch, num_heads, seq_len, total_seq)
        if extra_shape != expected_extra:
            raise ShapeInferenceError(
                f"Attention extra_add_qk shape must be {expected_extra}, "
                f"got {extra_shape}"
            )

    scale = float(node.attrs.get("scale", 1.0 / math.sqrt(qk_head_size)))
    unidirectional = bool(int(node.attrs.get("unidirectional", 0)))
    mask_filter_value = float(node.attrs.get("mask_filter_value", -10000.0))

    return MsAttentionSpec(
        batch=batch,
        seq_len=seq_len,
        num_heads=num_heads,
        qk_head_size=qk_head_size,
        v_head_size=v_head_size,
        q_hidden_size=q_hidden_size,
        k_hidden_size=k_hidden_size,
        v_hidden_size=v_hidden_size,
        input_hidden_size=input_hidden_size,
        past_seq=past_seq,
        total_seq=total_seq,
        scale=scale,
        unidirectional=unidirectional,
        mask_filter_value=mask_filter_value,
        mask_type=mask_type,
        mask_shape=mask_shape,
        has_past=has_past,
        has_present=has_present,
        has_extra_add_qk=has_extra_add_qk,
    )


def lower_ms_attention(graph: Graph, node: Node) -> MsAttentionOp:
    """Lower com.microsoft::Attention contrib op."""
    input0 = node.inputs[0]
    weight = node.inputs[1]
    bias = node.inputs[2]
    output = node.outputs[0]
    op_dtype = _node_dtype(graph, node, input0, output)
    spec = resolve_ms_attention_spec(graph, node, op_dtype)
    mask_name = _optional_name(node.inputs, 3)
    past_name = _optional_name(node.inputs, 4)
    extra_name = _optional_name(node.inputs, 5)
    present_name = _optional_name(node.outputs, 1)
    return MsAttentionOp(
        input0=input0,
        weight=weight,
        bias=bias,
        mask_index=mask_name,
        past=past_name,
        extra_add_qk=extra_name,
        output=output,
        present=present_name,
        batch=spec.batch,
        seq_len=spec.seq_len,
        num_heads=spec.num_heads,
        qk_head_size=spec.qk_head_size,
        v_head_size=spec.v_head_size,
        q_hidden_size=spec.q_hidden_size,
        k_hidden_size=spec.k_hidden_size,
        v_hidden_size=spec.v_hidden_size,
        input_hidden_size=spec.input_hidden_size,
        past_seq=spec.past_seq,
        total_seq=spec.total_seq,
        scale=spec.scale,
        unidirectional=spec.unidirectional,
        mask_filter_value=spec.mask_filter_value,
        mask_type=spec.mask_type,
        mask_shape=spec.mask_shape,
        dtype=op_dtype,
    )

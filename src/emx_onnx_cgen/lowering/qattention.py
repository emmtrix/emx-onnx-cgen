from __future__ import annotations

import math
from dataclasses import dataclass

from shared.scalar_types import ScalarType

from ..ir.ops import QAttentionOp
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from .common import optional_name as _optional_name
from .common import value_dtype as _value_dtype
from .common import value_shape as _value_shape
from .registry import register_lowering

# Mask type constants (same encoding as com.microsoft::Attention)
MASK_NONE = 0
MASK_1D_END = 1
MASK_1D_END_START = 2
MASK_2D = 3
MASK_3D = 4


@dataclass(frozen=True)
class QAttentionSpec:
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
    input_dtype: ScalarType
    weight_dtype: ScalarType
    dtype: ScalarType
    weight_scale_per_column: bool


def resolve_qattention_spec(graph: Graph, node: Node) -> QAttentionSpec:
    """Resolve and validate com.microsoft::QAttention parameters."""
    if len(node.inputs) < 8 or len(node.outputs) < 1:
        raise UnsupportedOpError("Unsupported op QAttention")
    supported_attrs = {
        "num_heads",
        "unidirectional",
        "mask_filter_value",
        "scale",
        "past_present_share_buffer",
        "do_rotary",
        "rotary_embedding_dim",
    }
    if set(node.attrs) - supported_attrs:
        raise UnsupportedOpError("Unsupported op QAttention")
    if int(node.attrs.get("do_rotary", 0)) != 0:
        raise UnsupportedOpError("Unsupported op QAttention")
    num_heads = node.attrs.get("num_heads")
    if num_heads is None:
        raise UnsupportedOpError("Unsupported op QAttention")
    num_heads = int(num_heads)

    input_dtype = _value_dtype(graph, node.inputs[0], node)
    weight_dtype = _value_dtype(graph, node.inputs[1], node)
    if input_dtype not in {ScalarType.U8, ScalarType.I8}:
        raise UnsupportedOpError(
            f"QAttention input must be uint8 or int8, got {input_dtype.onnx_name}"
        )
    if weight_dtype not in {ScalarType.U8, ScalarType.I8}:
        raise UnsupportedOpError(
            f"QAttention weight must be uint8 or int8, got {weight_dtype.onnx_name}"
        )

    bias_dtype = _value_dtype(graph, node.inputs[2], node)
    if not bias_dtype.is_float:
        raise UnsupportedOpError("QAttention bias must be float")

    input_scale_dtype = _value_dtype(graph, node.inputs[3], node)
    weight_scale_dtype = _value_dtype(graph, node.inputs[4], node)
    if not input_scale_dtype.is_float:
        raise UnsupportedOpError("QAttention input_scale must be float")
    if not weight_scale_dtype.is_float:
        raise UnsupportedOpError("QAttention weight_scale must be float")

    input_zp_dtype = _value_dtype(graph, node.inputs[6], node)
    weight_zp_dtype = _value_dtype(graph, node.inputs[7], node)
    if input_zp_dtype != input_dtype:
        raise UnsupportedOpError(
            "QAttention input_zero_point dtype must match input dtype"
        )
    if weight_zp_dtype != weight_dtype:
        raise UnsupportedOpError(
            "QAttention weight_zero_point dtype must match weight dtype"
        )

    # Output dtype is determined by the bias (always float)
    dtype = bias_dtype

    input_shape = _value_shape(graph, node.inputs[0], node)
    weight_shape = _value_shape(graph, node.inputs[1], node)
    bias_shape = _value_shape(graph, node.inputs[2], node)

    if len(input_shape) != 3:
        raise ShapeInferenceError(
            "QAttention input must be 3D (batch, seq, hidden), "
            f"got rank {len(input_shape)}"
        )
    if len(weight_shape) != 2:
        raise ShapeInferenceError(
            f"QAttention weight must be 2D, got rank {len(weight_shape)}"
        )
    if len(bias_shape) != 1:
        raise ShapeInferenceError(
            f"QAttention bias must be 1D, got rank {len(bias_shape)}"
        )

    batch, seq_len, input_hidden_size = input_shape
    if weight_shape[0] != input_hidden_size:
        raise ShapeInferenceError(
            "QAttention weight first dim must match input hidden size: "
            f"{weight_shape[0]} != {input_hidden_size}"
        )
    total_qkv = weight_shape[1]
    if bias_shape[0] != total_qkv:
        raise ShapeInferenceError(
            "QAttention bias size must match weight second dimension: "
            f"{bias_shape[0]} != {total_qkv}"
        )

    if total_qkv % 3 != 0:
        raise ShapeInferenceError(
            f"QAttention weight second dim must be divisible by 3, got {total_qkv}"
        )
    q_hidden_size = k_hidden_size = v_hidden_size = total_qkv // 3

    if q_hidden_size % num_heads != 0:
        raise ShapeInferenceError(
            "QAttention hidden size must be divisible by num_heads: "
            f"{q_hidden_size} % {num_heads} != 0"
        )
    qk_head_size = q_hidden_size // num_heads
    v_head_size = v_hidden_size // num_heads

    past_name = _optional_name(node.inputs, 8) if len(node.inputs) > 8 else None
    has_past = past_name is not None
    past_seq = 0
    if has_past:
        past_shape = _value_shape(graph, past_name, node)
        if len(past_shape) != 5:
            raise ShapeInferenceError(
                f"QAttention past must be 5D, got rank {len(past_shape)}"
            )
        if past_shape[0] != 2:
            raise ShapeInferenceError(
                "QAttention past first dimension must be 2 (key, value)"
            )
        if past_shape[1] != batch or past_shape[2] != num_heads:
            raise ShapeInferenceError("QAttention past batch/heads must match")
        if past_shape[4] != qk_head_size:
            raise ShapeInferenceError("QAttention past head size must match")
        past_seq = past_shape[3]

    total_seq = seq_len + past_seq

    present_name = _optional_name(node.outputs, 1) if len(node.outputs) > 1 else None
    has_present = present_name is not None

    mask_name = _optional_name(node.inputs, 5)
    mask_type = MASK_NONE
    mask_shape = None
    if mask_name is not None:
        mask_shape = _value_shape(graph, mask_name, node)
        mask_rank = len(mask_shape)
        mask_dtype = _value_dtype(graph, mask_name, node)
        if mask_dtype != ScalarType.I32:
            raise UnsupportedOpError("QAttention mask_index must be int32")
        if mask_rank == 1:
            if mask_shape[0] == batch:
                mask_type = MASK_1D_END
            elif mask_shape[0] == 2 * batch:
                mask_type = MASK_1D_END_START
            else:
                raise ShapeInferenceError(
                    "QAttention 1D mask must have batch or 2*batch elements"
                )
        elif mask_rank == 2:
            if mask_shape[0] != batch:
                raise ShapeInferenceError(
                    "QAttention 2D mask batch dimension must match"
                )
            mask_type = MASK_2D
        elif mask_rank == 3:
            if mask_shape[0] != batch:
                raise ShapeInferenceError(
                    "QAttention 3D mask batch dimension must match"
                )
            if mask_shape[1] != seq_len:
                raise ShapeInferenceError(
                    "QAttention 3D mask sequence dimension must match"
                )
            mask_type = MASK_3D
        else:
            raise UnsupportedOpError("QAttention mask_index must be 1D/2D/3D")

    scale = float(node.attrs.get("scale", 1.0 / math.sqrt(qk_head_size)))
    unidirectional = bool(int(node.attrs.get("unidirectional", 0)))
    mask_filter_value = float(node.attrs.get("mask_filter_value", -10000.0))

    # Per-column weight quantization: weight_scale/zp shape matches total_qkv
    weight_scale_shape = _value_shape(graph, node.inputs[4], node)
    weight_scale_per_column = weight_scale_shape not in {(), (1,)}
    if weight_scale_per_column and weight_scale_shape != (total_qkv,):
        raise ShapeInferenceError(
            "QAttention weight_scale must be scalar or have shape [total_qkv], "
            f"got {weight_scale_shape}"
        )
    if weight_scale_per_column:
        weight_zp_shape = _value_shape(graph, node.inputs[7], node)
        if weight_zp_shape != (total_qkv,):
            raise ShapeInferenceError(
                "QAttention per-column weight_zero_point must have shape [total_qkv], "
                f"got {weight_zp_shape}"
            )

    return QAttentionSpec(
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
        input_dtype=input_dtype,
        weight_dtype=weight_dtype,
        dtype=dtype,
        weight_scale_per_column=weight_scale_per_column,
    )


@register_lowering("QAttention")
def lower_qattention(graph: Graph, node: Node) -> QAttentionOp:
    """Lower com.microsoft::QAttention contrib op."""
    spec = resolve_qattention_spec(graph, node)
    input_name = node.inputs[0]
    weight_name = node.inputs[1]
    bias_name = node.inputs[2]
    input_scale_name = node.inputs[3]
    weight_scale_name = node.inputs[4]
    mask_name = _optional_name(node.inputs, 5)
    input_zp_name = node.inputs[6]
    weight_zp_name = node.inputs[7]
    past_name = _optional_name(node.inputs, 8) if len(node.inputs) > 8 else None
    output_name = node.outputs[0]
    present_name = _optional_name(node.outputs, 1) if len(node.outputs) > 1 else None
    return QAttentionOp(
        input=input_name,
        weight=weight_name,
        bias=bias_name,
        input_scale=input_scale_name,
        weight_scale=weight_scale_name,
        mask_index=mask_name,
        input_zero_point=input_zp_name,
        weight_zero_point=weight_zp_name,
        past=past_name,
        output=output_name,
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
        input_dtype=spec.input_dtype,
        weight_dtype=spec.weight_dtype,
        dtype=spec.dtype,
        weight_scale_per_column=spec.weight_scale_per_column,
    )

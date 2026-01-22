from __future__ import annotations

from dataclasses import dataclass

from shared.scalar_types import ScalarType

from ..codegen.c_emitter import RotaryEmbeddingOp
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from .common import optional_name, value_dtype, value_shape
from .registry import register_lowering


@dataclass(frozen=True)
class RotaryEmbeddingSpec:
    batch: int
    seq_len: int
    num_heads: int
    head_size: int
    rotary_dim: int
    rotary_dim_half: int
    input_rank: int


def _resolve_rotary_spec(
    graph: Graph, node: Node, dtype: ScalarType
) -> RotaryEmbeddingSpec:
    if not dtype.is_float:
        raise UnsupportedOpError("Unsupported op RotaryEmbedding")
    if len(node.inputs) < 3 or len(node.outputs) != 1:
        raise UnsupportedOpError("Unsupported op RotaryEmbedding")
    input_shape = value_shape(graph, node.inputs[0], node)
    input_rank = len(input_shape)
    if input_rank not in {3, 4}:
        raise ShapeInferenceError("RotaryEmbedding expects 3D or 4D input")
    if input_rank == 3:
        num_heads_attr = node.attrs.get("num_heads")
        if num_heads_attr is None:
            raise UnsupportedOpError(
                "RotaryEmbedding num_heads attribute is required for 3D inputs"
            )
        num_heads = int(num_heads_attr)
        if num_heads <= 0:
            raise ShapeInferenceError("RotaryEmbedding num_heads must be > 0")
        batch, seq_len, hidden_size = input_shape
        if hidden_size % num_heads != 0:
            raise ShapeInferenceError(
                "RotaryEmbedding hidden size must be divisible by num_heads"
            )
        head_size = hidden_size // num_heads
    else:
        batch, num_heads, seq_len, head_size = input_shape
        num_heads_attr = node.attrs.get("num_heads")
        if num_heads_attr is not None and int(num_heads_attr) != num_heads:
            raise ShapeInferenceError(
                "RotaryEmbedding num_heads must match input head dimension"
            )
    if head_size % 2 != 0:
        raise ShapeInferenceError("RotaryEmbedding head size must be even")
    rotary_dim = int(node.attrs.get("rotary_embedding_dim", 0))
    if rotary_dim == 0:
        rotary_dim = head_size
    if rotary_dim < 0 or rotary_dim > head_size:
        raise ShapeInferenceError(
            "RotaryEmbedding rotary_embedding_dim must be in [0, head_size]"
        )
    if rotary_dim % 2 != 0:
        raise ShapeInferenceError(
            "RotaryEmbedding rotary_embedding_dim must be even"
        )
    rotary_dim_half = rotary_dim // 2
    return RotaryEmbeddingSpec(
        batch=batch,
        seq_len=seq_len,
        num_heads=num_heads,
        head_size=head_size,
        rotary_dim=rotary_dim,
        rotary_dim_half=rotary_dim_half,
        input_rank=input_rank,
    )


@register_lowering("RotaryEmbedding")
def lower_rotary_embedding(graph: Graph, node: Node) -> RotaryEmbeddingOp:
    input_name = node.inputs[0]
    cos_name = node.inputs[1]
    sin_name = node.inputs[2]
    position_ids = optional_name(node.inputs, 3)
    dtype = value_dtype(graph, input_name, node)
    cos_dtype = value_dtype(graph, cos_name, node)
    sin_dtype = value_dtype(graph, sin_name, node)
    if cos_dtype != dtype or sin_dtype != dtype:
        raise ShapeInferenceError(
            "RotaryEmbedding inputs must share the same dtype"
        )
    spec = _resolve_rotary_spec(graph, node, dtype)
    input_shape = value_shape(graph, input_name, node)
    output_shape = value_shape(graph, node.outputs[0], node)
    if output_shape != input_shape:
        raise ShapeInferenceError(
            "RotaryEmbedding output shape must match input shape"
        )
    cos_shape = value_shape(graph, cos_name, node)
    sin_shape = value_shape(graph, sin_name, node)
    if cos_shape != sin_shape:
        raise ShapeInferenceError(
            "RotaryEmbedding cos/sin cache shapes must match"
        )
    position_shape = None
    position_dtype = None
    if position_ids is not None:
        position_shape = value_shape(graph, position_ids, node)
        if position_shape != (spec.batch, spec.seq_len):
            raise ShapeInferenceError(
                "RotaryEmbedding position_ids must match [batch, seq_len]"
            )
        position_dtype = value_dtype(graph, position_ids, node)
        if not position_dtype.is_integer:
            raise ShapeInferenceError(
                "RotaryEmbedding position_ids must be an integer tensor"
            )
        if len(cos_shape) != 2:
            raise ShapeInferenceError(
                "RotaryEmbedding expects 2D sin/cos caches with position_ids"
            )
        if cos_shape[1] != spec.rotary_dim_half:
            raise ShapeInferenceError(
                "RotaryEmbedding cos/sin cache last dim must match rotary_dim/2"
            )
    else:
        if len(cos_shape) != 3:
            raise ShapeInferenceError(
                "RotaryEmbedding expects 3D sin/cos caches without position_ids"
            )
        if cos_shape != (
            spec.batch,
            spec.seq_len,
            spec.rotary_dim_half,
        ):
            raise ShapeInferenceError(
                "RotaryEmbedding sin/cos cache shape must be "
                "[batch, seq_len, rotary_dim/2]"
            )
    interleaved = bool(int(node.attrs.get("interleaved", 0)))
    return RotaryEmbeddingOp(
        input0=input_name,
        cos_cache=cos_name,
        sin_cache=sin_name,
        position_ids=position_ids,
        output=node.outputs[0],
        input_shape=input_shape,
        cos_shape=cos_shape,
        sin_shape=sin_shape,
        position_ids_shape=position_shape,
        dtype=dtype,
        position_ids_dtype=position_dtype,
        rotary_dim=spec.rotary_dim,
        rotary_dim_half=spec.rotary_dim_half,
        head_size=spec.head_size,
        num_heads=spec.num_heads,
        seq_len=spec.seq_len,
        batch=spec.batch,
        input_rank=spec.input_rank,
        interleaved=interleaved,
    )

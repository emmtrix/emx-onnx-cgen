from __future__ import annotations

from dataclasses import dataclass

from shared.scalar_types import ScalarType

from ..ir.ops import RotaryEmbeddingOp
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from .common import value_dtype, value_shape
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
    graph: Graph, node: Node, dtype: ScalarType, cos_half_dim: int | None = None
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
        batch, seq_len, hidden_size = input_shape
        if num_heads_attr is not None:
            num_heads = int(num_heads_attr)
            if num_heads <= 0:
                raise ShapeInferenceError("RotaryEmbedding num_heads must be > 0")
            if hidden_size % num_heads != 0:
                raise ShapeInferenceError(
                    "RotaryEmbedding hidden size must be divisible by num_heads"
                )
            head_size = hidden_size // num_heads
        elif cos_half_dim is not None:
            # Infer num_heads from the cos/sin cache: each head is fully rotated,
            # so head_size = rotary_dim = 2 * cos_half_dim.
            rotary_embedding_dim = int(node.attrs.get("rotary_embedding_dim", 0))
            if rotary_embedding_dim == 0:
                head_size = 2 * cos_half_dim
            else:
                head_size = rotary_embedding_dim
            if hidden_size % head_size != 0:
                raise ShapeInferenceError(
                    "RotaryEmbedding hidden size must be divisible by inferred head_size"
                )
            num_heads = hidden_size // head_size
        else:
            raise UnsupportedOpError(
                "RotaryEmbedding num_heads attribute is required for 3D inputs"
            )
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
        raise ShapeInferenceError("RotaryEmbedding rotary_embedding_dim must be even")
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
    # Detect domain to determine input ordering.
    # ONNX standard (domain="" or None): [input, cos_cache, sin_cache, position_ids]
    # com.microsoft: [input, position_ids, cos_cache, sin_cache]
    domain = node.domain or ""
    if domain == "":
        # ONNX standard ordering
        if len(node.inputs) >= 4 and node.inputs[3]:
            position_ids = node.inputs[3]
        else:
            position_ids = None
        cos_name = node.inputs[1]
        sin_name = node.inputs[2]
    else:
        # com.microsoft ordering: position_ids before cos/sin
        if len(node.inputs) >= 4 and node.inputs[1]:
            position_ids = node.inputs[1]
            cos_name = node.inputs[2]
            sin_name = node.inputs[3]
        else:
            position_ids = None
            cos_name = node.inputs[1]
            sin_name = node.inputs[2]
    dtype = value_dtype(graph, input_name, node)
    cos_dtype = value_dtype(graph, cos_name, node)
    sin_dtype = value_dtype(graph, sin_name, node)
    if cos_dtype != dtype or sin_dtype != dtype:
        raise ShapeInferenceError("RotaryEmbedding inputs must share the same dtype")
    # Pre-fetch cos_shape so _resolve_rotary_spec can infer num_heads for 3D inputs
    # that omit the num_heads attribute.
    cos_shape = value_shape(graph, cos_name, node)
    cos_half_dim = cos_shape[-1] if len(cos_shape) >= 1 else None
    spec = _resolve_rotary_spec(graph, node, dtype, cos_half_dim=cos_half_dim)
    input_shape = value_shape(graph, input_name, node)
    output_shape = value_shape(graph, node.outputs[0], node)
    if output_shape != input_shape:
        raise ShapeInferenceError("RotaryEmbedding output shape must match input shape")
    sin_shape = value_shape(graph, sin_name, node)
    if cos_shape != sin_shape:
        raise ShapeInferenceError("RotaryEmbedding cos/sin cache shapes must match")
    position_shape = None
    position_dtype = None
    position_ids_broadcast = False
    if position_ids is not None:
        position_shape = value_shape(graph, position_ids, node)
        # Accept either (batch, seq_len) or (1,) — the latter broadcasts to all positions.
        valid_shapes = {
            (spec.batch, spec.seq_len),
            (1,),
        }
        if position_shape not in valid_shapes:
            raise ShapeInferenceError(
                "RotaryEmbedding position_ids shape must be [batch, seq_len] or [1], "
                f"got {position_shape}"
            )
        position_ids_broadcast = position_shape == (1,)
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
        position_ids_broadcast=position_ids_broadcast,
    )

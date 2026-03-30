from __future__ import annotations

from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from ..ir.ops import EmbedLayerNormOp
from shared.scalar_types import ScalarType
from .common import optional_name, value_dtype, value_shape
from .registry import register_lowering

_SUPPORTED_IDS_DTYPES = {ScalarType.I32, ScalarType.I64}


@register_lowering("EmbedLayerNormalization")
def lower_embed_layer_normalization(
    graph: Graph, node: Node
) -> EmbedLayerNormOp:
    if len(node.inputs) < 7:
        raise UnsupportedOpError(
            "EmbedLayerNormalization requires at least 7 inputs "
            "(input_ids, segment_ids, word_embedding, position_embedding, "
            "segment_embedding, gamma, beta)"
        )
    if len(node.outputs) < 1:
        raise UnsupportedOpError(
            "EmbedLayerNormalization requires at least 1 output"
        )

    input_ids_name = node.inputs[0]
    segment_ids_name = optional_name(node.inputs, 1)
    word_embedding_name = node.inputs[2]
    position_embedding_name = node.inputs[3]
    segment_embedding_name = optional_name(node.inputs, 4)
    gamma_name = node.inputs[5]
    beta_name = node.inputs[6]
    mask_name = optional_name(node.inputs, 7)
    position_ids_name = optional_name(node.inputs, 8)

    output_name = node.outputs[0]
    mask_index_name = optional_name(node.outputs, 1)
    embedding_sum_name = optional_name(node.outputs, 2)

    # Validate ids dtype
    ids_dtype = value_dtype(graph, input_ids_name, node)
    if ids_dtype not in _SUPPORTED_IDS_DTYPES:
        raise UnsupportedOpError(
            f"EmbedLayerNormalization input_ids dtype {ids_dtype} is not supported; "
            f"expected INT32 or INT64"
        )

    # Validate float dtype for embeddings
    float_dtype = value_dtype(graph, word_embedding_name, node)
    if not float_dtype.is_float:
        raise UnsupportedOpError(
            f"EmbedLayerNormalization word_embedding dtype {float_dtype} must be float"
        )

    # Resolve shapes
    input_ids_shape = value_shape(graph, input_ids_name, node)
    if len(input_ids_shape) != 2:
        raise ShapeInferenceError(
            f"EmbedLayerNormalization input_ids must be 2-D [batch, seq], "
            f"got shape {input_ids_shape}"
        )
    batch, seq = input_ids_shape

    word_emb_shape = value_shape(graph, word_embedding_name, node)
    if len(word_emb_shape) != 2:
        raise ShapeInferenceError(
            f"EmbedLayerNormalization word_embedding must be 2-D, "
            f"got shape {word_emb_shape}"
        )
    vocab_size, hidden_size = word_emb_shape

    pos_emb_shape = value_shape(graph, position_embedding_name, node)
    if len(pos_emb_shape) != 2 or pos_emb_shape[1] != hidden_size:
        raise ShapeInferenceError(
            f"EmbedLayerNormalization position_embedding must be [max_pos, {hidden_size}], "
            f"got shape {pos_emb_shape}"
        )
    max_pos = pos_emb_shape[0]

    if segment_embedding_name is not None:
        seg_emb_shape = value_shape(graph, segment_embedding_name, node)
        if len(seg_emb_shape) != 2 or seg_emb_shape[1] != hidden_size:
            raise ShapeInferenceError(
                f"EmbedLayerNormalization segment_embedding must be [*, {hidden_size}], "
                f"got shape {seg_emb_shape}"
            )

    # position_ids: shape [batch, seq] or [1, seq] (broadcast)
    pos_ids_batch = 1
    if position_ids_name is not None:
        pos_ids_shape = value_shape(graph, position_ids_name, node)
        if len(pos_ids_shape) != 2 or pos_ids_shape[1] != seq:
            raise ShapeInferenceError(
                f"EmbedLayerNormalization position_ids must be [batch_or_1, {seq}], "
                f"got shape {pos_ids_shape}"
            )
        pos_ids_batch = pos_ids_shape[0]

    epsilon = float(node.attrs.get("epsilon", 1e-12))

    return EmbedLayerNormOp(
        input_ids=input_ids_name,
        segment_ids=segment_ids_name,
        word_embedding=word_embedding_name,
        position_embedding=position_embedding_name,
        segment_embedding=segment_embedding_name,
        gamma=gamma_name,
        beta=beta_name,
        mask=mask_name,
        position_ids=position_ids_name,
        output=output_name,
        mask_index=mask_index_name,
        embedding_sum=embedding_sum_name,
        batch=batch,
        seq=seq,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        max_pos=max_pos,
        pos_ids_batch=pos_ids_batch,
        dtype=float_dtype,
        ids_dtype=ids_dtype,
        epsilon=epsilon,
    )

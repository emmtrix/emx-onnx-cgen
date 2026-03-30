from __future__ import annotations

from shared.scalar_types import ScalarType

from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from ..ir.ops import QEmbedLayerNormOp
from .common import optional_name, value_dtype, value_shape
from .registry import register_lowering

_SUPPORTED_IDS_DTYPES = {ScalarType.I32, ScalarType.I64}
_SUPPORTED_DATA_DTYPES = {ScalarType.I8, ScalarType.U8}


def _read_scalar_float(graph: Graph, name: str) -> float:
    for init in graph.initializers:
        if init.name == name:
            return float(init.data.flatten()[0])
    raise UnsupportedOpError(f"QEmbedLayerNormalization: {name!r} must be a constant initializer")


def _read_scalar_int(graph: Graph, name: str) -> int:
    for init in graph.initializers:
        if init.name == name:
            return int(init.data.flatten()[0])
    raise UnsupportedOpError(f"QEmbedLayerNormalization: {name!r} must be a constant initializer")


@register_lowering("QEmbedLayerNormalization")
def lower_qembed_layer_normalization(
    graph: Graph, node: Node
) -> QEmbedLayerNormOp:
    if len(node.inputs) < 13:
        raise UnsupportedOpError(
            "QEmbedLayerNormalization requires at least 13 inputs"
        )
    if len(node.outputs) < 1:
        raise UnsupportedOpError(
            "QEmbedLayerNormalization requires at least 1 output"
        )

    input_ids_name = node.inputs[0]
    segment_ids_name = optional_name(node.inputs, 1)
    word_embedding_name = node.inputs[2]
    position_embedding_name = node.inputs[3]
    segment_embedding_name = optional_name(node.inputs, 4)
    gamma_name = node.inputs[5]
    beta_name = node.inputs[6]
    mask_name = optional_name(node.inputs, 7)
    # inputs 8-17: scales and zero_points
    word_scale_name = node.inputs[8]
    pos_scale_name = node.inputs[9]
    seg_scale_name = optional_name(node.inputs, 10)
    gamma_scale_name = node.inputs[11]
    beta_scale_name = node.inputs[12]
    word_zp_name = optional_name(node.inputs, 13)
    pos_zp_name = optional_name(node.inputs, 14)
    seg_zp_name = optional_name(node.inputs, 15)
    gamma_zp_name = optional_name(node.inputs, 16)
    beta_zp_name = optional_name(node.inputs, 17)

    output_name = node.outputs[0]
    mask_index_name = optional_name(node.outputs, 1)

    ids_dtype = value_dtype(graph, input_ids_name, node)
    if ids_dtype not in _SUPPORTED_IDS_DTYPES:
        raise UnsupportedOpError(
            f"QEmbedLayerNormalization input_ids dtype {ids_dtype} is not supported"
        )

    data_dtype = value_dtype(graph, word_embedding_name, node)
    if data_dtype not in _SUPPORTED_DATA_DTYPES:
        raise UnsupportedOpError(
            f"QEmbedLayerNormalization embedding data dtype {data_dtype} must be INT8 or UINT8"
        )

    input_ids_shape = value_shape(graph, input_ids_name, node)
    if len(input_ids_shape) != 2:
        raise ShapeInferenceError(
            f"QEmbedLayerNormalization input_ids must be 2-D [batch, seq], "
            f"got shape {input_ids_shape}"
        )
    batch, seq = input_ids_shape

    word_emb_shape = value_shape(graph, word_embedding_name, node)
    if len(word_emb_shape) != 2:
        raise ShapeInferenceError(
            f"QEmbedLayerNormalization word_embedding must be 2-D, got {word_emb_shape}"
        )
    vocab_size, hidden_size = word_emb_shape

    pos_emb_shape = value_shape(graph, position_embedding_name, node)
    if len(pos_emb_shape) != 2 or pos_emb_shape[1] != hidden_size:
        raise ShapeInferenceError(
            f"QEmbedLayerNormalization position_embedding must be [max_pos, {hidden_size}], "
            f"got shape {pos_emb_shape}"
        )
    max_pos = pos_emb_shape[0]

    # Read scalar scale/zero_point constants
    word_scale = _read_scalar_float(graph, word_scale_name)
    pos_scale = _read_scalar_float(graph, pos_scale_name)
    seg_scale = _read_scalar_float(graph, seg_scale_name) if seg_scale_name else 1.0
    gamma_scale = _read_scalar_float(graph, gamma_scale_name)
    beta_scale = _read_scalar_float(graph, beta_scale_name)
    word_zp = _read_scalar_int(graph, word_zp_name) if word_zp_name else 0
    pos_zp = _read_scalar_int(graph, pos_zp_name) if pos_zp_name else 0
    seg_zp = _read_scalar_int(graph, seg_zp_name) if seg_zp_name else 0
    gamma_zp = _read_scalar_int(graph, gamma_zp_name) if gamma_zp_name else 0
    beta_zp = _read_scalar_int(graph, beta_zp_name) if beta_zp_name else 0

    epsilon = float(node.attrs.get("epsilon", 1e-12))

    return QEmbedLayerNormOp(
        input_ids=input_ids_name,
        segment_ids=segment_ids_name,
        word_embedding_data=word_embedding_name,
        position_embedding_data=position_embedding_name,
        segment_embedding_data=segment_embedding_name,
        gamma=gamma_name,
        beta=beta_name,
        mask=mask_name,
        output=output_name,
        mask_index=mask_index_name,
        batch=batch,
        seq=seq,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        max_pos=max_pos,
        word_scale=word_scale,
        word_zp=word_zp,
        pos_scale=pos_scale,
        pos_zp=pos_zp,
        seg_scale=seg_scale,
        seg_zp=seg_zp,
        gamma_scale=gamma_scale,
        gamma_zp=gamma_zp,
        beta_scale=beta_scale,
        beta_zp=beta_zp,
        ids_dtype=ids_dtype,
        data_dtype=data_dtype,
        epsilon=epsilon,
    )

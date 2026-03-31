from __future__ import annotations

from shared.scalar_types import ScalarType

from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from ..ir.ops import WordConvEmbeddingOp
from .common import value_dtype as _value_dtype
from .common import value_shape as _value_shape
from .registry import register_lowering


@register_lowering("WordConvEmbedding")
def lower_word_conv_embedding(graph: Graph, node: Node) -> WordConvEmbeddingOp:
    if len(node.inputs) != 4 or len(node.outputs) != 1:
        raise UnsupportedOpError(
            "WordConvEmbedding must have exactly 4 inputs and 1 output"
        )

    seq_name = node.inputs[0]
    weights_name = node.inputs[1]
    bias_name = node.inputs[2]
    char_emb_name = node.inputs[3]
    output_name = node.outputs[0]

    seq_shape = _value_shape(graph, seq_name, node)
    if len(seq_shape) != 2:
        raise ShapeInferenceError(
            f"WordConvEmbedding Sequence must be 2D [batch, max_word_len], got {seq_shape}"
        )
    batch, max_word_len = seq_shape

    seq_dtype = _value_dtype(graph, seq_name, node)
    if seq_dtype != ScalarType.I32:
        raise UnsupportedOpError(
            f"WordConvEmbedding Sequence must be int32, got {seq_dtype}"
        )

    w_shape = _value_shape(graph, weights_name, node)
    if len(w_shape) != 4:
        raise ShapeInferenceError(
            f"WordConvEmbedding W must be 4D [num_filters, 1, conv_window, char_emb_size], got {w_shape}"
        )
    num_filters, _, conv_window, char_emb_size = w_shape
    if _ != 1:
        raise ShapeInferenceError(
            f"WordConvEmbedding W dim 1 must be 1, got {_}"
        )

    b_shape = _value_shape(graph, bias_name, node)
    if b_shape != (num_filters,):
        raise ShapeInferenceError(
            f"WordConvEmbedding B shape must be ({num_filters},), got {b_shape}"
        )

    c_shape = _value_shape(graph, char_emb_name, node)
    if len(c_shape) != 2 or c_shape[1] != char_emb_size:
        raise ShapeInferenceError(
            f"WordConvEmbedding C shape must be [vocab_size, {char_emb_size}], got {c_shape}"
        )
    vocab_size = c_shape[0]

    if conv_window > max_word_len:
        raise ShapeInferenceError(
            f"WordConvEmbedding conv_window ({conv_window}) > max_word_len ({max_word_len})"
        )

    dtype = _value_dtype(graph, weights_name, node)
    if not dtype.is_float:
        raise UnsupportedOpError(
            f"WordConvEmbedding supports float weights, got {dtype}"
        )

    # Validate optional attributes against shapes
    if "char_embedding_size" in node.attrs:
        attr_char_emb = int(node.attrs["char_embedding_size"])
        if attr_char_emb != char_emb_size:
            raise ShapeInferenceError(
                f"WordConvEmbedding char_embedding_size attr ({attr_char_emb}) "
                f"does not match W shape ({char_emb_size})"
            )
    if "conv_window_size" in node.attrs:
        attr_conv_window = int(node.attrs["conv_window_size"])
        if attr_conv_window != conv_window:
            raise ShapeInferenceError(
                f"WordConvEmbedding conv_window_size attr ({attr_conv_window}) "
                f"does not match W shape ({conv_window})"
            )
    if "embedding_size" in node.attrs:
        attr_emb_size = int(node.attrs["embedding_size"])
        if attr_emb_size != num_filters:
            raise ShapeInferenceError(
                f"WordConvEmbedding embedding_size attr ({attr_emb_size}) "
                f"does not match W shape ({num_filters})"
            )

    return WordConvEmbeddingOp(
        sequence=seq_name,
        weights=weights_name,
        bias=bias_name,
        char_embedding=char_emb_name,
        output=output_name,
        batch=batch,
        max_word_len=max_word_len,
        vocab_size=vocab_size,
        char_emb_size=char_emb_size,
        conv_window=conv_window,
        num_filters=num_filters,
        dtype=dtype,
    )

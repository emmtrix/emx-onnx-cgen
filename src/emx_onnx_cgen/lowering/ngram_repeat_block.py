from __future__ import annotations

from shared.scalar_types import ScalarType

from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.ops import NGramRepeatBlockOp
from ..ir.model import Graph, Node
from .registry import register_lowering


@register_lowering("NGramRepeatBlock")
def lower_ngram_repeat_block(graph: Graph, node: Node) -> NGramRepeatBlockOp:
    if len(node.inputs) != 2 or len(node.outputs) != 1:
        raise UnsupportedOpError("NGramRepeatBlock must have 2 inputs and 1 output")

    input_ids_name = node.inputs[0]
    scores_name = node.inputs[1]
    output_name = node.outputs[0]

    input_ids_val = graph.find_value(input_ids_name)
    scores_val = graph.find_value(scores_name)
    output_val = graph.find_value(output_name)

    input_ids_dtype = input_ids_val.type.dtype
    scores_dtype = scores_val.type.dtype

    if input_ids_dtype not in {ScalarType.I64, ScalarType.I32}:
        raise UnsupportedOpError(
            f"NGramRepeatBlock: input_ids must be int32 or int64, got {input_ids_dtype.onnx_name}"
        )
    if scores_dtype != ScalarType.F32:
        raise UnsupportedOpError(
            f"NGramRepeatBlock: scores must be float32, got {scores_dtype.onnx_name}"
        )

    ids_shape = input_ids_val.type.shape
    scores_shape = scores_val.type.shape
    output_shape = output_val.type.shape

    if len(ids_shape) != 2:
        raise ShapeInferenceError(
            f"NGramRepeatBlock: input_ids must have rank 2, got shape {ids_shape}"
        )
    if len(scores_shape) != 2:
        raise ShapeInferenceError(
            f"NGramRepeatBlock: scores must have rank 2, got shape {scores_shape}"
        )
    if ids_shape[0] != scores_shape[0]:
        raise ShapeInferenceError(
            "NGramRepeatBlock: batch dimension mismatch between input_ids and scores"
        )

    batch_size = ids_shape[0]
    seq_len = ids_shape[1]
    vocab_size = scores_shape[1]

    if any(d < 0 for d in (batch_size, seq_len, vocab_size)):
        raise ShapeInferenceError(
            "NGramRepeatBlock does not support dynamic shapes; export with static shapes"
        )

    ngram_size_attr = node.attrs.get("ngram_size")
    if ngram_size_attr is None:
        raise UnsupportedOpError("NGramRepeatBlock: missing required attribute ngram_size")
    ngram_size = int(ngram_size_attr)
    if ngram_size < 1:
        raise UnsupportedOpError(
            f"NGramRepeatBlock: ngram_size must be >= 1, got {ngram_size}"
        )
    if ngram_size > seq_len:
        raise UnsupportedOpError(
            f"NGramRepeatBlock: ngram_size {ngram_size} > seq_len {seq_len}"
        )

    expected_output_shape = (batch_size, vocab_size)
    if tuple(output_shape) != expected_output_shape:
        raise ShapeInferenceError(
            f"NGramRepeatBlock: output shape must be {expected_output_shape}, got {output_shape}"
        )

    return NGramRepeatBlockOp(
        input_ids=input_ids_name,
        scores=scores_name,
        output=output_name,
        ngram_size=ngram_size,
        batch_size=batch_size,
        seq_len=seq_len,
        vocab_size=vocab_size,
    )

from __future__ import annotations

from ..errors import UnsupportedOpError
from ..ir.model import Graph, Node
from ..ir.ops import BifurcationDetectorOp
from ..lowering.common import value_shape
from .registry import register_lowering


@register_lowering("BifurcationDetector")
def lower_bifurcation_detector(graph: Graph, node: Node) -> BifurcationDetectorOp:
    n_inputs = len(node.inputs)
    if n_inputs not in {3, 4} or len(node.outputs) != 2:
        raise UnsupportedOpError(
            "BifurcationDetector expects 3 or 4 inputs and 2 outputs"
        )

    src_tokens_name = node.inputs[0]
    cur_tokens_name = node.inputs[1]
    match_idx_in_name = node.inputs[2]
    pred_tokens_name = node.inputs[3] if n_inputs == 4 else None
    tokens_name = node.outputs[0]
    match_idx_out_name = node.outputs[1]

    src_shape = value_shape(graph, src_tokens_name, node)
    cur_shape = value_shape(graph, cur_tokens_name, node)
    tokens_shape = value_shape(graph, tokens_name, node)

    if len(src_shape) != 1:
        raise UnsupportedOpError("BifurcationDetector src_tokens must be rank-1")
    if len(cur_shape) != 1:
        raise UnsupportedOpError("BifurcationDetector cur_tokens must be rank-1")
    if len(tokens_shape) != 1:
        raise UnsupportedOpError("BifurcationDetector tokens output must be rank-1")

    src_len = src_shape[0]
    cur_len = cur_shape[0]
    tokens_len = tokens_shape[0]

    if pred_tokens_name is not None:
        pred_shape = value_shape(graph, pred_tokens_name, node)
        if len(pred_shape) != 1:
            raise UnsupportedOpError("BifurcationDetector pred_tokens must be rank-1")
        pred_len = pred_shape[0]
    else:
        pred_len = 0

    min_ngram_size = int(node.attrs.get("min_ngram_size", 1))
    max_ngram_size = int(node.attrs.get("max_ngram_size", 3))

    return BifurcationDetectorOp(
        src_tokens=src_tokens_name,
        cur_tokens=cur_tokens_name,
        match_idx_in=match_idx_in_name,
        pred_tokens=pred_tokens_name,
        tokens=tokens_name,
        match_idx_out=match_idx_out_name,
        src_len=src_len,
        cur_len=cur_len,
        pred_len=pred_len,
        tokens_len=tokens_len,
        min_ngram_size=min_ngram_size,
        max_ngram_size=max_ngram_size,
    )

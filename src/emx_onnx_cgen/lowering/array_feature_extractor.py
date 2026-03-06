from __future__ import annotations

from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.context import GraphContext
from ..ir.model import Graph, Node
from ..ir.ops import ArrayFeatureExtractorOp
from .common import value_shape
from .registry import register_lowering


@register_lowering("ArrayFeatureExtractor")
def lower_array_feature_extractor(
    graph: Graph, node: Node
) -> ArrayFeatureExtractorOp:
    if len(node.inputs) != 2 or len(node.outputs) != 1:
        raise UnsupportedOpError(
            "ArrayFeatureExtractor must have 2 inputs and 1 output"
        )
    data_name, indices_name = node.inputs
    output_name = node.outputs[0]
    data_shape = value_shape(graph, data_name, node)
    if not data_shape:
        raise ShapeInferenceError("ArrayFeatureExtractor does not support scalar input")
    indices_shape = value_shape(graph, indices_name, node)
    if any(dim < 0 for dim in indices_shape):
        raise UnsupportedOpError(
            "ArrayFeatureExtractor does not support dynamic indices shapes"
        )
    feature_count = 1
    for dim in indices_shape:
        feature_count *= dim
    output_shape = (
        (1, feature_count) if len(data_shape) == 1 else (*data_shape[:-1], feature_count)
    )
    if isinstance(graph, GraphContext):
        graph.set_shape(output_name, output_shape)
    return ArrayFeatureExtractorOp(
        data=data_name,
        indices=indices_name,
        output=output_name,
    )

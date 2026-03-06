from __future__ import annotations

from ..errors import UnsupportedOpError
from ..ir.model import Graph, Node
from ..ir.ops import ArrayFeatureExtractorOp
from .common import value_dtype, value_shape
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
    indices_shape = value_shape(graph, indices_name, node)

    if not data_shape:
        raise UnsupportedOpError(
            "ArrayFeatureExtractor does not support scalar input tensors"
        )
    if not indices_shape:
        raise UnsupportedOpError(
            "ArrayFeatureExtractor requires indices to have rank >= 1"
        )

    data_dtype = value_dtype(graph, data_name, node)
    output_dtype = value_dtype(graph, output_name, node)
    if output_dtype != data_dtype:
        raise UnsupportedOpError(
            f"ArrayFeatureExtractor output dtype must match input dtype, "
            f"got {output_dtype.onnx_name} and {data_dtype.onnx_name}"
        )

    return ArrayFeatureExtractorOp(
        data=data_name,
        indices=indices_name,
        output=output_name,
    )

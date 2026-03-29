from __future__ import annotations

from shared.scalar_types import ScalarType

from ..errors import UnsupportedOpError
from ..ir.model import Graph, Node
from ..ir.ops import MurmurHash3Op
from ..ir.ops.misc import _MURMUR_HASH3_SUPPORTED_DTYPES
from .common import value_dtype, value_shape
from .registry import register_lowering

_OUTPUT_DTYPE_BY_POSITIVE = {
    0: ScalarType.I32,
    1: ScalarType.U32,
}


@register_lowering("MurmurHash3")
def lower_murmur_hash3(graph: Graph, node: Node) -> MurmurHash3Op:
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        raise UnsupportedOpError("MurmurHash3 must have 1 input and 1 output")

    input_name = node.inputs[0]
    output_name = node.outputs[0]

    input_dtype = value_dtype(graph, input_name, node)
    if input_dtype not in _MURMUR_HASH3_SUPPORTED_DTYPES:
        raise UnsupportedOpError(
            f"MurmurHash3 does not support input dtype {input_dtype.onnx_name}; "
            f"supported: int32, int64, float, double, string"
        )

    positive = int(node.attrs.get("positive", 0))
    if positive not in _OUTPUT_DTYPE_BY_POSITIVE:
        raise UnsupportedOpError(
            f"MurmurHash3 attribute 'positive' must be 0 or 1, got {positive}"
        )
    output_dtype = _OUTPUT_DTYPE_BY_POSITIVE[positive]

    declared_output_dtype = value_dtype(graph, output_name, node)
    if declared_output_dtype != output_dtype:
        raise UnsupportedOpError(
            f"MurmurHash3 output dtype mismatch: expected {output_dtype.onnx_name} "
            f"(positive={positive}), got {declared_output_dtype.onnx_name}"
        )

    input_shape = value_shape(graph, input_name, node)
    output_shape = value_shape(graph, output_name, node)
    if input_shape != output_shape:
        raise UnsupportedOpError(
            f"MurmurHash3 input shape {input_shape} does not match "
            f"output shape {output_shape}"
        )

    seed = int(node.attrs.get("seed", 0))

    return MurmurHash3Op(
        input0=input_name,
        output=output_name,
        seed=seed,
        input_dtype=input_dtype,
        output_dtype=output_dtype,
    )

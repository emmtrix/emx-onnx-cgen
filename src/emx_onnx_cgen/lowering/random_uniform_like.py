from __future__ import annotations

from shared.scalar_types import ScalarType

from ..dtypes import dtype_info
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from ..ir.ops import RandomUniformLikeOp
from .common import value_dtype, value_shape
from .registry import register_lowering

_SUPPORTED_OUTPUT_DTYPES = {
    ScalarType.F16,
    ScalarType.BF16,
    ScalarType.F32,
    ScalarType.F64,
}


def _normalized_seed(seed_attr: object | None) -> int | None:
    if seed_attr is None:
        return None
    return int(float(seed_attr))


@register_lowering("RandomUniformLike")
def lower_random_uniform_like(graph: Graph, node: Node) -> RandomUniformLikeOp:
    if len(node.inputs) != 1:
        raise UnsupportedOpError("RandomUniformLike must have 1 input")
    if len(node.outputs) != 1:
        raise UnsupportedOpError("RandomUniformLike must have 1 output")

    input_name = node.inputs[0]
    output_name = node.outputs[0]
    output_shape = value_shape(graph, output_name, node)
    input_shape = value_shape(graph, input_name, node)
    if output_shape != input_shape:
        raise ShapeInferenceError(
            "RandomUniformLike output shape must match input shape, "
            f"got {output_shape} for input {input_shape}"
        )

    input_dtype = value_dtype(graph, input_name, node)
    output_dtype = value_dtype(graph, output_name, node)
    dtype_attr = node.attrs.get("dtype")
    if dtype_attr is not None:
        attr_dtype = dtype_info(int(dtype_attr))
        if attr_dtype != output_dtype:
            raise UnsupportedOpError(
                "RandomUniformLike dtype attribute does not match output dtype"
            )
    elif input_dtype != output_dtype:
        raise UnsupportedOpError(
            "RandomUniformLike output dtype must match input dtype when dtype is omitted"
        )

    if output_dtype not in _SUPPORTED_OUTPUT_DTYPES:
        raise UnsupportedOpError(
            "RandomUniformLike output dtype must be float, "
            f"got {output_dtype.onnx_name}"
        )

    low = float(node.attrs.get("low", 0.0))
    high = float(node.attrs.get("high", 1.0))
    if high < low:
        raise UnsupportedOpError(
            "RandomUniformLike high must be greater than or equal to low, "
            f"got {high} < {low}"
        )

    return RandomUniformLikeOp(
        input0=input_name,
        output=output_name,
        low=low,
        high=high,
        seed=_normalized_seed(node.attrs.get("seed")),
    )

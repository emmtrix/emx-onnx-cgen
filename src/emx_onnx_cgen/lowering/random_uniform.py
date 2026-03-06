from __future__ import annotations

from shared.scalar_types import ScalarType

from ..dtypes import dtype_info
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.ops import RandomUniformOp
from ..ir.model import Graph, Node
from ..lowering.common import value_dtype, value_shape
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


@register_lowering("RandomUniform")
def lower_random_uniform(graph: Graph, node: Node) -> RandomUniformOp:
    if node.inputs:
        raise UnsupportedOpError("RandomUniform expects no inputs")
    if len(node.outputs) != 1:
        raise UnsupportedOpError("RandomUniform must have 1 output")

    output_name = node.outputs[0]
    output_dtype = value_dtype(graph, output_name, node)
    dtype_attr = node.attrs.get("dtype")
    if dtype_attr is not None:
        attr_dtype = dtype_info(int(dtype_attr))
        if attr_dtype != output_dtype:
            raise UnsupportedOpError(
                "RandomUniform dtype attribute does not match output dtype"
            )
    if output_dtype not in _SUPPORTED_OUTPUT_DTYPES:
        raise UnsupportedOpError(
            f"RandomUniform output dtype must be float, got {output_dtype.onnx_name}"
        )

    shape_attr = node.attrs.get("shape")
    if shape_attr is None:
        raise UnsupportedOpError("RandomUniform requires shape attribute")
    shape = tuple(int(dim) for dim in shape_attr)
    if any(dim < 0 for dim in shape):
        raise ShapeInferenceError("RandomUniform shape must contain non-negative dims")

    output_shape = value_shape(graph, output_name, node)
    if output_shape != shape:
        raise ShapeInferenceError(
            "RandomUniform output shape must match shape attribute, "
            f"got {output_shape} and {shape}"
        )

    low = float(node.attrs.get("low", 0.0))
    high = float(node.attrs.get("high", 1.0))
    if high < low:
        raise UnsupportedOpError(
            f"RandomUniform high must be greater than or equal to low, got {high} < {low}"
        )

    return RandomUniformOp(
        output=output_name,
        low=low,
        high=high,
        seed=_normalized_seed(node.attrs.get("seed")),
    )

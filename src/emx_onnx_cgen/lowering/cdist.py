from __future__ import annotations

from shared.scalar_types import ScalarType

from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.ops import CDistOp
from ..ir.model import Graph, Node
from .registry import register_lowering

_SUPPORTED_DTYPES = {ScalarType.F32, ScalarType.F64}
_SUPPORTED_METRICS = {"euclidean", "sqeuclidean"}


@register_lowering("CDist")
def lower_cdist(graph: Graph, node: Node) -> CDistOp:
    """Lower com.microsoft::CDist to CDistOp."""
    if len(node.inputs) != 2 or len(node.outputs) != 1:
        raise UnsupportedOpError("CDist must have 2 inputs and 1 output")

    input_a_name = node.inputs[0]
    input_b_name = node.inputs[1]
    output_name = node.outputs[0]

    input_a = graph.find_value(input_a_name)
    input_b = graph.find_value(input_b_name)
    output = graph.find_value(output_name)

    dtype_a = input_a.type.dtype
    dtype_b = input_b.type.dtype
    output_dtype = output.type.dtype

    if dtype_a not in _SUPPORTED_DTYPES:
        raise UnsupportedOpError(
            f"CDist does not support dtype {dtype_a.onnx_name}; expected float or double"
        )
    if dtype_a != dtype_b:
        raise UnsupportedOpError(
            f"CDist expects matching input dtypes, got {dtype_a.onnx_name} and {dtype_b.onnx_name}"
        )
    if dtype_a != output_dtype:
        raise UnsupportedOpError(
            f"CDist expects matching input/output dtypes, got {dtype_a.onnx_name} and {output_dtype.onnx_name}"
        )

    shape_a = input_a.type.shape
    shape_b = input_b.type.shape

    if len(shape_a) != 2:
        raise ShapeInferenceError(
            f"CDist expects input A to have rank 2, got shape {shape_a}"
        )
    if len(shape_b) != 2:
        raise ShapeInferenceError(
            f"CDist expects input B to have rank 2, got shape {shape_b}"
        )
    if shape_a[1] != shape_b[1]:
        raise ShapeInferenceError(
            f"CDist expects K dimension to match: A shape {shape_a}, B shape {shape_b}"
        )
    if shape_a[1] < 0:
        raise ShapeInferenceError(
            "CDist does not support dynamic K dimension; export with static shapes"
        )

    metric_attr = node.attrs.get("metric", b"euclidean")
    if isinstance(metric_attr, bytes):
        metric = metric_attr.decode("utf-8")
    else:
        metric = str(metric_attr)
    if metric not in _SUPPORTED_METRICS:
        raise UnsupportedOpError(
            f"CDist does not support metric '{metric}'; expected 'euclidean' or 'sqeuclidean'"
        )

    expected_output_shape = (shape_a[0], shape_b[0])
    output_shape = output.type.shape
    if tuple(output_shape) != expected_output_shape:
        raise ShapeInferenceError(
            f"CDist output shape must be {expected_output_shape}, got {output_shape}"
        )

    return CDistOp(
        input_a=input_a_name,
        input_b=input_b_name,
        output=output_name,
        metric=metric,
    )

from __future__ import annotations

import onnx

from shared.scalar_types import ScalarType

from ..dtypes import scalar_type_from_onnx
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.context import GraphContext
from ..ir.model import Graph, Node
from ..ir.ops import BitCastOp
from .common import ensure_supported_dtype, onnx_opset_version, value_dtype, value_shape
from .registry import register_lowering


def _shapes_match_or_are_dynamic(
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
) -> bool:
    if len(input_shape) != len(output_shape):
        return False
    return all(
        output_dim < 0 or output_dim == input_dim
        for input_dim, output_dim in zip(input_shape, output_shape)
    )


@register_lowering("BitCast")
def lower_bitcast(graph: Graph, node: Node) -> BitCastOp:
    opset = onnx_opset_version(graph, "")
    if opset is not None and opset < 26:
        raise UnsupportedOpError(
            f"BitCast requires opset >= 26, got opset {opset}. "
            "Hint: export the model with a newer ONNX opset."
        )
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        raise UnsupportedOpError("BitCast must have 1 input and 1 output")
    if "to" not in node.attrs:
        raise UnsupportedOpError("BitCast requires a 'to' attribute")

    input_dtype = ensure_supported_dtype(value_dtype(graph, node.inputs[0], node))
    if input_dtype == ScalarType.STRING:
        raise UnsupportedOpError("BitCast does not support string inputs")

    target_onnx_dtype = int(node.attrs["to"])
    target_dtype = scalar_type_from_onnx(target_onnx_dtype)
    if target_dtype is None:
        name = onnx.TensorProto.DataType.Name(target_onnx_dtype)
        raise UnsupportedOpError(
            f"BitCast 'to' dtype {target_onnx_dtype} ({name}) is not supported"
        )
    target_dtype = ensure_supported_dtype(target_dtype)
    if target_dtype == ScalarType.STRING:
        raise UnsupportedOpError("BitCast does not support string outputs")

    output_dtype = value_dtype(graph, node.outputs[0], node)
    if output_dtype != target_dtype:
        raise UnsupportedOpError(
            "BitCast output dtype must match 'to' attribute, "
            f"got {output_dtype.onnx_name} and {target_dtype.onnx_name}"
        )

    input_bits = input_dtype.bits
    if input_bits is None:
        input_bits = int(input_dtype.np_dtype.itemsize) * 8
    target_bits = target_dtype.bits
    if target_bits is None:
        target_bits = int(target_dtype.np_dtype.itemsize) * 8

    if input_bits != target_bits:
        raise UnsupportedOpError(
            "BitCast requires matching bit-width dtypes, "
            f"got {input_dtype.onnx_name} ({input_bits} bits) -> "
            f"{target_dtype.onnx_name} ({target_bits} bits)"
        )
    if input_bits % 8 != 0:
        raise UnsupportedOpError(
            "BitCast for sub-byte dtypes is not supported yet, "
            f"got {input_dtype.onnx_name} ({input_bits} bits)."
        )

    input_shape = value_shape(graph, node.inputs[0], node)
    output_shape = value_shape(graph, node.outputs[0], node)
    if input_shape == () and output_shape:
        input_shape = output_shape
    if output_shape == () and input_shape:
        output_shape = input_shape
    if not _shapes_match_or_are_dynamic(input_shape, output_shape):
        raise ShapeInferenceError("BitCast input and output shapes must match")
    output_shape = input_shape
    if isinstance(graph, GraphContext):
        graph.set_shape(node.outputs[0], output_shape)

    return BitCastOp(
        input0=node.inputs[0],
        output=node.outputs[0],
    )

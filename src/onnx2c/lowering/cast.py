from __future__ import annotations

import onnx

from ..codegen.c_emitter import CastOp
from ..dtypes import scalar_type_from_onnx
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from .common import ensure_supported_dtype, value_dtype, value_shape
from .registry import register_lowering


@register_lowering("Cast")
def lower_cast(graph: Graph, node: Node) -> CastOp:
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        raise UnsupportedOpError("Cast must have 1 input and 1 output")
    if "to" not in node.attrs:
        raise UnsupportedOpError("Cast requires a 'to' attribute")
    target_onnx_dtype = int(node.attrs["to"])
    target_dtype = scalar_type_from_onnx(target_onnx_dtype)
    if target_dtype is None:
        name = onnx.TensorProto.DataType.Name(target_onnx_dtype)
        raise UnsupportedOpError(
            f"Cast 'to' dtype {target_onnx_dtype} ({name}) is not supported"
        )
    target_dtype = ensure_supported_dtype(target_dtype)
    input_dtype = value_dtype(graph, node.inputs[0], node)
    output_dtype = value_dtype(graph, node.outputs[0], node)
    if output_dtype != target_dtype:
        raise UnsupportedOpError(
            "Cast output dtype must match 'to' attribute, "
            f"got {output_dtype.onnx_name} and {target_dtype.onnx_name}"
        )
    input_shape = value_shape(graph, node.inputs[0], node)
    output_shape = value_shape(graph, node.outputs[0], node)
    if input_shape != output_shape:
        raise ShapeInferenceError("Cast input and output shapes must match")
    return CastOp(
        input0=node.inputs[0],
        output=node.outputs[0],
        shape=output_shape,
        input_dtype=input_dtype,
        dtype=output_dtype,
    )


@register_lowering("CastLike")
def lower_castlike(graph: Graph, node: Node) -> CastOp:
    if len(node.inputs) != 2 or len(node.outputs) != 1:
        raise UnsupportedOpError("CastLike must have 2 inputs and 1 output")
    input_dtype = value_dtype(graph, node.inputs[0], node)
    like_dtype = value_dtype(graph, node.inputs[1], node)
    target_dtype = ensure_supported_dtype(like_dtype)
    output_dtype = value_dtype(graph, node.outputs[0], node)
    if output_dtype != target_dtype:
        raise UnsupportedOpError(
            "CastLike output dtype must match like input dtype, "
            f"got {output_dtype.onnx_name} and {target_dtype.onnx_name}"
        )
    input_shape = value_shape(graph, node.inputs[0], node)
    output_shape = value_shape(graph, node.outputs[0], node)
    if input_shape != output_shape:
        raise ShapeInferenceError("CastLike input and output shapes must match")
    return CastOp(
        input0=node.inputs[0],
        output=node.outputs[0],
        shape=output_shape,
        input_dtype=input_dtype,
        dtype=output_dtype,
    )

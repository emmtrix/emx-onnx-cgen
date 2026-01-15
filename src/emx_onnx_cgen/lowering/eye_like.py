from __future__ import annotations

from ..codegen.c_emitter import EyeLikeOp
from ..dtypes import scalar_type_from_onnx
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from .common import value_dtype, value_shape
from .registry import register_lowering


@register_lowering("EyeLike")
def lower_eye_like(graph: Graph, node: Node) -> EyeLikeOp:
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        raise UnsupportedOpError("EyeLike must have 1 input and 1 output")
    input_shape = value_shape(graph, node.inputs[0], node)
    output_shape = value_shape(graph, node.outputs[0], node)
    if input_shape != output_shape:
        raise ShapeInferenceError("EyeLike input and output shapes must match")
    if len(output_shape) < 2:
        raise UnsupportedOpError("EyeLike expects input rank >= 2")
    input_dtype = value_dtype(graph, node.inputs[0], node)
    output_dtype = value_dtype(graph, node.outputs[0], node)
    dtype_attr = node.attrs.get("dtype")
    if dtype_attr is not None:
        target_dtype = scalar_type_from_onnx(int(dtype_attr))
        if target_dtype is None:
            raise UnsupportedOpError(
                f"EyeLike dtype {dtype_attr} is not supported"
            )
        if output_dtype != target_dtype:
            raise UnsupportedOpError(
                "EyeLike output dtype must match dtype attribute, "
                f"got {output_dtype.onnx_name} and {target_dtype.onnx_name}"
            )
    k = int(node.attrs.get("k", 0))
    return EyeLikeOp(
        input0=node.inputs[0],
        output=node.outputs[0],
        output_shape=output_shape,
        k=k,
        dtype=output_dtype,
        input_dtype=input_dtype,
    )

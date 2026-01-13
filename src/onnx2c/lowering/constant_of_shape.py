from __future__ import annotations

from onnx import numpy_helper

from shared.scalar_types import ScalarType

from ..codegen.c_emitter import ConstantOfShapeOp
from ..dtypes import scalar_type_from_onnx
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from .registry import register_lowering


def _value_shape(graph: Graph, name: str, node: Node) -> tuple[int, ...]:
    try:
        return graph.find_value(name).type.shape
    except KeyError as exc:
        raise ShapeInferenceError(
            f"Missing shape for value '{name}' in op {node.op_type}. "
            "Hint: run ONNX shape inference or export with static shapes."
        ) from exc


def _value_dtype(graph: Graph, name: str, node: Node) -> ScalarType:
    try:
        return graph.find_value(name).type.dtype
    except KeyError as exc:
        raise ShapeInferenceError(
            f"Missing dtype for value '{name}' in op {node.op_type}. "
            "Hint: run ONNX shape inference or export with static shapes."
        ) from exc


def _parse_value_attr(node: Node) -> tuple[ScalarType, float | int | bool]:
    value_attr = node.attrs.get("value")
    if value_attr is None:
        return ScalarType.F32, 0.0
    dtype = scalar_type_from_onnx(value_attr.data_type)
    if dtype is None:
        raise UnsupportedOpError(
            f"ConstantOfShape has unsupported value dtype {value_attr.data_type}"
        )
    data = numpy_helper.to_array(value_attr)
    if data.size != 1:
        raise UnsupportedOpError("ConstantOfShape value must be a scalar")
    return dtype, data.reshape(-1)[0].item()


@register_lowering("ConstantOfShape")
def lower_constant_of_shape(graph: Graph, node: Node) -> ConstantOfShapeOp:
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        raise UnsupportedOpError("ConstantOfShape must have 1 input and 1 output")
    input_shape = _value_shape(graph, node.inputs[0], node)
    if len(input_shape) != 1:
        raise UnsupportedOpError("ConstantOfShape expects a 1D shape input")
    output_shape = _value_shape(graph, node.outputs[0], node)
    if input_shape[0] != len(output_shape):
        raise ShapeInferenceError(
            "ConstantOfShape input length must match output rank"
        )
    for dim in output_shape:
        if dim <= 0:
            raise ShapeInferenceError("Dynamic or zero dims are not supported")
    input_dtype = _value_dtype(graph, node.inputs[0], node)
    if input_dtype != ScalarType.I64:
        raise UnsupportedOpError(
            "ConstantOfShape expects int64 shape input, "
            f"got {input_dtype.onnx_name}"
        )
    output_dtype = _value_dtype(graph, node.outputs[0], node)
    value_dtype, value = _parse_value_attr(node)
    if output_dtype != value_dtype:
        raise UnsupportedOpError(
            "ConstantOfShape output dtype must match value dtype, "
            f"got {output_dtype.onnx_name} and {value_dtype.onnx_name}"
        )
    return ConstantOfShapeOp(
        input0=node.inputs[0],
        output=node.outputs[0],
        input_shape=input_shape,
        shape=output_shape,
        value=value,
        dtype=output_dtype,
        input_dtype=input_dtype,
    )

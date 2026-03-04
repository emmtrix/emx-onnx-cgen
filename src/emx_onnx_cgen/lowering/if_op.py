from __future__ import annotations

import numpy as np
import onnx
from onnx import numpy_helper

from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.context import GraphContext
from ..ir.model import Node, SequenceType, TensorType
from ..ir.ops import IfOptionalSequenceConstOp
from shared.scalar_types import ScalarType
from .common import value_dtype, value_shape
from .registry import register_lowering


def _branch_graph(node: Node, name: str) -> onnx.GraphProto:
    value = node.attrs.get(name)
    if not isinstance(value, onnx.GraphProto):
        raise UnsupportedOpError("If requires then_branch and else_branch graph attrs")
    return value


def _find_branch_node(
    branch: onnx.GraphProto, output_name: str
) -> onnx.NodeProto | None:
    for candidate in branch.node:
        if output_name in candidate.output:
            return candidate
    return None


def _branch_tensor(
    branch: onnx.GraphProto,
    output_name: str,
) -> tuple[bool, np.ndarray | None]:
    producer = _find_branch_node(branch, output_name)
    if producer is None:
        raise UnsupportedOpError("If optional sequence branch output has no producer")
    if producer.op_type == "Optional":
        if len(producer.input) == 0:
            return False, None
        if len(producer.input) != 1:
            raise UnsupportedOpError("Optional in If branch must have 0 or 1 input")
        return _branch_tensor(branch, producer.input[0])
    if producer.op_type == "SequenceEmpty":
        return True, np.array([], dtype=np.float32)
    if producer.op_type != "SequenceConstruct":
        raise UnsupportedOpError(
            "If optional sequence branches currently support SequenceConstruct only"
        )
    if len(producer.input) != 1 or not producer.input[0]:
        raise UnsupportedOpError(
            "If optional sequence branch SequenceConstruct must have exactly one input"
        )
    tensor_name = producer.input[0]
    for initializer in branch.initializer:
        if initializer.name == tensor_name:
            return True, numpy_helper.to_array(initializer)
    tensor_producer = _find_branch_node(branch, tensor_name)
    if tensor_producer is None or tensor_producer.op_type != "Constant":
        raise UnsupportedOpError(
            "If optional sequence branch tensor must be Constant or initializer"
        )
    for attr in tensor_producer.attribute:
        if attr.name == "value":
            return True, numpy_helper.to_array(attr.t)
    raise UnsupportedOpError("If optional sequence branch Constant must define value")


@register_lowering("If")
def lower_if_optional_sequence(
    graph: GraphContext, node: Node
) -> IfOptionalSequenceConstOp:
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        raise UnsupportedOpError(
            "If currently supports exactly one input and one output"
        )
    cond = node.inputs[0]
    if value_dtype(graph, cond, node) != ScalarType.BOOL:
        raise UnsupportedOpError("If condition must be bool")
    cond_shape = value_shape(graph, cond, node)
    if cond_shape not in {(), (1,)}:
        raise ShapeInferenceError("If condition must be a scalar or size-1 tensor")

    output_name = node.outputs[0]
    output_value = graph.find_value(output_name)
    if not isinstance(output_value.type, SequenceType):
        raise UnsupportedOpError("If currently supports sequence outputs only")
    elem_type = output_value.type.elem
    if not isinstance(elem_type, TensorType):
        raise UnsupportedOpError("If optional sequence output element must be tensor")

    then_branch = _branch_graph(node, "then_branch")
    else_branch = _branch_graph(node, "else_branch")
    if len(then_branch.output) != 1 or len(else_branch.output) != 1:
        raise UnsupportedOpError("If sequence branches must have one output")

    true_present, true_array = _branch_tensor(then_branch, then_branch.output[0].name)
    false_present, false_array = _branch_tensor(else_branch, else_branch.output[0].name)

    def _normalize(values: np.ndarray | None) -> tuple[float | int | bool, ...]:
        if values is None:
            return ()
        arr = np.asarray(values)
        if arr.size == 0:
            return ()
        expected_shape = elem_type.shape
        if expected_shape and tuple(arr.shape) != expected_shape:
            raise ShapeInferenceError(
                "If optional sequence branch tensor shape must match output element shape"
            )
        if arr.dtype != elem_type.dtype.np_dtype:
            arr = arr.astype(elem_type.dtype.np_dtype, copy=False)
        return tuple(arr.reshape(-1).tolist())

    true_values = _normalize(true_array)
    false_values = _normalize(false_array)
    elem_count = int(np.prod(elem_type.shape)) if elem_type.shape else 1
    if true_present and len(true_values) not in {0, elem_count}:
        raise ShapeInferenceError(
            "If true branch optional sequence tensor has invalid size"
        )
    if false_present and len(false_values) not in {0, elem_count}:
        raise ShapeInferenceError(
            "If false branch optional sequence tensor has invalid size"
        )

    output_present = (
        f"{output_name}_present" if output_value.type.is_optional else None
    )
    graph.set_shape(output_name, elem_type.shape)
    graph.set_dtype(output_name, elem_type.dtype)
    return IfOptionalSequenceConstOp(
        cond=cond,
        output_sequence=output_name,
        output_present=output_present,
        true_present=true_present,
        false_present=false_present,
        true_values=true_values,
        false_values=false_values,
    )

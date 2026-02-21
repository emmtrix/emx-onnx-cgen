from __future__ import annotations

from shared.scalar_types import ScalarType

from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.context import GraphContext
from ..ir.model import Graph, Node, SequenceType, TensorType
from ..ir.ops import SequenceInsertOp
from .common import resolve_int_list_from_value, value_dtype, value_shape
from .registry import register_lowering


def _sequence_type(graph: Graph | GraphContext, name: str, node: Node) -> SequenceType:
    try:
        value = graph.find_value(name)
    except KeyError as exc:
        raise ShapeInferenceError(
            f"Missing sequence value '{name}' in op {node.op_type}. "
            "Hint: run ONNX shape inference or export with static shapes."
        ) from exc
    if not isinstance(value.type, SequenceType):
        raise UnsupportedOpError(
            f"SequenceInsert expects a sequence input for '{name}', got tensor"
        )
    return value.type


def _is_fully_static_tensor_type(tensor_type: TensorType) -> bool:
    if tensor_type.shape == () and not tensor_type.dim_params:
        # ONNX sequence element shapes may be omitted by exporters; these are
        # imported as rank-0 placeholders and must be treated as unknown.
        return False
    return all(dim_param is None for dim_param in tensor_type.dim_params)


def _ensure_tensor_shape_compatible(
    actual: TensorType,
    expected: TensorType,
    *,
    message: str,
) -> None:
    if not _is_fully_static_tensor_type(actual):
        return
    if not _is_fully_static_tensor_type(expected):
        return
    if actual.shape != expected.shape:
        raise ShapeInferenceError(message)


def _producer_for_output(graph: Graph, output_name: str) -> Node | None:
    for producer in graph.nodes:
        if output_name in producer.outputs:
            return producer
    return None


@register_lowering("SequenceInsert")
def lower_sequence_insert(graph: Graph, node: Node) -> SequenceInsertOp:
    if len(node.inputs) not in {2, 3} or len(node.outputs) != 1:
        raise UnsupportedOpError("SequenceInsert must have 2 or 3 inputs and 1 output")
    input_sequence, tensor_name = node.inputs[0], node.inputs[1]
    if not input_sequence or not tensor_name:
        raise UnsupportedOpError("SequenceInsert inputs must be provided")
    position_name = node.inputs[2] if len(node.inputs) == 3 and node.inputs[2] else None
    output_sequence = node.outputs[0]
    if not output_sequence:
        raise UnsupportedOpError("SequenceInsert output must be provided")

    input_sequence_type = _sequence_type(graph, input_sequence, node)
    output_sequence_type = _sequence_type(graph, output_sequence, node)

    tensor_value = graph.find_value(tensor_name)
    if not isinstance(tensor_value.type, TensorType):
        raise UnsupportedOpError("SequenceInsert tensor input must be a tensor")
    tensor_type = tensor_value.type

    if input_sequence_type.elem.dtype != tensor_type.dtype:
        raise UnsupportedOpError(
            "SequenceInsert tensor dtype must match sequence element dtype"
        )
    _ensure_tensor_shape_compatible(
        tensor_type,
        input_sequence_type.elem,
        message="SequenceInsert tensor shape must match input sequence element shape",
    )
    if output_sequence_type.elem.dtype != input_sequence_type.elem.dtype:
        raise UnsupportedOpError(
            "SequenceInsert output sequence dtype must match input sequence dtype"
        )

    _ensure_tensor_shape_compatible(
        output_sequence_type.elem,
        input_sequence_type.elem,
        message="SequenceInsert output sequence element shape must match input sequence",
    )
    _ensure_tensor_shape_compatible(
        tensor_type,
        output_sequence_type.elem,
        message="SequenceInsert tensor shape must match output sequence element shape",
    )

    if position_name is not None:
        pos_dtype = value_dtype(graph, position_name, node)
        if pos_dtype not in {ScalarType.I32, ScalarType.I64}:
            raise UnsupportedOpError("SequenceInsert position must be int32 or int64")
        pos_shape = value_shape(graph, position_name, node)
        if pos_shape not in {(), (1,)}:
            raise ShapeInferenceError(
                "SequenceInsert position must be a scalar or size-1 tensor"
            )
        resolve_int_list_from_value(graph, position_name, node)

    return SequenceInsertOp(
        input_sequence=input_sequence,
        tensor=tensor_name,
        position=position_name,
        output_sequence=output_sequence,
    )

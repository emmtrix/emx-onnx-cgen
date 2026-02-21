from __future__ import annotations

from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node, SequenceType, TensorType
from ..ir.ops import ConcatFromSequenceOp, ConcatOp
from .registry import register_lowering


def _normalize_axis(axis: int, rank: int) -> int:
    normalized = axis + rank if axis < 0 else axis
    if normalized < 0 or normalized >= rank:
        raise UnsupportedOpError(
            f"ConcatFromSequence axis {axis} out of range for rank {rank}"
        )
    return normalized


def _dim_known(dim: int) -> bool:
    return dim > 0


def _dims_compatible(
    actual: int, expected: int, *, actual_dim_param: str | None
) -> bool:
    if actual_dim_param is not None:
        return True
    if not _dim_known(actual):
        return True
    if not _dim_known(expected):
        return True
    return actual == expected


@register_lowering("ConcatFromSequence")
def lower_concat_from_sequence(
    graph: Graph, node: Node
) -> ConcatFromSequenceOp | ConcatOp:
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        raise UnsupportedOpError("ConcatFromSequence must have 1 input and 1 output")
    input_sequence = node.inputs[0]
    output = node.outputs[0]
    if not input_sequence or not output:
        raise UnsupportedOpError("ConcatFromSequence input and output must be provided")

    input_value = graph.find_value(input_sequence)
    if not isinstance(input_value.type, SequenceType):
        raise UnsupportedOpError("ConcatFromSequence input must be a sequence")
    output_value = graph.find_value(output)
    if not isinstance(output_value.type, TensorType):
        raise UnsupportedOpError("ConcatFromSequence output must be a tensor")

    elem_type = input_value.type.elem
    output_type = output_value.type
    if elem_type.dtype != output_type.dtype:
        raise UnsupportedOpError(
            "ConcatFromSequence output dtype must match sequence element dtype"
        )

    new_axis = int(node.attrs.get("new_axis", 0))
    if new_axis not in {0, 1}:
        raise UnsupportedOpError("ConcatFromSequence new_axis must be 0 or 1")
    axis = _normalize_axis(
        int(node.attrs.get("axis", 0)), len(elem_type.shape) + new_axis
    )

    elem_rank = len(elem_type.shape)
    expected_rank = elem_rank + (1 if new_axis else 0)
    if len(output_type.shape) != expected_rank:
        raise ShapeInferenceError(
            "ConcatFromSequence output rank must match sequence element rank and new_axis"
        )

    if new_axis:
        for dim in range(axis):
            if not _dims_compatible(
                output_type.shape[dim],
                elem_type.shape[dim],
                actual_dim_param=output_type.dim_params[dim],
            ):
                raise ShapeInferenceError(
                    "ConcatFromSequence output shape mismatch before inserted axis"
                )
        for dim in range(axis, elem_rank):
            if not _dims_compatible(
                output_type.shape[dim + 1],
                elem_type.shape[dim],
                actual_dim_param=output_type.dim_params[dim + 1],
            ):
                raise ShapeInferenceError(
                    "ConcatFromSequence output shape mismatch after inserted axis"
                )
    else:
        for dim in range(elem_rank):
            if dim == axis:
                continue
            if not _dims_compatible(
                output_type.shape[dim],
                elem_type.shape[dim],
                actual_dim_param=output_type.dim_params[dim],
            ):
                raise ShapeInferenceError(
                    "ConcatFromSequence output non-concat dimensions must match sequence element shape"
                )

    producer = next((n for n in graph.nodes if input_sequence in n.outputs), None)

    if (
        producer is not None
        and producer.op_type == "SequenceConstruct"
        and not new_axis
    ):
        return ConcatOp(
            inputs=tuple(name for name in producer.inputs if name),
            output=output,
            axis=axis,
        )

    if producer is not None and producer.op_type == "SequenceConstruct":
        sequence_len = len(producer.inputs)
        if new_axis:
            if (
                output_type.dim_params[axis] is None
                and _dim_known(output_type.shape[axis])
                and output_type.shape[axis] != sequence_len
            ):
                raise ShapeInferenceError(
                    "ConcatFromSequence output axis size must match SequenceConstruct length"
                )
        else:
            expected = elem_type.shape[axis] * sequence_len
            if (
                output_type.dim_params[axis] is None
                and _dim_known(output_type.shape[axis])
                and output_type.shape[axis] != expected
            ):
                raise ShapeInferenceError(
                    "ConcatFromSequence concat axis size must match SequenceConstruct-expanded axis"
                )

    return ConcatFromSequenceOp(
        input_sequence=input_sequence,
        output=output,
        axis=axis,
        new_axis=bool(new_axis),
        elem_shape=elem_type.shape,
    )

from __future__ import annotations

from shared.scalar_types import ScalarType

from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node, SequenceType, TensorType
from ..ir.ops import SplitToSequenceOp
from ..validation import normalize_axis
from .common import optional_name, resolve_int_list_from_value, value_dtype, value_shape
from .registry import register_lowering


def _dim_compatible(actual: int, expected: int, *, has_dim_param: bool) -> bool:
    if has_dim_param:
        return True
    if actual <= 0 or expected <= 0:
        return True
    return actual == expected


def _split_metadata(
    graph: Graph,
    split_name: str | None,
    node: Node,
    *,
    axis_size: int,
) -> tuple[tuple[int, ...] | None, bool]:
    if split_name is None:
        return tuple(1 for _ in range(axis_size)), False

    split_shape = value_shape(graph, split_name, node)
    split_dtype = value_dtype(graph, split_name, node)
    if split_dtype not in {ScalarType.I32, ScalarType.I64}:
        raise UnsupportedOpError("SplitToSequence split input must be int32 or int64")
    if split_shape == ():
        split_values = resolve_int_list_from_value(graph, split_name, node)
        if split_values is None:
            return None, True
        if len(split_values) != 1:
            raise ShapeInferenceError(
                "SplitToSequence scalar split must contain one value"
            )
        split_size = split_values[0]
        if split_size <= 0:
            raise ShapeInferenceError("SplitToSequence scalar split must be positive")
        full_chunks, remainder = divmod(axis_size, split_size)
        sizes = [split_size] * full_chunks
        if remainder:
            sizes.append(remainder)
        return tuple(sizes), True

    if len(split_shape) != 1:
        raise UnsupportedOpError(
            "SplitToSequence split input must be a scalar or 1D tensor"
        )

    split_values = resolve_int_list_from_value(graph, split_name, node)
    if split_values is None:
        return None, False
    if any(size <= 0 for size in split_values):
        raise ShapeInferenceError("SplitToSequence split sizes must be positive")
    if sum(split_values) != axis_size:
        raise ShapeInferenceError(
            "SplitToSequence split sizes must sum to axis dimension"
        )
    return tuple(split_values), False


@register_lowering("SplitToSequence")
def lower_split_to_sequence(graph: Graph, node: Node) -> SplitToSequenceOp:
    if len(node.inputs) < 1 or len(node.inputs) > 2 or len(node.outputs) != 1:
        raise UnsupportedOpError("SplitToSequence must have 1-2 inputs and 1 output")
    input_name = node.inputs[0]
    if not input_name or not node.outputs[0]:
        raise UnsupportedOpError("SplitToSequence input and output must be provided")

    input_value = graph.find_value(input_name)
    if not isinstance(input_value.type, TensorType):
        raise UnsupportedOpError("SplitToSequence input must be a tensor")

    output_name = node.outputs[0]
    output_value = graph.find_value(output_name)
    if not isinstance(output_value.type, SequenceType):
        raise UnsupportedOpError("SplitToSequence output must be a sequence")
    if output_value.type.elem.dtype != input_value.type.dtype:
        raise UnsupportedOpError(
            "SplitToSequence output element dtype must match input dtype"
        )

    input_shape = value_shape(graph, input_name, node)
    axis = normalize_axis(int(node.attrs.get("axis", 0)), input_shape, node)
    axis_size = input_shape[axis]
    if axis_size < 0:
        raise ShapeInferenceError(
            "SplitToSequence requires a static axis dimension or shape-inference dims"
        )

    split_name = optional_name(node.inputs, 1)
    keepdims = bool(int(node.attrs.get("keepdims", 1)))
    split_sizes, split_scalar = _split_metadata(
        graph,
        split_name,
        node,
        axis_size=axis_size,
    )
    if split_name is not None:
        keepdims = True

    elem_shape = list(input_shape)
    if keepdims:
        if split_sizes and split_sizes[0] > 0:
            elem_shape[axis] = split_sizes[0]
    else:
        del elem_shape[axis]

    output_elem = output_value.type.elem
    if len(output_elem.shape) != len(elem_shape):
        raise ShapeInferenceError(
            "SplitToSequence output element rank must match inferred split rank"
        )
    for index, (actual, expected) in enumerate(zip(output_elem.shape, elem_shape)):
        if not _dim_compatible(
            actual,
            expected,
            has_dim_param=output_elem.dim_params[index] is not None,
        ):
            raise ShapeInferenceError(
                "SplitToSequence output element shape must match inferred split shape"
            )

    return SplitToSequenceOp(
        input0=input_name,
        split=split_name,
        output_sequence=output_name,
        axis=axis,
        keepdims=keepdims,
        split_sizes=split_sizes,
        split_scalar=split_scalar,
    )

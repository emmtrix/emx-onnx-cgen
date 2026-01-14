from __future__ import annotations

import numpy as np

from shared.scalar_types import ScalarType

from ..codegen.c_emitter import SplitOp
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Initializer, Node
from ..lowering.common import optional_name, value_dtype, value_shape
from ..validation import normalize_axis
from .registry import register_lowering


def _find_initializer(graph: Graph, name: str) -> Initializer | None:
    for initializer in graph.initializers:
        if initializer.name == name:
            return initializer
    return None


def _read_split_sizes(graph: Graph, name: str, node: Node) -> list[int]:
    initializer = _find_initializer(graph, name)
    if initializer is None:
        raise UnsupportedOpError(
            f"{node.op_type} split input must be a constant initializer"
        )
    if initializer.type.dtype not in {ScalarType.I64, ScalarType.I32}:
        raise UnsupportedOpError(
            f"{node.op_type} split input must be int64 or int32"
        )
    if len(initializer.type.shape) != 1:
        raise UnsupportedOpError(
            f"{node.op_type} split input must be a 1D tensor"
        )
    values = np.array(initializer.data, dtype=np.int64).reshape(-1)
    if values.size == 0:
        raise ShapeInferenceError(
            f"{node.op_type} split input cannot be empty"
        )
    return [int(value) for value in values]


def _validate_static_dims(shape: tuple[int, ...], node: Node) -> None:
    if any(dim <= 0 for dim in shape):
        raise ShapeInferenceError(
            f"{node.op_type} does not support zero or dynamic dims"
        )


def _normalize_num_outputs(node: Node, output_count: int) -> int:
    num_outputs_attr = node.attrs.get("num_outputs")
    if num_outputs_attr is None:
        return output_count
    num_outputs = int(num_outputs_attr)
    if num_outputs <= 0:
        raise UnsupportedOpError("Split num_outputs must be positive")
    if output_count != num_outputs:
        raise ShapeInferenceError(
            f"Split expects {num_outputs} outputs, got {output_count}"
        )
    return num_outputs


@register_lowering("Split")
def lower_split(graph: Graph, node: Node) -> SplitOp:
    if len(node.inputs) < 1 or len(node.outputs) < 1:
        raise UnsupportedOpError("Split must have at least 1 input and 1 output")
    if len(node.inputs) > 2:
        raise UnsupportedOpError("Split supports up to 2 inputs")
    input_name = node.inputs[0]
    if not input_name:
        raise UnsupportedOpError("Split input must be provided")
    input_shape = value_shape(graph, input_name, node)
    _validate_static_dims(input_shape, node)
    axis = normalize_axis(int(node.attrs.get("axis", 0)), input_shape, node)
    output_shapes = [
        value_shape(graph, output, node) for output in node.outputs
    ]
    input_dtype = value_dtype(graph, input_name, node)
    output_dtypes = {value_dtype(graph, output, node) for output in node.outputs}
    if output_dtypes != {input_dtype}:
        dtype_names = ", ".join(
            dtype.onnx_name for dtype in sorted(output_dtypes, key=str)
        )
        raise UnsupportedOpError(
            f"Split expects matching dtypes, got {dtype_names}"
        )
    split_name = optional_name(node.inputs, 1)
    if split_name is not None and "num_outputs" in node.attrs:
        raise UnsupportedOpError(
            "Split cannot specify both split input and num_outputs"
        )
    if split_name is not None:
        split_sizes = _read_split_sizes(graph, split_name, node)
        if len(split_sizes) != len(node.outputs):
            raise ShapeInferenceError(
                f"Split expects {len(split_sizes)} outputs, got {len(node.outputs)}"
            )
        if any(size <= 0 for size in split_sizes):
            raise ShapeInferenceError("Split sizes must be positive")
        if sum(split_sizes) != input_shape[axis]:
            raise ShapeInferenceError(
                "Split sizes must sum to the axis dimension"
            )
    else:
        num_outputs = _normalize_num_outputs(node, len(node.outputs))
        axis_dim = input_shape[axis]
        if axis_dim < num_outputs:
            raise ShapeInferenceError(
                "Split axis dimension must be >= num_outputs"
            )
        base = axis_dim // num_outputs
        remainder = axis_dim % num_outputs
        if base <= 0:
            raise ShapeInferenceError("Split axis size must be positive")
        split_sizes = [base + 1] * remainder + [base] * (
            num_outputs - remainder
        )
    computed_shapes: list[tuple[int, ...]] = []
    for size, output_shape in zip(split_sizes, output_shapes):
        if size <= 0:
            raise ShapeInferenceError("Split output size must be positive")
        shape = list(input_shape)
        shape[axis] = size
        computed_shape = tuple(shape)
        if output_shape != computed_shape:
            raise ShapeInferenceError(
                f"Split output shape must be {computed_shape}, got {output_shape}"
            )
        computed_shapes.append(computed_shape)
    return SplitOp(
        input0=input_name,
        outputs=tuple(node.outputs),
        input_shape=input_shape,
        output_shapes=tuple(computed_shapes),
        axis=axis,
        split_sizes=tuple(split_sizes),
        dtype=input_dtype,
        input_dtype=input_dtype,
    )

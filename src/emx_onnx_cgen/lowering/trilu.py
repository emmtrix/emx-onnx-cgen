from __future__ import annotations

import numpy as np

from shared.scalar_types import ScalarType

from ..ir.ops import TriluOp
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Initializer, Node
from ..lowering.common import optional_name, value_dtype, value_shape
from .registry import register_lowering


def _find_initializer(graph: Graph, name: str) -> Initializer | None:
    for initializer in graph.initializers:
        if initializer.name == name:
            return initializer
    return None


def _is_scalar_shape(shape: tuple[int, ...]) -> bool:
    return shape == () or shape == (1,)


def _read_k_initializer(initializer: Initializer, node: Node) -> int:
    if initializer.type.dtype != ScalarType.I64:
        raise UnsupportedOpError(
            f"{node.op_type} k input must be int64"
        )
    data = np.array(initializer.data, dtype=np.int64).reshape(-1)
    if data.size != 1:
        raise UnsupportedOpError(f"{node.op_type} k input must be scalar")
    return int(data[0])


@register_lowering("Trilu")
def lower_trilu(graph: Graph, node: Node) -> TriluOp:
    if len(node.inputs) not in {1, 2} or len(node.outputs) != 1:
        raise UnsupportedOpError("Trilu must have 1 or 2 inputs and 1 output")
    input_name = node.inputs[0]
    if not input_name:
        raise UnsupportedOpError("Trilu input must be provided")
    input_shape = value_shape(graph, input_name, node)
    output_shape = value_shape(graph, node.outputs[0], node)
    if input_shape != output_shape:
        raise ShapeInferenceError("Trilu input and output shapes must match")
    if len(output_shape) < 2:
        raise UnsupportedOpError("Trilu expects input rank >= 2")
    input_dtype = value_dtype(graph, input_name, node)
    output_dtype = value_dtype(graph, node.outputs[0], node)
    if input_dtype != output_dtype:
        raise UnsupportedOpError(
            "Trilu expects matching input/output dtypes, "
            f"got {input_dtype.onnx_name} and {output_dtype.onnx_name}"
        )
    upper_attr = node.attrs.get("upper", 1)
    upper = bool(int(upper_attr))
    k_input = optional_name(node.inputs, 1)
    k_value = 0
    k_input_name = None
    k_input_shape = None
    k_input_dtype = None
    if k_input:
        k_initializer = _find_initializer(graph, k_input)
        if k_initializer is not None:
            k_value = _read_k_initializer(k_initializer, node)
        else:
            k_shape = value_shape(graph, k_input, node)
            if not _is_scalar_shape(k_shape):
                raise UnsupportedOpError("Trilu k input must be scalar")
            k_dtype = value_dtype(graph, k_input, node)
            if k_dtype != ScalarType.I64:
                raise UnsupportedOpError("Trilu k input must be int64")
            k_input_name = k_input
            k_input_shape = k_shape
            k_input_dtype = k_dtype
    return TriluOp(
        input0=input_name,
        output=node.outputs[0],
        input_shape=input_shape,
        output_shape=output_shape,
        upper=upper,
        k_value=k_value,
        k_input=k_input_name,
        k_input_shape=k_input_shape,
        k_input_dtype=k_input_dtype,
        dtype=output_dtype,
        input_dtype=input_dtype,
    )

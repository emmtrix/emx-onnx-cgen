from __future__ import annotations

import numpy as np

from shared.scalar_types import ScalarType

from ..dtypes import dtype_info
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Initializer, Node
from ..ir.ops import HammingWindowOp
from ..lowering.common import value_dtype, value_shape
from .registry import register_lowering


_SUPPORTED_INPUT_DTYPES = {ScalarType.I32, ScalarType.I64}
_SUPPORTED_OUTPUT_DTYPES = {
    ScalarType.U8,
    ScalarType.U16,
    ScalarType.U32,
    ScalarType.U64,
    ScalarType.I8,
    ScalarType.I16,
    ScalarType.I32,
    ScalarType.I64,
    ScalarType.F16,
    ScalarType.F32,
    ScalarType.F64,
}


def _find_initializer(graph: Graph, name: str) -> Initializer | None:
    for initializer in graph.initializers:
        if initializer.name == name:
            return initializer
    return None


def _read_scalar_initializer(
    graph: Graph, name: str, node: Node
) -> int | None:
    initializer = _find_initializer(graph, name)
    if initializer is None:
        return None
    data = np.array(initializer.data)
    if data.size != 1:
        raise UnsupportedOpError(
            f"{node.op_type} size input must be a scalar"
        )
    return int(data.reshape(-1)[0].item())


def _is_scalar_shape(shape: tuple[int, ...]) -> bool:
    return shape == () or shape == (1,)


@register_lowering("HammingWindow")
def lower_hamming_window(graph: Graph, node: Node) -> HammingWindowOp:
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        raise UnsupportedOpError("HammingWindow must have 1 input and 1 output")
    size_shape = value_shape(graph, node.inputs[0], node)
    if not _is_scalar_shape(size_shape):
        raise UnsupportedOpError("HammingWindow size input must be a scalar")
    input_dtype = value_dtype(graph, node.inputs[0], node)
    if input_dtype not in _SUPPORTED_INPUT_DTYPES:
        raise UnsupportedOpError(
            f"HammingWindow size input must be int32 or int64, got {input_dtype.onnx_name}"
        )
    output_shape = value_shape(graph, node.outputs[0], node)
    if len(output_shape) != 1:
        raise ShapeInferenceError("HammingWindow output must be 1D")
    output_dtype = value_dtype(graph, node.outputs[0], node)
    if output_dtype not in _SUPPORTED_OUTPUT_DTYPES:
        raise UnsupportedOpError(
            "HammingWindow output dtype must be numeric, "
            f"got {output_dtype.onnx_name}"
        )
    output_datatype = node.attrs.get("output_datatype")
    if output_datatype is not None:
        attr_dtype = dtype_info(int(output_datatype))
        if attr_dtype != output_dtype:
            raise UnsupportedOpError(
                "HammingWindow output_datatype does not match output dtype"
            )
    periodic = int(node.attrs.get("periodic", 1))
    if periodic not in {0, 1}:
        raise UnsupportedOpError("HammingWindow periodic must be 0 or 1")
    size_value = _read_scalar_initializer(graph, node.inputs[0], node)
    if size_value is not None:
        if size_value < 0:
            raise ShapeInferenceError(
                "HammingWindow size must be non-negative"
            )
        if output_shape[0] != size_value:
            raise ShapeInferenceError(
                "HammingWindow output length does not match size input"
            )
    return HammingWindowOp(
        size=node.inputs[0],
        output=node.outputs[0],
        output_shape=output_shape,
        periodic=periodic == 1,
        dtype=output_dtype,
        input_dtype=input_dtype,
    )

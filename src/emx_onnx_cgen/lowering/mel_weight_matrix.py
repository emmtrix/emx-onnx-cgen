from __future__ import annotations

import numpy as np

from shared.scalar_types import ScalarType

from ..dtypes import dtype_info
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Initializer, Node
from ..ir.ops import MelWeightMatrixOp
from ..lowering.common import value_dtype, value_shape
from .registry import register_lowering

_SUPPORTED_INT_DTYPES = {ScalarType.I32, ScalarType.I64}
_SUPPORTED_FLOAT_DTYPES = {
    ScalarType.F16,
    ScalarType.BF16,
    ScalarType.F32,
    ScalarType.F64,
}
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
    ScalarType.BF16,
    ScalarType.F32,
    ScalarType.F64,
}


def _find_initializer(graph: Graph, name: str) -> Initializer | None:
    for initializer in graph.initializers:
        if initializer.name == name:
            return initializer
    return None


def _read_scalar_initializer(graph: Graph, name: str, node: Node) -> float | int | None:
    initializer = _find_initializer(graph, name)
    if initializer is None:
        return None
    data = np.array(initializer.data)
    if data.size != 1:
        raise UnsupportedOpError(f"{node.op_type} inputs must be scalars")
    return data.reshape(-1)[0].item()


def _is_scalar_shape(shape: tuple[int, ...]) -> bool:
    return shape == () or shape == (1,)


@register_lowering("MelWeightMatrix")
def lower_mel_weight_matrix(graph: Graph, node: Node) -> MelWeightMatrixOp:
    if len(node.inputs) != 5 or len(node.outputs) != 1:
        raise UnsupportedOpError("MelWeightMatrix must have 5 inputs and 1 output")

    int_inputs = (node.inputs[0], node.inputs[1], node.inputs[2])
    float_inputs = (node.inputs[3], node.inputs[4])

    for input_name in node.inputs:
        if not _is_scalar_shape(value_shape(graph, input_name, node)):
            raise UnsupportedOpError("MelWeightMatrix inputs must be scalars")

    for input_name in int_inputs:
        dtype = value_dtype(graph, input_name, node)
        if dtype not in _SUPPORTED_INT_DTYPES:
            raise UnsupportedOpError(
                "MelWeightMatrix num_mel_bins/dft_length/sample_rate must be int32 or int64"
            )

    for input_name in float_inputs:
        dtype = value_dtype(graph, input_name, node)
        if dtype not in _SUPPORTED_FLOAT_DTYPES:
            raise UnsupportedOpError(
                "MelWeightMatrix lower_edge_hertz/upper_edge_hertz must be float16/bfloat16/float/double"
            )

    output_shape = value_shape(graph, node.outputs[0], node)
    if len(output_shape) != 2:
        raise ShapeInferenceError("MelWeightMatrix output must be rank 2")

    output_dtype = value_dtype(graph, node.outputs[0], node)
    if output_dtype not in _SUPPORTED_OUTPUT_DTYPES:
        raise UnsupportedOpError(
            f"MelWeightMatrix output dtype must be numeric, got {output_dtype.onnx_name}"
        )

    output_datatype = node.attrs.get("output_datatype")
    if output_datatype is not None:
        attr_dtype = dtype_info(int(output_datatype))
        if attr_dtype != output_dtype:
            raise UnsupportedOpError(
                "MelWeightMatrix output_datatype does not match output dtype"
            )

    num_mel_bins = _read_scalar_initializer(graph, node.inputs[0], node)
    dft_length = _read_scalar_initializer(graph, node.inputs[1], node)
    sample_rate = _read_scalar_initializer(graph, node.inputs[2], node)
    lower_edge_hertz = _read_scalar_initializer(graph, node.inputs[3], node)
    upper_edge_hertz = _read_scalar_initializer(graph, node.inputs[4], node)

    if num_mel_bins is not None and int(num_mel_bins) <= 0:
        raise ShapeInferenceError("MelWeightMatrix num_mel_bins must be > 0")
    if dft_length is not None and int(dft_length) <= 0:
        raise ShapeInferenceError("MelWeightMatrix dft_length must be > 0")
    if sample_rate is not None and int(sample_rate) <= 0:
        raise ShapeInferenceError("MelWeightMatrix sample_rate must be > 0")
    if (
        lower_edge_hertz is not None
        and upper_edge_hertz is not None
        and float(lower_edge_hertz) > float(upper_edge_hertz)
    ):
        raise ShapeInferenceError(
            "MelWeightMatrix lower_edge_hertz must be <= upper_edge_hertz"
        )

    if num_mel_bins is not None and dft_length is not None:
        expected_shape = (int(dft_length) // 2 + 1, int(num_mel_bins))
        if output_shape != expected_shape:
            raise ShapeInferenceError(
                f"MelWeightMatrix output shape must be {expected_shape}, got {output_shape}"
            )

    return MelWeightMatrixOp(
        num_mel_bins=node.inputs[0],
        dft_length=node.inputs[1],
        sample_rate=node.inputs[2],
        lower_edge_hertz=node.inputs[3],
        upper_edge_hertz=node.inputs[4],
        output=node.outputs[0],
        values=(),
    )

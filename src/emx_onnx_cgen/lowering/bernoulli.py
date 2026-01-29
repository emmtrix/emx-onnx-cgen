from __future__ import annotations

from shared.scalar_types import ScalarType

from ..dtypes import dtype_info
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from ..ir.ops import BernoulliOp
from .common import value_dtype as _value_dtype
from .common import value_shape as _value_shape
from .registry import register_lowering


_SUPPORTED_INPUT_DTYPES = {ScalarType.F16, ScalarType.F32, ScalarType.F64}
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
    ScalarType.BOOL,
}


@register_lowering("Bernoulli")
def lower_bernoulli(graph: Graph, node: Node) -> BernoulliOp:
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        raise UnsupportedOpError("Bernoulli must have 1 input and 1 output")
    input_shape = _value_shape(graph, node.inputs[0], node)
    output_shape = _value_shape(graph, node.outputs[0], node)
    if input_shape != output_shape:
        raise ShapeInferenceError(
            "Bernoulli output shape must match input shape, "
            f"got {output_shape} for input {input_shape}"
        )
    input_dtype = _value_dtype(graph, node.inputs[0], node)
    if input_dtype not in _SUPPORTED_INPUT_DTYPES:
        raise UnsupportedOpError(
            "Bernoulli input dtype must be float, "
            f"got {input_dtype.onnx_name}"
        )
    output_dtype = _value_dtype(graph, node.outputs[0], node)
    dtype_attr = node.attrs.get("dtype")
    if dtype_attr is not None:
        attr_dtype = dtype_info(int(dtype_attr))
        if attr_dtype != output_dtype:
            raise UnsupportedOpError(
                "Bernoulli dtype attribute does not match output dtype"
            )
    if output_dtype not in _SUPPORTED_OUTPUT_DTYPES:
        raise UnsupportedOpError(
            "Bernoulli output dtype must be numeric or bool, "
            f"got {output_dtype.onnx_name}"
        )
    seed_value = node.attrs.get("seed")
    seed = None
    if seed_value is not None:
        seed = int(seed_value)
    return BernoulliOp(
        input0=node.inputs[0],
        output=node.outputs[0],
        input_shape=input_shape,
        output_shape=output_shape,
        input_dtype=input_dtype,
        dtype=output_dtype,
        seed=seed,
    )

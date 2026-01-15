from __future__ import annotations

from shared.scalar_functions import ScalarFunction
from shared.scalar_types import ScalarType

from ..codegen.c_emitter import MultiInputBinaryOp
from ..errors import UnsupportedOpError
from ..ir.model import Graph, Node
from ..lowering.common import node_dtype, value_dtype, value_shape
from ..lowering.registry import register_lowering
from ..ops import binary_op_symbol

VARIADIC_OP_FUNCTIONS: dict[str, ScalarFunction] = {
    "Sum": ScalarFunction.ADD,
    "Mean": ScalarFunction.MEAN,
    "Max": ScalarFunction.MAXIMUM,
    "Min": ScalarFunction.MINIMUM,
    "And": ScalarFunction.LOGICAL_AND,
    "Or": ScalarFunction.LOGICAL_OR,
    "Xor": ScalarFunction.LOGICAL_XOR,
    "BitwiseAnd": ScalarFunction.BITWISE_AND,
    "BitwiseOr": ScalarFunction.BITWISE_OR,
    "BitwiseXor": ScalarFunction.BITWISE_XOR,
}

BINARY_ONLY_OPS = {
    "And",
    "Or",
    "Xor",
    "BitwiseAnd",
    "BitwiseOr",
    "BitwiseXor",
}


def _validate_inputs(
    graph: Graph, node: Node, *, function: ScalarFunction
) -> tuple[ScalarType, tuple[int, ...]]:
    if len(node.outputs) != 1:
        raise UnsupportedOpError(f"{node.op_type} must have 1 output")
    if node.op_type in BINARY_ONLY_OPS:
        if len(node.inputs) != 2:
            raise UnsupportedOpError(
                f"{node.op_type} must have exactly 2 inputs"
            )
    elif len(node.inputs) < 2:
        raise UnsupportedOpError(
            f"{node.op_type} must have at least 2 inputs"
        )
    for name in node.inputs:
        if not name:
            raise UnsupportedOpError(f"{node.op_type} input must be provided")
    op_dtype = node_dtype(graph, node, *node.inputs, *node.outputs)
    output_dtype = value_dtype(graph, node.outputs[0], node)
    if op_dtype != output_dtype:
        raise UnsupportedOpError(
            f"{node.op_type} expects matching input/output dtypes, "
            f"got {op_dtype.onnx_name} and {output_dtype.onnx_name}"
        )
    output_shape = value_shape(graph, node.outputs[0], node)
    for name in node.inputs:
        input_shape = value_shape(graph, name, node)
        if input_shape != output_shape:
            raise UnsupportedOpError(
                f"{node.op_type} expects identical input/output shapes"
            )
    op_spec = binary_op_symbol(function, dtype=op_dtype, validate_attrs=False)
    if op_spec is None:
        raise UnsupportedOpError(
            f"{node.op_type} does not support dtype {op_dtype.onnx_name}"
        )
    return op_dtype, output_shape


def _lower_variadic(graph: Graph, node: Node) -> MultiInputBinaryOp:
    function = VARIADIC_OP_FUNCTIONS[node.op_type]
    op_dtype, output_shape = _validate_inputs(graph, node, function=function)
    op_spec = binary_op_symbol(function, dtype=op_dtype, validate_attrs=False)
    if op_spec is None:
        raise UnsupportedOpError(
            f"{node.op_type} does not support dtype {op_dtype.onnx_name}"
        )
    return MultiInputBinaryOp(
        inputs=tuple(node.inputs),
        output=node.outputs[0],
        function=function,
        operator_kind=op_spec.kind,
        shape=output_shape,
        dtype=op_dtype,
        input_dtype=op_dtype,
    )


for _op_type in VARIADIC_OP_FUNCTIONS:
    register_lowering(_op_type)(_lower_variadic)

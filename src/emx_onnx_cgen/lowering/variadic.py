from __future__ import annotations

from shared.scalar_functions import ScalarFunction

from ..errors import UnsupportedOpError
from ..ir.model import Graph, Node
from ..ir.ops import MultiInputBinaryOp
from ..lowering.registry import register_lowering
from ..ops import OperatorKind

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

VARIADIC_OP_OPERATOR_KINDS: dict[str, OperatorKind] = {
    "Sum": OperatorKind.INFIX,
    "Mean": OperatorKind.EXPR,
    "Max": OperatorKind.FUNC,
    "Min": OperatorKind.FUNC,
    "And": OperatorKind.INFIX,
    "Or": OperatorKind.INFIX,
    "Xor": OperatorKind.INFIX,
    "BitwiseAnd": OperatorKind.INFIX,
    "BitwiseOr": OperatorKind.INFIX,
    "BitwiseXor": OperatorKind.INFIX,
}


def _lower_variadic(graph: Graph, node: Node) -> MultiInputBinaryOp:
    if len(node.outputs) != 1:
        raise UnsupportedOpError(f"{node.op_type} must have 1 output")
    return MultiInputBinaryOp(
        op_type=node.op_type,
        inputs=tuple(node.inputs),
        output=node.outputs[0],
        function=VARIADIC_OP_FUNCTIONS[node.op_type],
        operator_kind=VARIADIC_OP_OPERATOR_KINDS[node.op_type],
        min_inputs=2,
        max_inputs=2 if node.op_type in BINARY_ONLY_OPS else None,
    )


for _op_type in VARIADIC_OP_FUNCTIONS:
    register_lowering(_op_type)(_lower_variadic)

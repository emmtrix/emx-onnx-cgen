from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum

import numpy as np

from shared.scalar_functions import ScalarFunction
from shared.scalar_types import ScalarType

from .errors import UnsupportedOpError


class OperatorKind(str, Enum):
    INFIX = "infix"
    FUNC = "func"
    EXPR = "expr"


@dataclass(frozen=True)
class BinaryOpSpec:
    operator: str
    kind: OperatorKind
    apply: Callable[[np.ndarray, np.ndarray], np.ndarray]


BINARY_OP_TYPES = {
    "Add",
    "And",
    "Div",
    "Equal",
    "Greater",
    "GreaterOrEqual",
    "Less",
    "LessOrEqual",
    "Max",
    "Mean",
    "Min",
    "Mod",
    "Mul",
    "Or",
    "PRelu",
    "Pow",
    "Sub",
    "Sum",
    "Xor",
}

COMPARE_OP_TYPES = {
    "Equal",
    "Greater",
    "GreaterOrEqual",
    "Less",
    "LessOrEqual",
}

UNARY_OP_TYPES = {
    "Abs",
    "Atanh",
    "Ceil",
    "Cos",
    "Exp",
    "Floor",
    "Identity",
    "Log",
    "Neg",
    "Not",
    "Relu",
    "Sin",
    "Sqrt",
    "Tan",
    "Tanh",
}


def _format_float_literal(value: float, dtype: ScalarType) -> str:
    formatted = f"{value:.9g}"
    if "e" not in formatted and "E" not in formatted and "." not in formatted:
        formatted = f"{formatted}.0"
    if dtype in {ScalarType.F16, ScalarType.F32}:
        return f"{formatted}f"
    return formatted


UNARY_SYMBOLS_BOOL = {
    ScalarFunction.POSITIVE: "identity",
    ScalarFunction.LOGICAL_NOT: "!",
}

UNARY_SYMBOLS_INT64 = {
    ScalarFunction.ABS: "llabs",
    ScalarFunction.POSITIVE: "identity",
    ScalarFunction.NEG: "neg",
}

UNARY_SYMBOLS_INT32 = {
    ScalarFunction.ABS: "abs",
    ScalarFunction.POSITIVE: "identity",
    ScalarFunction.NEG: "neg",
}

UNARY_SYMBOLS_INT16 = {
    ScalarFunction.ABS: "abs",
    ScalarFunction.POSITIVE: "identity",
    ScalarFunction.NEG: "neg",
}

UNARY_SYMBOLS_INT8 = {
    ScalarFunction.ABS: "abs",
    ScalarFunction.POSITIVE: "identity",
    ScalarFunction.NEG: "neg",
}

UNARY_SYMBOLS_DOUBLE = {
    ScalarFunction.ABS: "fabs",
    ScalarFunction.CEIL: "ceil",
    ScalarFunction.COS: "cos",
    ScalarFunction.EXP: "exp",
    ScalarFunction.FLOOR: "floor",
    ScalarFunction.POSITIVE: "identity",
    ScalarFunction.LOG: "log",
    ScalarFunction.NEG: "neg",
    ScalarFunction.RELU: "relu",
    ScalarFunction.SIN: "sin",
    ScalarFunction.SQRT: "sqrt",
    ScalarFunction.TAN: "tan",
    ScalarFunction.TANH: "tanh",
    ScalarFunction.ATANH: "atanh",
}

UNARY_SYMBOLS_FLOAT = {
    ScalarFunction.ABS: "fabsf",
    ScalarFunction.CEIL: "ceilf",
    ScalarFunction.COS: "cosf",
    ScalarFunction.EXP: "expf",
    ScalarFunction.FLOOR: "floorf",
    ScalarFunction.POSITIVE: "identity",
    ScalarFunction.LOG: "logf",
    ScalarFunction.NEG: "neg",
    ScalarFunction.RELU: "relu",
    ScalarFunction.SIN: "sinf",
    ScalarFunction.SQRT: "sqrtf",
    ScalarFunction.TAN: "tanf",
    ScalarFunction.TANH: "tanhf",
    ScalarFunction.ATANH: "atanhf",
}

BINARY_SPECS_BOOL = {
    ScalarFunction.LOGICAL_AND: BinaryOpSpec(
        "&&", OperatorKind.INFIX, lambda left, right: np.logical_and(left, right)
    ),
    ScalarFunction.LOGICAL_OR: BinaryOpSpec(
        "||", OperatorKind.INFIX, lambda left, right: np.logical_or(left, right)
    ),
    ScalarFunction.LOGICAL_XOR: BinaryOpSpec(
        "!=", OperatorKind.INFIX, lambda left, right: np.logical_xor(left, right)
    ),
}

COMPARE_SPECS = {
    ScalarFunction.EQ: BinaryOpSpec("==", OperatorKind.INFIX, np.equal),
    ScalarFunction.GT: BinaryOpSpec(">", OperatorKind.INFIX, np.greater),
    ScalarFunction.GE: BinaryOpSpec(">=", OperatorKind.INFIX, np.greater_equal),
    ScalarFunction.LT: BinaryOpSpec("<", OperatorKind.INFIX, np.less),
    ScalarFunction.LE: BinaryOpSpec("<=", OperatorKind.INFIX, np.less_equal),
}

BINARY_SPECS_INT = {
    ScalarFunction.ADD: BinaryOpSpec(
        "+", OperatorKind.INFIX, lambda left, right: left + right
    ),
    ScalarFunction.SUB: BinaryOpSpec(
        "-", OperatorKind.INFIX, lambda left, right: left - right
    ),
    ScalarFunction.MUL: BinaryOpSpec(
        "*", OperatorKind.INFIX, lambda left, right: left * right
    ),
}


def _mean_binary_spec(dtype: ScalarType) -> BinaryOpSpec:
    return BinaryOpSpec(
        f"({{left}} + {{right}}) * {_format_float_literal(0.5, dtype)}",
        OperatorKind.EXPR,
        lambda left, right: (left + right) * 0.5,
    )


def _prelu_binary_spec(dtype: ScalarType) -> BinaryOpSpec:
    zero_literal = _format_float_literal(0.0, dtype)
    return BinaryOpSpec(
        f"({{left}} > {zero_literal} ? {{left}} : {{right}} * {{left}})",
        OperatorKind.EXPR,
        lambda left, right: np.where(left > 0.0, left, right * left),
    )


BINARY_SPECS_DOUBLE = {
    ScalarFunction.ADD: BinaryOpSpec(
        "+", OperatorKind.INFIX, lambda left, right: left + right
    ),
    ScalarFunction.DIV: BinaryOpSpec(
        "/", OperatorKind.INFIX, lambda left, right: left / right
    ),
    ScalarFunction.MAXIMUM: BinaryOpSpec("fmax", OperatorKind.FUNC, np.maximum),
    ScalarFunction.MEAN: _mean_binary_spec(ScalarType.F64),
    ScalarFunction.MINIMUM: BinaryOpSpec("fmin", OperatorKind.FUNC, np.minimum),
    ScalarFunction.MUL: BinaryOpSpec(
        "*", OperatorKind.INFIX, lambda left, right: left * right
    ),
    ScalarFunction.POW: BinaryOpSpec("pow", OperatorKind.FUNC, np.power),
    ScalarFunction.PRELU: _prelu_binary_spec(ScalarType.F64),
    ScalarFunction.SUB: BinaryOpSpec(
        "-", OperatorKind.INFIX, lambda left, right: left - right
    ),
}

BINARY_SPECS_FLOAT = {
    ScalarFunction.ADD: BinaryOpSpec(
        "+", OperatorKind.INFIX, lambda left, right: left + right
    ),
    ScalarFunction.DIV: BinaryOpSpec(
        "/", OperatorKind.INFIX, lambda left, right: left / right
    ),
    ScalarFunction.MAXIMUM: BinaryOpSpec("fmaxf", OperatorKind.FUNC, np.maximum),
    ScalarFunction.MEAN: _mean_binary_spec(ScalarType.F32),
    ScalarFunction.MINIMUM: BinaryOpSpec("fminf", OperatorKind.FUNC, np.minimum),
    ScalarFunction.MUL: BinaryOpSpec(
        "*", OperatorKind.INFIX, lambda left, right: left * right
    ),
    ScalarFunction.POW: BinaryOpSpec("powf", OperatorKind.FUNC, np.power),
    ScalarFunction.PRELU: _prelu_binary_spec(ScalarType.F32),
    ScalarFunction.SUB: BinaryOpSpec(
        "-", OperatorKind.INFIX, lambda left, right: left - right
    ),
}

UNARY_SYMBOLS_BY_DTYPE = {
    ScalarType.BOOL: UNARY_SYMBOLS_BOOL,
    ScalarType.I64: UNARY_SYMBOLS_INT64,
    ScalarType.I32: UNARY_SYMBOLS_INT32,
    ScalarType.I16: UNARY_SYMBOLS_INT16,
    ScalarType.I8: UNARY_SYMBOLS_INT8,
    ScalarType.F64: UNARY_SYMBOLS_DOUBLE,
    ScalarType.F32: UNARY_SYMBOLS_FLOAT,
    ScalarType.F16: UNARY_SYMBOLS_FLOAT,
}

BINARY_SPECS_BY_DTYPE = {
    ScalarType.BOOL: BINARY_SPECS_BOOL,
    ScalarType.I64: BINARY_SPECS_INT,
    ScalarType.I32: BINARY_SPECS_INT,
    ScalarType.I16: BINARY_SPECS_INT,
    ScalarType.I8: BINARY_SPECS_INT,
    ScalarType.U64: BINARY_SPECS_INT,
    ScalarType.U32: BINARY_SPECS_INT,
    ScalarType.U16: BINARY_SPECS_INT,
    ScalarType.U8: BINARY_SPECS_INT,
    ScalarType.F64: BINARY_SPECS_DOUBLE,
    ScalarType.F32: BINARY_SPECS_FLOAT,
    ScalarType.F16: BINARY_SPECS_FLOAT,
}

UNARY_APPLY_FUNCS = {
    "fabsf": np.abs,
    "fabs": np.abs,
    "abs": np.abs,
    "llabs": np.abs,
    "!": np.logical_not,
    "identity": lambda value: value,
    "ceilf": np.ceil,
    "ceil": np.ceil,
    "cosf": np.cos,
    "cos": np.cos,
    "expf": np.exp,
    "exp": np.exp,
    "floorf": np.floor,
    "floor": np.floor,
    "logf": np.log,
    "log": np.log,
    "neg": lambda value: -value,
    "relu": lambda value: np.maximum(value, 0),
    "sinf": np.sin,
    "sin": np.sin,
    "sqrtf": np.sqrt,
    "sqrt": np.sqrt,
    "tanf": np.tan,
    "tan": np.tan,
    "tanhf": np.tanh,
    "tanh": np.tanh,
    "atanhf": np.arctanh,
    "atanh": np.arctanh,
}

COMPARE_FUNCTIONS = {
    ScalarFunction.EQ,
    ScalarFunction.GT,
    ScalarFunction.GE,
    ScalarFunction.LT,
    ScalarFunction.LE,
}


def binary_op_symbol(
    function: ScalarFunction,
    attrs: Mapping[str, object] | None = None,
    *,
    dtype: ScalarType,
    validate_attrs: bool = True,
) -> BinaryOpSpec | None:
    compare_spec = COMPARE_SPECS.get(function)
    if compare_spec is not None:
        return compare_spec
    specs = BINARY_SPECS_BY_DTYPE.get(dtype)
    if specs is not None:
        op_spec = specs.get(function)
        if op_spec is not None:
            return op_spec
    if not dtype.is_float:
        return None
    if function == ScalarFunction.FMOD:
        fmod = 0
        if attrs is not None:
            fmod = int(attrs.get("fmod", 0))
        if validate_attrs and fmod != 1:
            raise UnsupportedOpError(
                "Mod only supports fmod=1 for floating point types"
            )
        func = (
            "fmodf" if dtype in {ScalarType.F16, ScalarType.F32} else "fmod"
        )
        return BinaryOpSpec(func, OperatorKind.FUNC, np.fmod)
    return None


def unary_op_symbol(function: ScalarFunction, *, dtype: ScalarType) -> str | None:
    return UNARY_SYMBOLS_BY_DTYPE.get(dtype, {}).get(function)


def apply_binary_op(
    op_spec: BinaryOpSpec, left: np.ndarray, right: np.ndarray
) -> np.ndarray:
    return op_spec.apply(left, right)


def apply_unary_op(
    function: ScalarFunction, value: np.ndarray, *, dtype: ScalarType
) -> np.ndarray:
    op_symbol = unary_op_symbol(function, dtype=dtype)
    if op_symbol is None:
        raise UnsupportedOpError(f"Unsupported unary op {function.value}")
    func = UNARY_APPLY_FUNCS.get(op_symbol)
    if func is not None:
        return func(value)
    raise UnsupportedOpError(f"Unsupported unary op {op_symbol}")

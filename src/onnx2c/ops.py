from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

from shared.scalar_types import ScalarType

from .errors import UnsupportedOpError


@dataclass(frozen=True)
class BinaryOpSpec:
    operator: str
    kind: str
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
    "Identity": "identity",
    "Not": "!",
}

UNARY_SYMBOLS_INT64 = {
    "Abs": "llabs",
    "Identity": "identity",
    "Neg": "neg",
}

UNARY_SYMBOLS_INT32 = {
    "Abs": "abs",
    "Identity": "identity",
    "Neg": "neg",
}

UNARY_SYMBOLS_INT16 = {
    "Abs": "abs",
    "Identity": "identity",
    "Neg": "neg",
}

UNARY_SYMBOLS_INT8 = {
    "Abs": "abs",
    "Identity": "identity",
    "Neg": "neg",
}

UNARY_SYMBOLS_DOUBLE = {
    "Abs": "fabs",
    "Ceil": "ceil",
    "Cos": "cos",
    "Exp": "exp",
    "Floor": "floor",
    "Identity": "identity",
    "Log": "log",
    "Neg": "neg",
    "Relu": "relu",
    "Sin": "sin",
    "Sqrt": "sqrt",
    "Tan": "tan",
    "Tanh": "tanh",
    "Atanh": "atanh",
}

UNARY_SYMBOLS_FLOAT = {
    "Abs": "fabsf",
    "Ceil": "ceilf",
    "Cos": "cosf",
    "Exp": "expf",
    "Floor": "floorf",
    "Identity": "identity",
    "Log": "logf",
    "Neg": "neg",
    "Relu": "relu",
    "Sin": "sinf",
    "Sqrt": "sqrtf",
    "Tan": "tanf",
    "Tanh": "tanhf",
    "Atanh": "atanhf",
}

BINARY_SPECS_BOOL = {
    "And": BinaryOpSpec("&&", "infix", lambda left, right: np.logical_and(left, right)),
    "Or": BinaryOpSpec("||", "infix", lambda left, right: np.logical_or(left, right)),
    "Xor": BinaryOpSpec("!=", "infix", lambda left, right: np.logical_xor(left, right)),
}

COMPARE_SPECS = {
    "Equal": BinaryOpSpec("==", "infix", np.equal),
    "Greater": BinaryOpSpec(">", "infix", np.greater),
    "GreaterOrEqual": BinaryOpSpec(">=", "infix", np.greater_equal),
    "Less": BinaryOpSpec("<", "infix", np.less),
    "LessOrEqual": BinaryOpSpec("<=", "infix", np.less_equal),
}

BINARY_SPECS_INT = {
    "Add": BinaryOpSpec("+", "infix", lambda left, right: left + right),
    "Sub": BinaryOpSpec("-", "infix", lambda left, right: left - right),
    "Mul": BinaryOpSpec("*", "infix", lambda left, right: left * right),
    "Sum": BinaryOpSpec("+", "infix", lambda left, right: left + right),
}

BINARY_SPECS_DOUBLE = {
    "Add": BinaryOpSpec("+", "infix", lambda left, right: left + right),
    "Div": BinaryOpSpec("/", "infix", lambda left, right: left / right),
    "Max": BinaryOpSpec("fmax", "func", np.maximum),
    "Mean": BinaryOpSpec(
        f"({{left}} + {{right}}) * {_format_float_literal(0.5, ScalarType.F64)}",
        "expr",
        lambda left, right: (left + right) * 0.5,
    ),
    "Min": BinaryOpSpec("fmin", "func", np.minimum),
    "Mul": BinaryOpSpec("*", "infix", lambda left, right: left * right),
    "Pow": BinaryOpSpec("pow", "func", np.power),
    "Sub": BinaryOpSpec("-", "infix", lambda left, right: left - right),
    "Sum": BinaryOpSpec("+", "infix", lambda left, right: left + right),
}

BINARY_SPECS_FLOAT = {
    "Add": BinaryOpSpec("+", "infix", lambda left, right: left + right),
    "Div": BinaryOpSpec("/", "infix", lambda left, right: left / right),
    "Max": BinaryOpSpec("fmaxf", "func", np.maximum),
    "Mean": BinaryOpSpec(
        f"({{left}} + {{right}}) * {_format_float_literal(0.5, ScalarType.F32)}",
        "expr",
        lambda left, right: (left + right) * 0.5,
    ),
    "Min": BinaryOpSpec("fminf", "func", np.minimum),
    "Mul": BinaryOpSpec("*", "infix", lambda left, right: left * right),
    "Pow": BinaryOpSpec("powf", "func", np.power),
    "Sub": BinaryOpSpec("-", "infix", lambda left, right: left - right),
    "Sum": BinaryOpSpec("+", "infix", lambda left, right: left + right),
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

def binary_op_symbol(
    op_type: str, attrs: Mapping[str, object] | None = None, *, dtype: ScalarType
) -> BinaryOpSpec | None:
    compare_spec = COMPARE_SPECS.get(op_type)
    if compare_spec is not None:
        return compare_spec
    specs = BINARY_SPECS_BY_DTYPE.get(dtype)
    if specs is not None:
        op_spec = specs.get(op_type)
        if op_spec is not None:
            return op_spec
    if not dtype.is_float:
        return None
    if op_type == "Mod":
        fmod = 0
        if attrs is not None:
            fmod = int(attrs.get("fmod", 0))
        if fmod != 1:
            raise UnsupportedOpError(
                "Mod only supports fmod=1 for floating point types"
            )
        func = (
            "fmodf" if dtype in {ScalarType.F16, ScalarType.F32} else "fmod"
        )
        return BinaryOpSpec(func, "func", np.fmod)
    if op_type == "PRelu":
        zero_literal = _format_float_literal(0.0, dtype)
        return BinaryOpSpec(
            f"({{left}} > {zero_literal} ? {{left}} : {{right}} * {{left}})",
            "expr",
            lambda left, right: np.where(left > 0.0, left, right * left),
        )
    return None


def unary_op_symbol(op_type: str, *, dtype: ScalarType) -> str | None:
    return UNARY_SYMBOLS_BY_DTYPE.get(dtype, {}).get(op_type)


def apply_binary_op(
    op_spec: BinaryOpSpec, left: np.ndarray, right: np.ndarray
) -> np.ndarray:
    return op_spec.apply(left, right)


def apply_unary_op(op_symbol: str, value: np.ndarray) -> np.ndarray:
    func = UNARY_APPLY_FUNCS.get(op_symbol)
    if func is not None:
        return func(value)
    raise UnsupportedOpError(f"Unsupported unary op {op_symbol}")

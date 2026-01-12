from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

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

UNARY_OP_TYPES = {
    "Abs",
    "Atanh",
    "Ceil",
    "Cos",
    "Exp",
    "Floor",
    "Log",
    "Neg",
    "Not",
    "Relu",
    "Sin",
    "Sqrt",
    "Tan",
    "Tanh",
}


def _format_float_literal(value: float, dtype: str) -> str:
    formatted = f"{value:.9g}"
    if "e" not in formatted and "E" not in formatted and "." not in formatted:
        formatted = f"{formatted}.0"
    if dtype == "float":
        return f"{formatted}f"
    return formatted


UNARY_SYMBOLS_BOOL = {
    "Not": "!",
}

UNARY_SYMBOLS_INT64 = {
    "Abs": "llabs",
    "Neg": "neg",
}

UNARY_SYMBOLS_INT32 = {
    "Abs": "abs",
    "Neg": "neg",
}

UNARY_SYMBOLS_INT16 = {
    "Abs": "abs",
    "Neg": "neg",
}

UNARY_SYMBOLS_INT8 = {
    "Abs": "abs",
    "Neg": "neg",
}

UNARY_SYMBOLS_DOUBLE = {
    "Abs": "fabs",
    "Ceil": "ceil",
    "Cos": "cos",
    "Exp": "exp",
    "Floor": "floor",
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
        f"({{left}} + {{right}}) * {_format_float_literal(0.5, 'double')}",
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
        f"({{left}} + {{right}}) * {_format_float_literal(0.5, 'float')}",
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
    "bool": UNARY_SYMBOLS_BOOL,
    "int64": UNARY_SYMBOLS_INT64,
    "int32": UNARY_SYMBOLS_INT32,
    "int16": UNARY_SYMBOLS_INT16,
    "int8": UNARY_SYMBOLS_INT8,
    "double": UNARY_SYMBOLS_DOUBLE,
    "float": UNARY_SYMBOLS_FLOAT,
}

BINARY_SPECS_BY_DTYPE = {
    "bool": BINARY_SPECS_BOOL,
    "int64": BINARY_SPECS_INT,
    "int32": BINARY_SPECS_INT,
    "int16": BINARY_SPECS_INT,
    "int8": BINARY_SPECS_INT,
    "uint64": BINARY_SPECS_INT,
    "uint32": BINARY_SPECS_INT,
    "uint16": BINARY_SPECS_INT,
    "uint8": BINARY_SPECS_INT,
    "double": BINARY_SPECS_DOUBLE,
    "float": BINARY_SPECS_FLOAT,
}

UNARY_APPLY_FUNCS = {
    "fabsf": np.abs,
    "fabs": np.abs,
    "abs": np.abs,
    "llabs": np.abs,
    "!": np.logical_not,
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
    op_type: str, attrs: Mapping[str, object] | None = None, *, dtype: str
) -> BinaryOpSpec | None:
    specs = BINARY_SPECS_BY_DTYPE.get(dtype)
    if specs is not None:
        op_spec = specs.get(op_type)
        if op_spec is not None:
            return op_spec
    if dtype not in {"float", "double"}:
        return None
    if op_type == "Mod":
        fmod = 0
        if attrs is not None:
            fmod = int(attrs.get("fmod", 0))
        if fmod != 1:
            raise UnsupportedOpError(
                "Mod only supports fmod=1 for floating point types"
            )
        func = "fmodf" if dtype == "float" else "fmod"
        return BinaryOpSpec(func, "func", np.fmod)
    if op_type == "PRelu":
        zero_literal = _format_float_literal(0.0, dtype)
        return BinaryOpSpec(
            f"({{left}} > {zero_literal} ? {{left}} : {{right}} * {{left}})",
            "expr",
            lambda left, right: np.where(left > 0.0, left, right * left),
        )
    return None


def unary_op_symbol(op_type: str, *, dtype: str) -> str | None:
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

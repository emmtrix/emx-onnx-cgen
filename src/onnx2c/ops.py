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


def binary_op_symbol(
    op_type: str, attrs: Mapping[str, object] | None = None, *, dtype: str
) -> BinaryOpSpec | None:
    if dtype == "bool":
        if op_type == "And":
            return BinaryOpSpec(
                "&&", "infix", lambda left, right: np.logical_and(left, right)
            )
        if op_type == "Or":
            return BinaryOpSpec(
                "||", "infix", lambda left, right: np.logical_or(left, right)
            )
        if op_type == "Xor":
            return BinaryOpSpec(
                "!=", "infix", lambda left, right: np.logical_xor(left, right)
            )
        return None
    if dtype in {
        "int64",
        "int32",
        "int16",
        "int8",
        "uint64",
        "uint32",
        "uint16",
        "uint8",
    }:
        if op_type in {"Add", "Sum"}:
            return BinaryOpSpec("+", "infix", lambda left, right: left + right)
        if op_type == "Sub":
            return BinaryOpSpec("-", "infix", lambda left, right: left - right)
        if op_type == "Mul":
            return BinaryOpSpec("*", "infix", lambda left, right: left * right)
        return None
    if op_type == "Add":
        return BinaryOpSpec("+", "infix", lambda left, right: left + right)
    if op_type == "Div":
        return BinaryOpSpec("/", "infix", lambda left, right: left / right)
    if op_type == "Max":
        func = "fmaxf" if dtype == "float" else "fmax"
        return BinaryOpSpec(func, "func", np.maximum)
    if op_type == "Mean":
        mean_literal = _format_float_literal(0.5, dtype)
        return BinaryOpSpec(
            f"({{left}} + {{right}}) * {mean_literal}",
            "expr",
            lambda left, right: (left + right) * 0.5,
        )
    if op_type == "Min":
        func = "fminf" if dtype == "float" else "fmin"
        return BinaryOpSpec(func, "func", np.minimum)
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
    if op_type == "Mul":
        return BinaryOpSpec("*", "infix", lambda left, right: left * right)
    if op_type == "Pow":
        func = "powf" if dtype == "float" else "pow"
        return BinaryOpSpec(func, "func", np.power)
    if op_type == "PRelu":
        zero_literal = _format_float_literal(0.0, dtype)
        return BinaryOpSpec(
            f"({{left}} > {zero_literal} ? {{left}} : {{right}} * {{left}})",
            "expr",
            lambda left, right: np.where(left > 0.0, left, right * left),
        )
    if op_type == "Sub":
        return BinaryOpSpec("-", "infix", lambda left, right: left - right)
    if op_type == "Sum":
        return BinaryOpSpec("+", "infix", lambda left, right: left + right)
    return None


def unary_op_symbol(op_type: str, *, dtype: str) -> str | None:
    if dtype == "bool":
        if op_type == "Not":
            return "!"
        return None
    if dtype in {"int64", "int32", "int16", "int8"}:
        if op_type == "Abs":
            return "llabs" if dtype == "int64" else "abs"
        if op_type == "Neg":
            return "neg"
        return None
    if dtype == "double":
        if op_type == "Abs":
            return "fabs"
        if op_type == "Ceil":
            return "ceil"
        if op_type == "Cos":
            return "cos"
        if op_type == "Exp":
            return "exp"
        if op_type == "Floor":
            return "floor"
        if op_type == "Log":
            return "log"
        if op_type == "Neg":
            return "neg"
        if op_type == "Relu":
            return "relu"
        if op_type == "Sin":
            return "sin"
        if op_type == "Sqrt":
            return "sqrt"
        if op_type == "Tan":
            return "tan"
        if op_type == "Tanh":
            return "tanh"
        if op_type == "Atanh":
            return "atanh"
        return None
    if op_type == "Abs":
        return "fabsf"
    if op_type == "Ceil":
        return "ceilf"
    if op_type == "Cos":
        return "cosf"
    if op_type == "Exp":
        return "expf"
    if op_type == "Floor":
        return "floorf"
    if op_type == "Log":
        return "logf"
    if op_type == "Neg":
        return "neg"
    if op_type == "Relu":
        return "relu"
    if op_type == "Sin":
        return "sinf"
    if op_type == "Sqrt":
        return "sqrtf"
    if op_type == "Tan":
        return "tanf"
    if op_type == "Tanh":
        return "tanhf"
    if op_type == "Atanh":
        return "atanhf"
    return None


def apply_binary_op(
    op_spec: BinaryOpSpec, left: np.ndarray, right: np.ndarray
) -> np.ndarray:
    return op_spec.apply(left, right)


def apply_unary_op(op_symbol: str, value: np.ndarray) -> np.ndarray:
    if op_symbol in {"fabsf", "fabs"}:
        return np.abs(value)
    if op_symbol == "abs":
        return np.abs(value)
    if op_symbol == "llabs":
        return np.abs(value)
    if op_symbol == "!":
        return np.logical_not(value)
    if op_symbol in {"ceilf", "ceil"}:
        return np.ceil(value)
    if op_symbol in {"cosf", "cos"}:
        return np.cos(value)
    if op_symbol in {"expf", "exp"}:
        return np.exp(value)
    if op_symbol in {"floorf", "floor"}:
        return np.floor(value)
    if op_symbol in {"logf", "log"}:
        return np.log(value)
    if op_symbol == "neg":
        return -value
    if op_symbol == "relu":
        return np.maximum(value, 0)
    if op_symbol in {"sinf", "sin"}:
        return np.sin(value)
    if op_symbol in {"sqrtf", "sqrt"}:
        return np.sqrt(value)
    if op_symbol in {"tanf", "tan"}:
        return np.tan(value)
    if op_symbol in {"tanhf", "tanh"}:
        return np.tanh(value)
    if op_symbol in {"atanhf", "atanh"}:
        return np.arctanh(value)
    raise UnsupportedOpError(f"Unsupported unary op {op_symbol}")

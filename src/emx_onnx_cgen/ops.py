from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
import math

import numpy as np

from shared.scalar_functions import ScalarFunction
from shared.scalar_types import ScalarType

from .errors import UnsupportedOpError


_NP_ERF = getattr(np, "erf", None)
if _NP_ERF is None:
    _NP_ERF = np.vectorize(math.erf, otypes=[float])


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
    "BitShift",
    "BitwiseAnd",
    "BitwiseOr",
    "BitwiseXor",
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
    "Acos",
    "Acosh",
    "Asin",
    "Asinh",
    "Atan",
    "Atanh",
    "BitwiseNot",
    "Ceil",
    "Cos",
    "Cosh",
    "Elu",
    "Erf",
    "Exp",
    "Floor",
    "Gelu",
    "HardSigmoid",
    "HardSwish",
    "Identity",
    "LeakyRelu",
    "Log",
    "Mish",
    "Neg",
    "Not",
    "Reciprocal",
    "Relu",
    "Round",
    "Selu",
    "Sigmoid",
    "Sign",
    "Sin",
    "Sinh",
    "Softplus",
    "Softsign",
    "Sqrt",
    "Tan",
    "Tanh",
    "ThresholdedRelu",
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
    ScalarFunction.BITWISE_NOT: "bitwise_not",
}

UNARY_SYMBOLS_INT64 = {
    ScalarFunction.ABS: "llabs",
    ScalarFunction.BITWISE_NOT: "bitwise_not",
    ScalarFunction.POSITIVE: "identity",
    ScalarFunction.NEG: "neg",
    ScalarFunction.ROUND: "round",
    ScalarFunction.SIGN: "sign",
}

UNARY_SYMBOLS_INT32 = {
    ScalarFunction.ABS: "abs",
    ScalarFunction.BITWISE_NOT: "bitwise_not",
    ScalarFunction.POSITIVE: "identity",
    ScalarFunction.NEG: "neg",
    ScalarFunction.ROUND: "round",
    ScalarFunction.SIGN: "sign",
}

UNARY_SYMBOLS_INT16 = {
    ScalarFunction.ABS: "abs",
    ScalarFunction.BITWISE_NOT: "bitwise_not",
    ScalarFunction.POSITIVE: "identity",
    ScalarFunction.NEG: "neg",
    ScalarFunction.ROUND: "round",
    ScalarFunction.SIGN: "sign",
}

UNARY_SYMBOLS_INT8 = {
    ScalarFunction.ABS: "abs",
    ScalarFunction.BITWISE_NOT: "bitwise_not",
    ScalarFunction.POSITIVE: "identity",
    ScalarFunction.NEG: "neg",
    ScalarFunction.ROUND: "round",
    ScalarFunction.SIGN: "sign",
}

UNARY_SYMBOLS_DOUBLE = {
    ScalarFunction.ABS: "fabs",
    ScalarFunction.ACOS: "acos",
    ScalarFunction.ACOSH: "acosh",
    ScalarFunction.ASIN: "asin",
    ScalarFunction.ASINH: "asinh",
    ScalarFunction.ATAN: "atan",
    ScalarFunction.CEIL: "ceil",
    ScalarFunction.COS: "cos",
    ScalarFunction.COSH: "cosh",
    ScalarFunction.ELU: "elu",
    ScalarFunction.ERF: "erf",
    ScalarFunction.EXP: "exp",
    ScalarFunction.FLOOR: "floor",
    ScalarFunction.GELU: "gelu",
    ScalarFunction.HARDSIGMOID: "hardsigmoid",
    ScalarFunction.HARDSWISH: "hardswish",
    ScalarFunction.LEAKY_RELU: "leaky_relu",
    ScalarFunction.POSITIVE: "identity",
    ScalarFunction.LOG: "log",
    ScalarFunction.MISH: "mish",
    ScalarFunction.NEG: "neg",
    ScalarFunction.RECIPROCAL: "reciprocal",
    ScalarFunction.RELU: "relu",
    ScalarFunction.ROUND: "round",
    ScalarFunction.SELU: "selu",
    ScalarFunction.SIGMOID: "sigmoid",
    ScalarFunction.SIGN: "sign",
    ScalarFunction.SIN: "sin",
    ScalarFunction.SINH: "sinh",
    ScalarFunction.SOFTPLUS: "softplus",
    ScalarFunction.SOFTSIGN: "softsign",
    ScalarFunction.SQRT: "sqrt",
    ScalarFunction.TAN: "tan",
    ScalarFunction.TANH: "tanh",
    ScalarFunction.THRESHOLDED_RELU: "thresholded_relu",
    ScalarFunction.ATANH: "atanh",
}

UNARY_SYMBOLS_FLOAT = {
    ScalarFunction.ABS: "fabsf",
    ScalarFunction.ACOS: "acosf",
    ScalarFunction.ACOSH: "acoshf",
    ScalarFunction.ASIN: "asinf",
    ScalarFunction.ASINH: "asinhf",
    ScalarFunction.ATAN: "atanf",
    ScalarFunction.CEIL: "ceilf",
    ScalarFunction.COS: "cosf",
    ScalarFunction.COSH: "coshf",
    ScalarFunction.ELU: "elu",
    ScalarFunction.ERF: "erff",
    ScalarFunction.EXP: "expf",
    ScalarFunction.FLOOR: "floorf",
    ScalarFunction.GELU: "gelu",
    ScalarFunction.HARDSIGMOID: "hardsigmoid",
    ScalarFunction.HARDSWISH: "hardswish",
    ScalarFunction.LEAKY_RELU: "leaky_relu",
    ScalarFunction.POSITIVE: "identity",
    ScalarFunction.LOG: "logf",
    ScalarFunction.MISH: "mish",
    ScalarFunction.NEG: "neg",
    ScalarFunction.RECIPROCAL: "reciprocal",
    ScalarFunction.RELU: "relu",
    ScalarFunction.ROUND: "round",
    ScalarFunction.SELU: "selu",
    ScalarFunction.SIGMOID: "sigmoid",
    ScalarFunction.SIGN: "sign",
    ScalarFunction.SIN: "sinf",
    ScalarFunction.SINH: "sinhf",
    ScalarFunction.SOFTPLUS: "softplus",
    ScalarFunction.SOFTSIGN: "softsign",
    ScalarFunction.SQRT: "sqrtf",
    ScalarFunction.TAN: "tanf",
    ScalarFunction.TANH: "tanhf",
    ScalarFunction.THRESHOLDED_RELU: "thresholded_relu",
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
    ScalarFunction.BITWISE_AND: BinaryOpSpec(
        "&", OperatorKind.INFIX, lambda left, right: left & right
    ),
    ScalarFunction.BITWISE_OR: BinaryOpSpec(
        "|", OperatorKind.INFIX, lambda left, right: left | right
    ),
    ScalarFunction.BITWISE_XOR: BinaryOpSpec(
        "^", OperatorKind.INFIX, lambda left, right: left ^ right
    ),
    ScalarFunction.BITWISE_LEFT_SHIFT: BinaryOpSpec(
        "<<", OperatorKind.INFIX, np.left_shift
    ),
    ScalarFunction.BITWISE_RIGHT_SHIFT: BinaryOpSpec(
        ">>", OperatorKind.INFIX, np.right_shift
    ),
    ScalarFunction.DIV: BinaryOpSpec(
        "/", OperatorKind.INFIX, lambda left, right: left // right
    ),
    ScalarFunction.FMOD: BinaryOpSpec(
        "%", OperatorKind.INFIX, np.fmod
    ),
    ScalarFunction.REMAINDER: BinaryOpSpec(
        "remainder", OperatorKind.FUNC, np.mod
    ),
    ScalarFunction.MAXIMUM: BinaryOpSpec(
        "maximum", OperatorKind.FUNC, np.maximum
    ),
    ScalarFunction.MINIMUM: BinaryOpSpec(
        "minimum", OperatorKind.FUNC, np.minimum
    ),
    ScalarFunction.POW: BinaryOpSpec("pow", OperatorKind.FUNC, np.power),
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
    ScalarFunction.REMAINDER: BinaryOpSpec(
        "remainder", OperatorKind.FUNC, np.remainder
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
    ScalarFunction.REMAINDER: BinaryOpSpec(
        "remainder", OperatorKind.FUNC, np.remainder
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
    "acosf": np.arccos,
    "acos": np.arccos,
    "acoshf": np.arccosh,
    "acosh": np.arccosh,
    "fabsf": np.abs,
    "fabs": np.abs,
    "abs": np.abs,
    "llabs": np.abs,
    "asinf": np.arcsin,
    "asin": np.arcsin,
    "asinhf": np.arcsinh,
    "asinh": np.arcsinh,
    "atanf": np.arctan,
    "atan": np.arctan,
    "bitwise_not": np.bitwise_not,
    "!": np.logical_not,
    "identity": lambda value: value,
    "ceilf": np.ceil,
    "ceil": np.ceil,
    "cosf": np.cos,
    "cos": np.cos,
    "coshf": np.cosh,
    "cosh": np.cosh,
    "elu": lambda value: np.where(value > 0.0, value, np.exp(value) - 1.0),
    "erff": _NP_ERF,
    "erf": _NP_ERF,
    "expf": np.exp,
    "exp": np.exp,
    "floorf": np.floor,
    "floor": np.floor,
    "gelu": lambda value: 0.5
    * value
    * (1.0 + _NP_ERF(value / np.sqrt(2.0))),
    "hardsigmoid": lambda value: np.clip(value * 0.2 + 0.5, 0.0, 1.0),
    "hardswish": lambda value: value
    * np.clip(value + 3.0, 0.0, 6.0)
    / 6.0,
    "leaky_relu": lambda value: np.where(value > 0.0, value, 0.01 * value),
    "logf": np.log,
    "log": np.log,
    "neg": lambda value: -value,
    "reciprocal": lambda value: 1.0 / value,
    "relu": lambda value: np.maximum(value, 0),
    "round": np.round,
    "selu": lambda value: np.where(
        value > 0.0,
        1.0507009873554805 * value,
        1.0507009873554805
        * 1.6732632423543772
        * (np.exp(value) - 1.0),
    ),
    "sigmoid": lambda value: 1.0 / (1.0 + np.exp(-value)),
    "sign": np.sign,
    "sinf": np.sin,
    "sin": np.sin,
    "sqrtf": np.sqrt,
    "sqrt": np.sqrt,
    "softplus": lambda value: np.where(
        value > 20.0, value, np.log1p(np.exp(value))
    ),
    "softsign": lambda value: value / (1.0 + np.abs(value)),
    "sinhf": np.sinh,
    "sinh": np.sinh,
    "tanf": np.tan,
    "tan": np.tan,
    "tanhf": np.tanh,
    "tanh": np.tanh,
    "thresholded_relu": lambda value: np.where(
        value > 1.0, value, 0.0
    ),
    "mish": lambda value: value * np.tanh(np.log1p(np.exp(value))),
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

UNARY_ATTR_DEFAULTS: Mapping[str, Mapping[str, object]] = {
    "Elu": {"alpha": 1.0},
    "Gelu": {"approximate": "none"},
    "HardSigmoid": {"alpha": 0.2, "beta": 0.5},
    "LeakyRelu": {"alpha": 0.01},
    "Selu": {"alpha": 1.6732632423543772, "gamma": 1.0507009873554805},
    "Softplus": {"beta": 1.0, "threshold": 20.0},
    "ThresholdedRelu": {"alpha": 1.0},
}


def validate_unary_attrs(op_type: str, attrs: Mapping[str, object]) -> None:
    defaults = UNARY_ATTR_DEFAULTS.get(op_type)
    if defaults is None or not attrs:
        return
    for key in attrs:
        if key not in defaults:
            raise UnsupportedOpError(
                f"{op_type} does not support attribute {key}"
            )
    for key, default in defaults.items():
        if key not in attrs:
            continue
        value = attrs[key]
        if isinstance(default, str):
            if str(value) != default:
                raise UnsupportedOpError(
                    f"{op_type} only supports {key}={default}"
                )
            continue
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:
            raise UnsupportedOpError(
                f"{op_type} only supports {key}={default}"
            ) from exc
        if not math.isclose(numeric_value, float(default), abs_tol=1e-6):
            raise UnsupportedOpError(
                f"{op_type} only supports {key}={default}"
            )


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

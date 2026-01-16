from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Callable, Dict, List, Mapping, Set

from shared.scalar_types import ScalarFunctionError, ScalarType


@dataclass(frozen=True)
class _ScalarTypeInfo:
    scalar_type: ScalarType
    c_type: str
    prefix: str
    suffix: str
    is_float: bool
    is_bool: bool
    is_signed: bool
    is_small_int: bool
    bits: int | None


@dataclass(frozen=True)
class _GeneratedScalar:
    lines: List[str]
    deps: Set[ScalarFunctionKey]
    includes: Set[str]
    constants: Set[str] = field(default_factory=set)


def _scalar_function_spec(
    value: str,
    *,
    supports_float: bool = True,
    supports_signed_int: bool = True,
    supports_unsigned_int: bool = True,
    supports_bool: bool = True,
    int_from_f32_arity: int | None = None,
    bool_from_f32_arity: int | None = None,
) -> tuple[
    str,
    bool,
    bool,
    bool,
    bool,
    int | None,
    int | None,
]:
    return (
        value,
        supports_float,
        supports_signed_int,
        supports_unsigned_int,
        supports_bool,
        int_from_f32_arity,
        bool_from_f32_arity,
    )


def _common_unary_from_f32_spec(value: str) -> tuple[
    str, bool, bool, bool, bool, int | None, int | None
]:
    return _scalar_function_spec(value, int_from_f32_arity=1, bool_from_f32_arity=1)


def _common_binary_from_f32_spec(value: str) -> tuple[
    str, bool, bool, bool, bool, int | None, int | None
]:
    return _scalar_function_spec(value, int_from_f32_arity=2, bool_from_f32_arity=2)


def _bool_unary_from_f32_spec(
    value: str, *, supports_unsigned_int: bool = True
) -> tuple[str, bool, bool, bool, bool, int | None, int | None]:
    return _scalar_function_spec(
        value,
        supports_unsigned_int=supports_unsigned_int,
        bool_from_f32_arity=1,
    )


def _bool_binary_from_f32_spec(value: str) -> tuple[
    str, bool, bool, bool, bool, int | None, int | None
]:
    return _scalar_function_spec(value, bool_from_f32_arity=2)


def _no_float_spec(value: str) -> tuple[str, bool, bool, bool, bool, int | None, int | None]:
    return _scalar_function_spec(value, supports_float=False)


def _int_only_spec(value: str) -> tuple[str, bool, bool, bool, bool, int | None, int | None]:
    return _scalar_function_spec(value, supports_float=False, supports_bool=False)


def _conversion_spec(value: str) -> tuple[str, bool, bool, bool, bool, int | None, int | None]:
    return _scalar_function_spec(
        value,
        supports_float=False,
        supports_signed_int=False,
        supports_unsigned_int=False,
        supports_bool=False,
    )


class ScalarFunction(str, Enum):
    def __new__(
        cls,
        value: str,
        supports_float: bool,
        supports_signed_int: bool,
        supports_unsigned_int: bool,
        supports_bool: bool,
        int_from_f32_arity: int | None = None,
        bool_from_f32_arity: int | None = None,
    ) -> "ScalarFunction":
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.supports_float = supports_float
        obj.supports_signed_int = supports_signed_int
        obj.supports_unsigned_int = supports_unsigned_int
        obj.supports_bool = supports_bool
        obj.int_from_f32_arity = int_from_f32_arity
        obj.bool_from_f32_arity = bool_from_f32_arity
        return obj

    ABS = _bool_unary_from_f32_spec("abs")
    ABSOLUTE = _bool_unary_from_f32_spec("absolute")
    ACOS = _common_unary_from_f32_spec("acos")
    ACOSH = _common_unary_from_f32_spec("acosh")
    ADD = _bool_binary_from_f32_spec("add")
    ANGLE = _common_unary_from_f32_spec("angle")
    ARCCOS = _common_unary_from_f32_spec("arccos")
    ARCSIN = _common_unary_from_f32_spec("arcsin")
    ARCSINH = _common_unary_from_f32_spec("arcsinh")
    ARCTAN = _common_unary_from_f32_spec("arctan")
    ASIN = _common_unary_from_f32_spec("asin")
    ASINH = _common_unary_from_f32_spec("asinh")
    ATAN = _common_unary_from_f32_spec("atan")
    ATAN2 = _common_binary_from_f32_spec("atan2")
    ATANH = _common_unary_from_f32_spec("atanh")
    AFFINE = _common_unary_from_f32_spec("affine")
    BITWISE_AND = _no_float_spec("bitwise_and")
    BITWISE_LEFT_SHIFT = _int_only_spec("bitwise_left_shift")
    BITWISE_NOT = _no_float_spec("bitwise_not")
    BITWISE_OR = _no_float_spec("bitwise_or")
    BITWISE_RIGHT_SHIFT = _int_only_spec("bitwise_right_shift")
    BITWISE_XOR = _no_float_spec("bitwise_xor")
    CBRT = _common_unary_from_f32_spec("cbrt")
    CEIL = _bool_unary_from_f32_spec("ceil")
    CELU = _common_unary_from_f32_spec("celu")
    CLAMP_MAX = _bool_binary_from_f32_spec("clamp_max")
    CLAMP_MIN = _bool_binary_from_f32_spec("clamp_min")
    CONJ = _bool_unary_from_f32_spec("conj", supports_unsigned_int=False)
    CONJ_PHYSICAL = _bool_unary_from_f32_spec("conj_physical", supports_unsigned_int=False)
    COPYSIGN = _bool_binary_from_f32_spec("copysign")
    COS = _common_unary_from_f32_spec("cos")
    COSH = _common_unary_from_f32_spec("cosh")
    DEG2RAD = _common_unary_from_f32_spec("deg2rad")
    DIGAMMA = _common_unary_from_f32_spec("digamma")
    DIV = _bool_binary_from_f32_spec("div")
    ELU = _common_unary_from_f32_spec("elu")
    EQ = _scalar_function_spec("eq")
    ERF = _common_unary_from_f32_spec("erf")
    ERFC = _common_unary_from_f32_spec("erfc")
    ERFINV = _common_unary_from_f32_spec("erfinv")
    EXP = _common_unary_from_f32_spec("exp")
    EXP2 = _common_unary_from_f32_spec("exp2")
    EXPM1 = _common_unary_from_f32_spec("expm1")
    FLOOR = _bool_unary_from_f32_spec("floor")
    FLOOR_DIVIDE = _bool_binary_from_f32_spec("floor_divide")
    FMAX = _bool_binary_from_f32_spec("fmax")
    FMIN = _bool_binary_from_f32_spec("fmin")
    FMOD = _bool_binary_from_f32_spec("fmod")
    FRAC = _bool_unary_from_f32_spec("frac", supports_unsigned_int=False)
    GE = _scalar_function_spec("ge")
    GELU = _common_unary_from_f32_spec("gelu")
    GT = _scalar_function_spec("gt")
    HARDSIGMOID = _common_unary_from_f32_spec("hardsigmoid")
    HARDSWISH = _common_unary_from_f32_spec("hardswish")
    HEAVISIDE = _common_binary_from_f32_spec("heaviside")
    HYPOT = _common_binary_from_f32_spec("hypot")
    I0 = _common_unary_from_f32_spec("i0")
    ISFINITE = _common_unary_from_f32_spec("isfinite")
    ISINF = _common_unary_from_f32_spec("isinf")
    ISNAN = _common_unary_from_f32_spec("isnan")
    ISNEGINF = _common_unary_from_f32_spec("isneginf")
    ISPOSINF = _common_unary_from_f32_spec("isposinf")
    LDEXP = _common_binary_from_f32_spec("ldexp")
    LE = _scalar_function_spec("le")
    LEAKY_RELU = _common_unary_from_f32_spec("leaky_relu")
    LGAMMA = _common_unary_from_f32_spec("lgamma")
    LOG = _common_unary_from_f32_spec("log")
    LOG10 = _common_unary_from_f32_spec("log10")
    LOG1P = _common_unary_from_f32_spec("log1p")
    LOG2 = _common_unary_from_f32_spec("log2")
    LOG_SIGMOID = _common_unary_from_f32_spec("log_sigmoid")
    LOGADDEXP = _common_binary_from_f32_spec("logaddexp")
    LOGADDEXP2 = _common_binary_from_f32_spec("logaddexp2")
    LOGICAL_AND = _scalar_function_spec("logical_and")
    LOGICAL_NOT = _scalar_function_spec("logical_not")
    LOGICAL_OR = _scalar_function_spec("logical_or")
    LOGICAL_XOR = _scalar_function_spec("logical_xor")
    LOGIT = _common_unary_from_f32_spec("logit")
    LT = _scalar_function_spec("lt")
    MAXIMUM = _bool_binary_from_f32_spec("maximum")
    MEAN = _scalar_function_spec(
        "mean",
        supports_signed_int=False,
        supports_unsigned_int=False,
        supports_bool=False,
    )
    MINIMUM = _bool_binary_from_f32_spec("minimum")
    MISH = _common_unary_from_f32_spec("mish")
    MUL = _bool_binary_from_f32_spec("mul")
    NAN_TO_NUM = _common_unary_from_f32_spec("nan_to_num")
    NE = _scalar_function_spec("ne")
    NEG = _bool_unary_from_f32_spec("neg")
    NEXTAFTER = _common_binary_from_f32_spec("nextafter")
    POSITIVE = _bool_unary_from_f32_spec("positive", supports_unsigned_int=False)
    POW = _common_binary_from_f32_spec("pow")
    PRELU = _scalar_function_spec(
        "prelu",
        supports_signed_int=False,
        supports_unsigned_int=False,
        supports_bool=False,
    )
    RAD2DEG = _common_unary_from_f32_spec("rad2deg")
    SCALED_TANH = _common_unary_from_f32_spec("scaled_tanh")
    REAL = _bool_unary_from_f32_spec("real", supports_unsigned_int=False)
    RECIPROCAL = _bool_unary_from_f32_spec("reciprocal")
    RELU = _bool_unary_from_f32_spec("relu")
    RELU6 = _common_unary_from_f32_spec("relu6")
    REMAINDER = _bool_binary_from_f32_spec("remainder")
    ROUND = _bool_unary_from_f32_spec("round")
    RSQRT = _common_unary_from_f32_spec("rsqrt")
    SELU = _common_unary_from_f32_spec("selu")
    SGN = _bool_unary_from_f32_spec("sgn", supports_unsigned_int=False)
    SIGMOID = _common_unary_from_f32_spec("sigmoid")
    SIGN = _bool_unary_from_f32_spec("sign", supports_unsigned_int=False)
    SILU = _common_unary_from_f32_spec("silu")
    SIN = _common_unary_from_f32_spec("sin")
    SINC = _common_unary_from_f32_spec("sinc")
    SINH = _common_unary_from_f32_spec("sinh")
    SOFTPLUS = _common_unary_from_f32_spec("softplus")
    SOFTSIGN = _scalar_function_spec(
        "softsign",
        supports_signed_int=False,
        supports_unsigned_int=False,
        supports_bool=False,
    )
    SQRT = _common_unary_from_f32_spec("sqrt")
    SQUARE = _bool_unary_from_f32_spec("square", supports_unsigned_int=False)
    SHRINK = _common_unary_from_f32_spec("shrink")
    SUB = _bool_binary_from_f32_spec("sub")
    SWISH = _common_unary_from_f32_spec("swish")
    TAN = _common_unary_from_f32_spec("tan")
    TANH = _common_unary_from_f32_spec("tanh")
    THRESHOLDED_RELU = _scalar_function_spec(
        "thresholded_relu",
        supports_signed_int=False,
        supports_unsigned_int=False,
        supports_bool=False,
    )
    TRUNC = _bool_unary_from_f32_spec("trunc", supports_unsigned_int=False)
    XLOGY = _common_binary_from_f32_spec("xlogy")
    CONVERT_FROM_F32 = _conversion_spec("convert_from_f32")
    CONVERT_FROM_F64 = _conversion_spec("convert_from_f64")
    CONVERT_FROM_I8 = _conversion_spec("convert_from_i8")
    CONVERT_FROM_I16 = _conversion_spec("convert_from_i16")
    CONVERT_FROM_I32 = _conversion_spec("convert_from_i32")
    CONVERT_FROM_I64 = _conversion_spec("convert_from_i64")
    CONVERT_FROM_U8 = _conversion_spec("convert_from_u8")
    CONVERT_FROM_U16 = _conversion_spec("convert_from_u16")
    CONVERT_FROM_U32 = _conversion_spec("convert_from_u32")
    CONVERT_FROM_U64 = _conversion_spec("convert_from_u64")
    CONVERT_FROM_BOOL = _conversion_spec("convert_from_bool")

    def supports_dtype(self, dtype_info: _ScalarTypeInfo) -> bool:
        if dtype_info.is_float:
            return self.supports_float
        if dtype_info.is_bool:
            return self.supports_bool
        if dtype_info.is_signed:
            return self.supports_signed_int
        return self.supports_unsigned_int

    @classmethod
    def from_op_name(cls, op_name: str) -> "ScalarFunction":
        try:
            return cls(op_name)
        except ValueError as exc:
            raise ScalarFunctionError(
                f"unknown scalar function op name: {op_name}"
            ) from exc

    @classmethod
    def from_onnx_op(cls, op_type: str) -> "ScalarFunction":
        canonical = _normalize_op_name(op_type)
        if canonical != op_type:
            op_type = canonical
        try:
            return _ONNX_OP_TO_SCALAR_FUNCTION[op_type]
        except KeyError as exc:
            raise ScalarFunctionError(
                f"unsupported ONNX scalar op: {op_type}"
            ) from exc


@dataclass(frozen=True)
class ScalarFunctionKey:
    function: ScalarFunction
    return_type: ScalarType
    params: tuple[float, ...] = ()

    @classmethod
    def for_torch_dtype(
        cls, function: ScalarFunction, dtype: object
    ) -> "ScalarFunctionKey":
        return cls(function=function, return_type=ScalarType.from_torch_dtype(dtype))


def _conversion_key_from_alias(
    dtype_info: _ScalarTypeInfo, alias: str
) -> ScalarFunctionKey:
    if alias == "from_f32":
        return ScalarFunctionKey(
            function=ScalarFunction.CONVERT_FROM_F32,
            return_type=dtype_info.scalar_type,
            params=(),
        )
    if alias == "to_f32":
        return ScalarFunctionKey(
            function=ScalarFunction.CONVERT_FROM_BOOL,
            return_type=ScalarType.F32,
            params=(),
        )
    raise ScalarFunctionError(f"unknown conversion alias: {alias}")


def _scalar_key_from_op(
    dtype_info: _ScalarTypeInfo, op_name: str
) -> ScalarFunctionKey:
    canonical_name = _normalize_op_name(op_name)
    if canonical_name in {"from_f32", "to_f32"}:
        return _conversion_key_from_alias(dtype_info, canonical_name)
    return ScalarFunctionKey(
        function=ScalarFunction.from_op_name(canonical_name),
        return_type=dtype_info.scalar_type,
        params=(),
    )


_OP_ALIASES = {
    "absolute": "abs",
    "arccos": "acos",
    "arcsin": "asin",
    "arcsinh": "asinh",
    "arctan": "atan",
}

_ONNX_OP_TO_SCALAR_FUNCTION = {
    "Abs": ScalarFunction.ABS,
    "Acos": ScalarFunction.ACOS,
    "Acosh": ScalarFunction.ACOSH,
    "Add": ScalarFunction.ADD,
    "And": ScalarFunction.LOGICAL_AND,
    "Asin": ScalarFunction.ASIN,
    "Asinh": ScalarFunction.ASINH,
    "Atan": ScalarFunction.ATAN,
    "Atanh": ScalarFunction.ATANH,
    "BitwiseAnd": ScalarFunction.BITWISE_AND,
    "BitwiseNot": ScalarFunction.BITWISE_NOT,
    "BitwiseOr": ScalarFunction.BITWISE_OR,
    "BitwiseXor": ScalarFunction.BITWISE_XOR,
    "Ceil": ScalarFunction.CEIL,
    "Celu": ScalarFunction.CELU,
    "Cos": ScalarFunction.COS,
    "Cosh": ScalarFunction.COSH,
    "Div": ScalarFunction.DIV,
    "Elu": ScalarFunction.ELU,
    "Equal": ScalarFunction.EQ,
    "Erf": ScalarFunction.ERF,
    "Exp": ScalarFunction.EXP,
    "Floor": ScalarFunction.FLOOR,
    "Gelu": ScalarFunction.GELU,
    "Greater": ScalarFunction.GT,
    "GreaterOrEqual": ScalarFunction.GE,
    "HardSigmoid": ScalarFunction.HARDSIGMOID,
    "HardSwish": ScalarFunction.HARDSWISH,
    "Identity": ScalarFunction.POSITIVE,
    "LeakyRelu": ScalarFunction.LEAKY_RELU,
    "Less": ScalarFunction.LT,
    "LessOrEqual": ScalarFunction.LE,
    "Log": ScalarFunction.LOG,
    "Max": ScalarFunction.MAXIMUM,
    "Mean": ScalarFunction.MEAN,
    "Min": ScalarFunction.MINIMUM,
    "Mish": ScalarFunction.MISH,
    "Mod": ScalarFunction.FMOD,
    "Mul": ScalarFunction.MUL,
    "Neg": ScalarFunction.NEG,
    "Not": ScalarFunction.LOGICAL_NOT,
    "Or": ScalarFunction.LOGICAL_OR,
    "PRelu": ScalarFunction.PRELU,
    "Pow": ScalarFunction.POW,
    "Reciprocal": ScalarFunction.RECIPROCAL,
    "Relu": ScalarFunction.RELU,
    "Round": ScalarFunction.ROUND,
    "Selu": ScalarFunction.SELU,
    "Sigmoid": ScalarFunction.SIGMOID,
    "Sign": ScalarFunction.SIGN,
    "Sin": ScalarFunction.SIN,
    "Sinh": ScalarFunction.SINH,
    "Softplus": ScalarFunction.SOFTPLUS,
    "Softsign": ScalarFunction.SOFTSIGN,
    "Shrink": ScalarFunction.SHRINK,
    "Sqrt": ScalarFunction.SQRT,
    "Sub": ScalarFunction.SUB,
    "Sum": ScalarFunction.ADD,
    "Swish": ScalarFunction.SWISH,
    "Tan": ScalarFunction.TAN,
    "Tanh": ScalarFunction.TANH,
    "ThresholdedRelu": ScalarFunction.THRESHOLDED_RELU,
    "Xor": ScalarFunction.LOGICAL_XOR,
}


_NO_SUFFIX_MATH = {"isfinite", "isnan", "isinf", "signbit"}


def _float_literal(value: float, dtype_info: _ScalarTypeInfo) -> str:
    if dtype_info.suffix == "f32":
        if value == int(value):
            return f"{int(value)}.0f"
        literal = f"{value}"
        if "e" in literal or "E" in literal:
            return f"{literal}f"
        if "." not in literal:
            literal = f"{literal}.0"
        return f"{literal}f"
    if value == int(value):
        return f"{int(value)}.0"
    literal = f"{value}"
    if "." not in literal and "e" not in literal and "E" not in literal:
        literal = f"{literal}.0"
    return literal


def _param_suffix(params: tuple[float, ...]) -> str:
    if not params:
        return ""
    parts: list[str] = []
    for value in params:
        if math.isnan(value):
            encoded = "nan"
        elif math.isinf(value):
            encoded = "neg_inf" if value < 0 else "inf"
        else:
            encoded = format(value, ".17g")
            if encoded == "-0":
                encoded = "0"
            encoded = encoded.replace("e-", "e_neg").replace("e+", "e")
            encoded = encoded.replace("-", "neg").replace(".", "p")
        parts.append(encoded)
    return "__" + "_".join(parts)


def _math_fn(base: str, dtype_info: _ScalarTypeInfo) -> str:
    if dtype_info.suffix == "f32" and base not in _NO_SUFFIX_MATH:
        return f"{base}f"
    return base


def _normalize_op_name(op_name: str) -> str:
    return _OP_ALIASES.get(op_name, op_name)


def _cast_value(expr: str, dtype_info: _ScalarTypeInfo) -> str:
    if dtype_info.is_small_int:
        return f"({dtype_info.c_type})({expr})"
    return expr


def _simple_unary(dtype_info: _ScalarTypeInfo, name: str, expr: str) -> _GeneratedScalar:
    lines = [
        f"static inline {dtype_info.c_type} {dtype_info.prefix}{name}({dtype_info.c_type} a) {{",
        f"    return {expr};",
        "}",
    ]
    return _GeneratedScalar(lines=lines, deps=set(), includes=set())


def _simple_binary(dtype_info: _ScalarTypeInfo, name: str, expr: str) -> _GeneratedScalar:
    lines = [
        f"static inline {dtype_info.c_type} {dtype_info.prefix}{name}({dtype_info.c_type} a, {dtype_info.c_type} b) {{",
        f"    return {expr};",
        "}",
    ]
    return _GeneratedScalar(lines=lines, deps=set(), includes=set())


def _float_unary_math(dtype_info: _ScalarTypeInfo, name: str, base: str) -> _GeneratedScalar:
    return _simple_unary(dtype_info, name, f"{_math_fn(base, dtype_info)}(a)")


def _float_binary_math(dtype_info: _ScalarTypeInfo, name: str, base: str) -> _GeneratedScalar:
    return _simple_binary(dtype_info, name, f"{_math_fn(base, dtype_info)}(a, b)")


def _float_isfinite(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    one = _float_literal(1.0, dtype_info)
    zero = _float_literal(0.0, dtype_info)
    return _simple_unary(dtype_info, "isfinite", f"isfinite(a) ? {one} : {zero}")


def _float_isnan(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    one = _float_literal(1.0, dtype_info)
    zero = _float_literal(0.0, dtype_info)
    return _simple_unary(dtype_info, "isnan", f"isnan(a) ? {one} : {zero}")


def _float_isinf(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    one = _float_literal(1.0, dtype_info)
    zero = _float_literal(0.0, dtype_info)
    return _simple_unary(dtype_info, "isinf", f"isinf(a) ? {one} : {zero}")


def _float_isneginf(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    one = _float_literal(1.0, dtype_info)
    zero = _float_literal(0.0, dtype_info)
    return _simple_unary(
        dtype_info, "isneginf", f"(isinf(a) && signbit(a)) ? {one} : {zero}"
    )


def _float_isposinf(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    one = _float_literal(1.0, dtype_info)
    zero = _float_literal(0.0, dtype_info)
    return _simple_unary(
        dtype_info, "isposinf", f"(isinf(a) && !signbit(a)) ? {one} : {zero}"
    )


def _float_comparison(
    dtype_info: _ScalarTypeInfo, name: str, op: str
) -> _GeneratedScalar:
    one = _float_literal(1.0, dtype_info)
    zero = _float_literal(0.0, dtype_info)
    return _simple_binary(dtype_info, name, f"a {op} b ? {one} : {zero}")


def _float_logical_binary(
    dtype_info: _ScalarTypeInfo, name: str, expr: str
) -> _GeneratedScalar:
    one = _float_literal(1.0, dtype_info)
    zero = _float_literal(0.0, dtype_info)
    return _simple_binary(dtype_info, name, f"{expr} ? {one} : {zero}")


def _float_logical_not(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    one = _float_literal(1.0, dtype_info)
    zero = _float_literal(0.0, dtype_info)
    return _simple_unary(dtype_info, "logical_not", f"a == {zero} ? {one} : {zero}")


def _float_remainder(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    nan = "NAN"
    lines = [
        f"static inline {dtype_info.c_type} {dtype_info.prefix}remainder({dtype_info.c_type} a, {dtype_info.c_type} b) {{",
        "    if (isnan(a) || isnan(b)) {",
        f"        return {nan};",
        "    }",
        f"    if (b == {_float_literal(0.0, dtype_info)}) {{",
        f"        return {nan};",
        "    }",
        f"    {dtype_info.c_type} mod = {_math_fn('fmod', dtype_info)}(a, b);",
        f"    if (mod == {_float_literal(0.0, dtype_info)}) {{",
        "        return mod;",
        "    }",
        "    if ((mod < 0) != (b < 0)) {",
        "        mod += b;",
        "    }",
        "    return mod;",
        "}",
    ]
    return _GeneratedScalar(lines=lines, deps=set(), includes=set())


def _float_floor_divide(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    return _simple_binary(
        dtype_info, "floor_divide", f"{_math_fn('floor', dtype_info)}(a / b)"
    )


def _float_logaddexp(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    lines = [
        f"static inline {dtype_info.c_type} {dtype_info.prefix}logaddexp({dtype_info.c_type} a, {dtype_info.c_type} b) {{",
        "    if (isnan(a) || isnan(b)) {",
        "        return NAN;",
        "    }",
        f"    {dtype_info.c_type} max_val = {_math_fn('fmax', dtype_info)}(a, b);",
        f"    {dtype_info.c_type} min_val = {_math_fn('fmin', dtype_info)}(a, b);",
        "    if (max_val == -INFINITY) {",
        "        return -INFINITY;",
        "    }",
        f"    return max_val + {_math_fn('log1p', dtype_info)}({_math_fn('exp', dtype_info)}(min_val - max_val));",
        "}",
    ]
    return _GeneratedScalar(lines=lines, deps=set(), includes=set())


def _float_logaddexp2(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    one = _float_literal(1.0, dtype_info)
    lines = [
        f"static inline {dtype_info.c_type} {dtype_info.prefix}logaddexp2({dtype_info.c_type} a, {dtype_info.c_type} b) {{",
        "    if (isnan(a) || isnan(b)) {",
        "        return NAN;",
        "    }",
        f"    {dtype_info.c_type} max_val = {_math_fn('fmax', dtype_info)}(a, b);",
        f"    {dtype_info.c_type} min_val = {_math_fn('fmin', dtype_info)}(a, b);",
        "    if (max_val == -INFINITY) {",
        "        return -INFINITY;",
        "    }",
        f"    return max_val + {_math_fn('log2', dtype_info)}({one} + {_math_fn('exp2', dtype_info)}(min_val - max_val));",
        "}",
    ]
    return _GeneratedScalar(lines=lines, deps=set(), includes=set())


def _float_xlogy(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    zero = _float_literal(0.0, dtype_info)
    lines = [
        f"static inline {dtype_info.c_type} {dtype_info.prefix}xlogy({dtype_info.c_type} a, {dtype_info.c_type} b) {{",
        "    if (isnan(a) || isnan(b)) {",
        "        return NAN;",
        "    }",
        f"    if (a == {zero}) {{",
        f"        return {zero};",
        "    }",
        f"    return a * {_math_fn('log', dtype_info)}(b);",
        "}",
    ]
    return _GeneratedScalar(lines=lines, deps=set(), includes=set())


def _float_heaviside(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    zero = _float_literal(0.0, dtype_info)
    one = _float_literal(1.0, dtype_info)
    lines = [
        f"static inline {dtype_info.c_type} {dtype_info.prefix}heaviside({dtype_info.c_type} a, {dtype_info.c_type} b) {{",
        f"    if (a > {zero}) {{",
        f"        return {one};",
        "    }",
        f"    if (a == {zero}) {{",
        "        return b;",
        "    }",
        f"    return {zero};",
        "}",
    ]
    return _GeneratedScalar(lines=lines, deps=set(), includes=set())


def _float_ldexp(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    return _simple_binary(
        dtype_info, "ldexp", f"a * {_math_fn('exp2', dtype_info)}(b)"
    )


def _float_reciprocal(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    one = _float_literal(1.0, dtype_info)
    return _simple_unary(dtype_info, "reciprocal", f"{one} / a")


def _float_relu(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    zero = _float_literal(0.0, dtype_info)
    return _simple_unary(dtype_info, "relu", f"a > {zero} ? a : {zero}")


def _float_rsqrt(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    one = _float_literal(1.0, dtype_info)
    return _simple_unary(dtype_info, "rsqrt", f"{one} / {_math_fn('sqrt', dtype_info)}(a)")


def _float_sigmoid(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    one = _float_literal(1.0, dtype_info)
    return _simple_unary(dtype_info, "sigmoid", f"{one} / ({one} + {_math_fn('exp', dtype_info)}(-a))")


def _float_log_sigmoid(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    zero = _float_literal(0.0, dtype_info)
    lines = [
        f"static inline {dtype_info.c_type} {dtype_info.prefix}log_sigmoid({dtype_info.c_type} a) {{",
        f"    if (a >= {zero}) {{",
        f"        return -{_math_fn('log1p', dtype_info)}({_math_fn('exp', dtype_info)}(-a));",
        "    }",
        f"    return a - {_math_fn('log1p', dtype_info)}({_math_fn('exp', dtype_info)}(a));",
        "}",
    ]
    return _GeneratedScalar(lines=lines, deps=set(), includes=set())


def _float_gelu(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    inv_sqrt2 = _float_literal(0.7071067811865475, dtype_info)
    half = _float_literal(0.5, dtype_info)
    one = _float_literal(1.0, dtype_info)
    lines = [
        f"static inline {dtype_info.c_type} {dtype_info.prefix}gelu({dtype_info.c_type} a) {{",
        f"    const {dtype_info.c_type} inv_sqrt2 = {inv_sqrt2};",
        f"    return {half} * a * ({one} + {_math_fn('erf', dtype_info)}(a * inv_sqrt2));",
        "}",
    ]
    return _GeneratedScalar(lines=lines, deps=set(), includes=set())


def _float_elu(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    one = _float_literal(1.0, dtype_info)
    lines = [
        f"static inline {dtype_info.c_type} {dtype_info.prefix}elu({dtype_info.c_type} a) {{",
        f"    const {dtype_info.c_type} alpha = {one};",
        f"    const {dtype_info.c_type} scale = {one};",
        f"    const {dtype_info.c_type} input_scale = {one};",
        "    if (a > 0) {",
        "        return scale * a;",
        "    }",
        f"    return scale * alpha * ({_math_fn('exp', dtype_info)}(input_scale * a) - {one});",
        "}",
    ]
    return _GeneratedScalar(lines=lines, deps=set(), includes=set())


def _float_elu_param(
    dtype_info: _ScalarTypeInfo,
    params: tuple[float, ...],
    function_name: str,
) -> _GeneratedScalar:
    if params and len(params) != 1:
        raise ScalarFunctionError("elu expects 1 parameter: alpha")
    alpha_value = params[0] if params else 1.0
    alpha = _float_literal(alpha_value, dtype_info)
    one = _float_literal(1.0, dtype_info)
    lines = [
        f"static inline {dtype_info.c_type} {function_name}({dtype_info.c_type} a) {{",
        f"    const {dtype_info.c_type} alpha = {alpha};",
        "    if (a >= 0) {",
        "        return a;",
        "    }",
        f"    return alpha * ({_math_fn('exp', dtype_info)}(a) - {one});",
        "}",
    ]
    return _GeneratedScalar(lines=lines, deps=set(), includes=set())


def _float_celu(
    dtype_info: _ScalarTypeInfo,
    params: tuple[float, ...],
    function_name: str,
) -> _GeneratedScalar:
    if params and len(params) != 1:
        raise ScalarFunctionError("celu expects 1 parameter: alpha")
    alpha_value = params[0] if params else 1.0
    alpha = _float_literal(alpha_value, dtype_info)
    one = _float_literal(1.0, dtype_info)
    lines = [
        f"static inline {dtype_info.c_type} {function_name}({dtype_info.c_type} a) {{",
        f"    const {dtype_info.c_type} alpha = {alpha};",
        "    if (a > 0) {",
        "        return a;",
        "    }",
        f"    return alpha * ({_math_fn('exp', dtype_info)}(a / alpha) - {one});",
        "}",
    ]
    return _GeneratedScalar(lines=lines, deps=set(), includes=set())


def _float_affine(
    dtype_info: _ScalarTypeInfo,
    params: tuple[float, ...],
    function_name: str,
) -> _GeneratedScalar:
    if params and len(params) != 2:
        raise ScalarFunctionError("affine expects 2 parameters: alpha, beta")
    alpha_value = params[0] if params else 1.0
    beta_value = params[1] if len(params) > 1 else 0.0
    alpha = _float_literal(alpha_value, dtype_info)
    beta = _float_literal(beta_value, dtype_info)
    lines = [
        f"static inline {dtype_info.c_type} {function_name}({dtype_info.c_type} a) {{",
        f"    const {dtype_info.c_type} alpha = {alpha};",
        f"    const {dtype_info.c_type} beta = {beta};",
        "    return alpha * a + beta;",
        "}",
    ]
    return _GeneratedScalar(lines=lines, deps=set(), includes=set())


def _float_scaled_tanh(
    dtype_info: _ScalarTypeInfo,
    params: tuple[float, ...],
    function_name: str,
) -> _GeneratedScalar:
    if params and len(params) != 2:
        raise ScalarFunctionError("scaled_tanh expects 2 parameters: alpha, beta")
    alpha_value = params[0] if params else 1.0
    beta_value = params[1] if len(params) > 1 else 1.0
    alpha = _float_literal(alpha_value, dtype_info)
    beta = _float_literal(beta_value, dtype_info)
    lines = [
        f"static inline {dtype_info.c_type} {function_name}({dtype_info.c_type} a) {{",
        f"    const {dtype_info.c_type} alpha = {alpha};",
        f"    const {dtype_info.c_type} beta = {beta};",
        f"    return alpha * {_math_fn('tanh', dtype_info)}(beta * a);",
        "}",
    ]
    return _GeneratedScalar(lines=lines, deps=set(), includes=set())


def _float_swish(
    dtype_info: _ScalarTypeInfo,
    params: tuple[float, ...],
    function_name: str,
) -> _GeneratedScalar:
    if params and len(params) != 1:
        raise ScalarFunctionError("swish expects 1 parameter: alpha")
    alpha_value = params[0] if params else 1.0
    alpha = _float_literal(alpha_value, dtype_info)
    one = _float_literal(1.0, dtype_info)
    lines = [
        f"static inline {dtype_info.c_type} {function_name}({dtype_info.c_type} a) {{",
        f"    const {dtype_info.c_type} alpha = {alpha};",
        f"    return a / ({one} + {_math_fn('exp', dtype_info)}(-alpha * a));",
        "}",
    ]
    return _GeneratedScalar(lines=lines, deps=set(), includes=set())


def _float_leaky_relu_param(
    dtype_info: _ScalarTypeInfo,
    params: tuple[float, ...],
    function_name: str,
) -> _GeneratedScalar:
    if params and len(params) != 1:
        raise ScalarFunctionError("leaky_relu expects 1 parameter: alpha")
    alpha_value = params[0] if params else 0.01
    alpha = _float_literal(alpha_value, dtype_info)
    lines = [
        f"static inline {dtype_info.c_type} {function_name}({dtype_info.c_type} a) {{",
        f"    const {dtype_info.c_type} alpha = {alpha};",
        "    return a < 0 ? alpha * a : a;",
        "}",
    ]
    return _GeneratedScalar(lines=lines, deps=set(), includes=set())


def _float_thresholded_relu_param(
    dtype_info: _ScalarTypeInfo,
    params: tuple[float, ...],
    function_name: str,
) -> _GeneratedScalar:
    if params and len(params) != 1:
        raise ScalarFunctionError("thresholded_relu expects 1 parameter: alpha")
    alpha_value = params[0] if params else 1.0
    alpha = _float_literal(alpha_value, dtype_info)
    zero = _float_literal(0.0, dtype_info)
    lines = [
        f"static inline {dtype_info.c_type} {function_name}({dtype_info.c_type} a) {{",
        f"    const {dtype_info.c_type} alpha = {alpha};",
        f"    return a > alpha ? a : {zero};",
        "}",
    ]
    return _GeneratedScalar(lines=lines, deps=set(), includes=set())


def _float_hardsigmoid_param(
    dtype_info: _ScalarTypeInfo,
    params: tuple[float, ...],
    function_name: str,
) -> _GeneratedScalar:
    if params and len(params) != 2:
        raise ScalarFunctionError("hardsigmoid expects 2 parameters: alpha, beta")
    alpha_value = params[0] if params else 0.2
    beta_value = params[1] if len(params) > 1 else 0.5
    alpha = _float_literal(alpha_value, dtype_info)
    beta = _float_literal(beta_value, dtype_info)
    zero = _float_literal(0.0, dtype_info)
    one = _float_literal(1.0, dtype_info)
    lines = [
        f"static inline {dtype_info.c_type} {function_name}({dtype_info.c_type} a) {{",
        f"    {dtype_info.c_type} value = a * {alpha} + {beta};",
        f"    {dtype_info.c_type} clamped = {_math_fn('fmin', dtype_info)}({one}, {_math_fn('fmax', dtype_info)}({zero}, value));",
        "    return clamped;",
        "}",
    ]
    return _GeneratedScalar(lines=lines, deps=set(), includes=set())


def _float_shrink(
    dtype_info: _ScalarTypeInfo,
    params: tuple[float, ...],
    function_name: str,
) -> _GeneratedScalar:
    if params and len(params) != 2:
        raise ScalarFunctionError("shrink expects 2 parameters: bias, lambd")
    bias_value = params[0] if params else 0.0
    lambd_value = params[1] if params else 0.5
    bias = _float_literal(bias_value, dtype_info)
    lambd = _float_literal(lambd_value, dtype_info)
    zero = _float_literal(0.0, dtype_info)
    lines = [
        f"static inline {dtype_info.c_type} {function_name}({dtype_info.c_type} a) {{",
        f"    const {dtype_info.c_type} bias = {bias};",
        f"    const {dtype_info.c_type} lambd = {lambd};",
        "    if (a < -lambd) {",
        "        return a + bias;",
        "    }",
        "    if (a > lambd) {",
        "        return a - bias;",
        "    }",
        f"    return {zero};",
        "}",
    ]
    return _GeneratedScalar(lines=lines, deps=set(), includes=set())


def _float_leaky_relu(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    negative_slope = _float_literal(0.01, dtype_info)
    lines = [
        f"static inline {dtype_info.c_type} {dtype_info.prefix}leaky_relu({dtype_info.c_type} a) {{",
        f"    const {dtype_info.c_type} negative_slope = {negative_slope};",
        "    return a > 0 ? a : negative_slope * a;",
        "}",
    ]
    return _GeneratedScalar(lines=lines, deps=set(), includes=set())


def _float_softplus(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    beta = _float_literal(1.0, dtype_info)
    threshold = _float_literal(20.0, dtype_info)
    lines = [
        f"static inline {dtype_info.c_type} {dtype_info.prefix}softplus({dtype_info.c_type} a) {{",
        f"    const {dtype_info.c_type} beta = {beta};",
        f"    const {dtype_info.c_type} threshold = {threshold};",
        "    if (beta * a > threshold) {",
        "        return a;",
        "    }",
        f"    return {_math_fn('log1p', dtype_info)}({_math_fn('exp', dtype_info)}(beta * a)) / beta;",
        "}",
    ]
    return _GeneratedScalar(lines=lines, deps=set(), includes=set())


def _float_softsign(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    one = _float_literal(1.0, dtype_info)
    return _simple_unary(
        dtype_info,
        "softsign",
        f"a / ({one} + {_math_fn('fabs', dtype_info)}(a))",
    )


def _float_silu(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    one = _float_literal(1.0, dtype_info)
    return _simple_unary(dtype_info, "silu", f"a / ({one} + {_math_fn('exp', dtype_info)}(-a))")


def _float_mish(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    twenty = _float_literal(20.0, dtype_info)
    zero = _float_literal(0.0, dtype_info)
    lines = [
        f"static inline {dtype_info.c_type} {dtype_info.prefix}mish({dtype_info.c_type} a) {{",
        f"    if (a > {twenty}) {{",
        "        return a;",
        "    }",
        f"    if (a < -{twenty}) {{",
        f"        {dtype_info.c_type} exp_a = {_math_fn('exp', dtype_info)}(a);",
        "        return a * exp_a;",
        "    }",
        f"    {dtype_info.c_type} softplus = {_math_fn('log1p', dtype_info)}({_math_fn('exp', dtype_info)}(a));",
        f"    return a * {_math_fn('tanh', dtype_info)}(softplus);",
        "}",
    ]
    return _GeneratedScalar(lines=lines, deps=set(), includes=set())


def _float_selu(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    alpha = "1.6732632423543772848170429916717"
    scale = "1.0507009873554804934193349852946"
    alpha_literal = f"{alpha}f" if dtype_info.suffix == "f32" else alpha
    scale_literal = f"{scale}f" if dtype_info.suffix == "f32" else scale
    one = _float_literal(1.0, dtype_info)
    lines = [
        f"static inline {dtype_info.c_type} {dtype_info.prefix}selu({dtype_info.c_type} a) {{",
        f"    const {dtype_info.c_type} alpha = {alpha_literal};",
        f"    const {dtype_info.c_type} scale = {scale_literal};",
        "    if (a > 0) {",
        "        return scale * a;",
        "    }",
        f"    return scale * alpha * ({_math_fn('exp', dtype_info)}(a) - {one});",
        "}",
    ]
    return _GeneratedScalar(lines=lines, deps=set(), includes=set())


def _float_thresholded_relu(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    zero = _float_literal(0.0, dtype_info)
    alpha = _float_literal(1.0, dtype_info)
    return _simple_unary(
        dtype_info,
        "thresholded_relu",
        f"a > {alpha} ? a : {zero}",
    )


def _float_relu6(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    zero = _float_literal(0.0, dtype_info)
    six = _float_literal(6.0, dtype_info)
    return _simple_unary(
        dtype_info,
        "relu6",
        f"{_math_fn('fmin', dtype_info)}({six}, {_math_fn('fmax', dtype_info)}({zero}, a))",
    )


def _float_hardsigmoid(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    alpha = _float_literal(0.2, dtype_info)
    beta = _float_literal(0.5, dtype_info)
    zero = _float_literal(0.0, dtype_info)
    one = _float_literal(1.0, dtype_info)
    lines = [
        f"static inline {dtype_info.c_type} {dtype_info.prefix}hardsigmoid({dtype_info.c_type} a) {{",
        f"    {dtype_info.c_type} value = a * {alpha} + {beta};",
        f"    {dtype_info.c_type} clamped = {_math_fn('fmin', dtype_info)}({one}, {_math_fn('fmax', dtype_info)}({zero}, value));",
        "    return clamped;",
        "}",
    ]
    return _GeneratedScalar(lines=lines, deps=set(), includes=set())


def _float_hardswish(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    three = _float_literal(3.0, dtype_info)
    six = _float_literal(6.0, dtype_info)
    zero = _float_literal(0.0, dtype_info)
    lines = [
        f"static inline {dtype_info.c_type} {dtype_info.prefix}hardswish({dtype_info.c_type} a) {{",
        f"    {dtype_info.c_type} shifted = a + {three};",
        f"    {dtype_info.c_type} clamped = {_math_fn('fmin', dtype_info)}({six}, {_math_fn('fmax', dtype_info)}({zero}, shifted));",
        f"    return a * clamped / {six};",
        "}",
    ]
    return _GeneratedScalar(lines=lines, deps=set(), includes=set())


def _float_sign(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    zero = _float_literal(0.0, dtype_info)
    one = _float_literal(1.0, dtype_info)
    minus_one = _float_literal(-1.0, dtype_info)
    lines = [
        f"static inline {dtype_info.c_type} {dtype_info.prefix}sign({dtype_info.c_type} a) {{",
        "    if (isnan(a)) {",
        "        return a;",
        "    }",
        f"    if (a > {zero}) {{",
        f"        return {one};",
        "    }",
        f"    if (a < {zero}) {{",
        f"        return {minus_one};",
        "    }",
        f"    return {zero};",
        "}",
    ]
    return _GeneratedScalar(lines=lines, deps=set(), includes=set())


def _float_round(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    return _float_unary_math(dtype_info, "round", "round")


def _float_trunc(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    return _float_unary_math(dtype_info, "trunc", "trunc")


def _float_angle(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    zero = _float_literal(0.0, dtype_info)
    pi = "REF_PI_F" if dtype_info.suffix == "f32" else "REF_PI_D"
    lines = [
        f"static inline {dtype_info.c_type} {dtype_info.prefix}angle({dtype_info.c_type} a) {{",
        "    if (isnan(a)) {",
        "        return a;",
        "    }",
        f"    return a < {zero} ? {pi} : {zero};",
        "}",
    ]
    return _GeneratedScalar(lines=lines, deps=set(), includes=set(), constants={pi})


def _float_conj(dtype_info: _ScalarTypeInfo, name: str) -> _GeneratedScalar:
    return _simple_unary(dtype_info, name, "a")


def _float_deg2rad(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    pi = "REF_PI_F" if dtype_info.suffix == "f32" else "REF_PI_D"
    one_eighty = _float_literal(180.0, dtype_info)
    generated = _simple_unary(dtype_info, "deg2rad", f"a * ({pi} / {one_eighty})")
    return _GeneratedScalar(
        lines=generated.lines,
        deps=generated.deps,
        includes=generated.includes,
        constants={pi},
    )


def _float_rad2deg(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    pi = "REF_PI_F" if dtype_info.suffix == "f32" else "REF_PI_D"
    one_eighty = _float_literal(180.0, dtype_info)
    generated = _simple_unary(dtype_info, "rad2deg", f"a * ({one_eighty} / {pi})")
    return _GeneratedScalar(
        lines=generated.lines,
        deps=generated.deps,
        includes=generated.includes,
        constants={pi},
    )


def _float_digamma_f64() -> _GeneratedScalar:
    lines = [
        "static inline double ref_scalar_f64_digamma(double x) {",
        "    if (isnan(x) || isinf(x)) {",
        "        return x;",
        "    }",
        "    if (x <= 0.0) {",
        "        double frac = x - floor(x);",
        "        if (frac == 0.0) {",
        "            return NAN;",
        "        }",
        "        return ref_scalar_f64_digamma(1.0 - x) - REF_PI_D / tan(REF_PI_D * x);",
        "    }",
        "    double result = 0.0;",
        "    while (x < 10.0) {",
        "        result -= 1.0 / x;",
        "        x += 1.0;",
        "    }",
        "    double inv = 1.0 / x;",
        "    double inv2 = inv * inv;",
        "    result += log(x) - 0.5 * inv",
        "        - inv2 * (1.0 / 12.0 - inv2 * (1.0 / 120.0",
        "        - inv2 * (1.0 / 252.0 - inv2 * (1.0 / 240.0",
        "        - inv2 * (1.0 / 132.0 - inv2 * (691.0 / 32760.0))))));",
        "    return result;",
        "}",
    ]
    return _GeneratedScalar(
        lines=lines, deps=set(), includes=set(), constants={"REF_PI_D"}
    )


def _float_digamma(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    if dtype_info.suffix == "f64":
        return _float_digamma_f64()
    lines = [
        f"static inline {dtype_info.c_type} {dtype_info.prefix}digamma({dtype_info.c_type} x) {{",
        "    return (float)ref_scalar_f64_digamma((double)x);",
        "}",
    ]
    deps = {_scalar_key_from_op(_SCALAR_TYPES[ScalarType.F64], "digamma")}
    return _GeneratedScalar(lines=lines, deps=deps, includes=set())


def _float_erfinv(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    pi = "REF_PI_F" if dtype_info.suffix == "f32" else "REF_PI_D"
    one = _float_literal(1.0, dtype_info)
    zero = _float_literal(0.0, dtype_info)
    two = _float_literal(2.0, dtype_info)
    a_literal = _float_literal(0.147, dtype_info)
    lines = [
        f"static inline {dtype_info.c_type} {dtype_info.prefix}erfinv({dtype_info.c_type} x) {{",
        "    if (isnan(x)) {",
        "        return x;",
        "    }",
        f"    if (x <= -{one}) {{",
        f"        return x == -{one} ? -INFINITY : NAN;",
        "    }",
        f"    if (x >= {one}) {{",
        f"        return x == {one} ? INFINITY : NAN;",
        "    }",
        f"    if (x == {zero}) {{",
        f"        return {zero};",
        "    }",
        f"    {dtype_info.c_type} a = {a_literal};",
        f"    {dtype_info.c_type} ln = {_math_fn('log', dtype_info)}({one} - x * x);",
        f"    {dtype_info.c_type} term = {two} / ({pi} * a) + ln / {two};",
        f"    {dtype_info.c_type} inner = term * term - ln / a;",
        f"    {dtype_info.c_type} approx = {_math_fn('sqrt', dtype_info)}({_math_fn('fmax', dtype_info)}({zero}, {_math_fn('sqrt', dtype_info)}(inner) - term));",
        f"    if (x < {zero}) {{",
        "        approx = -approx;",
        "    }",
        "    for (int i = 0; i < 2; ++i) {",
        f"        {dtype_info.c_type} err = {_math_fn('erf', dtype_info)}(approx) - x;",
        f"        {dtype_info.c_type} deriv = {two} / {_math_fn('sqrt', dtype_info)}({pi}) * {_math_fn('exp', dtype_info)}(-approx * approx);",
        "        approx -= err / deriv;",
        "    }",
        "    return approx;",
        "}",
    ]
    return _GeneratedScalar(lines=lines, deps=set(), includes=set(), constants={pi})


def _float_frac(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    return _simple_unary(dtype_info, "frac", f"a - {_math_fn('trunc', dtype_info)}(a)")


def _float_i0(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    zero = _float_literal(0.0, dtype_info)
    three_seven_five = _float_literal(3.75, dtype_info)
    lines = [
        f"static inline {dtype_info.c_type} {dtype_info.prefix}i0({dtype_info.c_type} x) {{",
        f"    {dtype_info.c_type} ax = {_math_fn('fabs', dtype_info)}(x);",
        f"    if (ax < {three_seven_five}) {{",
        f"        {dtype_info.c_type} y = x / {three_seven_five};",
        "        y *= y;",
        f"        return { _float_literal(1.0, dtype_info)} + y * ({_float_literal(3.5156229, dtype_info)} + y * ({_float_literal(3.0899424, dtype_info)} + y * ({_float_literal(1.2067492, dtype_info)}",
        f"            + y * ({_float_literal(0.2659732, dtype_info)} + y * ({_float_literal(0.0360768, dtype_info)} + y * {_float_literal(0.0045813, dtype_info)})))));",
        "    }",
        f"    {dtype_info.c_type} y = {three_seven_five} / ax;",
        f"    return ({_math_fn('exp', dtype_info)}(ax) / {_math_fn('sqrt', dtype_info)}(ax)) * ({_float_literal(0.39894228, dtype_info)} + y * ({_float_literal(0.01328592, dtype_info)}",
        f"        + y * ({_float_literal(0.00225319, dtype_info)} + y * ({_float_literal(-0.00157565, dtype_info)} + y * ({_float_literal(0.00916281, dtype_info)}",
        f"        + y * ({_float_literal(-0.02057706, dtype_info)} + y * ({_float_literal(0.02635537, dtype_info)}",
        f"        + y * ({_float_literal(-0.01647633, dtype_info)} + y * {_float_literal(0.00392377, dtype_info)}))))))));",
        "}",
    ]
    return _GeneratedScalar(lines=lines, deps=set(), includes=set())


def _float_logit(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    zero = _float_literal(0.0, dtype_info)
    one = _float_literal(1.0, dtype_info)
    lines = [
        f"static inline {dtype_info.c_type} {dtype_info.prefix}logit({dtype_info.c_type} a) {{",
        "    if (isnan(a)) {",
        "        return a;",
        "    }",
        f"    if (a == {zero}) {{",
        "        return -INFINITY;",
        "    }",
        f"    if (a == {one}) {{",
        "        return INFINITY;",
        "    }",
        f"    if (a < {zero} || a > {one}) {{",
        "        return NAN;",
        "    }",
        f"    return {_math_fn('log', dtype_info)}(a / ({one} - a));",
        "}",
    ]
    return _GeneratedScalar(lines=lines, deps=set(), includes=set())


def _float_nan_to_num(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    zero = _float_literal(0.0, dtype_info)
    lines = [
        f"static inline {dtype_info.c_type} {dtype_info.prefix}nan_to_num({dtype_info.c_type} a) {{",
        "    if (isnan(a)) {",
        f"        return {zero};",
        "    }",
        "    if (isinf(a)) {",
        "        return signbit(a) ? -FLT_MAX : FLT_MAX;",
        "    }",
        "    return a;",
        "}",
    ]
    return _GeneratedScalar(lines=lines, deps=set(), includes=set())


def _float_sgn(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    zero = _float_literal(0.0, dtype_info)
    one = _float_literal(1.0, dtype_info)
    minus_one = _float_literal(-1.0, dtype_info)
    lines = [
        f"static inline {dtype_info.c_type} {dtype_info.prefix}sgn({dtype_info.c_type} a) {{",
        "    if (isnan(a)) {",
        f"        return {zero};",
        "    }",
        f"    if (a > {zero}) {{",
        f"        return {one};",
        "    }",
        f"    if (a < {zero}) {{",
        f"        return {minus_one};",
        "    }",
        f"    return {zero};",
        "}",
    ]
    return _GeneratedScalar(lines=lines, deps=set(), includes=set())


def _float_sinc(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    zero = _float_literal(0.0, dtype_info)
    one = _float_literal(1.0, dtype_info)
    pi = "REF_PI_F" if dtype_info.suffix == "f32" else "REF_PI_D"
    lines = [
        f"static inline {dtype_info.c_type} {dtype_info.prefix}sinc({dtype_info.c_type} a) {{",
        f"    if (a == {zero}) {{",
        f"        return {one};",
        "    }",
        f"    {dtype_info.c_type} x = {pi} * a;",
        f"    return {_math_fn('sin', dtype_info)}(x) / x;",
        "}",
    ]
    return _GeneratedScalar(lines=lines, deps=set(), includes=set(), constants={pi})


def _float_square(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    return _simple_unary(dtype_info, "square", "a * a")


def _float_positive(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    return _simple_unary(dtype_info, "positive", "a")


def _float_clamp_min(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    return _simple_binary(dtype_info, "clamp_min", f"{_math_fn('fmax', dtype_info)}(a, b)")


def _float_clamp_max(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    return _simple_binary(dtype_info, "clamp_max", f"{_math_fn('fmin', dtype_info)}(a, b)")

def _float_binary_op_handler(name: str, op: str) -> Callable[[_ScalarTypeInfo], _GeneratedScalar]:
    def handler(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
        return _simple_binary(dtype_info, name, f"a {op} b")

    return handler


def _float_unary_math_handler(name: str, base: str | None = None) -> Callable[
    [_ScalarTypeInfo], _GeneratedScalar
]:
    base_name = base or name

    def handler(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
        return _float_unary_math(dtype_info, name, base_name)

    return handler


def _float_binary_math_handler(name: str, base: str | None = None) -> Callable[
    [_ScalarTypeInfo], _GeneratedScalar
]:
    base_name = base or name

    def handler(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
        return _float_binary_math(dtype_info, name, base_name)

    return handler


def _float_comparison_handler(name: str, op: str) -> Callable[[_ScalarTypeInfo], _GeneratedScalar]:
    def handler(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
        return _float_comparison(dtype_info, name, op)

    return handler


def _float_logical_or(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    zero = _float_literal(0.0, dtype_info)
    return _float_logical_binary(dtype_info, "logical_or", f"(a != {zero} || b != {zero})")


def _float_logical_and(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    zero = _float_literal(0.0, dtype_info)
    return _float_logical_binary(dtype_info, "logical_and", f"(a != {zero} && b != {zero})")


def _float_logical_xor(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    zero = _float_literal(0.0, dtype_info)
    return _float_logical_binary(dtype_info, "logical_xor", f"((a != {zero}) != (b != {zero}))")


_FLOAT_OP_DISPATCH: Mapping[str, Callable[[_ScalarTypeInfo], _GeneratedScalar]] = {
    "abs": _float_unary_math_handler("abs", "fabs"),
    "add": _float_binary_op_handler("add", "+"),
    "sub": _float_binary_op_handler("sub", "-"),
    "mul": _float_binary_op_handler("mul", "*"),
    "div": _float_binary_op_handler("div", "/"),
    "maximum": _float_binary_math_handler("maximum", "fmax"),
    "fmax": _float_binary_math_handler("fmax", "fmax"),
    "minimum": _float_binary_math_handler("minimum", "fmin"),
    "fmin": _float_binary_math_handler("fmin", "fmin"),
    "le": _float_comparison_handler("le", "<="),
    "lt": _float_comparison_handler("lt", "<"),
    "ge": _float_comparison_handler("ge", ">="),
    "gt": _float_comparison_handler("gt", ">"),
    "eq": _float_comparison_handler("eq", "=="),
    "ne": _float_comparison_handler("ne", "!="),
    "logical_or": _float_logical_or,
    "logical_and": _float_logical_and,
    "logical_xor": _float_logical_xor,
    "logical_not": _float_logical_not,
    "copysign": _float_binary_math_handler("copysign"),
    "hypot": _float_binary_math_handler("hypot"),
    "atan2": _float_binary_math_handler("atan2"),
    "pow": _float_binary_math_handler("pow"),
    "fmod": _float_binary_math_handler("fmod"),
    "remainder": _float_remainder,
    "floor_divide": _float_floor_divide,
    "logaddexp": _float_logaddexp,
    "logaddexp2": _float_logaddexp2,
    "nextafter": _float_binary_math_handler("nextafter"),
    "xlogy": _float_xlogy,
    "heaviside": _float_heaviside,
    "ldexp": _float_ldexp,
    "clamp_min": _float_clamp_min,
    "clamp_max": _float_clamp_max,
    "neg": lambda dtype_info: _simple_unary(dtype_info, "neg", "-a"),
    "reciprocal": _float_reciprocal,
    "relu": _float_relu,
    "ceil": _float_unary_math_handler("ceil"),
    "floor": _float_unary_math_handler("floor"),
    "sin": _float_unary_math_handler("sin"),
    "cos": _float_unary_math_handler("cos"),
    "sqrt": _float_unary_math_handler("sqrt"),
    "cbrt": _float_unary_math_handler("cbrt"),
    "exp": _float_unary_math_handler("exp"),
    "tanh": _float_unary_math_handler("tanh"),
    "log": _float_unary_math_handler("log"),
    "acos": _float_unary_math_handler("acos"),
    "acosh": _float_unary_math_handler("acosh"),
    "asin": _float_unary_math_handler("asin"),
    "asinh": _float_unary_math_handler("asinh"),
    "atan": _float_unary_math_handler("atan"),
    "atanh": _float_unary_math_handler("atanh"),
    "cosh": _float_unary_math_handler("cosh"),
    "sinh": _float_unary_math_handler("sinh"),
    "tan": _float_unary_math_handler("tan"),
    "erf": _float_unary_math_handler("erf"),
    "erfc": _float_unary_math_handler("erfc"),
    "expm1": _float_unary_math_handler("expm1"),
    "log1p": _float_unary_math_handler("log1p"),
    "log2": _float_unary_math_handler("log2"),
    "log10": _float_unary_math_handler("log10"),
    "exp2": _float_unary_math_handler("exp2"),
    "lgamma": _float_unary_math_handler("lgamma"),
    "isfinite": _float_isfinite,
    "rsqrt": _float_rsqrt,
    "sigmoid": _float_sigmoid,
    "log_sigmoid": _float_log_sigmoid,
    "gelu": _float_gelu,
    "elu": _float_elu,
    "leaky_relu": _float_leaky_relu,
    "softplus": _float_softplus,
    "softsign": _float_softsign,
    "silu": _float_silu,
    "mish": _float_mish,
    "selu": _float_selu,
    "relu6": _float_relu6,
    "hardsigmoid": _float_hardsigmoid,
    "hardswish": _float_hardswish,
    "thresholded_relu": _float_thresholded_relu,
    "sign": _float_sign,
    "round": _float_round,
    "trunc": _float_trunc,
    "angle": _float_angle,
    "conj": lambda dtype_info: _float_conj(dtype_info, "conj"),
    "conj_physical": lambda dtype_info: _float_conj(dtype_info, "conj_physical"),
    "deg2rad": _float_deg2rad,
    "digamma": _float_digamma,
    "erfinv": _float_erfinv,
    "frac": _float_frac,
    "i0": _float_i0,
    "logit": _float_logit,
    "isnan": _float_isnan,
    "isinf": _float_isinf,
    "isneginf": _float_isneginf,
    "isposinf": _float_isposinf,
    "nan_to_num": _float_nan_to_num,
    "positive": _float_positive,
    "rad2deg": _float_rad2deg,
    "real": lambda dtype_info: _simple_unary(dtype_info, "real", "a"),
    "sgn": _float_sgn,
    "sinc": _float_sinc,
    "square": _float_square,
}

_PARAMETERIZED_FLOAT_OPS: Mapping[
    ScalarFunction,
    Callable[[_ScalarTypeInfo, tuple[float, ...], str], _GeneratedScalar],
] = {
    ScalarFunction.AFFINE: _float_affine,
    ScalarFunction.CELU: _float_celu,
    ScalarFunction.ELU: _float_elu_param,
    ScalarFunction.HARDSIGMOID: _float_hardsigmoid_param,
    ScalarFunction.LEAKY_RELU: _float_leaky_relu_param,
    ScalarFunction.SCALED_TANH: _float_scaled_tanh,
    ScalarFunction.SHRINK: _float_shrink,
    ScalarFunction.SWISH: _float_swish,
    ScalarFunction.THRESHOLDED_RELU: _float_thresholded_relu_param,
}


def _float_from_ops(dtype_info: _ScalarTypeInfo, name: str) -> _GeneratedScalar:
    canonical_name = _normalize_op_name(name)
    if canonical_name != name:
        lines = [
            f"static inline {dtype_info.c_type} {dtype_info.prefix}{name}({dtype_info.c_type} a) {{",
            f"    return {dtype_info.prefix}{canonical_name}(a);",
            "}",
        ]
        deps = {_scalar_key_from_op(dtype_info, canonical_name)}
        return _GeneratedScalar(lines=lines, deps=deps, includes=set())
    name = canonical_name
    handler = _FLOAT_OP_DISPATCH.get(name)
    if handler is None:
        raise ScalarFunctionError(f"unsupported float scalar op: {name}")
    return handler(dtype_info)


def _int_from_f32(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    lines: List[str] = []
    includes: Set[str] = {"#include <math.h>", "#include <stdint.h>"}
    if dtype_info.is_signed:
        includes.add("#include <limits.h>")
        min_name = f"INT{dtype_info.bits}_MIN"
        max_name = f"INT{dtype_info.bits}_MAX"
        lines.extend(
            [
                f"static inline {dtype_info.c_type} {dtype_info.prefix}from_f32(float value) {{",
                "    if (!isfinite(value)) {",
                f"        return {min_name};",
                "    }",
                f"    if (value > (float){max_name}) {{",
                f"        return {max_name};",
                "    }",
                f"    if (value < (float){min_name}) {{",
                f"        return {min_name};",
                "    }",
                f"    return ({dtype_info.c_type})value;",
                "}",
            ]
        )
        return _GeneratedScalar(lines=lines, deps=set(), includes=includes)
    max_name = f"UINT{dtype_info.bits}_MAX"
    if dtype_info.bits in {32, 64}:
        lines.extend(
            [
                f"static inline {dtype_info.c_type} {dtype_info.prefix}from_f32(float value) {{",
                "    if (!isfinite(value)) {",
                "        return 0;",
                "    }",
                "    if (value <= 0.0f) {",
                "        return 0;",
                "    }",
                f"    if (value >= (float){max_name}) {{",
                f"        return {max_name};",
                "    }",
                f"    return ({dtype_info.c_type})value;",
                "}",
            ]
        )
        return _GeneratedScalar(lines=lines, deps=set(), includes=includes)
    lines.extend(
        [
            f"static inline {dtype_info.c_type} {dtype_info.prefix}from_f32(float value) {{",
            "    if (!isfinite(value)) {",
            "        return 0;",
            "    }",
            f"    return ({dtype_info.c_type})value;",
            "}",
        ]
    )
    return _GeneratedScalar(lines=lines, deps=set(), includes=includes)


def _int_unary_from_f32(dtype_info: _ScalarTypeInfo, name: str) -> _GeneratedScalar:
    lines = [
        f"static inline {dtype_info.c_type} {dtype_info.prefix}{name}({dtype_info.c_type} a) {{",
        f"    return {dtype_info.prefix}from_f32(ref_scalar_f32_{name}((float)a));",
        "}",
    ]
    deps = {
        _conversion_key_from_alias(dtype_info, "from_f32"),
        _scalar_key_from_op(_SCALAR_TYPES[ScalarType.F32], name),
    }
    return _GeneratedScalar(lines=lines, deps=deps, includes=set())


def _int_binary_from_f32(dtype_info: _ScalarTypeInfo, name: str) -> _GeneratedScalar:
    lines = [
        f"static inline {dtype_info.c_type} {dtype_info.prefix}{name}({dtype_info.c_type} a, {dtype_info.c_type} b) {{",
        f"    return {dtype_info.prefix}from_f32(ref_scalar_f32_{name}((float)a, (float)b));",
        "}",
    ]
    deps = {
        _conversion_key_from_alias(dtype_info, "from_f32"),
        _scalar_key_from_op(_SCALAR_TYPES[ScalarType.F32], name),
    }
    return _GeneratedScalar(lines=lines, deps=deps, includes=set())


def _int_bool_literal(dtype_info: _ScalarTypeInfo, value: bool) -> str:
    if dtype_info.is_small_int:
        return f"({dtype_info.c_type}){1 if value else 0}"
    return "1" if value else "0"


def _int_binary_op(dtype_info: _ScalarTypeInfo, name: str, op: str) -> _GeneratedScalar:
    expr = _cast_value(f"a {op} b", dtype_info)
    return _simple_binary(dtype_info, name, expr)


def _int_unary_op(dtype_info: _ScalarTypeInfo, name: str, expr: str) -> _GeneratedScalar:
    return _simple_unary(dtype_info, name, _cast_value(expr, dtype_info))


def _int_comparison(dtype_info: _ScalarTypeInfo, name: str, op: str) -> _GeneratedScalar:
    one = _int_bool_literal(dtype_info, True)
    zero = _int_bool_literal(dtype_info, False)
    return _simple_binary(dtype_info, name, f"a {op} b ? {one} : {zero}")


def _int_logical(dtype_info: _ScalarTypeInfo, name: str, expr: str) -> _GeneratedScalar:
    one = _int_bool_literal(dtype_info, True)
    zero = _int_bool_literal(dtype_info, False)
    return _simple_binary(dtype_info, name, f"{expr} ? {one} : {zero}")


def _int_logical_not(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    zero = _int_bool_literal(dtype_info, False)
    one = _int_bool_literal(dtype_info, True)
    return _simple_unary(dtype_info, "logical_not", f"a == {zero} ? {one} : {zero}")


def _int_abs(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    if not dtype_info.is_signed:
        return _simple_unary(dtype_info, "abs", "a")
    min_name = f"INT{dtype_info.bits}_MIN"
    includes = {"#include <limits.h>"}
    lines = [
        f"static inline {dtype_info.c_type} {dtype_info.prefix}abs({dtype_info.c_type} a) {{",
        f"    if (a == {min_name}) {{",
        f"        return {min_name};",
        "    }",
        "    return a < 0 ? -a : a;",
        "}",
    ]
    return _GeneratedScalar(lines=lines, deps=set(), includes=includes)


def _int_absolute(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    lines = [
        f"static inline {dtype_info.c_type} {dtype_info.prefix}absolute({dtype_info.c_type} a) {{",
        f"    return {dtype_info.prefix}abs(a);",
        "}",
    ]
    deps = {_scalar_key_from_op(dtype_info, "abs")}
    return _GeneratedScalar(lines=lines, deps=deps, includes=set())


def _int_div(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    expr = _cast_value("a / b", dtype_info)
    lines = [
        f"static inline {dtype_info.c_type} {dtype_info.prefix}div({dtype_info.c_type} a, {dtype_info.c_type} b) {{",
        "    if (b == 0) {",
        "        return 0;",
        "    }",
        f"    return {expr};",
        "}",
    ]
    return _GeneratedScalar(lines=lines, deps=set(), includes=set())


def _int_fmod(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    expr = _cast_value("a % b", dtype_info)
    lines = [
        f"static inline {dtype_info.c_type} {dtype_info.prefix}fmod({dtype_info.c_type} a, {dtype_info.c_type} b) {{",
        "    if (b == 0) {",
        "        return 0;",
        "    }",
        f"    return {expr};",
        "}",
    ]
    return _GeneratedScalar(lines=lines, deps=set(), includes=set())


def _int_remainder(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    expr = _cast_value("a % b", dtype_info)
    lines = [
        f"static inline {dtype_info.c_type} {dtype_info.prefix}remainder({dtype_info.c_type} a, {dtype_info.c_type} b) {{",
        "    if (b == 0) {",
        "        return 0;",
        "    }",
        f"    {dtype_info.c_type} mod = {expr};",
        "    if (mod == 0) {",
        "        return mod;",
        "    }",
        "    if ((mod < 0) != (b < 0)) {",
        "        mod += b;",
        "    }",
        "    return mod;",
        "}",
    ]
    return _GeneratedScalar(lines=lines, deps=set(), includes=set())


def _int_floor_divide(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    expr = _cast_value("a / b", dtype_info)
    lines = [
        f"static inline {dtype_info.c_type} {dtype_info.prefix}floor_divide({dtype_info.c_type} a, {dtype_info.c_type} b) {{",
        "    if (b == 0) {",
        "        return 0;",
        "    }",
    ]
    if dtype_info.is_signed:
        lines.extend(
            [
                f"    {dtype_info.c_type} quo = a / b;",
                f"    {dtype_info.c_type} rem = a % b;",
                "    if (rem != 0 && ((rem < 0) != (b < 0))) {",
                "        quo -= 1;",
                "    }",
                "    return quo;",
                "}",
            ]
        )
    else:
        lines.extend(
            [
                f"    return {expr};",
                "}",
            ]
        )
    return _GeneratedScalar(lines=lines, deps=set(), includes=set())


def _int_copysign(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    lines = [
        f"static inline {dtype_info.c_type} {dtype_info.prefix}copysign({dtype_info.c_type} a, {dtype_info.c_type} b) {{",
        f"    {dtype_info.c_type} magnitude = {dtype_info.prefix}abs(a);",
        "    return b < 0 ? -magnitude : magnitude;",
        "}",
    ]
    deps = {_scalar_key_from_op(dtype_info, "abs")}
    return _GeneratedScalar(lines=lines, deps=deps, includes=set())


def _int_neg(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    if dtype_info.is_signed:
        min_name = f"INT{dtype_info.bits}_MIN"
        includes = {"#include <limits.h>"}
        lines = [
            f"static inline {dtype_info.c_type} {dtype_info.prefix}neg({dtype_info.c_type} a) {{",
            f"    if (a == {min_name}) {{",
            f"        return {min_name};",
            "    }",
            "    return -a;",
            "}",
        ]
        return _GeneratedScalar(lines=lines, deps=set(), includes=includes)
    expr = _cast_value("0 - a", dtype_info)
    return _simple_unary(dtype_info, "neg", expr)


def _int_reciprocal(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    expr = _cast_value("1 / a", dtype_info)
    lines = [
        f"static inline {dtype_info.c_type} {dtype_info.prefix}reciprocal({dtype_info.c_type} a) {{",
        "    if (a == 0) {",
        "        return 0;",
        "    }",
        f"    return {expr};",
        "}",
    ]
    return _GeneratedScalar(lines=lines, deps=set(), includes=set())


def _int_relu(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    if dtype_info.is_signed:
        return _simple_unary(dtype_info, "relu", "a > 0 ? a : 0")
    return _simple_unary(dtype_info, "relu", "a")


def _int_ceil_floor(dtype_info: _ScalarTypeInfo, name: str) -> _GeneratedScalar:
    return _simple_unary(dtype_info, name, "a")


def _int_round(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    return _simple_unary(dtype_info, "round", "a")


def _int_trunc(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    return _simple_unary(dtype_info, "trunc", "a")


def _int_frac(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    lines = [
        f"static inline {dtype_info.c_type} {dtype_info.prefix}frac({dtype_info.c_type} a) {{",
        "    (void)a;",
        "    return 0;",
        "}",
    ]
    return _GeneratedScalar(lines=lines, deps=set(), includes=set())


def _int_sign(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    if not dtype_info.is_signed:
        return _simple_unary(dtype_info, "sign", "a > 0 ? 1 : 0")
    lines = [
        f"static inline {dtype_info.c_type} {dtype_info.prefix}sign({dtype_info.c_type} a) {{",
        "    if (a > 0) {",
        "        return 1;",
        "    }",
        "    if (a < 0) {",
        "        return -1;",
        "    }",
        "    return 0;",
        "}",
    ]
    return _GeneratedScalar(lines=lines, deps=set(), includes=set())


def _int_conj(dtype_info: _ScalarTypeInfo, name: str) -> _GeneratedScalar:
    return _simple_unary(dtype_info, name, "a")


def _int_positive(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    return _simple_unary(dtype_info, "positive", "a")


def _int_sgn(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    if not dtype_info.is_signed:
        return _simple_unary(dtype_info, "sgn", "a > 0 ? 1 : 0")
    lines = [
        f"static inline {dtype_info.c_type} {dtype_info.prefix}sgn({dtype_info.c_type} a) {{",
        "    if (a > 0) {",
        "        return 1;",
        "    }",
        "    if (a < 0) {",
        "        return -1;",
        "    }",
        "    return 0;",
        "}",
    ]
    return _GeneratedScalar(lines=lines, deps=set(), includes=set())


def _int_square(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    return _simple_unary(dtype_info, "square", _cast_value("a * a", dtype_info))

def _int_binary_op_handler(name: str, op: str) -> Callable[[_ScalarTypeInfo], _GeneratedScalar]:
    def handler(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
        return _int_binary_op(dtype_info, name, op)

    return handler


def _int_simple_binary_handler(name: str, expr: str) -> Callable[[_ScalarTypeInfo], _GeneratedScalar]:
    def handler(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
        return _simple_binary(dtype_info, name, expr)

    return handler


def _int_comparison_handler(name: str, op: str) -> Callable[[_ScalarTypeInfo], _GeneratedScalar]:
    def handler(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
        return _int_comparison(dtype_info, name, op)

    return handler


def _int_logical_handler(name: str, expr: str) -> Callable[[_ScalarTypeInfo], _GeneratedScalar]:
    def handler(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
        return _int_logical(dtype_info, name, expr)

    return handler


_INT_OP_DISPATCH: Mapping[str, Callable[[_ScalarTypeInfo], _GeneratedScalar]] = {
    "abs": _int_abs,
    "absolute": _int_absolute,
    "add": _int_binary_op_handler("add", "+"),
    "sub": _int_binary_op_handler("sub", "-"),
    "mul": _int_binary_op_handler("mul", "*"),
    "bitwise_and": _int_binary_op_handler("bitwise_and", "&"),
    "bitwise_or": _int_binary_op_handler("bitwise_or", "|"),
    "bitwise_xor": _int_binary_op_handler("bitwise_xor", "^"),
    "bitwise_left_shift": _int_binary_op_handler("bitwise_left_shift", "<<"),
    "bitwise_right_shift": _int_binary_op_handler("bitwise_right_shift", ">>"),
    "bitwise_not": lambda dtype_info: _int_unary_op(dtype_info, "bitwise_not", "~a"),
    "div": _int_div,
    "maximum": _int_simple_binary_handler("maximum", "a > b ? a : b"),
    "minimum": _int_simple_binary_handler("minimum", "a < b ? a : b"),
    "le": _int_comparison_handler("le", "<="),
    "lt": _int_comparison_handler("lt", "<"),
    "ge": _int_comparison_handler("ge", ">="),
    "gt": _int_comparison_handler("gt", ">"),
    "eq": _int_comparison_handler("eq", "=="),
    "ne": _int_comparison_handler("ne", "!="),
    "logical_or": _int_logical_handler("logical_or", "(a != 0 || b != 0)"),
    "logical_and": _int_logical_handler("logical_and", "(a != 0 && b != 0)"),
    "logical_xor": _int_logical_handler("logical_xor", "((a != 0) != (b != 0))"),
    "logical_not": _int_logical_not,
    "fmax": _int_simple_binary_handler("fmax", "a > b ? a : b"),
    "fmin": _int_simple_binary_handler("fmin", "a < b ? a : b"),
    "copysign": _int_copysign,
    "fmod": _int_fmod,
    "remainder": _int_remainder,
    "floor_divide": _int_floor_divide,
    "clamp_min": _int_simple_binary_handler("clamp_min", "a > b ? a : b"),
    "clamp_max": _int_simple_binary_handler("clamp_max", "a < b ? a : b"),
    "neg": _int_neg,
    "reciprocal": _int_reciprocal,
    "relu": _int_relu,
    "ceil": lambda dtype_info: _int_ceil_floor(dtype_info, "ceil"),
    "floor": lambda dtype_info: _int_ceil_floor(dtype_info, "floor"),
    "round": _int_round,
    "trunc": _int_trunc,
    "frac": _int_frac,
    "sign": _int_sign,
    "conj": lambda dtype_info: _int_conj(dtype_info, "conj"),
    "conj_physical": lambda dtype_info: _int_conj(dtype_info, "conj_physical"),
    "positive": _int_positive,
    "real": lambda dtype_info: _simple_unary(dtype_info, "real", "a"),
    "sgn": _int_sgn,
    "square": _int_square,
}


def _int_from_ops(dtype_info: _ScalarTypeInfo, name: str) -> _GeneratedScalar:
    canonical_name = _normalize_op_name(name)
    if canonical_name != name:
        lines = [
            f"static inline {dtype_info.c_type} {dtype_info.prefix}{name}({dtype_info.c_type} a) {{",
            f"    return {dtype_info.prefix}{canonical_name}(a);",
            "}",
        ]
        deps = {_scalar_key_from_op(dtype_info, canonical_name)}
        return _GeneratedScalar(lines=lines, deps=deps, includes=set())
    name = canonical_name
    if name == "from_f32":
        return _int_from_f32(dtype_info)
    handler = _INT_OP_DISPATCH.get(name)
    if handler is not None:
        return handler(dtype_info)
    function = ScalarFunction.from_op_name(name)
    if function.int_from_f32_arity == 1:
        return _int_unary_from_f32(dtype_info, name)
    if function.int_from_f32_arity == 2:
        return _int_binary_from_f32(dtype_info, name)
    raise ScalarFunctionError(f"unsupported int scalar op: {name}")


def _bool_to_f32() -> _GeneratedScalar:
    lines = [
        "static inline float ref_scalar_bool_to_f32(bool value) {",
        "    return value ? 1.0f : 0.0f;",
        "}",
    ]
    return _GeneratedScalar(lines=lines, deps=set(), includes=set())


def _bool_from_f32() -> _GeneratedScalar:
    lines = [
        "static inline bool ref_scalar_bool_from_f32(float value) {",
        "    return value != 0.0f;",
        "}",
    ]
    return _GeneratedScalar(lines=lines, deps=set(), includes=set())


def _bool_bitwise(dtype_info: _ScalarTypeInfo, name: str, expr: str) -> _GeneratedScalar:
    return _simple_binary(dtype_info, name, expr)


def _bool_bitwise_not(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    return _simple_unary(dtype_info, "bitwise_not", "!a")


def _bool_logical(dtype_info: _ScalarTypeInfo, name: str, expr: str) -> _GeneratedScalar:
    return _simple_binary(dtype_info, name, expr)


def _bool_logical_not(dtype_info: _ScalarTypeInfo) -> _GeneratedScalar:
    return _simple_unary(dtype_info, "logical_not", "!a")


def _bool_comparison(dtype_info: _ScalarTypeInfo, name: str, op: str) -> _GeneratedScalar:
    return _simple_binary(dtype_info, name, f"a {op} b")


def _bool_unary_from_f32(name: str) -> _GeneratedScalar:
    lines = [
        f"static inline bool ref_scalar_bool_{name}(bool a) {{",
        f"    return ref_scalar_bool_from_f32(ref_scalar_f32_{name}(ref_scalar_bool_to_f32(a)));",
        "}",
    ]
    bool_info = _SCALAR_TYPES[ScalarType.BOOL]
    deps = {
        _conversion_key_from_alias(bool_info, "from_f32"),
        _conversion_key_from_alias(bool_info, "to_f32"),
        _scalar_key_from_op(_SCALAR_TYPES[ScalarType.F32], name),
    }
    return _GeneratedScalar(lines=lines, deps=deps, includes=set())


def _bool_binary_from_f32(name: str) -> _GeneratedScalar:
    lines = [
        f"static inline bool ref_scalar_bool_{name}(bool a, bool b) {{",
        "    return ref_scalar_bool_from_f32(",
        f"        ref_scalar_f32_{name}(ref_scalar_bool_to_f32(a), ref_scalar_bool_to_f32(b))",
        "    );",
        "}",
    ]
    bool_info = _SCALAR_TYPES[ScalarType.BOOL]
    deps = {
        _conversion_key_from_alias(bool_info, "from_f32"),
        _conversion_key_from_alias(bool_info, "to_f32"),
        _scalar_key_from_op(_SCALAR_TYPES[ScalarType.F32], name),
    }
    return _GeneratedScalar(lines=lines, deps=deps, includes=set())


_BOOL_OP_DISPATCH: Mapping[str, Callable[[], _GeneratedScalar]] = {
    "to_f32": _bool_to_f32,
    "from_f32": _bool_from_f32,
    "bitwise_and": lambda: _simple_binary(_SCALAR_TYPES[ScalarType.BOOL], "bitwise_and", "a & b"),
    "bitwise_or": lambda: _simple_binary(_SCALAR_TYPES[ScalarType.BOOL], "bitwise_or", "a | b"),
    "bitwise_xor": lambda: _simple_binary(_SCALAR_TYPES[ScalarType.BOOL], "bitwise_xor", "a ^ b"),
    "bitwise_not": lambda: _bool_bitwise_not(_SCALAR_TYPES[ScalarType.BOOL]),
    "logical_or": lambda: _bool_logical(_SCALAR_TYPES[ScalarType.BOOL], "logical_or", "a || b"),
    "logical_and": lambda: _bool_logical(_SCALAR_TYPES[ScalarType.BOOL], "logical_and", "a && b"),
    "logical_xor": lambda: _bool_logical(_SCALAR_TYPES[ScalarType.BOOL], "logical_xor", "a != b"),
    "logical_not": lambda: _bool_logical_not(_SCALAR_TYPES[ScalarType.BOOL]),
    "le": lambda: _bool_comparison(_SCALAR_TYPES[ScalarType.BOOL], "le", "<="),
    "lt": lambda: _bool_comparison(_SCALAR_TYPES[ScalarType.BOOL], "lt", "<"),
    "ge": lambda: _bool_comparison(_SCALAR_TYPES[ScalarType.BOOL], "ge", ">="),
    "gt": lambda: _bool_comparison(_SCALAR_TYPES[ScalarType.BOOL], "gt", ">"),
    "eq": lambda: _bool_comparison(_SCALAR_TYPES[ScalarType.BOOL], "eq", "=="),
    "ne": lambda: _bool_comparison(_SCALAR_TYPES[ScalarType.BOOL], "ne", "!="),
}


def _bool_from_ops(name: str) -> _GeneratedScalar:
    canonical_name = _normalize_op_name(name)
    if canonical_name != name:
        dtype_info = _SCALAR_TYPES[ScalarType.BOOL]
        lines = [
            f"static inline {dtype_info.c_type} {dtype_info.prefix}{name}({dtype_info.c_type} a) {{",
            f"    return {dtype_info.prefix}{canonical_name}(a);",
            "}",
        ]
        deps = {_scalar_key_from_op(dtype_info, canonical_name)}
        return _GeneratedScalar(lines=lines, deps=deps, includes=set())
    name = canonical_name
    handler = _BOOL_OP_DISPATCH.get(name)
    if handler is not None:
        return handler()
    function = ScalarFunction.from_op_name(name)
    if function.bool_from_f32_arity == 1:
        return _bool_unary_from_f32(name)
    if function.bool_from_f32_arity == 2:
        return _bool_binary_from_f32(name)
    raise ScalarFunctionError(f"unsupported bool scalar op: {name}")


_SCALAR_TYPES: Dict[ScalarType, _ScalarTypeInfo] = {
    ScalarType.F32: _ScalarTypeInfo(
        scalar_type=ScalarType.F32,
        c_type="float",
        prefix="ref_scalar_f32_",
        suffix="f32",
        is_float=True,
        is_bool=False,
        is_signed=True,
        is_small_int=False,
        bits=None,
    ),
    ScalarType.F64: _ScalarTypeInfo(
        scalar_type=ScalarType.F64,
        c_type="double",
        prefix="ref_scalar_f64_",
        suffix="f64",
        is_float=True,
        is_bool=False,
        is_signed=True,
        is_small_int=False,
        bits=None,
    ),
    ScalarType.I8: _ScalarTypeInfo(
        scalar_type=ScalarType.I8,
        c_type="int8_t",
        prefix="ref_scalar_i8_",
        suffix="i8",
        is_float=False,
        is_bool=False,
        is_signed=True,
        is_small_int=True,
        bits=8,
    ),
    ScalarType.I16: _ScalarTypeInfo(
        scalar_type=ScalarType.I16,
        c_type="int16_t",
        prefix="ref_scalar_i16_",
        suffix="i16",
        is_float=False,
        is_bool=False,
        is_signed=True,
        is_small_int=True,
        bits=16,
    ),
    ScalarType.I32: _ScalarTypeInfo(
        scalar_type=ScalarType.I32,
        c_type="int32_t",
        prefix="ref_scalar_i32_",
        suffix="i32",
        is_float=False,
        is_bool=False,
        is_signed=True,
        is_small_int=False,
        bits=32,
    ),
    ScalarType.I64: _ScalarTypeInfo(
        scalar_type=ScalarType.I64,
        c_type="int64_t",
        prefix="ref_scalar_i64_",
        suffix="i64",
        is_float=False,
        is_bool=False,
        is_signed=True,
        is_small_int=False,
        bits=64,
    ),
    ScalarType.U8: _ScalarTypeInfo(
        scalar_type=ScalarType.U8,
        c_type="uint8_t",
        prefix="ref_scalar_u8_",
        suffix="u8",
        is_float=False,
        is_bool=False,
        is_signed=False,
        is_small_int=True,
        bits=8,
    ),
    ScalarType.U16: _ScalarTypeInfo(
        scalar_type=ScalarType.U16,
        c_type="uint16_t",
        prefix="ref_scalar_u16_",
        suffix="u16",
        is_float=False,
        is_bool=False,
        is_signed=False,
        is_small_int=True,
        bits=16,
    ),
    ScalarType.U32: _ScalarTypeInfo(
        scalar_type=ScalarType.U32,
        c_type="uint32_t",
        prefix="ref_scalar_u32_",
        suffix="u32",
        is_float=False,
        is_bool=False,
        is_signed=False,
        is_small_int=False,
        bits=32,
    ),
    ScalarType.U64: _ScalarTypeInfo(
        scalar_type=ScalarType.U64,
        c_type="uint64_t",
        prefix="ref_scalar_u64_",
        suffix="u64",
        is_float=False,
        is_bool=False,
        is_signed=False,
        is_small_int=False,
        bits=64,
    ),
    ScalarType.BOOL: _ScalarTypeInfo(
        scalar_type=ScalarType.BOOL,
        c_type="bool",
        prefix="ref_scalar_bool_",
        suffix="bool",
        is_float=False,
        is_bool=True,
        is_signed=False,
        is_small_int=False,
        bits=None,
    ),
}


_SCALAR_TYPE_BY_ENUM: Mapping[ScalarType, _ScalarTypeInfo] = _SCALAR_TYPES


_CONVERSION_SOURCE_BY_FUNCTION: Mapping[ScalarFunction, ScalarType] = {
    ScalarFunction.CONVERT_FROM_F32: ScalarType.F32,
    ScalarFunction.CONVERT_FROM_F64: ScalarType.F64,
    ScalarFunction.CONVERT_FROM_I8: ScalarType.I8,
    ScalarFunction.CONVERT_FROM_I16: ScalarType.I16,
    ScalarFunction.CONVERT_FROM_I32: ScalarType.I32,
    ScalarFunction.CONVERT_FROM_I64: ScalarType.I64,
    ScalarFunction.CONVERT_FROM_U8: ScalarType.U8,
    ScalarFunction.CONVERT_FROM_U16: ScalarType.U16,
    ScalarFunction.CONVERT_FROM_U32: ScalarType.U32,
    ScalarFunction.CONVERT_FROM_U64: ScalarType.U64,
    ScalarFunction.CONVERT_FROM_BOOL: ScalarType.BOOL,
}


def _scalar_type_info(dtype: ScalarType) -> _ScalarTypeInfo:
    try:
        return _SCALAR_TYPE_BY_ENUM[dtype]
    except KeyError as exc:
        raise ScalarFunctionError(
            f"unsupported scalar dtype: {dtype.value}"
        ) from exc


def _supported_ops(dtype_info: _ScalarTypeInfo) -> Set[str]:
    supported = {
        _normalize_op_name(function.value)
        for function in ScalarFunction
        if not function.value.startswith("convert_from_")
        and function.supports_dtype(dtype_info)
    }
    if not dtype_info.is_float:
        supported.add("from_f32")
    if dtype_info.is_bool:
        supported.add("to_f32")
    return supported


def validate_scalar_function_supported_ops() -> None:
    scalar_ops = {
        _normalize_op_name(function.value)
        for function in ScalarFunction
        if not function.value.startswith("convert_from_")
    }
    conversion_aliases = {"from_f32", "to_f32"}
    categories = {
        "float": _supported_ops(_SCALAR_TYPES[ScalarType.F32]),
        "bool": _supported_ops(_SCALAR_TYPES[ScalarType.BOOL]),
        "signed_int": _supported_ops(_SCALAR_TYPES[ScalarType.I8]),
        "unsigned_int": _supported_ops(_SCALAR_TYPES[ScalarType.U8]),
    }
    errors: List[str] = []
    for category, supported in categories.items():
        missing = sorted(supported - scalar_ops - conversion_aliases)
        if missing:
            errors.append(
                f"{category} missing ScalarFunction ops (defined in _supported_ops): {missing}"
            )
    supported_union = set().union(*categories.values()) - conversion_aliases
    unexpected_extras = sorted(scalar_ops - supported_union)
    if unexpected_extras:
        errors.append(
            "ScalarFunction ops not supported by any dtype category: "
            f"{unexpected_extras}"
        )
    if errors:
        raise AssertionError(
            "ScalarFunction/_supported_ops drift detected:\n" + "\n".join(errors)
        )


def _scalar_info_for_key(key: ScalarFunctionKey) -> tuple[_ScalarTypeInfo, str]:
    if key.function in _CONVERSION_SOURCE_BY_FUNCTION:
        source_type = _CONVERSION_SOURCE_BY_FUNCTION[key.function]
        if source_type == ScalarType.F32:
            return _scalar_type_info(key.return_type), "from_f32"
        if source_type == ScalarType.BOOL:
            if key.return_type != ScalarType.F32:
                raise ScalarFunctionError(
                    f"unsupported scalar conversion from {source_type.value} to {key.return_type.value}"
                )
            return _scalar_type_info(source_type), "to_f32"
        raise ScalarFunctionError(
            f"unsupported scalar conversion from {source_type.value} to {key.return_type.value}"
        )
    return _scalar_type_info(key.return_type), key.function.value


def _generate_scalar(key: ScalarFunctionKey) -> _GeneratedScalar:
    dtype_info, op_name = _scalar_info_for_key(key)
    if _normalize_op_name(op_name) not in _supported_ops(dtype_info):
        raise ScalarFunctionError(
            f"unsupported scalar op {op_name} for {dtype_info.suffix}"
        )
    if dtype_info.is_float:
        param_handler = _PARAMETERIZED_FLOAT_OPS.get(key.function)
        if param_handler is not None:
            generated = param_handler(
                dtype_info, key.params, _function_name_for_key(key)
            )
        else:
            generated = _float_from_ops(dtype_info, op_name)
    elif dtype_info.is_bool:
        generated = _bool_from_ops(op_name)
    else:
        generated = _int_from_ops(dtype_info, op_name)
    includes = set(generated.includes)
    if dtype_info.is_float:
        includes.update({"#include <math.h>", "#include <float.h>"})
    if not dtype_info.is_float and not dtype_info.is_bool:
        includes.update({"#include <stdint.h>"})
        if dtype_info.is_signed:
            includes.add("#include <limits.h>")
    if dtype_info.is_bool:
        includes.add("#include <stdbool.h>")
    return _GeneratedScalar(
        lines=generated.lines,
        deps=generated.deps,
        includes=includes,
        constants=generated.constants,
    )


def _function_name_for_key(key: ScalarFunctionKey) -> str:
    param_suffix = _param_suffix(key.params)
    if key.function in _CONVERSION_SOURCE_BY_FUNCTION:
        source_type = _CONVERSION_SOURCE_BY_FUNCTION[key.function]
        if source_type == ScalarType.F32:
            if key.return_type in {
                ScalarType.I8,
                ScalarType.I16,
                ScalarType.I32,
                ScalarType.I64,
                ScalarType.U8,
                ScalarType.U16,
                ScalarType.U32,
                ScalarType.U64,
                ScalarType.BOOL,
            }:
                target_info = _scalar_type_info(key.return_type)
                return f"{target_info.prefix}from_f32{param_suffix}"
            raise ScalarFunctionError(
                f"unsupported scalar conversion from {source_type.value} to {key.return_type.value}"
            )
        if source_type == ScalarType.BOOL:
            if key.return_type == ScalarType.F32:
                source_info = _scalar_type_info(source_type)
                return f"{source_info.prefix}to_f32{param_suffix}"
            raise ScalarFunctionError(
                f"unsupported scalar conversion from {source_type.value} to {key.return_type.value}"
            )
        raise ScalarFunctionError(
            f"unsupported scalar conversion from {source_type.value} to {key.return_type.value}"
        )
    op_name = key.function.value
    dtype_info = _scalar_type_info(key.return_type)
    if _normalize_op_name(op_name) not in _supported_ops(dtype_info):
        raise ScalarFunctionError(
            f"unsupported scalar op {op_name} for {dtype_info.suffix}"
        )
    return f"{dtype_info.prefix}{op_name}{param_suffix}"


class ScalarFunctionRegistry:
    def __init__(self) -> None:
        self._requested: List[ScalarFunctionKey] = []
        self._requested_set: Set[ScalarFunctionKey] = set()
        self._key_to_name: Dict[ScalarFunctionKey, str] = {}
        self._generated: Dict[ScalarFunctionKey, _GeneratedScalar] = {}

    def request(self, key: ScalarFunctionKey) -> str:
        name = self._key_to_name.get(key)
        if name is None:
            name = _function_name_for_key(key)
            self._key_to_name[key] = name
        self._register_key(key)
        return name

    def _register_key(self, key: ScalarFunctionKey) -> None:
        if key in self._requested_set:
            return
        self._requested.append(key)
        self._requested_set.add(key)

    def include_lines(self) -> List[str]:
        includes: Set[str] = set()
        visited: Set[ScalarFunctionKey] = set()
        constants: Set[str] = set()

        def collect(key: ScalarFunctionKey) -> None:
            if key in visited:
                return
            self._ensure_generated(key)
            entry = self._generated[key]
            visited.add(key)
            for dep in entry.deps:
                collect(dep)
            includes.update(entry.includes)
            constants.update(entry.constants)

        for key in self._requested:
            collect(key)
        ordered = sorted(includes)
        preamble: List[str] = []
        if "REF_PI_F" in constants:
            preamble.extend(
                [
                    "#ifndef REF_PI_F",
                    "#define REF_PI_F 3.14159265358979323846f",
                    "#endif",
                ]
            )
        if "REF_PI_D" in constants:
            preamble.extend(
                [
                    "#ifndef REF_PI_D",
                    "#define REF_PI_D 3.14159265358979323846",
                    "#endif",
                ]
            )
        return ordered + preamble

    def render(self) -> List[str]:
        if not self._requested:
            return []
        lines: List[str] = []
        emitted: Set[ScalarFunctionKey] = set()

        def emit(key: ScalarFunctionKey) -> None:
            if key in emitted:
                return
            self._ensure_generated(key)
            entry = self._generated[key]
            for dep in sorted(entry.deps, key=_function_name_for_key):
                emit(dep)
            lines.extend(entry.lines)
            lines.append("")
            emitted.add(key)

        for key in self._requested:
            emit(key)
        while lines and lines[-1] == "":
            lines.pop()
        return lines

    def _ensure_generated(self, key: ScalarFunctionKey) -> None:
        if key in self._generated:
            return
        self._generated[key] = _generate_scalar(key)

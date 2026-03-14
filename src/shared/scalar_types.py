from __future__ import annotations

from enum import Enum

import numpy as np


def _bfloat16_numpy_dtype() -> np.dtype:
    try:
        import ml_dtypes

        return np.dtype(ml_dtypes.bfloat16)
    except Exception:
        try:
            return np.dtype("bfloat16")
        except TypeError:
            return np.dtype(np.float32)


def _float8_numpy_dtype(ml_name: str) -> np.dtype:
    """Return the canonical numpy dtype for a float8 type.

    Uses the corresponding ``ml_dtypes`` type when available, falling back to
    ``uint8``.
    """
    try:
        import ml_dtypes

        dtype_cls = getattr(ml_dtypes, ml_name, None)
        if dtype_cls is not None:
            return np.dtype(dtype_cls)
    except ImportError:
        pass
    return np.dtype(np.uint8)


def _subbyte_numpy_dtype(bits: int, *, signed: bool) -> np.dtype:
    """Return the canonical numpy dtype for a sub-byte integer type.

    Uses ``ml_dtypes.int2`` / ``ml_dtypes.uint2`` / ``ml_dtypes.int4`` /
    ``ml_dtypes.uint4`` when available, falling back to ``int8`` / ``uint8``.
    """
    try:
        import ml_dtypes

        name = f"{'int' if signed else 'uint'}{bits}"
        dtype_cls = getattr(ml_dtypes, name, None)
        if dtype_cls is not None:
            return np.dtype(dtype_cls)
    except ImportError:
        pass
    return np.dtype("int8" if signed else "uint8")


class ScalarFunctionError(RuntimeError):
    pass


class ScalarType(str, Enum):
    def __new__(
        cls,
        suffix: str,
        onnx_name: str,
        c_type: str,
        np_dtype: np.dtype,
        zero_literal: str,
        min_literal: str,
        max_literal: str,
        is_float: bool,
        is_signed: bool,
        is_bool: bool,
        bits: int | None,
    ) -> "ScalarType":
        obj = str.__new__(cls, suffix)
        obj._value_ = suffix
        obj.suffix = suffix
        obj.onnx_name = onnx_name
        obj.c_type = c_type
        obj.np_dtype = np.dtype(np_dtype)
        obj.zero_literal = zero_literal
        obj.min_literal = min_literal
        obj.max_literal = max_literal
        obj.is_float = is_float
        obj.is_signed = is_signed
        obj.is_bool = is_bool
        obj.bits = bits
        return obj

    @property
    def onnx_np_dtype(self) -> np.dtype:
        """Canonical numpy dtype for ONNX compatibility.

        For sub-byte integer types (INT2/UINT2/INT4/UINT4) this returns the
        corresponding ``ml_dtypes`` dtype (e.g. ``ml_dtypes.int4``) rather than
        the storage dtype (``int8`` / ``uint8``).  All other types return the
        same value as :attr:`np_dtype`.
        """
        if (
            self.bits is not None
            and self.bits < 8
            and not self.is_float
            and not self.is_bool
        ):
            return _subbyte_numpy_dtype(self.bits, signed=self.is_signed)
        return self.np_dtype

    F16 = (
        "f16",
        "float16",
        "_Float16",
        np.dtype("float16"),
        "0.0f",
        "-INFINITY",
        "INFINITY",
        True,
        True,
        False,
        16,
    )
    BF16 = (
        "bf16",
        "bfloat16",
        "__bf16",
        _bfloat16_numpy_dtype(),
        "0.0f",
        "-INFINITY",
        "INFINITY",
        True,
        True,
        False,
        16,
    )
    F8E4M3FN = (
        "f8e4m3fn",
        "float8e4m3fn",
        "emx_float8e4m3fn_t",
        _float8_numpy_dtype("float8_e4m3fn"),
        "0",
        "0xFEu",
        "0x7Eu",
        True,
        True,
        False,
        8,
    )
    F8E4M3FNUZ = (
        "f8e4m3fnuz",
        "float8e4m3fnuz",
        "emx_float8e4m3fnuz_t",
        _float8_numpy_dtype("float8_e4m3fnuz"),
        "0",
        "0xFFu",
        "0x7Fu",
        True,
        True,
        False,
        8,
    )
    F8E5M2 = (
        "f8e5m2",
        "float8e5m2",
        "emx_float8e5m2_t",
        _float8_numpy_dtype("float8_e5m2"),
        "0",
        "0xFCu",
        "0x7Cu",
        True,
        True,
        False,
        8,
    )
    F8E5M2FNUZ = (
        "f8e5m2fnuz",
        "float8e5m2fnuz",
        "emx_float8e5m2fnuz_t",
        _float8_numpy_dtype("float8_e5m2fnuz"),
        "0",
        "0xFFu",
        "0x7Fu",
        True,
        True,
        False,
        8,
    )
    F8E8M0FNU = (
        "f8e8m0fnu",
        "float8e8m0fnu",
        "emx_float8e8m0fnu_t",
        _float8_numpy_dtype("float8_e8m0fnu"),
        "0",
        "0x00u",
        "0xFEu",
        True,
        False,
        False,
        8,
    )
    F32 = (
        "f32",
        "float",
        "float",
        np.dtype("float32"),
        "0.0f",
        "-INFINITY",
        "INFINITY",
        True,
        True,
        False,
        32,
    )
    F64 = (
        "f64",
        "double",
        "double",
        np.dtype("float64"),
        "0.0",
        "-INFINITY",
        "INFINITY",
        True,
        True,
        False,
        64,
    )
    I2 = (
        "i2",
        "int2",
        "_BitInt(2)",
        np.dtype("int8"),
        "0",
        "-2",
        "1",
        False,
        True,
        False,
        2,
    )
    I4 = (
        "i4",
        "int4",
        "_BitInt(4)",
        np.dtype("int8"),
        "0",
        "-8",
        "7",
        False,
        True,
        False,
        4,
    )
    I8 = (
        "i8",
        "int8",
        "int8_t",
        np.dtype("int8"),
        "0",
        "INT8_MIN",
        "INT8_MAX",
        False,
        True,
        False,
        8,
    )
    I16 = (
        "i16",
        "int16",
        "int16_t",
        np.dtype("int16"),
        "0",
        "INT16_MIN",
        "INT16_MAX",
        False,
        True,
        False,
        16,
    )
    I32 = (
        "i32",
        "int32",
        "int32_t",
        np.dtype("int32"),
        "0",
        "INT32_MIN",
        "INT32_MAX",
        False,
        True,
        False,
        32,
    )
    I64 = (
        "i64",
        "int64",
        "int64_t",
        np.dtype("int64"),
        "0",
        "INT64_MIN",
        "INT64_MAX",
        False,
        True,
        False,
        64,
    )
    U2 = (
        "u2",
        "uint2",
        "unsigned _BitInt(2)",
        np.dtype("uint8"),
        "0",
        "0",
        "3",
        False,
        False,
        False,
        2,
    )
    U4 = (
        "u4",
        "uint4",
        "unsigned _BitInt(4)",
        np.dtype("uint8"),
        "0",
        "0",
        "15",
        False,
        False,
        False,
        4,
    )
    U8 = (
        "u8",
        "uint8",
        "uint8_t",
        np.dtype("uint8"),
        "0",
        "0",
        "UINT8_MAX",
        False,
        False,
        False,
        8,
    )
    U16 = (
        "u16",
        "uint16",
        "uint16_t",
        np.dtype("uint16"),
        "0",
        "0",
        "UINT16_MAX",
        False,
        False,
        False,
        16,
    )
    U32 = (
        "u32",
        "uint32",
        "uint32_t",
        np.dtype("uint32"),
        "0",
        "0",
        "UINT32_MAX",
        False,
        False,
        False,
        32,
    )
    U64 = (
        "u64",
        "uint64",
        "uint64_t",
        np.dtype("uint64"),
        "0",
        "0",
        "UINT64_MAX",
        False,
        False,
        False,
        64,
    )
    BOOL = (
        "bool",
        "bool",
        "bool",
        np.dtype("bool"),
        "false",
        "false",
        "true",
        False,
        False,
        True,
        None,
    )
    STRING = (
        "str",
        "string",
        "char",
        np.dtype("O"),
        '""',
        '""',
        '""',
        False,
        False,
        False,
        None,
    )

    @property
    def is_integer(self) -> bool:
        return not self.is_float and not self.is_bool

    @property
    def is_subbyte(self) -> bool:
        return self.bits is not None and self.bits < 8

    @property
    def is_float8(self) -> bool:
        return self.is_float and self.bits is not None and self.bits == 8

    @classmethod
    def from_torch_dtype(cls, dtype: object) -> "ScalarType":
        if isinstance(dtype, ScalarType):
            return dtype
        if isinstance(dtype, str):
            dtype_name = dtype
        else:
            dtype_name = getattr(dtype, "name", None) or str(dtype)
        normalized = dtype_name.lower()
        if normalized.startswith("torch."):
            normalized = normalized[len("torch.") :]
        mapping = {
            "float16": cls.F16,
            "bfloat16": cls.BF16,
            "float32": cls.F32,
            "float64": cls.F64,
            "int8": cls.I8,
            "int16": cls.I16,
            "int32": cls.I32,
            "int64": cls.I64,
            "uint8": cls.U8,
            "uint16": cls.U16,
            "uint32": cls.U32,
            "uint64": cls.U64,
            "bool": cls.BOOL,
            "string": cls.STRING,
        }
        try:
            return mapping[normalized]
        except KeyError as exc:
            raise ScalarFunctionError(
                f"unsupported dtype for scalar functions: {dtype_name}"
            ) from exc

    @classmethod
    def from_onnx_name(cls, name: str) -> "ScalarType":
        if isinstance(name, ScalarType):
            return name
        mapping = {scalar.onnx_name: scalar for scalar in cls}
        try:
            return mapping[name]
        except KeyError as exc:
            raise ScalarFunctionError(f"unsupported ONNX dtype: {name}") from exc

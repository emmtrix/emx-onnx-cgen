from __future__ import annotations

from enum import Enum

import numpy as np


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
        obj.isfloat = is_float
        obj.is_float = is_float
        obj.is_signed = is_signed
        obj.is_bool = is_bool
        obj.bits = bits
        return obj

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

    @property
    def is_integer(self) -> bool:
        return not self.is_float and not self.is_bool

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

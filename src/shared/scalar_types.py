from __future__ import annotations

from enum import Enum


class ScalarFunctionError(RuntimeError):
    pass


class ScalarType(str, Enum):
    def __new__(cls, suffix: str, c_type: str, is_float: bool) -> "ScalarType":
        obj = str.__new__(cls, suffix)
        obj._value_ = suffix
        obj.suffix = suffix
        obj.c_type = c_type
        obj.isfloat = is_float
        obj.is_float = is_float
        return obj

    F32 = ("f32", "float", True)
    F64 = ("f64", "double", True)
    I8 = ("i8", "int8_t", False)
    I16 = ("i16", "int16_t", False)
    I32 = ("i32", "int32_t", False)
    I64 = ("i64", "int64_t", False)
    U8 = ("u8", "uint8_t", False)
    U16 = ("u16", "uint16_t", False)
    U32 = ("u32", "uint32_t", False)
    U64 = ("u64", "uint64_t", False)
    BOOL = ("bool", "bool", False)

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

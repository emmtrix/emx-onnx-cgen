from __future__ import annotations

import onnx

from shared.scalar_types import ScalarFunctionError, ScalarType

from .errors import UnsupportedOpError

ONNX_TO_SCALAR_TYPE: dict[int, ScalarType] = {
    onnx.TensorProto.FLOAT16: ScalarType.F16,
    onnx.TensorProto.BFLOAT16: ScalarType.BF16,
    onnx.TensorProto.FLOAT8E4M3FN: ScalarType.F8E4M3FN,
    onnx.TensorProto.FLOAT8E4M3FNUZ: ScalarType.F8E4M3FNUZ,
    onnx.TensorProto.FLOAT8E5M2: ScalarType.F8E5M2,
    onnx.TensorProto.FLOAT8E5M2FNUZ: ScalarType.F8E5M2FNUZ,
    onnx.TensorProto.FLOAT8E8M0: ScalarType.F8E8M0FNU,
    onnx.TensorProto.FLOAT: ScalarType.F32,
    onnx.TensorProto.DOUBLE: ScalarType.F64,
    onnx.TensorProto.BOOL: ScalarType.BOOL,
    onnx.TensorProto.UINT2: ScalarType.U2,
    onnx.TensorProto.UINT4: ScalarType.U4,
    onnx.TensorProto.UINT8: ScalarType.U8,
    onnx.TensorProto.UINT16: ScalarType.U16,
    onnx.TensorProto.UINT32: ScalarType.U32,
    onnx.TensorProto.UINT64: ScalarType.U64,
    onnx.TensorProto.INT2: ScalarType.I2,
    onnx.TensorProto.INT4: ScalarType.I4,
    onnx.TensorProto.INT8: ScalarType.I8,
    onnx.TensorProto.INT16: ScalarType.I16,
    onnx.TensorProto.INT32: ScalarType.I32,
    onnx.TensorProto.INT64: ScalarType.I64,
    onnx.TensorProto.STRING: ScalarType.STRING,
}


def scalar_type_from_onnx(elem_type: int) -> ScalarType | None:
    return ONNX_TO_SCALAR_TYPE.get(elem_type)


def dtype_info(dtype: ScalarType | int | str) -> ScalarType:
    if isinstance(dtype, ScalarType):
        return dtype
    if isinstance(dtype, int):
        scalar = scalar_type_from_onnx(dtype)
        if scalar is None:
            raise UnsupportedOpError(f"Unsupported ONNX dtype enum: {dtype}")
        return scalar
    try:
        return ScalarType.from_onnx_name(dtype)
    except ScalarFunctionError:
        raise UnsupportedOpError(f"Unsupported dtype: {dtype}") from None

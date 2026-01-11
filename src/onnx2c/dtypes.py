from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import onnx


@dataclass(frozen=True)
class DTypeInfo:
    name: str
    c_type: str
    np_dtype: np.dtype
    zero_literal: str
    min_literal: str


ONNX_TO_DTYPE = {
    onnx.TensorProto.FLOAT: "float",
    onnx.TensorProto.BOOL: "bool",
    onnx.TensorProto.INT8: "int8",
    onnx.TensorProto.INT16: "int16",
    onnx.TensorProto.INT32: "int32",
    onnx.TensorProto.INT64: "int64",
}


DTYPE_INFO = {
    "float": DTypeInfo(
        name="float",
        c_type="float",
        np_dtype=np.dtype("float32"),
        zero_literal="0.0f",
        min_literal="-INFINITY",
    ),
    "bool": DTypeInfo(
        name="bool",
        c_type="bool",
        np_dtype=np.dtype("bool"),
        zero_literal="false",
        min_literal="false",
    ),
    "int64": DTypeInfo(
        name="int64",
        c_type="int64_t",
        np_dtype=np.dtype("int64"),
        zero_literal="0",
        min_literal="INT64_MIN",
    ),
    "int32": DTypeInfo(
        name="int32",
        c_type="int32_t",
        np_dtype=np.dtype("int32"),
        zero_literal="0",
        min_literal="INT32_MIN",
    ),
    "int16": DTypeInfo(
        name="int16",
        c_type="int16_t",
        np_dtype=np.dtype("int16"),
        zero_literal="0",
        min_literal="INT16_MIN",
    ),
    "int8": DTypeInfo(
        name="int8",
        c_type="int8_t",
        np_dtype=np.dtype("int8"),
        zero_literal="0",
        min_literal="INT8_MIN",
    ),
}


def dtype_info(name: str) -> DTypeInfo:
    return DTYPE_INFO[name]

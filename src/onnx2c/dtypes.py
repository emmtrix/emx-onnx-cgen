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
    max_literal: str


ONNX_TO_DTYPE = {
    onnx.TensorProto.FLOAT16: "float16",
    onnx.TensorProto.FLOAT: "float",
    onnx.TensorProto.DOUBLE: "double",
    onnx.TensorProto.BOOL: "bool",
    onnx.TensorProto.UINT8: "uint8",
    onnx.TensorProto.UINT16: "uint16",
    onnx.TensorProto.UINT32: "uint32",
    onnx.TensorProto.UINT64: "uint64",
    onnx.TensorProto.INT8: "int8",
    onnx.TensorProto.INT16: "int16",
    onnx.TensorProto.INT32: "int32",
    onnx.TensorProto.INT64: "int64",
}


DTYPE_INFO = {
    "float16": DTypeInfo(
        name="float16",
        c_type="_Float16",
        np_dtype=np.dtype("float16"),
        zero_literal="0.0f",
        min_literal="-INFINITY",
        max_literal="INFINITY",
    ),
    "float": DTypeInfo(
        name="float",
        c_type="float",
        np_dtype=np.dtype("float32"),
        zero_literal="0.0f",
        min_literal="-INFINITY",
        max_literal="INFINITY",
    ),
    "double": DTypeInfo(
        name="double",
        c_type="double",
        np_dtype=np.dtype("float64"),
        zero_literal="0.0",
        min_literal="-INFINITY",
        max_literal="INFINITY",
    ),
    "bool": DTypeInfo(
        name="bool",
        c_type="bool",
        np_dtype=np.dtype("bool"),
        zero_literal="false",
        min_literal="false",
        max_literal="true",
    ),
    "uint64": DTypeInfo(
        name="uint64",
        c_type="uint64_t",
        np_dtype=np.dtype("uint64"),
        zero_literal="0",
        min_literal="0",
        max_literal="UINT64_MAX",
    ),
    "uint32": DTypeInfo(
        name="uint32",
        c_type="uint32_t",
        np_dtype=np.dtype("uint32"),
        zero_literal="0",
        min_literal="0",
        max_literal="UINT32_MAX",
    ),
    "uint16": DTypeInfo(
        name="uint16",
        c_type="uint16_t",
        np_dtype=np.dtype("uint16"),
        zero_literal="0",
        min_literal="0",
        max_literal="UINT16_MAX",
    ),
    "uint8": DTypeInfo(
        name="uint8",
        c_type="uint8_t",
        np_dtype=np.dtype("uint8"),
        zero_literal="0",
        min_literal="0",
        max_literal="UINT8_MAX",
    ),
    "int64": DTypeInfo(
        name="int64",
        c_type="int64_t",
        np_dtype=np.dtype("int64"),
        zero_literal="0",
        min_literal="INT64_MIN",
        max_literal="INT64_MAX",
    ),
    "int32": DTypeInfo(
        name="int32",
        c_type="int32_t",
        np_dtype=np.dtype("int32"),
        zero_literal="0",
        min_literal="INT32_MIN",
        max_literal="INT32_MAX",
    ),
    "int16": DTypeInfo(
        name="int16",
        c_type="int16_t",
        np_dtype=np.dtype("int16"),
        zero_literal="0",
        min_literal="INT16_MIN",
        max_literal="INT16_MAX",
    ),
    "int8": DTypeInfo(
        name="int8",
        c_type="int8_t",
        np_dtype=np.dtype("int8"),
        zero_literal="0",
        min_literal="INT8_MIN",
        max_literal="INT8_MAX",
    ),
}


def dtype_info(name: str) -> DTypeInfo:
    return DTYPE_INFO[name]

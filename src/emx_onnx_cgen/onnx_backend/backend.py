from __future__ import annotations

import hashlib
import json
import logging
import os
import struct
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from onnx import ModelProto, TensorProto
from onnx.backend.base import Backend, BackendRep

from emx_onnx_cgen.cli import (
    _decode_image_decoder_inputs,
    _resolve_compiler,
    _serialize_string_tensor,
)
from emx_onnx_cgen.compiler import Compiler, CompilerOptions
from emx_onnx_cgen.ir.model import SequenceType, TensorType, Value
from emx_onnx_cgen.onnx_import import import_onnx
from shared.scalar_types import ScalarType

LOGGER = logging.getLogger(__name__)

_ELEM_TYPE_TO_DTYPE = {
    TensorProto.BOOL: np.dtype("bool"),
    TensorProto.BFLOAT16: ScalarType.BF16.np_dtype,
    TensorProto.DOUBLE: np.dtype("float64"),
    TensorProto.FLOAT: np.dtype("float32"),
    TensorProto.FLOAT16: np.dtype("float16"),
    TensorProto.INT8: np.dtype("int8"),
    TensorProto.INT16: np.dtype("int16"),
    TensorProto.INT32: np.dtype("int32"),
    TensorProto.INT64: np.dtype("int64"),
    TensorProto.STRING: np.dtype("O"),
    TensorProto.UINT8: np.dtype("uint8"),
    TensorProto.UINT16: np.dtype("uint16"),
    TensorProto.UINT32: np.dtype("uint32"),
    TensorProto.UINT64: np.dtype("uint64"),
}


@dataclass(frozen=True)
class _CompiledArtifact:
    executable: Path
    temp_dir: Path


@dataclass(frozen=True)
class _BackendMetadata:
    inputs: tuple[Value, ...]
    outputs: tuple[Value, ...]
    lowered_sequence_input_shapes: dict[str, tuple[int, ...]]


_COMPILE_CACHE: dict[str, _CompiledArtifact] = {}


def _model_hash(model: ModelProto) -> str:
    return hashlib.sha256(model.SerializeToString()).hexdigest()


def _resolve_executable_suffix() -> str:
    return ".exe" if os.name == "nt" else ""


def _resolve_output_infos(model: ModelProto) -> list[tuple[str, np.dtype]]:
    output_infos: list[tuple[str, np.dtype]] = []
    for output in model.graph.output:
        elem_type = output.type.tensor_type.elem_type
        dtype = _ELEM_TYPE_TO_DTYPE.get(elem_type, np.dtype("float32"))
        output_infos.append((output.name, dtype))
    return output_infos


def _decode_value(value: object, *, dtype: np.dtype) -> object:
    if dtype.kind in {"U", "S", "O"}:
        return value
    if isinstance(value, str):
        return float.fromhex(value)
    if isinstance(value, list):
        return [_decode_value(item, dtype=dtype) for item in value]
    return value


def _compile_model(model: ModelProto) -> _CompiledArtifact:
    compiler = _resolve_compiler(None, prefer_ccache=False)
    if compiler is None:
        raise RuntimeError("No C compiler found (set CC environment variable).")
    options = CompilerOptions(
        emit_testbench=True,
        testbench_output_format="json",
    )
    generated = Compiler(options).compile(model)
    temp_dir = Path(tempfile.mkdtemp(prefix="emx_onnx_backend_"))
    c_path = temp_dir / "model.c"
    exe_path = temp_dir / f"model{_resolve_executable_suffix()}"
    c_path.write_text(generated, encoding="utf-8")

    command = [*compiler, "-O2", str(c_path), "-o", str(exe_path), "-lm"]
    result = subprocess.run(
        command,
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"C compilation failed:\n{result.stderr}")
    return _CompiledArtifact(executable=exe_path, temp_dir=temp_dir)


def _serialize_tensor_payload(value: np.ndarray, dtype: np.dtype) -> bytes:
    if dtype.kind in {"U", "S", "O"}:
        return _serialize_string_tensor(np.asarray(value))
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    return array.tobytes(order="C")


def _build_backend_metadata(model: ModelProto) -> _BackendMetadata:
    graph = import_onnx(model)
    compiler = Compiler(CompilerOptions())
    compile_ctx = compiler._build_compile_context(model)
    lowered_sequence_input_shapes = {
        name: tuple(shape)
        for name, shape, value_type in zip(
            compile_ctx.lowered.input_names,
            compile_ctx.lowered.input_shapes,
            graph.inputs,
            strict=True,
        )
        if isinstance(value_type.type, SequenceType)
    }
    return _BackendMetadata(
        inputs=graph.inputs,
        outputs=graph.outputs,
        lowered_sequence_input_shapes=lowered_sequence_input_shapes,
    )


def _non_initializer_input_names(model: ModelProto) -> tuple[str, ...]:
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    return tuple(
        value_info.name
        for value_info in model.graph.input
        if value_info.name not in initializer_names
    )


def _normalize_runtime_inputs(
    model: ModelProto, inputs: tuple[object, ...]
) -> tuple[object, ...]:
    input_names = _non_initializer_input_names(model)
    named_inputs = {
        name: value for name, value in zip(input_names, inputs, strict=True)
    }
    _decode_image_decoder_inputs(model, named_inputs)
    return tuple(named_inputs[name] for name in input_names)


def _sequence_input_items(input_data: object, *, dtype: np.dtype) -> list[np.ndarray]:
    if isinstance(input_data, np.ndarray) and input_data.dtype != np.dtype("O"):
        if input_data.ndim < 1:
            raise TypeError("Sequence input must be at least 1D.")
        return [
            np.asarray(input_data[index]).astype(dtype, copy=False)
            for index in range(int(input_data.shape[0]))
        ]

    if isinstance(input_data, (list, tuple)):
        return [np.asarray(item).astype(dtype, copy=False) for item in input_data]

    array = np.asarray(input_data)
    if array.ndim < 1:
        raise TypeError("Sequence input must be at least 1D.")
    if array.dtype == np.dtype("O"):
        return [np.asarray(item).astype(dtype, copy=False) for item in array.tolist()]
    return [
        np.asarray(array[index]).astype(dtype, copy=False)
        for index in range(int(array.shape[0]))
    ]


def _serialize_runtime_input(
    handle,
    *,
    value: Value,
    input_data: object,
    lowered_sequence_input_shapes: dict[str, tuple[int, ...]],
) -> None:
    if isinstance(value.type, TensorType):
        dtype = value.type.dtype.np_dtype
        if value.type.is_optional:
            present = input_data is not None
            handle.write(struct.pack("<B", 1 if present else 0))
            if not present:
                return
        handle.write(_serialize_tensor_payload(np.asarray(input_data), dtype))
        return

    seq_type = value.type
    seq_dtype = seq_type.elem.dtype.np_dtype
    present = input_data is not None
    if seq_type.is_optional:
        handle.write(struct.pack("<B", 1 if present else 0))
        if not present:
            handle.write(struct.pack("<i", 0))
            elem_shape = tuple(seq_type.elem.shape)
            zero_payload = np.zeros((32, *elem_shape), dtype=seq_dtype)
            handle.write(np.ascontiguousarray(zero_payload).tobytes(order="C"))
            return
    seq_items = _sequence_input_items(input_data, dtype=seq_dtype)
    seq_count = len(seq_items)
    handle.write(struct.pack("<i", seq_count))
    elem_shape = lowered_sequence_input_shapes.get(
        value.name, tuple(seq_type.elem.shape)
    )
    normalized = np.zeros((32, *elem_shape), dtype=seq_dtype)
    limit = min(seq_count, 32)
    for idx in range(limit):
        item = seq_items[idx]
        target = normalized[idx]
        if target.ndim == item.ndim:
            slices = tuple(
                slice(0, min(cur, tgt)) for cur, tgt in zip(item.shape, target.shape)
            )
            if slices:
                target[slices] = item[slices]
            elif item.ndim == 0:
                normalized[idx] = item
            else:
                flat_src = item.reshape(-1)
                flat_dst = target.reshape(-1)
                copy_count = min(flat_src.size, flat_dst.size)
                if copy_count:
                    flat_dst[:copy_count] = flat_src[:copy_count]
        else:
            flat_src = item.reshape(-1)
            flat_dst = target.reshape(-1)
            copy_count = min(flat_src.size, flat_dst.size)
            if copy_count:
                flat_dst[:copy_count] = flat_src[:copy_count]
    handle.write(np.ascontiguousarray(normalized).tobytes(order="C"))


def _decode_runtime_output(value: Value, payload: dict[str, object]) -> object:
    if isinstance(value.type, TensorType):
        dtype = value.type.dtype.np_dtype
        array = np.array(_decode_value(payload["data"], dtype=dtype), dtype=dtype)
        shape = tuple(payload.get("shape", []))
        if shape:
            if array.size == 0:
                shape = tuple(
                    0
                    if index < len(value.type.dim_params)
                    and value.type.dim_params[index] is not None
                    else dim
                    for index, dim in enumerate(shape)
                )
            array = array.reshape(shape)
        elif array.size == 1:
            array = array.reshape(())
        return array

    dtype = value.type.elem.dtype.np_dtype
    shape = tuple(payload.get("shape", []) or value.type.elem.shape)
    sequence_count = int(payload.get("sequence_count", 0))
    array = np.array(_decode_value(payload["data"], dtype=dtype), dtype=dtype)
    if shape:
        array = array.reshape((-1, *shape))
    else:
        array = array.reshape((-1,))
    return [array[index] for index in range(sequence_count)]


class EmxOnnxCgenBackendRep(BackendRep):
    def __init__(
        self,
        *,
        executable: Path,
        metadata: _BackendMetadata,
        model: ModelProto,
    ) -> None:
        self._executable = executable
        self._metadata = metadata
        self._model = ModelProto()
        self._model.CopyFrom(model)

    def run(self, inputs, **kwargs):
        normalized_inputs = _normalize_runtime_inputs(self._model, tuple(inputs))
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as handle:
            input_path = Path(handle.name)
            for input_value, value in zip(
                normalized_inputs, self._metadata.inputs, strict=True
            ):
                _serialize_runtime_input(
                    handle,
                    value=value,
                    input_data=input_value,
                    lowered_sequence_input_shapes=self._metadata.lowered_sequence_input_shapes,
                )
        try:
            result = subprocess.run(
                [str(self._executable), str(input_path)],
                capture_output=True,
                check=False,
                encoding="utf-8",
                text=True,
                timeout=60,
            )
        finally:
            input_path.unlink(missing_ok=True)
        if result.returncode != 0:
            raise RuntimeError(f"Execution failed:\n{result.stderr}")
        payload = json.loads(result.stdout)
        outputs = payload.get("outputs", {})
        decoded: list[object] = []
        output_keys = list(outputs.keys())
        for index, value in enumerate(self._metadata.outputs):
            key = value.name if value.name in outputs else output_keys[index]
            decoded.append(_decode_runtime_output(value, outputs[key]))
        return decoded


class EmxOnnxCgenBackend(Backend):
    @classmethod
    def is_compatible(cls, model, device="CPU", **kwargs):
        return device == "CPU"

    @classmethod
    def prepare(cls, model, device="CPU", **kwargs):
        if device != "CPU":
            raise RuntimeError(f"Unsupported device: {device}")
        digest = _model_hash(model)
        artifact = _COMPILE_CACHE.get(digest)
        if artifact is None:
            artifact = _compile_model(model)
            _COMPILE_CACHE[digest] = artifact
            LOGGER.info("Compiled ONNX backend test model into %s", artifact.executable)
        return EmxOnnxCgenBackendRep(
            executable=artifact.executable,
            metadata=_build_backend_metadata(model),
            model=model,
        )

    @classmethod
    def run_model(cls, model, inputs, device="CPU", **kwargs):
        return cls.prepare(model, device=device, **kwargs).run(inputs, **kwargs)

    @classmethod
    def supports_device(cls, device):
        return device == "CPU"


prepare = EmxOnnxCgenBackend.prepare
run_model = EmxOnnxCgenBackend.run_model
supports_device = EmxOnnxCgenBackend.supports_device
backend_name = "emx-onnx-cgen"

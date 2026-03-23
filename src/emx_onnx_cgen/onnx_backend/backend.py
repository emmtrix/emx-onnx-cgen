from __future__ import annotations

import json
import logging
import os
import struct
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from onnx import ModelProto
from onnx.backend.base import Backend, BackendRep

from emx_onnx_cgen.cli import (
    _decode_image_decoder_inputs,
    _resolve_compiler,
    _serialize_string_tensor,
    _tensor_runtime_dim_values,
)
from emx_onnx_cgen.compiler import Compiler, CompilerOptions
from emx_onnx_cgen.ir.model import SequenceType, TensorType, Value
from emx_onnx_cgen.onnx_import import import_onnx
from emx_onnx_cgen.sequence_shape_hints import (
    SequenceElementDimHint,
    SequenceElementShapeHint,
    validate_runtime_shape,
)

LOGGER = logging.getLogger(__name__)
_SEQUENCE_MAX_LEN = 32


@dataclass(frozen=True)
class _CompiledArtifact:
    executable: Path
    temp_dir: tempfile.TemporaryDirectory[str]


@dataclass(frozen=True)
class _BackendMetadata:
    inputs: tuple[Value, ...]
    outputs: tuple[Value, ...]
    lowered_sequence_input_shapes: dict[str, tuple[int, ...]]
    sequence_element_shapes: dict[str, SequenceElementShapeHint]


def _resolve_executable_suffix() -> str:
    return ".exe" if os.name == "nt" else ""


def _decode_value(value: object, *, dtype: np.dtype) -> object:
    if dtype.kind in {"U", "S", "O"}:
        return value
    if isinstance(value, str):
        return float.fromhex(value)
    if isinstance(value, list):
        return [_decode_value(item, dtype=dtype) for item in value]
    return value


def _compile_model(
    model: ModelProto,
    *,
    sequence_element_shapes: dict[str, SequenceElementShapeHint] | None = None,
) -> _CompiledArtifact:
    compiler = _resolve_compiler(None, prefer_ccache=False)
    if compiler is None:
        raise RuntimeError("No C compiler found (set CC environment variable).")
    options = CompilerOptions(
        emit_testbench=True,
        testbench_output_format="json",
        sequence_element_shapes=sequence_element_shapes,
    )
    generated = Compiler(options).compile(model)
    temp_dir = tempfile.TemporaryDirectory(prefix="emx_onnx_backend_")
    temp_dir_path = Path(temp_dir.name)
    c_path = temp_dir_path / "model.c"
    exe_path = temp_dir_path / f"model{_resolve_executable_suffix()}"
    c_path.write_text(generated, encoding="utf-8")

    command = [*compiler, "-O2", str(c_path), "-o", str(exe_path), "-lm"]
    result = subprocess.run(
        command,
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode != 0:
        temp_dir.cleanup()
        raise RuntimeError(f"C compilation failed:\n{result.stderr}")
    return _CompiledArtifact(executable=exe_path, temp_dir=temp_dir)


def _serialize_tensor_payload(value: np.ndarray, dtype: np.dtype) -> bytes:
    if dtype.kind in {"U", "S", "O"}:
        return _serialize_string_tensor(np.asarray(value))
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    return array.tobytes(order="C")


def _build_backend_metadata(
    model: ModelProto,
    *,
    sequence_element_shapes: dict[str, SequenceElementShapeHint] | None = None,
) -> _BackendMetadata:
    graph = import_onnx(model)
    compiler = Compiler(
        CompilerOptions(sequence_element_shapes=sequence_element_shapes)
    )
    compile_ctx = compiler._build_compile_context(model)
    _, input_dim_names, _, _ = compiler._emitter._build_variable_dim_names(
        compile_ctx.lowered,
        compile_ctx.variable_dim_inputs,
        compile_ctx.variable_dim_outputs,
    )
    explicit_hints = dict(sequence_element_shapes or {})
    lowered_sequence_input_shapes = {
        name: (
            explicit_hints[name].max_shape
            if name in explicit_hints
            else tuple(
                (
                    int(input_dim_names.get(index, {}).get(axis, dim).expected_size)
                    if axis in input_dim_names.get(index, {})
                    else int(dim)
                )
                for axis, dim in enumerate(shape)
            )
        )
        for index, (name, shape, value_type) in enumerate(
            zip(
                compile_ctx.lowered.input_names,
                compile_ctx.lowered.input_shapes,
                graph.inputs,
                strict=True,
            )
        )
        if isinstance(value_type.type, SequenceType)
    }
    return _BackendMetadata(
        inputs=graph.inputs,
        outputs=graph.outputs,
        lowered_sequence_input_shapes=lowered_sequence_input_shapes,
        sequence_element_shapes=dict(sequence_element_shapes or {}),
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


def _model_sequence_input_requires_shape_hint(
    model: ModelProto, input_name: str
) -> bool:
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    for value_info in model.graph.input:
        if value_info.name in initializer_names or value_info.name != input_name:
            continue
        if value_info.type.WhichOneof("value") != "sequence_type":
            return False
        elem_type = value_info.type.sequence_type.elem_type
        if elem_type.WhichOneof("value") != "tensor_type":
            return False
        tensor_type = elem_type.tensor_type
        if not tensor_type.HasField("shape"):
            return True
        return any(
            dim.HasField("dim_param") or not dim.HasField("dim_value")
            for dim in tensor_type.shape.dim
        )
    return False


def _infer_sequence_element_shapes(
    model: ModelProto,
    inputs: tuple[object, ...],
    *,
    explicit_hints: dict[str, SequenceElementShapeHint] | None = None,
) -> dict[str, SequenceElementShapeHint]:
    graph = import_onnx(model)
    explicit_hints = dict(explicit_hints or {})
    input_names = _non_initializer_input_names(model)
    named_inputs = {name: value for name, value in zip(input_names, inputs, strict=True)}
    inferred = dict(explicit_hints)
    for value in graph.inputs:
        if not isinstance(value.type, SequenceType):
            continue
        if value.name in inferred:
            continue
        if not _model_sequence_input_requires_shape_hint(model, value.name):
            continue
        input_data = named_inputs.get(value.name)
        seq_dtype = value.type.elem.dtype.np_dtype
        seq_items = (
            [] if input_data is None else _sequence_input_items(input_data, dtype=seq_dtype)
        )
        if not seq_items:
            raise ValueError(
                f"Sequence input {value.name!r} requires explicit ragged-sequence "
                "bounds because the runtime input is empty."
            )
        rank = seq_items[0].ndim
        if any(item.ndim != rank for item in seq_items[1:]):
            raise ValueError(
                f"Sequence input {value.name!r} has mixed tensor ranks, "
                "but the generated backend ABI requires a uniform element rank."
            )
        if value.type.elem.shape and len(value.type.elem.shape) != rank:
            raise ValueError(
                f"Sequence input {value.name!r} has rank {rank}, but the model "
                f"declares rank {len(value.type.elem.shape)}."
            )
        dims: list[SequenceElementDimHint] = []
        for axis in range(rank):
            if axis < len(value.type.elem.shape) and value.type.elem.shape[axis] >= 0:
                dims.append(
                    SequenceElementDimHint(
                        max_size=int(value.type.elem.shape[axis]),
                        is_static=True,
                    )
                )
                continue
            max_size = max(int(item.shape[axis]) for item in seq_items)
            dims.append(
                SequenceElementDimHint(
                    max_size=max_size,
                    is_static=False,
                )
            )
        inferred[value.name] = SequenceElementShapeHint(
            input_name=value.name,
            dims=tuple(dims),
        )
    return inferred


def _sequence_shape_requires_concretization(shape: tuple[int, ...]) -> bool:
    return any(dim <= 0 for dim in shape)


def _fallback_concrete_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(int(dim) if dim > 0 else 1 for dim in shape)


def _copy_array_overlap(dst: np.ndarray, src: np.ndarray) -> None:
    if dst.ndim == src.ndim:
        slices = tuple(
            slice(0, min(cur, tgt)) for cur, tgt in zip(src.shape, dst.shape)
        )
        if slices:
            dst[slices] = src[slices]
            return
        if src.ndim == 0:
            dst[()] = src
            return
    flat_src = src.reshape(-1)
    flat_dst = dst.reshape(-1)
    copy_count = min(flat_src.size, flat_dst.size)
    if copy_count:
        flat_dst[:copy_count] = flat_src[:copy_count]


def _prepare_sequence_input_data(
    *,
    value: Value,
    input_data: object,
    lowered_sequence_input_shapes: dict[str, tuple[int, ...]],
    sequence_element_shapes: dict[str, SequenceElementShapeHint],
) -> tuple[np.ndarray, tuple[int, ...], int, dict[int, list[int]]]:
    seq_type = value.type
    seq_dtype = seq_type.elem.dtype.np_dtype
    seq_items = (
        [] if input_data is None else _sequence_input_items(input_data, dtype=seq_dtype)
    )
    seq_count = len(seq_items)
    if seq_count > _SEQUENCE_MAX_LEN:
        raise ValueError(
            f"Sequence input {value.name!r} has length {seq_count}, "
            f"but the generated ONNX backend ABI supports at most {_SEQUENCE_MAX_LEN} items."
        )

    hint = sequence_element_shapes.get(value.name)
    declared_shape = lowered_sequence_input_shapes.get(value.name, tuple(seq_type.elem.shape))
    if hint is not None:
        elem_shape = hint.max_shape
        dynamic_axes = hint.dynamic_axes
        for index, item in enumerate(seq_items):
            mismatch = validate_runtime_shape(hint, item.shape)
            if mismatch is not None:
                raise ValueError(
                    f"Sequence input {value.name!r} element {index} {mismatch}."
                )
    else:
        elem_shape = _fallback_concrete_shape(declared_shape)
        dynamic_axes = ()
        for index, item in enumerate(seq_items):
            if tuple(int(dim) for dim in item.shape) != tuple(int(dim) for dim in elem_shape):
                raise ValueError(
                    f"Sequence input {value.name!r} element {index} has shape "
                    f"{tuple(int(dim) for dim in item.shape)!r}, expected {elem_shape!r}."
                )

    normalized = np.zeros((_SEQUENCE_MAX_LEN, *elem_shape), dtype=seq_dtype)
    for idx, item in enumerate(seq_items):
        _copy_array_overlap(normalized[idx], item)
    axis_values = {
        axis: [int(item.shape[axis]) for item in seq_items] for axis in dynamic_axes
    }
    return normalized, elem_shape, seq_count, axis_values


def _sequence_outputs_need_shape_hints(outputs: tuple[Value, ...]) -> bool:
    return any(
        isinstance(value.type, SequenceType)
        and _sequence_shape_requires_concretization(tuple(value.type.elem.shape))
        for value in outputs
    )


def _infer_sequence_output_shapes(
    model: ModelProto,
    inputs: tuple[object, ...],
    outputs: tuple[Value, ...],
) -> dict[str, list[tuple[int, ...]]]:
    if not _sequence_outputs_need_shape_hints(outputs):
        return {}
    try:
        from onnx.reference import ReferenceEvaluator

        evaluator = ReferenceEvaluator(model)
        named_inputs = {
            name: value
            for name, value in zip(
                _non_initializer_input_names(model), inputs, strict=True
            )
        }
        reference_outputs = evaluator.run(None, named_inputs)
    except Exception as exc:
        LOGGER.warning("Failed to infer reference sequence output shapes: %s", exc)
        return {}

    shape_hints: dict[str, list[tuple[int, ...]]] = {}
    for value, output in zip(outputs, reference_outputs, strict=True):
        if not isinstance(value.type, SequenceType) or not isinstance(output, list):
            continue
        shape_hints[value.name] = [tuple(np.asarray(item).shape) for item in output]
    return shape_hints


def _serialize_runtime_input(
    handle,
    *,
    value: Value,
    input_data: object,
    lowered_sequence_input_shapes: dict[str, tuple[int, ...]],
    sequence_element_shapes: dict[str, SequenceElementShapeHint],
) -> None:
    if isinstance(value.type, TensorType):
        dtype = value.type.dtype.np_dtype
        array_data = np.asarray(input_data)
        if value.type.is_optional:
            present = input_data is not None
            handle.write(struct.pack("<B", 1 if present else 0))
            if not present:
                return
        runtime_dims = _tensor_runtime_dim_values(value, array_data)
        if runtime_dims:
            handle.write(struct.pack(f"<{len(runtime_dims)}i", *runtime_dims))
        handle.write(_serialize_tensor_payload(array_data, dtype))
        return

    seq_type = value.type
    present = input_data is not None
    normalized, elem_shape, seq_count, axis_values = _prepare_sequence_input_data(
        value=value,
        input_data=input_data,
        lowered_sequence_input_shapes=lowered_sequence_input_shapes,
        sequence_element_shapes=sequence_element_shapes,
    )
    if seq_type.is_optional:
        handle.write(struct.pack("<B", 1 if present else 0))
        if not present:
            handle.write(struct.pack("<i", 0))
            hint = sequence_element_shapes.get(value.name)
            if hint is not None:
                for _axis in hint.dynamic_axes:
                    handle.write(b"")
            zero_payload = np.zeros(
                (_SEQUENCE_MAX_LEN, *elem_shape), dtype=seq_type.elem.dtype.np_dtype
            )
            handle.write(np.ascontiguousarray(zero_payload).tobytes(order="C"))
            return
    handle.write(struct.pack("<i", seq_count))
    hint = sequence_element_shapes.get(value.name)
    if hint is not None:
        for axis in hint.dynamic_axes:
            dims = axis_values.get(axis, [])
            if dims:
                handle.write(struct.pack(f"<{len(dims)}i", *dims))
    handle.write(np.ascontiguousarray(normalized).tobytes(order="C"))


def _decode_runtime_output(
    value: Value,
    payload: dict[str, object],
    *,
    sequence_item_shapes: list[tuple[int, ...]] | None = None,
) -> object:
    if isinstance(value.type, TensorType):
        scalar_type = value.type.dtype
        dtype = scalar_type.np_dtype
        array = np.array(_decode_value(payload["data"], dtype=dtype), dtype=dtype)
        shape = tuple(payload.get("shape", []))
        if shape:
            if array.size == 0:
                shape = tuple(
                    (
                        0
                        if index < len(value.type.dim_params)
                        and value.type.dim_params[index] is not None
                        else dim
                    )
                    for index, dim in enumerate(shape)
                )
            array = array.reshape(shape)
        elif array.size == 1:
            array = array.reshape(())
        onnx_dtype = scalar_type.onnx_np_dtype
        if onnx_dtype != dtype:
            array = array.astype(onnx_dtype)
        return array

    dtype = value.type.elem.dtype.np_dtype
    shape = tuple(payload.get("shape", []) or value.type.elem.shape)
    sequence_count = int(payload.get("sequence_count", 0))
    array = np.array(_decode_value(payload["data"], dtype=dtype), dtype=dtype)
    if shape:
        try:
            array = array.reshape((-1, *shape))
        except ValueError:
            if array.ndim < 1:
                array = array.reshape((-1,))
    elif array.ndim < 1:
        array = array.reshape((-1,))
    payload_item_shapes = payload.get("item_shapes")
    if sequence_item_shapes is None and isinstance(payload_item_shapes, list):
        sequence_item_shapes = [
            tuple(int(dim) for dim in item_shape) for item_shape in payload_item_shapes
        ]
    outputs = [array[index] for index in range(sequence_count)]
    if sequence_item_shapes is None:
        return outputs

    cropped: list[np.ndarray] = []
    for index, item in enumerate(outputs):
        if index >= len(sequence_item_shapes):
            cropped.append(item)
            continue
        target_shape = sequence_item_shapes[index]
        if item.ndim == len(target_shape):
            slices = tuple(slice(0, dim) for dim in target_shape)
            cropped.append(item[slices])
            continue
        if int(np.prod(target_shape, dtype=np.int64)) == int(item.size):
            cropped.append(item.reshape(target_shape))
            continue
        cropped.append(item)
    return cropped


class EmxOnnxCgenBackendRep(BackendRep):
    def __init__(
        self,
        *,
        artifact: _CompiledArtifact,
        metadata: _BackendMetadata,
        model: ModelProto,
    ) -> None:
        self._artifact = artifact
        self._executable = artifact.executable
        self._metadata = metadata
        self._model = ModelProto()
        self._model.CopyFrom(model)

    def run(self, inputs, **kwargs):
        raw_inputs = tuple(inputs)
        normalized_inputs = _normalize_runtime_inputs(self._model, raw_inputs)
        sequence_output_shapes = _infer_sequence_output_shapes(
            self._model, raw_inputs, self._metadata.outputs
        )
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
                    sequence_element_shapes=self._metadata.sequence_element_shapes,
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
            decoded.append(
                _decode_runtime_output(
                    value,
                    outputs[key],
                    sequence_item_shapes=sequence_output_shapes.get(value.name),
                )
            )
        return decoded


class EmxOnnxCgenBackend(Backend):
    @classmethod
    def is_compatible(cls, model, device="CPU", **kwargs):
        return device == "CPU"

    @classmethod
    def prepare(cls, model, device="CPU", **kwargs):
        if device != "CPU":
            raise RuntimeError(f"Unsupported device: {device}")
        sequence_element_shapes = dict(kwargs.pop("sequence_element_shapes", {}) or {})
        artifact = _compile_model(
            model,
            sequence_element_shapes=sequence_element_shapes,
        )
        LOGGER.info("Compiled ONNX backend test model into %s", artifact.executable)
        return EmxOnnxCgenBackendRep(
            artifact=artifact,
            metadata=_build_backend_metadata(
                model,
                sequence_element_shapes=sequence_element_shapes,
            ),
            model=model,
        )

    @classmethod
    def run_model(cls, model, inputs, device="CPU", **kwargs):
        normalized_inputs = _normalize_runtime_inputs(model, tuple(inputs))
        explicit_hints = dict(kwargs.pop("sequence_element_shapes", {}) or {})
        sequence_element_shapes = _infer_sequence_element_shapes(
            model,
            normalized_inputs,
            explicit_hints=explicit_hints,
        )
        rep = cls.prepare(
            model,
            device=device,
            sequence_element_shapes=sequence_element_shapes,
            **kwargs,
        )
        return rep.run(inputs, **kwargs)

    @classmethod
    def supports_device(cls, device):
        return device == "CPU"


prepare = EmxOnnxCgenBackend.prepare
run_model = EmxOnnxCgenBackend.run_model
supports_device = EmxOnnxCgenBackend.supports_device
backend_name = "emx-onnx-cgen"

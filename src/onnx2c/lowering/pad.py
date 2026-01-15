from __future__ import annotations

import numpy as np

from shared.scalar_types import ScalarType

from ..codegen.c_emitter import PadOp
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Initializer, Node
from ..lowering.common import optional_name, value_dtype, value_shape
from ..validation import normalize_axis
from .registry import register_lowering


def _find_initializer(graph: Graph, name: str) -> Initializer | None:
    for initializer in graph.initializers:
        if initializer.name == name:
            return initializer
    return None


def _read_int_initializer(
    graph: Graph,
    name: str,
    node: Node,
    *,
    label: str,
) -> tuple[int, ...] | None:
    initializer = _find_initializer(graph, name)
    if initializer is None:
        return None
    if initializer.type.dtype not in {ScalarType.I64, ScalarType.I32}:
        raise UnsupportedOpError(
            f"Pad {label} input must be int64 or int32"
        )
    if len(initializer.type.shape) != 1:
        raise UnsupportedOpError(f"Pad {label} input must be a 1D tensor")
    values = np.array(initializer.data, dtype=np.int64).reshape(-1)
    return tuple(int(value) for value in values)


def _read_scalar_initializer(
    graph: Graph, name: str, node: Node, *, dtype: ScalarType
) -> float | int | bool | None:
    initializer = _find_initializer(graph, name)
    if initializer is None:
        return None
    if initializer.type.dtype != dtype:
        raise UnsupportedOpError(
            "Pad value input must match input dtype, "
            f"got {initializer.type.dtype.onnx_name}"
        )
    values = np.array(initializer.data).reshape(-1)
    if values.size != 1:
        raise UnsupportedOpError("Pad value input must be a scalar")
    return values.item()


def _normalize_axes(
    axes: tuple[int, ...], input_shape: tuple[int, ...], node: Node
) -> tuple[int, ...]:
    normalized = [normalize_axis(axis, input_shape, node) for axis in axes]
    if len(set(normalized)) != len(normalized):
        raise UnsupportedOpError("Pad axes must be unique")
    return tuple(normalized)


def _default_pad_value(dtype: ScalarType) -> float | int | bool:
    if dtype.is_bool:
        return False
    if dtype.is_float:
        return 0.0
    return 0


def _compute_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides: list[int] = []
    stride = 1
    for dim in reversed(shape):
        strides.append(stride)
        stride *= dim
    return tuple(reversed(strides))


@register_lowering("Pad")
def lower_pad(graph: Graph, node: Node) -> PadOp:
    if not node.inputs or len(node.outputs) != 1:
        raise UnsupportedOpError("Pad must have 1 output")
    input_name = node.inputs[0]
    if not input_name:
        raise UnsupportedOpError("Pad input must be provided")
    input_shape = value_shape(graph, input_name, node)
    output_shape = value_shape(graph, node.outputs[0], node)
    input_dtype = value_dtype(graph, input_name, node)
    output_dtype = value_dtype(graph, node.outputs[0], node)
    if input_dtype != output_dtype:
        raise UnsupportedOpError(
            "Pad expects matching input/output dtypes, "
            f"got {input_dtype.onnx_name} and {output_dtype.onnx_name}"
        )
    mode = node.attrs.get("mode", "constant")
    if isinstance(mode, bytes):
        mode = mode.decode("utf-8")
    if mode not in {"constant", "edge", "reflect", "wrap"}:
        raise UnsupportedOpError(f"Pad mode '{mode}' is not supported")
    pads_name = optional_name(node.inputs, 1)
    pads_attr = node.attrs.get("pads")
    if pads_name and pads_attr:
        raise UnsupportedOpError("Pad pads must be provided via input or attribute")
    pads = None
    pads_input = None
    pads_shape = None
    pads_dtype = None
    if pads_name:
        pads = _read_int_initializer(graph, pads_name, node, label="pads")
        if pads is None:
            pads_shape = value_shape(graph, pads_name, node)
            pads_dtype = value_dtype(graph, pads_name, node)
            if pads_dtype not in {ScalarType.I64, ScalarType.I32}:
                raise UnsupportedOpError(
                    "Pad pads input must be int64 or int32"
                )
            if len(pads_shape) != 1:
                raise UnsupportedOpError("Pad pads input must be a 1D tensor")
            pads_input = pads_name
    elif pads_attr is not None:
        pads = tuple(int(value) for value in pads_attr)
    if pads is None and pads_input is None:
        pads = tuple(0 for _ in range(2 * len(input_shape)))

    axes_name = optional_name(node.inputs, 3)
    axes = None
    if axes_name:
        axes = _read_int_initializer(graph, axes_name, node, label="axes")
        if axes is None:
            raise UnsupportedOpError("Pad axes input must be a constant initializer")
        axes = _normalize_axes(axes, input_shape, node)
    if pads_input is not None and axes is not None:
        raise UnsupportedOpError("Pad axes input must be a constant initializer")

    if axes is None:
        if pads is None:
            if pads_shape is None or pads_shape[0] != 2 * len(input_shape):
                raise ShapeInferenceError(
                    "Pad pads must have length 2 * rank of input"
                )
            pads_begin = None
            pads_end = None
        else:
            if len(pads) != 2 * len(input_shape):
                raise ShapeInferenceError(
                    "Pad pads must have length 2 * rank of input"
                )
            pads_begin = list(pads[: len(input_shape)])
            pads_end = list(pads[len(input_shape) :])
    else:
        if len(pads) != 2 * len(axes):
            raise ShapeInferenceError(
                "Pad pads must have length 2 * len(axes)"
            )
        pads_begin = [0] * len(input_shape)
        pads_end = [0] * len(input_shape)
        for index, axis in enumerate(axes):
            pads_begin[axis] = pads[index]
            pads_end[axis] = pads[index + len(axes)]

    if pads_begin is not None and pads_end is not None:
        if any(value < 0 for value in pads_begin + pads_end):
            raise UnsupportedOpError("Pad pads must be non-negative")

    if pads_begin is not None and pads_end is not None:
        expected_shape = tuple(
            dim + pad_before + pad_after
            for dim, pad_before, pad_after in zip(
                input_shape, pads_begin, pads_end
            )
        )
        if output_shape != expected_shape:
            raise ShapeInferenceError(
                "Pad output shape mismatch: "
                f"expected {expected_shape}, got {output_shape}"
            )

    value_name = optional_name(node.inputs, 2)
    pad_value = None
    value_input = None
    value_input_shape = None
    if value_name:
        pad_value = _read_scalar_initializer(
            graph, value_name, node, dtype=input_dtype
        )
        if pad_value is None:
            value_input_shape = value_shape(graph, value_name, node)
            input_value_dtype = value_dtype(graph, value_name, node)
            if input_value_dtype != input_dtype:
                raise UnsupportedOpError(
                    "Pad value input must match input dtype, "
                    f"got {input_value_dtype.onnx_name}"
                )
            if value_input_shape:
                raise UnsupportedOpError("Pad value input must be a scalar")
            value_input = value_name
    elif "value" in node.attrs:
        pad_value = node.attrs["value"]
    if pad_value is None and value_input is None:
        pad_value = _default_pad_value(input_dtype)

    return PadOp(
        input0=input_name,
        output=node.outputs[0],
        input_shape=input_shape,
        output_shape=output_shape,
        pads_begin=(
            tuple(int(value) for value in pads_begin)
            if pads_begin is not None
            else None
        ),
        pads_end=(
            tuple(int(value) for value in pads_end)
            if pads_end is not None
            else None
        ),
        pads_input=pads_input,
        pads_shape=pads_shape,
        pads_dtype=pads_dtype,
        mode=mode,
        value=pad_value,
        value_input=value_input,
        value_shape=value_input_shape,
        dtype=output_dtype,
        input_dtype=input_dtype,
        input_strides=_compute_strides(input_shape),
    )

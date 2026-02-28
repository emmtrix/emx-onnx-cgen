from __future__ import annotations

from shared.scalar_types import ScalarType

from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Initializer, Node
from ..ir.ops import ResizeOp
from .common import value_dtype, value_shape
from .registry import register_lowering

_SUPPORTED_MODES = {"nearest", "linear"}


def _decode_attr(value: object, default: str) -> str:
    if value is None:
        return default
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, str):
        return value
    return str(value)


def _find_initializer(graph: Graph, name: str) -> Initializer | None:
    for initializer in graph.initializers:
        if initializer.name == name:
            return initializer
    return None


def _load_initializer_values(
    graph: Graph, name: str, node: Node
) -> tuple[float | int, ...] | None:
    initializer = _find_initializer(graph, name)
    if initializer is None:
        return None
    if initializer.type.dtype not in {
        ScalarType.F16,
        ScalarType.F32,
        ScalarType.F64,
    }:
        raise UnsupportedOpError(
            "Upsample scales initializer must be bfloat16/float16/float32/float64"
        )
    data = initializer.data.reshape(-1)
    return tuple(data.tolist())


def _validate_output_shape(
    expected: tuple[int, ...],
    actual: tuple[int, ...],
) -> None:
    if expected != actual:
        raise ShapeInferenceError(
            f"Upsample output shape must be {expected}, got {actual}"
        )
    if any(dim < 0 for dim in actual):
        raise ShapeInferenceError("Upsample output shape must be non-negative")


@register_lowering("Upsample")
def lower_upsample(graph: Graph, node: Node) -> ResizeOp:
    if len(node.outputs) != 1:
        raise UnsupportedOpError("Upsample expects one output")
    if len(node.inputs) not in {1, 2}:
        raise UnsupportedOpError("Upsample expects 1 or 2 inputs")
    mode = _decode_attr(node.attrs.get("mode"), "nearest")
    if mode not in _SUPPORTED_MODES:
        raise UnsupportedOpError(f"Upsample mode {mode!r} is not supported")
    input_name = node.inputs[0]
    output_name = node.outputs[0]
    input_shape = value_shape(graph, input_name, node)
    output_shape = value_shape(graph, output_name, node)
    input_dtype = value_dtype(graph, input_name, node)
    output_dtype = value_dtype(graph, output_name, node)
    if input_dtype != output_dtype:
        raise UnsupportedOpError(
            "Upsample expects matching input/output dtypes, "
            f"got {input_dtype.onnx_name} and {output_dtype.onnx_name}"
        )
    rank = len(input_shape)
    axes = tuple(range(rank))
    scales_input = None
    scales: tuple[float, ...]
    if len(node.inputs) == 2 and node.inputs[1]:
        scales_input = node.inputs[1]
        scales_shape = value_shape(graph, scales_input, node)
        if len(scales_shape) != 1:
            raise UnsupportedOpError("Upsample expects scales to be 1D")
        if scales_shape[0] != rank:
            raise UnsupportedOpError("Upsample scales length mismatch")
        if value_dtype(graph, scales_input, node) not in {
            ScalarType.F16,
            ScalarType.BF16,
            ScalarType.F32,
            ScalarType.F64,
        }:
            raise UnsupportedOpError(
                "Upsample expects scales input to be bfloat16/float16/float32/float64"
            )
        values = _load_initializer_values(graph, scales_input, node)
        if values is None:
            scales = tuple(
                output_shape[axis] / input_shape[axis] for axis in range(rank)
            )
        else:
            scales = tuple(float(value) for value in values)
            expected = tuple(
                int(input_shape[axis] * scales[axis]) for axis in range(rank)
            )
            _validate_output_shape(expected, output_shape)
    else:
        scales_attr = node.attrs.get("scales")
        if scales_attr is None:
            raise UnsupportedOpError("Upsample requires scales attribute or input")
        scales = tuple(float(value) for value in scales_attr)
        if len(scales) != rank:
            raise UnsupportedOpError("Upsample scales length mismatch")
        expected = tuple(int(input_shape[axis] * scales[axis]) for axis in range(rank))
        _validate_output_shape(expected, output_shape)
    return ResizeOp(
        input0=input_name,
        output=output_name,
        scales=scales,
        scales_input=scales_input,
        sizes_input=None,
        roi_input=None,
        axes=axes,
        mode=mode,
        coordinate_transformation_mode="asymmetric",
        nearest_mode="floor",
        cubic_coeff_a=-0.75,
        exclude_outside=False,
        extrapolation_value=0.0,
        antialias=False,
        keep_aspect_ratio_policy="stretch",
    )

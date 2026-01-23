from __future__ import annotations

from dataclasses import dataclass

from shared.scalar_types import ScalarType

from ..ir.ops import GridSampleOp
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from .common import value_dtype, value_shape
from .registry import register_lowering

_SUPPORTED_MODES = {"linear", "nearest", "cubic"}
_SUPPORTED_PADDING_MODES = {"zeros", "border", "reflection"}


@dataclass(frozen=True)
class _GridSampleShapes:
    input_shape: tuple[int, ...]
    grid_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    spatial_rank: int


def _decode_attr(value: object, default: str) -> str:
    if value is None:
        return default
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, str):
        return value
    return str(value)


def _resolve_shapes(graph: Graph, node: Node) -> _GridSampleShapes:
    input_shape = value_shape(graph, node.inputs[0], node)
    grid_shape = value_shape(graph, node.inputs[1], node)
    output_shape = value_shape(graph, node.outputs[0], node)
    if len(input_shape) < 3:
        raise ShapeInferenceError(
            "GridSample expects input rank of at least 3"
        )
    spatial_rank = len(input_shape) - 2
    if any(dim < 0 for dim in (*input_shape, *grid_shape, *output_shape)):
        raise ShapeInferenceError(
            "GridSample requires static, non-negative shapes"
        )
    return _GridSampleShapes(
        input_shape=input_shape,
        grid_shape=grid_shape,
        output_shape=output_shape,
        spatial_rank=spatial_rank,
    )


def _validate_shapes(shapes: _GridSampleShapes) -> None:
    input_shape = shapes.input_shape
    grid_shape = shapes.grid_shape
    output_shape = shapes.output_shape
    spatial_rank = shapes.spatial_rank
    if len(grid_shape) != spatial_rank + 2:
        raise ShapeInferenceError(
            "GridSample expects grid rank to match input spatial rank"
        )
    if len(output_shape) != spatial_rank + 2:
        raise ShapeInferenceError(
            "GridSample expects output rank to match input spatial rank"
        )
    if grid_shape[0] != input_shape[0]:
        raise ShapeInferenceError("GridSample expects matching batch dimension")
    if grid_shape[-1] != spatial_rank:
        raise ShapeInferenceError(
            "GridSample expects grid last dimension to match spatial rank"
        )
    expected_output = (
        input_shape[0],
        input_shape[1],
        *grid_shape[1:-1],
    )
    if output_shape != expected_output:
        raise ShapeInferenceError(
            "GridSample output shape must be "
            f"{expected_output}, got {output_shape}"
        )


def _validate_dtypes(
    graph: Graph, node: Node
) -> tuple[ScalarType, ScalarType]:
    input_dtype = value_dtype(graph, node.inputs[0], node)
    grid_dtype = value_dtype(graph, node.inputs[1], node)
    output_dtype = value_dtype(graph, node.outputs[0], node)
    if input_dtype != output_dtype:
        raise UnsupportedOpError(
            "GridSample expects matching input/output dtypes, got "
            f"{input_dtype.onnx_name} and {output_dtype.onnx_name}"
        )
    if not input_dtype.is_float:
        raise UnsupportedOpError(
            "GridSample currently supports floating-point inputs only"
        )
    if not grid_dtype.is_float:
        raise UnsupportedOpError("GridSample expects floating-point grid")
    return input_dtype, grid_dtype


@register_lowering("GridSample")
def lower_grid_sample(graph: Graph, node: Node) -> GridSampleOp:
    if len(node.inputs) != 2 or len(node.outputs) != 1:
        raise UnsupportedOpError(
            "GridSample expects 2 inputs (X, grid) and 1 output"
        )
    shapes = _resolve_shapes(graph, node)
    _validate_shapes(shapes)
    mode = _decode_attr(node.attrs.get("mode"), "linear")
    padding_mode = _decode_attr(node.attrs.get("padding_mode"), "zeros")
    align_corners = int(node.attrs.get("align_corners", 0))
    if mode not in _SUPPORTED_MODES:
        raise UnsupportedOpError(
            f"GridSample mode {mode!r} is not supported"
        )
    if padding_mode not in _SUPPORTED_PADDING_MODES:
        raise UnsupportedOpError(
            "GridSample padding_mode "
            f"{padding_mode!r} is not supported"
        )
    if align_corners not in {0, 1}:
        raise UnsupportedOpError("GridSample align_corners must be 0 or 1")
    input_dtype, grid_dtype = _validate_dtypes(graph, node)
    if shapes.spatial_rank > 3:
        raise UnsupportedOpError(
            "GridSample supports up to 3 spatial dimensions"
        )
    return GridSampleOp(
        input0=node.inputs[0],
        grid=node.inputs[1],
        output=node.outputs[0],
        input_shape=shapes.input_shape,
        grid_shape=shapes.grid_shape,
        output_shape=shapes.output_shape,
        spatial_rank=shapes.spatial_rank,
        input_spatial=shapes.input_shape[2:],
        output_spatial=shapes.output_shape[2:],
        mode=mode,
        padding_mode=padding_mode,
        align_corners=bool(align_corners),
        dtype=input_dtype,
        grid_dtype=grid_dtype,
    )

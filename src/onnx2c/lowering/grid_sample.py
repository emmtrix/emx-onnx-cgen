from __future__ import annotations

from dataclasses import dataclass

from shared.scalar_types import ScalarType

from ..codegen.c_emitter import GridSampleOp
from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.model import Graph, Node
from .common import value_dtype, value_shape
from .registry import register_lowering

_SUPPORTED_MODES = {"linear", "nearest", "bilinear"}
_SUPPORTED_PADDING = {"zeros", "border", "reflection"}


@dataclass(frozen=True)
class _GridSampleConfig:
    input_shape: tuple[int, ...]
    grid_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    input_dtype: ScalarType
    grid_dtype: ScalarType
    output_dtype: ScalarType


def _decode_attr(value: object, default: str) -> str:
    if value is None:
        return default
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, str):
        return value
    return str(value)


def _resolve_config(graph: Graph, node: Node) -> _GridSampleConfig:
    input_shape = value_shape(graph, node.inputs[0], node)
    grid_shape = value_shape(graph, node.inputs[1], node)
    output_shape = value_shape(graph, node.outputs[0], node)
    input_dtype = value_dtype(graph, node.inputs[0], node)
    grid_dtype = value_dtype(graph, node.inputs[1], node)
    output_dtype = value_dtype(graph, node.outputs[0], node)
    return _GridSampleConfig(
        input_shape=input_shape,
        grid_shape=grid_shape,
        output_shape=output_shape,
        input_dtype=input_dtype,
        grid_dtype=grid_dtype,
        output_dtype=output_dtype,
    )


def _validate_shapes(config: _GridSampleConfig, node: Node) -> int:
    if len(config.input_shape) < 3:
        raise UnsupportedOpError("GridSample expects input rank >= 3")
    spatial_rank = len(config.input_shape) - 2
    if len(config.grid_shape) != spatial_rank + 2:
        raise ShapeInferenceError(
            "GridSample expects grid rank to match input spatial rank"
        )
    if config.grid_shape[-1] != spatial_rank:
        raise ShapeInferenceError(
            "GridSample grid last dimension must match input spatial rank"
        )
    if config.grid_shape[0] != config.input_shape[0]:
        raise ShapeInferenceError("GridSample batch size must match grid")
    expected_output_shape = (
        config.input_shape[0],
        config.input_shape[1],
        *config.grid_shape[1:-1],
    )
    if config.output_shape != expected_output_shape:
        raise ShapeInferenceError(
            "GridSample output shape must match grid spatial shape"
        )
    return spatial_rank


@register_lowering("GridSample")
def lower_grid_sample(graph: Graph, node: Node) -> GridSampleOp:
    if len(node.inputs) != 2 or len(node.outputs) != 1:
        raise UnsupportedOpError("GridSample must have 2 inputs and 1 output")
    if not node.inputs[0] or not node.inputs[1]:
        raise UnsupportedOpError("GridSample inputs must be provided")
    config = _resolve_config(graph, node)
    _validate_shapes(config, node)
    if config.input_dtype != config.output_dtype:
        raise UnsupportedOpError(
            "GridSample expects input and output dtypes to match"
        )
    if not config.grid_dtype.is_float:
        raise UnsupportedOpError("GridSample grid input must be float")
    mode = _decode_attr(node.attrs.get("mode"), "linear")
    if mode not in _SUPPORTED_MODES:
        raise UnsupportedOpError(f"GridSample mode {mode} is not supported")
    if mode == "bilinear":
        mode = "linear"
    padding_mode = _decode_attr(node.attrs.get("padding_mode"), "zeros")
    if padding_mode not in _SUPPORTED_PADDING:
        raise UnsupportedOpError(
            f"GridSample padding_mode {padding_mode} is not supported"
        )
    align_corners = int(node.attrs.get("align_corners", 0))
    if align_corners not in {0, 1}:
        raise UnsupportedOpError("GridSample align_corners must be 0 or 1")
    return GridSampleOp(
        input0=node.inputs[0],
        grid=node.inputs[1],
        output=node.outputs[0],
        input_shape=config.input_shape,
        grid_shape=config.grid_shape,
        output_shape=config.output_shape,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=bool(align_corners),
        dtype=config.output_dtype,
        grid_dtype=config.grid_dtype,
    )

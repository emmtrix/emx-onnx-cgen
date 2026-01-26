from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shared.scalar_types import ScalarType

from ...errors import ShapeInferenceError, UnsupportedOpError
from ..op_base import (
    BroadcastingOpBase,
    GatherLikeOpBase,
    RenderableOpBase,
    ShapeLikeOpBase,
)
from ..op_context import OpContext


def _compute_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides: list[int] = []
    stride = 1
    for dim in reversed(shape):
        strides.append(stride)
        stride *= dim
    return tuple(reversed(strides))


@dataclass(frozen=True)
class CastOp(RenderableOpBase):
    input0: str
    output: str
    shape: tuple[int, ...]
    input_dtype: ScalarType
    dtype: ScalarType

    def infer_types(self, ctx: OpContext) -> None:
        ctx.dtype(self.input0)
        ctx.dtype(self.output)

    def infer_shapes(self, ctx: OpContext) -> None:
        shape = ctx.shape(self.input0)
        ctx.set_shape(self.output, shape)

@dataclass(frozen=True)
class QuantizeLinearOp(RenderableOpBase):
    input0: str
    scale: str
    zero_point: str | None
    output: str
    input_shape: tuple[int, ...]
    axis: int | None
    dtype: ScalarType
    input_dtype: ScalarType
    scale_dtype: ScalarType

@dataclass(frozen=True)
class ConcatOp(RenderableOpBase):
    inputs: tuple[str, ...]
    output: str
    axis: int
    input_shapes: tuple[tuple[int, ...], ...]
    output_shape: tuple[int, ...]
    dtype: ScalarType

@dataclass(frozen=True)
class GatherElementsOp(RenderableOpBase):
    data: str
    indices: str
    output: str
    axis: int
    data_shape: tuple[int, ...]
    indices_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    dtype: ScalarType
    indices_dtype: ScalarType

@dataclass(frozen=True)
class GatherOp(GatherLikeOpBase):
    data: str
    indices: str
    output: str
    axis: int

    def _gather_data(self) -> str:
        return self.data

    def _gather_indices(self) -> str:
        return self.indices

    def _gather_output(self) -> str:
        return self.output

    def _gather_axis(self) -> int:
        return self.axis

    def _gather_mode(self) -> str:
        return "gather"

@dataclass(frozen=True)
class GatherNDOp(RenderableOpBase):
    data: str
    indices: str
    output: str
    batch_dims: int
    data_shape: tuple[int, ...]
    indices_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    dtype: ScalarType
    indices_dtype: ScalarType

@dataclass(frozen=True)
class ScatterNDOp(RenderableOpBase):
    data: str
    indices: str
    updates: str
    output: str
    data_shape: tuple[int, ...]
    indices_shape: tuple[int, ...]
    updates_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    reduction: str
    dtype: ScalarType
    indices_dtype: ScalarType

@dataclass(frozen=True)
class TensorScatterOp(RenderableOpBase):
    past_cache: str
    update: str
    write_indices: str | None
    output: str
    past_cache_shape: tuple[int, ...]
    update_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    write_indices_shape: tuple[int, ...] | None
    axis: int
    mode: str
    dtype: ScalarType
    write_indices_dtype: ScalarType | None

@dataclass(frozen=True)
class TransposeOp(RenderableOpBase):
    input0: str
    output: str
    perm: tuple[int, ...]
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    dtype: ScalarType
    input_dtype: ScalarType

    def infer_shapes(self, ctx: OpContext) -> None:
        input_shape = ctx.shape(self.input0)
        if len(self.perm) != len(input_shape):
            raise ShapeInferenceError(
                "Transpose perm rank must match input rank, "
                f"got perm {self.perm} for input shape {input_shape}"
            )
        output_shape = tuple(input_shape[axis] for axis in self.perm)
        ctx.set_shape(self.output, output_shape)

@dataclass(frozen=True)
class ReshapeOp(RenderableOpBase):
    input0: str
    output: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...] | None
    dtype: ScalarType
    input_dtype: ScalarType

    def infer_shapes(self, ctx: OpContext) -> None:
        input_shape = ctx.shape(self.input0)
        output_shape = (
            self.output_shape
            if self.output_shape is not None
            else ctx.shape(self.output)
        )
        ctx.set_shape(self.output, output_shape)

@dataclass(frozen=True)
class EyeLikeOp(RenderableOpBase):
    input0: str
    output: str
    output_shape: tuple[int, ...]
    k: int
    dtype: ScalarType
    input_dtype: ScalarType

@dataclass(frozen=True)
class TriluOp(RenderableOpBase):
    input0: str
    output: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    upper: bool
    k_value: int
    k_input: str | None
    k_input_shape: tuple[int, ...] | None
    k_input_dtype: ScalarType | None
    dtype: ScalarType
    input_dtype: ScalarType

@dataclass(frozen=True)
class TileOp(RenderableOpBase):
    input0: str
    output: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    repeats: tuple[int, ...]
    input_strides: tuple[int, ...]
    dtype: ScalarType
    input_dtype: ScalarType

@dataclass(frozen=True)
class PadOp(RenderableOpBase):
    input0: str
    output: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    pads_begin: tuple[int, ...] | None
    pads_end: tuple[int, ...] | None
    pads_input: str | None
    pads_shape: tuple[int, ...] | None
    pads_dtype: ScalarType | None
    pads_axis_map: tuple[int | None, ...] | None
    pads_values: tuple[int, ...] | None
    axes_input: str | None
    axes_shape: tuple[int, ...] | None
    axes_dtype: ScalarType | None
    mode: str
    value: float | int | bool
    value_input: str | None
    value_shape: tuple[int, ...] | None
    dtype: ScalarType
    input_dtype: ScalarType
    input_strides: tuple[int, ...]

@dataclass(frozen=True)
class DepthToSpaceOp(RenderableOpBase):
    input0: str
    output: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    blocksize: int
    mode: str
    dtype: ScalarType
    input_dtype: ScalarType

@dataclass(frozen=True)
class SpaceToDepthOp(RenderableOpBase):
    input0: str
    output: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    blocksize: int
    dtype: ScalarType
    input_dtype: ScalarType

@dataclass(frozen=True)
class SliceOp(RenderableOpBase):
    input0: str
    output: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    starts: tuple[int, ...] | None
    steps: tuple[int, ...] | None
    axes: tuple[int, ...] | None
    starts_input: str | None
    ends_input: str | None
    axes_input: str | None
    steps_input: str | None
    starts_shape: tuple[int, ...] | None
    ends_shape: tuple[int, ...] | None
    axes_shape: tuple[int, ...] | None
    steps_shape: tuple[int, ...] | None
    starts_dtype: ScalarType | None
    ends_dtype: ScalarType | None
    axes_dtype: ScalarType | None
    steps_dtype: ScalarType | None
    dtype: ScalarType
    input_dtype: ScalarType

@dataclass(frozen=True)
class ResizeOp(RenderableOpBase):
    input0: str
    output: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    scales: tuple[float, ...]
    scales_input: str | None
    sizes_input: str | None
    roi_input: str | None
    axes: tuple[int, ...]
    scales_shape: tuple[int, ...] | None
    sizes_shape: tuple[int, ...] | None
    roi_shape: tuple[int, ...] | None
    scales_dtype: ScalarType | None
    sizes_dtype: ScalarType | None
    roi_dtype: ScalarType | None
    scales_axes: tuple[int, ...] | None
    sizes_axes: tuple[int, ...] | None
    roi_axes: tuple[int, ...] | None
    mode: str
    coordinate_transformation_mode: str
    nearest_mode: str
    cubic_coeff_a: float
    exclude_outside: bool
    extrapolation_value: float
    antialias: bool
    keep_aspect_ratio_policy: str
    dtype: ScalarType

@dataclass(frozen=True)
class GridSampleOp(RenderableOpBase):
    input0: str
    grid: str
    output: str
    input_shape: tuple[int, ...]
    grid_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    spatial_rank: int
    input_spatial: tuple[int, ...]
    output_spatial: tuple[int, ...]
    mode: str
    padding_mode: str
    align_corners: bool
    dtype: ScalarType
    grid_dtype: ScalarType

@dataclass(frozen=True)
class ConstantOfShapeOp(RenderableOpBase):
    input0: str
    output: str
    input_shape: tuple[int, ...]
    shape: tuple[int, ...]
    value: float | int | bool
    dtype: ScalarType
    input_dtype: ScalarType

@dataclass(frozen=True)
class ShapeOp(RenderableOpBase):
    input0: str
    output: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    values: tuple[int, ...]
    dtype: ScalarType
    input_dtype: ScalarType

@dataclass(frozen=True)
class SizeOp(RenderableOpBase):
    input0: str
    output: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    value: int
    dtype: ScalarType
    input_dtype: ScalarType

@dataclass(frozen=True)
class NonZeroOp(RenderableOpBase):
    input0: str
    output: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    dtype: ScalarType
    input_dtype: ScalarType

@dataclass(frozen=True)
class NonMaxSuppressionOp(RenderableOpBase):
    boxes: str
    scores: str
    max_output_boxes_per_class: str | None
    iou_threshold: str | None
    score_threshold: str | None
    output: str
    boxes_shape: tuple[int, ...]
    scores_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    center_point_box: int
    boxes_dtype: ScalarType
    output_dtype: ScalarType
    max_output_dtype: ScalarType | None
    max_output_shape: tuple[int, ...] | None
    iou_threshold_dtype: ScalarType | None
    iou_threshold_shape: tuple[int, ...] | None
    score_threshold_dtype: ScalarType | None
    score_threshold_shape: tuple[int, ...] | None

@dataclass(frozen=True)
class ExpandOp(ShapeLikeOpBase):
    input0: str
    input_shape: str
    output: str

    def _shape_data(self) -> str:
        return self.input0

    def _shape_output(self) -> str:
        return self.output

    def _shape_mode(self) -> str:
        return "expand"

    def _shape_spec(self, ctx: OpContext) -> tuple[int, ...]:
        initializer = ctx.initializer(self.input_shape)
        if initializer is not None:
            if initializer.type.dtype not in {ScalarType.I64, ScalarType.I32}:
                raise UnsupportedOpError(
                    f"{self.kind} shape input must be int64 or int32"
                )
            if len(initializer.type.shape) != 1:
                raise UnsupportedOpError(
                    f"{self.kind} shape input must be a 1D tensor"
                )
            values = np.array(initializer.data, dtype=np.int64).reshape(-1)
            if values.size == 0:
                raise ShapeInferenceError(
                    f"{self.kind} shape input cannot be empty"
                )
            return tuple(int(value) for value in values)
        dtype = ctx.dtype(self.input_shape)
        if dtype not in {ScalarType.I64, ScalarType.I32}:
            raise UnsupportedOpError(
                f"{self.kind} shape input must be int64 or int32"
            )
        shape = ctx.shape(self.input_shape)
        if len(shape) != 1:
            raise UnsupportedOpError(
                f"{self.kind} shape input must be a 1D tensor"
            )
        if shape[0] <= 0:
            raise ShapeInferenceError(
                f"{self.kind} shape input cannot be empty"
            )
        output_shape = ctx.shape(self.output)
        if not output_shape:
            raise ShapeInferenceError(
                f"{self.kind} output shape must be specified"
            )
        return output_shape

    def _shape_derived(
        self,
        ctx: OpContext,
        *,
        data_shape: tuple[int, ...],
        target_shape: tuple[int, ...],
        output_shape: tuple[int, ...],
    ) -> None:
        input_shape_padded = (
            (1,) * (len(output_shape) - len(data_shape)) + data_shape
        )
        input_strides = _compute_strides(input_shape_padded)
        ctx.set_derived(self, "input_shape_padded", input_shape_padded)
        ctx.set_derived(self, "input_strides", input_strides)

@dataclass(frozen=True)
class CumSumOp(RenderableOpBase):
    input0: str
    axis_input: str | None
    axis_input_dtype: ScalarType | None
    axis: int | None
    output: str
    input_shape: tuple[int, ...]
    dtype: ScalarType
    input_dtype: ScalarType
    exclusive: bool
    reverse: bool

@dataclass(frozen=True)
class RangeOp(RenderableOpBase):
    start: str
    limit: str
    delta: str
    output: str
    output_shape: tuple[int, ...]
    length: int
    dtype: ScalarType
    input_dtype: ScalarType

@dataclass(frozen=True)
class HammingWindowOp(RenderableOpBase):
    size: str
    output: str
    output_shape: tuple[int, ...]
    periodic: bool
    dtype: ScalarType
    input_dtype: ScalarType

@dataclass(frozen=True)
class OneHotOp(RenderableOpBase):
    indices: str
    depth: str
    values: str
    output: str
    axis: int
    indices_shape: tuple[int, ...]
    values_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    depth_dim: int
    dtype: ScalarType
    indices_dtype: ScalarType
    depth_dtype: ScalarType

@dataclass(frozen=True)
class SplitOp(RenderableOpBase):
    input0: str
    outputs: tuple[str, ...]
    input_shape: tuple[int, ...]
    output_shapes: tuple[tuple[int, ...], ...]
    axis: int
    split_sizes: tuple[int, ...]
    dtype: ScalarType
    input_dtype: ScalarType

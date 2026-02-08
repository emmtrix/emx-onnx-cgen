from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shared.scalar_types import ScalarType

from ...errors import ShapeInferenceError, UnsupportedOpError
from ..op_base import (
    BroadcastingOpBase,
    EmitContext,
    Emitter,
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
    __io_inputs__ = ("input0",)
    __io_outputs__ = ("output",)
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
    __io_inputs__ = ("input0", "scale", "zero_point")
    __io_outputs__ = ("output",)
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
class DequantizeLinearOp(RenderableOpBase):
    __io_inputs__ = ("input0", "scale", "zero_point")
    __io_outputs__ = ("output",)
    input0: str
    scale: str
    zero_point: str | None
    output: str
    input_shape: tuple[int, ...]
    axis: int | None
    block_size: int | None
    dtype: ScalarType
    input_dtype: ScalarType
    scale_dtype: ScalarType


@dataclass(frozen=True)
class ConcatOp(RenderableOpBase):
    __io_inputs__ = ("inputs",)
    __io_outputs__ = ("output",)
    inputs: tuple[str, ...]
    output: str
    axis: int
    input_shapes: tuple[tuple[int, ...], ...]
    output_shape: tuple[int, ...]
    dtype: ScalarType


@dataclass(frozen=True)
class GatherElementsOp(RenderableOpBase):
    __io_inputs__ = ("data", "indices")
    __io_outputs__ = ("output",)
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
    __io_inputs__ = ("data", "indices")
    __io_outputs__ = ("output",)
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
    __io_inputs__ = ("data", "indices", "updates")
    __io_outputs__ = ("output",)
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
    __io_inputs__ = ("past_cache", "update", "write_indices")
    __io_outputs__ = ("output",)
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
    __io_inputs__ = ("input0",)
    __io_outputs__ = ("output",)
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
    __io_inputs__ = ("input0",)
    __io_outputs__ = ("output",)
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
    __io_inputs__ = ("input0",)
    __io_outputs__ = ("output",)
    input0: str
    output: str
    output_shape: tuple[int, ...]
    k: int
    dtype: ScalarType
    input_dtype: ScalarType


@dataclass(frozen=True)
class BernoulliOp(RenderableOpBase):
    __io_inputs__ = ("input0",)
    __io_outputs__ = ("output",)
    input0: str
    output: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    input_dtype: ScalarType
    dtype: ScalarType
    seed: int | None


@dataclass(frozen=True)
class TriluOp(RenderableOpBase):
    __io_inputs__ = ("input0", "k_input")
    __io_outputs__ = ("output",)
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

    def call_args(self) -> tuple[str, ...]:
        args = [self.input0, self.output]
        if self.k_input is not None:
            args.append(self.k_input)
        return tuple(args)


@dataclass(frozen=True)
class TileOp(RenderableOpBase):
    __io_inputs__ = ("input0",)
    __io_outputs__ = ("output",)
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
    __io_inputs__ = ("input0", "pads_input", "axes_input", "value_input")
    __io_outputs__ = ("output",)
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
    __io_inputs__ = ("input0",)
    __io_outputs__ = ("output",)
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
    __io_inputs__ = ("input0",)
    __io_outputs__ = ("output",)
    input0: str
    output: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    blocksize: int
    dtype: ScalarType
    input_dtype: ScalarType


@dataclass(frozen=True)
class SliceOp(RenderableOpBase):
    __io_inputs__ = (
        "input0",
        "starts_input",
        "ends_input",
        "axes_input",
        "steps_input",
    )
    __io_outputs__ = ("output",)
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
    __io_inputs__ = ("input0", "roi_input", "scales_input", "sizes_input")
    __io_outputs__ = ("output",)
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
    __io_inputs__ = ("input0", "grid")
    __io_outputs__ = ("output",)
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
    __io_inputs__ = ("input0",)
    __io_outputs__ = ("output",)
    input0: str
    output: str
    input_shape: tuple[int, ...]
    shape: tuple[int, ...]
    value: float | int | bool
    dtype: ScalarType
    input_dtype: ScalarType

    def emit(self, emitter: Emitter, ctx: EmitContext) -> str:
        return emitter.emit_generic_op(self, ctx)


@dataclass(frozen=True)
class ShapeOp(RenderableOpBase):
    __io_inputs__ = ("input0",)
    __io_outputs__ = ("output",)
    input0: str
    output: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    values: tuple[int, ...]
    dtype: ScalarType
    input_dtype: ScalarType

    def emit(self, emitter: Emitter, ctx: EmitContext) -> str:
        return emitter.emit_generic_op(self, ctx)


@dataclass(frozen=True)
class SizeOp(RenderableOpBase):
    def emit(self, emitter: Emitter, ctx: EmitContext) -> str:
        return emitter.emit_generic_op(self, ctx)

    __io_inputs__ = ("input0",)
    __io_outputs__ = ("output",)
    input0: str
    output: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    value: int
    dtype: ScalarType
    input_dtype: ScalarType


@dataclass(frozen=True)
class OptionalHasElementOp(RenderableOpBase):
    __io_inputs__ = ("input0",)
    __io_outputs__ = ("output",)
    input0: str
    output: str

    def emit(self, emitter: Emitter, ctx: EmitContext) -> str:
        return emitter.emit_generic_op(self, ctx)

    def call_args(self) -> tuple[str, ...]:
        return (self.input0, f"{self.input0}_present", self.output)

    def validate(self, ctx: OpContext) -> None:
        value = ctx.graph.find_value(self.input0)
        if not value.type.is_optional:
            raise UnsupportedOpError(
                f"{self.kind} expects optional input, got non-optional tensor."
            )
        try:
            output_dtype = ctx.dtype(self.output)
        except ShapeInferenceError:
            return None
        if output_dtype != ScalarType.BOOL:
            raise UnsupportedOpError(
                f"{self.kind} expects bool output, got {output_dtype.onnx_name}"
            )
        return None

    def infer_types(self, ctx: OpContext) -> None:
        ctx.dtype(self.input0)
        try:
            output_dtype = ctx.dtype(self.output)
        except ShapeInferenceError:
            ctx.set_dtype(self.output, ScalarType.BOOL)
            return None
        if output_dtype != ScalarType.BOOL:
            raise UnsupportedOpError(
                f"{self.kind} expects bool output, got {output_dtype.onnx_name}"
            )

    def infer_shapes(self, ctx: OpContext) -> None:
        try:
            output_shape = ctx.shape(self.output)
        except ShapeInferenceError:
            ctx.set_shape(self.output, ())
            return None
        if output_shape not in {(), (1,)}:
            raise UnsupportedOpError(
                f"{self.kind} expects scalar output, got shape {output_shape}"
            )


@dataclass(frozen=True)
class NonZeroOp(RenderableOpBase):
    __io_inputs__ = ("input0",)
    __io_outputs__ = ("output",)
    input0: str
    output: str

    def emit(self, emitter: Emitter, ctx: EmitContext) -> str:
        return emitter.emit_generic_op(self, ctx)

    def validate(self, ctx: OpContext) -> None:
        if len(ctx.shape(self.input0)) == 0:
            raise UnsupportedOpError(f"{self.kind} does not support scalar inputs")

    def infer_types(self, ctx: OpContext) -> None:
        ctx.dtype(self.input0)
        try:
            output_dtype = ctx.dtype(self.output)
        except ShapeInferenceError:
            ctx.set_dtype(self.output, ScalarType.I64)
            return
        if output_dtype != ScalarType.I64:
            raise UnsupportedOpError(
                f"{self.kind} output dtype must be int64, got {output_dtype.onnx_name}"
            )

    def infer_shapes(self, ctx: OpContext) -> None:
        input_shape = ctx.shape(self.input0)
        output_shape = ctx.shape(self.output)
        if len(output_shape) != 2:
            raise ShapeInferenceError(f"{self.kind} output must be 2D")
        if output_shape[0] != len(input_shape):
            raise ShapeInferenceError(
                f"{self.kind} output shape must be ({len(input_shape)}, N), got {output_shape}"
            )
        if output_shape[0] < 0 or output_shape[1] < 0:
            raise ShapeInferenceError(f"{self.kind} output shape must be non-negative")
        ctx.set_shape(self.output, output_shape)


@dataclass(frozen=True)
class NonMaxSuppressionOp(RenderableOpBase):
    __io_inputs__ = (
        "boxes",
        "scores",
        "max_output_boxes_per_class",
        "iou_threshold",
        "score_threshold",
    )
    __io_outputs__ = ("output",)
    boxes: str
    scores: str
    max_output_boxes_per_class: str | None
    iou_threshold: str | None
    score_threshold: str | None
    output: str
    center_point_box: int

    def _validate_scalar_input(
        self,
        ctx: OpContext,
        *,
        name: str,
        allowed_dtypes: set[ScalarType],
        label: str,
    ) -> None:
        dtype = ctx.dtype(name)
        if dtype not in allowed_dtypes:
            allowed = ", ".join(sorted(d.onnx_name for d in allowed_dtypes))
            raise UnsupportedOpError(
                f"{self.kind} {label} must be {allowed}, got {dtype.onnx_name}"
            )
        shape = ctx.shape(name)
        if shape not in {(), (1,)} and int(np.prod(shape, dtype=np.int64)) != 1:
            raise ShapeInferenceError(
                f"{self.kind} {label} must be a scalar tensor, got shape {shape}"
            )

    def validate(self, ctx: OpContext) -> None:
        boxes_shape = ctx.shape(self.boxes)
        scores_shape = ctx.shape(self.scores)
        if len(boxes_shape) != 3 or boxes_shape[2] != 4:
            raise ShapeInferenceError(
                f"{self.kind} boxes input must have shape "
                f"[num_batches, num_boxes, 4], got {boxes_shape}"
            )
        if len(scores_shape) != 3:
            raise ShapeInferenceError(
                f"{self.kind} scores input must have shape "
                f"[num_batches, num_classes, num_boxes], got {scores_shape}"
            )
        if boxes_shape[0] != scores_shape[0]:
            raise ShapeInferenceError(
                f"{self.kind} boxes/scores batch dims must match, "
                f"got {boxes_shape[0]} and {scores_shape[0]}"
            )
        if boxes_shape[1] != scores_shape[2]:
            raise ShapeInferenceError(
                f"{self.kind} boxes num_boxes dim {boxes_shape[1]} "
                f"must match scores num_boxes dim {scores_shape[2]}"
            )
        boxes_dtype = ctx.dtype(self.boxes)
        scores_dtype = ctx.dtype(self.scores)
        if boxes_dtype != scores_dtype or not boxes_dtype.is_float:
            raise UnsupportedOpError(
                f"{self.kind} boxes and scores must be the same float dtype, "
                f"got {boxes_dtype.onnx_name} and {scores_dtype.onnx_name}"
            )
        if self.max_output_boxes_per_class is not None:
            self._validate_scalar_input(
                ctx,
                name=self.max_output_boxes_per_class,
                allowed_dtypes={ScalarType.I32, ScalarType.I64},
                label="max_output_boxes_per_class input",
            )
        if self.iou_threshold is not None:
            self._validate_scalar_input(
                ctx,
                name=self.iou_threshold,
                allowed_dtypes={ScalarType.F32, ScalarType.F64},
                label="iou_threshold input",
            )
        if self.score_threshold is not None:
            self._validate_scalar_input(
                ctx,
                name=self.score_threshold,
                allowed_dtypes={ScalarType.F32, ScalarType.F64},
                label="score_threshold input",
            )
        if self.center_point_box not in {0, 1}:
            raise UnsupportedOpError(
                f"{self.kind} center_point_box must be 0 or 1, got {self.center_point_box}"
            )

    def infer_types(self, ctx: OpContext) -> None:
        ctx.dtype(self.boxes)
        ctx.dtype(self.scores)
        if self.max_output_boxes_per_class is not None:
            ctx.dtype(self.max_output_boxes_per_class)
        if self.iou_threshold is not None:
            ctx.dtype(self.iou_threshold)
        if self.score_threshold is not None:
            ctx.dtype(self.score_threshold)
        try:
            output_dtype = ctx.dtype(self.output)
        except ShapeInferenceError:
            ctx.set_dtype(self.output, ScalarType.I64)
            output_dtype = ScalarType.I64
        if output_dtype != ScalarType.I64:
            raise UnsupportedOpError(
                f"{self.kind} output dtype must be int64, got {output_dtype.onnx_name}"
            )

    def infer_shapes(self, ctx: OpContext) -> None:
        output_shape = ctx.shape(self.output)
        if len(output_shape) != 2 or output_shape[1] != 3:
            raise ShapeInferenceError(
                f"{self.kind} output must have shape [num_selected, 3], "
                f"got {output_shape}"
            )
        ctx.set_shape(self.output, output_shape)


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
                raise UnsupportedOpError(f"{self.kind} shape input must be a 1D tensor")
            values = np.array(initializer.data, dtype=np.int64).reshape(-1)
            if values.size == 0:
                raise ShapeInferenceError(f"{self.kind} shape input cannot be empty")
            return tuple(int(value) for value in values)
        dtype = ctx.dtype(self.input_shape)
        if dtype not in {ScalarType.I64, ScalarType.I32}:
            raise UnsupportedOpError(f"{self.kind} shape input must be int64 or int32")
        shape = ctx.shape(self.input_shape)
        if len(shape) != 1:
            raise UnsupportedOpError(f"{self.kind} shape input must be a 1D tensor")
        if shape[0] <= 0:
            raise ShapeInferenceError(f"{self.kind} shape input cannot be empty")
        output_shape = ctx.shape(self.output)
        if not output_shape:
            raise ShapeInferenceError(f"{self.kind} output shape must be specified")
        return output_shape

    def _shape_derived(
        self,
        ctx: OpContext,
        *,
        data_shape: tuple[int, ...],
        target_shape: tuple[int, ...],
        output_shape: tuple[int, ...],
    ) -> None:
        input_shape_padded = (1,) * (len(output_shape) - len(data_shape)) + data_shape
        input_strides = _compute_strides(input_shape_padded)
        ctx.set_derived(self, "input_shape_padded", input_shape_padded)
        ctx.set_derived(self, "input_strides", input_strides)


@dataclass(frozen=True)
class CumSumOp(RenderableOpBase):
    __io_inputs__ = ("input0", "axis_input")
    __io_outputs__ = ("output",)
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
    __io_inputs__ = ("start", "limit", "delta")
    __io_outputs__ = ("output",)
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
    __io_inputs__ = ("size",)
    __io_outputs__ = ("output",)
    size: str
    output: str
    output_shape: tuple[int, ...]
    periodic: bool
    dtype: ScalarType
    input_dtype: ScalarType


@dataclass(frozen=True)
class OneHotOp(RenderableOpBase):
    __io_inputs__ = ("indices", "depth", "values")
    __io_outputs__ = ("output",)
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
class TfIdfVectorizerOp(RenderableOpBase):
    __io_inputs__ = ("input0",)
    __io_outputs__ = ("output",)
    input0: str
    output: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    input_dtype: ScalarType
    output_dtype: ScalarType
    min_gram_length: int
    max_gram_length: int
    max_skip_count: int
    mode: str
    ngram_counts: tuple[int, ...]
    ngram_indexes: tuple[int, ...]
    pool_int64s: tuple[int, ...]
    weights: tuple[float, ...] | None


@dataclass(frozen=True)
class StringNormalizerOp(RenderableOpBase):
    __io_inputs__ = ("input0",)
    __io_outputs__ = ("output",)
    input0: str
    output: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    case_change_action: str
    is_case_sensitive: bool
    stopwords: tuple[str, ...]


@dataclass(frozen=True)
class SplitOp(RenderableOpBase):
    __io_inputs__ = ("input0",)
    __io_outputs__ = ("outputs",)
    input0: str
    outputs: tuple[str, ...]
    input_shape: tuple[int, ...]
    output_shapes: tuple[tuple[int, ...], ...]
    axis: int
    split_sizes: tuple[int, ...]
    dtype: ScalarType
    input_dtype: ScalarType

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shared.scalar_types import ScalarType

from ...errors import ShapeInferenceError, UnsupportedOpError
from ..op_base import (
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


def _shape_product(shape: tuple[int, ...]) -> int:
    product = 1
    for dim in shape:
        product *= dim
    return product


@dataclass(frozen=True)
class CastOp(RenderableOpBase):
    __io_inputs__ = ("input0",)
    __io_outputs__ = ("output",)
    input0: str
    output: str

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
    axis: int | None


@dataclass(frozen=True)
class DynamicQuantizeLinearOp(RenderableOpBase):
    __io_inputs__ = ("input0",)
    __io_outputs__ = ("output", "scale", "zero_point")
    input0: str
    output: str
    scale: str
    zero_point: str


@dataclass(frozen=True)
class DequantizeLinearOp(RenderableOpBase):
    __io_inputs__ = ("input0", "scale", "zero_point")
    __io_outputs__ = ("output",)
    input0: str
    scale: str
    zero_point: str | None
    output: str
    axis: int | None
    block_size: int | None


@dataclass(frozen=True)
class ConcatOp(RenderableOpBase):
    __io_inputs__ = ("inputs",)
    __io_outputs__ = ("output",)
    inputs: tuple[str, ...]
    output: str
    axis: int


@dataclass(frozen=True)
class CompressOp(RenderableOpBase):
    __io_inputs__ = ("data", "condition")
    __io_outputs__ = ("output",)
    data: str
    condition: str
    output: str
    axis: int | None


@dataclass(frozen=True)
class GatherElementsOp(RenderableOpBase):
    __io_inputs__ = ("data", "indices")
    __io_outputs__ = ("output",)
    data: str
    indices: str
    output: str
    axis: int


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


@dataclass(frozen=True)
class ScatterNDOp(RenderableOpBase):
    __io_inputs__ = ("data", "indices", "updates")
    __io_outputs__ = ("output",)
    data: str
    indices: str
    updates: str
    output: str
    reduction: str


@dataclass(frozen=True)
class ScatterOp(RenderableOpBase):
    __io_inputs__ = ("data", "indices", "updates")
    __io_outputs__ = ("output",)
    data: str
    indices: str
    updates: str
    output: str
    axis: int


@dataclass(frozen=True)
class TensorScatterOp(RenderableOpBase):
    __io_inputs__ = ("past_cache", "update", "write_indices")
    __io_outputs__ = ("output",)
    past_cache: str
    update: str
    write_indices: str | None
    output: str
    axis: int
    mode: str

    def infer_types(self, ctx: OpContext) -> None:
        past_dtype = ctx.dtype(self.past_cache)
        update_dtype = ctx.dtype(self.update)
        if past_dtype != update_dtype:
            raise UnsupportedOpError(
                f"{self.kind} expects matching past_cache/update dtypes, "
                f"got {past_dtype.onnx_name} and {update_dtype.onnx_name}"
            )
        if self.write_indices is not None:
            write_dtype = ctx.dtype(self.write_indices)
            if write_dtype not in {ScalarType.I32, ScalarType.I64}:
                raise UnsupportedOpError(
                    f"{self.kind} write_indices must be int32 or int64, "
                    f"got {write_dtype.onnx_name}"
                )
        try:
            output_dtype = ctx.dtype(self.output)
        except ShapeInferenceError:
            ctx.set_dtype(self.output, past_dtype)
            output_dtype = past_dtype
        if output_dtype != past_dtype:
            raise UnsupportedOpError(
                f"{self.kind} expects output dtype {past_dtype.onnx_name}, "
                f"got {output_dtype.onnx_name}"
            )

    def infer_shapes(self, ctx: OpContext) -> None:
        past_shape = ctx.shape(self.past_cache)
        update_shape = ctx.shape(self.update)
        output_shape = ctx.shape(self.output)
        if output_shape != past_shape:
            raise ShapeInferenceError(
                f"{self.kind} output shape must match past_cache shape, "
                f"got {output_shape} and {past_shape}"
            )
        if len(update_shape) != len(past_shape):
            raise ShapeInferenceError(
                f"{self.kind} update rank must match past_cache rank, "
                f"got {len(update_shape)} and {len(past_shape)}"
            )
        for index, (past_dim, update_dim) in enumerate(zip(past_shape, update_shape)):
            if index == self.axis:
                if update_dim > past_dim:
                    raise ShapeInferenceError(
                        f"{self.kind} update dim at axis {self.axis} must not exceed "
                        f"past_cache dim {past_dim}, got {update_dim}"
                    )
                continue
            if update_dim != past_dim:
                raise ShapeInferenceError(
                    f"{self.kind} update shape must match past_cache outside axis {self.axis}, "
                    f"got {update_shape} vs {past_shape}"
                )
        if self.write_indices is not None:
            write_shape = ctx.shape(self.write_indices)
            if len(write_shape) != 1:
                raise ShapeInferenceError(
                    f"{self.kind} write_indices must be rank-1, got shape {write_shape}"
                )
            if write_shape[0] != past_shape[0]:
                raise ShapeInferenceError(
                    f"{self.kind} write_indices length must match batch dim {past_shape[0]}, got {write_shape[0]}"
                )


@dataclass(frozen=True)
class TransposeOp(RenderableOpBase):
    __io_inputs__ = ("input0",)
    __io_outputs__ = ("output",)
    input0: str
    output: str
    perm: tuple[int, ...]

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

    def infer_shapes(self, ctx: OpContext) -> None:
        input_shape = ctx.shape(self.input0)
        output_shape = ctx.shape(self.output)
        if _shape_product(input_shape) != _shape_product(output_shape):
            raise ShapeInferenceError(
                f"{self.kind} input/output element counts must match, "
                f"got {input_shape} and {output_shape}"
            )


@dataclass(frozen=True)
class EyeLikeOp(RenderableOpBase):
    __io_inputs__ = ("input0",)
    __io_outputs__ = ("output",)
    input0: str
    output: str
    k: int


@dataclass(frozen=True)
class BernoulliOp(RenderableOpBase):
    __io_inputs__ = ("input0",)
    __io_outputs__ = ("output",)
    input0: str
    output: str
    seed: int | None


@dataclass(frozen=True)
class TriluOp(RenderableOpBase):
    __io_inputs__ = ("input0", "k_input")
    __io_outputs__ = ("output",)
    input0: str
    output: str
    upper: bool
    k_value: int
    k_input: str | None

    def call_args(self) -> tuple[str, ...]:
        args = [self.input0, self.output]
        if self.k_input is not None:
            args.append(self.k_input)
        return tuple(args)


@dataclass(frozen=True)
class TileOp(RenderableOpBase):
    __io_inputs__ = ("input0", "repeats_input")
    __io_outputs__ = ("output",)
    input0: str
    repeats_input: str
    output: str


@dataclass(frozen=True)
class PadOp(RenderableOpBase):
    __io_inputs__ = ("input0", "pads_input", "axes_input", "value_input")
    __io_outputs__ = ("output",)
    input0: str
    output: str
    pads_begin: tuple[int, ...] | None
    pads_end: tuple[int, ...] | None
    pads_input: str | None
    pads_values: tuple[int, ...] | None
    axes_input: str | None
    mode: str
    value: float | int | bool
    value_input: str | None


@dataclass(frozen=True)
class DepthToSpaceOp(RenderableOpBase):
    __io_inputs__ = ("input0",)
    __io_outputs__ = ("output",)
    input0: str
    output: str
    blocksize: int
    mode: str


@dataclass(frozen=True)
class SpaceToDepthOp(RenderableOpBase):
    __io_inputs__ = ("input0",)
    __io_outputs__ = ("output",)
    input0: str
    output: str
    blocksize: int


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
    starts: tuple[int, ...] | None
    steps: tuple[int, ...] | None
    axes: tuple[int, ...] | None
    starts_input: str | None
    ends_input: str | None
    axes_input: str | None
    steps_input: str | None


@dataclass(frozen=True)
class ResizeOp(RenderableOpBase):
    __io_inputs__ = ("input0", "roi_input", "scales_input", "sizes_input")
    __io_outputs__ = ("output",)
    input0: str
    output: str
    scales: tuple[float, ...]
    scales_input: str | None
    sizes_input: str | None
    roi_input: str | None
    axes: tuple[int, ...]
    mode: str
    coordinate_transformation_mode: str
    nearest_mode: str
    cubic_coeff_a: float
    exclude_outside: bool
    extrapolation_value: float
    antialias: bool
    keep_aspect_ratio_policy: str


@dataclass(frozen=True)
class GridSampleOp(RenderableOpBase):
    __io_inputs__ = ("input0", "grid")
    __io_outputs__ = ("output",)
    input0: str
    grid: str
    output: str
    mode: str
    padding_mode: str
    align_corners: bool

    def infer_types(self, ctx: OpContext) -> None:
        input_dtype = ctx.dtype(self.input0)
        grid_dtype = ctx.dtype(self.grid)
        if not grid_dtype.is_float:
            raise UnsupportedOpError(
                f"{self.kind} grid dtype must be float, got {grid_dtype.onnx_name}"
            )
        try:
            output_dtype = ctx.dtype(self.output)
        except ShapeInferenceError:
            ctx.set_dtype(self.output, input_dtype)
            output_dtype = input_dtype
        if output_dtype != input_dtype:
            raise UnsupportedOpError(
                f"{self.kind} output dtype must match input dtype {input_dtype.onnx_name}, "
                f"got {output_dtype.onnx_name}"
            )

    def infer_shapes(self, ctx: OpContext) -> None:
        input_shape = ctx.shape(self.input0)
        grid_shape = ctx.shape(self.grid)
        output_shape = ctx.shape(self.output)
        if len(input_shape) < 3:
            raise ShapeInferenceError(
                f"{self.kind} input rank must be >= 3, got {len(input_shape)}"
            )
        spatial_rank = len(input_shape) - 2
        if spatial_rank > 3:
            raise UnsupportedOpError(
                f"{self.kind} supports up to 3 spatial dims, got {spatial_rank}"
            )
        if len(grid_shape) != spatial_rank + 2:
            raise ShapeInferenceError(
                f"{self.kind} grid rank must be {spatial_rank + 2}, got {len(grid_shape)}"
            )
        if len(output_shape) != len(input_shape):
            raise ShapeInferenceError(
                f"{self.kind} output rank must match input rank {len(input_shape)}, got {len(output_shape)}"
            )
        if output_shape[:2] != input_shape[:2]:
            raise ShapeInferenceError(
                f"{self.kind} output batch/channels must match input, got {output_shape[:2]} vs {input_shape[:2]}"
            )
        if output_shape[2:] != grid_shape[1:-1]:
            raise ShapeInferenceError(
                f"{self.kind} output spatial dims must match grid dims {grid_shape[1:-1]}, got {output_shape[2:]}"
            )
        if grid_shape[-1] != spatial_rank:
            raise ShapeInferenceError(
                f"{self.kind} grid last dim must equal spatial rank {spatial_rank}, got {grid_shape[-1]}"
            )


@dataclass(frozen=True)
class ConstantOfShapeOp(RenderableOpBase):
    __io_inputs__ = ("input0",)
    __io_outputs__ = ("output",)
    input0: str
    output: str
    value: float | int | bool

    def emit(self, emitter: Emitter, ctx: EmitContext) -> str:
        return emitter.emit_generic_op(self, ctx)


@dataclass(frozen=True)
class ShapeOp(RenderableOpBase):
    __io_inputs__ = ("input0",)
    __io_outputs__ = ("output",)
    input0: str
    output: str
    start: int | None
    end: int | None

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
class UniqueOp(RenderableOpBase):
    __io_inputs__ = ("input0",)
    __io_outputs__ = ("y", "indices", "inverse_indices", "counts")
    input0: str
    y: str
    indices: str
    inverse_indices: str
    counts: str
    axis: int | None
    sorted: bool

    def infer_types(self, ctx: OpContext) -> None:
        input_dtype = ctx.dtype(self.input0)
        if input_dtype == ScalarType.STRING:
            raise UnsupportedOpError(f"{self.kind} does not support string input")
        y_dtype = ctx.dtype(self.y)
        if y_dtype != input_dtype:
            raise UnsupportedOpError(
                f"{self.kind} Y dtype must match input dtype {input_dtype.onnx_name}, got {y_dtype.onnx_name}"
            )
        for name in (self.indices, self.inverse_indices, self.counts):
            output_dtype = ctx.dtype(name)
            if output_dtype != ScalarType.I64:
                raise UnsupportedOpError(
                    f"{self.kind} output {name} must be int64, got {output_dtype.onnx_name}"
                )

    def infer_shapes(self, ctx: OpContext) -> None:
        input_shape = ctx.shape(self.input0)
        y_shape = ctx.shape(self.y)
        indices_shape = ctx.shape(self.indices)
        inverse_shape = ctx.shape(self.inverse_indices)
        counts_shape = ctx.shape(self.counts)
        if len(indices_shape) != 1 or len(counts_shape) != 1:
            raise ShapeInferenceError(
                f"{self.kind} indices and counts outputs must be rank-1"
            )
        if self.axis is None:
            if len(y_shape) != 1:
                raise ShapeInferenceError(f"{self.kind} Y output must be rank-1")
            if len(inverse_shape) != 1:
                raise ShapeInferenceError(
                    f"{self.kind} inverse_indices output must be rank-1"
                )
            if inverse_shape[0] != _shape_product(input_shape):
                raise ShapeInferenceError(
                    f"{self.kind} inverse_indices length must be {_shape_product(input_shape)}, got {inverse_shape[0]}"
                )
            if y_shape[0] != indices_shape[0] or y_shape[0] != counts_shape[0]:
                raise ShapeInferenceError(
                    f"{self.kind} Y/indices/counts lengths must match, got {y_shape[0]}, {indices_shape[0]}, {counts_shape[0]}"
                )
            return None
        axis = self.axis
        rank = len(input_shape)
        if axis < 0 or axis >= rank:
            raise ShapeInferenceError(
                f"{self.kind} axis {axis} out of range for rank {rank}"
            )
        if len(y_shape) != rank:
            raise ShapeInferenceError(
                f"{self.kind} Y output rank must be {rank}, got {len(y_shape)}"
            )
        for dim_index, (in_dim, out_dim) in enumerate(zip(input_shape, y_shape)):
            if dim_index == axis:
                continue
            if in_dim != out_dim:
                raise ShapeInferenceError(
                    f"{self.kind} Y output shape must match input outside axis {axis}, got {y_shape} for input {input_shape}"
                )
        if indices_shape[0] != y_shape[axis] or counts_shape[0] != y_shape[axis]:
            raise ShapeInferenceError(
                f"{self.kind} indices/counts length must match Y axis {axis} size {y_shape[axis]}"
            )
        if len(inverse_shape) != 1 or inverse_shape[0] != input_shape[axis]:
            raise ShapeInferenceError(
                f"{self.kind} inverse_indices length must be input axis {axis} size {input_shape[axis]}"
            )


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
    axis: int | None
    output: str
    exclusive: bool
    reverse: bool


@dataclass(frozen=True)
class LoopRangeOp(RenderableOpBase):
    __io_inputs__ = ("trip_count", "cond", "start", "delta")
    __io_outputs__ = ("final", "output")
    trip_count: str
    cond: str
    start: str
    delta: str
    final: str
    output: str
    add_table_data: tuple[float | int, ...] | None = None
    add_table_shape: tuple[int, ...] | None = None


@dataclass(frozen=True)
class LoopSequenceInsertOp(RenderableOpBase):
    __io_inputs__ = ("trip_count", "cond", "input_sequence")
    __io_outputs__ = ("output_sequence",)
    trip_count: str
    cond: str
    input_sequence: str
    output_sequence: str
    table_data: tuple[float | int, ...]
    table_shape: tuple[int, ...]
    elem_shape: tuple[int, ...]
    elem_dtype: ScalarType

    def call_args(self) -> tuple[str, ...]:
        return (
            self.trip_count,
            self.cond,
            self.input_sequence,
            f"{self.input_sequence}__count",
            self.output_sequence,
            f"{self.output_sequence}__count",
        )


@dataclass(frozen=True)
class LoopSequenceMapOp(RenderableOpBase):
    __io_inputs__ = ("trip_count", "cond")
    __io_outputs__ = ("output_sequences",)
    trip_count: str
    cond: str
    input_sequences: tuple[str, ...]
    input_tensors: tuple[str, ...]
    output_sequences: tuple[str, ...]
    output_kinds: tuple[str, ...]
    output_input0: tuple[str, ...]
    output_input1: tuple[str | None, ...]
    output_input0_is_sequence: tuple[bool, ...]
    output_input1_is_sequence: tuple[bool, ...]
    output_elem_shapes: tuple[tuple[int, ...], ...]
    output_elem_dtypes: tuple[ScalarType, ...]

    def call_args(self) -> tuple[str, ...]:
        args: list[str] = [self.trip_count, self.cond]
        for name in self.input_sequences:
            args.extend((name, f"{name}__count"))
        for name in self.input_tensors:
            args.append(name)
        for name in self.output_sequences:
            args.extend((name, f"{name}__count"))
        return tuple(args)


@dataclass(frozen=True)
class RangeOp(RenderableOpBase):
    __io_inputs__ = ("start", "limit", "delta")
    __io_outputs__ = ("output",)
    start: str
    limit: str
    delta: str
    output: str


@dataclass(frozen=True)
class HammingWindowOp(RenderableOpBase):
    __io_inputs__ = ("size",)
    __io_outputs__ = ("output",)
    size: str
    output: str
    periodic: bool


@dataclass(frozen=True)
class HannWindowOp(RenderableOpBase):
    __io_inputs__ = ("size",)
    __io_outputs__ = ("output",)
    size: str
    output: str
    periodic: bool


@dataclass(frozen=True)
class OneHotOp(RenderableOpBase):
    __io_inputs__ = ("indices", "depth", "values")
    __io_outputs__ = ("output",)
    indices: str
    depth: str
    values: str
    output: str
    axis: int


@dataclass(frozen=True)
class TfIdfVectorizerOp(RenderableOpBase):
    __io_inputs__ = ("input0",)
    __io_outputs__ = ("output",)
    input0: str
    output: str
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
    case_change_action: str
    is_case_sensitive: bool
    stopwords: tuple[str, ...]


@dataclass(frozen=True)
class TreeEnsembleClassifierOp(RenderableOpBase):
    __io_inputs__ = ("input0",)
    __io_outputs__ = ("label", "probabilities")
    input0: str
    label: str
    probabilities: str
    output: str
    post_transform: str
    class_labels: tuple[int, ...]
    node_tree_ids: tuple[int, ...]
    node_node_ids: tuple[int, ...]
    node_feature_ids: tuple[int, ...]
    node_modes: tuple[int, ...]
    node_values: tuple[float, ...]
    node_true_ids: tuple[int, ...]
    node_false_ids: tuple[int, ...]
    class_tree_ids: tuple[int, ...]
    class_node_ids: tuple[int, ...]
    class_ids: tuple[int, ...]
    class_weights: tuple[float, ...]
    dtype: ScalarType = ScalarType.F32
    output_shape: tuple[int, ...] = ()


@dataclass(frozen=True)
class TreeEnsembleOp(RenderableOpBase):
    __io_inputs__ = ("input0",)
    __io_outputs__ = ("output",)
    input0: str
    output: str
    aggregate_function: int
    post_transform: int
    tree_roots: tuple[int, ...]
    node_feature_ids: tuple[int, ...]
    node_modes: tuple[int, ...]
    node_splits: tuple[float, ...]
    node_true_ids: tuple[int, ...]
    node_true_leafs: tuple[int, ...]
    node_false_ids: tuple[int, ...]
    node_false_leafs: tuple[int, ...]
    membership_values: tuple[float, ...] | None
    member_range_starts: tuple[int, ...]
    member_range_ends: tuple[int, ...]
    leaf_target_ids: tuple[int, ...]
    leaf_weights: tuple[float, ...]
    n_targets: int


@dataclass(frozen=True)
class SplitOp(RenderableOpBase):
    __io_inputs__ = ("input0",)
    __io_outputs__ = ("outputs",)
    input0: str
    outputs: tuple[str, ...]
    axis: int
    split_sizes: tuple[int, ...]


@dataclass(frozen=True)
class SplitToSequenceOp(RenderableOpBase):
    __io_inputs__ = ("input0", "split")
    __io_outputs__ = ("output_sequence",)
    input0: str
    split: str | None
    output_sequence: str
    axis: int
    keepdims: bool
    split_sizes: tuple[int, ...] | None
    split_scalar: bool

    def call_args(self) -> tuple[str, ...]:
        args = [self.input0]
        if self.split is not None:
            args.append(self.split)
        args.extend([self.output_sequence, f"{self.output_sequence}__count"])
        return tuple(args)


@dataclass(frozen=True)
class ReverseSequenceOp(RenderableOpBase):
    __io_inputs__ = ("input0", "sequence_lens")
    __io_outputs__ = ("output",)
    input0: str
    sequence_lens: str
    output: str
    batch_axis: int
    time_axis: int


@dataclass(frozen=True)
class ConcatFromSequenceOp(RenderableOpBase):
    __io_inputs__ = ("input_sequence",)
    __io_outputs__ = ("output",)
    input_sequence: str
    output: str
    axis: int
    new_axis: bool
    elem_shape: tuple[int, ...]

    def call_args(self) -> tuple[str, ...]:
        return (
            self.input_sequence,
            f"{self.input_sequence}__count",
            self.output,
        )


@dataclass(frozen=True)
class SequenceAtOp(RenderableOpBase):
    __io_inputs__ = ("input_sequence", "position")
    __io_outputs__ = ("output",)
    input_sequence: str
    position: str
    output: str

    def call_args(self) -> tuple[str, ...]:
        return (
            self.input_sequence,
            f"{self.input_sequence}__count",
            self.position,
            self.output,
        )


@dataclass(frozen=True)
class SequenceLengthOp(RenderableOpBase):
    __io_inputs__ = ("input_sequence",)
    __io_outputs__ = ("output",)
    input_sequence: str
    output: str

    def call_args(self) -> tuple[str, ...]:
        return (
            self.input_sequence,
            f"{self.input_sequence}__count",
            self.output,
        )


@dataclass(frozen=True)
class SequenceIdentityOp(RenderableOpBase):
    __io_inputs__ = ("input_sequence",)
    __io_outputs__ = ("output_sequence",)
    input_sequence: str
    output_sequence: str
    input_present: str | None = None
    output_present: str | None = None

    def call_args(self) -> tuple[str, ...]:
        args = [
            self.input_sequence,
            f"{self.input_sequence}__count",
            self.output_sequence,
            f"{self.output_sequence}__count",
        ]
        if self.input_present is not None and self.output_present is not None:
            args.extend([self.input_present, self.output_present])
        return tuple(args)


@dataclass(frozen=True)
class SequenceInsertOp(RenderableOpBase):
    __io_inputs__ = ("input_sequence", "tensor", "position")
    __io_outputs__ = ("output_sequence",)
    input_sequence: str
    tensor: str
    position: str | None
    output_sequence: str

    def call_args(self) -> tuple[str, ...]:
        args = [self.input_sequence, f"{self.input_sequence}__count", self.tensor]
        if self.position is not None:
            args.append(self.position)
        args.extend([self.output_sequence, f"{self.output_sequence}__count"])
        return tuple(args)


@dataclass(frozen=True)
class SequenceEraseOp(RenderableOpBase):
    __io_inputs__ = ("input_sequence", "position")
    __io_outputs__ = ("output_sequence",)
    input_sequence: str
    position: str | None
    output_sequence: str

    def call_args(self) -> tuple[str, ...]:
        args = [self.input_sequence, f"{self.input_sequence}__count"]
        if self.position is not None:
            args.append(self.position)
        args.extend([self.output_sequence, f"{self.output_sequence}__count"])
        return tuple(args)


@dataclass(frozen=True)
class SequenceConstructOp(RenderableOpBase):
    __io_inputs__ = ("inputs",)
    __io_outputs__ = ("output_sequence",)
    inputs: tuple[str, ...]
    output_sequence: str

    def call_args(self) -> tuple[str, ...]:
        return (*self.inputs, self.output_sequence, f"{self.output_sequence}__count")


@dataclass(frozen=True)
class SequenceEmptyOp(RenderableOpBase):
    __io_inputs__ = ()
    __io_outputs__ = ("output_sequence",)
    output_sequence: str

    def call_args(self) -> tuple[str, ...]:
        return (self.output_sequence, f"{self.output_sequence}__count")

from __future__ import annotations

from dataclasses import dataclass

from shared.scalar_types import ScalarType

from ..op_base import ReduceOpBase
from ..op_context import OpContext


@dataclass(frozen=True)
class ReduceOp(ReduceOpBase):
    input0: str
    output: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    axes: tuple[int, ...]
    axes_input: str | None
    axes_input_shape: tuple[int, ...] | None
    axes_input_dtype: ScalarType | None
    keepdims: bool
    noop_with_empty_axes: bool
    reduce_kind: str
    reduce_count: int | None
    dtype: ScalarType

    def infer_types(self, ctx: OpContext) -> None:
        dtype = ctx.dtype(self.output)
        object.__setattr__(self, "dtype", dtype)

    def infer_shapes(self, ctx: OpContext) -> None:
        input_shape = ctx.shape(self.input0)
        if self.axes_input is None:
            axes = self.normalize_axes(self.axes, len(input_shape))
            output_shape = self.reduced_shape(
                input_shape, axes, keepdims=self.keepdims
            )
        else:
            axes = self.axes
            output_shape = ctx.shape(self.output)
        ctx.set_shape(self.output, output_shape)
        object.__setattr__(self, "input_shape", input_shape)
        object.__setattr__(self, "output_shape", output_shape)
        object.__setattr__(self, "axes", axes)


@dataclass(frozen=True)
class ArgReduceOp(ReduceOpBase):
    input0: str
    output: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    axis: int
    keepdims: bool
    select_last_index: bool
    reduce_kind: str
    input_dtype: ScalarType
    output_dtype: ScalarType

    def infer_types(self, ctx: OpContext) -> None:
        object.__setattr__(self, "input_dtype", ctx.dtype(self.input0))
        object.__setattr__(self, "output_dtype", ctx.dtype(self.output))

    def infer_shapes(self, ctx: OpContext) -> None:
        input_shape = ctx.shape(self.input0)
        axes = self.normalize_axes((self.axis,), len(input_shape))
        output_shape = self.reduced_shape(
            input_shape, axes, keepdims=self.keepdims
        )
        ctx.set_shape(self.output, output_shape)
        object.__setattr__(self, "input_shape", input_shape)
        object.__setattr__(self, "output_shape", output_shape)
        object.__setattr__(self, "axis", axes[0])


@dataclass(frozen=True)
class TopKOp(ReduceOpBase):
    input0: str
    output_values: str
    output_indices: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    axis: int
    k: int
    largest: bool
    sorted: bool
    input_dtype: ScalarType
    output_values_dtype: ScalarType
    output_indices_dtype: ScalarType

    def infer_types(self, ctx: OpContext) -> None:
        object.__setattr__(self, "input_dtype", ctx.dtype(self.input0))
        object.__setattr__(self, "output_values_dtype", ctx.dtype(self.output_values))
        object.__setattr__(self, "output_indices_dtype", ctx.dtype(self.output_indices))

    def infer_shapes(self, ctx: OpContext) -> None:
        input_shape = ctx.shape(self.input0)
        output_shape = ctx.shape(self.output_values)
        object.__setattr__(self, "input_shape", input_shape)
        object.__setattr__(self, "output_shape", output_shape)
        ctx.set_shape(self.output_values, output_shape)
        ctx.set_shape(self.output_indices, ctx.shape(self.output_indices))

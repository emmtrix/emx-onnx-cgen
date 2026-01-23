from __future__ import annotations

from dataclasses import dataclass

from ..op_base import ReduceOpBase
from ..op_context import OpContext


@dataclass(frozen=True)
class ReduceOp(ReduceOpBase):
    input0: str
    output: str
    axes: tuple[int, ...]
    axes_input: str | None
    keepdims: bool
    noop_with_empty_axes: bool
    reduce_kind: str
    reduce_count: int | None

    def infer_types(self, ctx: OpContext) -> None:
        ctx.dtype(self.output)

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
        ctx.set_derived(self, "axes", axes)


@dataclass(frozen=True)
class ArgReduceOp(ReduceOpBase):
    input0: str
    output: str
    axis: int
    keepdims: bool
    select_last_index: bool
    reduce_kind: str

    def infer_types(self, ctx: OpContext) -> None:
        ctx.dtype(self.input0)
        ctx.dtype(self.output)

    def infer_shapes(self, ctx: OpContext) -> None:
        input_shape = ctx.shape(self.input0)
        axes = self.normalize_axes((self.axis,), len(input_shape))
        output_shape = self.reduced_shape(
            input_shape, axes, keepdims=self.keepdims
        )
        ctx.set_shape(self.output, output_shape)
        ctx.set_derived(self, "axis", axes[0])


@dataclass(frozen=True)
class TopKOp(ReduceOpBase):
    input0: str
    k_input: str
    output_values: str
    output_indices: str
    axis: int
    k: int
    largest: bool
    sorted: bool

    def infer_types(self, ctx: OpContext) -> None:
        ctx.dtype(self.input0)
        ctx.dtype(self.output_values)
        ctx.dtype(self.output_indices)

    def infer_shapes(self, ctx: OpContext) -> None:
        input_shape = ctx.shape(self.input0)
        output_shape = ctx.shape(self.output_values)
        ctx.set_shape(self.output_values, output_shape)
        ctx.set_shape(self.output_indices, ctx.shape(self.output_indices))

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol

from shared.scalar_types import ScalarType

from ..errors import ShapeInferenceError, UnsupportedOpError
from .op_context import OpContext


class Emitter(Protocol):
    def render_op(self, op: "OpBase", ctx: "EmitContext") -> str:
        ...


@dataclass(frozen=True)
class EmitContext:
    op_index: int


class OpBase(ABC):
    """Ops should not mutate themselves; store derived values in OpContext."""
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]

    def __getattr__(self, name: str) -> str:
        if name == "kind":
            return self.__class__.__name__
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    def validate(self, ctx: OpContext) -> None:
        return None

    def infer_types(self, ctx: OpContext) -> None:
        return None

    def infer_shapes(self, ctx: OpContext) -> None:
        return None

    @abstractmethod
    def emit(self, emitter: Emitter, ctx: EmitContext) -> str:
        raise NotImplementedError


class RenderableOpBase(OpBase):
    def emit(self, emitter: Emitter, ctx: EmitContext) -> str:
        return emitter.render_op(self, ctx)


class ElementwiseOpBase(RenderableOpBase):
    """Elementwise ops should validate against OpContext and store no derived state."""

    def _elementwise_inputs(self) -> tuple[str, ...]:
        raise NotImplementedError

    def _elementwise_output(self) -> str:
        raise NotImplementedError

    def _elementwise_condition_inputs(self) -> tuple[str, ...]:
        return ()

    def _elementwise_compare(self) -> bool:
        return False

    def _elementwise_data_inputs(self) -> tuple[str, ...]:
        inputs = self._elementwise_inputs()
        condition_inputs = set(self._elementwise_condition_inputs())
        return tuple(name for name in inputs if name not in condition_inputs)

    def validate(self, ctx: OpContext) -> None:
        condition_inputs = self._elementwise_condition_inputs()
        for name in condition_inputs:
            dtype = ctx.dtype(name)
            if dtype != ScalarType.BOOL:
                raise UnsupportedOpError(
                    f"{self.kind} expects bool condition, got {dtype.onnx_name}"
                )
        data_inputs = self._elementwise_data_inputs()
        if not data_inputs:
            return None
        data_dtypes = tuple(ctx.dtype(name) for name in data_inputs)
        if any(dtype != data_dtypes[0] for dtype in data_dtypes[1:]):
            dtype_names = ", ".join(dtype.onnx_name for dtype in data_dtypes)
            raise UnsupportedOpError(
                f"{self.kind} expects matching input dtypes, got {dtype_names}"
            )
        output_dtype = ctx.dtype(self._elementwise_output())
        if self._elementwise_compare():
            if output_dtype != ScalarType.BOOL:
                raise UnsupportedOpError(
                    f"{self.kind} expects bool output, got {output_dtype.onnx_name}"
                )
            return None
        if output_dtype != data_dtypes[0]:
            raise UnsupportedOpError(
                f"{self.kind} expects output dtype {data_dtypes[0].onnx_name}, "
                f"got {output_dtype.onnx_name}"
            )
        return None

    def infer_types(self, ctx: OpContext) -> None:
        input_names = self._elementwise_inputs()
        output_name = self._elementwise_output()
        for name in input_names:
            ctx.dtype(name)
        ctx.dtype(output_name)

    def infer_shapes(self, ctx: OpContext) -> None:
        input_names = self._elementwise_inputs()
        output_name = self._elementwise_output()
        input_shapes = tuple(ctx.shape(name) for name in input_names)
        if len(input_shapes) == 1:
            output_shape = input_shapes[0]
        else:
            output_shape = BroadcastingOpBase.broadcast_shapes(*input_shapes)
        ctx.set_shape(output_name, output_shape)
        return None


class ReduceOpBase(RenderableOpBase):
    @staticmethod
    def normalize_axes(
        axes: tuple[int, ...] | None, rank: int
    ) -> tuple[int, ...]:
        if axes is None:
            axes = tuple(range(rank))
        normalized: list[int] = []
        for axis in axes:
            if axis < 0:
                axis += rank
            if axis < 0 or axis >= rank:
                raise ShapeInferenceError(
                    f"Reduce axis {axis} is out of bounds for rank {rank}"
                )
            normalized.append(axis)
        return tuple(dict.fromkeys(normalized))

    @staticmethod
    def reduced_shape(
        input_shape: tuple[int, ...],
        axes: tuple[int, ...] | None,
        *,
        keepdims: bool,
    ) -> tuple[int, ...]:
        rank = len(input_shape)
        normalized_axes = ReduceOpBase.normalize_axes(axes, rank)
        if keepdims:
            return tuple(
                1 if axis in normalized_axes else dim
                for axis, dim in enumerate(input_shape)
            )
        return tuple(
            dim for axis, dim in enumerate(input_shape) if axis not in normalized_axes
        )


class BroadcastingOpBase(RenderableOpBase):
    @staticmethod
    def broadcast_shapes(
        *shapes: tuple[int, ...],
    ) -> tuple[int, ...]:
        if not shapes:
            return ()
        max_rank = max(len(shape) for shape in shapes)
        padded_shapes = [
            (1,) * (max_rank - len(shape)) + shape for shape in shapes
        ]
        result: list[int] = []
        for dims in zip(*padded_shapes):
            dim = max(dims)
            if any(d not in {1, dim} for d in dims):
                raise ShapeInferenceError(
                    "Broadcasting mismatch for shapes: "
                    + ", ".join(str(shape) for shape in shapes)
                )
            result.append(dim)
        return tuple(result)


class MatMulLikeOpBase(RenderableOpBase):
    pass


class GemmLikeOpBase(RenderableOpBase):
    pass


class ConvLikeOpBase(RenderableOpBase):
    pass

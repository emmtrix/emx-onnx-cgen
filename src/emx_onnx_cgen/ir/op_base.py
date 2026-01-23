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
        try:
            output_dtype = ctx.dtype(self._elementwise_output())
        except ShapeInferenceError:
            return None
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
        desired_dtype = (
            ScalarType.BOOL if self._elementwise_compare() else None
        )
        if desired_dtype is None:
            data_inputs = self._elementwise_data_inputs()
            if data_inputs:
                desired_dtype = ctx.dtype(data_inputs[0])
        try:
            output_dtype = ctx.dtype(output_name)
        except ShapeInferenceError:
            if desired_dtype is not None:
                ctx.set_dtype(output_name, desired_dtype)
                return None
            raise
        if desired_dtype is not None and output_dtype != desired_dtype:
            raise UnsupportedOpError(
                f"{self.kind} expects output dtype {desired_dtype.onnx_name}, "
                f"got {output_dtype.onnx_name}"
            )

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


class GatherLikeOpBase(RenderableOpBase):
    def _gather_data(self) -> str:
        raise NotImplementedError

    def _gather_indices(self) -> str:
        raise NotImplementedError

    def _gather_output(self) -> str:
        raise NotImplementedError

    def _gather_axis(self) -> int:
        raise NotImplementedError

    def _gather_mode(self) -> str:
        raise NotImplementedError

    def validate(self, ctx: OpContext) -> None:
        indices_dtype = ctx.dtype(self._gather_indices())
        if indices_dtype not in {ScalarType.I64, ScalarType.I32}:
            raise UnsupportedOpError(
                f"{self.kind} indices must be int32 or int64, "
                f"got {indices_dtype.onnx_name}"
            )
        data_shape = ctx.shape(self._gather_data())
        if self._gather_mode() in {"gather", "gather_elements"}:
            if not data_shape:
                raise ShapeInferenceError(
                    f"{self.kind} does not support scalar inputs"
                )
            axis = self._gather_axis()
            if axis < 0:
                axis += len(data_shape)
            if axis < 0 or axis >= len(data_shape):
                raise ShapeInferenceError(
                    f"{self.kind} axis {axis} is out of range for rank "
                    f"{len(data_shape)}"
                )
        return None

    def infer_types(self, ctx: OpContext) -> None:
        data_dtype = ctx.dtype(self._gather_data())
        try:
            output_dtype = ctx.dtype(self._gather_output())
        except ShapeInferenceError:
            ctx.set_dtype(self._gather_output(), data_dtype)
            output_dtype = data_dtype
        if output_dtype != data_dtype:
            raise UnsupportedOpError(
                f"{self.kind} expects output dtype {data_dtype.onnx_name}, "
                f"got {output_dtype.onnx_name}"
            )

    def infer_shapes(self, ctx: OpContext) -> None:
        data_shape = ctx.shape(self._gather_data())
        indices_shape = ctx.shape(self._gather_indices())
        axis = self._gather_axis()
        if axis < 0:
            axis += len(data_shape)
        if axis < 0 or axis >= len(data_shape):
            raise ShapeInferenceError(
                f"{self.kind} axis {axis} is out of range for rank "
                f"{len(data_shape)}"
            )
        if self._gather_mode() == "gather":
            output_shape = (
                data_shape[:axis] + indices_shape + data_shape[axis + 1 :]
            )
        else:
            raise UnsupportedOpError(
                f"{self.kind} does not support gather mode "
                f"{self._gather_mode()}"
            )
        try:
            expected = ctx.shape(self._gather_output())
        except ShapeInferenceError:
            expected = None
        if expected is not None and expected != output_shape:
            raise ShapeInferenceError(
                f"{self.kind} output shape must be {output_shape}, got {expected}"
            )
        ctx.set_shape(self._gather_output(), output_shape)
        ctx.set_derived(self, "axis", axis)


class ShapeLikeOpBase(RenderableOpBase):
    def _shape_data(self) -> str:
        raise NotImplementedError

    def _shape_output(self) -> str:
        raise NotImplementedError

    def _shape_spec(self, ctx: OpContext) -> tuple[int, ...]:
        raise NotImplementedError

    def _shape_mode(self) -> str:
        raise NotImplementedError

    def _shape_derived(
        self,
        ctx: OpContext,
        *,
        data_shape: tuple[int, ...],
        target_shape: tuple[int, ...],
        output_shape: tuple[int, ...],
    ) -> None:
        return None

    @staticmethod
    def _validate_static_dims(shape: tuple[int, ...], kind: str) -> None:
        if any(dim < 0 for dim in shape):
            raise ShapeInferenceError(
                f"{kind} does not support dynamic dims"
            )

    @staticmethod
    def _broadcast_shape(
        input_shape: tuple[int, ...],
        target_shape: tuple[int, ...],
        *,
        kind: str,
    ) -> tuple[int, ...]:
        ShapeLikeOpBase._validate_static_dims(input_shape, kind)
        ShapeLikeOpBase._validate_static_dims(target_shape, kind)
        output_rank = max(len(input_shape), len(target_shape))
        input_padded = (1,) * (output_rank - len(input_shape)) + input_shape
        target_padded = (1,) * (output_rank - len(target_shape)) + target_shape
        result: list[int] = []
        for input_dim, target_dim in zip(input_padded, target_padded):
            if input_dim == 1:
                result.append(target_dim)
            elif target_dim == 1:
                result.append(input_dim)
            elif input_dim == target_dim:
                result.append(input_dim)
            else:
                raise ShapeInferenceError(
                    f"{kind} input shape {input_shape} is not "
                    f"broadcastable to {target_shape}"
                )
        return tuple(result)

    def validate(self, ctx: OpContext) -> None:
        data_shape = ctx.shape(self._shape_data())
        target_shape = self._shape_spec(ctx)
        if self._shape_mode() == "expand":
            self._broadcast_shape(
                data_shape, target_shape, kind=self.kind
            )
        return None

    def infer_types(self, ctx: OpContext) -> None:
        input_dtype = ctx.dtype(self._shape_data())
        try:
            output_dtype = ctx.dtype(self._shape_output())
        except ShapeInferenceError:
            ctx.set_dtype(self._shape_output(), input_dtype)
            output_dtype = input_dtype
        if output_dtype != input_dtype:
            raise UnsupportedOpError(
                f"{self.kind} expects output dtype {input_dtype.onnx_name}, "
                f"got {output_dtype.onnx_name}"
            )

    def infer_shapes(self, ctx: OpContext) -> None:
        data_shape = ctx.shape(self._shape_data())
        target_shape = self._shape_spec(ctx)
        if self._shape_mode() == "expand":
            output_shape = self._broadcast_shape(
                data_shape, target_shape, kind=self.kind
            )
        else:
            output_shape = target_shape
        try:
            expected = ctx.shape(self._shape_output())
        except ShapeInferenceError:
            expected = None
        if expected is not None and expected != output_shape:
            raise ShapeInferenceError(
                f"{self.kind} output shape must be {output_shape}, got {expected}"
            )
        ctx.set_shape(self._shape_output(), output_shape)
        self._shape_derived(
            ctx,
            data_shape=data_shape,
            target_shape=target_shape,
            output_shape=output_shape,
        )


class VariadicLikeOpBase(RenderableOpBase):
    def _variadic_inputs(self) -> tuple[str, ...]:
        raise NotImplementedError

    def _variadic_output(self) -> str:
        raise NotImplementedError

    def _variadic_kind(self) -> str:
        return self.kind

    def _variadic_min_inputs(self) -> int:
        return 2

    def _variadic_max_inputs(self) -> int | None:
        return None

    def _variadic_compare(self) -> bool:
        return False

    def _variadic_supports_dtype(self, dtype: ScalarType) -> bool:
        return True

    def validate(self, ctx: OpContext) -> None:
        inputs = self._variadic_inputs()
        if any(not name for name in inputs):
            raise UnsupportedOpError(
                f"{self._variadic_kind()} input must be provided"
            )
        min_inputs = self._variadic_min_inputs()
        max_inputs = self._variadic_max_inputs()
        if len(inputs) < min_inputs:
            raise UnsupportedOpError(
                f"{self._variadic_kind()} must have at least {min_inputs} inputs"
            )
        if max_inputs is not None and len(inputs) != max_inputs:
            raise UnsupportedOpError(
                f"{self._variadic_kind()} must have exactly {max_inputs} inputs"
            )
        input_dtypes = tuple(ctx.dtype(name) for name in inputs)
        if any(dtype != input_dtypes[0] for dtype in input_dtypes[1:]):
            dtype_names = ", ".join(
                dtype.onnx_name for dtype in input_dtypes
            )
            raise UnsupportedOpError(
                f"{self._variadic_kind()} expects matching input dtypes, "
                f"got {dtype_names}"
            )
        try:
            output_dtype = ctx.dtype(self._variadic_output())
        except ShapeInferenceError:
            output_dtype = None
        if output_dtype is not None:
            if self._variadic_compare():
                if output_dtype != ScalarType.BOOL:
                    raise UnsupportedOpError(
                        f"{self._variadic_kind()} expects bool output, "
                        f"got {output_dtype.onnx_name}"
                    )
            elif output_dtype != input_dtypes[0]:
                raise UnsupportedOpError(
                    f"{self._variadic_kind()} expects output dtype "
                    f"{input_dtypes[0].onnx_name}, got {output_dtype.onnx_name}"
                )
        if not self._variadic_supports_dtype(input_dtypes[0]):
            raise UnsupportedOpError(
                f"{self._variadic_kind()} does not support dtype "
                f"{input_dtypes[0].onnx_name}"
            )
        return None

    def infer_types(self, ctx: OpContext) -> None:
        for name in self._variadic_inputs():
            ctx.dtype(name)
        try:
            ctx.dtype(self._variadic_output())
        except ShapeInferenceError:
            ctx.set_dtype(
                self._variadic_output(),
                ctx.dtype(self._variadic_inputs()[0]),
            )

    def infer_shapes(self, ctx: OpContext) -> None:
        input_shapes = tuple(ctx.shape(name) for name in self._variadic_inputs())
        try:
            output_shape = BroadcastingOpBase.broadcast_shapes(*input_shapes)
        except ShapeInferenceError as exc:
            raise UnsupportedOpError(
                f"{self._variadic_kind()} expects broadcastable input shapes"
            ) from exc
        try:
            expected = ctx.shape(self._variadic_output())
        except ShapeInferenceError:
            expected = None
        if expected is not None and expected != output_shape:
            raise UnsupportedOpError(
                f"{self._variadic_kind()} output shape must be {output_shape}, "
                f"got {expected}"
            )
        ctx.set_shape(self._variadic_output(), output_shape)


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

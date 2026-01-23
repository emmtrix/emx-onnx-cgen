from __future__ import annotations

from dataclasses import dataclass

from shared.scalar_functions import ScalarFunction
from shared.scalar_types import ScalarType

from ...errors import UnsupportedOpError
from ...ops import COMPARE_FUNCTIONS, OperatorKind
from ..op_base import ElementwiseOpBase
from ..op_context import OpContext


@dataclass(frozen=True)
class BinaryOp(ElementwiseOpBase):
    input0: str
    input1: str
    output: str
    function: ScalarFunction
    operator_kind: OperatorKind
    input0_shape: tuple[int, ...]
    input1_shape: tuple[int, ...]
    shape: tuple[int, ...]
    dtype: ScalarType
    input_dtype: ScalarType

    def _elementwise_inputs(self) -> tuple[str, ...]:
        return (self.input0, self.input1)

    def _elementwise_output(self) -> str:
        return self.output

    def _store_elementwise_dtypes(
        self,
        input_dtypes: tuple[ScalarType, ...],
        output_dtype: ScalarType,
    ) -> None:
        input_dtype = input_dtypes[0]
        if self.function in COMPARE_FUNCTIONS and output_dtype != ScalarType.BOOL:
            raise UnsupportedOpError(
                f"{self.function.value} expects bool output"
            )
        object.__setattr__(self, "input_dtype", input_dtype)
        object.__setattr__(self, "dtype", output_dtype)

    def _store_elementwise_shapes(
        self,
        input_shapes: tuple[tuple[int, ...], ...],
        output_shape: tuple[int, ...],
    ) -> None:
        object.__setattr__(self, "input0_shape", input_shapes[0])
        object.__setattr__(self, "input1_shape", input_shapes[1])
        object.__setattr__(self, "shape", output_shape)


@dataclass(frozen=True)
class MultiInputBinaryOp(ElementwiseOpBase):
    inputs: tuple[str, ...]
    output: str
    function: ScalarFunction
    operator_kind: OperatorKind
    shape: tuple[int, ...]
    dtype: ScalarType
    input_dtype: ScalarType

    def _elementwise_inputs(self) -> tuple[str, ...]:
        return self.inputs

    def _elementwise_output(self) -> str:
        return self.output

    def _store_elementwise_dtypes(
        self,
        input_dtypes: tuple[ScalarType, ...],
        output_dtype: ScalarType,
    ) -> None:
        object.__setattr__(self, "input_dtype", input_dtypes[0])
        object.__setattr__(self, "dtype", output_dtype)

    def _store_elementwise_shapes(
        self,
        input_shapes: tuple[tuple[int, ...], ...],
        output_shape: tuple[int, ...],
    ) -> None:
        object.__setattr__(self, "shape", output_shape)


@dataclass(frozen=True)
class WhereOp(ElementwiseOpBase):
    condition: str
    input_x: str
    input_y: str
    output: str
    condition_shape: tuple[int, ...]
    x_shape: tuple[int, ...]
    y_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    dtype: ScalarType

    def _elementwise_inputs(self) -> tuple[str, ...]:
        return (self.condition, self.input_x, self.input_y)

    def _elementwise_output(self) -> str:
        return self.output

    def validate(self, ctx: OpContext) -> None:
        condition_dtype = ctx.dtype(self.condition)
        if condition_dtype != ScalarType.BOOL:
            raise UnsupportedOpError(
                f"Where expects bool condition, got {condition_dtype.onnx_name}"
            )
        x_dtype = ctx.dtype(self.input_x)
        y_dtype = ctx.dtype(self.input_y)
        output_dtype = ctx.dtype(self.output)
        if x_dtype != y_dtype or output_dtype != x_dtype:
            raise UnsupportedOpError(
                "Where expects matching input/output dtypes, "
                f"got {x_dtype.onnx_name}, {y_dtype.onnx_name}, {output_dtype.onnx_name}"
            )

    def _store_elementwise_dtypes(
        self,
        input_dtypes: tuple[ScalarType, ...],
        output_dtype: ScalarType,
    ) -> None:
        object.__setattr__(self, "dtype", output_dtype)

    def _store_elementwise_shapes(
        self,
        input_shapes: tuple[tuple[int, ...], ...],
        output_shape: tuple[int, ...],
    ) -> None:
        object.__setattr__(self, "condition_shape", input_shapes[0])
        object.__setattr__(self, "x_shape", input_shapes[1])
        object.__setattr__(self, "y_shape", input_shapes[2])
        object.__setattr__(self, "output_shape", output_shape)


@dataclass(frozen=True)
class UnaryOp(ElementwiseOpBase):
    input0: str
    output: str
    function: ScalarFunction
    shape: tuple[int, ...]
    dtype: ScalarType
    input_dtype: ScalarType
    params: tuple[float, ...] = ()

    def _elementwise_inputs(self) -> tuple[str, ...]:
        return (self.input0,)

    def _elementwise_output(self) -> str:
        return self.output

    def _store_elementwise_dtypes(
        self,
        input_dtypes: tuple[ScalarType, ...],
        output_dtype: ScalarType,
    ) -> None:
        object.__setattr__(self, "input_dtype", input_dtypes[0])
        object.__setattr__(self, "dtype", output_dtype)

    def _store_elementwise_shapes(
        self,
        input_shapes: tuple[tuple[int, ...], ...],
        output_shape: tuple[int, ...],
    ) -> None:
        object.__setattr__(self, "shape", output_shape)


@dataclass(frozen=True)
class ClipOp(ElementwiseOpBase):
    input0: str
    input_min: str | None
    input_max: str | None
    output: str
    input_shape: tuple[int, ...]
    min_shape: tuple[int, ...] | None
    max_shape: tuple[int, ...] | None
    output_shape: tuple[int, ...]
    dtype: ScalarType

    def _elementwise_inputs(self) -> tuple[str, ...]:
        inputs = [self.input0]
        if self.input_min is not None:
            inputs.append(self.input_min)
        if self.input_max is not None:
            inputs.append(self.input_max)
        return tuple(inputs)

    def _elementwise_output(self) -> str:
        return self.output

    def validate(self, ctx: OpContext) -> None:
        input_dtype = ctx.dtype(self.input0)
        output_dtype = ctx.dtype(self.output)
        if input_dtype != output_dtype:
            raise UnsupportedOpError(
                "Clip expects matching input/output dtypes, "
                f"got {input_dtype.onnx_name} and {output_dtype.onnx_name}"
            )
        if self.input_min is not None:
            min_dtype = ctx.dtype(self.input_min)
            if min_dtype != input_dtype:
                raise UnsupportedOpError(
                    "Clip min dtype must match input dtype, "
                    f"got {min_dtype.onnx_name}"
                )
        if self.input_max is not None:
            max_dtype = ctx.dtype(self.input_max)
            if max_dtype != input_dtype:
                raise UnsupportedOpError(
                    "Clip max dtype must match input dtype, "
                    f"got {max_dtype.onnx_name}"
                )

    def _store_elementwise_dtypes(
        self,
        input_dtypes: tuple[ScalarType, ...],
        output_dtype: ScalarType,
    ) -> None:
        object.__setattr__(self, "dtype", output_dtype)

    def _store_elementwise_shapes(
        self,
        input_shapes: tuple[tuple[int, ...], ...],
        output_shape: tuple[int, ...],
    ) -> None:
        object.__setattr__(self, "input_shape", input_shapes[0])
        min_shape = input_shapes[1] if self.input_min is not None else None
        max_shape = None
        if self.input_max is not None:
            index = 2 if self.input_min is not None else 1
            max_shape = input_shapes[index]
        object.__setattr__(self, "min_shape", min_shape)
        object.__setattr__(self, "max_shape", max_shape)
        object.__setattr__(self, "output_shape", output_shape)


@dataclass(frozen=True)
class IdentityOp(ElementwiseOpBase):
    input0: str
    output: str
    shape: tuple[int, ...]
    dtype: ScalarType
    input_dtype: ScalarType

    def _elementwise_inputs(self) -> tuple[str, ...]:
        return (self.input0,)

    def _elementwise_output(self) -> str:
        return self.output

    def _store_elementwise_dtypes(
        self,
        input_dtypes: tuple[ScalarType, ...],
        output_dtype: ScalarType,
    ) -> None:
        object.__setattr__(self, "dtype", output_dtype)
        object.__setattr__(self, "input_dtype", input_dtypes[0])

    def _store_elementwise_shapes(
        self,
        input_shapes: tuple[tuple[int, ...], ...],
        output_shape: tuple[int, ...],
    ) -> None:
        object.__setattr__(self, "shape", output_shape)

from __future__ import annotations

from dataclasses import dataclass

from shared.scalar_functions import ScalarFunction
from shared.scalar_types import ScalarType

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

    def _elementwise_compare(self) -> bool:
        return self.function in COMPARE_FUNCTIONS


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

    def _elementwise_compare(self) -> bool:
        return self.function in COMPARE_FUNCTIONS


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

    def _elementwise_condition_inputs(self) -> tuple[str, ...]:
        return (self.condition,)


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

    def validate(self, ctx: OpContext) -> None:
        super().validate(ctx)
        return None

    def _elementwise_compare(self) -> bool:
        return self.function in {ScalarFunction.ISINF, ScalarFunction.ISNAN}


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
        super().validate(ctx)
        return None


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

    def validate(self, ctx: OpContext) -> None:
        super().validate(ctx)
        return None

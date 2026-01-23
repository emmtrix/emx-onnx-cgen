from __future__ import annotations

from dataclasses import dataclass

from shared.scalar_functions import ScalarFunction
from shared.scalar_types import ScalarType

from ...ops import COMPARE_FUNCTIONS, OperatorKind, binary_op_symbol
from ..op_base import ElementwiseOpBase, VariadicLikeOpBase
from ..op_context import OpContext


@dataclass(frozen=True)
class BinaryOp(ElementwiseOpBase):
    input0: str
    input1: str
    output: str
    function: ScalarFunction
    operator_kind: OperatorKind

    def _elementwise_inputs(self) -> tuple[str, ...]:
        return (self.input0, self.input1)

    def _elementwise_output(self) -> str:
        return self.output

    def _elementwise_compare(self) -> bool:
        return self.function in COMPARE_FUNCTIONS


@dataclass(frozen=True)
class VariadicOp(VariadicLikeOpBase):
    op_type: str
    inputs: tuple[str, ...]
    output: str
    function: ScalarFunction
    operator_kind: OperatorKind
    min_inputs: int = 2
    max_inputs: int | None = None

    def _variadic_inputs(self) -> tuple[str, ...]:
        return self.inputs

    def _variadic_output(self) -> str:
        return self.output

    def _variadic_kind(self) -> str:
        return self.op_type

    def _variadic_compare(self) -> bool:
        return self.function in COMPARE_FUNCTIONS

    def _variadic_min_inputs(self) -> int:
        return self.min_inputs

    def _variadic_max_inputs(self) -> int | None:
        return self.max_inputs

    def _variadic_supports_dtype(self, dtype: ScalarType) -> bool:
        return (
            binary_op_symbol(
                self.function, dtype=dtype, validate_attrs=False
            )
            is not None
        )


class MultiInputBinaryOp(VariadicOp):
    pass


@dataclass(frozen=True)
class WhereOp(ElementwiseOpBase):
    condition: str
    input_x: str
    input_y: str
    output: str

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

    def _elementwise_inputs(self) -> tuple[str, ...]:
        return (self.input0,)

    def _elementwise_output(self) -> str:
        return self.output

    def validate(self, ctx: OpContext) -> None:
        super().validate(ctx)
        return None

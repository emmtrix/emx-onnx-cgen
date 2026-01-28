from __future__ import annotations

from dataclasses import dataclass

from shared.scalar_functions import ScalarFunction
from shared.scalar_types import ScalarType

from ...ops import COMPARE_FUNCTIONS, OperatorKind, binary_op_symbol
from ...errors import ShapeInferenceError, UnsupportedOpError
from ..op_base import ElementwiseOpBase, RenderableOpBase, VariadicLikeOpBase
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


_POW_BASE_DTYPES = {
    ScalarType.F16,
    ScalarType.F32,
    ScalarType.F64,
    ScalarType.I32,
    ScalarType.I64,
}
_POW_EXPONENT_DTYPES = {
    ScalarType.F16,
    ScalarType.F32,
    ScalarType.F64,
    ScalarType.I8,
    ScalarType.I16,
    ScalarType.I32,
    ScalarType.I64,
    ScalarType.U8,
    ScalarType.U16,
    ScalarType.U32,
    ScalarType.U64,
}


@dataclass(frozen=True)
class PowOp(BinaryOp):
    def validate(self, ctx: OpContext) -> None:
        base_dtype = ctx.dtype(self.input0)
        exponent_dtype = ctx.dtype(self.input1)
        if base_dtype not in _POW_BASE_DTYPES:
            raise UnsupportedOpError(
                "Pow base dtype must be one of "
                f"{', '.join(dtype.onnx_name for dtype in sorted(_POW_BASE_DTYPES, key=str))}, "
                f"got {base_dtype.onnx_name}"
            )
        if exponent_dtype not in _POW_EXPONENT_DTYPES:
            raise UnsupportedOpError(
                "Pow exponent dtype must be one of "
                f"{', '.join(dtype.onnx_name for dtype in sorted(_POW_EXPONENT_DTYPES, key=str))}, "
                f"got {exponent_dtype.onnx_name}"
            )
        try:
            output_dtype = ctx.dtype(self.output)
        except ShapeInferenceError:
            return None
        if output_dtype != base_dtype:
            raise UnsupportedOpError(
                "Pow expects output dtype "
                f"{base_dtype.onnx_name}, got {output_dtype.onnx_name}"
            )
        return None

    def infer_types(self, ctx: OpContext) -> None:
        base_dtype = ctx.dtype(self.input0)
        exponent_dtype = ctx.dtype(self.input1)
        if base_dtype not in _POW_BASE_DTYPES:
            raise UnsupportedOpError(
                "Pow base dtype must be one of "
                f"{', '.join(dtype.onnx_name for dtype in sorted(_POW_BASE_DTYPES, key=str))}, "
                f"got {base_dtype.onnx_name}"
            )
        if exponent_dtype not in _POW_EXPONENT_DTYPES:
            raise UnsupportedOpError(
                "Pow exponent dtype must be one of "
                f"{', '.join(dtype.onnx_name for dtype in sorted(_POW_EXPONENT_DTYPES, key=str))}, "
                f"got {exponent_dtype.onnx_name}"
            )
        try:
            output_dtype = ctx.dtype(self.output)
        except ShapeInferenceError:
            ctx.set_dtype(self.output, base_dtype)
            return None
        if output_dtype != base_dtype:
            raise UnsupportedOpError(
                "Pow expects output dtype "
                f"{base_dtype.onnx_name}, got {output_dtype.onnx_name}"
            )
        return None


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


@dataclass(frozen=True)
class QLinearMulOp(RenderableOpBase):
    input0: str
    input0_scale: str
    input0_zero_point: str
    input1: str
    input1_scale: str
    input1_zero_point: str
    output_scale: str
    output_zero_point: str
    output: str
    input0_shape: tuple[int, ...]
    input1_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    input0_dtype: ScalarType
    input1_dtype: ScalarType
    dtype: ScalarType
    input0_scale_dtype: ScalarType
    input1_scale_dtype: ScalarType
    output_scale_dtype: ScalarType
    input0_scale_shape: tuple[int, ...]
    input1_scale_shape: tuple[int, ...]
    output_scale_shape: tuple[int, ...]
    input0_zero_shape: tuple[int, ...]
    input1_zero_shape: tuple[int, ...]
    output_zero_shape: tuple[int, ...]

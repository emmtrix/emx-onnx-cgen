from __future__ import annotations

from dataclasses import dataclass

from shared.scalar_functions import ScalarFunction
from shared.scalar_types import ScalarType

from ...ops import COMPARE_FUNCTIONS, OperatorKind, binary_op_symbol
from ...errors import ShapeInferenceError, UnsupportedOpError
from ..op_base import (
    BroadcastingOpBase,
    CEmitterCompat,
    ElementwiseOpBase,
    EmitContext,
    Emitter,
    RenderableOpBase,
    VariadicLikeOpBase,
)
from ..op_context import OpContext


@dataclass(frozen=True)
class BinaryOp(ElementwiseOpBase):
    __io_inputs__ = ("input0", "input1")
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

    def infer_shapes(self, ctx: OpContext) -> None:
        if self.function != ScalarFunction.PRELU:
            return super().infer_shapes(ctx)
        input_shape = ctx.shape(self.input0)
        slope_shape = ctx.shape(self.input1)
        output_name = self.output
        if BroadcastingOpBase.unidirectional_broadcastable(slope_shape, input_shape):
            ctx.set_shape(output_name, input_shape)
            return None
        channel_axis = BroadcastingOpBase.prelu_channel_axis(input_shape, slope_shape)
        if channel_axis is not None:
            ctx.set_shape(output_name, input_shape)
            ctx.set_derived(self, "prelu_slope_axis", channel_axis)
            return None
        raise ShapeInferenceError(
            "Broadcasting mismatch for shapes: "
            + ", ".join(str(shape) for shape in (input_shape, slope_shape))
        )


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
            binary_op_symbol(self.function, dtype=dtype, validate_attrs=False)
            is not None
        )


class MultiInputBinaryOp(VariadicOp):
    pass


@dataclass(frozen=True)
class WhereOp(ElementwiseOpBase):
    __io_inputs__ = ("condition", "input_x", "input_y")
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
    __io_inputs__ = ("input0",)
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
    __io_inputs__ = ("input0", "input_min", "input_max")
    input0: str
    input_min: str | None
    input_max: str | None
    output: str
    min_value: float | None = None
    max_value: float | None = None

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
    __io_inputs__ = ("input0",)
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
    __io_inputs__ = (
        "input0",
        "input0_scale",
        "input0_zero_point",
        "input1",
        "input1_scale",
        "input1_zero_point",
        "output_scale",
        "output_zero_point",
    )
    __io_outputs__ = ("output",)
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

    def emit(self, emitter: Emitter, ctx: EmitContext) -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        params = emitter.shared_param_map(
            [
                ("input0", self.input0),
                ("input0_scale", self.input0_scale),
                ("input0_zero_point", self.input0_zero_point),
                ("input1", self.input1),
                ("input1_scale", self.input1_scale),
                ("input1_zero_point", self.input1_zero_point),
                ("output_scale", self.output_scale),
                ("output_zero_point", self.output_zero_point),
                ("output", self.output),
            ]
        )
        output_shape = CEmitterCompat.codegen_shape(self.output_shape)
        output_loop_vars = CEmitterCompat.loop_vars(self.output_shape)
        output_index_expr = f"{params['output']}" + "".join(
            f"[{var}]" for var in output_loop_vars
        )
        input0_index_expr = CEmitterCompat.broadcast_index_expr(
            params["input0"],
            self.input0_shape,
            self.output_shape,
            output_loop_vars,
        )
        input1_index_expr = CEmitterCompat.broadcast_index_expr(
            params["input1"],
            self.input1_shape,
            self.output_shape,
            output_loop_vars,
        )
        input0_suffix = emitter.param_array_suffix(self.input0_shape)
        input1_suffix = emitter.param_array_suffix(self.input1_shape)
        input0_scale_suffix = emitter.param_array_suffix(self.input0_scale_shape)
        input1_scale_suffix = emitter.param_array_suffix(self.input1_scale_shape)
        output_scale_suffix = emitter.param_array_suffix(self.output_scale_shape)
        input0_zero_suffix = emitter.param_array_suffix(self.input0_zero_shape)
        input1_zero_suffix = emitter.param_array_suffix(self.input1_zero_shape)
        output_zero_suffix = emitter.param_array_suffix(self.output_zero_shape)
        output_suffix = emitter.param_array_suffix(self.output_shape)
        param_decls = emitter.build_param_decls(
            [
                (
                    params["input0"],
                    self.input0_dtype.c_type,
                    input0_suffix,
                    True,
                ),
                (
                    params["input0_scale"],
                    self.input0_scale_dtype.c_type,
                    input0_scale_suffix,
                    True,
                ),
                (
                    params["input0_zero_point"],
                    self.input0_dtype.c_type,
                    input0_zero_suffix,
                    True,
                ),
                (
                    params["input1"],
                    self.input1_dtype.c_type,
                    input1_suffix,
                    True,
                ),
                (
                    params["input1_scale"],
                    self.input1_scale_dtype.c_type,
                    input1_scale_suffix,
                    True,
                ),
                (
                    params["input1_zero_point"],
                    self.input1_dtype.c_type,
                    input1_zero_suffix,
                    True,
                ),
                (
                    params["output_scale"],
                    self.output_scale_dtype.c_type,
                    output_scale_suffix,
                    True,
                ),
                (
                    params["output_zero_point"],
                    self.dtype.c_type,
                    output_zero_suffix,
                    True,
                ),
                (
                    params["output"],
                    self.dtype.c_type,
                    output_suffix,
                    False,
                ),
            ]
        )
        compute_dtype = (
            ScalarType.F64
            if ScalarType.F64
            in {
                self.input0_scale_dtype,
                self.input1_scale_dtype,
                self.output_scale_dtype,
            }
            else ScalarType.F32
        )
        compute_type = "double" if compute_dtype == ScalarType.F64 else "float"

        scale_index = "0"
        rendered = state.templates["qlinear_mul"].render(
            model_name=model.name,
            op_name=op_name,
            input0=params["input0"],
            input1=params["input1"],
            input0_scale=params["input0_scale"],
            input0_zero_point=params["input0_zero_point"],
            input1_scale=params["input1_scale"],
            input1_zero_point=params["input1_zero_point"],
            output_scale=params["output_scale"],
            output_zero_point=params["output_zero_point"],
            output=params["output"],
            params=param_decls,
            compute_type=compute_type,
            output_c_type=self.dtype.c_type,
            input0_index_expr=input0_index_expr,
            input1_index_expr=input1_index_expr,
            input0_scale_expr=f"{params['input0_scale']}[{scale_index}]",
            input1_scale_expr=f"{params['input1_scale']}[{scale_index}]",
            output_scale_expr=f"{params['output_scale']}[{scale_index}]",
            input0_zero_expr=f"{params['input0_zero_point']}[{scale_index}]",
            input1_zero_expr=f"{params['input1_zero_point']}[{scale_index}]",
            output_zero_expr=f"{params['output_zero_point']}[{scale_index}]",
            output_loop_vars=output_loop_vars,
            output_loop_bounds=output_shape,
            output_index_expr=output_index_expr,
            compute_dtype=compute_dtype,
            min_literal=self.dtype.min_literal,
            max_literal=self.dtype.max_literal,
            dim_args=emitter.dim_args_str(),
        ).rstrip()
        return emitter.with_node_comment(model, ctx.op_index, rendered)


@dataclass(frozen=True)
class QLinearAddOp(RenderableOpBase):
    __io_inputs__ = (
        "input0",
        "input0_scale",
        "input0_zero_point",
        "input1",
        "input1_scale",
        "input1_zero_point",
        "output_scale",
        "output_zero_point",
    )
    __io_outputs__ = ("output",)
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

    def emit(self, emitter: Emitter, ctx: EmitContext) -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        params = emitter.shared_param_map(
            [
                ("input0", self.input0),
                ("input0_scale", self.input0_scale),
                ("input0_zero_point", self.input0_zero_point),
                ("input1", self.input1),
                ("input1_scale", self.input1_scale),
                ("input1_zero_point", self.input1_zero_point),
                ("output_scale", self.output_scale),
                ("output_zero_point", self.output_zero_point),
                ("output", self.output),
            ]
        )
        output_shape = CEmitterCompat.codegen_shape(self.output_shape)
        output_loop_vars = CEmitterCompat.loop_vars(output_shape)
        output_index_expr = f"{params['output']}" + "".join(
            f"[{var}]" for var in output_loop_vars
        )
        input0_index_expr = CEmitterCompat.broadcast_index_expr(
            params["input0"], self.input0_shape, self.output_shape, output_loop_vars
        )
        input1_index_expr = CEmitterCompat.broadcast_index_expr(
            params["input1"], self.input1_shape, self.output_shape, output_loop_vars
        )
        input0_suffix = emitter.param_array_suffix(self.input0_shape)
        input1_suffix = emitter.param_array_suffix(self.input1_shape)
        input0_scale_suffix = emitter.param_array_suffix(self.input0_scale_shape)
        input1_scale_suffix = emitter.param_array_suffix(self.input1_scale_shape)
        output_scale_suffix = emitter.param_array_suffix(self.output_scale_shape)
        input0_zero_suffix = emitter.param_array_suffix(self.input0_zero_shape)
        input1_zero_suffix = emitter.param_array_suffix(self.input1_zero_shape)
        output_zero_suffix = emitter.param_array_suffix(self.output_zero_shape)
        output_suffix = emitter.param_array_suffix(self.output_shape)
        param_decls = emitter.build_param_decls(
            [
                (params["input0"], self.input0_dtype.c_type, input0_suffix, True),
                (
                    params["input0_scale"],
                    self.input0_scale_dtype.c_type,
                    input0_scale_suffix,
                    True,
                ),
                (
                    params["input0_zero_point"],
                    self.input0_dtype.c_type,
                    input0_zero_suffix,
                    True,
                ),
                (params["input1"], self.input1_dtype.c_type, input1_suffix, True),
                (
                    params["input1_scale"],
                    self.input1_scale_dtype.c_type,
                    input1_scale_suffix,
                    True,
                ),
                (
                    params["input1_zero_point"],
                    self.input1_dtype.c_type,
                    input1_zero_suffix,
                    True,
                ),
                (
                    params["output_scale"],
                    self.output_scale_dtype.c_type,
                    output_scale_suffix,
                    True,
                ),
                (
                    params["output_zero_point"],
                    self.dtype.c_type,
                    output_zero_suffix,
                    True,
                ),
                (params["output"], self.dtype.c_type, output_suffix, False),
            ]
        )
        compute_dtype = (
            ScalarType.F64
            if ScalarType.F64
            in {
                self.input0_scale_dtype,
                self.input1_scale_dtype,
                self.output_scale_dtype,
            }
            else ScalarType.F32
        )
        compute_type = "double" if compute_dtype == ScalarType.F64 else "float"

        scale_index = "0"
        rendered = state.templates["qlinear_add"].render(
            model_name=model.name,
            op_name=op_name,
            input0=params["input0"],
            input1=params["input1"],
            input0_scale=params["input0_scale"],
            input0_zero_point=params["input0_zero_point"],
            input1_scale=params["input1_scale"],
            input1_zero_point=params["input1_zero_point"],
            output_scale=params["output_scale"],
            output_zero_point=params["output_zero_point"],
            output=params["output"],
            params=param_decls,
            compute_type=compute_type,
            output_c_type=self.dtype.c_type,
            input0_index_expr=input0_index_expr,
            input1_index_expr=input1_index_expr,
            input0_scale_expr=f"{params['input0_scale']}[{scale_index}]",
            input1_scale_expr=f"{params['input1_scale']}[{scale_index}]",
            output_scale_expr=f"{params['output_scale']}[{scale_index}]",
            input0_zero_expr=f"{params['input0_zero_point']}[{scale_index}]",
            input1_zero_expr=f"{params['input1_zero_point']}[{scale_index}]",
            output_zero_expr=f"{params['output_zero_point']}[{scale_index}]",
            output_loop_vars=output_loop_vars,
            output_loop_bounds=output_shape,
            output_index_expr=output_index_expr,
            compute_dtype=compute_dtype,
            min_literal=self.dtype.min_literal,
            max_literal=self.dtype.max_literal,
            dim_args=emitter.dim_args_str(),
        ).rstrip()
        return emitter.with_node_comment(model, ctx.op_index, rendered)

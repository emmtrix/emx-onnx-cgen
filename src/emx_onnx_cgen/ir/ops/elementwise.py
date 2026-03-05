from __future__ import annotations

from dataclasses import dataclass

from shared.scalar_functions import ScalarFunction
from shared.scalar_types import ScalarType

from ...ops import COMPARE_FUNCTIONS, OperatorKind, binary_op_symbol, unary_op_symbol
from ...errors import CodegenError, ShapeInferenceError, UnsupportedOpError
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

    def emit(self, emitter: Emitter, ctx: EmitContext) -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        dim_args = emitter.dim_args_str()
        output_dtype_obj = emitter.op_output_dtype(self)
        zero_literal = output_dtype_obj.zero_literal
        input0_shape = emitter.ctx_shape(self.input0)
        input1_shape = emitter.ctx_shape(self.input1)
        output_shape = emitter.ctx_shape(self.output)
        input_dtype = emitter.ctx_dtype(self.input0)
        input1_dtype = (
            emitter.ctx_dtype(self.input1) if isinstance(self, PowOp) else input_dtype
        )
        output_dtype = emitter.ctx_dtype(self.output)
        params = emitter.shared_param_map(
            [
                ("input0", self.input0),
                ("input1", self.input1),
                ("output", self.output),
            ]
        )
        scalar_operator = None
        if self.function not in COMPARE_FUNCTIONS:
            scalar_operator = emitter.scalar_fn(self.function, input_dtype)
        op_spec = binary_op_symbol(
            self.function,
            dtype=input_dtype,
            validate_attrs=False,
        )
        if op_spec is None:
            raise CodegenError(
                f"Unsupported binary operator for rendering: {self.function.value}"
            )
        output_dim_names = emitter.dim_names_for(self.output)
        shape = CEmitterCompat.shape_dim_exprs(output_shape, output_dim_names)
        loop_vars = CEmitterCompat.loop_vars(output_shape)
        output_suffix = emitter.param_array_suffix(output_shape, output_dim_names)
        input0_suffix = emitter.param_array_suffix(
            input0_shape, emitter.dim_names_for(self.input0), dtype=input_dtype
        )
        input1_suffix = emitter.param_array_suffix(
            input1_shape, emitter.dim_names_for(self.input1), dtype=input1_dtype
        )
        input_c_type = input_dtype.c_type
        input1_c_type = input1_dtype.c_type
        output_c_type = output_dtype.c_type
        param_decls = emitter.build_param_decls(
            [
                (params["input0"], input_c_type, input0_suffix, True),
                (params["input1"], input1_c_type, input1_suffix, True),
                (params["output"], output_c_type, output_suffix, False),
            ]
        )
        common = {
            "model_name": model.name,
            "op_name": op_name,
            "element_count": CEmitterCompat.element_count_expr(shape),
            "array_suffix": output_suffix,
            "shape": shape,
            "loop_vars": loop_vars,
            "input_c_type": input_c_type,
            "output_c_type": output_c_type,
            "zero_literal": zero_literal,
            "dim_args": dim_args,
            "params": param_decls,
        }
        left_expr = CEmitterCompat.broadcast_index_expr(
            params["input0"],
            input0_shape,
            output_shape,
            loop_vars,
        )
        prelu_axis = None
        if self.function == ScalarFunction.PRELU:
            derived_axis = emitter.maybe_derived(self, "prelu_slope_axis")
            if isinstance(derived_axis, int):
                prelu_axis = derived_axis
        if prelu_axis is None:
            right_expr = CEmitterCompat.broadcast_index_expr(
                params["input1"],
                input1_shape,
                output_shape,
                loop_vars,
            )
        else:
            right_expr = f"{params['input1']}[{loop_vars[prelu_axis]}]"
        operator_expr = None
        operator = op_spec.operator
        operator_kind = self.operator_kind
        if scalar_operator is not None:
            operator = scalar_operator
            operator_kind = OperatorKind.FUNC
        if input_dtype == ScalarType.STRING:
            if self.function == ScalarFunction.EQ:
                operator_expr = f"(strcmp({left_expr}, {right_expr}) == 0)"
                operator_kind = OperatorKind.EXPR
            else:
                raise CodegenError(
                    "Unsupported string comparison for rendering: "
                    f"{self.function.value}"
                )
        if operator_kind == OperatorKind.EXPR:
            if operator_expr is None:
                operator_expr = op_spec.operator.format(
                    left=left_expr, right=right_expr
                )
        rendered = state.templates["binary"].render(
            **common,
            input0=params["input0"],
            input1=params["input1"],
            output=params["output"],
            operator=operator,
            operator_kind=operator_kind.value,
            left_expr=left_expr,
            right_expr=right_expr,
            operator_expr=operator_expr,
        ).rstrip()
        return emitter.with_node_comment(model, ctx.op_index, rendered)


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
    def emit(self, emitter: Emitter, ctx: EmitContext) -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        dim_args = emitter.dim_args_str()
        output_dtype_obj = emitter.op_output_dtype(self)
        zero_literal = output_dtype_obj.zero_literal
        output_shape_raw = emitter.ctx_shape(self.output)
        input_dtype = emitter.ctx_dtype(self.inputs[0])
        output_dtype = emitter.ctx_dtype(self.output)
        params = emitter.shared_param_map(
            [
                *((f"input{idx}", name) for idx, name in enumerate(self.inputs)),
                ("output", self.output),
            ]
        )
        scalar_operator = None
        if (
            self.function not in COMPARE_FUNCTIONS
            and self.function != ScalarFunction.MEAN
        ):
            scalar_operator = emitter.scalar_fn(self.function, input_dtype)
        op_spec = binary_op_symbol(
            self.function,
            dtype=input_dtype,
            validate_attrs=False,
        )
        if op_spec is None:
            raise CodegenError(
                "Unsupported multi-input operator for rendering: "
                f"{self.function.value}"
            )
        output_dim_names = emitter.dim_names_for(self.output)
        shape = CEmitterCompat.shape_dim_exprs(output_shape_raw, output_dim_names)
        loop_vars = CEmitterCompat.loop_vars(output_shape_raw)
        output_array_suffix = emitter.param_array_suffix(
            output_shape_raw, output_dim_names, dtype=output_dtype
        )
        input_c_type = input_dtype.c_type
        output_c_type = output_dtype.c_type
        input_names = [params[f"input{idx}"] for idx in range(len(self.inputs))]
        input_shapes = [emitter.ctx_shape(name) for name in self.inputs]
        input_dim_names = [emitter.dim_names_for(name) for name in self.inputs]
        input_array_suffixes = [
            emitter.param_array_suffix(shape, dim_names, dtype=input_dtype)
            for shape, dim_names in zip(input_shapes, input_dim_names)
        ]
        param_decls = emitter.build_param_decls(
            [
                *(
                    (name, input_c_type, array_suffix, True)
                    for name, array_suffix in zip(input_names, input_array_suffixes)
                ),
                (
                    params["output"],
                    output_c_type,
                    output_array_suffix,
                    False,
                ),
            ]
        )
        common = {
            "model_name": model.name,
            "op_name": op_name,
            "element_count": CEmitterCompat.element_count_expr(shape),
            "array_suffix": output_array_suffix,
            "shape": shape,
            "loop_vars": loop_vars,
            "input_c_type": input_c_type,
            "output_c_type": output_c_type,
            "zero_literal": zero_literal,
            "dim_args": dim_args,
            "params": param_decls,
        }
        input_exprs = [
            CEmitterCompat.broadcast_index_expr(
                name, shape, output_shape_raw, loop_vars
            )
            for name, shape in zip(input_names, input_shapes)
        ]
        output_expr = f"{params['output']}" + "".join(
            f"[{var}]" for var in loop_vars
        )
        operator = op_spec.operator
        operator_kind = self.operator_kind
        operator_expr = None
        mean_scale = None
        if self.function == ScalarFunction.MEAN:
            operator = "+"
            operator_kind = OperatorKind.INFIX
            mean_scale = len(self.inputs)
        if scalar_operator is not None:
            operator = scalar_operator
            operator_kind = OperatorKind.FUNC
        if operator_kind == OperatorKind.EXPR:
            raise CodegenError(
                "Multi-input operators do not support expression operators."
            )
        rendered = state.templates["multi_input"].render(
            **common,
            inputs=input_names,
            input_exprs=input_exprs,
            output=params["output"],
            output_expr=output_expr,
            operator=operator,
            operator_kind=operator_kind.value,
            operator_expr=operator_expr,
            mean_scale=mean_scale,
            is_mean=self.function == ScalarFunction.MEAN,
        ).rstrip()
        return emitter.with_node_comment(model, ctx.op_index, rendered)


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

    def emit(self, emitter: Emitter, ctx: EmitContext) -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        dim_args = emitter.dim_args_str()
        output_shape_raw = emitter.ctx_shape(self.output)
        condition_shape = emitter.ctx_shape(self.condition)
        x_shape = emitter.ctx_shape(self.input_x)
        y_shape = emitter.ctx_shape(self.input_y)
        output_dtype = emitter.ctx_dtype(self.output)
        params = emitter.shared_param_map(
            [
                ("condition", self.condition),
                ("input_x", self.input_x),
                ("input_y", self.input_y),
                ("output", self.output),
            ]
        )
        output_dim_names = emitter.dim_names_for(self.output)
        output_shape = CEmitterCompat.shape_dim_exprs(output_shape_raw, output_dim_names)
        loop_vars = CEmitterCompat.loop_vars(output_shape_raw)
        output_array_suffix = emitter.param_array_suffix(
            output_shape_raw, output_dim_names
        )
        condition_array_suffix = emitter.param_array_suffix(
            condition_shape, emitter.dim_names_for(self.condition)
        )
        x_array_suffix = emitter.param_array_suffix(
            x_shape, emitter.dim_names_for(self.input_x)
        )
        y_array_suffix = emitter.param_array_suffix(
            y_shape, emitter.dim_names_for(self.input_y)
        )
        condition_expr = CEmitterCompat.broadcast_index_expr(
            params["condition"],
            condition_shape,
            output_shape_raw,
            loop_vars,
        )
        x_expr = CEmitterCompat.broadcast_index_expr(
            params["input_x"], x_shape, output_shape_raw, loop_vars
        )
        y_expr = CEmitterCompat.broadcast_index_expr(
            params["input_y"], y_shape, output_shape_raw, loop_vars
        )
        output_expr = f"{params['output']}" + "".join(
            f"[{var}]" for var in loop_vars
        )
        param_decls = emitter.build_param_decls(
            [
                (
                    params["condition"],
                    ScalarType.BOOL.c_type,
                    condition_array_suffix,
                    True,
                ),
                (params["input_x"], output_dtype.c_type, x_array_suffix, True),
                (params["input_y"], output_dtype.c_type, y_array_suffix, True),
                (
                    params["output"],
                    output_dtype.c_type,
                    output_array_suffix,
                    False,
                ),
            ]
        )
        rendered = state.templates["where"].render(
            model_name=model.name,
            op_name=op_name,
            output_shape=output_shape,
            loop_vars=loop_vars,
            condition=params["condition"],
            input_x=params["input_x"],
            input_y=params["input_y"],
            output=params["output"],
            condition_array_suffix=condition_array_suffix,
            x_array_suffix=x_array_suffix,
            y_array_suffix=y_array_suffix,
            output_array_suffix=output_array_suffix,
            condition_expr=condition_expr,
            x_expr=x_expr,
            y_expr=y_expr,
            output_expr=output_expr,
            input_c_type=output_dtype.c_type,
            output_c_type=output_dtype.c_type,
            condition_c_type=ScalarType.BOOL.c_type,
            dim_args=dim_args,
            params=param_decls,
        ).rstrip()
        return emitter.with_node_comment(model, ctx.op_index, rendered)


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

    def emit(self, emitter: Emitter, ctx: EmitContext) -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        dim_args = emitter.dim_args_str()
        output_dtype_obj = emitter.op_output_dtype(self)
        zero_literal = output_dtype_obj.zero_literal
        input_dtype = emitter.ctx_dtype(self.input0)
        output_dtype = emitter.ctx_dtype(self.output)
        output_shape_raw = emitter.ctx_shape(self.output)
        params = emitter.shared_param_map(
            [("input0", self.input0), ("output", self.output)]
        )
        scalar_operator = None
        scalar_operator = emitter.scalar_fn(
            self.function, input_dtype, params=self.params
        )
        output_dim_names = emitter.dim_names_for(self.output)
        shape = CEmitterCompat.shape_dim_exprs(output_shape_raw, output_dim_names)
        loop_vars = CEmitterCompat.loop_vars(output_shape_raw)
        array_suffix = emitter.param_array_suffix(output_shape_raw, output_dim_names)
        param_decls = emitter.build_param_decls(
            [
                (params["input0"], input_dtype.c_type, array_suffix, True),
                (params["output"], output_dtype.c_type, array_suffix, False),
            ]
        )
        operator_symbol = unary_op_symbol(self.function, dtype=output_dtype)
        if self.function == ScalarFunction.ISINF and len(self.params) == 2:
            detect_negative, detect_positive = self.params
            detect_negative = int(detect_negative)
            detect_positive = int(detect_positive)
            if detect_negative and detect_positive:
                operator_symbol = "isinf"
            elif detect_negative:
                operator_symbol = "isneginf"
            elif detect_positive:
                operator_symbol = "isposinf"
            else:
                operator_symbol = "zero"
        elif self.function in {ScalarFunction.ISINF, ScalarFunction.ISNAN}:
            operator_symbol = (
                "isinf" if self.function == ScalarFunction.ISINF else "isnan"
            )
        if operator_symbol is None and scalar_operator is None:
            raise CodegenError(
                f"Unsupported unary operator for rendering: {self.function.value}"
            )
        common = {
            "model_name": model.name,
            "op_name": op_name,
            "element_count": CEmitterCompat.element_count_expr(shape),
            "array_suffix": array_suffix,
            "shape": shape,
            "loop_vars": loop_vars,
            "input_c_type": input_dtype.c_type,
            "output_c_type": output_dtype.c_type,
            "zero_literal": zero_literal,
            "dim_args": dim_args,
            "params": param_decls,
        }
        rendered = state.templates["unary"].render(
            **common,
            input0=params["input0"],
            output=params["output"],
            operator=scalar_operator or operator_symbol,
        ).rstrip()
        return emitter.with_node_comment(model, ctx.op_index, rendered)


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

    def emit(self, emitter: Emitter, ctx: EmitContext) -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        dim_args = emitter.dim_args_str()
        input_shape = emitter.ctx_shape(self.input0)
        output_shape_raw = emitter.ctx_shape(self.output)
        input_dtype = emitter.ctx_dtype(self.input0)
        output_dtype = emitter.ctx_dtype(self.output)
        min_shape = (
            emitter.ctx_shape(self.input_min) if self.input_min is not None else None
        )
        max_shape = (
            emitter.ctx_shape(self.input_max) if self.input_max is not None else None
        )
        params = emitter.shared_param_map(
            [
                ("input0", self.input0),
                ("input_min", self.input_min),
                ("input_max", self.input_max),
                ("output", self.output),
            ]
        )
        output_dim_names = emitter.dim_names_for(self.output)
        output_shape = CEmitterCompat.shape_dim_exprs(output_shape_raw, output_dim_names)
        loop_vars = CEmitterCompat.loop_vars(output_shape_raw)
        input_expr = CEmitterCompat.broadcast_index_expr(
            params["input0"],
            input_shape,
            output_shape_raw,
            loop_vars,
        )
        min_expr = (
            CEmitterCompat.broadcast_index_expr(
                params["input_min"],
                min_shape,
                output_shape_raw,
                loop_vars,
            )
            if self.input_min is not None
            else (
                emitter.format_literal(output_dtype, self.min_value)
                if self.min_value is not None
                else output_dtype.min_literal
            )
        )
        max_expr = (
            CEmitterCompat.broadcast_index_expr(
                params["input_max"],
                max_shape,
                output_shape_raw,
                loop_vars,
            )
            if self.input_max is not None
            else (
                emitter.format_literal(output_dtype, self.max_value)
                if self.max_value is not None
                else output_dtype.max_literal
            )
        )
        input_suffix = emitter.param_array_suffix(
            input_shape, emitter.dim_names_for(self.input0)
        )
        min_suffix = (
            emitter.param_array_suffix(min_shape, emitter.dim_names_for(self.input_min))
            if min_shape is not None
            else ""
        )
        max_suffix = (
            emitter.param_array_suffix(max_shape, emitter.dim_names_for(self.input_max))
            if max_shape is not None
            else ""
        )
        output_suffix = emitter.param_array_suffix(output_shape_raw, output_dim_names)
        param_decls = emitter.build_param_decls(
            [
                (params["input0"], input_dtype.c_type, input_suffix, True),
                (
                    (
                        params["input_min"],
                        input_dtype.c_type,
                        min_suffix,
                        True,
                    )
                    if params["input_min"]
                    else (None, "", "", True)
                ),
                (
                    (
                        params["input_max"],
                        input_dtype.c_type,
                        max_suffix,
                        True,
                    )
                    if params["input_max"]
                    else (None, "", "", True)
                ),
                (params["output"], output_dtype.c_type, output_suffix, False),
            ]
        )
        rendered = state.templates["clip"].render(
            model_name=model.name,
            op_name=op_name,
            input0=params["input0"],
            input_min=params["input_min"],
            input_max=params["input_max"],
            output=params["output"],
            params=param_decls,
            input_c_type=input_dtype.c_type,
            output_c_type=output_dtype.c_type,
            input_suffix=input_suffix,
            min_suffix=min_suffix,
            max_suffix=max_suffix,
            output_suffix=output_suffix,
            shape=output_shape,
            loop_vars=loop_vars,
            input_expr=input_expr,
            min_expr=min_expr,
            max_expr=max_expr,
            dtype=input_dtype,
            dim_args=dim_args,
        ).rstrip()
        return emitter.with_node_comment(model, ctx.op_index, rendered)


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

    def emit(self, emitter: "Emitter", ctx: "EmitContext") -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        dim_args = emitter.dim_args_str()
        output_dtype = emitter.op_output_dtype(self)
        c_type = output_dtype.c_type
        output_shape_raw = emitter.ctx_shape(self.output)
        params = emitter.shared_param_map(
            [("input0", self.input0), ("output", self.output)]
        )
        output_dim_names = emitter.dim_names_for(self.output)
        shape = CEmitterCompat.shape_dim_exprs(output_shape_raw, output_dim_names)
        loop_vars = CEmitterCompat.loop_vars(output_shape_raw)
        output_dtype = emitter.ctx_dtype(self.output)
        output_suffix = emitter.param_array_suffix(
            output_shape_raw, output_dim_names, dtype=output_dtype
        )
        input_suffix = emitter.param_array_suffix(
            output_shape_raw, emitter.dim_names_for(self.input0), dtype=output_dtype
        )
        param_decls = emitter.build_param_decls(
            [
                (params["input0"], c_type, input_suffix, True),
                (params["output"], c_type, output_suffix, False),
            ]
        )
        if c_type == "char":
            rendered = "\n".join(
                (
                    f"EMX_NODE_FN void {op_name}({dim_args}{', '.join(param_decls)}) {{",
                    f"    memcpy({params['output']}, {params['input0']}, sizeof(char) * {CEmitterCompat.element_count_expr(shape)} * EMX_STRING_MAX_LEN);",
                    "}",
                )
            )
        else:
            rendered = state.templates["identity"].render(
                model_name=model.name,
                op_name=op_name,
                input0=params["input0"],
                output=params["output"],
                params=param_decls,
                c_type=c_type,
                input_suffix=input_suffix,
                output_suffix=output_suffix,
                shape=shape,
                loop_vars=loop_vars,
            ).rstrip()
        return emitter.with_node_comment(model, ctx.op_index, rendered)


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

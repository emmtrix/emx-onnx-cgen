from __future__ import annotations

from dataclasses import dataclass

import itertools
import math
import re

import numpy as np

from shared.scalar_types import ScalarType

from ...errors import CodegenError, ShapeInferenceError, UnsupportedOpError
from ..model import SequenceType
from ..op_base import (
    CEmitterCompat,
    EmitContext,
    Emitter,
    GatherLikeOpBase,
    RenderableOpBase,
    ShapeLikeOpBase,
)
from ..op_context import OpContext


def _compute_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides: list[int] = []
    stride = 1
    for dim in reversed(shape):
        strides.append(stride)
        stride *= dim
    return tuple(reversed(strides))


def _shape_product(shape: tuple[int, ...]) -> int:
    product = 1
    for dim in shape:
        product *= dim
    return product


def _cast_fn_for_float8(
    input_dtype: ScalarType,
    output_dtype: ScalarType,
    emitter: Emitter,
    *,
    saturate: bool = True,
) -> tuple[str, str]:
    """Return ``(cast_prefix, cast_suffix)`` for the cast template.

    When either side is a float8 type a plain C cast ``(target_type)value`` is
    not sufficient because the storage type is ``uint8_t``.  Instead we chain
    the appropriate ``to_f32`` / ``from_f32`` scalar conversion helpers.

    For regular casts the pair ``("(target_type)", "")`` is returned.
    """
    from shared.scalar_functions import ScalarFunction, ScalarFunctionKey

    src_f8 = input_dtype.is_typedef_float
    dst_f8 = output_dtype.is_typedef_float

    if not src_f8 and not dst_f8:
        return f"({output_dtype.c_type})", ""

    registry = emitter.scalar_registry()
    if registry is None:
        return f"({output_dtype.c_type})", ""

    from_f32_fn = (
        ScalarFunction.FROM_F32_NO_SAT
        if not saturate
        else ScalarFunction.CONVERT_FROM_F32
    )

    if src_f8 and dst_f8:
        to_name = registry.request(
            ScalarFunctionKey(
                function=ScalarFunction.CONVERT_FROM_BOOL,
                return_type=input_dtype,
            )
        )
        from_name = registry.request(
            ScalarFunctionKey(
                function=from_f32_fn,
                return_type=output_dtype,
            )
        )
        return f"{from_name}({to_name}(", "))"

    if src_f8:
        to_name = registry.request(
            ScalarFunctionKey(
                function=ScalarFunction.CONVERT_FROM_BOOL,
                return_type=input_dtype,
            )
        )
        return f"({output_dtype.c_type}){to_name}(", ")"

    from_name = registry.request(
        ScalarFunctionKey(
            function=from_f32_fn,
            return_type=output_dtype,
        )
    )
    return f"{from_name}((float)", ")"


@dataclass(frozen=True)
class CastOp(RenderableOpBase):
    __io_inputs__ = ("input0",)
    __io_outputs__ = ("output",)
    input0: str
    output: str
    saturate: bool = True

    def infer_types(self, ctx: OpContext) -> None:
        ctx.dtype(self.input0)
        ctx.dtype(self.output)

    def infer_shapes(self, ctx: OpContext) -> None:
        shape = ctx.shape(self.input0)
        ctx.set_shape(self.output, shape)

    def emit(self, emitter: Emitter, ctx: EmitContext) -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        dim_args = emitter.dim_args_str()
        input_dtype = emitter.ctx_dtype(self.input0)
        output_dtype = emitter.ctx_dtype(self.output)
        output_shape_raw = emitter.ctx_shape(self.output)
        params = emitter.shared_param_map(
            [("input0", self.input0), ("output", self.output)]
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
        cast_prefix, cast_suffix = _cast_fn_for_float8(
            input_dtype, output_dtype, emitter, saturate=self.saturate
        )
        rendered = (
            state.templates["cast"]
            .render(
                model_name=model.name,
                op_name=op_name,
                input0=params["input0"],
                output=params["output"],
                params=param_decls,
                input_c_type=input_dtype.c_type,
                output_c_type=output_dtype.c_type,
                array_suffix=array_suffix,
                shape=shape,
                loop_vars=loop_vars,
                dim_args=dim_args,
                cast_prefix=cast_prefix,
                cast_suffix=cast_suffix,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)


@dataclass(frozen=True)
class QuantizeLinearOp(RenderableOpBase):
    __io_inputs__ = ("input0", "scale", "zero_point")
    __io_outputs__ = ("output",)
    input0: str
    scale: str
    zero_point: str | None
    output: str
    axis: int | None
    block_size: int | None

    def required_includes(self, ctx: OpContext) -> set[str]:
        includes: set[str] = {"#include <math.h>"}
        if ctx.dtype(self.output).is_integer:
            includes.add("#include <limits.h>")
        return includes

    def emit(self, emitter: Emitter, ctx: EmitContext) -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        dim_args = emitter.dim_args_str()
        params = emitter.shared_param_map(
            [
                ("input0", self.input0),
                ("scale", self.scale),
                ("zero_point", self.zero_point),
                ("output", self.output),
            ]
        )
        output_dim_names = emitter.dim_names_for(self.output)
        input_shape = emitter.ctx_shape(self.input0)
        output_dtype = emitter.ctx_dtype(self.output)
        input_dtype = emitter.ctx_dtype(self.input0)
        scale_dtype = emitter.ctx_dtype(self.scale)
        shape = CEmitterCompat.shape_dim_exprs(input_shape, output_dim_names)
        loop_vars = CEmitterCompat.loop_vars(input_shape)
        input_suffix = emitter.param_array_suffix(
            input_shape, emitter.dim_names_for(self.input0)
        )
        if self.axis is None:
            scale_shape = ()
        elif self.block_size:
            scale_shape_list = list(input_shape)
            scale_shape_list[self.axis] = input_shape[self.axis] // self.block_size
            scale_shape = tuple(scale_shape_list)
        else:
            scale_shape = (input_shape[self.axis],)
        scale_suffix = emitter.param_array_suffix(
            scale_shape, emitter.dim_names_for(self.scale)
        )
        zero_point_suffix = emitter.param_array_suffix(
            scale_shape, emitter.dim_names_for(self.zero_point or "")
        )
        param_decls = emitter.build_param_decls(
            [
                (params["input0"], input_dtype.c_type, input_suffix, True),
                (params["scale"], scale_dtype.c_type, scale_suffix, True),
                (
                    (
                        params["zero_point"],
                        output_dtype.c_type,
                        zero_point_suffix,
                        True,
                    )
                    if params["zero_point"]
                    else (None, "", "", True)
                ),
                (params["output"], output_dtype.c_type, input_suffix, False),
            ]
        )
        compute_type = "double" if input_dtype == ScalarType.F64 else "float"
        input_expr = f"{params['input0']}" + "".join(f"[{var}]" for var in loop_vars)
        output_expr = f"{params['output']}" + "".join(f"[{var}]" for var in loop_vars)
        if self.axis is None:
            scale_expr = f"{params['scale']}[0]"
        elif self.block_size:
            scale_indices = list(loop_vars)
            scale_indices[self.axis] = f"({loop_vars[self.axis]}) / {self.block_size}"
            scale_expr = f"{params['scale']}" + "".join(
                f"[{index}]" for index in scale_indices
            )
        else:
            scale_index = loop_vars[self.axis]
            scale_expr = f"{params['scale']}[{scale_index}]"
        if params["zero_point"]:
            if self.axis is None:
                zero_expr = f"{params['zero_point']}[0]"
            elif self.block_size:
                scale_indices = list(loop_vars)
                scale_indices[self.axis] = (
                    f"({loop_vars[self.axis]}) / {self.block_size}"
                )
                zero_expr = f"{params['zero_point']}" + "".join(
                    f"[{index}]" for index in scale_indices
                )
            else:
                zero_expr = f"{params['zero_point']}[{scale_index}]"
        else:
            zero_expr = "0"
        from_f32_fn = ""
        to_f32_fn = ""
        if output_dtype.is_typedef_float:
            from shared.scalar_functions import ScalarFunction, ScalarFunctionKey

            registry = emitter.scalar_registry()
            if registry is not None:
                from_f32_fn = registry.request(
                    ScalarFunctionKey(
                        function=ScalarFunction.CONVERT_FROM_F32,
                        return_type=output_dtype,
                    )
                )
                to_f32_fn = registry.request(
                    ScalarFunctionKey(
                        function=ScalarFunction.CONVERT_FROM_BOOL,
                        return_type=output_dtype,
                    )
                )
        rendered = (
            state.templates["quantize_linear"]
            .render(
                model_name=model.name,
                op_name=op_name,
                input0=params["input0"],
                scale=params["scale"],
                zero_point=params["zero_point"],
                output=params["output"],
                params=param_decls,
                compute_type=compute_type,
                input_c_type=input_dtype.c_type,
                output_c_type=output_dtype.c_type,
                shape=shape,
                loop_vars=loop_vars,
                input_expr=input_expr,
                scale_expr=scale_expr,
                zero_expr=zero_expr,
                output_expr=output_expr,
                compute_dtype=input_dtype,
                min_literal=output_dtype.min_literal,
                max_literal=output_dtype.max_literal,
                dim_args=dim_args,
                output_is_typedef_float=output_dtype.is_typedef_float,
                from_f32_fn=from_f32_fn,
                to_f32_fn=to_f32_fn,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)

    def c_op_inputs(
        self, emitter: "Emitter"
    ) -> tuple[tuple[str, tuple[int, ...]], ...]:
        if self.axis is None:
            scale_shape = ()
        elif self.block_size:
            input_shape = emitter.ctx_shape(self.input0)
            scale_shape_list = list(input_shape)
            scale_shape_list[self.axis] = input_shape[self.axis] // self.block_size
            scale_shape = tuple(scale_shape_list)
        else:
            scale_shape = (emitter.ctx_shape(self.input0)[self.axis],)
        inputs: list[tuple[str, tuple[int, ...]]] = [
            (self.input0, emitter.ctx_shape(self.input0)),
            (self.scale, scale_shape),
        ]
        if self.zero_point is not None:
            inputs.append((self.zero_point, scale_shape))
        return tuple(inputs)


@dataclass(frozen=True)
class DynamicQuantizeLinearOp(RenderableOpBase):
    __io_inputs__ = ("input0",)
    __io_outputs__ = ("output", "scale", "zero_point")
    input0: str
    output: str
    scale: str
    zero_point: str

    def required_includes(self, ctx: OpContext) -> set[str]:
        return {"#include <math.h>"}

    def emit(self, emitter: Emitter, ctx: EmitContext) -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        dim_args = emitter.dim_args_str()
        params = emitter.shared_param_map(
            [
                ("input0", self.input0),
                ("output", self.output),
                ("scale", self.scale),
                ("zero_point", self.zero_point),
            ]
        )
        input_shape = emitter.ctx_shape(self.input0)
        input_dtype = emitter.ctx_dtype(self.input0)
        output_dtype = emitter.ctx_dtype(self.output)
        scale_dtype = emitter.ctx_dtype(self.scale)
        zero_point_dtype = emitter.ctx_dtype(self.zero_point)
        output_dim_names = emitter.dim_names_for(self.output)
        shape = CEmitterCompat.shape_dim_exprs(input_shape, output_dim_names)
        loop_vars = CEmitterCompat.loop_vars(input_shape)
        input_suffix = emitter.param_array_suffix(
            input_shape, emitter.dim_names_for(self.input0)
        )
        param_decls = emitter.build_param_decls(
            [
                (params["input0"], input_dtype.c_type, input_suffix, True),
                (params["output"], output_dtype.c_type, input_suffix, False),
                (params["scale"], scale_dtype.c_type, "[1]", False),
                (params["zero_point"], zero_point_dtype.c_type, "[1]", False),
            ]
        )
        compute_type = "double" if input_dtype == ScalarType.F64 else "float"
        input_expr = f"{params['input0']}" + "".join(f"[{var}]" for var in loop_vars)
        output_expr = f"{params['output']}" + "".join(f"[{var}]" for var in loop_vars)
        rendered = (
            state.templates["dynamic_quantize_linear"]
            .render(
                model_name=model.name,
                op_name=op_name,
                params=param_decls,
                compute_type=compute_type,
                output_c_type=output_dtype.c_type,
                scale_c_type=scale_dtype.c_type,
                zero_point_c_type=zero_point_dtype.c_type,
                shape=shape,
                loop_vars=loop_vars,
                input_expr=input_expr,
                output_expr=output_expr,
                scale_expr=f"{params['scale']}[0]",
                zero_point_expr=f"{params['zero_point']}[0]",
                compute_dtype=input_dtype,
                dim_args=dim_args,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)

    def c_op_outputs(
        self, emitter: "Emitter"
    ) -> tuple[tuple[str, tuple[int, ...], "ScalarType"], ...]:
        return (
            (
                self.output,
                emitter.ctx_shape(self.output),
                emitter.ctx_dtype(self.output),
            ),
            (self.scale, emitter.ctx_shape(self.scale), emitter.ctx_dtype(self.scale)),
            (
                self.zero_point,
                emitter.ctx_shape(self.zero_point),
                emitter.ctx_dtype(self.zero_point),
            ),
        )


@dataclass(frozen=True)
class DequantizeLinearOp(RenderableOpBase):
    __io_inputs__ = ("input0", "scale", "zero_point")
    __io_outputs__ = ("output",)
    input0: str
    scale: str
    zero_point: str | None
    output: str
    axis: int | None
    block_size: int | None

    def emit(self, emitter: Emitter, ctx: EmitContext) -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        dim_args = emitter.dim_args_str()
        params = emitter.shared_param_map(
            [
                ("input0", self.input0),
                ("scale", self.scale),
                ("zero_point", self.zero_point),
                ("output", self.output),
            ]
        )
        output_dim_names = emitter.dim_names_for(self.output)
        input_shape = emitter.ctx_shape(self.input0)
        output_dtype = emitter.ctx_dtype(self.output)
        input_dtype = emitter.ctx_dtype(self.input0)
        scale_dtype = emitter.ctx_dtype(self.scale)
        shape = CEmitterCompat.shape_dim_exprs(input_shape, output_dim_names)
        loop_vars = CEmitterCompat.loop_vars(input_shape)
        input_suffix = emitter.param_array_suffix(
            input_shape, emitter.dim_names_for(self.input0)
        )
        if self.axis is None:
            scale_shape = ()
        elif self.block_size:
            scale_shape_list = list(input_shape)
            scale_shape_list[self.axis] = input_shape[self.axis] // self.block_size
            scale_shape = tuple(scale_shape_list)
        else:
            scale_shape = (input_shape[self.axis],)
        scale_suffix = emitter.param_array_suffix(
            scale_shape, emitter.dim_names_for(self.scale)
        )
        zero_point_suffix = emitter.param_array_suffix(
            scale_shape, emitter.dim_names_for(self.zero_point or "")
        )
        param_decls = emitter.build_param_decls(
            [
                (params["input0"], input_dtype.c_type, input_suffix, True),
                (params["scale"], scale_dtype.c_type, scale_suffix, True),
                (
                    (
                        params["zero_point"],
                        input_dtype.c_type,
                        zero_point_suffix,
                        True,
                    )
                    if params["zero_point"]
                    else (None, "", "", True)
                ),
                (params["output"], output_dtype.c_type, input_suffix, False),
            ]
        )
        compute_type = "double" if output_dtype == ScalarType.F64 else "float"
        input_expr = f"{params['input0']}" + "".join(f"[{var}]" for var in loop_vars)
        output_expr = f"{params['output']}" + "".join(f"[{var}]" for var in loop_vars)
        if self.axis is None:
            scale_expr = f"{params['scale']}[0]"
        elif self.block_size:
            scale_indices = list(loop_vars)
            scale_indices[self.axis] = f"({loop_vars[self.axis]}) / {self.block_size}"
            scale_expr = f"{params['scale']}" + "".join(
                f"[{index}]" for index in scale_indices
            )
        else:
            scale_index = loop_vars[self.axis]
            scale_expr = f"{params['scale']}[{scale_index}]"
        if params["zero_point"]:
            if self.axis is None:
                zero_expr = f"{params['zero_point']}[0]"
            elif self.block_size:
                scale_indices = list(loop_vars)
                scale_indices[self.axis] = (
                    f"({loop_vars[self.axis]}) / {self.block_size}"
                )
                zero_expr = f"{params['zero_point']}" + "".join(
                    f"[{index}]" for index in scale_indices
                )
            else:
                zero_expr = f"{params['zero_point']}[{scale_index}]"
        else:
            zero_expr = "0"
        to_f32_fn = ""
        if input_dtype.is_typedef_float:
            from shared.scalar_functions import ScalarFunction, ScalarFunctionKey

            registry = emitter.scalar_registry()
            if registry is not None:
                to_f32_fn = registry.request(
                    ScalarFunctionKey(
                        function=ScalarFunction.CONVERT_FROM_BOOL,
                        return_type=input_dtype,
                    )
                )
        rendered = (
            state.templates["dequantize_linear"]
            .render(
                model_name=model.name,
                op_name=op_name,
                input0=params["input0"],
                scale=params["scale"],
                zero_point=params["zero_point"],
                output=params["output"],
                params=param_decls,
                compute_type=compute_type,
                input_c_type=input_dtype.c_type,
                output_c_type=output_dtype.c_type,
                shape=shape,
                loop_vars=loop_vars,
                input_expr=input_expr,
                scale_expr=scale_expr,
                zero_expr=zero_expr,
                output_expr=output_expr,
                dim_args=dim_args,
                input_is_typedef_float=input_dtype.is_typedef_float,
                to_f32_fn=to_f32_fn,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)

    def c_op_inputs(
        self, emitter: "Emitter"
    ) -> tuple[tuple[str, tuple[int, ...]], ...]:
        if self.axis is None:
            scale_shape: tuple[int, ...] = ()
        elif self.block_size:
            input_shape = emitter.ctx_shape(self.input0)
            scale_shape_list = list(input_shape)
            scale_shape_list[self.axis] = input_shape[self.axis] // self.block_size
            scale_shape = tuple(scale_shape_list)
        else:
            scale_shape = (emitter.ctx_shape(self.input0)[self.axis],)
        inputs: list[tuple[str, tuple[int, ...]]] = [
            (self.input0, emitter.ctx_shape(self.input0)),
            (self.scale, scale_shape),
        ]
        if self.zero_point is not None:
            inputs.append((self.zero_point, scale_shape))
        return tuple(inputs)


@dataclass(frozen=True)
class ConcatOp(RenderableOpBase):
    __io_inputs__ = ("inputs",)
    __io_outputs__ = ("output",)
    inputs: tuple[str, ...]
    output: str
    axis: int

    def required_includes(self, ctx: OpContext) -> set[str]:
        return {"#include <string.h>"}

    def emit(self, emitter: Emitter, ctx: EmitContext) -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        c_type = emitter.ctx_dtype(self.output).c_type
        input_params = [
            (f"input_{index}", name) for index, name in enumerate(self.inputs)
        ]
        params = emitter.shared_param_map([*input_params, ("output", self.output)])
        input_names = tuple(
            params[f"input_{index}"] for index in range(len(self.inputs))
        )
        output_shape = emitter.ctx_shape(self.output)
        input_shapes = tuple(emitter.ctx_shape(name) for name in self.inputs)
        axis = self.axis
        if axis < 0:
            axis += len(output_shape)
        outer = CEmitterCompat.element_count(output_shape[:axis] or (1,))
        inner = CEmitterCompat.element_count(output_shape[axis + 1 :] or (1,))
        axis_sizes = tuple(shape[axis] for shape in input_shapes)
        input_suffixes = tuple(
            emitter.param_array_suffix(shape) for shape in input_shapes
        )
        output_suffix = emitter.param_array_suffix(output_shape)
        param_decls = emitter.build_param_decls(
            [
                *(
                    (name, c_type, suffix, True)
                    for name, suffix in zip(input_names, input_suffixes)
                ),
                (params["output"], c_type, output_suffix, False),
            ]
        )
        rendered = (
            state.templates["concat"]
            .render(
                model_name=model.name,
                op_name=op_name,
                inputs=input_names,
                output=params["output"],
                params=param_decls,
                c_type=c_type,
                input_suffixes=input_suffixes,
                output_suffix=output_suffix,
                axis_sizes=axis_sizes,
                input_count=len(self.inputs),
                outer=outer,
                inner=inner,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)


@dataclass(frozen=True)
class CompressOp(RenderableOpBase):
    __io_inputs__ = ("data", "condition")
    __io_outputs__ = ("output",)
    data: str
    condition: str
    output: str
    axis: int | None

    def emit(self, emitter: "Emitter", ctx: "EmitContext") -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        dim_args = emitter.dim_args_str()
        output_dtype = emitter.op_output_dtype(self)
        c_type = output_dtype.c_type
        params = emitter.shared_param_map(
            [
                ("data", self.data),
                ("condition", self.condition),
                ("output", self.output),
            ]
        )
        data_shape = emitter.ctx_shape(self.data)
        condition_shape = emitter.ctx_shape(self.condition)
        output_shape_raw = emitter.ctx_shape(self.output)
        output_shape = CEmitterCompat.codegen_shape(output_shape_raw)
        output_size = CEmitterCompat.element_count(output_shape_raw)
        condition_length = condition_shape[0]
        data_suffix = emitter.param_array_suffix(data_shape)
        condition_suffix = emitter.param_array_suffix(condition_shape)
        output_suffix = emitter.param_array_suffix(output_shape_raw)
        loop_vars = CEmitterCompat.loop_vars(output_shape_raw)
        output_strides: list[int] = []
        stride = 1
        for dim in reversed(output_shape_raw):
            output_strides.append(stride)
            stride *= dim
        output_strides = tuple(reversed(output_strides))
        param_decls = emitter.build_param_decls(
            [
                (params["data"], c_type, data_suffix, True),
                (
                    params["condition"],
                    emitter.ctx_dtype(self.condition).c_type,
                    condition_suffix,
                    True,
                ),
                (params["output"], c_type, output_suffix, False),
            ]
        )
        rendered = (
            state.templates["compress"]
            .render(
                model_name=model.name,
                op_name=op_name,
                data=params["data"],
                condition=params["condition"],
                output=params["output"],
                params=param_decls,
                c_type=c_type,
                data_suffix=data_suffix,
                condition_suffix=condition_suffix,
                output_suffix=output_suffix,
                output_shape=output_shape,
                output_size=output_size,
                output_strides=output_strides,
                condition_length=condition_length,
                axis=self.axis,
                axis_is_none=self.axis is None,
                loop_vars=loop_vars,
                dim_args=dim_args,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)


@dataclass(frozen=True)
class GatherElementsOp(RenderableOpBase):
    __io_inputs__ = ("data", "indices")
    __io_outputs__ = ("output",)
    data: str
    indices: str
    output: str
    axis: int

    def emit(self, emitter: "Emitter", ctx: "EmitContext") -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        emitter.dim_args_str()
        output_dtype = emitter.op_output_dtype(self)
        c_type = output_dtype.c_type
        params = emitter.shared_param_map(
            [
                ("data", self.data),
                ("indices", self.indices),
                ("output", self.output),
            ]
        )
        output_shape_raw = emitter.ctx_shape(self.output)
        output_shape = CEmitterCompat.codegen_shape(output_shape_raw)
        loop_vars = CEmitterCompat.loop_vars(output_shape_raw)
        data_indices = list(loop_vars)
        data_indices[self.axis] = "gather_index"
        data_shape = emitter.ctx_shape(self.data)
        indices_shape = emitter.ctx_shape(self.indices)
        data_suffix = emitter.param_array_suffix(data_shape)
        indices_suffix = emitter.param_array_suffix(indices_shape)
        output_suffix = emitter.param_array_suffix(output_shape_raw)
        param_decls = emitter.build_param_decls(
            [
                (params["data"], c_type, data_suffix, True),
                (
                    params["indices"],
                    emitter.ctx_dtype(self.indices).c_type,
                    indices_suffix,
                    True,
                ),
                (params["output"], c_type, output_suffix, False),
            ]
        )
        rendered = (
            state.templates["gather_elements"]
            .render(
                model_name=model.name,
                op_name=op_name,
                data=params["data"],
                indices=params["indices"],
                output=params["output"],
                params=param_decls,
                c_type=c_type,
                indices_c_type=emitter.ctx_dtype(self.indices).c_type,
                data_suffix=data_suffix,
                indices_suffix=indices_suffix,
                output_suffix=output_suffix,
                output_shape=output_shape,
                loop_vars=loop_vars,
                data_indices=data_indices,
                axis_dim=data_shape[self.axis],
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)


@dataclass(frozen=True)
class GatherOp(GatherLikeOpBase):
    data: str
    indices: str
    output: str
    axis: int

    def _gather_data(self) -> str:
        return self.data

    def _gather_indices(self) -> str:
        return self.indices

    def _gather_output(self) -> str:
        return self.output

    def _gather_axis(self) -> int:
        return self.axis

    def _gather_mode(self) -> str:
        return "gather"


@dataclass(frozen=True)
class ArrayFeatureExtractorOp(RenderableOpBase):
    __io_inputs__ = ("data", "indices")
    __io_outputs__ = ("output",)
    data: str
    indices: str
    output: str

    def infer_types(self, ctx: OpContext) -> None:
        data_dtype = ctx.dtype(self.data)
        if data_dtype == ScalarType.STRING:
            raise UnsupportedOpError(
                "ArrayFeatureExtractor with string tensors is not supported"
            )
        output_dtype = ctx.dtype(self.output)
        if output_dtype != data_dtype:
            raise UnsupportedOpError(
                f"ArrayFeatureExtractor output dtype must match input dtype, "
                f"got {output_dtype.onnx_name} and {data_dtype.onnx_name}"
            )
        indices_dtype = ctx.dtype(self.indices)
        if indices_dtype not in {ScalarType.I64, ScalarType.I32}:
            raise UnsupportedOpError(
                "ArrayFeatureExtractor indices must be int32 or int64, "
                f"got {indices_dtype.onnx_name}"
            )

    def infer_shapes(self, ctx: OpContext) -> None:
        data_shape = ctx.shape(self.data)
        indices_shape = ctx.shape(self.indices)
        if not data_shape:
            raise ShapeInferenceError(
                "ArrayFeatureExtractor does not support scalar input tensors"
            )
        if not indices_shape:
            raise ShapeInferenceError(
                "ArrayFeatureExtractor requires indices to have rank >= 1"
            )
        num_indices = _shape_product(indices_shape)
        if len(data_shape) == 1:
            output_shape = (num_indices,)
        else:
            output_shape = (*data_shape[:-1], num_indices)
        ctx.set_shape(self.output, output_shape)

    def emit(self, emitter: Emitter, ctx: EmitContext) -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        dim_args = emitter.dim_args_str()
        output_dtype = emitter.op_output_dtype(self)
        c_type = output_dtype.c_type
        params = emitter.shared_param_map(
            [
                ("data", self.data),
                ("indices", self.indices),
                ("output", self.output),
            ]
        )
        data_shape = emitter.ctx_shape(self.data)
        indices_shape = emitter.ctx_shape(self.indices)
        output_shape = emitter.ctx_shape(self.output)
        prefix_shape = output_shape[:-1]
        prefix_loop_vars = CEmitterCompat.loop_vars(prefix_shape)
        indices_loop_vars = tuple(f"j{index}" for index in range(len(indices_shape)))
        num_indices = _shape_product(indices_shape)
        indices_strides = _compute_strides(indices_shape)
        data_suffix = emitter.param_array_suffix(data_shape)
        indices_suffix = emitter.param_array_suffix(indices_shape)
        output_suffix = emitter.param_array_suffix(output_shape)
        indices_c_type = emitter.ctx_dtype(self.indices).c_type
        param_decls = emitter.build_param_decls(
            [
                (params["data"], c_type, data_suffix, True),
                (params["indices"], indices_c_type, indices_suffix, True),
                (params["output"], c_type, output_suffix, False),
            ]
        )
        rendered = (
            state.templates["array_feature_extractor"]
            .render(
                op_name=op_name,
                dim_args=dim_args,
                params=param_decls,
                data=params["data"],
                indices=params["indices"],
                output=params["output"],
                prefix_shape=CEmitterCompat.codegen_shape(prefix_shape),
                prefix_loop_vars=prefix_loop_vars,
                indices_shape=CEmitterCompat.codegen_shape(indices_shape),
                indices_loop_vars=indices_loop_vars,
                index_components=tuple(zip(indices_loop_vars, indices_strides)),
                num_indices=num_indices,
                last_axis_dim=data_shape[-1],
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)


@dataclass(frozen=True)
class GatherNDOp(RenderableOpBase):
    __io_inputs__ = ("data", "indices")
    __io_outputs__ = ("output",)
    data: str
    indices: str
    output: str
    batch_dims: int

    def emit(self, emitter: "Emitter", ctx: "EmitContext") -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        emitter.dim_args_str()
        output_dtype = emitter.op_output_dtype(self)
        c_type = output_dtype.c_type
        params = emitter.shared_param_map(
            [
                ("data", self.data),
                ("indices", self.indices),
                ("output", self.output),
            ]
        )
        indices_dim_names = emitter.dim_names_for(self.indices)
        data_dim_names = emitter.dim_names_for(self.data)
        data_shape_raw = emitter.ctx_shape(self.data)
        indices_shape_raw = emitter.ctx_shape(self.indices)
        output_shape_raw = emitter.ctx_shape(self.output)
        data_shape = CEmitterCompat.shape_dim_exprs(data_shape_raw, data_dim_names)
        indices_shape = CEmitterCompat.shape_dim_exprs(
            indices_shape_raw, indices_dim_names
        )
        indices_prefix_shape = indices_shape[:-1]
        indices_prefix_loop_vars = (
            CEmitterCompat.loop_vars(indices_shape_raw[:-1])
            if indices_shape_raw[:-1]
            else ()
        )
        index_depth = indices_shape_raw[-1]
        tail_shape = data_shape[self.batch_dims + index_depth :]
        tail_loop_vars = (
            tuple(f"t{index}" for index in range(len(tail_shape))) if tail_shape else ()
        )
        output_loop_vars = (*indices_prefix_loop_vars, *tail_loop_vars)
        if output_loop_vars:
            output_index_expr = params["output"] + "".join(
                f"[{var}]" for var in output_loop_vars
            )
        else:
            output_index_expr = f"{params['output']}[0]"
        data_index_vars = (
            *indices_prefix_loop_vars[: self.batch_dims],
            *tuple(f"index{idx}" for idx in range(index_depth)),
            *tail_loop_vars,
        )
        data_index_expr = params["data"] + "".join(
            f"[{var}]" for var in data_index_vars
        )
        data_suffix = emitter.param_array_suffix(data_shape_raw)
        indices_suffix = emitter.param_array_suffix(indices_shape_raw)
        output_suffix = emitter.param_array_suffix(output_shape_raw)
        param_decls = emitter.build_param_decls(
            [
                (params["data"], c_type, data_suffix, True),
                (
                    params["indices"],
                    emitter.ctx_dtype(self.indices).c_type,
                    indices_suffix,
                    True,
                ),
                (params["output"], c_type, output_suffix, False),
            ]
        )
        rendered = (
            state.templates["gather_nd"]
            .render(
                model_name=model.name,
                op_name=op_name,
                data=params["data"],
                indices=params["indices"],
                output=params["output"],
                params=param_decls,
                c_type=c_type,
                data_suffix=data_suffix,
                indices_suffix=indices_suffix,
                output_suffix=output_suffix,
                indices_prefix_shape=indices_prefix_shape,
                indices_prefix_loop_vars=indices_prefix_loop_vars,
                index_depth=index_depth,
                tail_shape=tail_shape,
                tail_loop_vars=tail_loop_vars,
                output_index_expr=output_index_expr,
                data_index_expr=data_index_expr,
                batch_dims=self.batch_dims,
                data_shape=data_shape,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)


@dataclass(frozen=True)
class ScatterNDOp(RenderableOpBase):
    __io_inputs__ = ("data", "indices", "updates")
    __io_outputs__ = ("output",)
    data: str
    indices: str
    updates: str
    output: str
    reduction: str

    def emit(self, emitter: "Emitter", ctx: "EmitContext") -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        emitter.dim_args_str()
        output_dtype = emitter.op_output_dtype(self)
        c_type = output_dtype.c_type
        params = emitter.shared_param_map(
            [
                ("data", self.data),
                ("indices", self.indices),
                ("updates", self.updates),
                ("output", self.output),
            ]
        )
        output_dim_names = emitter.dim_names_for(self.output)
        indices_dim_names = emitter.dim_names_for(self.indices)
        updates_dim_names = emitter.dim_names_for(self.updates)
        data_dim_names = emitter.dim_names_for(self.data)
        output_shape_raw = emitter.ctx_shape(self.output)
        data_shape_raw = emitter.ctx_shape(self.data)
        indices_shape_raw = emitter.ctx_shape(self.indices)
        updates_shape_raw = emitter.ctx_shape(self.updates)
        output_shape = CEmitterCompat.shape_dim_exprs(
            output_shape_raw, output_dim_names
        )
        data_shape = CEmitterCompat.shape_dim_exprs(data_shape_raw, data_dim_names)
        indices_shape = CEmitterCompat.shape_dim_exprs(
            indices_shape_raw, indices_dim_names
        )
        output_loop_vars = CEmitterCompat.loop_vars(output_shape_raw)
        indices_prefix_shape = indices_shape[:-1]
        indices_prefix_loop_vars = (
            CEmitterCompat.loop_vars(indices_shape_raw[:-1])
            if indices_shape_raw[:-1]
            else ()
        )
        index_depth = indices_shape_raw[-1]
        tail_shape = output_shape[index_depth:]
        tail_loop_vars = (
            tuple(f"t{index}" for index in range(len(output_shape_raw[index_depth:])))
            if output_shape_raw[index_depth:]
            else ()
        )
        index_vars = tuple(f"index{idx}" for idx in range(index_depth))
        output_index_expr = f"{params['output']}" + "".join(
            f"[{var}]" for var in (*index_vars, *tail_loop_vars)
        )
        updates_index_vars = (*indices_prefix_loop_vars, *tail_loop_vars)
        if not updates_shape_raw:
            updates_index_expr = f"{params['updates']}[0]"
        else:
            updates_index_expr = f"{params['updates']}" + "".join(
                f"[{var}]" for var in updates_index_vars
            )
        data_suffix = emitter.param_array_suffix(data_shape_raw, data_dim_names)
        indices_suffix = emitter.param_array_suffix(
            indices_shape_raw, indices_dim_names
        )
        updates_suffix = emitter.param_array_suffix(
            updates_shape_raw, updates_dim_names
        )
        output_suffix = emitter.param_array_suffix(output_shape_raw, output_dim_names)
        param_decls = emitter.build_param_decls(
            [
                (params["data"], c_type, data_suffix, True),
                (
                    params["indices"],
                    emitter.ctx_dtype(self.indices).c_type,
                    indices_suffix,
                    True,
                ),
                (params["updates"], c_type, updates_suffix, True),
                (params["output"], c_type, output_suffix, False),
            ]
        )
        rendered = (
            state.templates["scatter_nd"]
            .render(
                model_name=model.name,
                op_name=op_name,
                data=params["data"],
                indices=params["indices"],
                updates=params["updates"],
                output=params["output"],
                params=param_decls,
                c_type=c_type,
                output_shape=output_shape,
                output_loop_vars=output_loop_vars,
                indices_prefix_shape=indices_prefix_shape,
                indices_prefix_loop_vars=indices_prefix_loop_vars,
                index_depth=index_depth,
                data_shape=data_shape,
                tail_shape=tail_shape,
                tail_loop_vars=tail_loop_vars,
                output_index_expr=output_index_expr,
                updates_index_expr=updates_index_expr,
                reduction=self.reduction,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)

    def c_op_inputs(
        self, emitter: "Emitter"
    ) -> tuple[tuple[str, tuple[int, ...]], ...]:
        return ((self.data, emitter.ctx_shape(self.data)),)


@dataclass(frozen=True)
class ScatterElementsOp(RenderableOpBase):
    __io_inputs__ = ("data", "indices", "updates")
    __io_outputs__ = ("output",)
    data: str
    indices: str
    updates: str
    output: str
    axis: int
    reduction: str

    def emit(self, emitter: "Emitter", ctx: "EmitContext") -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        emitter.dim_args_str()
        output_dtype = emitter.op_output_dtype(self)
        c_type = output_dtype.c_type
        params = emitter.shared_param_map(
            [
                ("data", self.data),
                ("indices", self.indices),
                ("updates", self.updates),
                ("output", self.output),
            ]
        )
        output_shape_raw = emitter.ctx_shape(self.output)
        updates_shape_raw = emitter.ctx_shape(self.updates)
        output_shape = CEmitterCompat.codegen_shape(output_shape_raw)
        updates_shape = CEmitterCompat.codegen_shape(updates_shape_raw)
        loop_vars = CEmitterCompat.loop_vars(output_shape_raw)
        output_indices = list(loop_vars)
        output_indices[self.axis] = "scatter_index"
        data_shape = emitter.ctx_shape(self.data)
        indices_shape = emitter.ctx_shape(self.indices)
        data_suffix = emitter.param_array_suffix(data_shape)
        indices_suffix = emitter.param_array_suffix(indices_shape)
        updates_suffix = emitter.param_array_suffix(updates_shape_raw)
        output_suffix = emitter.param_array_suffix(output_shape_raw)
        param_decls = emitter.build_param_decls(
            [
                (params["data"], c_type, data_suffix, True),
                (
                    params["indices"],
                    emitter.ctx_dtype(self.indices).c_type,
                    indices_suffix,
                    True,
                ),
                (params["updates"], c_type, updates_suffix, True),
                (params["output"], c_type, output_suffix, False),
            ]
        )
        rendered = (
            state.templates["scatter_elements"]
            .render(
                model_name=model.name,
                op_name=op_name,
                data=params["data"],
                indices=params["indices"],
                updates=params["updates"],
                output=params["output"],
                params=param_decls,
                c_type=c_type,
                output_shape=output_shape,
                updates_shape=updates_shape,
                loop_vars=loop_vars,
                output_indices=output_indices,
                axis_dim=data_shape[self.axis],
                reduction=self.reduction,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)


@dataclass(frozen=True)
class ScatterOp(RenderableOpBase):
    __io_inputs__ = ("data", "indices", "updates")
    __io_outputs__ = ("output",)
    data: str
    indices: str
    updates: str
    output: str
    axis: int

    def emit(self, emitter: "Emitter", ctx: "EmitContext") -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        emitter.dim_args_str()
        output_dtype = emitter.op_output_dtype(self)
        c_type = output_dtype.c_type
        params = emitter.shared_param_map(
            [
                ("data", self.data),
                ("indices", self.indices),
                ("updates", self.updates),
                ("output", self.output),
            ]
        )
        output_shape_raw = emitter.ctx_shape(self.output)
        updates_shape_raw = emitter.ctx_shape(self.updates)
        output_shape = CEmitterCompat.codegen_shape(output_shape_raw)
        updates_shape = CEmitterCompat.codegen_shape(updates_shape_raw)
        loop_vars = CEmitterCompat.loop_vars(output_shape_raw)
        output_indices = list(loop_vars)
        output_indices[self.axis] = "scatter_index"
        data_shape = emitter.ctx_shape(self.data)
        indices_shape = emitter.ctx_shape(self.indices)
        data_suffix = emitter.param_array_suffix(data_shape)
        indices_suffix = emitter.param_array_suffix(indices_shape)
        updates_suffix = emitter.param_array_suffix(updates_shape_raw)
        output_suffix = emitter.param_array_suffix(output_shape_raw)
        param_decls = emitter.build_param_decls(
            [
                (params["data"], c_type, data_suffix, True),
                (
                    params["indices"],
                    emitter.ctx_dtype(self.indices).c_type,
                    indices_suffix,
                    True,
                ),
                (params["updates"], c_type, updates_suffix, True),
                (params["output"], c_type, output_suffix, False),
            ]
        )
        rendered = (
            state.templates["scatter"]
            .render(
                model_name=model.name,
                op_name=op_name,
                data=params["data"],
                indices=params["indices"],
                updates=params["updates"],
                output=params["output"],
                params=param_decls,
                c_type=c_type,
                output_shape=output_shape,
                updates_shape=updates_shape,
                loop_vars=loop_vars,
                output_indices=output_indices,
                axis_dim=data_shape[self.axis],
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)


@dataclass(frozen=True)
class TensorScatterOp(RenderableOpBase):
    __io_inputs__ = ("past_cache", "update", "write_indices")
    __io_outputs__ = ("output",)
    past_cache: str
    update: str
    write_indices: str | None
    output: str
    axis: int
    mode: str

    def infer_types(self, ctx: OpContext) -> None:
        past_dtype = ctx.dtype(self.past_cache)
        update_dtype = ctx.dtype(self.update)
        if past_dtype != update_dtype:
            raise UnsupportedOpError(
                f"{self.kind} expects matching past_cache/update dtypes, "
                f"got {past_dtype.onnx_name} and {update_dtype.onnx_name}"
            )
        if self.write_indices is not None:
            write_dtype = ctx.dtype(self.write_indices)
            if write_dtype not in {ScalarType.I32, ScalarType.I64}:
                raise UnsupportedOpError(
                    f"{self.kind} write_indices must be int32 or int64, "
                    f"got {write_dtype.onnx_name}"
                )
        try:
            output_dtype = ctx.dtype(self.output)
        except ShapeInferenceError:
            ctx.set_dtype(self.output, past_dtype)
            output_dtype = past_dtype
        if output_dtype != past_dtype:
            raise UnsupportedOpError(
                f"{self.kind} expects output dtype {past_dtype.onnx_name}, "
                f"got {output_dtype.onnx_name}"
            )

    def infer_shapes(self, ctx: OpContext) -> None:
        past_shape = ctx.shape(self.past_cache)
        update_shape = ctx.shape(self.update)
        output_shape = ctx.shape(self.output)
        if output_shape != past_shape:
            raise ShapeInferenceError(
                f"{self.kind} output shape must match past_cache shape, "
                f"got {output_shape} and {past_shape}"
            )
        if len(update_shape) != len(past_shape):
            raise ShapeInferenceError(
                f"{self.kind} update rank must match past_cache rank, "
                f"got {len(update_shape)} and {len(past_shape)}"
            )
        for index, (past_dim, update_dim) in enumerate(zip(past_shape, update_shape)):
            if index == self.axis:
                if update_dim > past_dim:
                    raise ShapeInferenceError(
                        f"{self.kind} update dim at axis {self.axis} must not exceed "
                        f"past_cache dim {past_dim}, got {update_dim}"
                    )
                continue
            if update_dim != past_dim:
                raise ShapeInferenceError(
                    f"{self.kind} update shape must match past_cache outside axis {self.axis}, "
                    f"got {update_shape} vs {past_shape}"
                )
        if self.write_indices is not None:
            write_shape = ctx.shape(self.write_indices)
            if len(write_shape) != 1:
                raise ShapeInferenceError(
                    f"{self.kind} write_indices must be rank-1, got shape {write_shape}"
                )
            if write_shape[0] != past_shape[0]:
                raise ShapeInferenceError(
                    f"{self.kind} write_indices length must match batch dim {past_shape[0]}, got {write_shape[0]}"
                )

    def emit(self, emitter: "Emitter", ctx: "EmitContext") -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        emitter.dim_args_str()
        output_dtype = emitter.op_output_dtype(self)
        c_type = output_dtype.c_type
        param_pairs = [
            ("past_cache", self.past_cache),
            ("update", self.update),
            ("output", self.output),
        ]
        if self.write_indices is not None:
            param_pairs.insert(2, ("write_indices", self.write_indices))
        params = emitter.shared_param_map(param_pairs)
        output_dim_names = emitter.dim_names_for(self.output)
        update_dim_names = emitter.dim_names_for(self.update)
        past_dim_names = emitter.dim_names_for(self.past_cache)
        write_indices_dim_names = (
            emitter.dim_names_for(self.write_indices) if self.write_indices else None
        )
        output_shape_tuple = emitter.ctx_shape(self.output)
        update_shape_tuple = emitter.ctx_shape(self.update)
        past_shape_tuple = emitter.ctx_shape(self.past_cache)
        output_shape = CEmitterCompat.shape_dim_exprs(
            output_shape_tuple, output_dim_names
        )
        update_shape = CEmitterCompat.shape_dim_exprs(
            update_shape_tuple, update_dim_names
        )
        prefix_shape = output_shape[: self.axis]
        prefix_loop_vars = (
            CEmitterCompat.loop_vars(output_shape_tuple[: self.axis])
            if output_shape_tuple[: self.axis]
            else ()
        )
        tail_shape = output_shape[self.axis + 1 :]
        tail_loop_vars = (
            tuple(
                f"t{index}" for index in range(len(output_shape_tuple[self.axis + 1 :]))
            )
            if output_shape_tuple[self.axis + 1 :]
            else ()
        )
        output_loop_vars = CEmitterCompat.loop_vars(output_shape_tuple)
        sequence_loop_var = "seq"
        cache_index_var = "cache_index"
        write_index_var = "write_index"
        index_vars = (*prefix_loop_vars, cache_index_var, *tail_loop_vars)
        output_index_expr = f"{params['output']}" + "".join(
            f"[{var}]" for var in index_vars
        )
        update_index_vars = (
            *prefix_loop_vars,
            sequence_loop_var,
            *tail_loop_vars,
        )
        update_index_expr = f"{params['update']}" + "".join(
            f"[{var}]" for var in update_index_vars
        )
        past_suffix = emitter.param_array_suffix(past_shape_tuple, past_dim_names)
        update_suffix = emitter.param_array_suffix(update_shape_tuple, update_dim_names)
        output_suffix = emitter.param_array_suffix(output_shape_tuple, output_dim_names)
        param_decls = [
            (params["past_cache"], c_type, past_suffix, True),
            (params["update"], c_type, update_suffix, True),
        ]
        if self.write_indices is not None:
            write_indices_shape = emitter.ctx_shape(self.write_indices)
            write_indices_dtype = emitter.ctx_dtype(self.write_indices)
            write_indices_suffix = emitter.param_array_suffix(
                write_indices_shape, write_indices_dim_names
            )
            param_decls.append(
                (
                    params["write_indices"],
                    write_indices_dtype.c_type,
                    write_indices_suffix,
                    True,
                )
            )
        param_decls.append((params["output"], c_type, output_suffix, False))
        param_decls_rendered = emitter.build_param_decls(param_decls)
        rendered = (
            state.templates["tensor_scatter"]
            .render(
                model_name=model.name,
                op_name=op_name,
                past_cache=params["past_cache"],
                update=params["update"],
                write_indices=(
                    params.get("write_indices") if self.write_indices else None
                ),
                output=params["output"],
                params=param_decls_rendered,
                c_type=c_type,
                output_shape=output_shape,
                output_loop_vars=output_loop_vars,
                prefix_shape=prefix_shape,
                prefix_loop_vars=prefix_loop_vars,
                sequence_dim=update_shape[self.axis],
                sequence_loop_var=sequence_loop_var,
                tail_shape=tail_shape,
                tail_loop_vars=tail_loop_vars,
                output_index_expr=output_index_expr,
                update_index_expr=update_index_expr,
                max_sequence_length=output_shape[self.axis],
                write_indices_present=self.write_indices is not None,
                batch_index_var=prefix_loop_vars[0] if prefix_loop_vars else "0",
                write_index_var=write_index_var,
                cache_index_var=cache_index_var,
                circular=self.mode == "circular",
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)

    def c_op_inputs(
        self, emitter: "Emitter"
    ) -> tuple[tuple[str, tuple[int, ...]], ...]:
        inputs: list[tuple[str, tuple[int, ...]]] = [
            (self.past_cache, emitter.ctx_shape(self.past_cache)),
            (self.update, emitter.ctx_shape(self.update)),
        ]
        if self.write_indices is not None:
            inputs.append((self.write_indices, emitter.ctx_shape(self.write_indices)))
        return tuple(inputs)


@dataclass(frozen=True)
class TransposeOp(RenderableOpBase):
    __io_inputs__ = ("input0",)
    __io_outputs__ = ("output",)
    input0: str
    output: str
    perm: tuple[int, ...]

    def emit(self, emitter: Emitter, ctx: EmitContext) -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        c_type = emitter.ctx_dtype(self.output).c_type
        input_shape = emitter.ctx_shape(self.input0)
        output_shape_raw = emitter.ctx_shape(self.output)
        input_dim_names = emitter.dim_names_for(self.input0)
        output_dim_names = emitter.dim_names_for(self.output)
        params = emitter.shared_param_map(
            [("input0", self.input0), ("output", self.output)]
        )
        output_shape = CEmitterCompat.shape_dim_exprs(
            output_shape_raw, output_dim_names
        )
        loop_vars = CEmitterCompat.loop_vars(output_shape)
        output_suffix = emitter.param_array_suffix(output_shape_raw, output_dim_names)
        input_suffix = emitter.param_array_suffix(input_shape, input_dim_names)
        param_decls = emitter.build_param_decls(
            [
                (params["input0"], c_type, input_suffix, True),
                (params["output"], c_type, output_suffix, False),
            ]
        )
        if not input_shape:
            input_indices = [loop_vars[0]]
        else:
            input_indices = [None] * len(self.perm)
            for output_axis, input_axis in enumerate(self.perm):
                input_indices[input_axis] = loop_vars[output_axis]
        rendered = (
            state.templates["transpose"]
            .render(
                model_name=model.name,
                op_name=op_name,
                input0=params["input0"],
                output=params["output"],
                params=param_decls,
                c_type=c_type,
                input_suffix=input_suffix,
                output_suffix=output_suffix,
                output_shape=output_shape,
                loop_vars=loop_vars,
                input_indices=input_indices,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)

    def infer_shapes(self, ctx: OpContext) -> None:
        input_shape = ctx.shape(self.input0)
        if len(self.perm) != len(input_shape):
            raise ShapeInferenceError(
                "Transpose perm rank must match input rank, "
                f"got perm {self.perm} for input shape {input_shape}"
            )
        output_shape = tuple(input_shape[axis] for axis in self.perm)
        ctx.set_shape(self.output, output_shape)


@dataclass(frozen=True)
class ReshapeOp(RenderableOpBase):
    __io_inputs__ = ("input0",)
    __io_outputs__ = ("output",)
    input0: str
    output: str

    def required_includes(self, ctx: OpContext) -> set[str]:
        return {"#include <string.h>"}

    def emit(self, emitter: Emitter, ctx: EmitContext) -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        c_type = emitter.ctx_dtype(self.output).c_type
        input_shape = emitter.ctx_shape(self.input0)
        output_shape_raw = emitter.ctx_shape(self.output)
        input_dim_names = emitter.dim_names_for(self.input0)
        output_dim_names = emitter.dim_names_for(self.output)
        params = emitter.shared_param_map(
            [("input0", self.input0), ("output", self.output)]
        )
        input_suffix = emitter.param_array_suffix(input_shape, input_dim_names)
        output_shape = CEmitterCompat.shape_dim_exprs(
            output_shape_raw, output_dim_names
        )
        output_suffix = emitter.param_array_suffix(output_shape_raw, output_dim_names)
        param_decls = emitter.build_param_decls(
            [
                (params["input0"], c_type, input_suffix, True),
                (params["output"], c_type, output_suffix, False),
            ]
        )
        loop_vars = CEmitterCompat.loop_vars(output_shape_raw)
        rendered = (
            state.templates["reshape"]
            .render(
                model_name=model.name,
                op_name=op_name,
                input0=params["input0"],
                output=params["output"],
                params=param_decls,
                c_type=c_type,
                input_suffix=input_suffix,
                output_suffix=output_suffix,
                output_shape=output_shape,
                loop_vars=loop_vars,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)

    def infer_shapes(self, ctx: OpContext) -> None:
        input_shape = ctx.shape(self.input0)
        output_shape = ctx.shape(self.output)
        if _shape_product(input_shape) != _shape_product(output_shape):
            input_dim_params = ctx.graph.find_value(self.input0).type.dim_params
            output_dim_params = ctx.graph.find_value(self.output).type.dim_params
            if any(input_dim_params) or any(output_dim_params):
                return
            raise ShapeInferenceError(
                f"{self.kind} input/output element counts must match, "
                f"got {input_shape} and {output_shape}"
            )


@dataclass(frozen=True)
class EyeLikeOp(RenderableOpBase):
    __io_inputs__ = ("input0",)
    __io_outputs__ = ("output",)
    input0: str
    output: str
    k: int

    def emit(self, emitter: "Emitter", ctx: "EmitContext") -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        emitter.dim_args_str()
        output_dtype = emitter.op_output_dtype(self)
        c_type = output_dtype.c_type
        zero_literal = output_dtype.zero_literal
        input_dtype = emitter.ctx_dtype(self.input0)
        output_shape = emitter.ctx_shape(self.output)
        params = emitter.shared_param_map(
            [("input0", self.input0), ("output", self.output)]
        )
        output_dim_names = emitter.dim_names_for(self.output)
        shape = CEmitterCompat.shape_dim_exprs(output_shape, output_dim_names)
        output_suffix = emitter.param_array_suffix(output_shape, output_dim_names)
        input_suffix = emitter.param_array_suffix(
            output_shape, emitter.dim_names_for(self.input0)
        )
        batch_dims = output_shape[:-2]
        batch_size = CEmitterCompat.element_count(batch_dims or (1,))
        param_decls = emitter.build_param_decls(
            [
                (params["input0"], input_dtype.c_type, input_suffix, True),
                (params["output"], c_type, output_suffix, False),
            ]
        )
        rendered = (
            state.templates["eye_like"]
            .render(
                model_name=model.name,
                op_name=op_name,
                input0=params["input0"],
                output=params["output"],
                params=param_decls,
                c_type=c_type,
                input_suffix=input_suffix,
                output_suffix=output_suffix,
                shape=shape,
                batch_size=batch_size,
                rows=output_shape[-2],
                cols=output_shape[-1],
                k=self.k,
                zero_literal=zero_literal,
                one_literal=f"(({c_type})1)",
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)


@dataclass(frozen=True)
class BernoulliOp(RenderableOpBase):
    __io_inputs__ = ("input0",)
    __io_outputs__ = ("output",)
    input0: str
    output: str
    seed: int | None

    def emit(self, emitter: "Emitter", ctx: "EmitContext") -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        dim_args = emitter.dim_args_str()
        output_dtype = emitter.op_output_dtype(self)
        zero_literal = output_dtype.zero_literal
        output_dim_names = emitter.dim_names_for(self.output)
        output_shape = emitter.ctx_shape(self.output)
        input_shape = emitter.ctx_shape(self.input0)
        input_dtype = emitter.ctx_dtype(self.input0)
        output_dtype = emitter.ctx_dtype(self.output)
        shape = CEmitterCompat.shape_dim_exprs(output_shape, output_dim_names)
        loop_vars = CEmitterCompat.loop_vars(output_shape)
        output_suffix = emitter.param_array_suffix(output_shape, output_dim_names)
        input_suffix = emitter.param_array_suffix(
            input_shape, emitter.dim_names_for(self.input0)
        )
        params = emitter.shared_param_map(
            [("input0", self.input0), ("output", self.output)]
        )
        param_decls = emitter.build_param_decls(
            [
                (params["input0"], input_dtype.c_type, input_suffix, True),
                (params["output"], output_dtype.c_type, output_suffix, False),
            ]
        )
        one_literal = (
            "true" if output_dtype == ScalarType.BOOL else f"({output_dtype.c_type})1"
        )
        zero_literal = (
            "false" if output_dtype == ScalarType.BOOL else output_dtype.zero_literal
        )
        rendered = (
            state.templates["bernoulli"]
            .render(
                model_name=model.name,
                op_name=op_name,
                input0=params["input0"],
                output=params["output"],
                input_index_expr="".join(f"[{var}]" for var in loop_vars),
                output_index_expr="".join(f"[{var}]" for var in loop_vars),
                shape=shape,
                loop_vars=loop_vars,
                seed=self.seed if self.seed is not None else 0,
                one_literal=one_literal,
                zero_literal=zero_literal,
                dim_args=dim_args,
                params=param_decls,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)


@dataclass(frozen=True)
class RandomUniformOp(RenderableOpBase):
    __io_inputs__ = ()
    __io_outputs__ = ("output",)
    output: str
    low: float
    high: float
    seed: int | None

    def emit(self, emitter: "Emitter", ctx: "EmitContext") -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        dim_args = emitter.dim_args_str()
        output_dtype = emitter.op_output_dtype(self)
        output_shape = emitter.ctx_shape(self.output)
        output_dim_names = emitter.dim_names_for(self.output)
        output_suffix = emitter.param_array_suffix(output_shape, output_dim_names)
        params = emitter.shared_param_map([("output", self.output)])
        param_decls = emitter.build_param_decls(
            [
                (params["output"], output_dtype.c_type, output_suffix, False),
            ]
        )
        rendered = (
            state.templates["random_uniform"]
            .render(
                op_name=op_name,
                dim_args=dim_args,
                params=param_decls,
                output=params["output"],
                shape=CEmitterCompat.shape_dim_exprs(output_shape, output_dim_names),
                loop_vars=CEmitterCompat.loop_vars(output_shape),
                output_index_expr="".join(
                    f"[{var}]" for var in CEmitterCompat.loop_vars(output_shape)
                ),
                low=self.low,
                high=self.high,
                c_type=output_dtype.c_type,
                seed=self.seed if self.seed is not None else 0,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)

    def call_args(self) -> tuple[str, ...]:
        return (self.output,)


@dataclass(frozen=True)
class RandomUniformLikeOp(RenderableOpBase):
    __io_inputs__ = ("input0",)
    __io_outputs__ = ("output",)
    input0: str
    output: str
    low: float
    high: float
    seed: int | None

    def emit(self, emitter: "Emitter", ctx: "EmitContext") -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        dim_args = emitter.dim_args_str()
        output_dtype = emitter.op_output_dtype(self)
        output_shape = emitter.ctx_shape(self.output)
        output_dim_names = emitter.dim_names_for(self.output)
        output_suffix = emitter.param_array_suffix(output_shape, output_dim_names)
        input_shape = emitter.ctx_shape(self.input0)
        input_suffix = emitter.param_array_suffix(
            input_shape,
            emitter.dim_names_for(self.input0),
            dtype=emitter.ctx_dtype(self.input0),
        )
        params = emitter.shared_param_map(
            [("input0", self.input0), ("output", self.output)]
        )
        param_decls = emitter.build_param_decls(
            [
                (
                    params["input0"],
                    emitter.ctx_dtype(self.input0).c_type,
                    input_suffix,
                    True,
                ),
                (params["output"], output_dtype.c_type, output_suffix, False),
            ]
        )
        rendered = (
            state.templates["random_uniform"]
            .render(
                op_name=op_name,
                dim_args=dim_args,
                params=param_decls,
                input0=params["input0"],
                output=params["output"],
                shape=CEmitterCompat.shape_dim_exprs(output_shape, output_dim_names),
                loop_vars=CEmitterCompat.loop_vars(output_shape),
                output_index_expr="".join(
                    f"[{var}]" for var in CEmitterCompat.loop_vars(output_shape)
                ),
                low=self.low,
                high=self.high,
                c_type=output_dtype.c_type,
                seed=self.seed if self.seed is not None else 0,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)


@dataclass(frozen=True)
class DropoutOp(RenderableOpBase):
    __io_inputs__ = ("input0", "ratio", "training_mode")
    __io_outputs__ = ("output", "mask")
    input0: str
    output: str
    ratio: str | None
    training_mode: str | None
    mask: str | None
    seed: int | None

    def required_includes(self, ctx: OpContext) -> set[str]:
        if self.training_mode is not None or self.mask is not None:
            return {"#include <stdbool.h>"}
        return set()

    def emit(self, emitter: "Emitter", ctx: "EmitContext") -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        dim_args = emitter.dim_args_str()
        output_shape = emitter.ctx_shape(self.output)
        output_dim_names = emitter.dim_names_for(self.output)
        loop_vars = CEmitterCompat.loop_vars(output_shape)
        shape = CEmitterCompat.shape_dim_exprs(output_shape, output_dim_names)
        input_dtype = emitter.ctx_dtype(self.input0)
        output_dtype = emitter.ctx_dtype(self.output)
        output_suffix = emitter.param_array_suffix(output_shape, output_dim_names)
        input_suffix = emitter.param_array_suffix(
            output_shape, emitter.dim_names_for(self.input0), dtype=input_dtype
        )
        ratio_shape = emitter.ctx_shape(self.ratio) if self.ratio is not None else ()
        training_shape = (
            emitter.ctx_shape(self.training_mode)
            if self.training_mode is not None
            else ()
        )
        ratio_suffix = (
            emitter.param_array_suffix(ratio_shape, emitter.dim_names_for(self.ratio))
            if self.ratio is not None
            else ""
        )
        training_suffix = (
            emitter.param_array_suffix(
                training_shape,
                emitter.dim_names_for(self.training_mode),
                dtype=ScalarType.BOOL,
            )
            if self.training_mode is not None
            else ""
        )
        mask_suffix = (
            emitter.param_array_suffix(
                output_shape, emitter.dim_names_for(self.mask), dtype=ScalarType.BOOL
            )
            if self.mask is not None
            else ""
        )
        params = emitter.shared_param_map(
            [
                ("input0", self.input0),
                ("ratio", self.ratio),
                ("training_mode", self.training_mode),
                ("output", self.output),
                ("mask", self.mask),
            ]
        )
        param_decl_specs = [
            (params["input0"], input_dtype.c_type, input_suffix, True),
        ]
        if self.ratio is not None:
            ratio_dtype = emitter.ctx_dtype(self.ratio)
            param_decl_specs.append(
                (params["ratio"], ratio_dtype.c_type, ratio_suffix, True)
            )
        if self.training_mode is not None:
            param_decl_specs.append(
                (params["training_mode"], "bool", training_suffix, True)
            )
        param_decl_specs.append(
            (params["output"], output_dtype.c_type, output_suffix, False)
        )
        if self.mask is not None:
            param_decl_specs.append((params["mask"], "bool", mask_suffix, False))
        param_decls = emitter.build_param_decls(param_decl_specs)
        rendered = (
            state.templates["dropout"]
            .render(
                model_name=model.name,
                op_name=op_name,
                input0=params["input0"],
                ratio=params.get("ratio"),
                training_mode=params.get("training_mode"),
                output=params["output"],
                mask=params.get("mask"),
                params=param_decls,
                dim_args=dim_args,
                shape=shape,
                loop_vars=loop_vars,
                c_type=output_dtype.c_type,
                ratio_index_expr="[0]" if self.ratio is not None else "",
                training_index_expr="[0]" if self.training_mode is not None else "",
                index_expr="".join(f"[{var}]" for var in loop_vars),
                seed=self.seed if self.seed is not None else 0,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)

    def c_op_outputs(
        self, emitter: "Emitter"
    ) -> tuple[tuple[str, tuple[int, ...], "ScalarType"], ...]:
        outputs: list[tuple[str, tuple[int, ...], ScalarType]] = [
            (
                self.output,
                emitter.ctx_shape(self.output),
                emitter.ctx_dtype(self.output),
            )
        ]
        if self.mask is not None:
            outputs.append(
                (self.mask, emitter.ctx_shape(self.mask), emitter.ctx_dtype(self.mask))
            )
        return tuple(outputs)


@dataclass(frozen=True)
class TriluOp(RenderableOpBase):
    __io_inputs__ = ("input0", "k_input")
    __io_outputs__ = ("output",)
    input0: str
    output: str
    upper: bool
    k_value: int
    k_input: str | None

    def extra_model_dtypes(self, ctx: OpContext) -> set["ScalarType"]:
        if self.k_input is not None:
            return {ctx.dtype(self.k_input)}
        return set()

    def call_args(self) -> tuple[str, ...]:
        args = [self.input0, self.output]
        if self.k_input is not None:
            args.append(self.k_input)
        return tuple(args)

    def emit(self, emitter: "Emitter", ctx: "EmitContext") -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        emitter.dim_args_str()
        output_dtype = emitter.op_output_dtype(self)
        c_type = output_dtype.c_type
        zero_literal = output_dtype.zero_literal
        input_shape = emitter.ctx_shape(self.input0)
        output_shape = emitter.ctx_shape(self.output)
        emitter.ctx_dtype(self.input0)
        param_specs = [("input0", self.input0), ("output", self.output)]
        if self.k_input is not None:
            param_specs.append(("k_input", self.k_input))
        params = emitter.shared_param_map(param_specs)
        output_dim_names = emitter.dim_names_for(self.output)
        shape = CEmitterCompat.shape_dim_exprs(output_shape, output_dim_names)
        output_suffix = emitter.param_array_suffix(output_shape, output_dim_names)
        input_suffix = emitter.param_array_suffix(
            input_shape, emitter.dim_names_for(self.input0)
        )
        k_suffix = ""
        if self.k_input is not None:
            k_suffix = emitter.param_array_suffix(
                emitter.ctx_shape(self.k_input), emitter.dim_names_for(self.k_input)
            )
        batch_dims = output_shape[:-2]
        batch_size = CEmitterCompat.element_count(batch_dims or (1,))
        param_decls = [
            (params["input0"], c_type, input_suffix, True),
            (params["output"], c_type, output_suffix, False),
        ]
        if self.k_input is not None:
            param_decls.append(
                (
                    params["k_input"],
                    emitter.ctx_dtype(self.k_input).c_type,
                    k_suffix,
                    True,
                )
            )
        rendered = (
            state.templates["trilu"]
            .render(
                model_name=model.name,
                op_name=op_name,
                input0=params["input0"],
                output=params["output"],
                k_input=params.get("k_input"),
                params=emitter.build_param_decls(param_decls),
                c_type=c_type,
                k_c_type=(
                    emitter.ctx_dtype(self.k_input).c_type
                    if self.k_input is not None
                    else ScalarType.I64.c_type
                ),
                input_suffix=input_suffix,
                output_suffix=output_suffix,
                shape=shape,
                batch_size=batch_size,
                rows=output_shape[-2],
                cols=output_shape[-1],
                k_value=self.k_value,
                upper=self.upper,
                zero_literal=zero_literal,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)

    def c_op_inputs(
        self, emitter: "Emitter"
    ) -> tuple[tuple[str, tuple[int, ...]], ...]:
        inputs: list[tuple[str, tuple[int, ...]]] = [
            (self.input0, emitter.ctx_shape(self.input0))
        ]
        if self.k_input is not None:
            inputs.append((self.k_input, emitter.ctx_shape(self.k_input)))
        return tuple(inputs)


@dataclass(frozen=True)
class TileOp(RenderableOpBase):
    __io_inputs__ = ("input0", "repeats_input")
    __io_outputs__ = ("output",)
    input0: str
    repeats_input: str
    output: str

    def emit(self, emitter: "Emitter", ctx: "EmitContext") -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        emitter.dim_args_str()
        output_dtype = emitter.op_output_dtype(self)
        c_type = output_dtype.c_type
        input_shape = emitter.ctx_shape(self.input0)
        output_shape_raw = emitter.ctx_shape(self.output)
        params = emitter.shared_param_map(
            [
                ("input0", self.input0),
                ("repeats_input", self.repeats_input),
                ("output", self.output),
            ]
        )
        output_dim_names = emitter.dim_names_for(self.output)
        output_shape = CEmitterCompat.shape_dim_exprs(
            output_shape_raw, output_dim_names
        )
        loop_vars = CEmitterCompat.loop_vars(output_shape_raw)
        input_suffix = emitter.param_array_suffix(
            input_shape, emitter.dim_names_for(self.input0)
        )
        output_suffix = emitter.param_array_suffix(output_shape_raw, output_dim_names)
        param_decls = emitter.build_param_decls(
            [
                (params["input0"], c_type, input_suffix, True),
                (
                    params["repeats_input"],
                    emitter.ctx_dtype(self.repeats_input).c_type,
                    emitter.param_array_suffix(
                        emitter.ctx_shape(self.repeats_input),
                        emitter.dim_names_for(self.repeats_input),
                    ),
                    True,
                ),
                (params["output"], c_type, output_suffix, False),
            ]
        )
        input_strides: list[int] = []
        stride = 1
        for dim in reversed(input_shape):
            input_strides.append(stride)
            stride *= dim
        input_strides = list(reversed(input_strides))
        input_index_terms = [
            f"({var} % {dim}) * {stride}"
            for var, dim, stride in zip(loop_vars, input_shape, input_strides)
        ]
        input_index_expr = " + ".join(input_index_terms) or "0"
        rendered = (
            state.templates["tile"]
            .render(
                model_name=model.name,
                op_name=op_name,
                input0=params["input0"],
                output=params["output"],
                params=param_decls,
                c_type=c_type,
                input_suffix=input_suffix,
                output_suffix=output_suffix,
                output_shape=output_shape,
                loop_vars=loop_vars,
                input_index_expr=input_index_expr,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)


@dataclass(frozen=True)
class CenterCropPadOp(RenderableOpBase):
    __io_inputs__ = ("input0", "shape_input")
    __io_outputs__ = ("output",)
    input0: str
    shape_input: str
    output: str
    axes: (
        tuple[int, ...] | None
    )  # None means all axes; already normalized (non-negative)
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]

    def emit(self, emitter: "Emitter", ctx: "EmitContext") -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        emitter.dim_args_str()
        output_dtype = emitter.op_output_dtype(self)
        c_type = output_dtype.c_type
        input_shape_raw = list(self.input_shape)
        output_shape_raw = list(self.output_shape)
        rank = len(output_shape_raw)
        params = emitter.shared_param_map(
            [
                ("input0", self.input0),
                ("shape_input", self.shape_input),
                ("output", self.output),
            ]
        )
        output_dim_names = emitter.dim_names_for(self.output)
        output_shape = CEmitterCompat.shape_dim_exprs(
            tuple(output_shape_raw), output_dim_names
        )
        out_loop_vars = CEmitterCompat.loop_vars(tuple(output_shape_raw))
        input_suffix = emitter.param_array_suffix(
            tuple(input_shape_raw), emitter.dim_names_for(self.input0)
        )
        output_suffix = emitter.param_array_suffix(
            tuple(output_shape_raw), output_dim_names
        )
        shape_input_dtype = emitter.ctx_dtype(self.shape_input)
        param_decls = emitter.build_param_decls(
            [
                (params["input0"], c_type, input_suffix, True),
                (
                    params["shape_input"],
                    shape_input_dtype.c_type,
                    emitter.param_array_suffix(
                        emitter.ctx_shape(self.shape_input),
                        emitter.dim_names_for(self.shape_input),
                    ),
                    True,
                ),
                (params["output"], c_type, output_suffix, False),
            ]
        )
        # Compute per-axis crop_start and pad_start
        axes = self.axes if self.axes is not None else tuple(range(rank))
        crop_starts = [0] * rank
        pad_starts = [0] * rank
        for a in axes:
            in_dim = input_shape_raw[a]
            out_dim = output_shape_raw[a]
            if in_dim > out_dim:
                crop_starts[a] = (in_dim - out_dim) // 2
            elif out_dim > in_dim:
                pad_starts[a] = (out_dim - in_dim) // 2
        # Compute row-major strides of input
        input_strides: list[int] = []
        stride = 1
        for dim in reversed(input_shape_raw):
            input_strides.append(stride)
            stride *= dim
        input_strides = list(reversed(input_strides))
        rendered = (
            state.templates["center_crop_pad"]
            .render(
                model_name=model.name,
                op_name=op_name,
                input0=params["input0"],
                shape_input=params["shape_input"],
                output=params["output"],
                params=param_decls,
                c_type=c_type,
                input_suffix=input_suffix,
                output_suffix=output_suffix,
                output_shape=output_shape,
                input_shape=input_shape_raw,
                out_loop_vars=out_loop_vars,
                crop_starts=crop_starts,
                pad_starts=pad_starts,
                input_strides=input_strides,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)


@dataclass(frozen=True)
class PadOp(RenderableOpBase):
    __io_inputs__ = ("input0", "pads_input", "axes_input", "value_input")
    __io_outputs__ = ("output",)
    input0: str
    output: str
    pads_begin: tuple[int, ...] | None
    pads_end: tuple[int, ...] | None
    pads_input: str | None
    pads_values: tuple[int, ...] | None
    axes_input: str | None
    mode: str
    value: float | int | bool
    value_input: str | None

    def required_includes(self, ctx: OpContext) -> set[str]:
        return {"#include <stddef.h>"}

    def emit(self, emitter: "Emitter", ctx: "EmitContext") -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        emitter.dim_args_str()
        output_dtype = emitter.op_output_dtype(self)
        c_type = output_dtype.c_type
        input_shape_raw = emitter.ctx_shape(self.input0)
        output_shape_raw = emitter.ctx_shape(self.output)
        input_dim_names = emitter.dim_names_for(self.input0)
        output_dim_names = emitter.dim_names_for(self.output)
        input_shape = CEmitterCompat.shape_dim_exprs(input_shape_raw, input_dim_names)
        output_shape = CEmitterCompat.shape_dim_exprs(
            output_shape_raw, output_dim_names
        )
        in_loop_vars = CEmitterCompat.loop_vars(input_shape_raw)
        out_loop_vars = CEmitterCompat.loop_vars(output_shape_raw)
        idx_vars = tuple(f"pad_idx{index}" for index in range(len(output_shape_raw)))
        reflect_vars = tuple(
            f"pad_reflect{index}" for index in range(len(output_shape_raw))
        )
        pads_c_type = None
        pads_suffix = None
        pads_values = self.pads_values
        if self.pads_input is not None:
            pads_c_type = emitter.ctx_dtype(self.pads_input).c_type
            pads_suffix = emitter.param_array_suffix(
                emitter.ctx_shape(self.pads_input),
                emitter.dim_names_for(self.pads_input),
            )
        elif pads_values is not None:
            pads_c_type = "int64_t"

        axes_c_type = None
        axes_suffix = None
        axes_length = None
        if self.axes_input is not None:
            axes_c_type = emitter.ctx_dtype(self.axes_input).c_type
            axes_suffix = emitter.param_array_suffix(
                emitter.ctx_shape(self.axes_input),
                emitter.dim_names_for(self.axes_input),
            )
            axes_length = emitter.ctx_shape(self.axes_input)[0]
            pad_begin_exprs = tuple(
                f"pad_begin[{index}]" for index in range(len(output_shape_raw))
            )
        elif self.pads_input is not None:
            pad_begin_exprs = tuple(
                f"{self.pads_input}[{index}]" for index in range(len(output_shape_raw))
            )
        else:
            pad_begin_exprs = tuple(str(value) for value in (self.pads_begin or ()))
        if self.value_input is not None:
            value_suffix = emitter.param_array_suffix(
                emitter.ctx_shape(self.value_input),
                emitter.dim_names_for(self.value_input),
            )
            pad_value_expr = f"{self.value_input}[0]"
        else:
            value_suffix = None
            pad_value_expr = emitter.format_literal(
                emitter.ctx_dtype(self.output), self.value
            )
        input_strides: list[int] = []
        stride = 1
        for dim in reversed(input_shape_raw):
            input_strides.append(stride)
            stride *= dim
        input_strides = list(reversed(input_strides))
        rendered = (
            state.templates["pad"]
            .render(
                model_name=model.name,
                op_name=op_name,
                input0=self.input0,
                pads_input=self.pads_input,
                axes_input=self.axes_input,
                value_input=self.value_input,
                output=self.output,
                c_type=c_type,
                pads_c_type=pads_c_type,
                axes_c_type=axes_c_type,
                input_suffix=emitter.param_array_suffix(
                    input_shape_raw, input_dim_names
                ),
                pads_suffix=pads_suffix,
                axes_suffix=axes_suffix,
                value_suffix=value_suffix,
                output_suffix=emitter.param_array_suffix(
                    output_shape_raw, output_dim_names
                ),
                input_shape=input_shape,
                output_shape=output_shape,
                in_loop_vars=in_loop_vars,
                out_loop_vars=out_loop_vars,
                pad_begin_exprs=pad_begin_exprs,
                axes_length=axes_length,
                pads_values=pads_values,
                input_strides=tuple(input_strides),
                mode=self.mode,
                pad_value_expr=pad_value_expr,
                input0_flat="input_flat",
                base_index="pad_index",
                idx_vars=idx_vars,
                reflect_vars=reflect_vars,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)

    def c_op_inputs(
        self, emitter: "Emitter"
    ) -> tuple[tuple[str, tuple[int, ...]], ...]:
        inputs: list[tuple[str, tuple[int, ...]]] = [
            (self.input0, emitter.ctx_shape(self.input0))
        ]
        if self.pads_input is not None:
            inputs.append((self.pads_input, emitter.ctx_shape(self.pads_input)))
        if self.axes_input is not None:
            inputs.append((self.axes_input, emitter.ctx_shape(self.axes_input)))
        if self.value_input is not None:
            inputs.append((self.value_input, emitter.ctx_shape(self.value_input)))
        return tuple(inputs)


@dataclass(frozen=True)
class DepthToSpaceOp(RenderableOpBase):
    __io_inputs__ = ("input0",)
    __io_outputs__ = ("output",)
    input0: str
    output: str
    blocksize: int
    mode: str

    def emit(self, emitter: "Emitter", ctx: "EmitContext") -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        emitter.dim_args_str()
        output_dtype = emitter.op_output_dtype(self)
        c_type = output_dtype.c_type
        input_shape = emitter.ctx_shape(self.input0)
        output_shape = emitter.ctx_shape(self.output)
        params = emitter.shared_param_map(
            [("input0", self.input0), ("output", self.output)]
        )
        output_suffix = emitter.param_array_suffix(output_shape)
        input_suffix = emitter.param_array_suffix(input_shape)
        param_decls = emitter.build_param_decls(
            [
                (params["input0"], c_type, input_suffix, True),
                (params["output"], c_type, output_suffix, False),
            ]
        )
        rendered = (
            state.templates["depth_to_space"]
            .render(
                model_name=model.name,
                op_name=op_name,
                input0=params["input0"],
                output=params["output"],
                params=param_decls,
                c_type=c_type,
                input_suffix=input_suffix,
                output_suffix=output_suffix,
                batch=input_shape[0],
                in_channels=input_shape[1],
                out_channels=output_shape[1],
                in_h=input_shape[2],
                in_w=input_shape[3],
                out_h=output_shape[2],
                out_w=output_shape[3],
                blocksize=self.blocksize,
                mode=self.mode,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)


@dataclass(frozen=True)
class SpaceToDepthOp(RenderableOpBase):
    __io_inputs__ = ("input0",)
    __io_outputs__ = ("output",)
    input0: str
    output: str
    blocksize: int

    def emit(self, emitter: "Emitter", ctx: "EmitContext") -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        emitter.dim_args_str()
        output_dtype = emitter.op_output_dtype(self)
        c_type = output_dtype.c_type
        input_shape = emitter.ctx_shape(self.input0)
        output_shape = emitter.ctx_shape(self.output)
        params = emitter.shared_param_map(
            [("input0", self.input0), ("output", self.output)]
        )
        output_suffix = emitter.param_array_suffix(output_shape)
        input_suffix = emitter.param_array_suffix(input_shape)
        param_decls = emitter.build_param_decls(
            [
                (params["input0"], c_type, input_suffix, True),
                (params["output"], c_type, output_suffix, False),
            ]
        )
        rendered = (
            state.templates["space_to_depth"]
            .render(
                model_name=model.name,
                op_name=op_name,
                input0=params["input0"],
                output=params["output"],
                params=param_decls,
                c_type=c_type,
                input_suffix=input_suffix,
                output_suffix=output_suffix,
                batch=input_shape[0],
                in_channels=input_shape[1],
                out_channels=output_shape[1],
                in_h=input_shape[2],
                in_w=input_shape[3],
                out_h=output_shape[2],
                out_w=output_shape[3],
                blocksize=self.blocksize,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)


@dataclass(frozen=True)
class SliceOp(RenderableOpBase):
    __io_inputs__ = (
        "input0",
        "starts_input",
        "ends_input",
        "axes_input",
        "steps_input",
    )
    __io_outputs__ = ("output",)
    input0: str
    output: str
    starts: tuple[int, ...] | None
    steps: tuple[int, ...] | None
    axes: tuple[int, ...] | None
    starts_input: str | None
    ends_input: str | None
    axes_input: str | None
    steps_input: str | None

    def extra_model_dtypes(self, ctx: OpContext) -> set["ScalarType"]:
        dtypes: set[ScalarType] = set()
        for name in (
            self.starts_input,
            self.ends_input,
            self.axes_input,
            self.steps_input,
        ):
            if name is not None:
                dtypes.add(ctx.dtype(name))
        return dtypes

    def emit(self, emitter: "Emitter", ctx: "EmitContext") -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        emitter.dim_args_str()
        output_dtype = emitter.op_output_dtype(self)
        c_type = output_dtype.c_type
        name_params = emitter.shared_param_map(
            [
                ("input0", self.input0),
                ("starts_input", self.starts_input),
                ("ends_input", self.ends_input),
                ("axes_input", self.axes_input),
                ("steps_input", self.steps_input),
                ("output", self.output),
            ]
        )
        input_shape_raw = emitter.ctx_shape(self.input0)
        output_shape_raw = emitter.ctx_shape(self.output)
        output_shape = CEmitterCompat.codegen_shape(output_shape_raw)
        loop_vars = CEmitterCompat.loop_vars(output_shape)
        if self.starts is not None and self.steps is not None:
            input_indices: list[str] = []
            for start, step, loop_var in zip(self.starts, self.steps, loop_vars):
                if step == 1:
                    if start == 0:
                        input_indices.append(loop_var)
                    else:
                        input_indices.append(f"{start} + {loop_var}")
                else:
                    input_indices.append(f"{start} + {step} * {loop_var}")
            input_suffix = emitter.param_array_suffix(input_shape_raw)
            output_suffix = emitter.param_array_suffix(output_shape_raw)
            param_decls = emitter.build_param_decls(
                [
                    (name_params["input0"], c_type, input_suffix, True),
                    (name_params["output"], c_type, output_suffix, False),
                ]
            )
            rendered = (
                state.templates["slice"]
                .render(
                    model_name=model.name,
                    op_name=op_name,
                    input0=name_params["input0"],
                    output=name_params["output"],
                    params=param_decls,
                    c_type=c_type,
                    input_suffix=input_suffix,
                    output_suffix=output_suffix,
                    output_shape=output_shape,
                    loop_vars=loop_vars,
                    input_indices=input_indices,
                )
                .rstrip()
            )
            return emitter.with_node_comment(model, ctx.op_index, rendered)
        input_suffix = emitter.param_array_suffix(input_shape_raw)
        output_suffix = emitter.param_array_suffix(output_shape_raw)
        params = emitter.build_param_decls(
            [
                (name_params["input0"], c_type, input_suffix, True),
                (
                    (
                        name_params["starts_input"],
                        emitter.ctx_dtype(self.starts_input).c_type,
                        emitter.param_array_suffix(
                            emitter.ctx_shape(self.starts_input)
                        ),
                        True,
                    )
                    if self.starts_input
                    else (None, "", "", True)
                ),
                (
                    (
                        name_params["ends_input"],
                        emitter.ctx_dtype(self.ends_input).c_type,
                        emitter.param_array_suffix(emitter.ctx_shape(self.ends_input)),
                        True,
                    )
                    if self.ends_input
                    else (None, "", "", True)
                ),
                (
                    (
                        name_params["axes_input"],
                        emitter.ctx_dtype(self.axes_input).c_type,
                        emitter.param_array_suffix(emitter.ctx_shape(self.axes_input)),
                        True,
                    )
                    if self.axes_input
                    else (None, "", "", True)
                ),
                (
                    (
                        name_params["steps_input"],
                        emitter.ctx_dtype(self.steps_input).c_type,
                        emitter.param_array_suffix(emitter.ctx_shape(self.steps_input)),
                        True,
                    )
                    if self.steps_input
                    else (None, "", "", True)
                ),
                (name_params["output"], c_type, output_suffix, False),
            ]
        )
        input_dims = CEmitterCompat.codegen_shape(input_shape_raw)
        rendered = (
            state.templates["slice_dynamic"]
            .render(
                model_name=model.name,
                op_name=op_name,
                params=params,
                input0=name_params["input0"],
                starts_input=name_params["starts_input"],
                ends_input=name_params["ends_input"],
                axes_input=name_params["axes_input"],
                steps_input=name_params["steps_input"],
                output=name_params["output"],
                c_type=c_type,
                input_shape=input_dims,
                output_shape=output_shape,
                output_loop_vars=loop_vars,
                input_rank=len(input_shape_raw),
                starts_len=(
                    emitter.ctx_shape(self.starts_input)[0] if self.starts_input else 0
                ),
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)


@dataclass(frozen=True)
class ResizeOp(RenderableOpBase):
    __io_inputs__ = ("input0", "roi_input", "scales_input", "sizes_input")
    __io_outputs__ = ("output",)
    input0: str
    output: str
    scales: tuple[float, ...]
    scales_input: str | None
    sizes_input: str | None
    roi_input: str | None
    axes: tuple[int, ...]
    mode: str
    coordinate_transformation_mode: str
    nearest_mode: str
    cubic_coeff_a: float
    exclude_outside: bool
    extrapolation_value: float
    antialias: bool
    keep_aspect_ratio_policy: str

    def required_includes(self, ctx: OpContext) -> set[str]:
        return {"#include <math.h>"}

    def emit(self, emitter: "Emitter", ctx: "EmitContext") -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        emitter.dim_args_str()
        output_dtype = emitter.op_output_dtype(self)
        c_type = output_dtype.c_type
        name_params = emitter.shared_param_map(
            [
                ("input0", self.input0),
                ("roi_input", self.roi_input),
                ("scales_input", self.scales_input),
                ("sizes_input", self.sizes_input),
                ("output", self.output),
            ]
        )
        input_shape_raw = emitter.ctx_shape(self.input0)
        output_shape_raw = emitter.ctx_shape(self.output)
        input_suffix = emitter.param_array_suffix(input_shape_raw)
        output_suffix = emitter.param_array_suffix(output_shape_raw)
        roi_suffix = None
        scales_suffix = None
        sizes_suffix = None
        roi_c_type = None
        scales_c_type = None
        sizes_c_type = None
        if self.roi_input:
            roi_suffix = emitter.param_array_suffix(emitter.ctx_shape(self.roi_input))
            roi_c_type = emitter.ctx_dtype(self.roi_input).c_type
        if self.scales_input:
            scales_suffix = emitter.param_array_suffix(
                emitter.ctx_shape(self.scales_input)
            )
            scales_c_type = emitter.ctx_dtype(self.scales_input).c_type
        if self.sizes_input:
            sizes_suffix = emitter.param_array_suffix(
                emitter.ctx_shape(self.sizes_input)
            )
            sizes_c_type = emitter.ctx_dtype(self.sizes_input).c_type
        params = emitter.build_param_decls(
            [
                (name_params["input0"], c_type, input_suffix, True),
                (
                    (
                        name_params["roi_input"],
                        roi_c_type or "",
                        roi_suffix or "",
                        True,
                    )
                    if roi_c_type
                    else (None, "", "", True)
                ),
                (
                    (
                        name_params["scales_input"],
                        scales_c_type or "",
                        scales_suffix or "",
                        True,
                    )
                    if scales_c_type
                    else (None, "", "", True)
                ),
                (
                    (
                        name_params["sizes_input"],
                        sizes_c_type or "",
                        sizes_suffix or "",
                        True,
                    )
                    if sizes_c_type
                    else (None, "", "", True)
                ),
                (name_params["output"], c_type, output_suffix, False),
            ]
        )
        rank = len(input_shape_raw)
        scales_axis_map = None
        if self.scales_input:
            scales_len = emitter.ctx_shape(self.scales_input)[0]
            scales_axis_map = (
                tuple(range(scales_len))
                if scales_len == rank
                else tuple(range(len(self.axes)))
            )
        sizes_axis_map = None
        if self.sizes_input:
            sizes_len = emitter.ctx_shape(self.sizes_input)[0]
            sizes_axis_map = (
                tuple(range(sizes_len))
                if sizes_len == rank
                else tuple(range(len(self.axes)))
            )
        roi_axis_map = None
        if self.roi_input:
            roi_len = emitter.ctx_shape(self.roi_input)[0]
            roi_axis_map = (
                tuple(range(roi_len // 2))
                if roi_len == 2 * rank
                else tuple(range(len(self.axes)))
            )
        rendered = (
            state.templates["resize"]
            .render(
                model_name=model.name,
                op_name=op_name,
                params=params,
                input0=name_params["input0"],
                output=name_params["output"],
                c_type=c_type,
                input_suffix=input_suffix,
                output_suffix=output_suffix,
                input_shape=input_shape_raw,
                output_shape=output_shape_raw,
                rank=len(input_shape_raw),
                loop_vars=CEmitterCompat.loop_vars(output_shape_raw),
                scales=self.scales,
                scales_input=name_params["scales_input"],
                sizes_input=name_params["sizes_input"],
                roi_input=name_params["roi_input"],
                roi_suffix=roi_suffix,
                scales_suffix=scales_suffix,
                sizes_suffix=sizes_suffix,
                roi_c_type=roi_c_type,
                scales_c_type=scales_c_type,
                sizes_c_type=sizes_c_type,
                axes=self.axes,
                scales_axis_map=scales_axis_map,
                sizes_axis_map=sizes_axis_map,
                roi_axis_map=roi_axis_map,
                mode=self.mode,
                coordinate_transformation_mode=self.coordinate_transformation_mode,
                nearest_mode=self.nearest_mode,
                cubic_coeff_a=emitter.format_double(self.cubic_coeff_a),
                exclude_outside=self.exclude_outside,
                extrapolation_value=emitter.format_double(self.extrapolation_value),
                antialias=self.antialias,
                keep_aspect_ratio_policy=self.keep_aspect_ratio_policy,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)


@dataclass(frozen=True)
class GridSampleOp(RenderableOpBase):
    __io_inputs__ = ("input0", "grid")
    __io_outputs__ = ("output",)
    input0: str
    grid: str
    output: str
    mode: str
    padding_mode: str
    align_corners: bool

    def required_includes(self, ctx: OpContext) -> set[str]:
        return {"#include <math.h>"}

    def infer_types(self, ctx: OpContext) -> None:
        input_dtype = ctx.dtype(self.input0)
        grid_dtype = ctx.dtype(self.grid)
        if not grid_dtype.is_float:
            raise UnsupportedOpError(
                f"{self.kind} grid dtype must be float, got {grid_dtype.onnx_name}"
            )
        try:
            output_dtype = ctx.dtype(self.output)
        except ShapeInferenceError:
            ctx.set_dtype(self.output, input_dtype)
            output_dtype = input_dtype
        if output_dtype != input_dtype:
            raise UnsupportedOpError(
                f"{self.kind} output dtype must match input dtype {input_dtype.onnx_name}, "
                f"got {output_dtype.onnx_name}"
            )

    def infer_shapes(self, ctx: OpContext) -> None:
        input_shape = ctx.shape(self.input0)
        grid_shape = ctx.shape(self.grid)
        output_shape = ctx.shape(self.output)
        if len(input_shape) < 3:
            raise ShapeInferenceError(
                f"{self.kind} input rank must be >= 3, got {len(input_shape)}"
            )
        spatial_rank = len(input_shape) - 2
        if spatial_rank > 3:
            raise UnsupportedOpError(
                f"{self.kind} supports up to 3 spatial dims, got {spatial_rank}"
            )
        if len(grid_shape) != spatial_rank + 2:
            raise ShapeInferenceError(
                f"{self.kind} grid rank must be {spatial_rank + 2}, got {len(grid_shape)}"
            )
        if len(output_shape) != len(input_shape):
            raise ShapeInferenceError(
                f"{self.kind} output rank must match input rank {len(input_shape)}, got {len(output_shape)}"
            )
        if output_shape[:2] != input_shape[:2]:
            raise ShapeInferenceError(
                f"{self.kind} output batch/channels must match input, got {output_shape[:2]} vs {input_shape[:2]}"
            )
        if output_shape[2:] != grid_shape[1:-1]:
            raise ShapeInferenceError(
                f"{self.kind} output spatial dims must match grid dims {grid_shape[1:-1]}, got {output_shape[2:]}"
            )
        if grid_shape[-1] != spatial_rank:
            raise ShapeInferenceError(
                f"{self.kind} grid last dim must equal spatial rank {spatial_rank}, got {grid_shape[-1]}"
            )

    def emit(self, emitter: "Emitter", ctx: "EmitContext") -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        emitter.dim_args_str()
        output_dtype = emitter.op_output_dtype(self)
        c_type = output_dtype.c_type
        input_shape = emitter.ctx_shape(self.input0)
        grid_shape = emitter.ctx_shape(self.grid)
        output_shape = emitter.ctx_shape(self.output)
        spatial_rank = len(input_shape) - 2
        input_spatial = input_shape[2:]
        output_spatial = output_shape[2:]
        grid_dtype = emitter.ctx_dtype(self.grid)
        input_suffix = emitter.param_array_suffix(input_shape)
        grid_suffix = emitter.param_array_suffix(grid_shape)
        output_suffix = emitter.param_array_suffix(output_shape)
        params = [
            f"const {c_type} {self.input0}{input_suffix}",
            f"const {grid_dtype.c_type} {self.grid}{grid_suffix}",
            f"{c_type} {self.output}{output_suffix}",
        ]
        output_loop_vars = CEmitterCompat.loop_vars(output_shape)
        rendered = (
            state.templates["grid_sample"]
            .render(
                model_name=model.name,
                op_name=op_name,
                params=params,
                input0=self.input0,
                grid=self.grid,
                output=self.output,
                c_type=c_type,
                grid_c_type=grid_dtype.c_type,
                input_suffix=input_suffix,
                grid_suffix=grid_suffix,
                output_suffix=output_suffix,
                input_shape=input_shape,
                grid_shape=grid_shape,
                output_shape=output_shape,
                spatial_rank=spatial_rank,
                input_spatial=input_spatial,
                output_spatial=output_spatial,
                output_loop_vars=output_loop_vars,
                mode=self.mode,
                padding_mode=self.padding_mode,
                align_corners=self.align_corners,
                linear_offsets=tuple(itertools.product((0, 1), repeat=spatial_rank)),
                cubic_offsets=tuple(itertools.product(range(4), repeat=spatial_rank)),
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)


@dataclass(frozen=True)
class AffineGridOp(RenderableOpBase):
    __io_inputs__ = ("theta", "size")
    __io_outputs__ = ("grid",)
    theta: str
    size: str
    grid: str
    align_corners: bool
    spatial_rank: int  # 2 or 3

    def required_includes(self, ctx: OpContext) -> set[str]:
        return {"#include <math.h>"}

    def infer_types(self, ctx: OpContext) -> None:
        theta_dtype = ctx.dtype(self.theta)
        if not theta_dtype.is_float:
            raise UnsupportedOpError(
                f"{self.kind} theta dtype must be float, got {theta_dtype.onnx_name}"
            )
        try:
            grid_dtype = ctx.dtype(self.grid)
        except ShapeInferenceError:
            ctx.set_dtype(self.grid, theta_dtype)
            grid_dtype = theta_dtype
        if grid_dtype != theta_dtype:
            raise UnsupportedOpError(
                f"{self.kind} grid dtype must match theta dtype {theta_dtype.onnx_name}, "
                f"got {grid_dtype.onnx_name}"
            )

    def infer_shapes(self, ctx: OpContext) -> None:
        theta_shape = ctx.shape(self.theta)
        size_shape = ctx.shape(self.size)
        if len(theta_shape) != 3:
            raise ShapeInferenceError(
                f"{self.kind} theta must have rank 3, got {len(theta_shape)}"
            )
        if len(size_shape) != 1:
            raise ShapeInferenceError(
                f"{self.kind} size must have rank 1, got {len(size_shape)}"
            )
        n = theta_shape[0]
        spatial_rank = self.spatial_rank
        grid_shape = ctx.shape(self.grid)
        if len(grid_shape) != spatial_rank + 2:
            raise ShapeInferenceError(
                f"{self.kind} grid rank must be {spatial_rank + 2}, got {len(grid_shape)}"
            )
        if grid_shape[0] != n:
            raise ShapeInferenceError(
                f"{self.kind} grid batch dim must be {n}, got {grid_shape[0]}"
            )
        if grid_shape[-1] != spatial_rank:
            raise ShapeInferenceError(
                f"{self.kind} grid last dim must be {spatial_rank}, got {grid_shape[-1]}"
            )

    def emit(self, emitter: "Emitter", ctx: "EmitContext") -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        emitter.dim_args_str()
        output_dtype = emitter.op_output_dtype(self)
        c_type = output_dtype.c_type
        acc_type = emitter.accumulation_dtype(output_dtype).c_type
        grid_shape = emitter.ctx_shape(self.grid)
        theta_shape = emitter.ctx_shape(self.theta)
        spatial_rank = self.spatial_rank
        n = theta_shape[0]
        spatial_dims = grid_shape[1:-1]  # (H, W) for 2D, (D, H, W) for 3D
        theta_suffix = emitter.param_array_suffix(theta_shape)
        grid_suffix = emitter.param_array_suffix(grid_shape)
        size_len = 2 + spatial_rank
        params = [
            f"const {c_type} {self.theta}{theta_suffix}",
            f"const int64_t {self.size}[{size_len}]",
            f"{c_type} {self.grid}{grid_suffix}",
        ]
        rendered = (
            state.templates["affine_grid"]
            .render(
                model_name=model.name,
                op_name=op_name,
                params=params,
                theta=self.theta,
                size=self.size,
                grid=self.grid,
                c_type=c_type,
                acc_type=acc_type,
                theta_shape=theta_shape,
                grid_shape=grid_shape,
                theta_suffix=theta_suffix,
                grid_suffix=grid_suffix,
                n=n,
                spatial_dims=spatial_dims,
                spatial_rank=spatial_rank,
                align_corners=self.align_corners,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)


@dataclass(frozen=True)
class ConstantOfShapeOp(RenderableOpBase):
    __io_inputs__ = ("input0",)
    __io_outputs__ = ("output",)
    input0: str
    output: str
    value: float | int | bool

    def extra_model_dtypes(self, ctx: OpContext) -> set["ScalarType"]:
        return {ctx.dtype(self.input0)}

    def emit(self, emitter: "Emitter", ctx: "EmitContext") -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        emitter.dim_args_str()
        output_dtype = emitter.op_output_dtype(self)
        params = emitter.shared_param_map(
            [("input0", self.input0), ("output", self.output)]
        )
        output_shape_raw = emitter.ctx_shape(self.output)
        input_shape = emitter.ctx_shape(self.input0)
        input_dtype = emitter.ctx_dtype(self.input0)
        output_dtype = emitter.ctx_dtype(self.output)
        shape = CEmitterCompat.codegen_shape(output_shape_raw)
        loop_vars = CEmitterCompat.loop_vars(output_shape_raw)
        array_suffix = emitter.param_array_suffix(output_shape_raw)
        input_suffix = emitter.param_array_suffix(input_shape)
        param_decls = emitter.build_param_decls(
            [
                (params["input0"], input_dtype.c_type, input_suffix, True),
                (params["output"], output_dtype.c_type, array_suffix, False),
            ]
        )
        rendered = (
            state.templates["constant_of_shape"]
            .render(
                model_name=model.name,
                op_name=op_name,
                input0=params["input0"],
                output=params["output"],
                params=param_decls,
                input_c_type=input_dtype.c_type,
                c_type=output_dtype.c_type,
                input_suffix=input_suffix,
                array_suffix=array_suffix,
                shape=shape,
                loop_vars=loop_vars,
                value_literal=emitter.format_literal(output_dtype, self.value),
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)


@dataclass(frozen=True)
class ShapeOp(RenderableOpBase):
    __io_inputs__ = ("input0",)
    __io_outputs__ = ("output",)
    input0: str
    output: str
    start: int | None
    end: int | None

    def emit(self, emitter: "Emitter", ctx: "EmitContext") -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        emitter.dim_args_str()
        output_dtype = emitter.op_output_dtype(self)
        params = emitter.shared_param_map(
            [("input0", self.input0), ("output", self.output)]
        )
        input_shape = emitter.ctx_shape(self.input0)
        output_shape = emitter.ctx_shape(self.output)
        input_dtype = emitter.ctx_dtype(self.input0)
        output_dtype = emitter.ctx_dtype(self.output)
        input_suffix = emitter.param_array_suffix(input_shape)
        output_suffix = emitter.param_array_suffix(output_shape)
        param_decls = emitter.build_param_decls(
            [
                (params["input0"], input_dtype.c_type, input_suffix, True),
                (params["output"], output_dtype.c_type, output_suffix, False),
            ]
        )
        rank = len(input_shape)
        start_index = 0 if self.start is None else self.start
        end_index = rank if self.end is None else self.end
        if start_index < 0:
            start_index += rank
        if end_index < 0:
            end_index += rank
        start_index = max(0, min(start_index, rank))
        end_index = max(0, min(end_index, rank))
        rendered = (
            state.templates["shape"]
            .render(
                model_name=model.name,
                op_name=op_name,
                input0=params["input0"],
                output=params["output"],
                params=param_decls,
                input_c_type=input_dtype.c_type,
                c_type=output_dtype.c_type,
                input_suffix=input_suffix,
                output_suffix=output_suffix,
                values=[
                    emitter.format_literal(output_dtype, value)
                    for value in input_shape[start_index:end_index]
                ],
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)


@dataclass(frozen=True)
class SizeOp(RenderableOpBase):
    def emit(self, emitter: "Emitter", ctx: "EmitContext") -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        emitter.dim_args_str()
        output_dtype = emitter.op_output_dtype(self)
        params = emitter.shared_param_map(
            [("input0", self.input0), ("output", self.output)]
        )
        input_shape = emitter.ctx_shape(self.input0)
        output_shape = emitter.ctx_shape(self.output)
        input_dtype = emitter.ctx_dtype(self.input0)
        output_dtype = emitter.ctx_dtype(self.output)
        input_suffix = emitter.param_array_suffix(input_shape)
        output_suffix = emitter.param_array_suffix(output_shape)
        param_decls = emitter.build_param_decls(
            [
                (params["input0"], input_dtype.c_type, input_suffix, True),
                (params["output"], output_dtype.c_type, output_suffix, False),
            ]
        )
        rendered = (
            state.templates["size"]
            .render(
                model_name=model.name,
                op_name=op_name,
                input0=params["input0"],
                output=params["output"],
                params=param_decls,
                input_c_type=input_dtype.c_type,
                c_type=output_dtype.c_type,
                input_suffix=input_suffix,
                output_suffix=output_suffix,
                value=emitter.format_literal(
                    output_dtype, CEmitterCompat.element_count(input_shape)
                ),
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)

    __io_inputs__ = ("input0",)
    __io_outputs__ = ("output",)
    input0: str
    output: str


@dataclass(frozen=True)
class OptionalHasElementOp(RenderableOpBase):
    __io_inputs__ = ("input0",)
    __io_outputs__ = ("output",)
    input0: str
    output: str
    input_is_optional: bool

    def emit(self, emitter: "Emitter", ctx: "EmitContext") -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        emitter.dim_args_str()
        output_dtype = emitter.op_output_dtype(self)
        params = emitter.shared_param_map(
            [("input0", self.input0), ("output", self.output)]
        )
        input_shape = emitter.ctx_shape(self.input0)
        output_shape = emitter.ctx_shape(self.output)
        input_dim_names = emitter.dim_names_for(self.input0)
        output_dim_names = emitter.dim_names_for(self.output)
        input_suffix = emitter.param_array_suffix(input_shape, input_dim_names)
        output_suffix = emitter.param_array_suffix(output_shape, output_dim_names)
        input_dtype = emitter.ctx_dtype(self.input0)
        output_dtype = emitter.ctx_dtype(self.output)
        optional_flags = emitter.optional_input_flag_map(model)
        input_flag = optional_flags.get(self.input0)
        param_defs: list[tuple[str, str, str, bool]] = [
            (params["input0"], input_dtype.c_type, input_suffix, True),
        ]
        input_present_expr = "1"
        if self.input_is_optional:
            if input_flag is None:
                raise CodegenError("OptionalHasElement expects an optional input flag.")
            param_defs.append((input_flag, "_Bool", "", True))
            input_present_expr = input_flag
        param_defs.append((params["output"], output_dtype.c_type, output_suffix, False))
        param_decls = emitter.build_param_decls(param_defs)
        rendered = (
            state.templates["optional_has_element"]
            .render(
                model_name=model.name,
                op_name=op_name,
                input0=params["input0"],
                input_present=input_present_expr,
                output=params["output"],
                params=param_decls,
                input_c_type=input_dtype.c_type,
                output_c_type=output_dtype.c_type,
                input_suffix=input_suffix,
                output_suffix=output_suffix,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)

    def call_args(self) -> tuple[str, ...]:
        if self.input_is_optional:
            return (self.input0, f"{self.input0}_present", self.output)
        return (self.input0, self.output)

    def validate(self, ctx: OpContext) -> None:
        value = ctx.graph.find_value(self.input0)
        if value.type.is_optional != self.input_is_optional:
            raise UnsupportedOpError(
                f"{self.kind} optional input typing mismatch for {self.input0}."
            )
        try:
            output_dtype = ctx.dtype(self.output)
        except ShapeInferenceError:
            return None
        if output_dtype != ScalarType.BOOL:
            raise UnsupportedOpError(
                f"{self.kind} expects bool output, got {output_dtype.onnx_name}"
            )
        return None

    def infer_types(self, ctx: OpContext) -> None:
        ctx.dtype(self.input0)
        try:
            output_dtype = ctx.dtype(self.output)
        except ShapeInferenceError:
            ctx.set_dtype(self.output, ScalarType.BOOL)
            return None
        if output_dtype != ScalarType.BOOL:
            raise UnsupportedOpError(
                f"{self.kind} expects bool output, got {output_dtype.onnx_name}"
            )

    def infer_shapes(self, ctx: OpContext) -> None:
        try:
            output_shape = ctx.shape(self.output)
        except ShapeInferenceError:
            ctx.set_shape(self.output, ())
            return None
        if output_shape not in {(), (1,)}:
            raise UnsupportedOpError(
                f"{self.kind} expects scalar output, got shape {output_shape}"
            )


@dataclass(frozen=True)
class OptionalHasElementAbsentOp(RenderableOpBase):
    __io_inputs__ = ()
    __io_outputs__ = ("output",)
    output: str

    def emit(self, emitter: "Emitter", ctx: "EmitContext") -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        dim_args = emitter.dim_args_str()
        output_dtype = emitter.op_output_dtype(self)
        params = emitter.shared_param_map([("output", self.output)])
        output_shape = emitter.ctx_shape(self.output)
        output_dim_names = emitter.dim_names_for(self.output)
        output_suffix = emitter.param_array_suffix(output_shape, output_dim_names)
        param_decls = emitter.build_param_decls(
            [
                (params["output"], output_dtype.c_type, output_suffix, False),
            ]
        )
        rendered = (
            f"EMX_NODE_FN void {op_name}({dim_args}{', '.join(param_decls)}) {{\n"
            f"    {params['output']}[0] = 0;\n"
            "}"
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)

    def call_args(self) -> tuple[str, ...]:
        return (self.output,)

    def validate(self, ctx: OpContext) -> None:
        try:
            output_dtype = ctx.dtype(self.output)
        except ShapeInferenceError:
            return None
        if output_dtype != ScalarType.BOOL:
            raise UnsupportedOpError(
                f"{self.kind} expects bool output, got {output_dtype.onnx_name}"
            )


@dataclass(frozen=True)
class OptionalGetElementOp(RenderableOpBase):
    __io_inputs__ = ("input0",)
    __io_outputs__ = ("output",)
    input0: str
    output: str

    def emit(self, emitter: "Emitter", ctx: "EmitContext") -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        emitter.dim_args_str()
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
        rendered = (
            state.templates["optional_get_element"]
            .render(
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
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)

    def infer_types(self, ctx: OpContext) -> None:
        input_dtype = ctx.dtype(self.input0)
        try:
            output_dtype = ctx.dtype(self.output)
        except ShapeInferenceError:
            ctx.set_dtype(self.output, input_dtype)
            return
        if output_dtype != input_dtype:
            raise UnsupportedOpError(
                f"{self.kind} expects matching input/output dtypes, "
                f"got {input_dtype.onnx_name} and {output_dtype.onnx_name}"
            )

    def infer_shapes(self, ctx: OpContext) -> None:
        input_shape = ctx.shape(self.input0)
        try:
            output_shape = ctx.shape(self.output)
        except ShapeInferenceError:
            ctx.set_shape(self.output, input_shape)
            return
        if output_shape != input_shape:
            raise UnsupportedOpError(
                f"{self.kind} expects matching input/output shapes, "
                f"got {input_shape} and {output_shape}"
            )


@dataclass(frozen=True)
class NonZeroOp(RenderableOpBase):
    __io_inputs__ = ("input0",)
    __io_outputs__ = ("output",)
    input0: str
    output: str

    def emit(self, emitter: "Emitter", ctx: "EmitContext") -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        emitter.dim_args_str()
        output_dtype = emitter.op_output_dtype(self)
        c_type = output_dtype.c_type
        params = emitter.shared_param_map(
            [("input0", self.input0), ("output", self.output)]
        )
        input_dim_names = emitter.dim_names_for(self.input0)
        output_dim_names = emitter.dim_names_for(self.output)
        input_shape_raw = emitter.ctx_shape(self.input0)
        output_shape_raw = emitter.ctx_shape(self.output)
        input_dtype = emitter.ctx_dtype(self.input0)
        input_shape = CEmitterCompat.shape_dim_exprs(input_shape_raw, input_dim_names)
        loop_vars = CEmitterCompat.loop_vars(input_shape_raw)
        input_suffix = emitter.param_array_suffix(input_shape_raw, input_dim_names)
        output_suffix = emitter.param_array_suffix(output_shape_raw, output_dim_names)
        param_decls = emitter.build_param_decls(
            [
                (params["input0"], input_dtype.c_type, input_suffix, True),
                (params["output"], c_type, output_suffix, False),
            ]
        )
        input_expr = f"{params['input0']}" + "".join(f"[{var}]" for var in loop_vars)
        rendered = (
            state.templates["nonzero"]
            .render(
                model_name=model.name,
                op_name=op_name,
                input0=params["input0"],
                output=params["output"],
                params=param_decls,
                input_c_type=input_dtype.c_type,
                output_c_type=c_type,
                input_suffix=input_suffix,
                output_suffix=output_suffix,
                input_shape=input_shape,
                loop_vars=loop_vars,
                input_expr=input_expr,
                zero_literal=input_dtype.zero_literal,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)

    def validate(self, ctx: OpContext) -> None:
        if len(ctx.shape(self.input0)) == 0:
            raise UnsupportedOpError(f"{self.kind} does not support scalar inputs")

    def infer_types(self, ctx: OpContext) -> None:
        ctx.dtype(self.input0)
        try:
            output_dtype = ctx.dtype(self.output)
        except ShapeInferenceError:
            ctx.set_dtype(self.output, ScalarType.I64)
            return
        if output_dtype != ScalarType.I64:
            raise UnsupportedOpError(
                f"{self.kind} output dtype must be int64, got {output_dtype.onnx_name}"
            )

    def infer_shapes(self, ctx: OpContext) -> None:
        input_shape = ctx.shape(self.input0)
        output_shape = ctx.shape(self.output)
        if len(output_shape) != 2:
            raise ShapeInferenceError(f"{self.kind} output must be 2D")
        if output_shape[0] != len(input_shape):
            raise ShapeInferenceError(
                f"{self.kind} output shape must be ({len(input_shape)}, N), got {output_shape}"
            )
        if output_shape[0] < 0 or output_shape[1] < 0:
            raise ShapeInferenceError(f"{self.kind} output shape must be non-negative")
        ctx.set_shape(self.output, output_shape)


@dataclass(frozen=True)
class UniqueOp(RenderableOpBase):
    __io_inputs__ = ("input0",)
    __io_outputs__ = ("y", "indices", "inverse_indices", "counts")
    input0: str
    y: str
    indices: str
    inverse_indices: str
    counts: str
    axis: int | None
    sorted: bool

    def infer_types(self, ctx: OpContext) -> None:
        input_dtype = ctx.dtype(self.input0)
        if input_dtype == ScalarType.STRING:
            raise UnsupportedOpError(f"{self.kind} does not support string input")
        y_dtype = ctx.dtype(self.y)
        if y_dtype != input_dtype:
            raise UnsupportedOpError(
                f"{self.kind} Y dtype must match input dtype {input_dtype.onnx_name}, got {y_dtype.onnx_name}"
            )
        for name in (self.indices, self.inverse_indices, self.counts):
            output_dtype = ctx.dtype(name)
            if output_dtype != ScalarType.I64:
                raise UnsupportedOpError(
                    f"{self.kind} output {name} must be int64, got {output_dtype.onnx_name}"
                )

    def infer_shapes(self, ctx: OpContext) -> None:
        input_shape = ctx.shape(self.input0)
        y_shape = ctx.shape(self.y)
        indices_shape = ctx.shape(self.indices)
        inverse_shape = ctx.shape(self.inverse_indices)
        counts_shape = ctx.shape(self.counts)
        if len(indices_shape) != 1 or len(counts_shape) != 1:
            raise ShapeInferenceError(
                f"{self.kind} indices and counts outputs must be rank-1"
            )
        if self.axis is None:
            if len(y_shape) != 1:
                raise ShapeInferenceError(f"{self.kind} Y output must be rank-1")
            if len(inverse_shape) != 1:
                raise ShapeInferenceError(
                    f"{self.kind} inverse_indices output must be rank-1"
                )
            if inverse_shape[0] != _shape_product(input_shape):
                raise ShapeInferenceError(
                    f"{self.kind} inverse_indices length must be {_shape_product(input_shape)}, got {inverse_shape[0]}"
                )
            if y_shape[0] != indices_shape[0] or y_shape[0] != counts_shape[0]:
                raise ShapeInferenceError(
                    f"{self.kind} Y/indices/counts lengths must match, got {y_shape[0]}, {indices_shape[0]}, {counts_shape[0]}"
                )
            return None
        axis = self.axis
        rank = len(input_shape)
        if axis < 0 or axis >= rank:
            raise ShapeInferenceError(
                f"{self.kind} axis {axis} out of range for rank {rank}"
            )
        if len(y_shape) != rank:
            raise ShapeInferenceError(
                f"{self.kind} Y output rank must be {rank}, got {len(y_shape)}"
            )
        for dim_index, (in_dim, out_dim) in enumerate(zip(input_shape, y_shape)):
            if dim_index == axis:
                continue
            if in_dim != out_dim:
                raise ShapeInferenceError(
                    f"{self.kind} Y output shape must match input outside axis {axis}, got {y_shape} for input {input_shape}"
                )
        if indices_shape[0] != y_shape[axis] or counts_shape[0] != y_shape[axis]:
            raise ShapeInferenceError(
                f"{self.kind} indices/counts length must match Y axis {axis} size {y_shape[axis]}"
            )
        if len(inverse_shape) != 1 or inverse_shape[0] != input_shape[axis]:
            raise ShapeInferenceError(
                f"{self.kind} inverse_indices length must be input axis {axis} size {input_shape[axis]}"
            )

    def emit(self, emitter: "Emitter", ctx: "EmitContext") -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        dim_args = emitter.dim_args_str()
        emitter.op_output_dtype(self)
        params = emitter.shared_param_map(
            [
                ("input0", self.input0),
                ("y", self.y),
                ("indices", self.indices),
                ("inverse_indices", self.inverse_indices),
                ("counts", self.counts),
            ]
        )
        input_shape = emitter.ctx_shape(self.input0)
        y_shape = emitter.ctx_shape(self.y)
        input_dtype = emitter.ctx_dtype(self.input0)
        y_dtype = emitter.ctx_dtype(self.y)
        indices_dtype = emitter.ctx_dtype(self.indices)
        inverse_dtype = emitter.ctx_dtype(self.inverse_indices)
        counts_dtype = emitter.ctx_dtype(self.counts)
        input_suffix = emitter.param_array_suffix(
            input_shape, emitter.dim_names_for(self.input0)
        )
        y_suffix = emitter.param_array_suffix(y_shape, emitter.dim_names_for(self.y))
        indices_suffix = emitter.param_array_suffix(
            emitter.ctx_shape(self.indices), emitter.dim_names_for(self.indices)
        )
        inverse_suffix = emitter.param_array_suffix(
            emitter.ctx_shape(self.inverse_indices),
            emitter.dim_names_for(self.inverse_indices),
        )
        counts_suffix = emitter.param_array_suffix(
            emitter.ctx_shape(self.counts), emitter.dim_names_for(self.counts)
        )
        param_decls = emitter.build_param_decls(
            [
                (params["input0"], input_dtype.c_type, input_suffix, True),
                (params["y"], y_dtype.c_type, y_suffix, False),
                (params["indices"], indices_dtype.c_type, indices_suffix, False),
                (
                    params["inverse_indices"],
                    inverse_dtype.c_type,
                    inverse_suffix,
                    False,
                ),
                (params["counts"], counts_dtype.c_type, counts_suffix, False),
            ]
        )
        rendered = (
            state.templates["unique"]
            .render(
                model_name=model.name,
                op_name=op_name,
                params=param_decls,
                dim_args=dim_args,
                input0=params["input0"],
                y=params["y"],
                indices=params["indices"],
                inverse_indices=params["inverse_indices"],
                counts=params["counts"],
                input_shape=CEmitterCompat.shape_dim_exprs(
                    input_shape, emitter.dim_names_for(self.input0)
                ),
                y_shape=CEmitterCompat.shape_dim_exprs(
                    y_shape, emitter.dim_names_for(self.y)
                ),
                input_rank=len(input_shape),
                axis=-1 if self.axis is None else self.axis,
                flat_size=CEmitterCompat.element_count_expr(input_shape),
                axis_dim=(0 if self.axis is None else input_shape[self.axis]),
                y_axis_dim=(y_shape[0] if self.axis is None else y_shape[self.axis]),
                outer=(
                    0
                    if self.axis is None
                    else CEmitterCompat.element_count_expr(input_shape[: self.axis])
                ),
                inner=(
                    0
                    if self.axis is None
                    else CEmitterCompat.element_count_expr(input_shape[self.axis + 1 :])
                ),
                sorted_output=1 if self.sorted else 0,
                input_c_type=input_dtype.c_type,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)

    def c_op_outputs(
        self, emitter: "Emitter"
    ) -> tuple[tuple[str, tuple[int, ...], "ScalarType"], ...]:
        return (
            (self.y, emitter.ctx_shape(self.y), emitter.ctx_dtype(self.y)),
            (
                self.indices,
                emitter.ctx_shape(self.indices),
                emitter.ctx_dtype(self.indices),
            ),
            (
                self.inverse_indices,
                emitter.ctx_shape(self.inverse_indices),
                emitter.ctx_dtype(self.inverse_indices),
            ),
            (
                self.counts,
                emitter.ctx_shape(self.counts),
                emitter.ctx_dtype(self.counts),
            ),
        )


@dataclass(frozen=True)
class NonMaxSuppressionOp(RenderableOpBase):
    __io_inputs__ = (
        "boxes",
        "scores",
        "max_output_boxes_per_class",
        "iou_threshold",
        "score_threshold",
    )
    __io_outputs__ = ("output",)
    boxes: str
    scores: str
    max_output_boxes_per_class: str | None
    iou_threshold: str | None
    score_threshold: str | None
    output: str
    center_point_box: int

    def required_includes(self, ctx: OpContext) -> set[str]:
        return {"#include <limits.h>"}

    def _validate_scalar_input(
        self,
        ctx: OpContext,
        *,
        name: str,
        allowed_dtypes: set[ScalarType],
        label: str,
    ) -> None:
        dtype = ctx.dtype(name)
        if dtype not in allowed_dtypes:
            allowed = ", ".join(sorted(d.onnx_name for d in allowed_dtypes))
            raise UnsupportedOpError(
                f"{self.kind} {label} must be {allowed}, got {dtype.onnx_name}"
            )
        shape = ctx.shape(name)
        if shape not in {(), (1,)} and int(np.prod(shape, dtype=np.int64)) != 1:
            raise ShapeInferenceError(
                f"{self.kind} {label} must be a scalar tensor, got shape {shape}"
            )

    def validate(self, ctx: OpContext) -> None:
        boxes_shape = ctx.shape(self.boxes)
        scores_shape = ctx.shape(self.scores)
        if len(boxes_shape) != 3 or boxes_shape[2] != 4:
            raise ShapeInferenceError(
                f"{self.kind} boxes input must have shape "
                f"[num_batches, num_boxes, 4], got {boxes_shape}"
            )
        if len(scores_shape) != 3:
            raise ShapeInferenceError(
                f"{self.kind} scores input must have shape "
                f"[num_batches, num_classes, num_boxes], got {scores_shape}"
            )
        if boxes_shape[0] != scores_shape[0]:
            raise ShapeInferenceError(
                f"{self.kind} boxes/scores batch dims must match, "
                f"got {boxes_shape[0]} and {scores_shape[0]}"
            )
        if boxes_shape[1] != scores_shape[2]:
            raise ShapeInferenceError(
                f"{self.kind} boxes num_boxes dim {boxes_shape[1]} "
                f"must match scores num_boxes dim {scores_shape[2]}"
            )
        boxes_dtype = ctx.dtype(self.boxes)
        scores_dtype = ctx.dtype(self.scores)
        if boxes_dtype != scores_dtype or not boxes_dtype.is_float:
            raise UnsupportedOpError(
                f"{self.kind} boxes and scores must be the same float dtype, "
                f"got {boxes_dtype.onnx_name} and {scores_dtype.onnx_name}"
            )
        if self.max_output_boxes_per_class is not None:
            self._validate_scalar_input(
                ctx,
                name=self.max_output_boxes_per_class,
                allowed_dtypes={ScalarType.I32, ScalarType.I64},
                label="max_output_boxes_per_class input",
            )
        if self.iou_threshold is not None:
            self._validate_scalar_input(
                ctx,
                name=self.iou_threshold,
                allowed_dtypes={ScalarType.F32, ScalarType.F64},
                label="iou_threshold input",
            )
        if self.score_threshold is not None:
            self._validate_scalar_input(
                ctx,
                name=self.score_threshold,
                allowed_dtypes={ScalarType.F32, ScalarType.F64},
                label="score_threshold input",
            )
        if self.center_point_box not in {0, 1}:
            raise UnsupportedOpError(
                f"{self.kind} center_point_box must be 0 or 1, got {self.center_point_box}"
            )

    def infer_types(self, ctx: OpContext) -> None:
        ctx.dtype(self.boxes)
        ctx.dtype(self.scores)
        if self.max_output_boxes_per_class is not None:
            ctx.dtype(self.max_output_boxes_per_class)
        if self.iou_threshold is not None:
            ctx.dtype(self.iou_threshold)
        if self.score_threshold is not None:
            ctx.dtype(self.score_threshold)
        try:
            output_dtype = ctx.dtype(self.output)
        except ShapeInferenceError:
            ctx.set_dtype(self.output, ScalarType.I64)
            output_dtype = ScalarType.I64
        if output_dtype != ScalarType.I64:
            raise UnsupportedOpError(
                f"{self.kind} output dtype must be int64, got {output_dtype.onnx_name}"
            )

    def infer_shapes(self, ctx: OpContext) -> None:
        output_shape = ctx.shape(self.output)
        if len(output_shape) != 2 or output_shape[1] != 3:
            raise ShapeInferenceError(
                f"{self.kind} output must have shape [num_selected, 3], "
                f"got {output_shape}"
            )
        ctx.set_shape(self.output, output_shape)

    def emit(self, emitter: "Emitter", ctx: "EmitContext") -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        dim_args = emitter.dim_args_str()
        output_dtype = emitter.op_output_dtype(self)
        boxes_shape = emitter.ctx_shape(self.boxes)
        scores_shape = emitter.ctx_shape(self.scores)
        output_shape = emitter.ctx_shape(self.output)
        boxes_dtype = emitter.ctx_dtype(self.boxes)
        output_dtype = emitter.ctx_dtype(self.output)
        max_output_shape = (
            emitter.ctx_shape(self.max_output_boxes_per_class)
            if self.max_output_boxes_per_class is not None
            else None
        )
        max_output_dtype = (
            emitter.ctx_dtype(self.max_output_boxes_per_class)
            if self.max_output_boxes_per_class is not None
            else None
        )
        iou_threshold_shape = (
            emitter.ctx_shape(self.iou_threshold)
            if self.iou_threshold is not None
            else None
        )
        iou_threshold_dtype = (
            emitter.ctx_dtype(self.iou_threshold)
            if self.iou_threshold is not None
            else None
        )
        score_threshold_shape = (
            emitter.ctx_shape(self.score_threshold)
            if self.score_threshold is not None
            else None
        )
        score_threshold_dtype = (
            emitter.ctx_dtype(self.score_threshold)
            if self.score_threshold is not None
            else None
        )

        params = emitter.shared_param_map(
            [
                ("boxes", self.boxes),
                ("scores", self.scores),
                ("max_output_boxes_per_class", self.max_output_boxes_per_class),
                ("iou_threshold", self.iou_threshold),
                ("score_threshold", self.score_threshold),
                ("output", self.output),
            ]
        )
        boxes_suffix = emitter.param_array_suffix(
            boxes_shape, emitter.dim_names_for(self.boxes)
        )
        scores_suffix = emitter.param_array_suffix(
            scores_shape, emitter.dim_names_for(self.scores)
        )
        output_suffix = emitter.param_array_suffix(
            output_shape, emitter.dim_names_for(self.output)
        )
        max_output_suffix = (
            emitter.param_array_suffix(
                max_output_shape,
                emitter.dim_names_for(self.max_output_boxes_per_class or ""),
            )
            if max_output_shape is not None
            else ""
        )
        iou_threshold_suffix = (
            emitter.param_array_suffix(
                iou_threshold_shape,
                emitter.dim_names_for(self.iou_threshold or ""),
            )
            if iou_threshold_shape is not None
            else ""
        )
        score_threshold_suffix = (
            emitter.param_array_suffix(
                score_threshold_shape,
                emitter.dim_names_for(self.score_threshold or ""),
            )
            if score_threshold_shape is not None
            else ""
        )
        param_decls = emitter.build_param_decls(
            [
                (params["boxes"], boxes_dtype.c_type, boxes_suffix, True),
                (params["scores"], boxes_dtype.c_type, scores_suffix, True),
                (
                    (
                        params["max_output_boxes_per_class"],
                        max_output_dtype.c_type if max_output_dtype else "",
                        max_output_suffix,
                        True,
                    )
                    if params["max_output_boxes_per_class"]
                    else (None, "", "", True)
                ),
                (
                    (
                        params["iou_threshold"],
                        iou_threshold_dtype.c_type if iou_threshold_dtype else "",
                        iou_threshold_suffix,
                        True,
                    )
                    if params["iou_threshold"]
                    else (None, "", "", True)
                ),
                (
                    (
                        params["score_threshold"],
                        (score_threshold_dtype.c_type if score_threshold_dtype else ""),
                        score_threshold_suffix,
                        True,
                    )
                    if params["score_threshold"]
                    else (None, "", "", True)
                ),
                (params["output"], output_dtype.c_type, output_suffix, False),
            ]
        )
        rendered = (
            state.templates["nonmax_suppression"]
            .render(
                model_name=model.name,
                op_name=op_name,
                boxes=params["boxes"],
                scores=params["scores"],
                max_output_boxes_per_class=params["max_output_boxes_per_class"],
                iou_threshold=params["iou_threshold"],
                score_threshold=params["score_threshold"],
                output=params["output"],
                params=param_decls,
                input_c_type=boxes_dtype.c_type,
                output_c_type=output_dtype.c_type,
                compute_type=boxes_dtype.c_type,
                output_capacity=output_shape[0],
                num_batches=boxes_shape[0],
                num_boxes=boxes_shape[1],
                num_classes=scores_shape[1],
                center_point_box=self.center_point_box,
                compute_scalar_dtype=boxes_dtype,
                iou_threshold_default=boxes_dtype.zero_literal,
                score_threshold_default=boxes_dtype.zero_literal,
                score_threshold_enabled=self.score_threshold is not None,
                dim_args=dim_args,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)

    def c_op_inputs(
        self, emitter: "Emitter"
    ) -> tuple[tuple[str, tuple[int, ...]], ...]:
        inputs: list[tuple[str, tuple[int, ...]]] = [
            (self.boxes, emitter.ctx_shape(self.boxes)),
            (self.scores, emitter.ctx_shape(self.scores)),
        ]
        if self.max_output_boxes_per_class is not None:
            inputs.append(
                (
                    self.max_output_boxes_per_class,
                    emitter.ctx_shape(self.max_output_boxes_per_class),
                )
            )
        if self.iou_threshold is not None:
            inputs.append((self.iou_threshold, emitter.ctx_shape(self.iou_threshold)))
        if self.score_threshold is not None:
            inputs.append(
                (self.score_threshold, emitter.ctx_shape(self.score_threshold))
            )
        return tuple(inputs)


@dataclass(frozen=True)
class ExpandOp(ShapeLikeOpBase):
    input0: str
    input_shape: str
    output: str

    def _shape_data(self) -> str:
        return self.input0

    def _shape_output(self) -> str:
        return self.output

    def _shape_mode(self) -> str:
        return "expand"

    def _shape_spec(self, ctx: OpContext) -> tuple[int, ...]:
        initializer = ctx.initializer(self.input_shape)
        if initializer is not None:
            if initializer.type.dtype not in {ScalarType.I64, ScalarType.I32}:
                raise UnsupportedOpError(
                    f"{self.kind} shape input must be int64 or int32"
                )
            if len(initializer.type.shape) != 1:
                raise UnsupportedOpError(f"{self.kind} shape input must be a 1D tensor")
            values = np.array(initializer.data, dtype=np.int64).reshape(-1)
            if values.size == 0:
                raise ShapeInferenceError(f"{self.kind} shape input cannot be empty")
            return tuple(int(value) for value in values)
        dtype = ctx.dtype(self.input_shape)
        if dtype not in {ScalarType.I64, ScalarType.I32}:
            raise UnsupportedOpError(f"{self.kind} shape input must be int64 or int32")
        shape = ctx.shape(self.input_shape)
        if len(shape) != 1:
            raise UnsupportedOpError(f"{self.kind} shape input must be a 1D tensor")
        if shape[0] <= 0:
            raise ShapeInferenceError(f"{self.kind} shape input cannot be empty")
        output_shape = ctx.shape(self.output)
        if not output_shape:
            raise ShapeInferenceError(f"{self.kind} output shape must be specified")
        return output_shape

    def _shape_derived(
        self,
        ctx: OpContext,
        *,
        data_shape: tuple[int, ...],
        target_shape: tuple[int, ...],
        output_shape: tuple[int, ...],
    ) -> None:
        input_shape_padded = (1,) * (len(output_shape) - len(data_shape)) + data_shape
        input_strides = _compute_strides(input_shape_padded)
        ctx.set_derived(self, "input_shape_padded", input_shape_padded)
        ctx.set_derived(self, "input_strides", input_strides)


@dataclass(frozen=True)
class CumSumOp(RenderableOpBase):
    __io_inputs__ = ("input0", "axis_input")
    __io_outputs__ = ("output",)
    input0: str
    axis_input: str | None
    axis: int | None
    output: str
    exclusive: bool
    reverse: bool

    def emit(self, emitter: "Emitter", ctx: "EmitContext") -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        dim_args = emitter.dim_args_str()
        output_dtype = emitter.op_output_dtype(self)
        c_type = output_dtype.c_type
        input_shape = emitter.ctx_shape(self.input0)
        params = emitter.unique_param_map(
            [
                ("input0", self.input0),
                ("axis_input", self.axis_input),
                ("output", self.output),
            ]
        )
        input_dim_names = emitter.dim_names_for(self.input0)
        output_dim_names = emitter.dim_names_for(self.output)
        axis_c_type = (
            emitter.ctx_dtype(self.axis_input).c_type
            if self.axis_input is not None
            else "int64_t"
        )
        rendered = (
            state.templates["cumsum"]
            .render(
                model_name=model.name,
                op_name=op_name,
                input0=params["input0"],
                axis_input=params["axis_input"],
                axis_c_type=axis_c_type,
                axis_suffix=emitter.param_array_suffix(()),
                axis_literal=self.axis,
                output=params["output"],
                c_type=c_type,
                input_suffix=emitter.param_array_suffix(input_shape, input_dim_names),
                output_suffix=emitter.param_array_suffix(input_shape, output_dim_names),
                input_shape=CEmitterCompat.shape_dim_exprs(
                    input_shape, input_dim_names
                ),
                rank=len(input_shape),
                exclusive=self.exclusive,
                reverse=self.reverse,
                dim_args=dim_args,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)

    def c_op_inputs(
        self, emitter: "Emitter"
    ) -> tuple[tuple[str, tuple[int, ...]], ...]:
        return ((self.input0, emitter.ctx_shape(self.input0)),)


@dataclass(frozen=True)
class STFTOp(RenderableOpBase):
    __io_inputs__ = ("signal", "frame_step", "window", "frame_length_input")
    __io_outputs__ = ("output",)
    signal: str
    frame_step: str
    window: str | None
    frame_length_input: str | None
    output: str
    onesided: bool
    input_is_complex: bool
    fft_length: int
    window_length: int
    frame_length_literal: int
    use_runtime_frame_length: bool

    def emit(self, emitter: "Emitter", ctx: "EmitContext") -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        dim_args = emitter.dim_args_str()
        emitter.op_output_dtype(self)
        signal_shape = emitter.ctx_shape(self.signal)
        output_shape = emitter.ctx_shape(self.output)
        stft_dtype = emitter.ctx_dtype(self.output)
        if stft_dtype != emitter.ctx_dtype(self.signal):
            raise CodegenError("STFT signal and output dtypes must match")
        if len(signal_shape) != 3 or len(output_shape) != 4:
            raise CodegenError("STFT expects signal rank 3 and output rank 4")
        if signal_shape[2] not in {1, 2}:
            raise CodegenError("STFT signal last dimension must be 1 or 2")
        if output_shape[3] != 2:
            raise CodegenError("STFT output last dimension must be 2")
        if self.onesided and self.input_is_complex:
            raise CodegenError("STFT onesided output is invalid for complex input")
        if self.fft_length <= 0:
            raise CodegenError("STFT frame_length must be > 0")
        if self.window_length <= 0:
            raise CodegenError("STFT window length must be > 0")

        frame_step_dtype = emitter.ctx_dtype(self.frame_step)
        if frame_step_dtype not in {ScalarType.I32, ScalarType.I64}:
            raise CodegenError("STFT frame_step must be int32 or int64")

        frame_length_c_type = frame_step_dtype.c_type
        if self.frame_length_input is not None:
            frame_length_dtype = emitter.ctx_dtype(self.frame_length_input)
            if frame_length_dtype not in {ScalarType.I32, ScalarType.I64}:
                raise CodegenError("STFT frame_length input must be int32 or int64")
            frame_length_c_type = frame_length_dtype.c_type

        params = emitter.unique_param_map(
            [
                ("signal", self.signal),
                ("frame_step", self.frame_step),
                ("window", self.window),
                ("frame_length_input", self.frame_length_input),
                ("output", self.output),
            ]
        )
        signal_dim_names = emitter.dim_names_for(self.signal)
        output_dim_names = emitter.dim_names_for(self.output)
        signal_shape_expr = CEmitterCompat.shape_dim_exprs(
            signal_shape, signal_dim_names
        )
        output_shape_expr = CEmitterCompat.shape_dim_exprs(
            output_shape, output_dim_names
        )

        signal_suffix = emitter.param_array_suffix(
            signal_shape,
            signal_dim_names,
            dtype=stft_dtype,
        )
        output_suffix = emitter.param_array_suffix(
            output_shape,
            output_dim_names,
            dtype=stft_dtype,
        )
        window_suffix = None
        if self.window is not None:
            window_shape = emitter.ctx_shape(self.window)
            window_suffix = emitter.param_array_suffix(
                window_shape,
                dtype=stft_dtype,
            )
            if emitter.ctx_dtype(self.window) != stft_dtype:
                raise CodegenError("STFT window dtype must match signal dtype")

        scalar_suffix = emitter.param_array_suffix(())
        param_decls = emitter.build_param_decls(
            [
                (params["signal"], stft_dtype.c_type, signal_suffix, True),
                (
                    params["frame_step"],
                    frame_step_dtype.c_type,
                    scalar_suffix,
                    True,
                ),
                (params["window"], stft_dtype.c_type, window_suffix or "", True),
                (
                    params["frame_length_input"],
                    frame_length_c_type,
                    scalar_suffix,
                    True,
                ),
                (params["output"], stft_dtype.c_type, output_suffix, False),
            ]
        )

        rendered = (
            state.templates["stft"]
            .render(
                model_name=model.name,
                op_name=op_name,
                params=param_decls,
                signal=params["signal"],
                frame_step=params["frame_step"],
                window=params["window"],
                frame_length_input=params["frame_length_input"],
                output=params["output"],
                c_type=stft_dtype.c_type,
                signal_suffix=signal_suffix,
                frame_step_suffix=scalar_suffix,
                window_suffix=window_suffix,
                frame_length_suffix=scalar_suffix,
                output_suffix=output_suffix,
                frame_step_c_type=frame_step_dtype.c_type,
                frame_length_c_type=frame_length_c_type,
                batch_expr=str(signal_shape_expr[0]),
                signal_length_expr=str(signal_shape_expr[1]),
                output_frames_expr=str(output_shape_expr[1]),
                output_bins_expr=str(output_shape_expr[2]),
                input_components=2 if self.input_is_complex else 1,
                fft_dtype=stft_dtype,
                fft_length=self.fft_length,
                window_length=self.window_length,
                frame_length_literal=self.frame_length_literal,
                use_runtime_frame_length=self.use_runtime_frame_length,
                use_window=self.window is not None,
                zero_literal=stft_dtype.zero_literal,
                dim_args=dim_args,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)

    def c_op_inputs(
        self, emitter: "Emitter"
    ) -> tuple[tuple[str, tuple[int, ...]], ...]:
        inputs: list[tuple[str, tuple[int, ...]]] = [
            (self.signal, emitter.ctx_shape(self.signal)),
            (self.frame_step, ()),
        ]
        if self.window is not None:
            inputs.append((self.window, emitter.ctx_shape(self.window)))
        if self.frame_length_input is not None:
            inputs.append((self.frame_length_input, ()))
        return tuple(inputs)


@dataclass(frozen=True)
class LoopRangeOp(RenderableOpBase):
    __io_inputs__ = ("trip_count", "cond", "start", "delta")
    __io_outputs__ = ("final", "output")
    trip_count: str
    cond: str
    start: str
    delta: str
    final: str
    output: str
    add_table_data: tuple[float | int, ...] | None = None
    add_table_shape: tuple[int, ...] | None = None

    def emit(self, emitter: "Emitter", ctx: "EmitContext") -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        dim_args = emitter.dim_args_str()
        output_dtype = emitter.op_output_dtype(self)
        c_type = output_dtype.c_type
        params = emitter.shared_param_map(
            [
                ("trip_count", self.trip_count),
                ("cond", self.cond),
                ("start", self.start),
                ("delta", self.delta),
                ("final", self.final),
                ("output", self.output),
            ]
        )
        scalar_suffix = emitter.param_array_suffix(())
        output_shape = emitter.ctx_shape(self.output)
        output_suffix = emitter.param_array_suffix(output_shape)
        count_dtype = emitter.ctx_dtype(self.trip_count).c_type
        cond_dtype = emitter.ctx_dtype(self.cond).c_type
        param_decls = emitter.build_param_decls(
            [
                (params["trip_count"], count_dtype, scalar_suffix, True),
                (params["cond"], cond_dtype, scalar_suffix, True),
                (params["start"], c_type, scalar_suffix, True),
                (params["delta"], c_type, scalar_suffix, True),
                (params["final"], c_type, scalar_suffix, False),
                (params["output"], c_type, output_suffix, False),
            ]
        )
        add_table_data = None
        state_size = 1
        if self.add_table_data is not None:
            add_table_data = [
                emitter.format_value(value, emitter.ctx_dtype(self.output))
                for value in self.add_table_data
            ]
            start_shape = emitter.ctx_shape(self.start)
            state_size = int(math.prod(start_shape)) if start_shape else 1
        rendered = (
            state.templates["loop_range"]
            .render(
                model_name=model.name,
                op_name=op_name,
                trip_count=params["trip_count"],
                cond=params["cond"],
                start=params["start"],
                delta=params["delta"],
                final=params["final"],
                output=params["output"],
                params=param_decls,
                c_type=c_type,
                output_suffix=output_suffix,
                dim_args=dim_args,
                add_table_data=add_table_data,
                state_size=state_size,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)

    def c_op_inputs(
        self, emitter: "Emitter"
    ) -> tuple[tuple[str, tuple[int, ...]], ...]:
        return (
            (self.trip_count, ()),
            (self.cond, ()),
            (self.start, ()),
            (self.delta, ()),
        )

    def c_op_outputs(
        self, emitter: "Emitter"
    ) -> tuple[tuple[str, tuple[int, ...], "ScalarType"], ...]:
        return (
            (self.final, emitter.ctx_shape(self.final), emitter.ctx_dtype(self.final)),
            (
                self.output,
                emitter.ctx_shape(self.output),
                emitter.ctx_dtype(self.output),
            ),
        )


@dataclass(frozen=True)
class LoopSequenceInsertOp(RenderableOpBase):
    __io_inputs__ = ("trip_count", "cond", "input_sequence")
    __io_outputs__ = ("output_sequence",)
    trip_count: str
    cond: str
    input_sequence: str
    output_sequence: str
    table_data: tuple[float | int, ...]
    table_shape: tuple[int, ...]
    elem_shape: tuple[int, ...]
    elem_dtype: ScalarType
    input_sequence_present: str | None = None
    # Name of the _Bool present flag passed to the C function. Set when
    # input_sequence is an Optional[Sequence]; None for plain Sequence inputs.
    default_sequence_data: tuple[float | int, ...] = ()
    # Initial sequence values used when input_sequence_present is False (absent).
    # Extracted from the then-branch SequenceConstruct in the loop body.
    prefix_slices: bool = False
    # When True, append x[:i+1] style slices from table_data instead of scalars.

    def call_args(self) -> tuple[str, ...]:
        args: list[str] = [
            self.trip_count,
            self.cond,
            self.input_sequence,
            f"{self.input_sequence}__count",
        ]
        if self.input_sequence_present is not None:
            args.append(self.input_sequence_present)
        args.extend(
            [
                self.output_sequence,
                f"{self.output_sequence}__count",
            ]
        )
        return tuple(args)

    def emit(self, emitter: "Emitter", ctx: "EmitContext") -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        dim_args = emitter.dim_args_str()
        emitter.op_output_dtype(self)
        params = emitter.shared_param_map(
            [
                ("trip_count", self.trip_count),
                ("cond", self.cond),
                ("input_sequence", self.input_sequence),
                ("input_sequence_present", self.input_sequence_present),
                ("output_sequence", self.output_sequence),
            ]
        )
        elem_shape = self.elem_shape
        elem_suffix = emitter.param_array_suffix(elem_shape)
        seq_dtype = self.elem_dtype
        scalar_suffix = emitter.param_array_suffix(())
        input_present_param = params.get("input_sequence_present")
        param_specs: list[tuple[str | None, str, str, bool]] = [
            (
                params["trip_count"],
                emitter.ctx_dtype(self.trip_count).c_type,
                scalar_suffix,
                True,
            ),
            (
                params["cond"],
                emitter.ctx_dtype(self.cond).c_type,
                scalar_suffix,
                True,
            ),
            (
                params["input_sequence"],
                seq_dtype.c_type,
                f"[EMX_SEQUENCE_MAX_LEN]{elem_suffix}",
                True,
            ),
            (f"{params['input_sequence']}__count", "idx_t", "", True),
        ]
        if input_present_param is not None:
            param_specs.append((input_present_param, "_Bool", "", True))
        output_dynamic_axes = emitter.sequence_dynamic_axes(self.output_sequence)
        param_specs.extend(
            [
                (
                    params["output_sequence"],
                    seq_dtype.c_type,
                    f"[EMX_SEQUENCE_MAX_LEN]{elem_suffix}",
                    False,
                ),
                (
                    f"{params['output_sequence']}__count",
                    "idx_t *",
                    "",
                    False,
                ),
                *[
                    (
                        emitter.sequence_dim_array_name(self.output_sequence, axis),
                        "idx_t",
                        "[EMX_SEQUENCE_MAX_LEN]",
                        False,
                    )
                    for axis in output_dynamic_axes
                ],
            ]
        )
        param_decls = emitter.build_param_decls(param_specs)
        table_data = [
            emitter.format_value(value, seq_dtype) for value in self.table_data
        ]
        default_data = [
            emitter.format_value(value, seq_dtype)
            for value in self.default_sequence_data
        ]
        rendered = (
            state.templates["loop_sequence_insert"]
            .render(
                op_name=op_name,
                dim_args=dim_args,
                params=param_decls,
                trip_count=params["trip_count"],
                cond=params["cond"],
                input_sequence=params["input_sequence"],
                input_present=input_present_param,
                output_sequence=params["output_sequence"],
                table_data=table_data,
                table_len=self.table_shape[0],
                element_count=CEmitterCompat.element_count_expr(elem_shape),
                c_type=seq_dtype.c_type,
                default_data=default_data,
                default_count=len(self.default_sequence_data),
                prefix_slices=self.prefix_slices,
                output_dim_arrays=tuple(
                    {
                        "axis": axis,
                        "name": emitter.sequence_dim_array_name(
                            self.output_sequence, axis
                        ),
                    }
                    for axis in output_dynamic_axes
                ),
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)

    def resolved_output_dtype(self, ctx: OpContext) -> ScalarType:
        return self.elem_dtype

    def computed_output_shape(self, emitter: "Emitter") -> tuple[int, ...]:
        return self.elem_shape

    def computed_output_dtype(self, emitter: "Emitter") -> "ScalarType":
        return self.elem_dtype

    def c_op_inputs(
        self, emitter: "Emitter"
    ) -> tuple[tuple[str, tuple[int, ...]], ...]:
        return ((self.trip_count, ()), (self.cond, ()))

    def c_op_outputs(
        self, emitter: "Emitter"
    ) -> tuple[tuple[str, tuple[int, ...], "ScalarType"], ...]:
        return ((self.output_sequence, self.elem_shape, self.elem_dtype),)


@dataclass(frozen=True)
class LoopSequenceMapOp(RenderableOpBase):
    __io_inputs__ = ("trip_count", "cond")
    __io_outputs__ = ("output_sequences",)
    __io_remap_extra__ = (
        "input_sequences",
        "input_tensors",
        "output_input0",
        "output_input1",
    )
    trip_count: str
    cond: str
    input_sequences: tuple[str, ...]
    input_tensors: tuple[str, ...]
    output_sequences: tuple[str, ...]
    output_kinds: tuple[str, ...]
    output_input0: tuple[str, ...]
    output_input1: tuple[str | None, ...]
    output_input0_is_sequence: tuple[bool, ...]
    output_input1_is_sequence: tuple[bool, ...]
    output_elem_shapes: tuple[tuple[int, ...], ...]
    output_elem_dtypes: tuple[ScalarType, ...]

    def call_args(self) -> tuple[str, ...]:
        args: list[str] = [self.trip_count, self.cond]
        for name in self.input_sequences:
            args.extend((name, f"{name}__count"))
        for name in self.input_tensors:
            args.append(name)
        for name in self.output_sequences:
            args.extend((name, f"{name}__count"))
        return tuple(args)

    def emit(self, emitter: "Emitter", ctx: "EmitContext") -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        dim_args = emitter.dim_args_str()
        emitter.op_output_dtype(self)
        params = emitter.shared_param_map(
            [
                ("trip_count", self.trip_count),
                ("cond", self.cond),
                *[
                    (f"input_sequence_{idx}", name)
                    for idx, name in enumerate(self.input_sequences)
                ],
                *[
                    (f"input_tensor_{idx}", name)
                    for idx, name in enumerate(self.input_tensors)
                ],
                *[
                    (f"output_sequence_{idx}", name)
                    for idx, name in enumerate(self.output_sequences)
                ],
            ]
        )
        scalar_suffix = emitter.param_array_suffix(())
        decls: list[tuple[str, str, str, bool]] = [
            (
                params["trip_count"],
                emitter.ctx_dtype(self.trip_count).c_type,
                scalar_suffix,
                True,
            ),
            (params["cond"], emitter.ctx_dtype(self.cond).c_type, scalar_suffix, True),
        ]
        seq_param_by_name = {
            name: params[f"input_sequence_{idx}"]
            for idx, name in enumerate(self.input_sequences)
        }
        tensor_param_by_name = {
            name: params[f"input_tensor_{idx}"]
            for idx, name in enumerate(self.input_tensors)
        }
        for idx, name in enumerate(self.input_sequences):
            seq_dtype = emitter.ctx_sequence_elem_type(name).dtype
            seq_shape = emitter.sequence_storage_shape(name)
            decls.append(
                (
                    params[f"input_sequence_{idx}"],
                    seq_dtype.c_type,
                    f"[EMX_SEQUENCE_MAX_LEN]{emitter.param_array_suffix(seq_shape)}",
                    True,
                )
            )
            decls.append(
                (f"{params[f'input_sequence_{idx}']}__count", "idx_t", "", True)
            )
            for axis in emitter.sequence_dynamic_axes(name):
                decls.append(
                    (
                        emitter.sequence_dim_array_name(name, axis),
                        "idx_t",
                        "[EMX_SEQUENCE_MAX_LEN]",
                        True,
                    )
                )
        for idx, name in enumerate(self.input_tensors):
            dtype = emitter.ctx_dtype(name)
            shape = emitter.ctx_shape(name)
            decls.append(
                (
                    params[f"input_tensor_{idx}"],
                    dtype.c_type,
                    emitter.param_array_suffix(shape),
                    True,
                )
            )
        for idx, name in enumerate(self.output_sequences):
            seq_dtype = emitter.ctx_sequence_elem_type(name).dtype
            seq_shape = emitter.sequence_storage_shape(name)
            decls.append(
                (
                    params[f"output_sequence_{idx}"],
                    seq_dtype.c_type,
                    f"[EMX_SEQUENCE_MAX_LEN]{emitter.param_array_suffix(seq_shape)}",
                    False,
                )
            )
            decls.append(
                (f"{params[f'output_sequence_{idx}']}__count", "idx_t *", "", False)
            )
            for axis in emitter.sequence_dynamic_axes(name):
                decls.append(
                    (
                        emitter.sequence_dim_array_name(name, axis),
                        "idx_t",
                        "[EMX_SEQUENCE_MAX_LEN]",
                        False,
                    )
                )
        param_decls = emitter.build_param_decls(decls)

        lines = [f"void {op_name}({dim_args}{', '.join(param_decls)}) {{"]
        lines.append(
            f"    const int64_t trip_raw = (int64_t){params['trip_count']}[0];"
        )
        lines.append(f"    const bool enabled = {params['cond']}[0];")
        lines.append(
            "    int64_t iter_limit = (enabled && trip_raw > 0) ? trip_raw : 0;"
        )
        for idx, name in enumerate(self.input_sequences):
            lines.append(
                f"    if (iter_limit > (int64_t){params[f'input_sequence_{idx}']}__count) iter_limit = (int64_t){params[f'input_sequence_{idx}']}__count;"
            )
        lines.append(
            "    if (iter_limit > (int64_t)EMX_SEQUENCE_MAX_LEN) iter_limit = (int64_t)EMX_SEQUENCE_MAX_LEN;"
        )
        lines.append("    for (int64_t i = 0; i < iter_limit; ++i) {")
        for out_idx, out_name in enumerate(self.output_sequences):
            out_param = params[f"output_sequence_{out_idx}"]
            kind = self.output_kinds[out_idx]
            in0 = self.output_input0[out_idx]
            in1 = self.output_input1[out_idx]
            in0_seq = self.output_input0_is_sequence[out_idx]
            in1_seq = self.output_input1_is_sequence[out_idx]
            output_elem_shape = emitter.sequence_storage_shape(out_name)
            elem_count = CEmitterCompat.element_count_expr(
                CEmitterCompat.shape_dim_exprs(output_elem_shape, {})
            )
            for axis in emitter.sequence_dynamic_axes(out_name):
                if kind == "shape":
                    if in0_seq:
                        lines.append(
                            f"        {emitter.sequence_dim_array_name(out_name, axis)}[(idx_t)i] = 1;"
                        )
                    else:
                        dim_ref = emitter.dim_names_for(in0).get(axis)
                        dim_expr = dim_ref.name if dim_ref is not None else emitter.ctx_shape(in0)[axis]
                        lines.append(
                            f"        {emitter.sequence_dim_array_name(out_name, axis)}[(idx_t)i] = {dim_expr};"
                        )
                elif in0_seq:
                    lines.append(
                        f"        {emitter.sequence_dim_array_name(out_name, axis)}[(idx_t)i] = "
                        f"{emitter.sequence_dim_array_name(in0, axis)}[(idx_t)i];"
                    )
                else:
                    dim_ref = emitter.dim_names_for(in0).get(axis)
                    dim_expr = dim_ref.name if dim_ref is not None else emitter.ctx_shape(in0)[axis]
                    lines.append(
                        f"        {emitter.sequence_dim_array_name(out_name, axis)}[(idx_t)i] = {dim_expr};"
                    )
            if kind == "shape":
                source_shape = (
                    CEmitterCompat.shape_dim_exprs(
                        emitter.sequence_storage_shape(in0),
                        {},
                    )
                    if in0_seq
                    else CEmitterCompat.shape_dim_exprs(
                        emitter.ctx_shape(in0),
                        emitter.dim_names_for(in0),
                    )
                )
                for dim_idx, dim in enumerate(source_shape):
                    if in0_seq and dim_idx in emitter.sequence_dynamic_axes(in0):
                        dim_expr = (
                            f"{emitter.sequence_dim_array_name(in0, dim_idx)}[(idx_t)i]"
                        )
                    else:
                        dim_expr = dim
                    lines.append(f"        {out_param}[(idx_t)i][{dim_idx}] = {dim_expr};")
            elif kind == "identity":
                src_name = (
                    seq_param_by_name[in0] if in0_seq else tensor_param_by_name[in0]
                )
                src = f"{src_name}[(idx_t)i]" if in0_seq else src_name
                lines.append(
                    f"        for (idx_t e = 0; e < {elem_count}; ++e) {{ {out_param}[(idx_t)i][e] = {src}[e]; }}"
                )
            elif kind == "add" and in1 is not None:
                src0_name = (
                    seq_param_by_name[in0] if in0_seq else tensor_param_by_name[in0]
                )
                src1_name = (
                    seq_param_by_name[in1] if in1_seq else tensor_param_by_name[in1]
                )
                src0 = f"{src0_name}[(idx_t)i]" if in0_seq else src0_name
                src1 = f"{src1_name}[(idx_t)i]" if in1_seq else src1_name
                lines.append(
                    f"        for (idx_t e = 0; e < {elem_count}; ++e) {{ {out_param}[(idx_t)i][e] = {src0}[e] + {src1}[e]; }}"
                )
        lines.append("    }")
        for out_idx, _ in enumerate(self.output_sequences):
            lines.append(
                f"    *{params[f'output_sequence_{out_idx}']}__count = (idx_t)iter_limit;"
            )
        lines.append("}")
        return emitter.with_node_comment(model, ctx.op_index, "\n".join(lines))

    def resolved_output_dtype(self, ctx: OpContext) -> ScalarType:
        return self.output_elem_dtypes[0]

    def computed_output_shape(self, emitter: "Emitter") -> tuple[int, ...]:
        return self.output_elem_shapes[0]

    def computed_output_dtype(self, emitter: "Emitter") -> "ScalarType":
        return self.output_elem_dtypes[0]

    def c_op_inputs(
        self, emitter: "Emitter"
    ) -> tuple[tuple[str, tuple[int, ...]], ...]:
        return ((self.trip_count, ()), (self.cond, ()))

    def c_op_outputs(
        self, emitter: "Emitter"
    ) -> tuple[tuple[str, tuple[int, ...], "ScalarType"], ...]:
        return tuple(
            (
                output_name,
                emitter.sequence_storage_shape(output_name),
                output_dtype,
            )
            for output_name, output_dtype in zip(
                self.output_sequences, self.output_elem_dtypes
            )
        )


@dataclass(frozen=True)
class RangeOp(RenderableOpBase):
    __io_inputs__ = ("start", "limit", "delta")
    __io_outputs__ = ("output",)
    start: str
    limit: str
    delta: str
    output: str

    def emit(self, emitter: "Emitter", ctx: "EmitContext") -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        emitter.dim_args_str()
        output_dtype = emitter.op_output_dtype(self)
        c_type = output_dtype.c_type
        params = emitter.shared_param_map(
            [
                ("start", self.start),
                ("limit", self.limit),
                ("delta", self.delta),
                ("output", self.output),
            ]
        )
        scalar_suffix = emitter.param_array_suffix(())
        output_shape = emitter.ctx_shape(self.output)
        output_suffix = emitter.param_array_suffix(output_shape)
        param_decls = emitter.build_param_decls(
            [
                (params["start"], c_type, scalar_suffix, True),
                (params["limit"], c_type, scalar_suffix, True),
                (params["delta"], c_type, scalar_suffix, True),
                (params["output"], c_type, output_suffix, False),
            ]
        )
        rendered = (
            state.templates["range"]
            .render(
                model_name=model.name,
                op_name=op_name,
                start=params["start"],
                limit=params["limit"],
                delta=params["delta"],
                output=params["output"],
                params=param_decls,
                c_type=c_type,
                input_suffix=scalar_suffix,
                output_suffix=output_suffix,
                length=output_shape[0],
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)

    def c_op_inputs(
        self, emitter: "Emitter"
    ) -> tuple[tuple[str, tuple[int, ...]], ...]:
        return ((self.start, ()), (self.limit, ()), (self.delta, ()))


@dataclass(frozen=True)
class DFTOp(RenderableOpBase):
    __io_inputs__ = ("input0", "axis_input")
    __io_outputs__ = ("output",)
    input0: str
    axis_input: str | None
    output: str
    axis: int | None
    dft_length: int | None
    axis_variants: tuple[int, ...]
    dft_lengths: tuple[int, ...]
    inverse: bool
    onesided: bool
    input_is_complex: bool
    fft_mode: str = "stockham_auto"

    def emit(self, emitter: "Emitter", ctx: "EmitContext") -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        dim_args = emitter.dim_args_str()
        emitter.op_output_dtype(self)
        input_shape = emitter.ctx_shape(self.input0)
        output_shape = emitter.ctx_shape(self.output)
        dft_dtype = emitter.ctx_dtype(self.output)
        if dft_dtype != emitter.ctx_dtype(self.input0):
            raise CodegenError("DFT input and output dtypes must match")
        if not self.axis_variants or len(self.axis_variants) != len(self.dft_lengths):
            raise CodegenError("DFT requires at least one valid axis variant")

        rank = len(input_shape)
        params = emitter.shared_param_map(
            [
                ("input0", self.input0),
                ("axis_input", self.axis_input),
                ("output", self.output),
            ]
        )
        input_dim_names = emitter.dim_names_for(self.input0)
        output_dim_names = emitter.dim_names_for(self.output)
        input_shape_expr = CEmitterCompat.shape_dim_exprs(input_shape, input_dim_names)
        output_shape_expr = CEmitterCompat.shape_dim_exprs(
            output_shape, output_dim_names
        )
        axis_c_type = "int64_t"
        if self.axis_input is not None:
            axis_dtype = emitter.ctx_dtype(self.axis_input)
            if axis_dtype not in {ScalarType.I32, ScalarType.I64}:
                raise CodegenError("DFT axis input must be int32 or int64")
            axis_c_type = axis_dtype.c_type

        input_suffix = emitter.param_array_suffix(
            input_shape,
            input_dim_names,
            dtype=dft_dtype,
        )
        output_suffix = emitter.param_array_suffix(
            output_shape,
            output_dim_names,
            dtype=dft_dtype,
        )
        param_decls = emitter.build_param_decls(
            [
                (params["input0"], dft_dtype.c_type, input_suffix, True),
                (
                    params["axis_input"],
                    axis_c_type,
                    emitter.param_array_suffix(()),
                    True,
                ),
                (params["output"], dft_dtype.c_type, output_suffix, False),
            ]
        )
        axis_variants: list[dict[str, object]] = []
        signal_rank = rank - 1
        for axis, fft_length in zip(self.axis_variants, self.dft_lengths):
            if axis < 0 or axis >= signal_rank:
                raise CodegenError("DFT axis must target a signal dimension")
            if fft_length <= 0:
                raise CodegenError("DFT length must be > 0")
            stage_plan = emitter.dft_stockham_stage_plan(fft_length)
            use_fft = bool(stage_plan) and fft_length > 1
            twiddles = emitter.dft_twiddle_table(
                fft_length,
                inverse=self.inverse,
                dtype=dft_dtype,
            )
            axis_variants.append(
                {
                    "axis": axis,
                    "outer_expr": CEmitterCompat.element_count_expr(
                        input_shape_expr[:axis]
                    ),
                    "inner_expr": CEmitterCompat.element_count_expr(
                        input_shape_expr[axis + 1 : signal_rank]
                    ),
                    "input_axis_expr": str(input_shape_expr[axis]),
                    "output_axis_expr": str(output_shape_expr[axis]),
                    "fft_length": fft_length,
                    "use_fft": use_fft,
                    "stage_data": [
                        {"kind": kind, "m": m, "l": stage_span}
                        for kind, m, stage_span in stage_plan
                    ],
                    "twiddle_re_values": [real for real, _ in twiddles],
                    "twiddle_im_values": [imag for _, imag in twiddles],
                    "twiddle_len": fft_length,
                    "inverse_scale": emitter.format_literal(
                        dft_dtype, 1.0 / float(fft_length)
                    ),
                }
            )

        if self.axis_input is None and len(axis_variants) != 1:
            raise CodegenError(
                "DFT with constant axis must lower to exactly one axis variant"
            )

        rendered = (
            state.templates["dft"]
            .render(
                model_name=model.name,
                op_name=op_name,
                params=param_decls,
                input0=params["input0"],
                axis_input=params["axis_input"],
                axis_c_type=axis_c_type,
                output=params["output"],
                c_type=dft_dtype.c_type,
                input_suffix=input_suffix,
                output_suffix=output_suffix,
                rank=rank,
                axis_variants=axis_variants,
                input_components=2 if self.input_is_complex else 1,
                inverse=self.inverse,
                onesided=self.onesided,
                zero_literal=dft_dtype.zero_literal,
                dim_args=dim_args,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)

    def c_op_inputs(
        self, emitter: "Emitter"
    ) -> tuple[tuple[str, tuple[int, ...]], ...]:
        return ((self.input0, emitter.ctx_shape(self.input0)),)


@dataclass(frozen=True)
class HammingWindowOp(RenderableOpBase):
    __io_inputs__ = ("size",)
    __io_outputs__ = ("output",)
    size: str
    output: str
    periodic: bool

    def required_includes(self, ctx: OpContext) -> set[str]:
        return {"#include <math.h>"}

    def emit(self, emitter: "Emitter", ctx: "EmitContext") -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        emitter.dim_args_str()
        output_dtype = emitter.op_output_dtype(self)
        c_type = output_dtype.c_type
        params = emitter.shared_param_map(
            [
                ("size", self.size),
                ("output", self.output),
            ]
        )
        scalar_suffix = emitter.param_array_suffix(())
        output_shape = emitter.ctx_shape(self.output)
        output_suffix = emitter.param_array_suffix(output_shape)
        param_decls = emitter.build_param_decls(
            [
                (
                    params["size"],
                    emitter.ctx_dtype(self.size).c_type,
                    scalar_suffix,
                    True,
                ),
                (params["output"], c_type, output_suffix, False),
            ]
        )
        output_dtype = emitter.ctx_dtype(self.output)
        compute_dtype = (
            ScalarType.F64 if output_dtype == ScalarType.F64 else ScalarType.F32
        )
        compute_type = "double" if compute_dtype == ScalarType.F64 else "float"

        rendered = (
            state.templates["hamming_window"]
            .render(
                model_name=model.name,
                op_name=op_name,
                size=params["size"],
                output=params["output"],
                params=param_decls,
                c_type=c_type,
                output_suffix=output_suffix,
                length=output_shape[0],
                periodic_literal="1" if self.periodic else "0",
                compute_type=compute_type,
                one_literal=emitter.format_literal(compute_dtype, 1.0),
                two_literal=emitter.format_literal(compute_dtype, 2.0),
                alpha_literal=emitter.format_literal(compute_dtype, 25.0 / 46.0),
                beta_literal=emitter.format_literal(compute_dtype, 1.0 - (25.0 / 46.0)),
                pi_literal=emitter.format_literal(compute_dtype, math.pi),
                compute_scalar_dtype=compute_dtype,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)

    def c_op_inputs(
        self, emitter: "Emitter"
    ) -> tuple[tuple[str, tuple[int, ...]], ...]:
        return ((self.size, ()),)


@dataclass(frozen=True)
class BlackmanWindowOp(RenderableOpBase):
    __io_inputs__ = ("size",)
    __io_outputs__ = ("output",)
    size: str
    output: str
    periodic: bool

    def required_includes(self, ctx: OpContext) -> set[str]:
        return {"#include <math.h>"}

    def emit(self, emitter: "Emitter", ctx: "EmitContext") -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        emitter.dim_args_str()
        output_dtype = emitter.op_output_dtype(self)
        c_type = output_dtype.c_type
        params = emitter.shared_param_map(
            [
                ("size", self.size),
                ("output", self.output),
            ]
        )
        scalar_suffix = emitter.param_array_suffix(())
        output_shape = emitter.ctx_shape(self.output)
        output_suffix = emitter.param_array_suffix(output_shape)
        param_decls = emitter.build_param_decls(
            [
                (
                    params["size"],
                    emitter.ctx_dtype(self.size).c_type,
                    scalar_suffix,
                    True,
                ),
                (params["output"], c_type, output_suffix, False),
            ]
        )
        output_dtype = emitter.ctx_dtype(self.output)
        compute_dtype = (
            ScalarType.F64 if output_dtype == ScalarType.F64 else ScalarType.F32
        )
        compute_type = "double" if compute_dtype == ScalarType.F64 else "float"

        rendered = (
            state.templates["blackman_window"]
            .render(
                model_name=model.name,
                op_name=op_name,
                size=params["size"],
                output=params["output"],
                params=param_decls,
                c_type=c_type,
                output_suffix=output_suffix,
                length=output_shape[0],
                periodic_literal="1" if self.periodic else "0",
                compute_type=compute_type,
                one_literal=emitter.format_literal(compute_dtype, 1.0),
                two_literal=emitter.format_literal(compute_dtype, 2.0),
                c042_literal=emitter.format_literal(compute_dtype, 0.42),
                c05_literal=emitter.format_literal(compute_dtype, 0.5),
                c008_literal=emitter.format_literal(compute_dtype, 0.08),
                pi_literal=emitter.format_literal(compute_dtype, math.pi),
                compute_scalar_dtype=compute_dtype,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)

    def c_op_inputs(
        self, emitter: "Emitter"
    ) -> tuple[tuple[str, tuple[int, ...]], ...]:
        return ((self.size, ()),)


@dataclass(frozen=True)
class HannWindowOp(RenderableOpBase):
    __io_inputs__ = ("size",)
    __io_outputs__ = ("output",)
    size: str
    output: str
    periodic: bool

    def required_includes(self, ctx: OpContext) -> set[str]:
        return {"#include <math.h>"}

    def emit(self, emitter: "Emitter", ctx: "EmitContext") -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        emitter.dim_args_str()
        output_dtype = emitter.op_output_dtype(self)
        c_type = output_dtype.c_type
        params = emitter.shared_param_map(
            [
                ("size", self.size),
                ("output", self.output),
            ]
        )
        scalar_suffix = emitter.param_array_suffix(())
        output_shape = emitter.ctx_shape(self.output)
        output_suffix = emitter.param_array_suffix(output_shape)
        param_decls = emitter.build_param_decls(
            [
                (
                    params["size"],
                    emitter.ctx_dtype(self.size).c_type,
                    scalar_suffix,
                    True,
                ),
                (params["output"], c_type, output_suffix, False),
            ]
        )
        output_dtype = emitter.ctx_dtype(self.output)
        compute_dtype = (
            ScalarType.F64 if output_dtype == ScalarType.F64 else ScalarType.F32
        )
        compute_type = "double" if compute_dtype == ScalarType.F64 else "float"

        rendered = (
            state.templates["hann_window"]
            .render(
                model_name=model.name,
                op_name=op_name,
                size=params["size"],
                output=params["output"],
                params=param_decls,
                c_type=c_type,
                output_suffix=output_suffix,
                length=output_shape[0],
                periodic_literal="1" if self.periodic else "0",
                compute_type=compute_type,
                one_literal=emitter.format_literal(compute_dtype, 1.0),
                two_literal=emitter.format_literal(compute_dtype, 2.0),
                alpha_literal=emitter.format_literal(compute_dtype, 0.5),
                beta_literal=emitter.format_literal(compute_dtype, 0.5),
                pi_literal=emitter.format_literal(compute_dtype, math.pi),
                compute_scalar_dtype=compute_dtype,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)

    def c_op_inputs(
        self, emitter: "Emitter"
    ) -> tuple[tuple[str, tuple[int, ...]], ...]:
        return ((self.size, ()),)


@dataclass(frozen=True)
class MelWeightMatrixOp(RenderableOpBase):
    __io_inputs__ = (
        "num_mel_bins",
        "dft_length",
        "sample_rate",
        "lower_edge_hertz",
        "upper_edge_hertz",
    )
    __io_outputs__ = ("output",)
    num_mel_bins: str
    dft_length: str
    sample_rate: str
    lower_edge_hertz: str
    upper_edge_hertz: str
    output: str
    values: tuple[float, ...]

    def required_includes(self, ctx: OpContext) -> set[str]:
        return {"#include <math.h>"}

    def emit(self, emitter: "Emitter", ctx: "EmitContext") -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        emitter.dim_args_str()
        output_dtype = emitter.op_output_dtype(self)
        c_type = output_dtype.c_type
        params = emitter.shared_param_map(
            [
                ("num_mel_bins", self.num_mel_bins),
                ("dft_length", self.dft_length),
                ("sample_rate", self.sample_rate),
                ("lower_edge_hertz", self.lower_edge_hertz),
                ("upper_edge_hertz", self.upper_edge_hertz),
                ("output", self.output),
            ]
        )
        scalar_suffix = emitter.param_array_suffix(())
        output_shape = emitter.ctx_shape(self.output)
        output_suffix = emitter.param_array_suffix(output_shape)
        param_decls = emitter.build_param_decls(
            [
                (
                    params["num_mel_bins"],
                    emitter.ctx_dtype(self.num_mel_bins).c_type,
                    scalar_suffix,
                    True,
                ),
                (
                    params["dft_length"],
                    emitter.ctx_dtype(self.dft_length).c_type,
                    scalar_suffix,
                    True,
                ),
                (
                    params["sample_rate"],
                    emitter.ctx_dtype(self.sample_rate).c_type,
                    scalar_suffix,
                    True,
                ),
                (
                    params["lower_edge_hertz"],
                    emitter.ctx_dtype(self.lower_edge_hertz).c_type,
                    scalar_suffix,
                    True,
                ),
                (
                    params["upper_edge_hertz"],
                    emitter.ctx_dtype(self.upper_edge_hertz).c_type,
                    scalar_suffix,
                    True,
                ),
                (params["output"], c_type, output_suffix, False),
            ]
        )
        output_dtype = emitter.ctx_dtype(self.output)
        compute_dtype = (
            ScalarType.F64 if output_dtype == ScalarType.F64 else ScalarType.F32
        )
        compute_type = "double" if compute_dtype == ScalarType.F64 else "float"
        rendered = (
            state.templates["mel_weight_matrix"]
            .render(
                model_name=model.name,
                op_name=op_name,
                num_mel_bins=params["num_mel_bins"],
                dft_length=params["dft_length"],
                sample_rate=params["sample_rate"],
                lower_edge_hertz=params["lower_edge_hertz"],
                upper_edge_hertz=params["upper_edge_hertz"],
                output=params["output"],
                params=param_decls,
                c_type=c_type,
                output_suffix=output_suffix,
                num_spectrogram_bins=output_shape[0],
                num_mel_bins_dim=output_shape[1],
                compute_type=compute_type,
                seven_hundred_literal=emitter.format_literal(compute_dtype, 700.0),
                mel_scale_literal=emitter.format_literal(compute_dtype, 2595.0),
                ten_literal=emitter.format_literal(compute_dtype, 10.0),
                one_literal=emitter.format_literal(compute_dtype, 1.0),
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)


@dataclass(frozen=True)
class OneHotOp(RenderableOpBase):
    __io_inputs__ = ("indices", "depth", "values")
    __io_outputs__ = ("output",)
    indices: str
    depth: str
    values: str
    output: str
    axis: int

    def emit(self, emitter: "Emitter", ctx: "EmitContext") -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        emitter.dim_args_str()
        output_dtype = emitter.op_output_dtype(self)
        c_type = output_dtype.c_type
        params = emitter.shared_param_map(
            [
                ("indices", self.indices),
                ("depth", self.depth),
                ("values", self.values),
                ("output", self.output),
            ]
        )
        output_dim_names = emitter.dim_names_for(self.output)
        indices_dim_names = emitter.dim_names_for(self.indices)
        values_dim_names = emitter.dim_names_for(self.values)
        output_shape_raw = emitter.ctx_shape(self.output)
        indices_shape_raw = emitter.ctx_shape(self.indices)
        values_shape_raw = emitter.ctx_shape(self.values)
        depth_dtype = emitter.ctx_dtype(self.depth)
        output_shape = CEmitterCompat.codegen_shape(output_shape_raw)
        loop_vars = CEmitterCompat.loop_vars(output_shape)
        indices_indices = tuple(
            var for idx, var in enumerate(loop_vars) if idx != self.axis
        )
        if not indices_indices:
            indices_indices = ("0",)
        output_suffix = emitter.param_array_suffix(output_shape_raw, output_dim_names)
        indices_suffix = emitter.param_array_suffix(
            indices_shape_raw, indices_dim_names
        )
        values_suffix = emitter.param_array_suffix(values_shape_raw, values_dim_names)
        depth_suffix = emitter.param_array_suffix(())
        param_decls = emitter.build_param_decls(
            [
                (
                    params["indices"],
                    emitter.ctx_dtype(self.indices).c_type,
                    indices_suffix,
                    True,
                ),
                (
                    params["depth"],
                    depth_dtype.c_type,
                    depth_suffix,
                    True,
                ),
                (params["values"], c_type, values_suffix, True),
                (params["output"], c_type, output_suffix, False),
            ]
        )
        rendered = (
            state.templates["one_hot"]
            .render(
                model_name=model.name,
                op_name=op_name,
                indices=params["indices"],
                depth=params["depth"],
                values=params["values"],
                output=params["output"],
                params=param_decls,
                indices_suffix=indices_suffix,
                depth_suffix=depth_suffix,
                values_suffix=values_suffix,
                output_suffix=output_suffix,
                output_shape=output_shape,
                loop_vars=loop_vars,
                indices_indices=indices_indices,
                axis_index=loop_vars[self.axis],
                depth_dim=output_shape_raw[self.axis],
                indices_c_type=emitter.ctx_dtype(self.indices).c_type,
                c_type=c_type,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)

    def c_op_inputs(
        self, emitter: "Emitter"
    ) -> tuple[tuple[str, tuple[int, ...]], ...]:
        return (
            (self.indices, emitter.ctx_shape(self.indices)),
            (self.depth, ()),
            (self.values, emitter.ctx_shape(self.values)),
        )


@dataclass(frozen=True)
class TfIdfVectorizerOp(RenderableOpBase):
    __io_inputs__ = ("input0",)
    __io_outputs__ = ("output",)
    input0: str
    output: str
    min_gram_length: int
    max_gram_length: int
    max_skip_count: int
    mode: str
    ngram_counts: tuple[int, ...]
    ngram_indexes: tuple[int, ...]
    pool_int64s: tuple[int, ...]
    weights: tuple[float, ...] | None

    def emit(self, emitter: "Emitter", ctx: "EmitContext") -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        emitter.dim_args_str()
        output_dtype = emitter.op_output_dtype(self)
        params = emitter.shared_param_map(
            [("input0", self.input0), ("output", self.output)]
        )
        input_dim_names = emitter.dim_names_for(self.input0)
        output_dim_names = emitter.dim_names_for(self.output)
        input_shape = emitter.ctx_shape(self.input0)
        output_shape = emitter.ctx_shape(self.output)
        input_dtype = emitter.ctx_dtype(self.input0)
        output_dtype = emitter.ctx_dtype(self.output)
        input_suffix = emitter.param_array_suffix(input_shape, input_dim_names)
        output_suffix = emitter.param_array_suffix(output_shape, output_dim_names)
        param_decls = emitter.build_param_decls(
            [
                (
                    params["input0"],
                    input_dtype.c_type,
                    input_suffix,
                    True,
                ),
                (
                    params["output"],
                    output_dtype.c_type,
                    output_suffix,
                    False,
                ),
            ]
        )
        output_dim = output_shape[-1] if output_shape else 0
        mode_id = {"TF": 0, "IDF": 1, "TFIDF": 2}[self.mode]
        pool_values = [
            emitter.format_literal(ScalarType.I64, value) for value in self.pool_int64s
        ]
        ngram_counts_values = [
            emitter.format_literal(ScalarType.I64, value) for value in self.ngram_counts
        ]
        ngram_indexes_values = [
            emitter.format_literal(ScalarType.I64, value)
            for value in self.ngram_indexes
        ]
        weights_values = (
            [emitter.format_literal(output_dtype, value) for value in self.weights]
            if self.weights is not None
            else None
        )
        rendered = (
            state.templates["tfidf_vectorizer"]
            .render(
                model_name=model.name,
                op_name=op_name,
                input0=params["input0"],
                output=params["output"],
                params=param_decls,
                input_suffix=input_suffix,
                output_suffix=output_suffix,
                input_shape=input_shape,
                output_shape=output_shape,
                input_rank=len(input_shape),
                output_dim=output_dim,
                min_gram_length=self.min_gram_length,
                max_gram_length=self.max_gram_length,
                max_skip_count=self.max_skip_count,
                mode_id=mode_id,
                ngram_counts_len=len(self.ngram_counts),
                pool_size=len(self.pool_int64s),
                ngram_index_len=len(self.ngram_indexes),
                pool_values=pool_values,
                ngram_counts_values=ngram_counts_values,
                ngram_indexes_values=ngram_indexes_values,
                weights_values=weights_values,
                zero_literal=output_dtype.zero_literal,
                one_literal=emitter.format_literal(output_dtype, 1.0),
                c_type=output_dtype.c_type,
                input_c_type=input_dtype.c_type,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)


@dataclass(frozen=True)
class StringConcatOp(RenderableOpBase):
    __io_inputs__ = ("input0", "input1")
    __io_outputs__ = ("output",)
    input0: str
    input1: str
    output: str

    def required_includes(self, ctx: OpContext) -> set[str]:
        return {"#include <string.h>"}

    def emit(self, emitter: "Emitter", ctx: "EmitContext") -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        dim_args = emitter.dim_args_str()
        input0_shape = emitter.ctx_shape(self.input0)
        input1_shape = emitter.ctx_shape(self.input1)
        output_shape = emitter.ctx_shape(self.output)
        params = emitter.shared_param_map(
            [
                ("input0", self.input0),
                ("input1", self.input1),
                ("output", self.output),
            ]
        )
        input0_suffix = emitter.param_array_suffix(
            input0_shape, emitter.dim_names_for(self.input0), dtype=ScalarType.STRING
        )
        input1_suffix = emitter.param_array_suffix(
            input1_shape, emitter.dim_names_for(self.input1), dtype=ScalarType.STRING
        )
        output_suffix = emitter.param_array_suffix(
            output_shape, emitter.dim_names_for(self.output), dtype=ScalarType.STRING
        )
        param_decls = emitter.build_param_decls(
            [
                (params["input0"], "char", input0_suffix, True),
                (params["input1"], "char", input1_suffix, True),
                (params["output"], "char", output_suffix, False),
            ]
        )
        output_dim_names = emitter.dim_names_for(self.output)
        shape = CEmitterCompat.shape_dim_exprs(output_shape, output_dim_names)
        loop_vars = CEmitterCompat.loop_vars(output_shape)
        input0_expr = CEmitterCompat.broadcast_index_expr(
            params["input0"], input0_shape, output_shape, loop_vars
        )
        input1_expr = CEmitterCompat.broadcast_index_expr(
            params["input1"], input1_shape, output_shape, loop_vars
        )
        output_expr = CEmitterCompat.broadcast_index_expr(
            params["output"], output_shape, output_shape, loop_vars
        )
        rendered = (
            state.templates["string_concat"]
            .render(
                op_name=op_name,
                dim_args=dim_args,
                params=param_decls,
                shape=shape,
                loop_vars=loop_vars,
                input0_expr=input0_expr,
                input1_expr=input1_expr,
                output_expr=output_expr,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)

    def computed_output_dtype(self, emitter: "Emitter") -> "ScalarType":
        return ScalarType.STRING


@dataclass(frozen=True)
class StringNormalizerOp(RenderableOpBase):
    __io_inputs__ = ("input0",)
    __io_outputs__ = ("output",)
    input0: str
    output: str
    case_change_action: str
    is_case_sensitive: bool
    stopwords: tuple[str, ...]

    def required_includes(self, ctx: OpContext) -> set[str]:
        return {
            "#include <string.h>",
            "#include <strings.h>",
            "#include <ctype.h>",
            "#include <stdbool.h>",
        }

    def emit(self, emitter: "Emitter", ctx: "EmitContext") -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        dim_args = emitter.dim_args_str()
        params = emitter.shared_param_map(
            [("input0", self.input0), ("output", self.output)]
        )
        input_shape = emitter.ctx_shape(self.input0)
        output_shape = emitter.ctx_shape(self.output)
        input_suffix = emitter.param_array_suffix(
            input_shape, emitter.dim_names_for(self.input0), dtype=ScalarType.STRING
        )
        output_suffix = emitter.param_array_suffix(
            output_shape, emitter.dim_names_for(self.output), dtype=ScalarType.STRING
        )
        param_decls = emitter.build_param_decls(
            [
                (params["input0"], "char", input_suffix, True),
                (params["output"], "char", output_suffix, False),
            ]
        )
        compare_fn = "strcmp" if self.is_case_sensitive else "strcasecmp"
        stopword_checks = tuple(
            f"if ({compare_fn}(src, {emitter.format_c_string_literal(stopword)}) == 0) keep = false;"
            for stopword in self.stopwords
        )
        case_mode = {"NONE": 0, "LOWER": 1, "UPPER": 2}[self.case_change_action]
        rendered = (
            state.templates["string_normalizer"]
            .render(
                op_name=op_name,
                dim_args=dim_args,
                params=param_decls,
                input0=params["input0"],
                output=params["output"],
                input_count=CEmitterCompat.element_count_expr(input_shape),
                output_count=CEmitterCompat.element_count_expr(output_shape),
                stopword_checks=stopword_checks,
                case_mode=case_mode,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)

    def computed_output_dtype(self, emitter: "Emitter") -> "ScalarType":
        return ScalarType.STRING


@dataclass(frozen=True)
class LabelEncoderOp(RenderableOpBase):
    __io_inputs__ = ("input0",)
    __io_outputs__ = ("output",)
    input0: str
    output: str
    keys_strings: tuple[str, ...]
    keys_int64s: tuple[int, ...]
    keys_floats: tuple[float, ...]
    values_strings: tuple[str, ...]
    values_int64s: tuple[int, ...]
    values_floats: tuple[float, ...]
    default_string: str
    default_int64: int
    default_float: float

    def required_includes(self, ctx: OpContext) -> set[str]:
        if self.keys_strings:
            return {"#include <string.h>"}
        return set()

    def infer_types(self, ctx: OpContext) -> None:
        ctx.dtype(self.input0)
        ctx.dtype(self.output)

    def infer_shapes(self, ctx: OpContext) -> None:
        shape = ctx.shape(self.input0)
        ctx.set_shape(self.output, shape)

    def emit(self, emitter: "Emitter", ctx: "EmitContext") -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        dim_args = emitter.dim_args_str()
        output_dtype = emitter.op_output_dtype(self)
        params = emitter.shared_param_map(
            [("input0", self.input0), ("output", self.output)]
        )
        input_dtype = emitter.ctx_dtype(self.input0)
        output_dtype = emitter.ctx_dtype(self.output)
        output_shape_raw = emitter.ctx_shape(self.output)
        output_dim_names = emitter.dim_names_for(self.output)
        shape = CEmitterCompat.shape_dim_exprs(output_shape_raw, output_dim_names)
        loop_vars = CEmitterCompat.loop_vars(output_shape_raw)
        if input_dtype == ScalarType.STRING:
            input_suffix = emitter.param_array_suffix(
                output_shape_raw, output_dim_names, dtype=ScalarType.STRING
            )
        else:
            input_suffix = emitter.param_array_suffix(
                output_shape_raw, output_dim_names
            )
        output_suffix = emitter.param_array_suffix(output_shape_raw, output_dim_names)
        input_c_type = input_dtype.c_type
        output_c_type = output_dtype.c_type
        if input_dtype == ScalarType.STRING:
            input_c_type = "char"
        param_decls = emitter.build_param_decls(
            [
                (params["input0"], input_c_type, input_suffix, True),
                (params["output"], output_c_type, output_suffix, False),
            ]
        )
        if self.keys_strings:
            key_type = "string"
            key_c_type = "char"
            keys = tuple(emitter.format_c_string_literal(k) for k in self.keys_strings)
        elif self.keys_int64s:
            key_type = "int64"
            key_c_type = input_dtype.c_type
            keys = tuple(str(k) for k in self.keys_int64s)
        else:
            key_type = "float"
            key_c_type = input_dtype.c_type
            keys = tuple(
                emitter.format_literal(input_dtype, k) for k in self.keys_floats
            )
        if self.values_int64s:
            values = tuple(str(v) for v in self.values_int64s)
            default_value = str(self.default_int64)
        elif self.values_floats:
            values = tuple(
                emitter.format_literal(output_dtype, v) for v in self.values_floats
            )
            default_value = emitter.format_literal(output_dtype, self.default_float)
        else:
            values = tuple(
                emitter.format_c_string_literal(v) for v in self.values_strings
            )
            default_value = emitter.format_c_string_literal(self.default_string)
        rendered = (
            state.templates["label_encoder"]
            .render(
                op_name=op_name,
                dim_args=dim_args,
                params=param_decls,
                input0=params["input0"],
                output=params["output"],
                shape=shape,
                loop_vars=loop_vars,
                key_type=key_type,
                key_c_type=key_c_type,
                output_c_type=output_c_type,
                keys=keys,
                values=values,
                default_value=default_value,
                num_entries=len(keys),
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)


@dataclass(frozen=True)
class StringSplitOp(RenderableOpBase):
    __io_inputs__ = ("input0",)
    __io_outputs__ = ("output_y", "output_z")
    input0: str
    output_y: str
    output_z: str
    delimiter: str
    maxsplit: int

    def required_includes(self, ctx: OpContext) -> set[str]:
        return {"#include <string.h>"}

    def emit(self, emitter: "Emitter", ctx: "EmitContext") -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        dim_args = emitter.dim_args_str()
        params = emitter.shared_param_map(
            [
                ("input0", self.input0),
                ("output_y", self.output_y),
                ("output_z", self.output_z),
            ]
        )
        input_shape = emitter.ctx_shape(self.input0)
        output_y_shape = emitter.ctx_shape(self.output_y)
        output_z_shape = emitter.ctx_shape(self.output_z)
        input_suffix = emitter.param_array_suffix(
            input_shape, emitter.dim_names_for(self.input0), dtype=ScalarType.STRING
        )
        output_y_suffix = emitter.param_array_suffix(
            output_y_shape,
            emitter.dim_names_for(self.output_y),
            dtype=ScalarType.STRING,
        )
        output_z_suffix = emitter.param_array_suffix(
            output_z_shape, emitter.dim_names_for(self.output_z)
        )
        param_decls = emitter.build_param_decls(
            [
                (params["input0"], "char", input_suffix, True),
                (params["output_y"], "char", output_y_suffix, False),
                (params["output_z"], "int64_t", output_z_suffix, False),
            ]
        )
        is_whitespace = not self.delimiter
        max_count = output_y_shape[-1] if output_y_shape else 0
        rendered = (
            state.templates["string_split"]
            .render(
                op_name=op_name,
                dim_args=dim_args,
                params=param_decls,
                input0=params["input0"],
                output_y=params["output_y"],
                output_z=params["output_z"],
                input_count=CEmitterCompat.element_count_expr(input_shape),
                max_count=max_count,
                is_whitespace=is_whitespace,
                maxsplit=self.maxsplit,
                delimiter_literal=emitter.format_c_string_literal(self.delimiter),
                delimiter_len=len(self.delimiter),
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)

    def resolved_output_dtype(self, ctx: OpContext) -> ScalarType:
        return ctx.dtype(self.output_z)

    def computed_output_dtype(self, emitter: "Emitter") -> "ScalarType":
        return ScalarType.STRING

    def c_op_outputs(
        self, emitter: "Emitter"
    ) -> tuple[tuple[str, tuple[int, ...], "ScalarType"], ...]:
        return (
            (self.output_y, emitter.ctx_shape(self.output_y), ScalarType.STRING),
            (
                self.output_z,
                emitter.ctx_shape(self.output_z),
                emitter.ctx_dtype(self.output_z),
            ),
        )


def _normalize_regex_fullmatch_pattern(pattern: str) -> str:
    """Strip word-boundary assertions that are redundant under fullmatch.

    Rewrites ``\\b`` adjacent to literal word characters where the boundary
    is implied by the surrounding context (e.g. ``\\b`` at pattern start/end
    next to a word character, or between a punctuation literal and a word
    character).  Used as a fallback: the original pattern is tried first;
    the normalised form is only attempted if the original is rejected.
    """

    normalized = pattern
    normalized = re.sub(r"\\([^\wds])\\b(?=[A-Za-z0-9_])", r"\\\1", normalized)
    normalized = re.sub(r"(?<=[A-Za-z0-9_])\\b\\([^\wds])", r"\\\1", normalized)
    normalized = re.sub(r"^\\b(?=[A-Za-z0-9_])", "", normalized)
    normalized = re.sub(r"(?<=[A-Za-z0-9_])\\b$", "", normalized)
    return normalized


def _compile_regex_fullmatch_codegen(pattern: str, *, prefix: str) -> tuple[str, str]:
    try:
        from emx_regex_cgen import generate as generate_regex_c
    except ImportError as exc:  # pragma: no cover - dependency failure
        raise CodegenError(
            "RegexFullMatch requires the emx-regex-cgen package to be installed"
        ) from exc

    attempts = [pattern]
    normalized = _normalize_regex_fullmatch_pattern(pattern)
    if normalized != pattern:
        attempts.append(normalized)

    errors: list[str] = []
    for candidate in attempts:
        try:
            generated = generate_regex_c(candidate, prefix=prefix)
        except ValueError as exc:
            errors.append(f"{candidate!r}: {exc}")
            continue
        match_function = generated.match_function.replace(
            f"bool {prefix}_match(",
            f"static inline bool {prefix}_match(",
            1,
        )
        return generated.globals.rstrip(), match_function.rstrip()

    raise UnsupportedOpError(
        "RegexFullMatch pattern is not supported by emx-regex-cgen: "
        + "; ".join(errors)
    )


@dataclass(frozen=True)
class RegexFullMatchOp(RenderableOpBase):
    __io_inputs__ = ("input0",)
    __io_outputs__ = ("output",)
    input0: str
    output: str
    pattern: str

    def required_includes(self, ctx: OpContext) -> set[str]:
        return {
            "#include <stdbool.h>",
            "#include <stddef.h>",
            "#include <stdint.h>",
            "#include <string.h>",
        }

    def emit(self, emitter: "Emitter", ctx: "EmitContext") -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        matcher_prefix = f"{op_name}_regex"
        matcher_globals, matcher_function = _compile_regex_fullmatch_codegen(
            self.pattern,
            prefix=matcher_prefix,
        )
        dim_args = emitter.dim_args_str()
        params = emitter.shared_param_map(
            [("input0", self.input0), ("output", self.output)]
        )
        input_shape = emitter.ctx_shape(self.input0)
        input_suffix = emitter.param_array_suffix(
            input_shape, emitter.dim_names_for(self.input0), dtype=ScalarType.STRING
        )
        output_dtype = emitter.ctx_dtype(self.output)
        output_suffix = emitter.param_array_suffix(
            emitter.ctx_shape(self.output), emitter.dim_names_for(self.output)
        )
        param_decls = emitter.build_param_decls(
            [
                (params["input0"], "char", input_suffix, True),
                (params["output"], output_dtype.c_type, output_suffix, False),
            ]
        )
        input_count = CEmitterCompat.element_count_expr(input_shape)
        lines = [
            matcher_globals,
            "",
            matcher_function,
            "",
            f"EMX_NODE_FN void {op_name}({dim_args}{', '.join(param_decls)}) {{",
            f"const char (*input_flat)[EMX_STRING_MAX_LEN] = (const char (*)[EMX_STRING_MAX_LEN]){params['input0']};",
            f"{output_dtype.c_type} *output_flat = ({output_dtype.c_type} *){params['output']};",
            f"for (idx_t i = 0; i < {input_count}; ++i) {{",
            f"    output_flat[i] = {matcher_prefix}_match(input_flat[i], strlen(input_flat[i]));",
            "}",
            "}",
        ]
        return emitter.with_node_comment(model, ctx.op_index, "\n".join(lines))

    def computed_output_dtype(self, emitter: "Emitter") -> "ScalarType":
        return ScalarType.BOOL


@dataclass(frozen=True)
class TreeEnsembleClassifierOp(RenderableOpBase):
    __io_inputs__ = ("input0",)
    __io_outputs__ = ("label", "probabilities")
    __io_remap_extra__ = ("output",)
    input0: str
    label: str
    probabilities: str
    output: str
    post_transform: str
    class_labels: tuple[int, ...]
    node_tree_ids: tuple[int, ...]
    node_node_ids: tuple[int, ...]
    node_feature_ids: tuple[int, ...]
    node_modes: tuple[int, ...]
    node_values: tuple[float, ...]
    node_true_ids: tuple[int, ...]
    node_false_ids: tuple[int, ...]
    class_tree_ids: tuple[int, ...]
    class_node_ids: tuple[int, ...]
    class_ids: tuple[int, ...]
    class_weights: tuple[float, ...]
    dtype: ScalarType = ScalarType.F32
    output_shape: tuple[int, ...] = ()

    @property
    def primary_output_name(self) -> str:
        return self.probabilities

    def emit(self, emitter: "Emitter", ctx: "EmitContext") -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        dim_args = emitter.dim_args_str()
        emitter.op_output_dtype(self)
        params = emitter.shared_param_map(
            [
                ("input0", self.input0),
                ("label", self.label),
                ("probabilities", self.probabilities),
            ]
        )
        input_shape = emitter.ctx_shape(self.input0)
        label_shape = emitter.ctx_shape(self.label)
        prob_shape = emitter.ctx_shape(self.probabilities)
        input_dtype = emitter.ctx_dtype(self.input0)
        label_dtype = emitter.ctx_dtype(self.label)
        prob_dtype = emitter.ctx_dtype(self.probabilities)
        input_suffix = emitter.param_array_suffix(
            input_shape, emitter.dim_names_for(self.input0)
        )
        label_suffix = emitter.param_array_suffix(
            label_shape, emitter.dim_names_for(self.label)
        )
        prob_suffix = emitter.param_array_suffix(
            prob_shape, emitter.dim_names_for(self.probabilities)
        )
        param_decls = emitter.build_param_decls(
            [
                (params["input0"], input_dtype.c_type, input_suffix, True),
                (params["label"], label_dtype.c_type, label_suffix, False),
                (params["probabilities"], prob_dtype.c_type, prob_suffix, False),
            ]
        )
        tree_ids = sorted(set(self.node_tree_ids))
        tree_id_literals = [
            emitter.format_literal(ScalarType.I64, value) for value in tree_ids
        ]
        rendered = (
            state.templates["tree_ensemble_classifier"]
            .render(
                op_name=op_name,
                dim_args=dim_args,
                params=param_decls,
                input0=params["input0"],
                label=params["label"],
                probabilities=params["probabilities"],
                class_labels=[
                    emitter.format_literal(ScalarType.I64, v) for v in self.class_labels
                ],
                node_tree_ids=[
                    emitter.format_literal(ScalarType.I64, v)
                    for v in self.node_tree_ids
                ],
                node_node_ids=[
                    emitter.format_literal(ScalarType.I64, v)
                    for v in self.node_node_ids
                ],
                node_feature_ids=[
                    emitter.format_literal(ScalarType.I64, v)
                    for v in self.node_feature_ids
                ],
                node_modes=[str(v) for v in self.node_modes],
                node_values=[
                    emitter.format_literal(input_dtype, v) for v in self.node_values
                ],
                node_true_ids=[
                    emitter.format_literal(ScalarType.I64, v)
                    for v in self.node_true_ids
                ],
                node_false_ids=[
                    emitter.format_literal(ScalarType.I64, v)
                    for v in self.node_false_ids
                ],
                class_tree_ids=[
                    emitter.format_literal(ScalarType.I64, v)
                    for v in self.class_tree_ids
                ],
                class_node_ids=[
                    emitter.format_literal(ScalarType.I64, v)
                    for v in self.class_node_ids
                ],
                class_ids=[
                    emitter.format_literal(ScalarType.I64, v) for v in self.class_ids
                ],
                class_weights=[
                    emitter.format_literal(ScalarType.F32, v)
                    for v in self.class_weights
                ],
                batch_size=input_shape[0],
                class_count=len(self.class_labels),
                node_count=len(self.node_node_ids),
                leaf_value_count=len(self.class_ids),
                tree_count=len(tree_ids),
                tree_ids=tree_id_literals,
                logistic=self.post_transform == "LOGISTIC",
                input_c_type=input_dtype.c_type,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)

    def c_op_outputs(
        self, emitter: "Emitter"
    ) -> tuple[tuple[str, tuple[int, ...], "ScalarType"], ...]:
        return (
            (self.label, emitter.ctx_shape(self.label), emitter.ctx_dtype(self.label)),
            (
                self.probabilities,
                emitter.ctx_shape(self.probabilities),
                emitter.ctx_dtype(self.probabilities),
            ),
        )


@dataclass(frozen=True)
class TreeEnsembleOp(RenderableOpBase):
    __io_inputs__ = ("input0",)
    __io_outputs__ = ("output",)
    input0: str
    output: str
    aggregate_function: int
    post_transform: int
    tree_roots: tuple[int, ...]
    node_feature_ids: tuple[int, ...]
    node_modes: tuple[int, ...]
    node_splits: tuple[float, ...]
    node_true_ids: tuple[int, ...]
    node_true_leafs: tuple[int, ...]
    node_false_ids: tuple[int, ...]
    node_false_leafs: tuple[int, ...]
    membership_values: tuple[float, ...] | None
    member_range_starts: tuple[int, ...]
    member_range_ends: tuple[int, ...]
    leaf_target_ids: tuple[int, ...]
    leaf_weights: tuple[float, ...]
    n_targets: int

    def emit(self, emitter: "Emitter", ctx: "EmitContext") -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        dim_args = emitter.dim_args_str()
        output_dtype = emitter.op_output_dtype(self)
        params = emitter.shared_param_map(
            [("input0", self.input0), ("output", self.output)]
        )
        input_shape = emitter.ctx_shape(self.input0)
        output_shape = emitter.ctx_shape(self.output)
        input_dtype = emitter.ctx_dtype(self.input0)
        output_dtype = emitter.ctx_dtype(self.output)
        input_suffix = emitter.param_array_suffix(
            input_shape, emitter.dim_names_for(self.input0)
        )
        output_suffix = emitter.param_array_suffix(
            output_shape, emitter.dim_names_for(self.output)
        )
        param_decls = emitter.build_param_decls(
            [
                (params["input0"], input_dtype.c_type, input_suffix, True),
                (params["output"], output_dtype.c_type, output_suffix, False),
            ]
        )
        rendered = (
            state.templates["tree_ensemble"]
            .render(
                op_name=op_name,
                dim_args=dim_args,
                params=param_decls,
                input0=params["input0"],
                output=params["output"],
                tree_roots=[
                    emitter.format_literal(ScalarType.I64, v) for v in self.tree_roots
                ],
                node_feature_ids=[
                    emitter.format_literal(ScalarType.I64, v)
                    for v in self.node_feature_ids
                ],
                node_modes=[str(v) for v in self.node_modes],
                node_splits=[
                    emitter.format_literal(input_dtype, v) for v in self.node_splits
                ],
                node_true_ids=[
                    emitter.format_literal(ScalarType.I64, v)
                    for v in self.node_true_ids
                ],
                node_true_leafs=[
                    emitter.format_literal(ScalarType.I64, v)
                    for v in self.node_true_leafs
                ],
                node_false_ids=[
                    emitter.format_literal(ScalarType.I64, v)
                    for v in self.node_false_ids
                ],
                node_false_leafs=[
                    emitter.format_literal(ScalarType.I64, v)
                    for v in self.node_false_leafs
                ],
                membership_values=[
                    emitter.format_literal(input_dtype, v)
                    for v in (self.membership_values or ())
                ],
                member_range_starts=[
                    emitter.format_literal(ScalarType.I64, v)
                    for v in self.member_range_starts
                ],
                member_range_ends=[
                    emitter.format_literal(ScalarType.I64, v)
                    for v in self.member_range_ends
                ],
                leaf_target_ids=[
                    emitter.format_literal(ScalarType.I64, v)
                    for v in self.leaf_target_ids
                ],
                leaf_weights=[
                    emitter.format_literal(output_dtype, v) for v in self.leaf_weights
                ],
                batch_size=input_shape[0],
                tree_count=len(self.tree_roots),
                target_count=self.n_targets,
                node_count=len(self.node_modes),
                leaf_count=len(self.leaf_target_ids),
                zero_literal=output_dtype.zero_literal,
                input_c_type=input_dtype.c_type,
                output_c_type=output_dtype.c_type,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)


@dataclass(frozen=True)
class SplitOp(RenderableOpBase):
    __io_inputs__ = ("input0",)
    __io_outputs__ = ("outputs",)
    input0: str
    outputs: tuple[str, ...]
    axis: int
    split_sizes: tuple[int, ...]

    def required_includes(self, ctx: OpContext) -> set[str]:
        return {"#include <string.h>"}

    def emit(self, emitter: "Emitter", ctx: "EmitContext") -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        emitter.dim_args_str()
        output_dtype = emitter.op_output_dtype(self)
        c_type = output_dtype.c_type
        input_shape = emitter.ctx_shape(self.input0)
        output_shapes = tuple(emitter.ctx_shape(name) for name in self.outputs)
        output_params = [
            (f"output_{index}", name) for index, name in enumerate(self.outputs)
        ]
        params = emitter.shared_param_map([("input0", self.input0), *output_params])
        output_names = tuple(
            params[f"output_{index}"] for index in range(len(self.outputs))
        )
        output_suffixes = tuple(
            emitter.param_array_suffix(shape, emitter.dim_names_for(name))
            for name, shape in zip(output_names, output_shapes)
        )
        outer = 1
        for dim in input_shape[: self.axis]:
            outer *= dim
        inner = 1
        for dim in input_shape[self.axis + 1 :]:
            inner *= dim
        input_suffix = emitter.param_array_suffix(
            input_shape, emitter.dim_names_for(self.input0)
        )
        param_decls = emitter.build_param_decls(
            [
                (params["input0"], c_type, input_suffix, True),
                *(
                    (name, c_type, suffix, False)
                    for name, suffix in zip(output_names, output_suffixes)
                ),
            ]
        )
        rendered = (
            state.templates["split"]
            .render(
                model_name=model.name,
                op_name=op_name,
                input0=params["input0"],
                outputs=output_names,
                params=param_decls,
                output_suffixes=output_suffixes,
                c_type=c_type,
                input_suffix=input_suffix,
                axis_sizes=self.split_sizes,
                axis_total=input_shape[self.axis],
                outer=outer,
                inner=inner,
                output_count=len(output_names),
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)

    def c_op_outputs(
        self, emitter: "Emitter"
    ) -> tuple[tuple[str, tuple[int, ...], "ScalarType"], ...]:
        return tuple(
            (name, emitter.ctx_shape(name), emitter.ctx_dtype(name))
            for name in self.outputs
        )


@dataclass(frozen=True)
class SplitToSequenceOp(RenderableOpBase):
    __io_inputs__ = ("input0", "split")
    __io_outputs__ = ("output_sequence",)
    input0: str
    split: str | None
    output_sequence: str
    axis: int
    keepdims: bool
    split_sizes: tuple[int, ...] | None
    split_scalar: bool

    def required_includes(self, ctx: OpContext) -> set[str]:
        return {"#include <string.h>"}

    def call_args(self) -> tuple[str, ...]:
        args = [self.input0]
        if self.split is not None:
            args.append(self.split)
        args.extend(
            [
                self.output_sequence,
                f"{self.output_sequence}__count",
            ]
        )
        return tuple(args)

    def emit(self, emitter: "Emitter", ctx: "EmitContext") -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        dim_args = emitter.dim_args_str()
        output_dtype = emitter.op_output_dtype(self)
        c_type = output_dtype.c_type
        input_shape = emitter.ctx_shape(self.input0)
        storage_shape = emitter.sequence_storage_shape(self.output_sequence)
        params = emitter.shared_param_map(
            [
                ("input0", self.input0),
                ("split", self.split),
                ("output_sequence", self.output_sequence),
            ]
        )
        input_suffix = emitter.param_array_suffix(
            input_shape, emitter.dim_names_for(self.input0)
        )
        output_suffix = emitter.param_array_suffix(
            storage_shape, emitter.dim_names_for(self.output_sequence)
        )
        input_shape_expr = CEmitterCompat.shape_dim_exprs(
            input_shape, emitter.dim_names_for(self.input0)
        )
        output_shape_expr = CEmitterCompat.shape_dim_exprs(
            storage_shape, emitter.dim_names_for(self.output_sequence)
        )
        dynamic_output_axes = emitter.sequence_dynamic_axes(self.output_sequence)
        param_specs = [
            (params["input0"], c_type, input_suffix, True),
        ]
        if params["split"] is not None:
            split_shape = emitter.ctx_shape(self.split)
            split_dtype = emitter.ctx_dtype(self.split)
            param_specs.append(
                (
                    params["split"],
                    split_dtype.c_type,
                    emitter.param_array_suffix(
                        split_shape, emitter.dim_names_for(self.split)
                    ),
                    True,
                )
            )
        param_specs.extend(
            [
                (
                    params["output_sequence"],
                    c_type,
                    f"[EMX_SEQUENCE_MAX_LEN]{output_suffix}",
                    False,
                ),
                (f"{params['output_sequence']}__count", "idx_t *", "", False),
                *[
                    (
                        emitter.sequence_dim_array_name(self.output_sequence, axis),
                        "idx_t",
                        "[EMX_SEQUENCE_MAX_LEN]",
                        False,
                    )
                    for axis in dynamic_output_axes
                ],
            ]
        )
        param_decls = emitter.build_param_decls(param_specs)
        outer = CEmitterCompat.element_count_expr(input_shape_expr[: self.axis])
        inner = CEmitterCompat.element_count_expr(input_shape_expr[self.axis + 1 :])
        split_len = (
            emitter.ctx_shape(self.split)[0]
            if self.split is not None and len(emitter.ctx_shape(self.split)) == 1
            else 0
        )
        rendered = (
            state.templates["split_to_sequence"]
            .render(
                op_name=op_name,
                dim_args=dim_args,
                params=param_decls,
                input0=params["input0"],
                split=params["split"],
                split_dtype=(
                    emitter.ctx_dtype(self.split).c_type
                    if self.split is not None
                    else "int64_t"
                ),
                split_scalar=self.split_scalar,
                split_len=split_len,
                axis_sizes=self.split_sizes,
                output_sequence=params["output_sequence"],
                axis_total=input_shape_expr[self.axis],
                outer=outer,
                inner=inner,
                keepdims=self.keepdims,
                output_axis_capacity=(
                    output_shape_expr[self.axis] if self.keepdims else 1
                ),
                output_dim_arrays=tuple(
                    {
                        "axis": axis,
                        "name": emitter.sequence_dim_array_name(
                            self.output_sequence, axis
                        ),
                        "expr": (
                            "chunk"
                            if self.keepdims and axis == self.axis
                            else (
                                input_shape_expr[axis]
                                if self.keepdims
                                else input_shape_expr[
                                    axis if axis < self.axis else axis + 1
                                ]
                            )
                        ),
                    }
                    for axis in dynamic_output_axes
                ),
                c_type=c_type,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)

    def resolved_output_dtype(self, ctx: OpContext) -> ScalarType:
        value = ctx.find_value(self.output_sequence)
        if isinstance(value.type, SequenceType):
            return value.type.elem.dtype
        raise CodegenError("SplitToSequence output must be a sequence")

    def computed_output_shape(self, emitter: "Emitter") -> tuple[int, ...]:
        return emitter.sequence_storage_shape(self.output_sequence)

    def computed_output_dtype(self, emitter: "Emitter") -> "ScalarType":
        return emitter.ctx_sequence_elem_type(self.output_sequence).dtype

    def c_op_outputs(
        self, emitter: "Emitter"
    ) -> tuple[tuple[str, tuple[int, ...], "ScalarType"], ...]:
        elem_type = emitter.ctx_sequence_elem_type(self.output_sequence)
        return (
            (
                self.output_sequence,
                emitter.sequence_storage_shape(self.output_sequence),
                elem_type.dtype,
            ),
        )


@dataclass(frozen=True)
class ReverseSequenceOp(RenderableOpBase):
    __io_inputs__ = ("input0", "sequence_lens")
    __io_outputs__ = ("output",)
    input0: str
    sequence_lens: str
    output: str
    batch_axis: int
    time_axis: int

    def emit(self, emitter: "Emitter", ctx: "EmitContext") -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        dim_args = emitter.dim_args_str()
        output_dtype = emitter.op_output_dtype(self)
        c_type = output_dtype.c_type
        input_shape = emitter.ctx_shape(self.input0)
        params = emitter.shared_param_map(
            [
                ("input0", self.input0),
                ("sequence_lens", self.sequence_lens),
                ("output", self.output),
            ]
        )
        seq_dtype = emitter.ctx_dtype(self.sequence_lens)
        input_suffix = emitter.param_array_suffix(
            input_shape, emitter.dim_names_for(self.input0)
        )
        output_suffix = emitter.param_array_suffix(
            input_shape, emitter.dim_names_for(self.output)
        )
        sequence_lens_suffix = emitter.param_array_suffix(
            (input_shape[self.batch_axis],)
        )
        param_decls = emitter.build_param_decls(
            [
                (params["input0"], c_type, input_suffix, True),
                (
                    params["sequence_lens"],
                    seq_dtype.c_type,
                    sequence_lens_suffix,
                    True,
                ),
                (params["output"], c_type, output_suffix, False),
            ]
        )
        rendered = (
            state.templates["reverse_sequence"]
            .render(
                op_name=op_name,
                dim_args=dim_args,
                params=param_decls,
                input0=params["input0"],
                sequence_lens=params["sequence_lens"],
                output=params["output"],
                rank=len(input_shape),
                dims=CEmitterCompat.shape_dim_exprs(
                    input_shape, emitter.dim_names_for(self.input0)
                ),
                batch_axis=self.batch_axis,
                time_axis=self.time_axis,
                seq_c_type=seq_dtype.c_type,
                c_type=c_type,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)


@dataclass(frozen=True)
class ConcatFromSequenceOp(RenderableOpBase):
    __io_inputs__ = ("input_sequence",)
    __io_outputs__ = ("output",)
    input_sequence: str
    output: str
    axis: int
    new_axis: bool
    elem_shape: tuple[int, ...]

    def call_args(self) -> tuple[str, ...]:
        return (
            self.input_sequence,
            f"{self.input_sequence}__count",
            self.output,
        )

    def emit(self, emitter: "Emitter", ctx: "EmitContext") -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        dim_args = emitter.dim_args_str()
        output_dtype = emitter.op_output_dtype(self)
        c_type = output_dtype.c_type
        elem_shape = self.elem_shape
        output_shape = emitter.ctx_shape(self.output)
        params = emitter.shared_param_map(
            [
                ("input_sequence", self.input_sequence),
                ("output", self.output),
            ]
        )
        elem_suffix = emitter.param_array_suffix(
            elem_shape, emitter.dim_names_for(self.input_sequence)
        )
        output_suffix = emitter.param_array_suffix(
            output_shape, emitter.dim_names_for(self.output)
        )
        param_decls = emitter.build_param_decls(
            [
                (
                    params["input_sequence"],
                    c_type,
                    f"[EMX_SEQUENCE_MAX_LEN]{elem_suffix}",
                    True,
                ),
                (f"{params['input_sequence']}__count", "idx_t", "", True),
                (params["output"], c_type, output_suffix, False),
            ]
        )
        if self.new_axis:
            outer = CEmitterCompat.element_count(elem_shape[: self.axis] or (1,))
            inner = CEmitterCompat.element_count(elem_shape[self.axis :] or (1,))
            axis_extent = 1
        else:
            outer = CEmitterCompat.element_count(elem_shape[: self.axis] or (1,))
            inner = CEmitterCompat.element_count(elem_shape[self.axis + 1 :] or (1,))
            axis_extent = elem_shape[self.axis]
        rendered = (
            state.templates["concat_from_sequence"]
            .render(
                op_name=op_name,
                dim_args=dim_args,
                params=param_decls,
                input_sequence=params["input_sequence"],
                output=params["output"],
                c_type=c_type,
                outer=outer,
                inner=inner,
                axis_extent=axis_extent,
                new_axis=self.new_axis,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)

    def c_op_inputs(
        self, emitter: "Emitter"
    ) -> tuple[tuple[str, tuple[int, ...]], ...]:
        return ()


@dataclass(frozen=True)
class SequenceAtOp(RenderableOpBase):
    __io_inputs__ = ("input_sequence", "position")
    __io_outputs__ = ("output",)
    input_sequence: str
    position: str
    output: str

    def call_args(self) -> tuple[str, ...]:
        return (
            self.input_sequence,
            f"{self.input_sequence}__count",
            self.position,
            self.output,
        )

    def emit(self, emitter: "Emitter", ctx: "EmitContext") -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        dim_args = emitter.dim_args_str()
        output_dtype = emitter.op_output_dtype(self)
        c_type = output_dtype.c_type
        output_shape = emitter.ctx_shape(self.output)
        params = emitter.shared_param_map(
            [
                ("input_sequence", self.input_sequence),
                ("position", self.position),
                ("output", self.output),
            ]
        )
        position_dtype = emitter.ctx_dtype(self.position)
        tensor_suffix = emitter.param_array_suffix(
            output_shape, emitter.dim_names_for(self.output)
        )
        input_suffix = emitter.param_array_suffix(
            emitter.sequence_storage_shape(self.input_sequence),
            emitter.dim_names_for(self.input_sequence),
        )
        param_decls = emitter.build_param_decls(
            [
                (
                    params["input_sequence"],
                    c_type,
                    f"[EMX_SEQUENCE_MAX_LEN]{input_suffix}",
                    True,
                ),
                (f"{params['input_sequence']}__count", "idx_t", "", True),
                (
                    params["position"],
                    position_dtype.c_type,
                    emitter.param_array_suffix((1,)),
                    True,
                ),
                (params["output"], c_type, tensor_suffix, False),
            ]
        )
        rendered = (
            state.templates["sequence_at"]
            .render(
                op_name=op_name,
                dim_args=dim_args,
                params=param_decls,
                input_sequence=params["input_sequence"],
                position=params["position"],
                output=params["output"],
                element_count=CEmitterCompat.element_count_expr(output_shape),
                c_type=c_type,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)

    def c_op_inputs(
        self, emitter: "Emitter"
    ) -> tuple[tuple[str, tuple[int, ...]], ...]:
        return ()


@dataclass(frozen=True)
class SequenceLengthOp(RenderableOpBase):
    __io_inputs__ = ("input_sequence",)
    __io_outputs__ = ("output",)
    input_sequence: str
    output: str

    def call_args(self) -> tuple[str, ...]:
        return (
            self.input_sequence,
            f"{self.input_sequence}__count",
            self.output,
        )

    def emit(self, emitter: "Emitter", ctx: "EmitContext") -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        dim_args = emitter.dim_args_str()
        output_dtype = emitter.op_output_dtype(self)
        output_shape = emitter.ctx_shape(self.output)
        params = emitter.shared_param_map(
            [
                ("input_sequence", self.input_sequence),
                ("output", self.output),
            ]
        )
        output_dtype = emitter.ctx_dtype(self.output)
        elem_type = emitter.ctx_sequence_elem_type(self.input_sequence)
        input_suffix = emitter.param_array_suffix(
            emitter.sequence_storage_shape(self.input_sequence),
            emitter.dim_names_for(self.input_sequence),
        )
        output_suffix = emitter.param_array_suffix(output_shape)
        param_decls = emitter.build_param_decls(
            [
                (
                    params["input_sequence"],
                    elem_type.dtype.c_type,
                    f"[EMX_SEQUENCE_MAX_LEN]{input_suffix}",
                    True,
                ),
                (f"{params['input_sequence']}__count", "idx_t", "", True),
                (params["output"], output_dtype.c_type, output_suffix, False),
            ]
        )
        rendered = (
            state.templates["sequence_length"]
            .render(
                op_name=op_name,
                dim_args=dim_args,
                params=param_decls,
                input_sequence=params["input_sequence"],
                output=params["output"],
                c_type=output_dtype.c_type,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)

    def c_op_inputs(
        self, emitter: "Emitter"
    ) -> tuple[tuple[str, tuple[int, ...]], ...]:
        return ()


@dataclass(frozen=True)
class SequenceIdentityOp(RenderableOpBase):
    __io_inputs__ = ("input_sequence",)
    __io_outputs__ = ("output_sequence",)
    __io_remap_extra__ = ("input_present", "output_present")
    input_sequence: str
    output_sequence: str
    input_present: str | None = None
    output_present: str | None = None

    def call_args(self) -> tuple[str, ...]:
        args = [
            self.input_sequence,
            f"{self.input_sequence}__count",
            self.output_sequence,
            f"{self.output_sequence}__count",
        ]
        if self.input_present is not None and self.output_present is not None:
            args.extend([self.input_present, self.output_present])
        return tuple(args)

    def emit(self, emitter: "Emitter", ctx: "EmitContext") -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        dim_args = emitter.dim_args_str()
        emitter.op_output_dtype(self)
        elem_type = emitter.ctx_sequence_elem_type(self.input_sequence)
        elem_shape = emitter.sequence_storage_shape(self.input_sequence)
        tensor_suffix = emitter.param_array_suffix(elem_shape)
        params = emitter.shared_param_map(
            [
                ("input_sequence", self.input_sequence),
                ("output_sequence", self.output_sequence),
                ("input_present", self.input_present),
                ("output_present", self.output_present),
                *[
                    (
                        f"input_sequence_dim_{axis}",
                        emitter.sequence_dim_array_name(self.input_sequence, axis),
                    )
                    for axis in emitter.sequence_dynamic_axes(self.input_sequence)
                ],
                *[
                    (
                        f"output_sequence_dim_{axis}",
                        emitter.sequence_dim_array_name(self.output_sequence, axis),
                    )
                    for axis in emitter.sequence_dynamic_axes(self.output_sequence)
                ],
            ]
        )
        param_specs = [
            (
                params["input_sequence"],
                elem_type.dtype.c_type,
                f"[EMX_SEQUENCE_MAX_LEN]{tensor_suffix}",
                True,
            ),
            (f"{params['input_sequence']}__count", "idx_t", "", True),
            (
                params["output_sequence"],
                elem_type.dtype.c_type,
                f"[EMX_SEQUENCE_MAX_LEN]{tensor_suffix}",
                False,
            ),
            (f"{params['output_sequence']}__count", "idx_t *", "", False),
            *[
                (
                    params[f"input_sequence_dim_{axis}"],
                    "idx_t",
                    "[EMX_SEQUENCE_MAX_LEN]",
                    True,
                )
                for axis in emitter.sequence_dynamic_axes(self.input_sequence)
            ],
            *[
                (
                    params[f"output_sequence_dim_{axis}"],
                    "idx_t",
                    "[EMX_SEQUENCE_MAX_LEN]",
                    False,
                )
                for axis in emitter.sequence_dynamic_axes(self.output_sequence)
            ],
        ]
        if params["input_present"] is not None and params["output_present"] is not None:
            param_specs.extend(
                [
                    (params["input_present"], "_Bool", "", True),
                    (params["output_present"], "_Bool *", "", False),
                ]
            )
        rendered = (
            state.templates["sequence_identity"]
            .render(
                op_name=op_name,
                dim_args=dim_args,
                params=emitter.build_param_decls(param_specs),
                input_sequence=params["input_sequence"],
                output_sequence=params["output_sequence"],
                input_present=params["input_present"],
                output_present=params["output_present"],
                input_dim_arrays=tuple(
                    {
                        "axis": axis,
                        "name": params[f"input_sequence_dim_{axis}"],
                    }
                    for axis in emitter.sequence_dynamic_axes(self.input_sequence)
                ),
                output_dim_arrays=tuple(
                    {
                        "axis": axis,
                        "name": params[f"output_sequence_dim_{axis}"],
                    }
                    for axis in emitter.sequence_dynamic_axes(self.output_sequence)
                ),
                element_count=CEmitterCompat.element_count_expr(elem_shape),
                c_type=elem_type.dtype.c_type,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)

    def resolved_output_dtype(self, ctx: OpContext) -> ScalarType:
        value = ctx.find_value(self.output_sequence)
        if isinstance(value.type, SequenceType):
            return value.type.elem.dtype
        raise CodegenError("SequenceIdentity output must be a sequence")

    def computed_output_dtype(self, emitter: "Emitter") -> "ScalarType":
        return emitter.ctx_sequence_elem_type(self.output_sequence).dtype

    def c_op_outputs(
        self, emitter: "Emitter"
    ) -> tuple[tuple[str, tuple[int, ...], "ScalarType"], ...]:
        return (
            (
                self.output_sequence,
                emitter.sequence_storage_shape(self.output_sequence),
                emitter.ctx_sequence_elem_type(self.output_sequence).dtype,
            ),
        )

    def c_op_inputs(
        self, emitter: "Emitter"
    ) -> tuple[tuple[str, tuple[int, ...]], ...]:
        return ()


@dataclass(frozen=True)
class SequenceInsertOp(RenderableOpBase):
    __io_inputs__ = ("input_sequence", "tensor", "position")
    __io_outputs__ = ("output_sequence",)
    input_sequence: str
    tensor: str
    position: str | None
    output_sequence: str

    def call_args(self) -> tuple[str, ...]:
        args = [self.input_sequence, f"{self.input_sequence}__count", self.tensor]
        if self.position is not None:
            args.append(self.position)
        args.extend([self.output_sequence, f"{self.output_sequence}__count"])
        return tuple(args)

    def emit(self, emitter: "Emitter", ctx: "EmitContext") -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        dim_args = emitter.dim_args_str()
        output_dtype = emitter.op_output_dtype(self)
        c_type = output_dtype.c_type
        tensor_shape = emitter.ctx_shape(self.tensor)
        params = emitter.shared_param_map(
            [
                ("input_sequence", self.input_sequence),
                ("tensor", self.tensor),
                ("position", self.position),
                ("output_sequence", self.output_sequence),
            ]
        )
        tensor_suffix = emitter.param_array_suffix(
            tensor_shape, emitter.dim_names_for(self.tensor)
        )
        sequence_suffix = emitter.param_array_suffix(
            emitter.sequence_storage_shape(self.input_sequence),
        )
        position_dtype = (
            emitter.ctx_dtype(self.position)
            if self.position is not None
            else ScalarType.I64
        )
        dynamic_input_axes = emitter.sequence_dynamic_axes(self.input_sequence)
        dynamic_output_axes = emitter.sequence_dynamic_axes(self.output_sequence)
        param_specs = [
            (
                params["input_sequence"],
                c_type,
                f"[EMX_SEQUENCE_MAX_LEN]{sequence_suffix}",
                True,
            ),
            (f"{params['input_sequence']}__count", "idx_t", "", True),
            *[
                (
                    emitter.sequence_dim_array_name(self.input_sequence, axis),
                    "idx_t",
                    "[EMX_SEQUENCE_MAX_LEN]",
                    True,
                )
                for axis in dynamic_input_axes
            ],
            (params["tensor"], c_type, tensor_suffix, True),
        ]
        if params["position"] is not None:
            param_specs.append(
                (
                    params["position"],
                    position_dtype.c_type,
                    emitter.param_array_suffix((1,)),
                    True,
                )
            )
        param_specs.extend(
            [
                (
                    params["output_sequence"],
                    c_type,
                    f"[EMX_SEQUENCE_MAX_LEN]{sequence_suffix}",
                    False,
                ),
                (f"{params['output_sequence']}__count", "idx_t *", "", False),
                *[
                    (
                        emitter.sequence_dim_array_name(self.output_sequence, axis),
                        "idx_t",
                        "[EMX_SEQUENCE_MAX_LEN]",
                        False,
                    )
                    for axis in dynamic_output_axes
                ],
            ]
        )
        param_decls = emitter.build_param_decls(param_specs)
        rendered = (
            state.templates["sequence_insert"]
            .render(
                op_name=op_name,
                dim_args=dim_args,
                params=param_decls,
                input_sequence=params["input_sequence"],
                tensor=params["tensor"],
                position=params["position"],
                output_sequence=params["output_sequence"],
                input_dim_arrays=tuple(
                    {
                        "axis": axis,
                        "name": emitter.sequence_dim_array_name(
                            self.input_sequence, axis
                        ),
                    }
                    for axis in dynamic_input_axes
                ),
                output_dim_arrays=tuple(
                    {
                        "axis": axis,
                        "name": emitter.sequence_dim_array_name(
                            self.output_sequence, axis
                        ),
                        "tensor_dim": (
                            emitter.dim_names_for(self.tensor).get(axis)
                            if axis < len(tensor_shape)
                            else None
                        ),
                        "tensor_shape": tensor_shape[axis]
                        if axis < len(tensor_shape)
                        else 1,
                    }
                    for axis in dynamic_output_axes
                ),
                sequence_element_count=CEmitterCompat.element_count_expr(
                    emitter.sequence_storage_shape(self.input_sequence)
                ),
                tensor_element_count=CEmitterCompat.element_count_expr(tensor_shape),
                c_type=c_type,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)

    def resolved_output_dtype(self, ctx: OpContext) -> ScalarType:
        return ctx.dtype(self.tensor)

    def computed_output_shape(self, emitter: "Emitter") -> tuple[int, ...]:
        return emitter.sequence_storage_shape(self.output_sequence)

    def computed_output_dtype(self, emitter: "Emitter") -> "ScalarType":
        return emitter.ctx_sequence_elem_type(self.output_sequence).dtype

    def c_op_outputs(
        self, emitter: "Emitter"
    ) -> tuple[tuple[str, tuple[int, ...], "ScalarType"], ...]:
        return (
            (
                self.output_sequence,
                emitter.sequence_storage_shape(self.output_sequence),
                emitter.ctx_sequence_elem_type(self.output_sequence).dtype,
            ),
        )

    def c_op_inputs(
        self, emitter: "Emitter"
    ) -> tuple[tuple[str, tuple[int, ...]], ...]:
        return ()


@dataclass(frozen=True)
class IfOptionalSequenceConstOp(RenderableOpBase):
    __io_inputs__ = ("cond",)
    __io_outputs__ = ("output_sequence", "output_present")
    cond: str
    output_sequence: str
    output_present: str | None
    true_present: bool
    false_present: bool
    true_values: tuple[float | int | bool, ...]
    false_values: tuple[float | int | bool, ...]

    def call_args(self) -> tuple[str, ...]:
        args = [self.cond, self.output_sequence]
        if self.output_present is not None:
            args.append(self.output_present)
        args.append(f"{self.output_sequence}__count")
        return tuple(args)

    def emit(self, emitter: "Emitter", ctx: "EmitContext") -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        dim_args = emitter.dim_args_str()
        emitter.op_output_dtype(self)
        elem_type = emitter.ctx_sequence_elem_type(self.output_sequence)
        elem_shape = CEmitterCompat.codegen_shape(elem_type.shape)
        true_has_values = len(self.true_values) > 0
        false_has_values = len(self.false_values) > 0
        sequence_suffix = emitter.param_array_suffix(
            emitter.sequence_storage_shape(self.output_sequence),
            emitter.dim_names_for(self.output_sequence),
        )
        cond_shape = CEmitterCompat.codegen_shape(emitter.ctx_shape(self.cond))
        params = emitter.shared_param_map(
            [
                ("cond", self.cond),
                ("output_sequence", self.output_sequence),
                ("output_present", self.output_present),
            ]
        )
        param_specs = [
            (
                params["cond"],
                ScalarType.BOOL.c_type,
                emitter.param_array_suffix(cond_shape),
                True,
            ),
            (
                params["output_sequence"],
                elem_type.dtype.c_type,
                f"[EMX_SEQUENCE_MAX_LEN]{sequence_suffix}",
                False,
            ),
        ]
        if params["output_present"] is not None:
            param_specs.append((params["output_present"], "_Bool *", "", False))
        param_specs.append(
            (f"{params['output_sequence']}__count", "idx_t *", "", False)
        )
        param_decls = emitter.build_param_decls(param_specs)
        true_values_name = f"{op_name}_true_values"
        false_values_name = f"{op_name}_false_values"
        elem_count = CEmitterCompat.element_count(elem_shape)
        lines = [f"void {op_name}({dim_args}{', '.join(param_decls)}) {{"]
        if true_has_values:
            true_literals = [
                emitter.format_literal(elem_type.dtype, value)
                for value in self.true_values
            ]
            lines.append(
                f"const {elem_type.dtype.c_type} {true_values_name}[{elem_count}] = {{"
            )
            lines.extend(emitter.emit_initializer_lines(true_literals, (elem_count,)))
            lines.append("};")
        if false_has_values:
            false_literals = [
                emitter.format_literal(elem_type.dtype, value)
                for value in self.false_values
            ]
            lines.append(
                f"const {elem_type.dtype.c_type} {false_values_name}[{elem_count}] = {{"
            )
            lines.extend(emitter.emit_initializer_lines(false_literals, (elem_count,)))
            lines.append("};")
        lines.append(f"if ({params['cond']}[0]) {{")
        if self.true_present:
            lines.append(
                f"    *{params['output_sequence']}__count = {1 if true_has_values else 0};"
            )
            if true_has_values:
                loop_vars = CEmitterCompat.loop_vars(elem_shape)
                for depth, var in enumerate(loop_vars):
                    lines.append(
                        f"    for (idx_t {var} = 0; {var} < {elem_shape[depth]}; ++{var}) {{"
                    )
                index_expr = emitter.index_expr(elem_shape, loop_vars)
                output_index = "".join(f"[{var}]" for var in loop_vars)
                lines.append(
                    f"        {params['output_sequence']}[0]{output_index} = {true_values_name}[{index_expr}];"
                )
                for _ in loop_vars:
                    lines.append("    }")
        else:
            lines.append(f"    *{params['output_sequence']}__count = 0;")
        if params["output_present"] is not None:
            lines.append(
                f"    *{params['output_present']} = {'1' if self.true_present else '0'};"
            )
        lines.append("} else {")
        if self.false_present:
            lines.append(
                f"    *{params['output_sequence']}__count = {1 if false_has_values else 0};"
            )
            if false_has_values:
                loop_vars = CEmitterCompat.loop_vars(elem_shape)
                for depth, var in enumerate(loop_vars):
                    lines.append(
                        f"    for (idx_t {var} = 0; {var} < {elem_shape[depth]}; ++{var}) {{"
                    )
                index_expr = emitter.index_expr(elem_shape, loop_vars)
                output_index = "".join(f"[{var}]" for var in loop_vars)
                lines.append(
                    f"        {params['output_sequence']}[0]{output_index} = {false_values_name}[{index_expr}];"
                )
                for _ in loop_vars:
                    lines.append("    }")
        else:
            lines.append(f"    *{params['output_sequence']}__count = 0;")
        if params["output_present"] is not None:
            lines.append(
                f"    *{params['output_present']} = {'1' if self.false_present else '0'};"
            )
        lines.append("}")
        lines.append("}")
        return emitter.with_node_comment(
            model, ctx.op_index, emitter.format_c_indentation("\n".join(lines))
        )

    def resolved_output_dtype(self, ctx: OpContext) -> ScalarType:
        value = ctx.find_value(self.output_sequence)
        if isinstance(value.type, SequenceType):
            return value.type.elem.dtype
        raise CodegenError("IfOptionalSequenceConst output must be a sequence")

    def computed_output_shape(self, emitter: "Emitter") -> tuple[int, ...]:
        return emitter.sequence_storage_shape(self.output_sequence)

    def computed_output_dtype(self, emitter: "Emitter") -> "ScalarType":
        return emitter.ctx_sequence_elem_type(self.output_sequence).dtype

    def c_op_outputs(
        self, emitter: "Emitter"
    ) -> tuple[tuple[str, tuple[int, ...], "ScalarType"], ...]:
        return (
            (
                self.output_sequence,
                emitter.sequence_storage_shape(self.output_sequence),
                emitter.ctx_sequence_elem_type(self.output_sequence).dtype,
            ),
        )


@dataclass(frozen=True)
class SequenceEraseOp(RenderableOpBase):
    __io_inputs__ = ("input_sequence", "position")
    __io_outputs__ = ("output_sequence",)
    input_sequence: str
    position: str | None
    output_sequence: str

    def call_args(self) -> tuple[str, ...]:
        args = [self.input_sequence, f"{self.input_sequence}__count"]
        if self.position is not None:
            args.append(self.position)
        args.extend([self.output_sequence, f"{self.output_sequence}__count"])
        return tuple(args)

    def emit(self, emitter: "Emitter", ctx: "EmitContext") -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        dim_args = emitter.dim_args_str()
        output_dtype = emitter.op_output_dtype(self)
        c_type = output_dtype.c_type
        elem_shape = emitter.sequence_storage_shape(self.input_sequence)
        params = emitter.shared_param_map(
            [
                ("input_sequence", self.input_sequence),
                ("position", self.position),
                ("output_sequence", self.output_sequence),
            ]
        )
        position_dtype = (
            emitter.ctx_dtype(self.position)
            if self.position is not None
            else ScalarType.I64
        )
        tensor_suffix = emitter.param_array_suffix(elem_shape)
        sequence_suffix = tensor_suffix
        dynamic_input_axes = emitter.sequence_dynamic_axes(self.input_sequence)
        dynamic_output_axes = emitter.sequence_dynamic_axes(self.output_sequence)
        param_specs = [
            (
                params["input_sequence"],
                c_type,
                f"[EMX_SEQUENCE_MAX_LEN]{tensor_suffix}",
                True,
            ),
            (f"{params['input_sequence']}__count", "idx_t", "", True),
            *[
                (
                    emitter.sequence_dim_array_name(self.input_sequence, axis),
                    "idx_t",
                    "[EMX_SEQUENCE_MAX_LEN]",
                    True,
                )
                for axis in dynamic_input_axes
            ],
        ]
        if params["position"] is not None:
            param_specs.append(
                (
                    params["position"],
                    position_dtype.c_type,
                    emitter.param_array_suffix((1,)),
                    True,
                )
            )
        param_specs.extend(
            [
                (
                    params["output_sequence"],
                    c_type,
                    f"[EMX_SEQUENCE_MAX_LEN]{sequence_suffix}",
                    False,
                ),
                (f"{params['output_sequence']}__count", "idx_t *", "", False),
                *[
                    (
                        emitter.sequence_dim_array_name(self.output_sequence, axis),
                        "idx_t",
                        "[EMX_SEQUENCE_MAX_LEN]",
                        False,
                    )
                    for axis in dynamic_output_axes
                ],
            ]
        )
        param_decls = emitter.build_param_decls(param_specs)
        rendered = (
            state.templates["sequence_erase"]
            .render(
                op_name=op_name,
                dim_args=dim_args,
                params=param_decls,
                input_sequence=params["input_sequence"],
                position=params["position"],
                output_sequence=params["output_sequence"],
                input_dim_arrays=tuple(
                    {
                        "axis": axis,
                        "name": emitter.sequence_dim_array_name(
                            self.input_sequence, axis
                        ),
                    }
                    for axis in dynamic_input_axes
                ),
                output_dim_arrays=tuple(
                    {
                        "axis": axis,
                        "name": emitter.sequence_dim_array_name(
                            self.output_sequence, axis
                        ),
                    }
                    for axis in dynamic_output_axes
                ),
                element_count=CEmitterCompat.element_count_expr(elem_shape),
                c_type=c_type,
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)

    def resolved_output_dtype(self, ctx: OpContext) -> ScalarType:
        value = ctx.find_value(self.input_sequence)
        if isinstance(value.type, SequenceType):
            return value.type.elem.dtype
        raise CodegenError("SequenceErase input must be a sequence")

    def computed_output_shape(self, emitter: "Emitter") -> tuple[int, ...]:
        return emitter.sequence_storage_shape(self.input_sequence)

    def computed_output_dtype(self, emitter: "Emitter") -> "ScalarType":
        return emitter.ctx_sequence_elem_type(self.input_sequence).dtype

    def c_op_outputs(
        self, emitter: "Emitter"
    ) -> tuple[tuple[str, tuple[int, ...], "ScalarType"], ...]:
        elem_type = emitter.ctx_sequence_elem_type(self.input_sequence)
        return (
            (
                self.output_sequence,
                emitter.sequence_storage_shape(self.input_sequence),
                elem_type.dtype,
            ),
        )

    def c_op_inputs(
        self, emitter: "Emitter"
    ) -> tuple[tuple[str, tuple[int, ...]], ...]:
        return ()


@dataclass(frozen=True)
class SequenceConstructOp(RenderableOpBase):
    __io_inputs__ = ("inputs",)
    __io_outputs__ = ("output_sequence",)
    inputs: tuple[str, ...]
    output_sequence: str

    def call_args(self) -> tuple[str, ...]:
        return (*self.inputs, self.output_sequence, f"{self.output_sequence}__count")

    def emit(self, emitter: "Emitter", ctx: "EmitContext") -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        dim_args = emitter.dim_args_str()
        output_dtype = emitter.op_output_dtype(self)
        c_type = output_dtype.c_type
        if not self.inputs:
            raise CodegenError("SequenceConstruct requires at least one input")
        elem_shape = emitter.ctx_shape(self.inputs[0])
        input_params = [
            (f"input_{index}", name) for index, name in enumerate(self.inputs)
        ]
        params = emitter.shared_param_map(
            [*input_params, ("output_sequence", self.output_sequence)]
        )
        tensor_suffix = emitter.param_array_suffix(
            elem_shape, emitter.dim_names_for(self.inputs[0])
        )
        param_specs = [
            (params[f"input_{index}"], c_type, tensor_suffix, True)
            for index in range(len(self.inputs))
        ]
        param_specs.extend(
            [
                (
                    params["output_sequence"],
                    c_type,
                    f"[EMX_SEQUENCE_MAX_LEN]{tensor_suffix}",
                    False,
                ),
                (f"{params['output_sequence']}__count", "idx_t *", "", False),
            ]
        )
        param_decls = emitter.build_param_decls(param_specs)
        rendered = (
            state.templates["sequence_construct"]
            .render(
                op_name=op_name,
                dim_args=dim_args,
                params=param_decls,
                inputs=tuple(
                    params[f"input_{index}"] for index in range(len(self.inputs))
                ),
                output_sequence=params["output_sequence"],
                element_count=CEmitterCompat.element_count_expr(elem_shape),
                c_type=c_type,
                input_count=len(self.inputs),
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)

    def resolved_output_dtype(self, ctx: OpContext) -> ScalarType:
        return ctx.dtype(self.inputs[0])

    def computed_output_dtype(self, emitter: "Emitter") -> "ScalarType":
        return emitter.ctx_dtype(self.inputs[0])

    def c_op_outputs(
        self, emitter: "Emitter"
    ) -> tuple[tuple[str, tuple[int, ...], "ScalarType"], ...]:
        return (
            (
                self.output_sequence,
                emitter.ctx_shape(self.inputs[0]),
                emitter.ctx_dtype(self.inputs[0]),
            ),
        )


@dataclass(frozen=True)
class SequenceEmptyOp(RenderableOpBase):
    __io_inputs__ = ()
    __io_outputs__ = ("output_sequence",)
    output_sequence: str

    def call_args(self) -> tuple[str, ...]:
        return (self.output_sequence, f"{self.output_sequence}__count")

    def emit(self, emitter: "Emitter", ctx: "EmitContext") -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        dim_args = emitter.dim_args_str()
        emitter.op_output_dtype(self)
        elem_type = emitter.ctx_sequence_elem_type(self.output_sequence)
        tensor_suffix = emitter.param_array_suffix(
            emitter.sequence_storage_shape(self.output_sequence),
            emitter.dim_names_for(self.output_sequence),
        )
        params = emitter.shared_param_map([("output_sequence", self.output_sequence)])
        param_decls = emitter.build_param_decls(
            [
                (
                    params["output_sequence"],
                    elem_type.dtype.c_type,
                    f"[EMX_SEQUENCE_MAX_LEN]{tensor_suffix}",
                    False,
                ),
                (f"{params['output_sequence']}__count", "idx_t *", "", False),
            ]
        )
        rendered = (
            state.templates["sequence_empty"]
            .render(
                op_name=op_name,
                dim_args=dim_args,
                params=param_decls,
                output_sequence=params["output_sequence"],
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)

    def resolved_output_dtype(self, ctx: OpContext) -> ScalarType:
        value = ctx.find_value(self.output_sequence)
        if isinstance(value.type, SequenceType):
            return value.type.elem.dtype
        raise CodegenError("SequenceEmpty output must be a sequence")

    def computed_output_shape(self, emitter: "Emitter") -> tuple[int, ...]:
        return emitter.sequence_storage_shape(self.output_sequence)

    def computed_output_dtype(self, emitter: "Emitter") -> "ScalarType":
        return emitter.ctx_sequence_elem_type(self.output_sequence).dtype

    def c_op_outputs(
        self, emitter: "Emitter"
    ) -> tuple[tuple[str, tuple[int, ...], "ScalarType"], ...]:
        elem_type = emitter.ctx_sequence_elem_type(self.output_sequence)
        return (
            (
                self.output_sequence,
                emitter.sequence_storage_shape(self.output_sequence),
                elem_type.dtype,
            ),
        )

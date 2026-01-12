from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..errors import CodegenError
from ..dtypes import dtype_info


@dataclass(frozen=True)
class BinaryOp:
    input0: str
    input1: str
    output: str
    operator: str
    operator_kind: str
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class UnaryOp:
    input0: str
    output: str
    operator: str
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class MatMulOp:
    input0: str
    input1: str
    output: str
    m: int
    n: int
    k: int
    dtype: str


@dataclass(frozen=True)
class GemmOp:
    input_a: str
    input_b: str
    input_c: str | None
    output: str
    m: int
    n: int
    k: int
    trans_a: bool
    trans_b: bool
    alpha: float | int
    beta: float | int
    c_shape: tuple[int, ...] | None
    dtype: str


@dataclass(frozen=True)
class AttentionOp:
    input_q: str
    input_k: str
    input_v: str
    output: str
    batch: int
    heads: int
    q_seq: int
    kv_seq: int
    qk_head_size: int
    v_head_size: int
    scale: float
    is_causal: bool
    dtype: str


@dataclass(frozen=True)
class ConvOp:
    input0: str
    weights: str
    bias: str | None
    output: str
    batch: int
    in_channels: int
    out_channels: int
    spatial_rank: int
    in_spatial: tuple[int, ...]
    out_spatial: tuple[int, ...]
    kernel_shape: tuple[int, ...]
    strides: tuple[int, ...]
    pads: tuple[int, ...]
    dilations: tuple[int, ...]
    group: int
    dtype: str

    @property
    def out_h(self) -> int:
        if self.spatial_rank < 1:
            raise ValueError("Conv output height is undefined for spatial_rank < 1")
        return self.out_spatial[0]

    @property
    def out_w(self) -> int:
        if self.spatial_rank < 2:
            raise ValueError("Conv output width is undefined for spatial_rank < 2")
        return self.out_spatial[1]


@dataclass(frozen=True)
class AveragePoolOp:
    input0: str
    output: str
    batch: int
    channels: int
    in_h: int
    in_w: int
    out_h: int
    out_w: int
    kernel_h: int
    kernel_w: int
    stride_h: int
    stride_w: int
    pad_top: int
    pad_left: int
    pad_bottom: int
    pad_right: int
    count_include_pad: bool
    dtype: str


@dataclass(frozen=True)
class SoftmaxOp:
    input0: str
    output: str
    outer: int
    axis_size: int
    inner: int
    axis: int
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class BatchNormOp:
    input0: str
    scale: str
    bias: str
    mean: str
    variance: str
    output: str
    shape: tuple[int, ...]
    channels: int
    epsilon: float
    dtype: str


@dataclass(frozen=True)
class LrnOp:
    input0: str
    output: str
    shape: tuple[int, ...]
    channels: int
    size: int
    half: int
    alpha: float
    beta: float
    bias: float
    dtype: str


@dataclass(frozen=True)
class MaxPoolOp:
    input0: str
    output: str
    batch: int
    channels: int
    spatial_rank: int
    in_spatial: tuple[int, ...]
    out_spatial: tuple[int, ...]
    kernel_shape: tuple[int, ...]
    strides: tuple[int, ...]
    pads: tuple[int, ...]
    dilations: tuple[int, ...]
    ceil_mode: bool
    dtype: str


@dataclass(frozen=True)
class ConcatOp:
    inputs: tuple[str, ...]
    output: str
    axis: int
    input_shapes: tuple[tuple[int, ...], ...]
    output_shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class TransposeOp:
    input0: str
    output: str
    perm: tuple[int, ...]
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class ReshapeOp:
    input0: str
    output: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class ReduceOp:
    input0: str
    output: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    axes: tuple[int, ...]
    keepdims: bool
    reduce_kind: str
    reduce_count: int
    dtype: str


@dataclass(frozen=True)
class ConstantOfShapeOp:
    input0: str
    output: str
    input_shape: tuple[int, ...]
    shape: tuple[int, ...]
    value: float | int | bool
    dtype: str
    input_dtype: str


@dataclass(frozen=True)
class ConstTensor:
    name: str
    shape: tuple[int, ...]
    data: tuple[float | int | bool, ...]
    dtype: str


@dataclass(frozen=True)
class TempBuffer:
    name: str
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class LoweredModel:
    name: str
    input_names: tuple[str, ...]
    input_shapes: tuple[tuple[int, ...], ...]
    input_dtypes: tuple[str, ...]
    output_names: tuple[str, ...]
    output_shapes: tuple[tuple[int, ...], ...]
    output_dtypes: tuple[str, ...]
    constants: tuple[ConstTensor, ...]
    ops: tuple[
        BinaryOp
        | UnaryOp
        | MatMulOp
        | GemmOp
        | AttentionOp
        | ConvOp
        | AveragePoolOp
        | BatchNormOp
        | LrnOp
        | SoftmaxOp
        | MaxPoolOp
        | ConcatOp
        | TransposeOp
        | ReshapeOp
        | ReduceOp
        | ConstantOfShapeOp,
        ...,
    ]


class CEmitter:
    def __init__(self, template_dir: Path) -> None:
        self._env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=select_autoescape(enabled_extensions=()),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def emit_model(self, model: LoweredModel, *, emit_testbench: bool = False) -> str:
        try:
            binary_template = self._env.get_template("binary_op.c.j2")
            unary_template = self._env.get_template("unary_op.c.j2")
            matmul_template = self._env.get_template("matmul_op.c.j2")
            gemm_template = self._env.get_template("gemm_op.c.j2")
            attention_template = self._env.get_template("attention_op.c.j2")
            conv_template = self._env.get_template("conv_op.c.j2")
            avg_pool_template = self._env.get_template("average_pool_op.c.j2")
            batch_norm_template = self._env.get_template("batch_norm_op.c.j2")
            lrn_template = self._env.get_template("lrn_op.c.j2")
            softmax_template = self._env.get_template("softmax_op.c.j2")
            maxpool_template = self._env.get_template("maxpool_op.c.j2")
            concat_template = self._env.get_template("concat_op.c.j2")
            transpose_template = self._env.get_template("transpose_op.c.j2")
            reshape_template = self._env.get_template("reshape_op.c.j2")
            reduce_template = self._env.get_template("reduce_op.c.j2")
            constant_of_shape_template = self._env.get_template(
                "constant_of_shape_op.c.j2"
            )
            testbench_template = None
            if emit_testbench:
                testbench_template = self._env.get_template("testbench.c.j2")
        except Exception as exc:  # pragma: no cover - template load failure
            raise CodegenError("Failed to load C template") from exc
        temp_buffers = self._temp_buffers(model)
        temp_name_map = {
            original: buffer.name for original, buffer in temp_buffers.items()
        }
        resolved_ops = [self._resolve_op(op, temp_name_map) for op in model.ops]
        operator_fns = "\n\n".join(
            self._render_op(
                model,
                op,
                index,
                array_suffix="",
                loop_vars=(),
                loop_indents=(),
                inner_indent="",
                c_type=dtype_info(op.dtype).c_type,
                zero_literal=dtype_info(op.dtype).zero_literal,
                min_literal=dtype_info(op.dtype).min_literal,
                max_literal=dtype_info(op.dtype).max_literal,
                binary_template=binary_template,
                unary_template=unary_template,
                matmul_template=matmul_template,
                gemm_template=gemm_template,
                attention_template=attention_template,
                conv_template=conv_template,
                avg_pool_template=avg_pool_template,
                batch_norm_template=batch_norm_template,
                lrn_template=lrn_template,
                softmax_template=softmax_template,
                maxpool_template=maxpool_template,
                concat_template=concat_template,
                transpose_template=transpose_template,
                reshape_template=reshape_template,
                reduce_template=reduce_template,
                constant_of_shape_template=constant_of_shape_template,
            )
            for index, op in enumerate(resolved_ops)
        )
        wrapper_fn = self._emit_model_wrapper(
            model,
            resolved_ops,
            tuple(temp_buffers.values()),
        )
        includes = ["#include <stddef.h>"]
        if emit_testbench:
            includes.extend(("#include <stdio.h>", "#include <stdint.h>"))
        constant_of_shape_inputs = {
            op.input_dtype
            for op in resolved_ops
            if isinstance(op, ConstantOfShapeOp)
        }
        model_dtypes = {
            *model.input_dtypes,
            *model.output_dtypes,
            *(const.dtype for const in model.constants),
            *(op.dtype for op in resolved_ops),
            *constant_of_shape_inputs,
        }
        if (
            any(dtype in {"int64", "int32", "int16", "int8"} for dtype in model_dtypes)
            and "#include <stdint.h>" not in includes
        ):
            includes.append("#include <stdint.h>")
        if "bool" in model_dtypes:
            includes.append("#include <stdbool.h>")
        math_ops = {
            "atanhf",
            "ceilf",
            "cosf",
            "expf",
            "fabsf",
            "floorf",
            "logf",
            "sinf",
            "sqrtf",
            "tanf",
            "tanhf",
        }
        if any(
            isinstance(op, UnaryOp) and op.operator in {"llabs", "abs"}
            for op in resolved_ops
        ):
            includes.append("#include <stdlib.h>")
        binary_math_ops = {"fmaxf", "fminf", "fmodf", "powf"}
        if any(
            isinstance(op, UnaryOp) and op.operator in math_ops
            for op in resolved_ops
        ):
            includes.append("#include <math.h>")
        if any(
            isinstance(op, BinaryOp) and op.operator in binary_math_ops
            for op in resolved_ops
        ):
            if "#include <math.h>" not in includes:
                includes.append("#include <math.h>")
        if any(isinstance(op, AttentionOp) for op in resolved_ops):
            if "#include <math.h>" not in includes:
                includes.append("#include <math.h>")
        if any(isinstance(op, BatchNormOp) for op in resolved_ops):
            if "#include <math.h>" not in includes:
                includes.append("#include <math.h>")
        if any(isinstance(op, LrnOp) for op in resolved_ops):
            if "#include <math.h>" not in includes:
                includes.append("#include <math.h>")
        if any(isinstance(op, SoftmaxOp) for op in resolved_ops):
            if "#include <math.h>" not in includes:
                includes.append("#include <math.h>")
        if any(
            isinstance(op, ReduceOp)
            and op.reduce_kind
            in {"l1", "l2", "logsum", "logsumexp"}
            for op in resolved_ops
        ):
            if "#include <math.h>" not in includes:
                includes.append("#include <math.h>")
        if any(
            isinstance(op, ReduceOp) and op.reduce_kind in {"min", "max"}
            for op in resolved_ops
        ):
            if any(
                op.dtype in {"int64", "int32", "int16", "int8"}
                for op in resolved_ops
                if isinstance(op, ReduceOp)
                and op.reduce_kind in {"min", "max"}
            ):
                if "#include <limits.h>" not in includes:
                    includes.append("#include <limits.h>")
            if any(
                op.dtype == "float"
                for op in resolved_ops
                if isinstance(op, ReduceOp)
                and op.reduce_kind in {"min", "max"}
            ):
                if "#include <math.h>" not in includes:
                    includes.append("#include <math.h>")
        if any(isinstance(op, MaxPoolOp) for op in resolved_ops):
            if any(
                op.dtype == "float"
                for op in resolved_ops
                if isinstance(op, MaxPoolOp)
            ):
                if "#include <math.h>" not in includes:
                    includes.append("#include <math.h>")
            if any(
                op.dtype in {"int64", "int32", "int16", "int8"}
                for op in resolved_ops
                if isinstance(op, MaxPoolOp)
            ):
                if "#include <limits.h>" not in includes:
                    includes.append("#include <limits.h>")
        if any(isinstance(op, ConcatOp) for op in resolved_ops):
            includes.append("#include <string.h>")
        if any(isinstance(op, ReshapeOp) for op in resolved_ops):
            if "#include <string.h>" not in includes:
                includes.append("#include <string.h>")
        sections = ["// Generated by emx-onnx2c (MVP)", *includes, ""]
        constants_section = self._emit_constants(model.constants)
        if constants_section:
            sections.extend((constants_section.rstrip(), ""))
        sections.extend(
            (
                operator_fns.rstrip(),
                "",
                wrapper_fn,
            )
        )
        if emit_testbench and testbench_template is not None:
            sections.extend(("", self._emit_testbench(model, testbench_template)))
        sections.append("")
        rendered = "\n".join(sections)
        if not rendered.endswith("\n"):
            rendered += "\n"
        return rendered

    def _emit_model_wrapper(
        self,
        model: LoweredModel,
        resolved_ops: list[
            BinaryOp
            | UnaryOp
            | MatMulOp
            | GemmOp
            | AttentionOp
            | ConvOp
            | AveragePoolOp
            | BatchNormOp
            | LrnOp
            | SoftmaxOp
            | MaxPoolOp
            | ConcatOp
            | ReshapeOp
            | ReduceOp
            | ConstantOfShapeOp
        ],
        temp_buffers: tuple[TempBuffer, ...],
    ) -> str:
        params = [
            f"const {dtype_info(dtype).c_type} {name}{self._array_suffix(shape)}"
            for name, shape, dtype in zip(
                model.input_names, model.input_shapes, model.input_dtypes
            )
        ]
        for name, shape, dtype in zip(
            model.output_names, model.output_shapes, model.output_dtypes
        ):
            output_type = dtype_info(dtype).c_type
            params.append(f"{output_type} {name}{self._array_suffix(shape)}")
        signature = ", ".join(params)
        lines = [f"void {model.name}({signature}) {{"]
        for temp in temp_buffers:
            c_type = dtype_info(temp.dtype).c_type
            lines.append(
                f"    {c_type} {temp.name}{self._array_suffix(temp.shape)};"
            )
        for index, op in enumerate(resolved_ops):
            if isinstance(op, (BinaryOp, MatMulOp)):
                call = f"{op.input0}, {op.input1}, {op.output}"
            elif isinstance(op, GemmOp):
                if op.input_c is None:
                    call = f"{op.input_a}, {op.input_b}, {op.output}"
                else:
                    call = (
                        f"{op.input_a}, {op.input_b}, {op.input_c}, {op.output}"
                    )
            elif isinstance(op, AttentionOp):
                call = (
                    f"{op.input_q}, {op.input_k}, {op.input_v}, {op.output}"
                )
            elif isinstance(op, ConvOp):
                if op.bias is None:
                    call = f"{op.input0}, {op.weights}, {op.output}"
                else:
                    call = f"{op.input0}, {op.weights}, {op.bias}, {op.output}"
            elif isinstance(op, AveragePoolOp):
                call = f"{op.input0}, {op.output}"
            elif isinstance(op, BatchNormOp):
                call = (
                    f"{op.input0}, {op.scale}, {op.bias}, "
                    f"{op.mean}, {op.variance}, {op.output}"
                )
            elif isinstance(op, SoftmaxOp):
                call = f"{op.input0}, {op.output}"
            elif isinstance(op, ConcatOp):
                call = ", ".join((*op.inputs, op.output))
            elif isinstance(op, ConstantOfShapeOp):
                call = f"{op.input0}, {op.output}"
            elif isinstance(op, ReshapeOp):
                call = f"{op.input0}, {op.output}"
            else:
                call = f"{op.input0}, {op.output}"
            lines.append(f"    {model.name}_op{index}({call});")
        lines.append("}")
        return "\n".join(lines)

    def _temp_buffers(self, model: LoweredModel) -> dict[str, TempBuffer]:
        output_names = set(model.output_names)
        intermediates = [
            (
                self._op_output(op),
                self._op_output_shape(op),
                self._op_output_dtype(op),
            )
            for op in model.ops
            if self._op_output(op) not in output_names
        ]
        if not intermediates:
            return {}
        if len(intermediates) == 1:
            name, shape, dtype = intermediates[0]
            return {name: TempBuffer(name="tmp", shape=shape, dtype=dtype)}
        return {
            name: TempBuffer(name=f"tmp{index}", shape=shape, dtype=dtype)
            for index, (name, shape, dtype) in enumerate(intermediates)
        }

    @staticmethod
    def _resolve_op(
        op: BinaryOp
        | UnaryOp
        | MatMulOp
        | GemmOp
        | AttentionOp
        | ConvOp
        | AveragePoolOp
        | BatchNormOp
        | LrnOp
        | SoftmaxOp
        | MaxPoolOp
        | ConcatOp
        | TransposeOp
        | ReshapeOp
        | ReduceOp
        | ConstantOfShapeOp,
        temp_map: dict[str, str],
    ) -> (
        BinaryOp
        | UnaryOp
        | MatMulOp
        | GemmOp
        | AttentionOp
        | ConvOp
        | AveragePoolOp
        | BatchNormOp
        | LrnOp
        | SoftmaxOp
        | MaxPoolOp
        | ConcatOp
        | TransposeOp
        | ReshapeOp
        | ReduceOp
        | ConstantOfShapeOp
    ):
        if isinstance(op, BinaryOp):
            return BinaryOp(
                input0=temp_map.get(op.input0, op.input0),
                input1=temp_map.get(op.input1, op.input1),
                output=temp_map.get(op.output, op.output),
                operator=op.operator,
                operator_kind=op.operator_kind,
                shape=op.shape,
                dtype=op.dtype,
            )
        if isinstance(op, MatMulOp):
            return MatMulOp(
                input0=temp_map.get(op.input0, op.input0),
                input1=temp_map.get(op.input1, op.input1),
                output=temp_map.get(op.output, op.output),
                m=op.m,
                n=op.n,
                k=op.k,
                dtype=op.dtype,
            )
        if isinstance(op, GemmOp):
            return GemmOp(
                input_a=temp_map.get(op.input_a, op.input_a),
                input_b=temp_map.get(op.input_b, op.input_b),
                input_c=(
                    temp_map.get(op.input_c, op.input_c)
                    if op.input_c is not None
                    else None
                ),
                output=temp_map.get(op.output, op.output),
                m=op.m,
                n=op.n,
                k=op.k,
                trans_a=op.trans_a,
                trans_b=op.trans_b,
                alpha=op.alpha,
                beta=op.beta,
                c_shape=op.c_shape,
                dtype=op.dtype,
            )
        if isinstance(op, AttentionOp):
            return AttentionOp(
                input_q=temp_map.get(op.input_q, op.input_q),
                input_k=temp_map.get(op.input_k, op.input_k),
                input_v=temp_map.get(op.input_v, op.input_v),
                output=temp_map.get(op.output, op.output),
                batch=op.batch,
                heads=op.heads,
                q_seq=op.q_seq,
                kv_seq=op.kv_seq,
                qk_head_size=op.qk_head_size,
                v_head_size=op.v_head_size,
                scale=op.scale,
                is_causal=op.is_causal,
                dtype=op.dtype,
            )
        if isinstance(op, ConvOp):
            return ConvOp(
                input0=temp_map.get(op.input0, op.input0),
                weights=temp_map.get(op.weights, op.weights),
                bias=temp_map.get(op.bias, op.bias) if op.bias else None,
                output=temp_map.get(op.output, op.output),
                batch=op.batch,
                in_channels=op.in_channels,
                out_channels=op.out_channels,
                spatial_rank=op.spatial_rank,
                in_spatial=op.in_spatial,
                out_spatial=op.out_spatial,
                kernel_shape=op.kernel_shape,
                strides=op.strides,
                pads=op.pads,
                dilations=op.dilations,
                group=op.group,
                dtype=op.dtype,
            )
        if isinstance(op, AveragePoolOp):
            return AveragePoolOp(
                input0=temp_map.get(op.input0, op.input0),
                output=temp_map.get(op.output, op.output),
                batch=op.batch,
                channels=op.channels,
                in_h=op.in_h,
                in_w=op.in_w,
                out_h=op.out_h,
                out_w=op.out_w,
                kernel_h=op.kernel_h,
                kernel_w=op.kernel_w,
                stride_h=op.stride_h,
                stride_w=op.stride_w,
                pad_top=op.pad_top,
                pad_left=op.pad_left,
                pad_bottom=op.pad_bottom,
                pad_right=op.pad_right,
                count_include_pad=op.count_include_pad,
                dtype=op.dtype,
            )
        if isinstance(op, BatchNormOp):
            return BatchNormOp(
                input0=temp_map.get(op.input0, op.input0),
                scale=temp_map.get(op.scale, op.scale),
                bias=temp_map.get(op.bias, op.bias),
                mean=temp_map.get(op.mean, op.mean),
                variance=temp_map.get(op.variance, op.variance),
                output=temp_map.get(op.output, op.output),
                shape=op.shape,
                channels=op.channels,
                epsilon=op.epsilon,
                dtype=op.dtype,
            )
        if isinstance(op, LrnOp):
            return LrnOp(
                input0=temp_map.get(op.input0, op.input0),
                output=temp_map.get(op.output, op.output),
                shape=op.shape,
                channels=op.channels,
                size=op.size,
                half=op.half,
                alpha=op.alpha,
                beta=op.beta,
                bias=op.bias,
                dtype=op.dtype,
            )
        if isinstance(op, SoftmaxOp):
            return SoftmaxOp(
                input0=temp_map.get(op.input0, op.input0),
                output=temp_map.get(op.output, op.output),
                outer=op.outer,
                axis_size=op.axis_size,
                inner=op.inner,
                axis=op.axis,
                shape=op.shape,
                dtype=op.dtype,
            )
        if isinstance(op, MaxPoolOp):
            return MaxPoolOp(
                input0=temp_map.get(op.input0, op.input0),
                output=temp_map.get(op.output, op.output),
                batch=op.batch,
                channels=op.channels,
                spatial_rank=op.spatial_rank,
                in_spatial=op.in_spatial,
                out_spatial=op.out_spatial,
                kernel_shape=op.kernel_shape,
                strides=op.strides,
                pads=op.pads,
                dilations=op.dilations,
                ceil_mode=op.ceil_mode,
                dtype=op.dtype,
            )
        if isinstance(op, ConcatOp):
            return ConcatOp(
                inputs=tuple(temp_map.get(name, name) for name in op.inputs),
                output=temp_map.get(op.output, op.output),
                axis=op.axis,
                input_shapes=op.input_shapes,
                output_shape=op.output_shape,
                dtype=op.dtype,
            )
        if isinstance(op, ConstantOfShapeOp):
            return ConstantOfShapeOp(
                input0=temp_map.get(op.input0, op.input0),
                output=temp_map.get(op.output, op.output),
                input_shape=op.input_shape,
                shape=op.shape,
                value=op.value,
                dtype=op.dtype,
                input_dtype=op.input_dtype,
            )
        if isinstance(op, TransposeOp):
            return TransposeOp(
                input0=temp_map.get(op.input0, op.input0),
                output=temp_map.get(op.output, op.output),
                perm=op.perm,
                input_shape=op.input_shape,
                output_shape=op.output_shape,
                dtype=op.dtype,
            )
        if isinstance(op, ReshapeOp):
            return ReshapeOp(
                input0=temp_map.get(op.input0, op.input0),
                output=temp_map.get(op.output, op.output),
                input_shape=op.input_shape,
                output_shape=op.output_shape,
                dtype=op.dtype,
            )
        if isinstance(op, ReduceOp):
            return ReduceOp(
                input0=temp_map.get(op.input0, op.input0),
                output=temp_map.get(op.output, op.output),
                input_shape=op.input_shape,
                output_shape=op.output_shape,
                axes=op.axes,
                keepdims=op.keepdims,
                reduce_kind=op.reduce_kind,
                reduce_count=op.reduce_count,
                dtype=op.dtype,
            )
        return UnaryOp(
            input0=temp_map.get(op.input0, op.input0),
            output=temp_map.get(op.output, op.output),
            operator=op.operator,
            shape=op.shape,
            dtype=op.dtype,
        )

    @staticmethod
    def _render_op(
        model: LoweredModel,
        op: BinaryOp
        | UnaryOp
        | MatMulOp
        | GemmOp
        | AttentionOp
        | ConvOp
        | AveragePoolOp
        | BatchNormOp
        | LrnOp
        | SoftmaxOp
        | MaxPoolOp
        | ConcatOp
        | TransposeOp
        | ReshapeOp
        | ReduceOp
        | ConstantOfShapeOp,
        index: int,
        *,
        array_suffix: str,
        loop_vars: tuple[str, ...],
        loop_indents: tuple[str, ...],
        inner_indent: str,
        c_type: str,
        zero_literal: str,
        min_literal: str,
        max_literal: str,
        binary_template,
        unary_template,
        matmul_template,
        gemm_template,
        attention_template,
        conv_template,
        avg_pool_template,
        batch_norm_template,
        lrn_template,
        softmax_template,
        maxpool_template,
        concat_template,
        transpose_template,
        reshape_template,
        reduce_template,
        constant_of_shape_template,
    ) -> str:
        if isinstance(op, BinaryOp):
            shape = op.shape
            loop_vars = CEmitter._loop_vars(shape)
            loop_indents = CEmitter._loop_indents(shape)
            inner_indent = CEmitter._inner_indent(shape)
            array_suffix = CEmitter._array_suffix(shape)
            common = {
                "model_name": model.name,
                "op_name": f"{model.name}_op{index}",
                "element_count": CEmitter._element_count(shape),
                "array_suffix": array_suffix,
                "shape": shape,
                "loop_vars": loop_vars,
                "loop_indents": loop_indents,
                "inner_indent": inner_indent,
                "c_type": c_type,
                "zero_literal": zero_literal,
            }
            left_expr = f"{op.input0}" + "".join(
                f"[{var}]" for var in loop_vars
            )
            right_expr = f"{op.input1}" + "".join(
                f"[{var}]" for var in loop_vars
            )
            operator_expr = None
            if op.operator_kind == "expr":
                operator_expr = op.operator.format(
                    left=left_expr, right=right_expr
                )
            return binary_template.render(
                **common,
                input0=op.input0,
                input1=op.input1,
                output=op.output,
                operator=op.operator,
                operator_kind=op.operator_kind,
                left_expr=left_expr,
                right_expr=right_expr,
                operator_expr=operator_expr,
            ).rstrip()
        if isinstance(op, MatMulOp):
            return matmul_template.render(
                model_name=model.name,
                op_name=f"{model.name}_op{index}",
                input0=op.input0,
                input1=op.input1,
                output=op.output,
                c_type=c_type,
                acc_type=c_type,
                zero_literal=zero_literal,
                input0_suffix=CEmitter._array_suffix((op.m, op.k)),
                input1_suffix=CEmitter._array_suffix((op.k, op.n)),
                output_suffix=CEmitter._array_suffix((op.m, op.n)),
                m=op.m,
                n=op.n,
                k=op.k,
            ).rstrip()
        if isinstance(op, GemmOp):
            input_a_shape = (op.k, op.m) if op.trans_a else (op.m, op.k)
            input_b_shape = (op.n, op.k) if op.trans_b else (op.k, op.n)
            alpha_literal = CEmitter._format_literal(op.dtype, op.alpha)
            beta_literal = CEmitter._format_literal(op.dtype, op.beta)
            if op.c_shape is None:
                c_rank = 0
                c_dim0 = 0
                c_dim1 = 0
            elif len(op.c_shape) == 1:
                c_rank = 1
                c_dim0 = 1
                c_dim1 = op.c_shape[0]
            else:
                c_rank = 2
                c_dim0 = op.c_shape[0]
                c_dim1 = op.c_shape[1]
            return gemm_template.render(
                model_name=model.name,
                op_name=f"{model.name}_op{index}",
                input_a=op.input_a,
                input_b=op.input_b,
                input_c=op.input_c,
                output=op.output,
                c_type=c_type,
                acc_type=c_type,
                zero_literal=zero_literal,
                alpha_literal=alpha_literal,
                beta_literal=beta_literal,
                trans_a=int(op.trans_a),
                trans_b=int(op.trans_b),
                m=op.m,
                n=op.n,
                k=op.k,
                input_a_suffix=CEmitter._array_suffix(input_a_shape),
                input_b_suffix=CEmitter._array_suffix(input_b_shape),
                output_suffix=CEmitter._array_suffix((op.m, op.n)),
                c_suffix=(
                    CEmitter._array_suffix(op.c_shape)
                    if op.c_shape is not None
                    else None
                ),
                c_rank=c_rank,
                c_dim0=c_dim0,
                c_dim1=c_dim1,
            ).rstrip()
        if isinstance(op, AttentionOp):
            return attention_template.render(
                model_name=model.name,
                op_name=f"{model.name}_op{index}",
                input_q=op.input_q,
                input_k=op.input_k,
                input_v=op.input_v,
                output=op.output,
                c_type=c_type,
                zero_literal=zero_literal,
                scale_literal=CEmitter._format_float(op.scale),
                is_causal=int(op.is_causal),
                batch=op.batch,
                heads=op.heads,
                q_seq=op.q_seq,
                kv_seq=op.kv_seq,
                qk_head_size=op.qk_head_size,
                v_head_size=op.v_head_size,
                input_q_suffix=CEmitter._array_suffix(
                    (op.batch, op.heads, op.q_seq, op.qk_head_size)
                ),
                input_k_suffix=CEmitter._array_suffix(
                    (op.batch, op.heads, op.kv_seq, op.qk_head_size)
                ),
                input_v_suffix=CEmitter._array_suffix(
                    (op.batch, op.heads, op.kv_seq, op.v_head_size)
                ),
                output_suffix=CEmitter._array_suffix(
                    (op.batch, op.heads, op.q_seq, op.v_head_size)
                ),
            ).rstrip()
        if isinstance(op, ConvOp):
            input_shape = (op.batch, op.in_channels, *op.in_spatial)
            weight_shape = (
                op.out_channels,
                op.in_channels // op.group,
                *op.kernel_shape,
            )
            output_shape = (op.batch, op.out_channels, *op.out_spatial)
            out_indices = tuple(f"od{dim}" for dim in range(op.spatial_rank))
            kernel_indices = tuple(
                f"kd{dim}" for dim in range(op.spatial_rank)
            )
            in_indices = tuple(f"id{dim}" for dim in range(op.spatial_rank))
            pad_begin = op.pads[: op.spatial_rank]
            group_in_channels = op.in_channels // op.group
            group_out_channels = op.out_channels // op.group
            return conv_template.render(
                model_name=model.name,
                op_name=f"{model.name}_op{index}",
                input0=op.input0,
                weights=op.weights,
                bias=op.bias,
                output=op.output,
                c_type=c_type,
                zero_literal=zero_literal,
                input_suffix=CEmitter._array_suffix(input_shape),
                weight_suffix=CEmitter._array_suffix(weight_shape),
                bias_suffix=CEmitter._array_suffix((op.out_channels,)),
                output_suffix=CEmitter._array_suffix(output_shape),
                batch=op.batch,
                in_channels=op.in_channels,
                out_channels=op.out_channels,
                spatial_rank=op.spatial_rank,
                in_spatial=op.in_spatial,
                out_spatial=op.out_spatial,
                kernel_shape=op.kernel_shape,
                strides=op.strides,
                pads_begin=pad_begin,
                dilations=op.dilations,
                group=op.group,
                group_in_channels=group_in_channels,
                group_out_channels=group_out_channels,
                out_indices=out_indices,
                kernel_indices=kernel_indices,
                in_indices=in_indices,
            ).rstrip()
        if isinstance(op, AveragePoolOp):
            input_shape = (op.batch, op.channels, op.in_h, op.in_w)
            output_shape = (op.batch, op.channels, op.out_h, op.out_w)
            return avg_pool_template.render(
                model_name=model.name,
                op_name=f"{model.name}_op{index}",
                input0=op.input0,
                output=op.output,
                c_type=c_type,
                zero_literal=zero_literal,
                input_suffix=CEmitter._array_suffix(input_shape),
                output_suffix=CEmitter._array_suffix(output_shape),
                batch=op.batch,
                channels=op.channels,
                in_h=op.in_h,
                in_w=op.in_w,
                out_h=op.out_h,
                out_w=op.out_w,
                kernel_h=op.kernel_h,
                kernel_w=op.kernel_w,
                stride_h=op.stride_h,
                stride_w=op.stride_w,
                pad_top=op.pad_top,
                pad_left=op.pad_left,
                pad_bottom=op.pad_bottom,
                pad_right=op.pad_right,
                count_include_pad=int(op.count_include_pad),
            ).rstrip()
        if isinstance(op, BatchNormOp):
            shape = op.shape
            loop_vars = CEmitter._loop_vars(shape)
            loop_indents = CEmitter._loop_indents(shape)
            inner_indent = CEmitter._inner_indent(shape)
            return batch_norm_template.render(
                model_name=model.name,
                op_name=f"{model.name}_op{index}",
                input0=op.input0,
                scale=op.scale,
                bias=op.bias,
                mean=op.mean,
                variance=op.variance,
                output=op.output,
                c_type=c_type,
                input_suffix=CEmitter._array_suffix(shape),
                output_suffix=CEmitter._array_suffix(shape),
                scale_suffix=CEmitter._array_suffix((op.channels,)),
                bias_suffix=CEmitter._array_suffix((op.channels,)),
                mean_suffix=CEmitter._array_suffix((op.channels,)),
                variance_suffix=CEmitter._array_suffix((op.channels,)),
                shape=shape,
                loop_vars=loop_vars,
                loop_indents=loop_indents,
                inner_indent=inner_indent,
                epsilon_literal=CEmitter._format_float(op.epsilon),
            ).rstrip()
        if isinstance(op, LrnOp):
            shape = op.shape
            loop_vars = CEmitter._loop_vars(shape)
            loop_indents = CEmitter._loop_indents(shape)
            inner_indent = CEmitter._inner_indent(shape)
            return lrn_template.render(
                model_name=model.name,
                op_name=f"{model.name}_op{index}",
                input0=op.input0,
                output=op.output,
                c_type=c_type,
                input_suffix=CEmitter._array_suffix(shape),
                output_suffix=CEmitter._array_suffix(shape),
                shape=shape,
                channels=op.channels,
                half=op.half,
                loop_vars=loop_vars,
                loop_indents=loop_indents,
                inner_indent=inner_indent,
                zero_literal=zero_literal,
                alpha_div_size_literal=CEmitter._format_float(op.alpha / op.size),
                beta_literal=CEmitter._format_float(op.beta),
                bias_literal=CEmitter._format_float(op.bias),
            ).rstrip()
        if isinstance(op, SoftmaxOp):
            return softmax_template.render(
                model_name=model.name,
                op_name=f"{model.name}_op{index}",
                input0=op.input0,
                output=op.output,
                c_type=c_type,
                array_suffix=CEmitter._array_suffix(op.shape),
                outer=op.outer,
                axis_size=op.axis_size,
                inner=op.inner,
            ).rstrip()
        if isinstance(op, MaxPoolOp):
            input_shape = (op.batch, op.channels, *op.in_spatial)
            output_shape = (op.batch, op.channels, *op.out_spatial)
            return maxpool_template.render(
                model_name=model.name,
                op_name=f"{model.name}_op{index}",
                input0=op.input0,
                output=op.output,
                c_type=c_type,
                min_literal=min_literal,
                input_suffix=CEmitter._array_suffix(input_shape),
                output_suffix=CEmitter._array_suffix(output_shape),
                batch=op.batch,
                channels=op.channels,
                spatial_rank=op.spatial_rank,
                in_spatial=op.in_spatial,
                out_spatial=op.out_spatial,
                kernel_shape=op.kernel_shape,
                strides=op.strides,
                pads=op.pads,
                dilations=op.dilations,
                ceil_mode=int(op.ceil_mode),
            ).rstrip()
        if isinstance(op, ConcatOp):
            axis = op.axis
            if axis < 0:
                axis += len(op.output_shape)
            outer = CEmitter._element_count(op.output_shape[:axis] or (1,))
            inner = CEmitter._element_count(op.output_shape[axis + 1 :] or (1,))
            axis_sizes = tuple(shape[axis] for shape in op.input_shapes)
            return concat_template.render(
                model_name=model.name,
                op_name=f"{model.name}_op{index}",
                inputs=op.inputs,
                output=op.output,
                c_type=c_type,
                input_suffixes=tuple(
                    CEmitter._array_suffix(shape) for shape in op.input_shapes
                ),
                output_suffix=CEmitter._array_suffix(op.output_shape),
                axis_sizes=axis_sizes,
                input_count=len(op.inputs),
                outer=outer,
                inner=inner,
            ).rstrip()
        if isinstance(op, TransposeOp):
            output_shape = op.output_shape
            loop_vars = CEmitter._loop_vars(output_shape)
            loop_indents = CEmitter._loop_indents(output_shape)
            inner_indent = CEmitter._inner_indent(output_shape)
            output_suffix = CEmitter._array_suffix(output_shape)
            input_suffix = CEmitter._array_suffix(op.input_shape)
            input_indices = [None] * len(op.perm)
            for output_axis, input_axis in enumerate(op.perm):
                input_indices[input_axis] = loop_vars[output_axis]
            return transpose_template.render(
                model_name=model.name,
                op_name=f"{model.name}_op{index}",
                input0=op.input0,
                output=op.output,
                c_type=c_type,
                input_suffix=input_suffix,
                output_suffix=output_suffix,
                output_shape=output_shape,
                loop_vars=loop_vars,
                loop_indents=loop_indents,
                inner_indent=inner_indent,
                input_indices=input_indices,
            ).rstrip()
        if isinstance(op, ReshapeOp):
            return reshape_template.render(
                model_name=model.name,
                op_name=f"{model.name}_op{index}",
                input0=op.input0,
                output=op.output,
                c_type=c_type,
                input_suffix=CEmitter._array_suffix(op.input_shape),
                output_suffix=CEmitter._array_suffix(op.output_shape),
                element_count=CEmitter._element_count(op.output_shape),
            ).rstrip()
        if isinstance(op, ReduceOp):
            output_shape = op.output_shape
            output_loop_vars = CEmitter._loop_vars(output_shape)
            output_loop_indents = CEmitter._loop_indents(output_shape)
            output_inner_indent = CEmitter._inner_indent(output_shape)
            reduce_loop_vars = tuple(f"r{idx}" for idx in range(len(op.axes)))
            reduce_dims = tuple(op.input_shape[axis] for axis in op.axes)
            reduce_loop_indents = tuple(
                output_inner_indent + "    " * idx
                for idx in range(len(reduce_loop_vars))
            )
            reduce_inner_indent = output_inner_indent + "    " * len(
                reduce_loop_vars
            )
            if op.keepdims:
                input_indices = [
                    reduce_loop_vars[op.axes.index(axis)]
                    if axis in op.axes
                    else output_loop_vars[axis]
                    for axis in range(len(op.input_shape))
                ]
            else:
                kept_axes = [
                    axis
                    for axis in range(len(op.input_shape))
                    if axis not in op.axes
                ]
                input_indices = [
                    reduce_loop_vars[op.axes.index(axis)]
                    if axis in op.axes
                    else output_loop_vars[kept_axes.index(axis)]
                    for axis in range(len(op.input_shape))
                ]
            input_index_expr = "".join(f"[{var}]" for var in input_indices)
            output_index_expr = "".join(
                f"[{var}]" for var in output_loop_vars
            )
            value_expr = f"{op.input0}{input_index_expr}"
            update_expr = None
            init_literal = None
            final_expr = "acc"
            count_literal = CEmitter._format_literal(
                op.dtype, op.reduce_count
            )
            if op.reduce_kind == "sum":
                init_literal = zero_literal
                update_expr = f"acc += {value_expr};"
            elif op.reduce_kind == "mean":
                init_literal = zero_literal
                update_expr = f"acc += {value_expr};"
                final_expr = f"acc / {count_literal}"
            elif op.reduce_kind == "max":
                init_literal = min_literal
                update_expr = f"if ({value_expr} > acc) acc = {value_expr};"
            elif op.reduce_kind == "min":
                init_literal = max_literal
                update_expr = f"if ({value_expr} < acc) acc = {value_expr};"
            elif op.reduce_kind == "prod":
                init_literal = CEmitter._format_literal(op.dtype, 1)
                update_expr = f"acc *= {value_expr};"
            elif op.reduce_kind == "l1":
                init_literal = zero_literal
                update_expr = f"acc += fabsf({value_expr});"
            elif op.reduce_kind == "l2":
                init_literal = zero_literal
                update_expr = f"acc += {value_expr} * {value_expr};"
                final_expr = "sqrtf(acc)"
            elif op.reduce_kind == "logsum":
                init_literal = zero_literal
                update_expr = f"acc += {value_expr};"
                final_expr = "logf(acc)"
            elif op.reduce_kind == "logsumexp":
                init_literal = zero_literal
                update_expr = f"acc += expf({value_expr});"
                final_expr = "logf(acc)"
            elif op.reduce_kind == "sumsquare":
                init_literal = zero_literal
                update_expr = f"acc += {value_expr} * {value_expr};"
            else:
                raise CodegenError(
                    f"Unsupported reduce kind {op.reduce_kind}"
                )
            return reduce_template.render(
                model_name=model.name,
                op_name=f"{model.name}_op{index}",
                input0=op.input0,
                output=op.output,
                c_type=c_type,
                input_suffix=CEmitter._array_suffix(op.input_shape),
                output_suffix=CEmitter._array_suffix(op.output_shape),
                output_shape=output_shape,
                output_loop_vars=output_loop_vars,
                output_loop_indents=output_loop_indents,
                output_inner_indent=output_inner_indent,
                reduce_loop_vars=reduce_loop_vars,
                reduce_dims=reduce_dims,
                reduce_loop_indents=reduce_loop_indents,
                reduce_inner_indent=reduce_inner_indent,
                output_index_expr=output_index_expr,
                init_literal=init_literal,
                update_expr=update_expr,
                final_expr=final_expr,
            ).rstrip()
        if isinstance(op, ConstantOfShapeOp):
            shape = op.shape
            loop_vars = CEmitter._loop_vars(shape)
            loop_indents = CEmitter._loop_indents(shape)
            inner_indent = CEmitter._inner_indent(shape)
            array_suffix = CEmitter._array_suffix(shape)
            return constant_of_shape_template.render(
                model_name=model.name,
                op_name=f"{model.name}_op{index}",
                input0=op.input0,
                output=op.output,
                input_c_type=dtype_info(op.input_dtype).c_type,
                c_type=c_type,
                input_suffix=CEmitter._array_suffix(op.input_shape),
                array_suffix=array_suffix,
                shape=shape,
                loop_vars=loop_vars,
                loop_indents=loop_indents,
                inner_indent=inner_indent,
                value_literal=CEmitter._format_literal(op.dtype, op.value),
            ).rstrip()
        shape = op.shape
        loop_vars = CEmitter._loop_vars(shape)
        loop_indents = CEmitter._loop_indents(shape)
        inner_indent = CEmitter._inner_indent(shape)
        array_suffix = CEmitter._array_suffix(shape)
        common = {
            "model_name": model.name,
            "op_name": f"{model.name}_op{index}",
            "element_count": CEmitter._element_count(shape),
            "array_suffix": array_suffix,
            "shape": shape,
            "loop_vars": loop_vars,
            "loop_indents": loop_indents,
            "inner_indent": inner_indent,
            "c_type": c_type,
            "zero_literal": zero_literal,
        }
        return unary_template.render(
            **common,
            input0=op.input0,
            output=op.output,
            operator=op.operator,
        ).rstrip()

    @staticmethod
    def _op_output(
        op: BinaryOp
        | UnaryOp
        | MatMulOp
        | GemmOp
        | AttentionOp
        | ConvOp
        | AveragePoolOp
        | BatchNormOp
        | LrnOp
        | SoftmaxOp
        | MaxPoolOp
        | ConcatOp
        | TransposeOp
        | ReshapeOp
        | ReduceOp
        | ConstantOfShapeOp,
    ) -> str:
        return op.output

    @staticmethod
    def _op_output_shape(
        op: BinaryOp
        | UnaryOp
        | MatMulOp
        | GemmOp
        | AttentionOp
        | ConvOp
        | AveragePoolOp
        | BatchNormOp
        | LrnOp
        | SoftmaxOp
        | MaxPoolOp
        | ConcatOp
        | TransposeOp
        | ReshapeOp
        | ReduceOp
        | ConstantOfShapeOp,
    ) -> tuple[int, ...]:
        if isinstance(op, BinaryOp):
            return op.shape
        if isinstance(op, UnaryOp):
            return op.shape
        if isinstance(op, MatMulOp):
            return (op.m, op.n)
        if isinstance(op, GemmOp):
            return (op.m, op.n)
        if isinstance(op, ConvOp):
            return (op.batch, op.out_channels, *op.out_spatial)
        if isinstance(op, AveragePoolOp):
            return (op.batch, op.channels, op.out_h, op.out_w)
        if isinstance(op, BatchNormOp):
            return op.shape
        if isinstance(op, LrnOp):
            return op.shape
        if isinstance(op, SoftmaxOp):
            return op.shape
        if isinstance(op, MaxPoolOp):
            return (op.batch, op.channels, *op.out_spatial)
        if isinstance(op, ConcatOp):
            return op.output_shape
        if isinstance(op, TransposeOp):
            return op.output_shape
        if isinstance(op, ReshapeOp):
            return op.output_shape
        if isinstance(op, ReduceOp):
            return op.output_shape
        if isinstance(op, ConstantOfShapeOp):
            return op.shape
        return (op.batch, op.heads, op.q_seq, op.v_head_size)

    @staticmethod
    def _op_output_dtype(
        op: BinaryOp
        | UnaryOp
        | MatMulOp
        | GemmOp
        | AttentionOp
        | ConvOp
        | AveragePoolOp
        | BatchNormOp
        | SoftmaxOp
        | MaxPoolOp
        | ConcatOp
        | TransposeOp
        | ReshapeOp
        | ReduceOp
        | ConstantOfShapeOp,
    ) -> str:
        return op.dtype

    @staticmethod
    def _array_suffix(shape: tuple[int, ...]) -> str:
        if not shape:
            raise CodegenError("Scalar outputs are not supported")
        return "".join(f"[{dim}]" for dim in shape)

    @staticmethod
    def _loop_vars(shape: tuple[int, ...]) -> tuple[str, ...]:
        if not shape:
            raise CodegenError("Scalar outputs are not supported")
        return tuple(f"i{index}" for index in range(len(shape)))

    @staticmethod
    def _loop_indents(shape: tuple[int, ...]) -> tuple[str, ...]:
        if not shape:
            raise CodegenError("Scalar outputs are not supported")
        return tuple("    " * (index + 1) for index in range(len(shape)))

    @staticmethod
    def _inner_indent(shape: tuple[int, ...]) -> str:
        if not shape:
            raise CodegenError("Scalar outputs are not supported")
        return "    " * (len(shape) + 1)

    @staticmethod
    def _element_count(shape: tuple[int, ...]) -> int:
        count = 1
        for dim in shape:
            if dim <= 0:
                raise CodegenError("Dynamic or zero dims are not supported")
            count *= dim
        return count

    def _emit_testbench(self, model: LoweredModel, testbench_template) -> str:
        input_counts = tuple(
            self._element_count(shape) for shape in model.input_shapes
        )
        inputs = []
        for name, shape, count, dtype in zip(
            model.input_names, model.input_shapes, input_counts, model.input_dtypes
        ):
            info = dtype_info(dtype)
            loop_vars = self._loop_vars(shape)
            if dtype == "float":
                random_expr = "rng_next_float()"
            elif dtype == "bool":
                random_expr = "((rng_next_u64() & 1ull) != 0)"
            else:
                random_expr = f"({info.c_type})rng_next_i64()"
            inputs.append(
                {
                    "name": name,
                    "shape": shape,
                    "shape_literal": ",".join(str(dim) for dim in shape),
                    "count": count,
                    "array_suffix": self._array_suffix(shape),
                    "loop_vars": loop_vars,
                    "loop_indents": self._loop_indents(shape),
                    "inner_indent": self._inner_indent(shape),
                    "rank": len(shape),
                    "index_expr": self._index_expr(shape, loop_vars),
                    "dtype": dtype,
                    "c_type": info.c_type,
                    "random_expr": random_expr,
                    "print_format": self._print_format(dtype),
                    "print_cast": self._print_cast(dtype),
                }
            )
        outputs = []
        for name, shape, dtype in zip(
            model.output_names, model.output_shapes, model.output_dtypes
        ):
            output_info = dtype_info(dtype)
            output_loop_vars = self._loop_vars(shape)
            outputs.append(
                {
                    "name": name,
                    "shape": shape,
                    "shape_literal": ",".join(str(dim) for dim in shape),
                    "count": self._element_count(shape),
                    "array_suffix": self._array_suffix(shape),
                    "loop_vars": output_loop_vars,
                    "loop_indents": self._loop_indents(shape),
                    "inner_indent": self._inner_indent(shape),
                    "rank": len(shape),
                    "index_expr": self._index_expr(shape, output_loop_vars),
                    "dtype": dtype,
                    "c_type": output_info.c_type,
                    "print_format": self._print_format(dtype),
                    "print_cast": self._print_cast(dtype),
                }
            )
        return testbench_template.render(
            model_name=model.name,
            inputs=inputs,
            outputs=outputs,
        ).rstrip()

    def _emit_constants(self, constants: tuple[ConstTensor, ...]) -> str:
        if not constants:
            return ""
        lines: list[str] = []
        for const in constants:
            c_type = dtype_info(const.dtype).c_type
            array_suffix = self._array_suffix(const.shape)
            values = [
                self._format_value(value, const.dtype) for value in const.data
            ]
            lines.append(f"static const {c_type} {const.name}{array_suffix} = {{")
            if values:
                chunk_size = 8
                chunks = [
                    values[index : index + chunk_size]
                    for index in range(0, len(values), chunk_size)
                ]
                for chunk_index, chunk in enumerate(chunks):
                    line = "    " + ", ".join(chunk)
                    if chunk_index != len(chunks) - 1:
                        line += ","
                    lines.append(line)
            lines.append("};")
            lines.append("")
        if lines and not lines[-1]:
            lines.pop()
        return "\n".join(lines)

    @staticmethod
    def _index_expr(shape: tuple[int, ...], loop_vars: tuple[str, ...]) -> str:
        if len(shape) != len(loop_vars):
            raise CodegenError("Loop variables must match shape rank")
        if not shape:
            raise CodegenError("Scalar outputs are not supported")
        expr = loop_vars[0]
        for dim, var in zip(shape[1:], loop_vars[1:]):
            expr = f"({expr} * {dim} + {var})"
        return expr

    @staticmethod
    def _format_float(value: float) -> str:
        formatted = f"{value:.9g}"
        if "e" not in formatted and "E" not in formatted and "." not in formatted:
            formatted = f"{formatted}.0"
        return f"{formatted}f"

    @staticmethod
    def _format_int64(value: int) -> str:
        min_value = -(2**63)
        if value == min_value:
            return "INT64_MIN"
        return f"{int(value)}LL"

    @staticmethod
    def _format_int(value: int, bits: int, min_macro: str) -> str:
        min_value = -(2 ** (bits - 1))
        if value == min_value:
            return min_macro
        return str(int(value))

    @staticmethod
    def _format_literal(dtype: str, value: float | int | bool) -> str:
        if dtype == "float":
            return CEmitter._format_float(float(value))
        if dtype == "bool":
            return "true" if bool(value) else "false"
        if dtype == "int64":
            return CEmitter._format_int64(int(value))
        if dtype == "int32":
            return CEmitter._format_int(int(value), 32, "INT32_MIN")
        if dtype == "int16":
            return CEmitter._format_int(int(value), 16, "INT16_MIN")
        if dtype == "int8":
            return CEmitter._format_int(int(value), 8, "INT8_MIN")
        raise CodegenError(f"Unsupported dtype {dtype}")

    def _format_value(self, value: float | int | bool, dtype: str) -> str:
        if dtype == "float":
            return self._format_float(float(value))
        if dtype == "bool":
            return "true" if bool(value) else "false"
        if dtype == "int64":
            return self._format_int64(int(value))
        if dtype == "int32":
            return self._format_int(int(value), 32, "INT32_MIN")
        if dtype == "int16":
            return self._format_int(int(value), 16, "INT16_MIN")
        if dtype == "int8":
            return self._format_int(int(value), 8, "INT8_MIN")
        raise CodegenError(f"Unsupported dtype {dtype}")

    @staticmethod
    def _print_format(dtype: str) -> str:
        if dtype == "float":
            return "%.8g"
        if dtype == "bool":
            return "%d"
        if dtype == "int64":
            return "%lld"
        if dtype == "int32":
            return "%d"
        if dtype == "int16":
            return "%hd"
        if dtype == "int8":
            return "%hhd"
        raise CodegenError(f"Unsupported dtype {dtype}")

    @staticmethod
    def _print_cast(dtype: str) -> str:
        if dtype == "float":
            return "(double)"
        if dtype == "bool":
            return "(int)"
        if dtype == "int64":
            return "(long long)"
        if dtype in {"int32", "int16", "int8"}:
            return "(int)"
        raise CodegenError(f"Unsupported dtype {dtype}")

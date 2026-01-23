from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from math import prod
from pathlib import Path
import re
import struct
from typing import Mapping, Sequence

from jinja2 import Environment, FileSystemLoader, Template, select_autoescape
import numpy as np

from ..errors import CodegenError
from ..ops import (
    COMPARE_FUNCTIONS,
    OperatorKind,
    binary_op_symbol,
    unary_op_symbol,
)
from ..ir.op_base import (
    BroadcastingOpBase,
    ConvLikeOpBase,
    ElementwiseOpBase,
    GemmLikeOpBase,
    MatMulLikeOpBase,
    ReduceOpBase,
    RenderableOpBase,
    OpBase,
    EmitContext,
)
from ..ir.ops import (
    AdagradOp,
    ArgReduceOp,
    AttentionOp,
    AveragePoolOp,
    BatchNormOp,
    BinaryOp,
    CastOp,
    ClipOp,
    ConcatOp,
    ConstantOfShapeOp,
    ConvOp,
    ConvTransposeOp,
    CumSumOp,
    DepthToSpaceOp,
    EinsumKind,
    EinsumOp,
    ExpandOp,
    EyeLikeOp,
    GatherElementsOp,
    GatherNDOp,
    GatherOp,
    GemmOp,
    GridSampleOp,
    GroupNormalizationOp,
    HardmaxOp,
    IdentityOp,
    InstanceNormalizationOp,
    LayerNormalizationOp,
    LogSoftmaxOp,
    LpNormalizationOp,
    LpPoolOp,
    LrnOp,
    LstmOp,
    MatMulOp,
    MaxPoolOp,
    MeanVarianceNormalizationOp,
    MultiInputBinaryOp,
    NegativeLogLikelihoodLossOp,
    NonMaxSuppressionOp,
    NonZeroOp,
    OneHotOp,
    PadOp,
    QuantizeLinearOp,
    QLinearMatMulOp,
    RangeOp,
    ReduceOp,
    ReshapeOp,
    ResizeOp,
    RMSNormalizationOp,
    RotaryEmbeddingOp,
    ScatterNDOp,
    ShapeOp,
    SizeOp,
    SliceOp,
    SoftmaxCrossEntropyLossOp,
    SoftmaxOp,
    SpaceToDepthOp,
    SplitOp,
    TensorScatterOp,
    TileOp,
    TopKOp,
    TransposeOp,
    TriluOp,
    UnaryOp,
    WhereOp,
)
from shared.scalar_functions import (
    ScalarFunction,
    ScalarFunctionKey,
    ScalarFunctionRegistry,
)
from shared.scalar_types import ScalarFunctionError, ScalarType


def _format_c_indentation(source: str, *, indent: str = "    ") -> str:
    def strip_string_literals(line: str) -> str:
        sanitized: list[str] = []
        in_string = False
        in_char = False
        escape = False
        for char in line:
            if escape:
                escape = False
                if not (in_string or in_char):
                    sanitized.append(char)
                continue
            if in_string:
                if char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if in_char:
                if char == "\\":
                    escape = True
                elif char == "'":
                    in_char = False
                continue
            if char == '"':
                in_string = True
                continue
            if char == "'":
                in_char = True
                continue
            sanitized.append(char)
        return "".join(sanitized)

    formatted_lines: list[str] = []
    indent_level = 0
    for line in source.splitlines():
        stripped = line.lstrip()
        if not stripped:
            formatted_lines.append("")
            continue
        if stripped.startswith("}"):
            indent_level = max(indent_level - 1, 0)
        formatted_lines.append(f"{indent * indent_level}{stripped}")
        sanitized = strip_string_literals(stripped)
        open_count = sanitized.count("{")
        close_count = sanitized.count("}")
        if stripped.startswith("}"):
            close_count = max(close_count - 1, 0)
        indent_level += open_count - close_count
        indent_level = max(indent_level, 0)
    return "\n".join(formatted_lines)


_SCALAR_TYPE_BY_DTYPE: dict[str, ScalarType] = {
    "float": ScalarType.F32,
    "double": ScalarType.F64,
    "int8": ScalarType.I8,
    "int16": ScalarType.I16,
    "int32": ScalarType.I32,
    "int64": ScalarType.I64,
    "uint8": ScalarType.U8,
    "uint16": ScalarType.U16,
    "uint32": ScalarType.U32,
    "uint64": ScalarType.U64,
    "bool": ScalarType.BOOL,
}

_LSTM_ACTIVATION_SPECS: dict[int, tuple[ScalarFunction, int]] = {
    0: (ScalarFunction.RELU, 0),
    1: (ScalarFunction.TANH, 0),
    2: (ScalarFunction.SIGMOID, 0),
    3: (ScalarFunction.AFFINE, 2),
    4: (ScalarFunction.LEAKY_RELU, 1),
    5: (ScalarFunction.THRESHOLDED_RELU, 1),
    6: (ScalarFunction.SCALED_TANH, 2),
    7: (ScalarFunction.HARDSIGMOID, 2),
    8: (ScalarFunction.ELU, 1),
    9: (ScalarFunction.SOFTSIGN, 0),
    10: (ScalarFunction.SOFTPLUS, 0),
}

_C_IDENTIFIER_RE = re.compile(r"[^a-zA-Z0-9_]")
_C_KEYWORDS = {
    "_Bool",
    "_Complex",
    "_Imaginary",
    "auto",
    "break",
    "case",
    "char",
    "const",
    "continue",
    "default",
    "do",
    "double",
    "else",
    "enum",
    "extern",
    "float",
    "for",
    "goto",
    "if",
    "inline",
    "int",
    "long",
    "register",
    "restrict",
    "return",
    "short",
    "signed",
    "sizeof",
    "static",
    "struct",
    "switch",
    "typedef",
    "union",
    "unsigned",
    "void",
    "volatile",
    "while",
}

@dataclass(frozen=True)
class NodeInfo:
    op_type: str
    name: str | None
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    attrs: dict[str, object]


@dataclass(frozen=True)
class ConstTensor:
    name: str
    shape: tuple[int, ...]
    data: tuple[float | int | bool, ...]
    dtype: ScalarType


@dataclass(frozen=True)
class TempBuffer:
    name: str
    shape: tuple[int, ...]
    dtype: ScalarType


@dataclass(frozen=True)
class ModelHeader:
    generator: str
    command_line: str | None
    model_checksum: str | None
    model_name: str | None
    graph_name: str | None
    description: str | None
    graph_description: str | None
    producer_name: str | None
    producer_version: str | None
    domain: str | None
    model_version: int | None
    ir_version: int | None
    opset_imports: tuple[tuple[str, int], ...]
    metadata_props: tuple[tuple[str, str], ...]
    input_count: int
    output_count: int
    node_count: int
    initializer_count: int


@dataclass(frozen=True)
class LoweredModel:
    name: str
    input_names: tuple[str, ...]
    input_shapes: tuple[tuple[int, ...], ...]
    input_dtypes: tuple[ScalarType, ...]
    output_names: tuple[str, ...]
    output_shapes: tuple[tuple[int, ...], ...]
    output_dtypes: tuple[ScalarType, ...]
    constants: tuple[ConstTensor, ...]
    ops: tuple[OpBase, ...]
    node_infos: tuple[NodeInfo, ...]
    header: ModelHeader


@dataclass
class _EmitState:
    model: LoweredModel
    templates: dict[str, Template]
    scalar_registry: ScalarFunctionRegistry
    dim_args: str
    tensor_dim_names: Mapping[str, Mapping[int, str]]


class CEmitter:
    def __init__(
        self,
        template_dir: Path,
        *,
        restrict_arrays: bool = True,
        truncate_weights_after: int | None = None,
        large_temp_threshold_bytes: int = 1024,
        large_weight_threshold: int = 1024,
    ) -> None:
        self._env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=select_autoescape(enabled_extensions=()),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self._restrict_arrays = restrict_arrays
        if truncate_weights_after is not None and truncate_weights_after < 1:
            raise CodegenError("truncate_weights_after must be >= 1")
        self._truncate_weights_after = truncate_weights_after
        if large_temp_threshold_bytes < 0:
            raise CodegenError("large_temp_threshold_bytes must be >= 0")
        self._large_temp_threshold_bytes = large_temp_threshold_bytes
        if large_weight_threshold < 0:
            raise CodegenError("large_weight_threshold must be >= 0")
        self._large_weight_threshold = large_weight_threshold
        self._emit_state: _EmitState | None = None

    @staticmethod
    def _sanitize_identifier(name: str) -> str:
        sanitized = _C_IDENTIFIER_RE.sub("_", name)
        if not sanitized:
            sanitized = "v"
        if sanitized[0].isdigit():
            sanitized = f"v_{sanitized}"
        return sanitized

    def _op_function_name(self, model: LoweredModel, index: int) -> str:
        node_info = model.node_infos[index]
        suffix = node_info.name or node_info.op_type
        base_name = f"node{index}_{suffix}".lower()
        return self._sanitize_identifier(base_name)

    @staticmethod
    def _ensure_unique_identifier(base: str, used: set[str]) -> str:
        if base not in used and base not in _C_KEYWORDS:
            return base
        index = 1
        while True:
            candidate = f"{base}_{index}"
            if candidate not in used and candidate not in _C_KEYWORDS:
                return candidate
            index += 1

    def _unique_param_map(
        self, params: Sequence[tuple[str, str | None]]
    ) -> dict[str, str | None]:
        used: set[str] = set()
        mapped: dict[str, str | None] = {}
        for key, name in params:
            if name is None:
                mapped[key] = None
                continue
            sanitized = self._sanitize_identifier(key)
            unique = self._ensure_unique_identifier(sanitized, used)
            used.add(unique)
            mapped[key] = unique
        return mapped

    def _shared_param_map(
        self, params: Sequence[tuple[str, str | None]]
    ) -> dict[str, str | None]:
        used: set[str] = set()
        mapped: dict[str, str | None] = {}
        name_map: dict[str, str] = {}
        for key, name in params:
            if name is None:
                mapped[key] = None
                continue
            if key in name_map:
                mapped[key] = name_map[key]
                continue
            sanitized = self._sanitize_identifier(key)
            unique = self._ensure_unique_identifier(sanitized, used)
            used.add(unique)
            name_map[key] = unique
            mapped[key] = unique
        return mapped

    @staticmethod
    def _build_param_decls(
        specs: Sequence[tuple[str | None, str, str, bool]]
    ) -> list[str]:
        ordered: list[str] = []
        grouped: dict[str, dict[str, object]] = {}
        for name, c_type, suffix, is_const in specs:
            if name is None:
                continue
            if name not in grouped:
                grouped[name] = {
                    "c_type": c_type,
                    "suffix": suffix,
                    "is_const": is_const,
                }
                ordered.append(name)
            else:
                if not is_const:
                    grouped[name]["is_const"] = False
        decls: list[str] = []
        for name in ordered:
            info = grouped[name]
            const_prefix = "const " if info["is_const"] else ""
            decls.append(
                f"{const_prefix}{info['c_type']} {name}{info['suffix']}"
            )
        return decls

    @staticmethod
    def _op_names(
        op: BinaryOp
        | MultiInputBinaryOp
        | WhereOp
        | UnaryOp
        | ClipOp
        | CastOp
        | QuantizeLinearOp
        | QLinearMatMulOp
        | MatMulOp
        | EinsumOp
        | GemmOp
        | AttentionOp
        | RotaryEmbeddingOp
        | ConvOp
        | AveragePoolOp
        | BatchNormOp
        | LpNormalizationOp
        | InstanceNormalizationOp
        | GroupNormalizationOp
        | LayerNormalizationOp
        | MeanVarianceNormalizationOp
        | RMSNormalizationOp
        | LrnOp
        | LstmOp
        | AdagradOp
        | SoftmaxOp
        | LogSoftmaxOp
        | HardmaxOp
        | NegativeLogLikelihoodLossOp
        | SoftmaxCrossEntropyLossOp
        | MaxPoolOp
        | ConcatOp
        | GatherElementsOp
        | GatherOp
        | GatherNDOp
        | ScatterNDOp
        | TensorScatterOp
        | TransposeOp
        | ReshapeOp
        | IdentityOp
        | EyeLikeOp
        | TriluOp
        | TileOp
        | PadOp
        | DepthToSpaceOp
        | SpaceToDepthOp
        | SliceOp
        | ResizeOp
        | GridSampleOp
        | ReduceOp
        | ArgReduceOp
        | TopKOp
        | ConstantOfShapeOp
        | ShapeOp
        | SizeOp
        | NonZeroOp
        | NonMaxSuppressionOp
        | ExpandOp
        | CumSumOp
        | RangeOp
        | OneHotOp
        | SplitOp,
    ) -> tuple[str, ...]:
        if isinstance(op, BinaryOp):
            return (op.input0, op.input1, op.output)
        if isinstance(op, MultiInputBinaryOp):
            return (*op.inputs, op.output)
        if isinstance(op, WhereOp):
            return (op.condition, op.input_x, op.input_y, op.output)
        if isinstance(op, UnaryOp):
            return (op.input0, op.output)
        if isinstance(op, ClipOp):
            names = [op.input0]
            if op.input_min is not None:
                names.append(op.input_min)
            if op.input_max is not None:
                names.append(op.input_max)
            names.append(op.output)
            return tuple(names)
        if isinstance(op, CastOp):
            return (op.input0, op.output)
        if isinstance(op, QuantizeLinearOp):
            names = [op.input0, op.scale]
            if op.zero_point is not None:
                names.append(op.zero_point)
            names.append(op.output)
            return tuple(names)
        if isinstance(op, QLinearMatMulOp):
            return (
                op.input0,
                op.input0_scale,
                op.input0_zero_point,
                op.input1,
                op.input1_scale,
                op.input1_zero_point,
                op.output_scale,
                op.output_zero_point,
                op.output,
            )
        if isinstance(op, MatMulOp):
            return (op.input0, op.input1, op.output)
        if isinstance(op, EinsumOp):
            return (*op.inputs, op.output)
        if isinstance(op, GemmOp):
            names = [op.input_a, op.input_b]
            if op.input_c is not None:
                names.append(op.input_c)
            names.append(op.output)
            return tuple(names)
        if isinstance(op, AttentionOp):
            names = [op.input_q, op.input_k, op.input_v]
            if op.input_attn_mask is not None:
                names.append(op.input_attn_mask)
            if op.input_past_key is not None:
                names.append(op.input_past_key)
            if op.input_past_value is not None:
                names.append(op.input_past_value)
            if op.input_nonpad_kv_seqlen is not None:
                names.append(op.input_nonpad_kv_seqlen)
            names.append(op.output)
            if op.output_present_key is not None:
                names.append(op.output_present_key)
            if op.output_present_value is not None:
                names.append(op.output_present_value)
            if op.output_qk_matmul is not None:
                names.append(op.output_qk_matmul)
            return tuple(names)
        if isinstance(op, RotaryEmbeddingOp):
            names = [op.input0, op.cos_cache, op.sin_cache]
            if op.position_ids is not None:
                names.append(op.position_ids)
            names.append(op.output)
            return tuple(names)
        if isinstance(op, ConvOp):
            names = [op.input0, op.weights]
            if op.bias is not None:
                names.append(op.bias)
            names.append(op.output)
            return tuple(names)
        if isinstance(op, ConvTransposeOp):
            names = [op.input0, op.weights]
            if op.bias is not None:
                names.append(op.bias)
            names.append(op.output)
            return tuple(names)
        if isinstance(op, AveragePoolOp):
            return (op.input0, op.output)
        if isinstance(op, LpPoolOp):
            return (op.input0, op.output)
        if isinstance(op, BatchNormOp):
            return (op.input0, op.scale, op.bias, op.mean, op.variance, op.output)
        if isinstance(op, LpNormalizationOp):
            return (op.input0, op.output)
        if isinstance(op, InstanceNormalizationOp):
            return (op.input0, op.scale, op.bias, op.output)
        if isinstance(op, GroupNormalizationOp):
            return (op.input0, op.scale, op.bias, op.output)
        if isinstance(op, LayerNormalizationOp):
            names = [op.input0, op.scale]
            if op.bias is not None:
                names.append(op.bias)
            names.append(op.output)
            if op.mean_output is not None:
                names.append(op.mean_output)
            if op.invstd_output is not None:
                names.append(op.invstd_output)
            return tuple(names)
        if isinstance(op, MeanVarianceNormalizationOp):
            return (op.input0, op.output)
        if isinstance(op, RMSNormalizationOp):
            return (op.input0, op.scale, op.output)
        if isinstance(op, LstmOp):
            names = [op.input_x, op.input_w, op.input_r]
            if op.input_b is not None:
                names.append(op.input_b)
            if op.input_sequence_lens is not None:
                names.append(op.input_sequence_lens)
            if op.input_initial_h is not None:
                names.append(op.input_initial_h)
            if op.input_initial_c is not None:
                names.append(op.input_initial_c)
            if op.input_p is not None:
                names.append(op.input_p)
            if op.output_y is not None:
                names.append(op.output_y)
            if op.output_y_h is not None:
                names.append(op.output_y_h)
            if op.output_y_c is not None:
                names.append(op.output_y_c)
            return tuple(names)
        if isinstance(op, AdagradOp):
            return (
                op.rate,
                op.timestep,
                *op.inputs,
                *op.gradients,
                *op.accumulators,
                *op.outputs,
                *op.accumulator_outputs,
            )
        if isinstance(op, (SoftmaxOp, LogSoftmaxOp, HardmaxOp)):
            return (op.input0, op.output)
        if isinstance(op, NegativeLogLikelihoodLossOp):
            names = [op.input0, op.target]
            if op.weight is not None:
                names.append(op.weight)
            names.append(op.output)
            return tuple(names)
        if isinstance(op, SoftmaxCrossEntropyLossOp):
            names = [op.input0, op.target]
            if op.weight is not None:
                names.append(op.weight)
            names.append(op.output)
            if op.log_prob is not None:
                names.append(op.log_prob)
            return tuple(names)
        if isinstance(op, MaxPoolOp):
            names = [op.input0, op.output]
            if op.indices is not None:
                names.append(op.indices)
            return tuple(names)
        if isinstance(op, GatherElementsOp):
            return (op.data, op.indices, op.output)
        if isinstance(op, GatherOp):
            return (op.data, op.indices, op.output)
        if isinstance(op, GatherNDOp):
            return (op.data, op.indices, op.output)
        if isinstance(op, ScatterNDOp):
            return (op.data, op.indices, op.updates, op.output)
        if isinstance(op, TensorScatterOp):
            names = [op.past_cache, op.update]
            if op.write_indices is not None:
                names.append(op.write_indices)
            names.append(op.output)
            return tuple(names)
        if isinstance(op, ConcatOp):
            return (*op.inputs, op.output)
        if isinstance(op, ConstantOfShapeOp):
            return (op.input0, op.output)
        if isinstance(op, ShapeOp):
            return (op.input0, op.output)
        if isinstance(op, SizeOp):
            return (op.input0, op.output)
        if isinstance(op, NonZeroOp):
            return (op.input0, op.output)
        if isinstance(op, NonMaxSuppressionOp):
            names = [op.boxes, op.scores]
            if op.max_output_boxes_per_class is not None:
                names.append(op.max_output_boxes_per_class)
            if op.iou_threshold is not None:
                names.append(op.iou_threshold)
            if op.score_threshold is not None:
                names.append(op.score_threshold)
            names.append(op.output)
            return tuple(names)
        if isinstance(op, ExpandOp):
            return (op.input0, op.output)
        if isinstance(op, CumSumOp):
            names = [op.input0]
            if op.axis_input is not None:
                names.append(op.axis_input)
            names.append(op.output)
            return tuple(names)
        if isinstance(op, RangeOp):
            return (op.start, op.limit, op.delta, op.output)
        if isinstance(op, OneHotOp):
            return (op.indices, op.depth, op.values, op.output)
        if isinstance(op, SplitOp):
            return (op.input0, *op.outputs)
        if isinstance(op, ReshapeOp):
            return (op.input0, op.output)
        if isinstance(op, IdentityOp):
            return (op.input0, op.output)
        if isinstance(op, EyeLikeOp):
            return (op.input0, op.output)
        if isinstance(op, TriluOp):
            names = [op.input0]
            if op.k_input is not None:
                names.append(op.k_input)
            names.append(op.output)
            return tuple(names)
        if isinstance(op, TileOp):
            return (op.input0, op.output)
        if isinstance(op, PadOp):
            names = [op.input0, op.output]
            if op.pads_input is not None:
                names.append(op.pads_input)
            if op.axes_input is not None:
                names.append(op.axes_input)
            if op.value_input is not None:
                names.append(op.value_input)
            return tuple(names)
        if isinstance(op, DepthToSpaceOp):
            return (op.input0, op.output)
        if isinstance(op, SpaceToDepthOp):
            return (op.input0, op.output)
        if isinstance(op, SliceOp):
            names = [op.input0]
            if op.starts_input is not None:
                names.append(op.starts_input)
            if op.ends_input is not None:
                names.append(op.ends_input)
            if op.axes_input is not None:
                names.append(op.axes_input)
            if op.steps_input is not None:
                names.append(op.steps_input)
            names.append(op.output)
            return tuple(names)
        if isinstance(op, ResizeOp):
            names = [op.input0]
            if op.roi_input is not None:
                names.append(op.roi_input)
            if op.scales_input is not None:
                names.append(op.scales_input)
            if op.sizes_input is not None:
                names.append(op.sizes_input)
            names.append(op.output)
            return tuple(names)
        if isinstance(op, GridSampleOp):
            return (op.input0, op.grid, op.output)
        if isinstance(op, TopKOp):
            return (op.input0, op.output_values, op.output_indices)
        if isinstance(op, ReduceOp):
            names = [op.input0]
            if op.axes_input is not None:
                names.append(op.axes_input)
            names.append(op.output)
            return tuple(names)
        return (op.input0, op.output)

    def _build_name_map(self, model: LoweredModel) -> dict[str, str]:
        used: set[str] = set()
        name_map: dict[str, str] = {}
        constant_names = {const.name for const in model.constants}
        names = [model.name]
        names.extend(model.input_names)
        names.extend(model.output_names)
        for op in model.ops:
            names.extend(
                name for name in self._op_names(op) if name not in constant_names
            )
        for name in names:
            if name in name_map:
                continue
            sanitized = self._sanitize_identifier(name)
            unique = self._ensure_unique_identifier(sanitized, used)
            name_map[name] = unique
            used.add(unique)
        for index, const in enumerate(model.constants, start=1):
            if const.name in name_map:
                continue
            base_name = self._sanitize_identifier(const.name.lower())
            weight_name = f"weight{index}_{base_name}"
            unique = self._ensure_unique_identifier(weight_name, used)
            name_map[const.name] = unique
            used.add(unique)
        return name_map

    @staticmethod
    def _map_optional_name(
        name_map: dict[str, str], name: str | None
    ) -> str | None:
        if name is None:
            return None
        return name_map.get(name, name)

    def _map_op_names(
        self,
        op: BinaryOp
        | MultiInputBinaryOp
        | WhereOp
        | UnaryOp
        | ClipOp
        | CastOp
        | QuantizeLinearOp
        | QLinearMatMulOp
        | MatMulOp
        | EinsumOp
        | GemmOp
        | AttentionOp
        | RotaryEmbeddingOp
        | ConvOp
        | ConvTransposeOp
        | AveragePoolOp
        | LpPoolOp
        | BatchNormOp
        | LpNormalizationOp
        | InstanceNormalizationOp
        | GroupNormalizationOp
        | LayerNormalizationOp
        | MeanVarianceNormalizationOp
        | RMSNormalizationOp
        | LrnOp
        | LstmOp
        | AdagradOp
        | SoftmaxOp
        | LogSoftmaxOp
        | HardmaxOp
        | NegativeLogLikelihoodLossOp
        | SoftmaxCrossEntropyLossOp
        | MaxPoolOp
        | ConcatOp
        | GatherElementsOp
        | GatherOp
        | GatherNDOp
        | ScatterNDOp
        | TensorScatterOp
        | TransposeOp
        | ReshapeOp
        | IdentityOp
        | EyeLikeOp
        | TriluOp
        | TileOp
        | PadOp
        | DepthToSpaceOp
        | SpaceToDepthOp
        | SliceOp
        | ResizeOp
        | GridSampleOp
        | ReduceOp
        | ArgReduceOp
        | TopKOp
        | ConstantOfShapeOp
        | ShapeOp
        | SizeOp
        | NonZeroOp
        | NonMaxSuppressionOp
        | ExpandOp
        | CumSumOp
        | RangeOp
        | OneHotOp
        | SplitOp,
        name_map: dict[str, str],
    ) -> (
        BinaryOp
        | MultiInputBinaryOp
        | WhereOp
        | UnaryOp
        | ClipOp
        | CastOp
        | QuantizeLinearOp
        | QLinearMatMulOp
        | MatMulOp
        | EinsumOp
        | GemmOp
        | AttentionOp
        | RotaryEmbeddingOp
        | ConvOp
        | ConvTransposeOp
        | AveragePoolOp
        | LpPoolOp
        | BatchNormOp
        | LpNormalizationOp
        | InstanceNormalizationOp
        | GroupNormalizationOp
        | LayerNormalizationOp
        | MeanVarianceNormalizationOp
        | RMSNormalizationOp
        | LrnOp
        | LstmOp
        | AdagradOp
        | SoftmaxOp
        | LogSoftmaxOp
        | HardmaxOp
        | NegativeLogLikelihoodLossOp
        | SoftmaxCrossEntropyLossOp
        | MaxPoolOp
        | ConcatOp
        | GatherElementsOp
        | GatherOp
        | GatherNDOp
        | ScatterNDOp
        | TensorScatterOp
        | TransposeOp
        | ReshapeOp
        | IdentityOp
        | EyeLikeOp
        | TriluOp
        | TileOp
        | PadOp
        | DepthToSpaceOp
        | SpaceToDepthOp
        | SliceOp
        | ResizeOp
        | GridSampleOp
        | ReduceOp
        | ArgReduceOp
        | TopKOp
        | ConstantOfShapeOp
        | ShapeOp
        | SizeOp
        | NonZeroOp
        | NonMaxSuppressionOp
        | ExpandOp
        | CumSumOp
        | RangeOp
        | OneHotOp
        | SplitOp
    ):
        if isinstance(op, BinaryOp):
            return BinaryOp(
                input0=name_map.get(op.input0, op.input0),
                input1=name_map.get(op.input1, op.input1),
                output=name_map.get(op.output, op.output),
                function=op.function,
                operator_kind=op.operator_kind,
                input0_shape=op.input0_shape,
                input1_shape=op.input1_shape,
                shape=op.shape,
                dtype=op.dtype,
                input_dtype=op.input_dtype,
            )
        if isinstance(op, MultiInputBinaryOp):
            return MultiInputBinaryOp(
                inputs=tuple(name_map.get(name, name) for name in op.inputs),
                output=name_map.get(op.output, op.output),
                function=op.function,
                operator_kind=op.operator_kind,
                shape=op.shape,
                dtype=op.dtype,
                input_dtype=op.input_dtype,
            )
        if isinstance(op, WhereOp):
            return WhereOp(
                condition=name_map.get(op.condition, op.condition),
                input_x=name_map.get(op.input_x, op.input_x),
                input_y=name_map.get(op.input_y, op.input_y),
                output=name_map.get(op.output, op.output),
                condition_shape=op.condition_shape,
                x_shape=op.x_shape,
                y_shape=op.y_shape,
                output_shape=op.output_shape,
                dtype=op.dtype,
            )
        if isinstance(op, UnaryOp):
            return UnaryOp(
                input0=name_map.get(op.input0, op.input0),
                output=name_map.get(op.output, op.output),
                function=op.function,
                shape=op.shape,
                dtype=op.dtype,
                input_dtype=op.input_dtype,
                params=op.params,
            )
        if isinstance(op, ClipOp):
            return ClipOp(
                input0=name_map.get(op.input0, op.input0),
                input_min=self._map_optional_name(name_map, op.input_min),
                input_max=self._map_optional_name(name_map, op.input_max),
                output=name_map.get(op.output, op.output),
                input_shape=op.input_shape,
                min_shape=op.min_shape,
                max_shape=op.max_shape,
                output_shape=op.output_shape,
                dtype=op.dtype,
            )
        if isinstance(op, CastOp):
            return CastOp(
                input0=name_map.get(op.input0, op.input0),
                output=name_map.get(op.output, op.output),
                shape=op.shape,
                input_dtype=op.input_dtype,
                dtype=op.dtype,
            )
        if isinstance(op, QuantizeLinearOp):
            return QuantizeLinearOp(
                input0=name_map.get(op.input0, op.input0),
                scale=name_map.get(op.scale, op.scale),
                zero_point=self._map_optional_name(name_map, op.zero_point),
                output=name_map.get(op.output, op.output),
                input_shape=op.input_shape,
                axis=op.axis,
                dtype=op.dtype,
                input_dtype=op.input_dtype,
                scale_dtype=op.scale_dtype,
            )
        if isinstance(op, QLinearMatMulOp):
            return QLinearMatMulOp(
                input0=name_map.get(op.input0, op.input0),
                input0_scale=name_map.get(op.input0_scale, op.input0_scale),
                input0_zero_point=name_map.get(
                    op.input0_zero_point, op.input0_zero_point
                ),
                input1=name_map.get(op.input1, op.input1),
                input1_scale=name_map.get(op.input1_scale, op.input1_scale),
                input1_zero_point=name_map.get(
                    op.input1_zero_point, op.input1_zero_point
                ),
                output_scale=name_map.get(op.output_scale, op.output_scale),
                output_zero_point=name_map.get(
                    op.output_zero_point, op.output_zero_point
                ),
                output=name_map.get(op.output, op.output),
                input0_shape=op.input0_shape,
                input1_shape=op.input1_shape,
                output_shape=op.output_shape,
                batch_shape=op.batch_shape,
                input0_batch_shape=op.input0_batch_shape,
                input1_batch_shape=op.input1_batch_shape,
                m=op.m,
                n=op.n,
                k=op.k,
                left_vector=op.left_vector,
                right_vector=op.right_vector,
                input0_dtype=op.input0_dtype,
                input1_dtype=op.input1_dtype,
                dtype=op.dtype,
                input0_scale_dtype=op.input0_scale_dtype,
                input1_scale_dtype=op.input1_scale_dtype,
                output_scale_dtype=op.output_scale_dtype,
                input0_scale_shape=op.input0_scale_shape,
                input1_scale_shape=op.input1_scale_shape,
                output_scale_shape=op.output_scale_shape,
                input0_zero_shape=op.input0_zero_shape,
                input1_zero_shape=op.input1_zero_shape,
                output_zero_shape=op.output_zero_shape,
            )
        if isinstance(op, MatMulOp):
            return MatMulOp(
                input0=name_map.get(op.input0, op.input0),
                input1=name_map.get(op.input1, op.input1),
                output=name_map.get(op.output, op.output),
                input0_shape=op.input0_shape,
                input1_shape=op.input1_shape,
                output_shape=op.output_shape,
                batch_shape=op.batch_shape,
                input0_batch_shape=op.input0_batch_shape,
                input1_batch_shape=op.input1_batch_shape,
                m=op.m,
                n=op.n,
                k=op.k,
                left_vector=op.left_vector,
                right_vector=op.right_vector,
                dtype=op.dtype,
            )
        if isinstance(op, EinsumOp):
            return EinsumOp(
                inputs=tuple(name_map.get(name, name) for name in op.inputs),
                output=name_map.get(op.output, op.output),
                kind=op.kind,
                input_shapes=op.input_shapes,
                output_shape=op.output_shape,
                dtype=op.dtype,
                input_dtype=op.input_dtype,
            )
        if isinstance(op, GemmOp):
            return GemmOp(
                input_a=name_map.get(op.input_a, op.input_a),
                input_b=name_map.get(op.input_b, op.input_b),
                input_c=self._map_optional_name(name_map, op.input_c),
                output=name_map.get(op.output, op.output),
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
                input_q=name_map.get(op.input_q, op.input_q),
                input_k=name_map.get(op.input_k, op.input_k),
                input_v=name_map.get(op.input_v, op.input_v),
                input_attn_mask=self._map_optional_name(
                    name_map, op.input_attn_mask
                ),
                input_past_key=self._map_optional_name(
                    name_map, op.input_past_key
                ),
                input_past_value=self._map_optional_name(
                    name_map, op.input_past_value
                ),
                input_nonpad_kv_seqlen=self._map_optional_name(
                    name_map, op.input_nonpad_kv_seqlen
                ),
                output_present_key=self._map_optional_name(
                    name_map, op.output_present_key
                ),
                output_present_value=self._map_optional_name(
                    name_map, op.output_present_value
                ),
                output_qk_matmul=self._map_optional_name(
                    name_map, op.output_qk_matmul
                ),
                output=name_map.get(op.output, op.output),
                batch=op.batch,
                q_heads=op.q_heads,
                kv_heads=op.kv_heads,
                q_seq=op.q_seq,
                kv_seq=op.kv_seq,
                total_seq=op.total_seq,
                past_seq=op.past_seq,
                qk_head_size=op.qk_head_size,
                v_head_size=op.v_head_size,
                q_hidden_size=op.q_hidden_size,
                k_hidden_size=op.k_hidden_size,
                v_hidden_size=op.v_hidden_size,
                scale=op.scale,
                is_causal=op.is_causal,
                softcap=op.softcap,
                qk_matmul_output_mode=op.qk_matmul_output_mode,
                q_rank=op.q_rank,
                k_rank=op.k_rank,
                v_rank=op.v_rank,
                output_rank=op.output_rank,
                mask_shape=op.mask_shape,
                mask_is_bool=op.mask_is_bool,
                mask_rank=op.mask_rank,
                mask_broadcast_batch=op.mask_broadcast_batch,
                mask_broadcast_heads=op.mask_broadcast_heads,
                mask_broadcast_q_seq=op.mask_broadcast_q_seq,
                mask_q_seq=op.mask_q_seq,
                mask_kv_seq=op.mask_kv_seq,
                head_group_size=op.head_group_size,
                dtype=op.dtype,
            )
        if isinstance(op, RotaryEmbeddingOp):
            return RotaryEmbeddingOp(
                input0=name_map.get(op.input0, op.input0),
                cos_cache=name_map.get(op.cos_cache, op.cos_cache),
                sin_cache=name_map.get(op.sin_cache, op.sin_cache),
                position_ids=self._map_optional_name(
                    name_map, op.position_ids
                ),
                output=name_map.get(op.output, op.output),
                input_shape=op.input_shape,
                cos_shape=op.cos_shape,
                sin_shape=op.sin_shape,
                position_ids_shape=op.position_ids_shape,
                dtype=op.dtype,
                position_ids_dtype=op.position_ids_dtype,
                rotary_dim=op.rotary_dim,
                rotary_dim_half=op.rotary_dim_half,
                head_size=op.head_size,
                num_heads=op.num_heads,
                seq_len=op.seq_len,
                batch=op.batch,
                input_rank=op.input_rank,
                interleaved=op.interleaved,
            )
        if isinstance(op, ConvOp):
            return ConvOp(
                input0=name_map.get(op.input0, op.input0),
                weights=name_map.get(op.weights, op.weights),
                bias=self._map_optional_name(name_map, op.bias),
                output=name_map.get(op.output, op.output),
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
        if isinstance(op, ConvTransposeOp):
            return ConvTransposeOp(
                input0=name_map.get(op.input0, op.input0),
                weights=name_map.get(op.weights, op.weights),
                bias=self._map_optional_name(name_map, op.bias),
                output=name_map.get(op.output, op.output),
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
                output_padding=op.output_padding,
                group=op.group,
                dtype=op.dtype,
            )
        if isinstance(op, AveragePoolOp):
            return AveragePoolOp(
                input0=name_map.get(op.input0, op.input0),
                output=name_map.get(op.output, op.output),
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
        if isinstance(op, LpPoolOp):
            return LpPoolOp(
                input0=name_map.get(op.input0, op.input0),
                output=name_map.get(op.output, op.output),
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
                p=op.p,
                dtype=op.dtype,
            )
        if isinstance(op, BatchNormOp):
            return BatchNormOp(
                input0=name_map.get(op.input0, op.input0),
                scale=name_map.get(op.scale, op.scale),
                bias=name_map.get(op.bias, op.bias),
                mean=name_map.get(op.mean, op.mean),
                variance=name_map.get(op.variance, op.variance),
                output=name_map.get(op.output, op.output),
                shape=op.shape,
                channels=op.channels,
                epsilon=op.epsilon,
                dtype=op.dtype,
            )
        if isinstance(op, LpNormalizationOp):
            return LpNormalizationOp(
                input0=name_map.get(op.input0, op.input0),
                output=name_map.get(op.output, op.output),
                shape=op.shape,
                axis=op.axis,
                p=op.p,
                outer=op.outer,
                axis_size=op.axis_size,
                inner=op.inner,
                dtype=op.dtype,
            )
        if isinstance(op, InstanceNormalizationOp):
            return InstanceNormalizationOp(
                input0=name_map.get(op.input0, op.input0),
                scale=name_map.get(op.scale, op.scale),
                bias=name_map.get(op.bias, op.bias),
                output=name_map.get(op.output, op.output),
                shape=op.shape,
                channels=op.channels,
                spatial_size=op.spatial_size,
                epsilon=op.epsilon,
                dtype=op.dtype,
            )
        if isinstance(op, GroupNormalizationOp):
            return GroupNormalizationOp(
                input0=name_map.get(op.input0, op.input0),
                scale=name_map.get(op.scale, op.scale),
                bias=name_map.get(op.bias, op.bias),
                output=name_map.get(op.output, op.output),
                shape=op.shape,
                channels=op.channels,
                num_groups=op.num_groups,
                group_size=op.group_size,
                spatial_size=op.spatial_size,
                epsilon=op.epsilon,
                dtype=op.dtype,
            )
        if isinstance(op, LayerNormalizationOp):
            return LayerNormalizationOp(
                input0=name_map.get(op.input0, op.input0),
                scale=name_map.get(op.scale, op.scale),
                bias=self._map_optional_name(name_map, op.bias),
                output=name_map.get(op.output, op.output),
                mean_output=self._map_optional_name(name_map, op.mean_output),
                invstd_output=self._map_optional_name(name_map, op.invstd_output),
                shape=op.shape,
                normalized_shape=op.normalized_shape,
                scale_shape=op.scale_shape,
                bias_shape=op.bias_shape,
                outer=op.outer,
                inner=op.inner,
                axis=op.axis,
                epsilon=op.epsilon,
                dtype=op.dtype,
            )
        if isinstance(op, MeanVarianceNormalizationOp):
            return MeanVarianceNormalizationOp(
                input0=name_map.get(op.input0, op.input0),
                output=name_map.get(op.output, op.output),
                shape=op.shape,
                axes=op.axes,
                non_axes=op.non_axes,
                reduce_count=op.reduce_count,
                epsilon=op.epsilon,
                dtype=op.dtype,
            )
        if isinstance(op, RMSNormalizationOp):
            return RMSNormalizationOp(
                input0=name_map.get(op.input0, op.input0),
                scale=name_map.get(op.scale, op.scale),
                output=name_map.get(op.output, op.output),
                shape=op.shape,
                normalized_shape=op.normalized_shape,
                scale_shape=op.scale_shape,
                outer=op.outer,
                inner=op.inner,
                axis=op.axis,
                epsilon=op.epsilon,
                dtype=op.dtype,
            )
        if isinstance(op, LrnOp):
            return LrnOp(
                input0=name_map.get(op.input0, op.input0),
                output=name_map.get(op.output, op.output),
                shape=op.shape,
                channels=op.channels,
                size=op.size,
                half=op.half,
                alpha=op.alpha,
                beta=op.beta,
                bias=op.bias,
                dtype=op.dtype,
            )
        if isinstance(op, LstmOp):
            return LstmOp(
                input_x=name_map.get(op.input_x, op.input_x),
                input_w=name_map.get(op.input_w, op.input_w),
                input_r=name_map.get(op.input_r, op.input_r),
                input_b=self._map_optional_name(name_map, op.input_b),
                input_sequence_lens=self._map_optional_name(
                    name_map, op.input_sequence_lens
                ),
                input_initial_h=self._map_optional_name(
                    name_map, op.input_initial_h
                ),
                input_initial_c=self._map_optional_name(
                    name_map, op.input_initial_c
                ),
                input_p=self._map_optional_name(name_map, op.input_p),
                output_y=self._map_optional_name(name_map, op.output_y),
                output_y_h=self._map_optional_name(name_map, op.output_y_h),
                output_y_c=self._map_optional_name(name_map, op.output_y_c),
                seq_length=op.seq_length,
                batch_size=op.batch_size,
                input_size=op.input_size,
                hidden_size=op.hidden_size,
                num_directions=op.num_directions,
                direction=op.direction,
                layout=op.layout,
                input_forget=op.input_forget,
                clip=op.clip,
                activation_kinds=op.activation_kinds,
                activation_alphas=op.activation_alphas,
                activation_betas=op.activation_betas,
                dtype=op.dtype,
                sequence_lens_dtype=op.sequence_lens_dtype,
            )
        if isinstance(op, AdagradOp):
            return AdagradOp(
                rate=name_map.get(op.rate, op.rate),
                timestep=name_map.get(op.timestep, op.timestep),
                inputs=tuple(name_map.get(name, name) for name in op.inputs),
                gradients=tuple(
                    name_map.get(name, name) for name in op.gradients
                ),
                accumulators=tuple(
                    name_map.get(name, name) for name in op.accumulators
                ),
                outputs=tuple(name_map.get(name, name) for name in op.outputs),
                accumulator_outputs=tuple(
                    name_map.get(name, name)
                    for name in op.accumulator_outputs
                ),
                rate_shape=op.rate_shape,
                timestep_shape=op.timestep_shape,
                tensor_shapes=op.tensor_shapes,
                output_shapes=op.output_shapes,
                dtype=op.dtype,
                rate_dtype=op.rate_dtype,
                timestep_dtype=op.timestep_dtype,
                norm_coefficient=op.norm_coefficient,
                epsilon=op.epsilon,
                decay_factor=op.decay_factor,
            )
        if isinstance(op, SoftmaxOp):
            return SoftmaxOp(
                input0=name_map.get(op.input0, op.input0),
                output=name_map.get(op.output, op.output),
                outer=op.outer,
                axis_size=op.axis_size,
                inner=op.inner,
                axis=op.axis,
                shape=op.shape,
                dtype=op.dtype,
            )
        if isinstance(op, LogSoftmaxOp):
            return LogSoftmaxOp(
                input0=name_map.get(op.input0, op.input0),
                output=name_map.get(op.output, op.output),
                outer=op.outer,
                axis_size=op.axis_size,
                inner=op.inner,
                axis=op.axis,
                shape=op.shape,
                dtype=op.dtype,
            )
        if isinstance(op, HardmaxOp):
            return HardmaxOp(
                input0=name_map.get(op.input0, op.input0),
                output=name_map.get(op.output, op.output),
                outer=op.outer,
                axis_size=op.axis_size,
                inner=op.inner,
                axis=op.axis,
                shape=op.shape,
                dtype=op.dtype,
            )
        if isinstance(op, NegativeLogLikelihoodLossOp):
            return NegativeLogLikelihoodLossOp(
                input0=name_map.get(op.input0, op.input0),
                target=name_map.get(op.target, op.target),
                weight=self._map_optional_name(name_map, op.weight),
                output=name_map.get(op.output, op.output),
                input_shape=op.input_shape,
                target_shape=op.target_shape,
                output_shape=op.output_shape,
                n=op.n,
                c=op.c,
                d=op.d,
                ignore_index=op.ignore_index,
                reduction=op.reduction,
                input_dtype=op.input_dtype,
                weight_dtype=op.weight_dtype,
                weight_shape=op.weight_shape,
                dtype=op.dtype,
                target_dtype=op.target_dtype,
            )
        if isinstance(op, SoftmaxCrossEntropyLossOp):
            return SoftmaxCrossEntropyLossOp(
                input0=name_map.get(op.input0, op.input0),
                target=name_map.get(op.target, op.target),
                weight=self._map_optional_name(name_map, op.weight),
                output=name_map.get(op.output, op.output),
                log_prob=self._map_optional_name(name_map, op.log_prob),
                input_shape=op.input_shape,
                target_shape=op.target_shape,
                output_shape=op.output_shape,
                log_prob_shape=op.log_prob_shape,
                n=op.n,
                c=op.c,
                d=op.d,
                ignore_index=op.ignore_index,
                reduction=op.reduction,
                input_dtype=op.input_dtype,
                weight_dtype=op.weight_dtype,
                weight_shape=op.weight_shape,
                dtype=op.dtype,
                target_dtype=op.target_dtype,
            )
        if isinstance(op, MaxPoolOp):
            return MaxPoolOp(
                input0=name_map.get(op.input0, op.input0),
                output=name_map.get(op.output, op.output),
                indices=self._map_optional_name(name_map, op.indices),
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
                storage_order=op.storage_order,
                dtype=op.dtype,
                indices_dtype=op.indices_dtype,
            )
        if isinstance(op, ConcatOp):
            return ConcatOp(
                inputs=tuple(
                    name_map.get(name, name) for name in op.inputs
                ),
                output=name_map.get(op.output, op.output),
                input_shapes=op.input_shapes,
                output_shape=op.output_shape,
                axis=op.axis,
                dtype=op.dtype,
            )
        if isinstance(op, GatherElementsOp):
            return GatherElementsOp(
                data=name_map.get(op.data, op.data),
                indices=name_map.get(op.indices, op.indices),
                output=name_map.get(op.output, op.output),
                data_shape=op.data_shape,
                indices_shape=op.indices_shape,
                output_shape=op.output_shape,
                axis=op.axis,
                dtype=op.dtype,
                indices_dtype=op.indices_dtype,
            )
        if isinstance(op, GatherOp):
            return GatherOp(
                data=name_map.get(op.data, op.data),
                indices=name_map.get(op.indices, op.indices),
                output=name_map.get(op.output, op.output),
                data_shape=op.data_shape,
                indices_shape=op.indices_shape,
                output_shape=op.output_shape,
                axis=op.axis,
                dtype=op.dtype,
                indices_dtype=op.indices_dtype,
            )
        if isinstance(op, GatherNDOp):
            return GatherNDOp(
                data=name_map.get(op.data, op.data),
                indices=name_map.get(op.indices, op.indices),
                output=name_map.get(op.output, op.output),
                batch_dims=op.batch_dims,
                data_shape=op.data_shape,
                indices_shape=op.indices_shape,
                output_shape=op.output_shape,
                dtype=op.dtype,
                indices_dtype=op.indices_dtype,
            )
        if isinstance(op, ScatterNDOp):
            return ScatterNDOp(
                data=name_map.get(op.data, op.data),
                indices=name_map.get(op.indices, op.indices),
                updates=name_map.get(op.updates, op.updates),
                output=name_map.get(op.output, op.output),
                data_shape=op.data_shape,
                indices_shape=op.indices_shape,
                updates_shape=op.updates_shape,
                output_shape=op.output_shape,
                reduction=op.reduction,
                dtype=op.dtype,
                indices_dtype=op.indices_dtype,
            )
        if isinstance(op, TensorScatterOp):
            return TensorScatterOp(
                past_cache=name_map.get(op.past_cache, op.past_cache),
                update=name_map.get(op.update, op.update),
                write_indices=(
                    name_map.get(op.write_indices, op.write_indices)
                    if op.write_indices is not None
                    else None
                ),
                output=name_map.get(op.output, op.output),
                past_cache_shape=op.past_cache_shape,
                update_shape=op.update_shape,
                output_shape=op.output_shape,
                write_indices_shape=op.write_indices_shape,
                axis=op.axis,
                mode=op.mode,
                dtype=op.dtype,
                write_indices_dtype=op.write_indices_dtype,
            )
        if isinstance(op, TransposeOp):
            return TransposeOp(
                input0=name_map.get(op.input0, op.input0),
                output=name_map.get(op.output, op.output),
                input_shape=op.input_shape,
                output_shape=op.output_shape,
                perm=op.perm,
                dtype=op.dtype,
                input_dtype=op.input_dtype,
            )
        if isinstance(op, ReshapeOp):
            return ReshapeOp(
                input0=name_map.get(op.input0, op.input0),
                output=name_map.get(op.output, op.output),
                input_shape=op.input_shape,
                output_shape=op.output_shape,
                dtype=op.dtype,
                input_dtype=op.input_dtype,
            )
        if isinstance(op, IdentityOp):
            return IdentityOp(
                input0=name_map.get(op.input0, op.input0),
                output=name_map.get(op.output, op.output),
                shape=op.shape,
                dtype=op.dtype,
                input_dtype=op.input_dtype,
            )
        if isinstance(op, EyeLikeOp):
            return EyeLikeOp(
                input0=name_map.get(op.input0, op.input0),
                output=name_map.get(op.output, op.output),
                output_shape=op.output_shape,
                k=op.k,
                dtype=op.dtype,
                input_dtype=op.input_dtype,
            )
        if isinstance(op, TriluOp):
            return TriluOp(
                input0=name_map.get(op.input0, op.input0),
                output=name_map.get(op.output, op.output),
                input_shape=op.input_shape,
                output_shape=op.output_shape,
                upper=op.upper,
                k_value=op.k_value,
                k_input=self._map_optional_name(name_map, op.k_input),
                k_input_shape=op.k_input_shape,
                k_input_dtype=op.k_input_dtype,
                dtype=op.dtype,
                input_dtype=op.input_dtype,
            )
        if isinstance(op, TileOp):
            return TileOp(
                input0=name_map.get(op.input0, op.input0),
                output=name_map.get(op.output, op.output),
                input_shape=op.input_shape,
                output_shape=op.output_shape,
                repeats=op.repeats,
                input_strides=op.input_strides,
                dtype=op.dtype,
                input_dtype=op.input_dtype,
            )
        if isinstance(op, PadOp):
            return PadOp(
                input0=name_map.get(op.input0, op.input0),
                output=name_map.get(op.output, op.output),
                input_shape=op.input_shape,
                output_shape=op.output_shape,
                pads_begin=op.pads_begin,
                pads_end=op.pads_end,
                pads_input=self._map_optional_name(name_map, op.pads_input),
                pads_shape=op.pads_shape,
                pads_dtype=op.pads_dtype,
                pads_axis_map=op.pads_axis_map,
                pads_values=op.pads_values,
                axes_input=self._map_optional_name(name_map, op.axes_input),
                axes_shape=op.axes_shape,
                axes_dtype=op.axes_dtype,
                mode=op.mode,
                value=op.value,
                value_input=self._map_optional_name(name_map, op.value_input),
                value_shape=op.value_shape,
                dtype=op.dtype,
                input_dtype=op.input_dtype,
                input_strides=op.input_strides,
            )
        if isinstance(op, DepthToSpaceOp):
            return DepthToSpaceOp(
                input0=name_map.get(op.input0, op.input0),
                output=name_map.get(op.output, op.output),
                input_shape=op.input_shape,
                output_shape=op.output_shape,
                blocksize=op.blocksize,
                mode=op.mode,
                dtype=op.dtype,
                input_dtype=op.input_dtype,
            )
        if isinstance(op, SpaceToDepthOp):
            return SpaceToDepthOp(
                input0=name_map.get(op.input0, op.input0),
                output=name_map.get(op.output, op.output),
                input_shape=op.input_shape,
                output_shape=op.output_shape,
                blocksize=op.blocksize,
                dtype=op.dtype,
                input_dtype=op.input_dtype,
            )
        if isinstance(op, SliceOp):
            return SliceOp(
                input0=name_map.get(op.input0, op.input0),
                output=name_map.get(op.output, op.output),
                input_shape=op.input_shape,
                output_shape=op.output_shape,
                starts=op.starts,
                steps=op.steps,
                axes=op.axes,
                starts_input=self._map_optional_name(name_map, op.starts_input),
                ends_input=self._map_optional_name(name_map, op.ends_input),
                axes_input=self._map_optional_name(name_map, op.axes_input),
                steps_input=self._map_optional_name(name_map, op.steps_input),
                starts_shape=op.starts_shape,
                ends_shape=op.ends_shape,
                axes_shape=op.axes_shape,
                steps_shape=op.steps_shape,
                starts_dtype=op.starts_dtype,
                ends_dtype=op.ends_dtype,
                axes_dtype=op.axes_dtype,
                steps_dtype=op.steps_dtype,
                dtype=op.dtype,
                input_dtype=op.input_dtype,
            )
        if isinstance(op, ResizeOp):
            return ResizeOp(
                input0=name_map.get(op.input0, op.input0),
                output=name_map.get(op.output, op.output),
                roi_input=self._map_optional_name(name_map, op.roi_input),
                scales_input=self._map_optional_name(name_map, op.scales_input),
                sizes_input=self._map_optional_name(name_map, op.sizes_input),
                input_shape=op.input_shape,
                output_shape=op.output_shape,
                scales=op.scales,
                axes=op.axes,
                scales_shape=op.scales_shape,
                sizes_shape=op.sizes_shape,
                roi_shape=op.roi_shape,
                scales_dtype=op.scales_dtype,
                sizes_dtype=op.sizes_dtype,
                roi_dtype=op.roi_dtype,
                scales_axes=op.scales_axes,
                sizes_axes=op.sizes_axes,
                roi_axes=op.roi_axes,
                mode=op.mode,
                nearest_mode=op.nearest_mode,
                coordinate_transformation_mode=op.coordinate_transformation_mode,
                cubic_coeff_a=op.cubic_coeff_a,
                extrapolation_value=op.extrapolation_value,
                exclude_outside=op.exclude_outside,
                antialias=op.antialias,
                keep_aspect_ratio_policy=op.keep_aspect_ratio_policy,
                dtype=op.dtype,
            )
        if isinstance(op, GridSampleOp):
            return GridSampleOp(
                input0=name_map.get(op.input0, op.input0),
                grid=name_map.get(op.grid, op.grid),
                output=name_map.get(op.output, op.output),
                input_shape=op.input_shape,
                grid_shape=op.grid_shape,
                output_shape=op.output_shape,
                spatial_rank=op.spatial_rank,
                input_spatial=op.input_spatial,
                output_spatial=op.output_spatial,
                mode=op.mode,
                padding_mode=op.padding_mode,
                align_corners=op.align_corners,
                dtype=op.dtype,
                grid_dtype=op.grid_dtype,
            )
        if isinstance(op, ReduceOp):
            return ReduceOp(
                input0=name_map.get(op.input0, op.input0),
                output=name_map.get(op.output, op.output),
                input_shape=op.input_shape,
                output_shape=op.output_shape,
                axes=op.axes,
                axes_input=self._map_optional_name(name_map, op.axes_input),
                axes_input_shape=op.axes_input_shape,
                axes_input_dtype=op.axes_input_dtype,
                keepdims=op.keepdims,
                noop_with_empty_axes=op.noop_with_empty_axes,
                reduce_kind=op.reduce_kind,
                reduce_count=op.reduce_count,
                dtype=op.dtype,
            )
        if isinstance(op, ArgReduceOp):
            return ArgReduceOp(
                input0=name_map.get(op.input0, op.input0),
                output=name_map.get(op.output, op.output),
                input_shape=op.input_shape,
                output_shape=op.output_shape,
                axis=op.axis,
                keepdims=op.keepdims,
                select_last_index=op.select_last_index,
                reduce_kind=op.reduce_kind,
                input_dtype=op.input_dtype,
                output_dtype=op.output_dtype,
            )
        if isinstance(op, TopKOp):
            return TopKOp(
                input0=name_map.get(op.input0, op.input0),
                output_values=name_map.get(op.output_values, op.output_values),
                output_indices=name_map.get(op.output_indices, op.output_indices),
                input_shape=op.input_shape,
                output_shape=op.output_shape,
                axis=op.axis,
                k=op.k,
                largest=op.largest,
                sorted=op.sorted,
                input_dtype=op.input_dtype,
                output_values_dtype=op.output_values_dtype,
                output_indices_dtype=op.output_indices_dtype,
            )
        if isinstance(op, ConstantOfShapeOp):
            return ConstantOfShapeOp(
                input0=name_map.get(op.input0, op.input0),
                output=name_map.get(op.output, op.output),
                input_shape=op.input_shape,
                shape=op.shape,
                value=op.value,
                dtype=op.dtype,
                input_dtype=op.input_dtype,
            )
        if isinstance(op, ShapeOp):
            return ShapeOp(
                input0=name_map.get(op.input0, op.input0),
                output=name_map.get(op.output, op.output),
                input_shape=op.input_shape,
                output_shape=op.output_shape,
                values=op.values,
                dtype=op.dtype,
                input_dtype=op.input_dtype,
            )
        if isinstance(op, SizeOp):
            return SizeOp(
                input0=name_map.get(op.input0, op.input0),
                output=name_map.get(op.output, op.output),
                input_shape=op.input_shape,
                output_shape=op.output_shape,
                value=op.value,
                dtype=op.dtype,
                input_dtype=op.input_dtype,
            )
        if isinstance(op, NonZeroOp):
            return NonZeroOp(
                input0=name_map.get(op.input0, op.input0),
                output=name_map.get(op.output, op.output),
                input_shape=op.input_shape,
                output_shape=op.output_shape,
                dtype=op.dtype,
                input_dtype=op.input_dtype,
            )
        if isinstance(op, NonMaxSuppressionOp):
            return NonMaxSuppressionOp(
                boxes=name_map.get(op.boxes, op.boxes),
                scores=name_map.get(op.scores, op.scores),
                max_output_boxes_per_class=self._map_optional_name(
                    name_map, op.max_output_boxes_per_class
                ),
                iou_threshold=self._map_optional_name(
                    name_map, op.iou_threshold
                ),
                score_threshold=self._map_optional_name(
                    name_map, op.score_threshold
                ),
                output=name_map.get(op.output, op.output),
                boxes_shape=op.boxes_shape,
                scores_shape=op.scores_shape,
                output_shape=op.output_shape,
                center_point_box=op.center_point_box,
                boxes_dtype=op.boxes_dtype,
                output_dtype=op.output_dtype,
                max_output_dtype=op.max_output_dtype,
                max_output_shape=op.max_output_shape,
                iou_threshold_dtype=op.iou_threshold_dtype,
                iou_threshold_shape=op.iou_threshold_shape,
                score_threshold_dtype=op.score_threshold_dtype,
                score_threshold_shape=op.score_threshold_shape,
            )
        if isinstance(op, ExpandOp):
            return ExpandOp(
                input0=name_map.get(op.input0, op.input0),
                output=name_map.get(op.output, op.output),
                input_shape=op.input_shape,
                output_shape=op.output_shape,
                input_shape_padded=op.input_shape_padded,
                input_strides=op.input_strides,
                dtype=op.dtype,
                input_dtype=op.input_dtype,
            )
        if isinstance(op, CumSumOp):
            return CumSumOp(
                input0=name_map.get(op.input0, op.input0),
                axis_input=self._map_optional_name(name_map, op.axis_input),
                axis_input_dtype=op.axis_input_dtype,
                axis=op.axis,
                output=name_map.get(op.output, op.output),
                input_shape=op.input_shape,
                dtype=op.dtype,
                input_dtype=op.input_dtype,
                exclusive=op.exclusive,
                reverse=op.reverse,
            )
        if isinstance(op, RangeOp):
            return RangeOp(
                start=name_map.get(op.start, op.start),
                limit=name_map.get(op.limit, op.limit),
                delta=name_map.get(op.delta, op.delta),
                output=name_map.get(op.output, op.output),
                output_shape=op.output_shape,
                length=op.length,
                dtype=op.dtype,
                input_dtype=op.input_dtype,
            )
        if isinstance(op, OneHotOp):
            return OneHotOp(
                indices=name_map.get(op.indices, op.indices),
                depth=name_map.get(op.depth, op.depth),
                values=name_map.get(op.values, op.values),
                output=name_map.get(op.output, op.output),
                axis=op.axis,
                indices_shape=op.indices_shape,
                values_shape=op.values_shape,
                output_shape=op.output_shape,
                depth_dim=op.depth_dim,
                dtype=op.dtype,
                indices_dtype=op.indices_dtype,
                depth_dtype=op.depth_dtype,
            )
        if isinstance(op, SplitOp):
            return SplitOp(
                input0=name_map.get(op.input0, op.input0),
                outputs=tuple(
                    name_map.get(name, name) for name in op.outputs
                ),
                input_shape=op.input_shape,
                output_shapes=op.output_shapes,
                axis=op.axis,
                split_sizes=op.split_sizes,
                dtype=op.dtype,
                input_dtype=op.input_dtype,
            )
        return UnaryOp(
            input0=name_map.get(op.input0, op.input0),
            output=name_map.get(op.output, op.output),
            function=op.function,
            shape=op.shape,
            dtype=op.dtype,
        )

    def _sanitize_model_names_with_map(
        self, model: LoweredModel
    ) -> tuple[LoweredModel, dict[str, str]]:
        name_map = self._build_name_map(model)
        constants = tuple(
            ConstTensor(
                name=name_map.get(const.name, const.name),
                shape=const.shape,
                data=const.data,
                dtype=const.dtype,
            )
            for const in model.constants
        )
        ops = tuple(self._map_op_names(op, name_map) for op in model.ops)
        sanitized = LoweredModel(
            name=name_map.get(model.name, model.name),
            input_names=tuple(
                name_map.get(name, name) for name in model.input_names
            ),
            input_shapes=model.input_shapes,
            input_dtypes=model.input_dtypes,
            output_names=tuple(
                name_map.get(name, name) for name in model.output_names
            ),
            output_shapes=model.output_shapes,
            output_dtypes=model.output_dtypes,
            constants=constants,
            ops=ops,
            node_infos=model.node_infos,
            header=model.header,
        )
        return sanitized, name_map

    def _sanitize_model_names(self, model: LoweredModel) -> LoweredModel:
        return self._sanitize_model_names_with_map(model)[0]

    @staticmethod
    def _sanitize_testbench_inputs(
        testbench_inputs: Mapping[str, tuple[float | int | bool, ...]] | None,
        name_map: Mapping[str, str],
    ) -> Mapping[str, tuple[float | int | bool, ...]] | None:
        if not testbench_inputs:
            return None
        return {
            name_map.get(name, name): values
            for name, values in testbench_inputs.items()
        }

    def _load_templates(self, emit_testbench: bool) -> dict[str, Template]:
        try:
            templates = {
                "binary": self._env.get_template("binary_op.c.j2"),
                "multi_input": self._env.get_template(
                    "multi_input_op.c.j2"
                ),
                "where": self._env.get_template("where_op.c.j2"),
                "unary": self._env.get_template("unary_op.c.j2"),
                "clip": self._env.get_template("clip_op.c.j2"),
                "cast": self._env.get_template("cast_op.c.j2"),
                "quantize_linear": self._env.get_template(
                    "quantize_linear_op.c.j2"
                ),
                "qlinear_matmul": self._env.get_template(
                    "qlinear_matmul_op.c.j2"
                ),
                "matmul": self._env.get_template("matmul_op.c.j2"),
                "einsum": self._env.get_template("einsum_op.c.j2"),
                "gemm": self._env.get_template("gemm_op.c.j2"),
                "attention": self._env.get_template("attention_op.c.j2"),
                "rotary_embedding": self._env.get_template(
                    "rotary_embedding_op.c.j2"
                ),
                "conv": self._env.get_template("conv_op.c.j2"),
                "conv_transpose": self._env.get_template(
                    "conv_transpose_op.c.j2"
                ),
                "avg_pool": self._env.get_template("average_pool_op.c.j2"),
                "lp_pool": self._env.get_template("lp_pool_op.c.j2"),
                "batch_norm": self._env.get_template("batch_norm_op.c.j2"),
                "lp_norm": self._env.get_template("lp_normalization_op.c.j2"),
                "instance_norm": self._env.get_template(
                    "instance_normalization_op.c.j2"
                ),
                "group_norm": self._env.get_template(
                    "group_normalization_op.c.j2"
                ),
                "layer_norm": self._env.get_template(
                    "layer_normalization_op.c.j2"
                ),
                "mean_variance_norm": self._env.get_template(
                    "mean_variance_normalization_op.c.j2"
                ),
                "rms_norm": self._env.get_template("rms_normalization_op.c.j2"),
                "lrn": self._env.get_template("lrn_op.c.j2"),
                "lstm": self._env.get_template("lstm_op.c.j2"),
                "adagrad": self._env.get_template("adagrad_op.c.j2"),
                "softmax": self._env.get_template("softmax_op.c.j2"),
                "logsoftmax": self._env.get_template("logsoftmax_op.c.j2"),
                "hardmax": self._env.get_template("hardmax_op.c.j2"),
                "nllloss": self._env.get_template(
                    "negative_log_likelihood_loss_op.c.j2"
                ),
                "softmax_cross_entropy_loss": self._env.get_template(
                    "softmax_cross_entropy_loss_op.c.j2"
                ),
                "maxpool": self._env.get_template("maxpool_op.c.j2"),
                "concat": self._env.get_template("concat_op.c.j2"),
                "gather_elements": self._env.get_template("gather_elements_op.c.j2"),
                "gather": self._env.get_template("gather_op.c.j2"),
                "gather_nd": self._env.get_template("gather_nd_op.c.j2"),
                "scatter_nd": self._env.get_template("scatter_nd_op.c.j2"),
                "tensor_scatter": self._env.get_template(
                    "tensor_scatter_op.c.j2"
                ),
                "transpose": self._env.get_template("transpose_op.c.j2"),
                "reshape": self._env.get_template("reshape_op.c.j2"),
                "identity": self._env.get_template("identity_op.c.j2"),
                "eye_like": self._env.get_template("eye_like_op.c.j2"),
                "trilu": self._env.get_template("trilu_op.c.j2"),
                "tile": self._env.get_template("tile_op.c.j2"),
                "pad": self._env.get_template("pad_op.c.j2"),
                "depth_to_space": self._env.get_template("depth_to_space_op.c.j2"),
                "space_to_depth": self._env.get_template("space_to_depth_op.c.j2"),
                "slice": self._env.get_template("slice_op.c.j2"),
                "slice_dynamic": self._env.get_template(
                    "slice_op_dynamic.c.j2"
                ),
                "resize": self._env.get_template("resize_op.c.j2"),
                "grid_sample": self._env.get_template("grid_sample_op.c.j2"),
                "reduce": self._env.get_template("reduce_op.c.j2"),
                "reduce_dynamic": self._env.get_template(
                    "reduce_op_dynamic.c.j2"
                ),
                "arg_reduce": self._env.get_template("arg_reduce_op.c.j2"),
                "topk": self._env.get_template("topk_op.c.j2"),
                "constant_of_shape": self._env.get_template(
                    "constant_of_shape_op.c.j2"
                ),
                "shape": self._env.get_template("shape_op.c.j2"),
                "size": self._env.get_template("size_op.c.j2"),
                "nonzero": self._env.get_template("nonzero_op.c.j2"),
                "nonmax_suppression": self._env.get_template(
                    "nonmax_suppression_op.c.j2"
                ),
                "expand": self._env.get_template("expand_op.c.j2"),
                "cumsum": self._env.get_template("cumsum_op.c.j2"),
                "range": self._env.get_template("range_op.c.j2"),
                "one_hot": self._env.get_template("one_hot_op.c.j2"),
                "split": self._env.get_template("split_op.c.j2"),
            }
            if emit_testbench:
                templates["testbench"] = self._env.get_template("testbench.c.j2")
        except Exception as exc:  # pragma: no cover - template load failure
            raise CodegenError("Failed to load C template") from exc
        return templates

    def emit_model(
        self,
        model: LoweredModel,
        *,
        emit_testbench: bool = False,
        testbench_inputs: Mapping[str, tuple[float | int | bool, ...]] | None = None,
        variable_dim_inputs: Mapping[int, Mapping[int, str]] | None = None,
        variable_dim_outputs: Mapping[int, Mapping[int, str]] | None = None,
    ) -> str:
        model, name_map = self._sanitize_model_names_with_map(model)
        testbench_inputs = self._sanitize_testbench_inputs(
            testbench_inputs, name_map
        )
        inline_constants, large_constants = self._partition_constants(
            model.constants
        )
        (
            dim_order,
            input_dim_names,
            output_dim_names,
            dim_values,
        ) = self._build_variable_dim_names(
            model,
            variable_dim_inputs,
            variable_dim_outputs,
        )
        tensor_dim_names = self._build_tensor_dim_names(
            model,
            input_dim_names,
            output_dim_names,
        )
        dim_args = self._format_dim_args_prefix(dim_order)
        self._env.globals["dim_args"] = dim_args
        templates = self._load_templates(emit_testbench)
        scalar_registry = ScalarFunctionRegistry()
        testbench_template = templates.get("testbench")
        reserved_names = {
            model.name,
            *model.input_names,
            *model.output_names,
            *(const.name for const in model.constants),
        }
        temp_buffers = self._temp_buffers(model, reserved_names=reserved_names)
        temp_name_map = {
            original: buffer.name for original, buffer in temp_buffers.items()
        }
        resolved_ops = [self._resolve_op(op, temp_name_map) for op in model.ops]
        self._propagate_tensor_dim_names(resolved_ops, tensor_dim_names)
        self._emit_state = _EmitState(
            model=model,
            templates=templates,
            scalar_registry=scalar_registry,
            dim_args=dim_args,
            tensor_dim_names=tensor_dim_names,
        )
        operator_fns = "\n\n".join(
            op.emit(self, EmitContext(op_index=index))
            for index, op in enumerate(resolved_ops)
        )
        wrapper_fn = self._emit_model_wrapper(
            model,
            resolved_ops,
            tuple(temp_buffers.values()),
            dim_order=dim_order,
            input_dim_names=input_dim_names,
            output_dim_names=output_dim_names,
        )
        scalar_functions = scalar_registry.render()
        scalar_include_lines = (
            scalar_registry.include_lines() if scalar_functions else []
        )
        scalar_includes = {
            line for line in scalar_include_lines if line.startswith("#include ")
        }
        scalar_preamble = [
            line for line in scalar_include_lines if not line.startswith("#include ")
        ]
        testbench_math_include = set()
        if emit_testbench and self._testbench_requires_math(
            model, testbench_inputs
        ):
            testbench_math_include.add("#include <math.h>")
        includes = self._collect_includes(
            model,
            resolved_ops,
            emit_testbench=emit_testbench,
            extra_includes=scalar_includes | testbench_math_include,
            needs_weight_loader=bool(large_constants),
        )
        sections = [
            self._emit_header_comment(model.header),
            "",
            *includes,
            "",
            self._emit_index_type_define(),
        ]
        if scalar_preamble:
            sections.extend(("", *scalar_preamble))
        sections.append("")
        constants_section = self._emit_constant_definitions(inline_constants)
        if constants_section:
            sections.extend((constants_section.rstrip(), ""))
        large_constants_section = self._emit_constant_storage_definitions(
            large_constants
        )
        if large_constants_section:
            sections.extend((large_constants_section.rstrip(), ""))
        if scalar_functions:
            sections.extend(("\n".join(scalar_functions), ""))
        weight_loader = self._emit_weight_loader(model, large_constants)
        sections.extend(
            (
                operator_fns.rstrip(),
                "",
                weight_loader.rstrip(),
                "",
                wrapper_fn,
            )
        )
        if emit_testbench and testbench_template is not None:
            sections.extend(
                (
                    "",
                    self._emit_testbench(
                        model,
                        testbench_template,
                        testbench_inputs=testbench_inputs,
                        dim_order=dim_order,
                        dim_values=dim_values,
                        weight_data_filename=self._weight_data_filename(model),
                    ),
                )
            )
        sections.append("")
        rendered = "\n".join(sections)
        if not rendered.endswith("\n"):
            rendered += "\n"
        return rendered

    def emit_model_with_data_file(
        self,
        model: LoweredModel,
        *,
        emit_testbench: bool = False,
        testbench_inputs: Mapping[str, tuple[float | int | bool, ...]] | None = None,
        variable_dim_inputs: Mapping[int, Mapping[int, str]] | None = None,
        variable_dim_outputs: Mapping[int, Mapping[int, str]] | None = None,
    ) -> tuple[str, str]:
        model, name_map = self._sanitize_model_names_with_map(model)
        testbench_inputs = self._sanitize_testbench_inputs(
            testbench_inputs, name_map
        )
        inline_constants, large_constants = self._partition_constants(
            model.constants
        )
        (
            dim_order,
            input_dim_names,
            output_dim_names,
            dim_values,
        ) = self._build_variable_dim_names(
            model,
            variable_dim_inputs,
            variable_dim_outputs,
        )
        tensor_dim_names = self._build_tensor_dim_names(
            model,
            input_dim_names,
            output_dim_names,
        )
        dim_args = self._format_dim_args_prefix(dim_order)
        self._env.globals["dim_args"] = dim_args
        templates = self._load_templates(emit_testbench)
        scalar_registry = ScalarFunctionRegistry()
        testbench_template = templates.get("testbench")
        reserved_names = {
            model.name,
            *model.input_names,
            *model.output_names,
            *(const.name for const in model.constants),
        }
        temp_buffers = self._temp_buffers(model, reserved_names=reserved_names)
        temp_name_map = {
            original: buffer.name for original, buffer in temp_buffers.items()
        }
        resolved_ops = [self._resolve_op(op, temp_name_map) for op in model.ops]
        self._propagate_tensor_dim_names(resolved_ops, tensor_dim_names)
        self._emit_state = _EmitState(
            model=model,
            templates=templates,
            scalar_registry=scalar_registry,
            dim_args=dim_args,
            tensor_dim_names=tensor_dim_names,
        )
        operator_fns = "\n\n".join(
            op.emit(self, EmitContext(op_index=index))
            for index, op in enumerate(resolved_ops)
        )
        wrapper_fn = self._emit_model_wrapper(
            model,
            resolved_ops,
            tuple(temp_buffers.values()),
            dim_order=dim_order,
            input_dim_names=input_dim_names,
            output_dim_names=output_dim_names,
        )
        scalar_functions = scalar_registry.render()
        scalar_include_lines = (
            scalar_registry.include_lines() if scalar_functions else []
        )
        scalar_includes = {
            line for line in scalar_include_lines if line.startswith("#include ")
        }
        scalar_preamble = [
            line for line in scalar_include_lines if not line.startswith("#include ")
        ]
        testbench_math_include = set()
        if emit_testbench and self._testbench_requires_math(
            model, testbench_inputs
        ):
            testbench_math_include.add("#include <math.h>")
        includes = self._collect_includes(
            model,
            resolved_ops,
            emit_testbench=emit_testbench,
            extra_includes=scalar_includes | testbench_math_include,
            needs_weight_loader=bool(large_constants),
        )
        sections = [
            self._emit_header_comment(model.header),
            "",
            *includes,
            "",
            self._emit_index_type_define(),
        ]
        if scalar_preamble:
            sections.extend(("", *scalar_preamble))
        sections.append("")
        constants_section = self._emit_constant_declarations(inline_constants)
        if constants_section:
            sections.extend((constants_section.rstrip(), ""))
        large_constants_section = self._emit_constant_storage_definitions(
            large_constants
        )
        if large_constants_section:
            sections.extend((large_constants_section.rstrip(), ""))
        if scalar_functions:
            sections.extend(("\n".join(scalar_functions), ""))
        weight_loader = self._emit_weight_loader(model, large_constants)
        sections.extend(
            (
                operator_fns.rstrip(),
                "",
                weight_loader.rstrip(),
                "",
                wrapper_fn,
            )
        )
        if emit_testbench and testbench_template is not None:
            sections.extend(
                (
                    "",
                    self._emit_testbench(
                        model,
                        testbench_template,
                        testbench_inputs=testbench_inputs,
                        dim_order=dim_order,
                        dim_values=dim_values,
                        weight_data_filename=self._weight_data_filename(model),
                    ),
                )
            )
        sections.append("")
        main_rendered = "\n".join(sections)
        if not main_rendered.endswith("\n"):
            main_rendered += "\n"
        data_includes = self._collect_constant_includes(inline_constants)
        data_sections = [self._emit_header_comment(model.header), ""]
        if data_includes:
            data_sections.extend((*data_includes, ""))
        else:
            data_sections.append("")
        data_constants = self._emit_constant_definitions(
            inline_constants, storage_prefix="const"
        )
        if data_constants:
            data_sections.append(data_constants.rstrip())
        data_sections.append("")
        data_rendered = "\n".join(data_sections)
        if not data_rendered.endswith("\n"):
            data_rendered += "\n"
        return main_rendered, data_rendered

    @staticmethod
    def _emit_header_comment(header: ModelHeader) -> str:
        lines: list[str] = [header.generator, ""]
        lines.append(f"Command line: {header.command_line or 'n/a'}")
        lines.append(f"Model checksum (sha256): {header.model_checksum or 'n/a'}")
        lines.append(f"Model name: {header.model_name or 'n/a'}")
        lines.append(f"Graph name: {header.graph_name or 'n/a'}")
        lines.append(
            "Inputs: "
            f"{header.input_count} Outputs: {header.output_count} "
            f"Nodes: {header.node_count} Initializers: {header.initializer_count}"
        )
        lines.append(f"IR version: {header.ir_version or 'n/a'}")
        lines.append(f"Model version: {header.model_version or 'n/a'}")
        lines.append(f"Domain: {header.domain or 'n/a'}")
        producer = header.producer_name or "n/a"
        producer_version = header.producer_version or "n/a"
        lines.append(f"Producer: {producer} (version: {producer_version})")
        if header.opset_imports:
            opset_items = []
            for domain, version in header.opset_imports:
                opset_domain = domain or "ai.onnx"
                opset_items.append(f"{opset_domain}={version}")
            lines.append(f"Opset imports: {', '.join(opset_items)}")
        else:
            lines.append("Opset imports: n/a")
        lines.append("Description:")
        lines.extend(_format_multiline_value(header.description))
        lines.append("Graph description:")
        lines.extend(_format_multiline_value(header.graph_description))
        lines.append("Metadata:")
        if header.metadata_props:
            for key, value in header.metadata_props:
                lines.append(f"  {key}: {value}")
        else:
            lines.append("  n/a")
        comment_lines = ["/*"]
        comment_lines.extend(
            f" * {line}" if line else " *" for line in lines
        )
        comment_lines.append(" */")
        return "\n".join(comment_lines)

    @staticmethod
    def _format_node_attr_value(value: object) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        if isinstance(value, str):
            return value
        if isinstance(value, (list, tuple)):
            rendered = ", ".join(CEmitter._format_node_attr_value(item) for item in value)
            return f"[{rendered}]"
        if hasattr(value, "tolist"):
            try:
                return CEmitter._format_node_attr_value(value.tolist())
            except Exception:
                return repr(value)
        return repr(value)

    @staticmethod
    def _emit_node_comment(node_info: NodeInfo, index: int) -> str:
        lines = [
            f"Node {index}:",
            f"OpType: {node_info.op_type}",
            f"Name: {node_info.name}" if node_info.name else "Name: n/a",
            "Inputs: "
            + (", ".join(node_info.inputs) if node_info.inputs else "n/a"),
            "Outputs: "
            + (", ".join(node_info.outputs) if node_info.outputs else "n/a"),
        ]
        if node_info.attrs:
            lines.append("Attrs:")
            for key, value in sorted(node_info.attrs.items()):
                rendered = CEmitter._format_node_attr_value(value)
                lines.append(f"  {key}: {rendered}")
        else:
            lines.append("Attrs: n/a")
        comment_lines = ["/*"]
        comment_lines.extend(
            f" * {line}" if line else " *" for line in lines
        )
        comment_lines.append(" */")
        return "\n".join(comment_lines)

    @staticmethod
    def _emit_constant_comment(constant: ConstTensor, index: int) -> str:
        shape = constant.shape
        lines = [
            f"Weight {index}:",
            f"Name: {constant.name}",
            f"Shape: {shape if shape else '[]'}",
            f"Elements: {CEmitter._element_count(shape)}",
            f"Dtype: {constant.dtype.onnx_name}",
        ]
        comment_lines = ["/*"]
        comment_lines.extend(
            f" * {line}" if line else " *" for line in lines
        )
        comment_lines.append(" */")
        return "\n".join(comment_lines)

    @staticmethod
    def _collect_constant_includes(constants: tuple[ConstTensor, ...]) -> list[str]:
        if not constants:
            return []
        includes: list[str] = []
        dtypes = {const.dtype for const in constants}
        if any(
            dtype
            in {
                ScalarType.I64,
                ScalarType.I32,
                ScalarType.I16,
                ScalarType.I8,
                ScalarType.U64,
                ScalarType.U32,
                ScalarType.U16,
                ScalarType.U8,
            }
            for dtype in dtypes
        ):
            includes.append("#include <stdint.h>")
        if ScalarType.BOOL in dtypes:
            includes.append("#include <stdbool.h>")
        return includes

    @staticmethod
    def _scalar_function_name(
        function: ScalarFunction,
        dtype: ScalarType,
        registry: ScalarFunctionRegistry,
        *,
        params: tuple[float, ...] = (),
    ) -> str | None:
        registry_functions = {
            ScalarFunction.ABS,
            ScalarFunction.ADD,
            ScalarFunction.ACOS,
            ScalarFunction.ACOSH,
            ScalarFunction.AFFINE,
            ScalarFunction.LOGICAL_AND,
            ScalarFunction.ASIN,
            ScalarFunction.ASINH,
            ScalarFunction.ATAN,
            ScalarFunction.ATANH,
            ScalarFunction.BITWISE_AND,
            ScalarFunction.BITWISE_NOT,
            ScalarFunction.BITWISE_OR,
            ScalarFunction.BITWISE_XOR,
            ScalarFunction.CEIL,
            ScalarFunction.CELU,
            ScalarFunction.COS,
            ScalarFunction.COSH,
            ScalarFunction.DIV,
            ScalarFunction.ELU,
            ScalarFunction.ERF,
            ScalarFunction.EXP,
            ScalarFunction.FLOOR,
            ScalarFunction.GELU,
            ScalarFunction.HARDSIGMOID,
            ScalarFunction.HARDSWISH,
            ScalarFunction.LOG,
            ScalarFunction.FMOD,
            ScalarFunction.REMAINDER,
            ScalarFunction.LEAKY_RELU,
            ScalarFunction.MISH,
            ScalarFunction.MUL,
            ScalarFunction.NEG,
            ScalarFunction.LOGICAL_NOT,
            ScalarFunction.LOGICAL_OR,
            ScalarFunction.POW,
            ScalarFunction.RECIPROCAL,
            ScalarFunction.RELU,
            ScalarFunction.ROUND,
            ScalarFunction.SELU,
            ScalarFunction.SHRINK,
            ScalarFunction.SIGMOID,
            ScalarFunction.SIGN,
            ScalarFunction.SIN,
            ScalarFunction.SINH,
            ScalarFunction.SOFTPLUS,
            ScalarFunction.SOFTSIGN,
            ScalarFunction.SQRT,
            ScalarFunction.SUB,
            ScalarFunction.SWISH,
            ScalarFunction.TAN,
            ScalarFunction.TANH,
            ScalarFunction.SCALED_TANH,
            ScalarFunction.THRESHOLDED_RELU,
            ScalarFunction.LOGICAL_XOR,
            ScalarFunction.ISNEGINF,
            ScalarFunction.ISPOSINF,
        }
        if function in {ScalarFunction.MAXIMUM, ScalarFunction.MINIMUM}:
            if dtype in {ScalarType.F32, ScalarType.F64}:
                scalar_function = (
                    ScalarFunction.FMAX
                    if function == ScalarFunction.MAXIMUM
                    else ScalarFunction.FMIN
                )
            else:
                scalar_function = function
        elif function in registry_functions:
            scalar_function = function
        else:
            return None
        try:
            return registry.request(
                ScalarFunctionKey(
                    function=scalar_function, return_type=dtype, params=params
                )
            )
        except ScalarFunctionError:
            return None

    def _lstm_activation_function_name(
        self,
        kind: int,
        alpha: float,
        beta: float,
        dtype: ScalarType,
        registry: ScalarFunctionRegistry,
    ) -> str:
        spec = _LSTM_ACTIVATION_SPECS.get(kind)
        if spec is None:
            raise CodegenError(
                f"Unsupported LSTM activation kind for codegen: {kind}"
            )
        function, param_count = spec
        if param_count == 0:
            params = ()
        elif param_count == 1:
            params = (alpha,)
        else:
            params = (alpha, beta)
        name = self._scalar_function_name(
            function, dtype, registry, params=params
        )
        if name is None:
            raise CodegenError(
                f"Failed to resolve scalar function for LSTM activation kind {kind}"
            )
        return name

    @staticmethod
    def _collect_includes(
        model: LoweredModel,
        resolved_ops: list[
            BinaryOp
            | WhereOp
            | UnaryOp
            | ClipOp
            | CastOp
            | QuantizeLinearOp
            | QLinearMatMulOp
            | MatMulOp
            | EinsumOp
            | GemmOp
            | AttentionOp
            | ConvOp
            | ConvTransposeOp
            | AveragePoolOp
            | LpPoolOp
            | BatchNormOp
            | LpNormalizationOp
            | InstanceNormalizationOp
            | GroupNormalizationOp
            | LayerNormalizationOp
            | MeanVarianceNormalizationOp
            | RMSNormalizationOp
            | LrnOp
            | LstmOp
            | AdagradOp
            | SoftmaxOp
            | LogSoftmaxOp
            | HardmaxOp
            | NegativeLogLikelihoodLossOp
            | SoftmaxCrossEntropyLossOp
            | MaxPoolOp
            | ConcatOp
            | GatherElementsOp
            | GatherOp
            | GatherNDOp
            | ScatterNDOp
            | TensorScatterOp
            | TransposeOp
            | ReshapeOp
            | IdentityOp
            | EyeLikeOp
            | TriluOp
            | TileOp
            | DepthToSpaceOp
            | SpaceToDepthOp
            | SliceOp
            | ResizeOp
            | GridSampleOp
            | ReduceOp
            | ArgReduceOp
            | TopKOp
            | ConstantOfShapeOp
            | ShapeOp
            | SizeOp
            | NonZeroOp
            | NonMaxSuppressionOp
            | ExpandOp
            | CumSumOp
            | RangeOp
            | OneHotOp
            | SplitOp
        ],
        *,
        emit_testbench: bool,
        extra_includes: set[str] | None = None,
        needs_weight_loader: bool = False,
    ) -> list[str]:
        includes: set[str] = {"#include <stdint.h>"}
        if emit_testbench:
            includes.add("#include <stdio.h>")
        if needs_weight_loader:
            includes.add("#include <stdio.h>")
        if extra_includes:
            includes.update(extra_includes)
        if any(
            isinstance(op, UnaryOp)
            and op.function in {ScalarFunction.ISINF, ScalarFunction.ISNAN}
            for op in resolved_ops
        ):
            includes.add("#include <math.h>")
        constant_of_shape_inputs = {
            op.input_dtype
            for op in resolved_ops
            if isinstance(op, ConstantOfShapeOp)
        }
        model_dtypes = {
            *model.input_dtypes,
            *model.output_dtypes,
            *(const.dtype for const in model.constants),
            *constant_of_shape_inputs,
        }
        model_dtypes.update(
            op.dtype
            for op in resolved_ops
            if not isinstance(op, (ArgReduceOp, TopKOp))
        )
        arg_reduce_dtypes = {
            dtype
            for op in resolved_ops
            if isinstance(op, ArgReduceOp)
            for dtype in (op.input_dtype, op.output_dtype)
        }
        model_dtypes.update(arg_reduce_dtypes)
        topk_dtypes = {
            dtype
            for op in resolved_ops
            if isinstance(op, TopKOp)
            for dtype in (
                op.input_dtype,
                op.output_values_dtype,
                op.output_indices_dtype,
            )
        }
        model_dtypes.update(topk_dtypes)
        slice_input_dtypes = {
            dtype
            for op in resolved_ops
            if isinstance(op, SliceOp)
            for dtype in (
                op.starts_dtype,
                op.ends_dtype,
                op.axes_dtype,
                op.steps_dtype,
            )
            if dtype is not None
        }
        trilu_k_dtypes = {
            op.k_input_dtype
            for op in resolved_ops
            if isinstance(op, TriluOp) and op.k_input_dtype is not None
        }
        maxpool_indices_dtypes = {
            op.indices_dtype
            for op in resolved_ops
            if isinstance(op, MaxPoolOp) and op.indices_dtype is not None
        }
        model_dtypes.update(maxpool_indices_dtypes)
        model_dtypes.update(trilu_k_dtypes)
        nll_target_dtypes = {
            op.target_dtype
            for op in resolved_ops
            if isinstance(op, NegativeLogLikelihoodLossOp)
        }
        sce_target_dtypes = {
            op.target_dtype
            for op in resolved_ops
            if isinstance(op, SoftmaxCrossEntropyLossOp)
        }
        if CEmitter._needs_stdint(
            model_dtypes,
            (nll_target_dtypes, sce_target_dtypes, slice_input_dtypes),
            has_resize=any(isinstance(op, ResizeOp) for op in resolved_ops),
        ):
            includes.add("#include <stdint.h>")
        if ScalarType.BOOL in model_dtypes:
            includes.add("#include <stdbool.h>")
        if any(
            isinstance(op, ReduceOp) and op.axes_input is not None
            for op in resolved_ops
        ):
            includes.add("#include <stdbool.h>")
        if any(
            isinstance(op, SoftmaxCrossEntropyLossOp)
            and op.ignore_index is not None
            for op in resolved_ops
        ):
            includes.add("#include <stdbool.h>")
        if any(
            isinstance(op, UnaryOp)
            and unary_op_symbol(op.function, dtype=op.dtype) in {"llabs", "abs"}
            for op in resolved_ops
        ):
            includes.add("#include <stdlib.h>")
        if any(isinstance(op, PadOp) for op in resolved_ops):
            includes.add("#include <stddef.h>")
        if CEmitter._needs_math(resolved_ops):
            includes.add("#include <math.h>")
        if CEmitter._needs_limits(resolved_ops):
            includes.add("#include <limits.h>")
        if any(
            isinstance(op, (ConcatOp, ReshapeOp, SplitOp, IdentityOp))
            for op in resolved_ops
        ):
            includes.add("#include <string.h>")
        ordered_includes = (
            "#include <stdint.h>",
            "#include <stdio.h>",
            "#include <stddef.h>",
            "#include <stdbool.h>",
            "#include <stdlib.h>",
            "#include <math.h>",
            "#include <float.h>",
            "#include <limits.h>",
            "#include <string.h>",
        )
        return [include for include in ordered_includes if include in includes]

    @staticmethod
    def _emit_index_type_define() -> str:
        return "\n".join(
            (
                "#ifndef idx_t",
                "#define idx_t int32_t",
                "#endif",
            )
        )

    @staticmethod
    def _needs_stdint(
        model_dtypes: set[ScalarType],
        targets: tuple[set[ScalarType], ...],
        *,
        has_resize: bool,
    ) -> bool:
        integer_dtypes = {
            ScalarType.I64,
            ScalarType.I32,
            ScalarType.I16,
            ScalarType.I8,
            ScalarType.U64,
            ScalarType.U32,
            ScalarType.U16,
            ScalarType.U8,
        }
        if any(dtype in integer_dtypes for dtype in model_dtypes):
            return True
        if any(
            dtype in {ScalarType.I64, ScalarType.I32}
            for target_dtypes in targets
            for dtype in target_dtypes
        ):
            return True
        return has_resize

    @staticmethod
    def _needs_math(
        resolved_ops: list[
            BinaryOp
            | UnaryOp
            | ClipOp
            | CastOp
            | QuantizeLinearOp
            | QLinearMatMulOp
            | MatMulOp
            | EinsumOp
            | GemmOp
            | AttentionOp
            | ConvOp
            | ConvTransposeOp
            | AveragePoolOp
            | LpPoolOp
            | BatchNormOp
            | LpNormalizationOp
            | InstanceNormalizationOp
            | GroupNormalizationOp
            | LayerNormalizationOp
            | MeanVarianceNormalizationOp
            | RMSNormalizationOp
            | LrnOp
            | LstmOp
            | AdagradOp
            | SoftmaxOp
            | LogSoftmaxOp
            | HardmaxOp
            | NegativeLogLikelihoodLossOp
            | SoftmaxCrossEntropyLossOp
            | MaxPoolOp
            | ConcatOp
            | GatherElementsOp
            | GatherOp
            | GatherNDOp
            | ScatterNDOp
            | TensorScatterOp
            | TransposeOp
            | ReshapeOp
            | IdentityOp
            | EyeLikeOp
            | TriluOp
            | TileOp
            | DepthToSpaceOp
            | SpaceToDepthOp
            | SliceOp
            | ResizeOp
            | GridSampleOp
            | ReduceOp
            | ArgReduceOp
            | TopKOp
            | ConstantOfShapeOp
            | ShapeOp
            | SizeOp
            | NonZeroOp
            | NonMaxSuppressionOp
            | ExpandOp
            | CumSumOp
            | RangeOp
            | OneHotOp
            | SplitOp
        ],
    ) -> bool:
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
        binary_math_ops = {"fmaxf", "fminf", "fmodf", "powf"}

        def is_binary_math_op(op: BinaryOp) -> bool:
            op_spec = binary_op_symbol(
                op.function, dtype=op.input_dtype, validate_attrs=False
            )
            return op_spec is not None and op_spec.operator in binary_math_ops

        if any(
            isinstance(op, UnaryOp)
            and unary_op_symbol(op.function, dtype=op.dtype) in math_ops
            for op in resolved_ops
        ):
            return True
        if any(
            isinstance(op, UnaryOp)
            and op.function in {ScalarFunction.CELU, ScalarFunction.SWISH}
            for op in resolved_ops
        ):
            return True
        if any(
            isinstance(op, UnaryOp)
            and op.function in {ScalarFunction.ISINF, ScalarFunction.ISNAN}
            for op in resolved_ops
        ):
            return True
        if any(
            isinstance(op, ClipOp)
            and op.dtype.is_float
            and (op.input_min is None or op.input_max is None)
            for op in resolved_ops
        ):
            return True
        if any(
            isinstance(op, BinaryOp) and is_binary_math_op(op)
            for op in resolved_ops
        ):
            return True
        if any(
            isinstance(
                op,
                (
                    AttentionOp,
                    BatchNormOp,
                    LpNormalizationOp,
                    InstanceNormalizationOp,
                    GroupNormalizationOp,
                    LayerNormalizationOp,
                    MeanVarianceNormalizationOp,
                    RMSNormalizationOp,
                    LrnOp,
                    LstmOp,
                    AdagradOp,
                    SoftmaxOp,
                    LogSoftmaxOp,
                    SoftmaxCrossEntropyLossOp,
                    ResizeOp,
                    GridSampleOp,
                ),
            )
            for op in resolved_ops
        ):
            return True
        if any(
            isinstance(op, ReduceOp)
            and op.reduce_kind in {"l1", "l2", "logsum", "logsumexp"}
            for op in resolved_ops
        ):
            return True
        if any(
            isinstance(op, ReduceOp)
            and op.reduce_kind in {"min", "max"}
            and op.dtype.is_float
            for op in resolved_ops
        ):
            return True
        if any(
            isinstance(op, MaxPoolOp) and op.dtype.is_float
            for op in resolved_ops
        ):
            return True
        if any(
            isinstance(op, (LpPoolOp, QuantizeLinearOp, QLinearMatMulOp))
            for op in resolved_ops
        ):
            return True
        return False

    @staticmethod
    def _needs_limits(
        resolved_ops: list[
            BinaryOp
            | UnaryOp
            | ClipOp
            | CastOp
            | QuantizeLinearOp
            | QLinearMatMulOp
            | MatMulOp
            | EinsumOp
            | GemmOp
            | AttentionOp
            | ConvOp
            | ConvTransposeOp
            | AveragePoolOp
            | LpPoolOp
            | BatchNormOp
            | LpNormalizationOp
            | InstanceNormalizationOp
            | GroupNormalizationOp
            | LayerNormalizationOp
            | MeanVarianceNormalizationOp
            | RMSNormalizationOp
            | LrnOp
            | LstmOp
            | SoftmaxOp
            | LogSoftmaxOp
            | HardmaxOp
            | NegativeLogLikelihoodLossOp
            | SoftmaxCrossEntropyLossOp
            | MaxPoolOp
            | ConcatOp
            | GatherElementsOp
            | GatherOp
            | GatherNDOp
            | TransposeOp
            | ReshapeOp
            | IdentityOp
            | EyeLikeOp
            | TriluOp
            | TileOp
            | DepthToSpaceOp
            | SpaceToDepthOp
            | SliceOp
            | ResizeOp
            | GridSampleOp
            | ReduceOp
            | ArgReduceOp
            | TopKOp
            | ConstantOfShapeOp
            | ShapeOp
            | SizeOp
            | NonZeroOp
            | NonMaxSuppressionOp
            | ExpandOp
            | CumSumOp
            | RangeOp
            | OneHotOp
            | SplitOp
        ],
    ) -> bool:
        if any(
            isinstance(op, ReduceOp)
            and op.reduce_kind in {"min", "max"}
            and op.dtype in {
                ScalarType.I64,
                ScalarType.I32,
                ScalarType.I16,
                ScalarType.I8,
            }
            for op in resolved_ops
        ):
            return True
        if any(
            isinstance(op, ClipOp)
            and op.dtype.is_integer
            and (op.input_min is None or op.input_max is None)
            for op in resolved_ops
        ):
            return True
        if any(
            isinstance(op, MaxPoolOp)
            and op.dtype
            in {ScalarType.I64, ScalarType.I32, ScalarType.I16, ScalarType.I8}
            for op in resolved_ops
        ):
            return True
        if any(
            isinstance(op, (QuantizeLinearOp, QLinearMatMulOp))
            and op.dtype.is_integer
            for op in resolved_ops
        ):
            return True
        if any(isinstance(op, NonMaxSuppressionOp) for op in resolved_ops):
            return True
        return False

    def _emit_model_wrapper(
        self,
        model: LoweredModel,
        resolved_ops: list[
            BinaryOp
            | WhereOp
            | UnaryOp
            | ClipOp
            | CastOp
            | QuantizeLinearOp
            | QLinearMatMulOp
            | MatMulOp
            | EinsumOp
            | GemmOp
            | AttentionOp
            | ConvOp
            | ConvTransposeOp
            | AveragePoolOp
            | LpPoolOp
            | BatchNormOp
            | LpNormalizationOp
            | InstanceNormalizationOp
            | GroupNormalizationOp
            | LayerNormalizationOp
            | MeanVarianceNormalizationOp
            | RMSNormalizationOp
            | LrnOp
            | LstmOp
            | SoftmaxOp
            | LogSoftmaxOp
            | HardmaxOp
            | NegativeLogLikelihoodLossOp
            | SoftmaxCrossEntropyLossOp
            | MaxPoolOp
            | ConcatOp
            | GatherElementsOp
            | GatherOp
            | GatherNDOp
            | TransposeOp
            | ReshapeOp
            | IdentityOp
            | EyeLikeOp
            | TriluOp
            | TileOp
            | DepthToSpaceOp
            | SpaceToDepthOp
            | SliceOp
            | ResizeOp
            | GridSampleOp
            | ReduceOp
            | ArgReduceOp
            | TopKOp
            | ConstantOfShapeOp
            | ShapeOp
            | SizeOp
            | NonZeroOp
            | NonMaxSuppressionOp
            | ExpandOp
            | CumSumOp
            | RangeOp
            | OneHotOp
            | SplitOp
        ],
        temp_buffers: tuple[TempBuffer, ...],
        *,
        dim_order: Sequence[str],
        input_dim_names: Mapping[int, Mapping[int, str]],
        output_dim_names: Mapping[int, Mapping[int, str]],
    ) -> str:
        params = []
        if dim_order:
            params.extend(self._format_dim_args(dim_order))
        for index, (name, shape, dtype) in enumerate(
            zip(model.input_names, model.input_shapes, model.input_dtypes)
        ):
            params.append(
                f"const {dtype.c_type} {name}"
                f"{self._param_array_suffix(shape, input_dim_names.get(index))}"
            )
        for index, (name, shape, dtype) in enumerate(
            zip(model.output_names, model.output_shapes, model.output_dtypes)
        ):
            params.append(
                f"{dtype.c_type} {name}"
                f"{self._param_array_suffix(shape, output_dim_names.get(index))}"
            )
        signature = ", ".join(params)
        lines = [f"void {model.name}({signature}) {{"]
        for temp in temp_buffers:
            c_type = temp.dtype.c_type
            storage = (
                "static "
                if self._temp_buffer_size_bytes(temp)
                > self._large_temp_threshold_bytes
                else ""
            )
            lines.append(
                f"    {storage}{c_type} {temp.name}{self._array_suffix(temp.shape)};"
            )
        for index, op in enumerate(resolved_ops):
            op_name = self._op_function_name(model, index)
            call = self._build_op_call(op, dim_order)
            lines.append(f"    {op_name}({call});")
        lines.append("}")
        return "\n".join(lines)

    @staticmethod
    def _temp_buffer_size_bytes(temp: TempBuffer) -> int:
        element_count = 1
        for dim in temp.shape:
            element_count *= dim
        return element_count * temp.dtype.np_dtype.itemsize

    @staticmethod
    def _build_op_call(
        op: BinaryOp
        | WhereOp
        | UnaryOp
        | ClipOp
        | CastOp
        | QuantizeLinearOp
        | QLinearMatMulOp
        | MatMulOp
        | EinsumOp
        | GemmOp
        | AttentionOp
        | RotaryEmbeddingOp
        | ConvOp
        | ConvTransposeOp
        | AveragePoolOp
        | LpPoolOp
        | BatchNormOp
        | LpNormalizationOp
        | InstanceNormalizationOp
        | GroupNormalizationOp
        | LayerNormalizationOp
        | MeanVarianceNormalizationOp
        | RMSNormalizationOp
        | LrnOp
        | LstmOp
        | AdagradOp
        | SoftmaxOp
        | LogSoftmaxOp
        | HardmaxOp
        | NegativeLogLikelihoodLossOp
        | SoftmaxCrossEntropyLossOp
        | MaxPoolOp
        | ConcatOp
        | GatherElementsOp
        | GatherOp
        | GatherNDOp
        | ScatterNDOp
        | TensorScatterOp
        | TransposeOp
        | ReshapeOp
        | IdentityOp
        | EyeLikeOp
        | TriluOp
        | TileOp
        | PadOp
        | DepthToSpaceOp
        | SpaceToDepthOp
        | SliceOp
        | ResizeOp
        | GridSampleOp
        | ReduceOp
        | ArgReduceOp
        | TopKOp
        | ConstantOfShapeOp
        | ShapeOp
        | SizeOp
        | NonZeroOp
        | NonMaxSuppressionOp
        | ExpandOp
        | CumSumOp
        | RangeOp
        | OneHotOp
        | SplitOp,
        dim_order: Sequence[str],
    ) -> str:
        args: list[str] = []
        if dim_order:
            args.extend(dim_order)
        if isinstance(op, BinaryOp):
            args.extend([op.input0, op.input1, op.output])
            return ", ".join(args)
        if isinstance(op, MultiInputBinaryOp):
            args.extend([*op.inputs, op.output])
            return ", ".join(args)
        if isinstance(op, WhereOp):
            args.extend([op.condition, op.input_x, op.input_y, op.output])
            return ", ".join(args)
        if isinstance(op, QLinearMatMulOp):
            args.extend(
                [
                    op.input0,
                    op.input0_scale,
                    op.input0_zero_point,
                    op.input1,
                    op.input1_scale,
                    op.input1_zero_point,
                    op.output_scale,
                    op.output_zero_point,
                    op.output,
                ]
            )
            return ", ".join(args)
        if isinstance(op, MatMulOp):
            args.extend([op.input0, op.input1, op.output])
            return ", ".join(args)
        if isinstance(op, EinsumOp):
            args.extend([*op.inputs, op.output])
            return ", ".join(args)
        if isinstance(op, GemmOp):
            if op.input_c is None:
                args.extend([op.input_a, op.input_b, op.output])
                return ", ".join(args)
            args.extend([op.input_a, op.input_b, op.input_c, op.output])
            return ", ".join(args)
        if isinstance(op, UnaryOp):
            args.extend([op.input0, op.output])
            return ", ".join(args)
        if isinstance(op, ClipOp):
            call_parts = [op.input0]
            if op.input_min is not None:
                call_parts.append(op.input_min)
            if op.input_max is not None:
                call_parts.append(op.input_max)
            call_parts.append(op.output)
            args.extend(call_parts)
            return ", ".join(args)
        if isinstance(op, QuantizeLinearOp):
            call_parts = [op.input0, op.scale]
            if op.zero_point is not None:
                call_parts.append(op.zero_point)
            call_parts.append(op.output)
            args.extend(call_parts)
            return ", ".join(args)
        if isinstance(op, AttentionOp):
            call_parts = [op.input_q, op.input_k, op.input_v]
            if op.input_attn_mask is not None:
                call_parts.append(op.input_attn_mask)
            if op.input_past_key is not None:
                call_parts.append(op.input_past_key)
            if op.input_past_value is not None:
                call_parts.append(op.input_past_value)
            if op.input_nonpad_kv_seqlen is not None:
                call_parts.append(op.input_nonpad_kv_seqlen)
            call_parts.append(op.output)
            if op.output_present_key is not None:
                call_parts.append(op.output_present_key)
            if op.output_present_value is not None:
                call_parts.append(op.output_present_value)
            if op.output_qk_matmul is not None:
                call_parts.append(op.output_qk_matmul)
            args.extend(call_parts)
            return ", ".join(args)
        if isinstance(op, ConvOp):
            if op.bias is None:
                args.extend([op.input0, op.weights, op.output])
                return ", ".join(args)
            args.extend([op.input0, op.weights, op.bias, op.output])
            return ", ".join(args)
        if isinstance(op, ConvTransposeOp):
            if op.bias is None:
                args.extend([op.input0, op.weights, op.output])
                return ", ".join(args)
            args.extend([op.input0, op.weights, op.bias, op.output])
            return ", ".join(args)
        if isinstance(op, AveragePoolOp):
            args.extend([op.input0, op.output])
            return ", ".join(args)
        if isinstance(op, LpPoolOp):
            args.extend([op.input0, op.output])
            return ", ".join(args)
        if isinstance(op, BatchNormOp):
            args.extend(
                [op.input0, op.scale, op.bias, op.mean, op.variance, op.output]
            )
            return ", ".join(args)
        if isinstance(op, LpNormalizationOp):
            args.extend([op.input0, op.output])
            return ", ".join(args)
        if isinstance(op, InstanceNormalizationOp):
            args.extend([op.input0, op.scale, op.bias, op.output])
            return ", ".join(args)
        if isinstance(op, GroupNormalizationOp):
            args.extend([op.input0, op.scale, op.bias, op.output])
            return ", ".join(args)
        if isinstance(op, LayerNormalizationOp):
            call_parts = [op.input0, op.scale]
            if op.bias is not None:
                call_parts.append(op.bias)
            call_parts.append(op.output)
            if op.mean_output is not None:
                call_parts.append(op.mean_output)
            if op.invstd_output is not None:
                call_parts.append(op.invstd_output)
            args.extend(call_parts)
            return ", ".join(args)
        if isinstance(op, MeanVarianceNormalizationOp):
            args.extend([op.input0, op.output])
            return ", ".join(args)
        if isinstance(op, RMSNormalizationOp):
            args.extend([op.input0, op.scale, op.output])
            return ", ".join(args)
        if isinstance(op, LstmOp):
            call_parts = [op.input_x, op.input_w, op.input_r]
            if op.input_b is not None:
                call_parts.append(op.input_b)
            if op.input_sequence_lens is not None:
                call_parts.append(op.input_sequence_lens)
            if op.input_initial_h is not None:
                call_parts.append(op.input_initial_h)
            if op.input_initial_c is not None:
                call_parts.append(op.input_initial_c)
            if op.input_p is not None:
                call_parts.append(op.input_p)
            if op.output_y is not None:
                call_parts.append(op.output_y)
            if op.output_y_h is not None:
                call_parts.append(op.output_y_h)
            if op.output_y_c is not None:
                call_parts.append(op.output_y_c)
            args.extend(call_parts)
            return ", ".join(args)
        if isinstance(op, AdagradOp):
            args.extend(
                [
                    op.rate,
                    op.timestep,
                    *op.inputs,
                    *op.gradients,
                    *op.accumulators,
                    *op.outputs,
                    *op.accumulator_outputs,
                ]
            )
            return ", ".join(args)
        if isinstance(op, (SoftmaxOp, LogSoftmaxOp, HardmaxOp)):
            args.extend([op.input0, op.output])
            return ", ".join(args)
        if isinstance(op, NegativeLogLikelihoodLossOp):
            call_parts = [op.input0, op.target]
            if op.weight is not None:
                call_parts.append(op.weight)
            call_parts.append(op.output)
            args.extend(call_parts)
            return ", ".join(args)
        if isinstance(op, SoftmaxCrossEntropyLossOp):
            call_parts = [op.input0, op.target]
            if op.weight is not None:
                call_parts.append(op.weight)
            call_parts.append(op.output)
            if op.log_prob is not None:
                call_parts.append(op.log_prob)
            args.extend(call_parts)
            return ", ".join(args)
        if isinstance(op, MaxPoolOp):
            if op.indices is not None:
                args.extend([op.input0, op.output, op.indices])
                return ", ".join(args)
            args.extend([op.input0, op.output])
            return ", ".join(args)
        if isinstance(op, GatherElementsOp):
            args.extend([op.data, op.indices, op.output])
            return ", ".join(args)
        if isinstance(op, GatherOp):
            args.extend([op.data, op.indices, op.output])
            return ", ".join(args)
        if isinstance(op, GatherNDOp):
            args.extend([op.data, op.indices, op.output])
            return ", ".join(args)
        if isinstance(op, ScatterNDOp):
            args.extend([op.data, op.indices, op.updates, op.output])
            return ", ".join(args)
        if isinstance(op, TensorScatterOp):
            args.extend([op.past_cache, op.update])
            if op.write_indices is not None:
                args.append(op.write_indices)
            args.append(op.output)
            return ", ".join(args)
        if isinstance(op, ConcatOp):
            args.extend([*op.inputs, op.output])
            return ", ".join(args)
        if isinstance(op, ConstantOfShapeOp):
            args.extend([op.input0, op.output])
            return ", ".join(args)
        if isinstance(op, ShapeOp):
            args.extend([op.input0, op.output])
            return ", ".join(args)
        if isinstance(op, SizeOp):
            args.extend([op.input0, op.output])
            return ", ".join(args)
        if isinstance(op, NonZeroOp):
            args.extend([op.input0, op.output])
            return ", ".join(args)
        if isinstance(op, NonMaxSuppressionOp):
            call_parts = [op.boxes, op.scores]
            if op.max_output_boxes_per_class is not None:
                call_parts.append(op.max_output_boxes_per_class)
            if op.iou_threshold is not None:
                call_parts.append(op.iou_threshold)
            if op.score_threshold is not None:
                call_parts.append(op.score_threshold)
            call_parts.append(op.output)
            args.extend(call_parts)
            return ", ".join(args)
        if isinstance(op, ExpandOp):
            args.extend([op.input0, op.output])
            return ", ".join(args)
        if isinstance(op, TriluOp):
            call_parts = [op.input0, op.output]
            if op.k_input is not None:
                call_parts.append(op.k_input)
            args.extend(call_parts)
            return ", ".join(args)
        if isinstance(op, CumSumOp):
            args.append(op.input0)
            if op.axis_input is not None:
                args.append(op.axis_input)
            args.append(op.output)
            return ", ".join(args)
        if isinstance(op, RangeOp):
            args.extend([op.start, op.limit, op.delta, op.output])
            return ", ".join(args)
        if isinstance(op, OneHotOp):
            args.extend([op.indices, op.depth, op.values, op.output])
            return ", ".join(args)
        if isinstance(op, SplitOp):
            args.extend([op.input0, *op.outputs])
            return ", ".join(args)
        if isinstance(op, ReshapeOp):
            args.extend([op.input0, op.output])
            return ", ".join(args)
        if isinstance(op, PadOp):
            call_parts = [op.input0]
            if op.pads_input is not None:
                call_parts.append(op.pads_input)
            if op.axes_input is not None:
                call_parts.append(op.axes_input)
            if op.value_input is not None:
                call_parts.append(op.value_input)
            call_parts.append(op.output)
            args.extend(call_parts)
            return ", ".join(args)
        if isinstance(op, SliceOp):
            call_parts = [op.input0]
            if op.starts_input is not None:
                call_parts.append(op.starts_input)
            if op.ends_input is not None:
                call_parts.append(op.ends_input)
            if op.axes_input is not None:
                call_parts.append(op.axes_input)
            if op.steps_input is not None:
                call_parts.append(op.steps_input)
            call_parts.append(op.output)
            args.extend(call_parts)
            return ", ".join(args)
        if isinstance(op, ResizeOp):
            call_parts = [op.input0]
            if op.roi_input is not None:
                call_parts.append(op.roi_input)
            if op.scales_input is not None:
                call_parts.append(op.scales_input)
            if op.sizes_input is not None:
                call_parts.append(op.sizes_input)
            call_parts.append(op.output)
            args.extend(call_parts)
            return ", ".join(args)
        if isinstance(op, TriluOp):
            call_parts = [op.input0, op.output]
            if op.k_input is not None:
                call_parts.append(op.k_input)
            args.extend(call_parts)
            return ", ".join(args)
        if isinstance(op, GridSampleOp):
            args.extend([op.input0, op.grid, op.output])
            return ", ".join(args)
        if isinstance(op, ReduceOp):
            if op.axes_input is not None:
                args.extend([op.input0, op.axes_input, op.output])
                return ", ".join(args)
            args.extend([op.input0, op.output])
            return ", ".join(args)
        if isinstance(op, ArgReduceOp):
            args.extend([op.input0, op.output])
            return ", ".join(args)
        if isinstance(op, TopKOp):
            args.extend([op.input0, op.output_values, op.output_indices])
            return ", ".join(args)
        args.extend([op.input0, op.output])
        return ", ".join(args)

    def _temp_buffers(
        self, model: LoweredModel, *, reserved_names: set[str] | None = None
    ) -> dict[str, TempBuffer]:
        output_names = set(model.output_names)
        used_names = set(reserved_names or ())

        def allocate_temp_name(base: str) -> str:
            if base not in used_names:
                used_names.add(base)
                return base
            index = 0
            while True:
                candidate = f"{base}{index}"
                if candidate not in used_names:
                    used_names.add(candidate)
                    return candidate
                index += 1

        intermediates = [
            (name, shape, dtype)
            for op in model.ops
            for name, shape, dtype in self._op_outputs(op)
            if name not in output_names
        ]
        if not intermediates:
            return {}
        if len(intermediates) == 1:
            name, shape, dtype = intermediates[0]
            temp_name = allocate_temp_name(f"tmp0_{name}")
            return {name: TempBuffer(name=temp_name, shape=shape, dtype=dtype)}
        return {
            name: TempBuffer(
                name=allocate_temp_name(f"tmp{index}_{name}"),
                shape=shape,
                dtype=dtype,
            )
            for index, (name, shape, dtype) in enumerate(intermediates)
        }

    @staticmethod
    def _resolve_op(
        op: BinaryOp
        | MultiInputBinaryOp
        | WhereOp
        | UnaryOp
        | ClipOp
        | CastOp
        | QuantizeLinearOp
        | QLinearMatMulOp
        | MatMulOp
        | EinsumOp
        | GemmOp
        | AttentionOp
        | RotaryEmbeddingOp
        | ConvOp
        | ConvTransposeOp
        | AveragePoolOp
        | LpPoolOp
        | BatchNormOp
        | LpNormalizationOp
        | InstanceNormalizationOp
        | GroupNormalizationOp
        | LayerNormalizationOp
        | MeanVarianceNormalizationOp
        | RMSNormalizationOp
        | LrnOp
        | LstmOp
        | AdagradOp
        | SoftmaxOp
        | LogSoftmaxOp
        | HardmaxOp
        | NegativeLogLikelihoodLossOp
        | SoftmaxCrossEntropyLossOp
        | MaxPoolOp
        | ConcatOp
        | GatherElementsOp
        | GatherOp
        | GatherNDOp
        | ScatterNDOp
        | TensorScatterOp
        | TransposeOp
        | ReshapeOp
        | IdentityOp
        | EyeLikeOp
        | TriluOp
        | TileOp
        | DepthToSpaceOp
        | SpaceToDepthOp
        | SliceOp
        | ResizeOp
        | GridSampleOp
        | ReduceOp
        | ArgReduceOp
        | TopKOp
        | ConstantOfShapeOp
        | ShapeOp
        | SizeOp
        | NonZeroOp
        | NonMaxSuppressionOp
        | ExpandOp
        | CumSumOp
        | RangeOp
        | OneHotOp
        | SplitOp,
        temp_map: dict[str, str],
    ) -> (
        BinaryOp
        | MultiInputBinaryOp
        | WhereOp
        | UnaryOp
        | ClipOp
        | CastOp
        | QuantizeLinearOp
        | QLinearMatMulOp
        | MatMulOp
        | EinsumOp
        | GemmOp
        | AttentionOp
        | ConvOp
        | ConvTransposeOp
        | AveragePoolOp
        | LpPoolOp
        | BatchNormOp
        | LpNormalizationOp
        | InstanceNormalizationOp
        | GroupNormalizationOp
        | LayerNormalizationOp
        | MeanVarianceNormalizationOp
        | RMSNormalizationOp
        | LrnOp
        | LstmOp
        | AdagradOp
        | SoftmaxOp
        | LogSoftmaxOp
        | HardmaxOp
        | NegativeLogLikelihoodLossOp
        | SoftmaxCrossEntropyLossOp
        | MaxPoolOp
        | ConcatOp
        | GatherElementsOp
        | GatherOp
        | GatherNDOp
        | ScatterNDOp
        | TensorScatterOp
        | TransposeOp
        | ReshapeOp
        | IdentityOp
        | EyeLikeOp
        | TriluOp
        | TileOp
        | DepthToSpaceOp
        | SpaceToDepthOp
        | SliceOp
        | ResizeOp
        | GridSampleOp
        | ReduceOp
        | ArgReduceOp
        | TopKOp
        | ConstantOfShapeOp
        | ShapeOp
        | SizeOp
        | NonZeroOp
        | NonMaxSuppressionOp
        | ExpandOp
        | CumSumOp
        | RangeOp
        | OneHotOp
        | SplitOp
    ):
        if isinstance(op, BinaryOp):
            return BinaryOp(
                input0=temp_map.get(op.input0, op.input0),
                input1=temp_map.get(op.input1, op.input1),
                output=temp_map.get(op.output, op.output),
                function=op.function,
                operator_kind=op.operator_kind,
                input0_shape=op.input0_shape,
                input1_shape=op.input1_shape,
                shape=op.shape,
                dtype=op.dtype,
                input_dtype=op.input_dtype,
            )
        if isinstance(op, MultiInputBinaryOp):
            return MultiInputBinaryOp(
                inputs=tuple(temp_map.get(name, name) for name in op.inputs),
                output=temp_map.get(op.output, op.output),
                function=op.function,
                operator_kind=op.operator_kind,
                shape=op.shape,
                dtype=op.dtype,
                input_dtype=op.input_dtype,
            )
        if isinstance(op, WhereOp):
            return WhereOp(
                condition=temp_map.get(op.condition, op.condition),
                input_x=temp_map.get(op.input_x, op.input_x),
                input_y=temp_map.get(op.input_y, op.input_y),
                output=temp_map.get(op.output, op.output),
                condition_shape=op.condition_shape,
                x_shape=op.x_shape,
                y_shape=op.y_shape,
                output_shape=op.output_shape,
                dtype=op.dtype,
            )
        if isinstance(op, UnaryOp):
            return UnaryOp(
                input0=temp_map.get(op.input0, op.input0),
                output=temp_map.get(op.output, op.output),
                function=op.function,
                shape=op.shape,
                dtype=op.dtype,
                input_dtype=op.input_dtype,
                params=op.params,
            )
        if isinstance(op, ClipOp):
            return ClipOp(
                input0=temp_map.get(op.input0, op.input0),
                input_min=temp_map.get(op.input_min, op.input_min)
                if op.input_min is not None
                else None,
                input_max=temp_map.get(op.input_max, op.input_max)
                if op.input_max is not None
                else None,
                output=temp_map.get(op.output, op.output),
                input_shape=op.input_shape,
                min_shape=op.min_shape,
                max_shape=op.max_shape,
                output_shape=op.output_shape,
                dtype=op.dtype,
            )
        if isinstance(op, MatMulOp):
            return MatMulOp(
                input0=temp_map.get(op.input0, op.input0),
                input1=temp_map.get(op.input1, op.input1),
                output=temp_map.get(op.output, op.output),
                input0_shape=op.input0_shape,
                input1_shape=op.input1_shape,
                output_shape=op.output_shape,
                batch_shape=op.batch_shape,
                input0_batch_shape=op.input0_batch_shape,
                input1_batch_shape=op.input1_batch_shape,
                m=op.m,
                n=op.n,
                k=op.k,
                left_vector=op.left_vector,
                right_vector=op.right_vector,
                dtype=op.dtype,
            )
        if isinstance(op, EinsumOp):
            return EinsumOp(
                inputs=tuple(temp_map.get(name, name) for name in op.inputs),
                output=temp_map.get(op.output, op.output),
                kind=op.kind,
                input_shapes=op.input_shapes,
                output_shape=op.output_shape,
                dtype=op.dtype,
                input_dtype=op.input_dtype,
            )
        if isinstance(op, CastOp):
            return CastOp(
                input0=temp_map.get(op.input0, op.input0),
                output=temp_map.get(op.output, op.output),
                shape=op.shape,
                input_dtype=op.input_dtype,
                dtype=op.dtype,
            )
        if isinstance(op, QuantizeLinearOp):
            return QuantizeLinearOp(
                input0=temp_map.get(op.input0, op.input0),
                scale=temp_map.get(op.scale, op.scale),
                zero_point=(
                    temp_map.get(op.zero_point, op.zero_point)
                    if op.zero_point is not None
                    else None
                ),
                output=temp_map.get(op.output, op.output),
                input_shape=op.input_shape,
                axis=op.axis,
                dtype=op.dtype,
                input_dtype=op.input_dtype,
                scale_dtype=op.scale_dtype,
            )
        if isinstance(op, QLinearMatMulOp):
            return QLinearMatMulOp(
                input0=temp_map.get(op.input0, op.input0),
                input0_scale=temp_map.get(op.input0_scale, op.input0_scale),
                input0_zero_point=temp_map.get(
                    op.input0_zero_point, op.input0_zero_point
                ),
                input1=temp_map.get(op.input1, op.input1),
                input1_scale=temp_map.get(op.input1_scale, op.input1_scale),
                input1_zero_point=temp_map.get(
                    op.input1_zero_point, op.input1_zero_point
                ),
                output_scale=temp_map.get(op.output_scale, op.output_scale),
                output_zero_point=temp_map.get(
                    op.output_zero_point, op.output_zero_point
                ),
                output=temp_map.get(op.output, op.output),
                input0_shape=op.input0_shape,
                input1_shape=op.input1_shape,
                output_shape=op.output_shape,
                batch_shape=op.batch_shape,
                input0_batch_shape=op.input0_batch_shape,
                input1_batch_shape=op.input1_batch_shape,
                m=op.m,
                n=op.n,
                k=op.k,
                left_vector=op.left_vector,
                right_vector=op.right_vector,
                input0_dtype=op.input0_dtype,
                input1_dtype=op.input1_dtype,
                dtype=op.dtype,
                input0_scale_dtype=op.input0_scale_dtype,
                input1_scale_dtype=op.input1_scale_dtype,
                output_scale_dtype=op.output_scale_dtype,
                input0_scale_shape=op.input0_scale_shape,
                input1_scale_shape=op.input1_scale_shape,
                output_scale_shape=op.output_scale_shape,
                input0_zero_shape=op.input0_zero_shape,
                input1_zero_shape=op.input1_zero_shape,
                output_zero_shape=op.output_zero_shape,
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
                input_attn_mask=(
                    temp_map.get(op.input_attn_mask, op.input_attn_mask)
                    if op.input_attn_mask is not None
                    else None
                ),
                input_past_key=(
                    temp_map.get(op.input_past_key, op.input_past_key)
                    if op.input_past_key is not None
                    else None
                ),
                input_past_value=(
                    temp_map.get(op.input_past_value, op.input_past_value)
                    if op.input_past_value is not None
                    else None
                ),
                input_nonpad_kv_seqlen=(
                    temp_map.get(
                        op.input_nonpad_kv_seqlen, op.input_nonpad_kv_seqlen
                    )
                    if op.input_nonpad_kv_seqlen is not None
                    else None
                ),
                output=temp_map.get(op.output, op.output),
                output_present_key=(
                    temp_map.get(op.output_present_key, op.output_present_key)
                    if op.output_present_key is not None
                    else None
                ),
                output_present_value=(
                    temp_map.get(op.output_present_value, op.output_present_value)
                    if op.output_present_value is not None
                    else None
                ),
                output_qk_matmul=(
                    temp_map.get(op.output_qk_matmul, op.output_qk_matmul)
                    if op.output_qk_matmul is not None
                    else None
                ),
                batch=op.batch,
                q_heads=op.q_heads,
                kv_heads=op.kv_heads,
                q_seq=op.q_seq,
                kv_seq=op.kv_seq,
                total_seq=op.total_seq,
                past_seq=op.past_seq,
                qk_head_size=op.qk_head_size,
                v_head_size=op.v_head_size,
                q_hidden_size=op.q_hidden_size,
                k_hidden_size=op.k_hidden_size,
                v_hidden_size=op.v_hidden_size,
                scale=op.scale,
                is_causal=op.is_causal,
                softcap=op.softcap,
                qk_matmul_output_mode=op.qk_matmul_output_mode,
                q_rank=op.q_rank,
                k_rank=op.k_rank,
                v_rank=op.v_rank,
                output_rank=op.output_rank,
                mask_shape=op.mask_shape,
                mask_is_bool=op.mask_is_bool,
                mask_rank=op.mask_rank,
                mask_broadcast_batch=op.mask_broadcast_batch,
                mask_broadcast_heads=op.mask_broadcast_heads,
                mask_broadcast_q_seq=op.mask_broadcast_q_seq,
                mask_q_seq=op.mask_q_seq,
                mask_kv_seq=op.mask_kv_seq,
                head_group_size=op.head_group_size,
                dtype=op.dtype,
            )
        if isinstance(op, RotaryEmbeddingOp):
            return RotaryEmbeddingOp(
                input0=temp_map.get(op.input0, op.input0),
                cos_cache=temp_map.get(op.cos_cache, op.cos_cache),
                sin_cache=temp_map.get(op.sin_cache, op.sin_cache),
                position_ids=(
                    temp_map.get(op.position_ids, op.position_ids)
                    if op.position_ids is not None
                    else None
                ),
                output=temp_map.get(op.output, op.output),
                input_shape=op.input_shape,
                cos_shape=op.cos_shape,
                sin_shape=op.sin_shape,
                position_ids_shape=op.position_ids_shape,
                dtype=op.dtype,
                position_ids_dtype=op.position_ids_dtype,
                rotary_dim=op.rotary_dim,
                rotary_dim_half=op.rotary_dim_half,
                head_size=op.head_size,
                num_heads=op.num_heads,
                seq_len=op.seq_len,
                batch=op.batch,
                input_rank=op.input_rank,
                interleaved=op.interleaved,
            )
        if isinstance(op, LstmOp):
            return LstmOp(
                input_x=temp_map.get(op.input_x, op.input_x),
                input_w=temp_map.get(op.input_w, op.input_w),
                input_r=temp_map.get(op.input_r, op.input_r),
                input_b=(
                    temp_map.get(op.input_b, op.input_b)
                    if op.input_b is not None
                    else None
                ),
                input_sequence_lens=(
                    temp_map.get(op.input_sequence_lens, op.input_sequence_lens)
                    if op.input_sequence_lens is not None
                    else None
                ),
                input_initial_h=(
                    temp_map.get(op.input_initial_h, op.input_initial_h)
                    if op.input_initial_h is not None
                    else None
                ),
                input_initial_c=(
                    temp_map.get(op.input_initial_c, op.input_initial_c)
                    if op.input_initial_c is not None
                    else None
                ),
                input_p=(
                    temp_map.get(op.input_p, op.input_p)
                    if op.input_p is not None
                    else None
                ),
                output_y=(
                    temp_map.get(op.output_y, op.output_y)
                    if op.output_y is not None
                    else None
                ),
                output_y_h=(
                    temp_map.get(op.output_y_h, op.output_y_h)
                    if op.output_y_h is not None
                    else None
                ),
                output_y_c=(
                    temp_map.get(op.output_y_c, op.output_y_c)
                    if op.output_y_c is not None
                    else None
                ),
                seq_length=op.seq_length,
                batch_size=op.batch_size,
                input_size=op.input_size,
                hidden_size=op.hidden_size,
                num_directions=op.num_directions,
                direction=op.direction,
                layout=op.layout,
                input_forget=op.input_forget,
                clip=op.clip,
                activation_kinds=op.activation_kinds,
                activation_alphas=op.activation_alphas,
                activation_betas=op.activation_betas,
                dtype=op.dtype,
                sequence_lens_dtype=op.sequence_lens_dtype,
            )
        if isinstance(op, AdagradOp):
            return AdagradOp(
                rate=temp_map.get(op.rate, op.rate),
                timestep=temp_map.get(op.timestep, op.timestep),
                inputs=tuple(temp_map.get(name, name) for name in op.inputs),
                gradients=tuple(
                    temp_map.get(name, name) for name in op.gradients
                ),
                accumulators=tuple(
                    temp_map.get(name, name) for name in op.accumulators
                ),
                outputs=tuple(temp_map.get(name, name) for name in op.outputs),
                accumulator_outputs=tuple(
                    temp_map.get(name, name)
                    for name in op.accumulator_outputs
                ),
                rate_shape=op.rate_shape,
                timestep_shape=op.timestep_shape,
                tensor_shapes=op.tensor_shapes,
                output_shapes=op.output_shapes,
                dtype=op.dtype,
                rate_dtype=op.rate_dtype,
                timestep_dtype=op.timestep_dtype,
                norm_coefficient=op.norm_coefficient,
                epsilon=op.epsilon,
                decay_factor=op.decay_factor,
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
        if isinstance(op, ConvTransposeOp):
            return ConvTransposeOp(
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
                output_padding=op.output_padding,
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
        if isinstance(op, LpPoolOp):
            return LpPoolOp(
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
                p=op.p,
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
        if isinstance(op, LpNormalizationOp):
            return LpNormalizationOp(
                input0=temp_map.get(op.input0, op.input0),
                output=temp_map.get(op.output, op.output),
                shape=op.shape,
                axis=op.axis,
                p=op.p,
                outer=op.outer,
                axis_size=op.axis_size,
                inner=op.inner,
                dtype=op.dtype,
            )
        if isinstance(op, InstanceNormalizationOp):
            return InstanceNormalizationOp(
                input0=temp_map.get(op.input0, op.input0),
                scale=temp_map.get(op.scale, op.scale),
                bias=temp_map.get(op.bias, op.bias),
                output=temp_map.get(op.output, op.output),
                shape=op.shape,
                channels=op.channels,
                spatial_size=op.spatial_size,
                epsilon=op.epsilon,
                dtype=op.dtype,
            )
        if isinstance(op, GroupNormalizationOp):
            return GroupNormalizationOp(
                input0=temp_map.get(op.input0, op.input0),
                scale=temp_map.get(op.scale, op.scale),
                bias=temp_map.get(op.bias, op.bias),
                output=temp_map.get(op.output, op.output),
                shape=op.shape,
                channels=op.channels,
                num_groups=op.num_groups,
                group_size=op.group_size,
                spatial_size=op.spatial_size,
                epsilon=op.epsilon,
                dtype=op.dtype,
            )
        if isinstance(op, LayerNormalizationOp):
            return LayerNormalizationOp(
                input0=temp_map.get(op.input0, op.input0),
                scale=temp_map.get(op.scale, op.scale),
                bias=(temp_map.get(op.bias, op.bias) if op.bias else None),
                output=temp_map.get(op.output, op.output),
                mean_output=(
                    temp_map.get(op.mean_output, op.mean_output)
                    if op.mean_output
                    else None
                ),
                invstd_output=(
                    temp_map.get(op.invstd_output, op.invstd_output)
                    if op.invstd_output
                    else None
                ),
                shape=op.shape,
                normalized_shape=op.normalized_shape,
                scale_shape=op.scale_shape,
                bias_shape=op.bias_shape,
                outer=op.outer,
                inner=op.inner,
                axis=op.axis,
                epsilon=op.epsilon,
                dtype=op.dtype,
            )
        if isinstance(op, MeanVarianceNormalizationOp):
            return MeanVarianceNormalizationOp(
                input0=temp_map.get(op.input0, op.input0),
                output=temp_map.get(op.output, op.output),
                shape=op.shape,
                axes=op.axes,
                non_axes=op.non_axes,
                reduce_count=op.reduce_count,
                epsilon=op.epsilon,
                dtype=op.dtype,
            )
        if isinstance(op, RMSNormalizationOp):
            return RMSNormalizationOp(
                input0=temp_map.get(op.input0, op.input0),
                scale=temp_map.get(op.scale, op.scale),
                output=temp_map.get(op.output, op.output),
                shape=op.shape,
                normalized_shape=op.normalized_shape,
                scale_shape=op.scale_shape,
                outer=op.outer,
                inner=op.inner,
                axis=op.axis,
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
        if isinstance(op, LogSoftmaxOp):
            return LogSoftmaxOp(
                input0=temp_map.get(op.input0, op.input0),
                output=temp_map.get(op.output, op.output),
                outer=op.outer,
                axis_size=op.axis_size,
                inner=op.inner,
                axis=op.axis,
                shape=op.shape,
                dtype=op.dtype,
            )
        if isinstance(op, HardmaxOp):
            return HardmaxOp(
                input0=temp_map.get(op.input0, op.input0),
                output=temp_map.get(op.output, op.output),
                outer=op.outer,
                axis_size=op.axis_size,
                inner=op.inner,
                axis=op.axis,
                shape=op.shape,
                dtype=op.dtype,
            )
        if isinstance(op, NegativeLogLikelihoodLossOp):
            return NegativeLogLikelihoodLossOp(
                input0=temp_map.get(op.input0, op.input0),
                target=temp_map.get(op.target, op.target),
                weight=(
                    temp_map.get(op.weight, op.weight)
                    if op.weight is not None
                    else None
                ),
                output=temp_map.get(op.output, op.output),
                input_shape=op.input_shape,
                target_shape=op.target_shape,
                output_shape=op.output_shape,
                n=op.n,
                c=op.c,
                d=op.d,
                reduction=op.reduction,
                ignore_index=op.ignore_index,
                input_dtype=op.input_dtype,
                weight_dtype=op.weight_dtype,
                weight_shape=op.weight_shape,
                dtype=op.dtype,
                target_dtype=op.target_dtype,
            )
        if isinstance(op, SoftmaxCrossEntropyLossOp):
            return SoftmaxCrossEntropyLossOp(
                input0=temp_map.get(op.input0, op.input0),
                target=temp_map.get(op.target, op.target),
                weight=(
                    temp_map.get(op.weight, op.weight)
                    if op.weight is not None
                    else None
                ),
                output=temp_map.get(op.output, op.output),
                log_prob=(
                    temp_map.get(op.log_prob, op.log_prob)
                    if op.log_prob is not None
                    else None
                ),
                input_shape=op.input_shape,
                target_shape=op.target_shape,
                output_shape=op.output_shape,
                log_prob_shape=op.log_prob_shape,
                n=op.n,
                c=op.c,
                d=op.d,
                reduction=op.reduction,
                ignore_index=op.ignore_index,
                input_dtype=op.input_dtype,
                weight_dtype=op.weight_dtype,
                weight_shape=op.weight_shape,
                dtype=op.dtype,
                target_dtype=op.target_dtype,
            )
        if isinstance(op, MaxPoolOp):
            return MaxPoolOp(
                input0=temp_map.get(op.input0, op.input0),
                output=temp_map.get(op.output, op.output),
                indices=(
                    temp_map.get(op.indices, op.indices)
                    if op.indices is not None
                    else None
                ),
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
                storage_order=op.storage_order,
                dtype=op.dtype,
                indices_dtype=op.indices_dtype,
            )
        if isinstance(op, GatherElementsOp):
            return GatherElementsOp(
                data=temp_map.get(op.data, op.data),
                indices=temp_map.get(op.indices, op.indices),
                output=temp_map.get(op.output, op.output),
                axis=op.axis,
                data_shape=op.data_shape,
                indices_shape=op.indices_shape,
                output_shape=op.output_shape,
                dtype=op.dtype,
                indices_dtype=op.indices_dtype,
            )
        if isinstance(op, GatherOp):
            return GatherOp(
                data=temp_map.get(op.data, op.data),
                indices=temp_map.get(op.indices, op.indices),
                output=temp_map.get(op.output, op.output),
                axis=op.axis,
                data_shape=op.data_shape,
                indices_shape=op.indices_shape,
                output_shape=op.output_shape,
                dtype=op.dtype,
                indices_dtype=op.indices_dtype,
            )
        if isinstance(op, GatherNDOp):
            return GatherNDOp(
                data=temp_map.get(op.data, op.data),
                indices=temp_map.get(op.indices, op.indices),
                output=temp_map.get(op.output, op.output),
                batch_dims=op.batch_dims,
                data_shape=op.data_shape,
                indices_shape=op.indices_shape,
                output_shape=op.output_shape,
                dtype=op.dtype,
                indices_dtype=op.indices_dtype,
            )
        if isinstance(op, ScatterNDOp):
            return ScatterNDOp(
                data=temp_map.get(op.data, op.data),
                indices=temp_map.get(op.indices, op.indices),
                updates=temp_map.get(op.updates, op.updates),
                output=temp_map.get(op.output, op.output),
                data_shape=op.data_shape,
                indices_shape=op.indices_shape,
                updates_shape=op.updates_shape,
                output_shape=op.output_shape,
                reduction=op.reduction,
                dtype=op.dtype,
                indices_dtype=op.indices_dtype,
            )
        if isinstance(op, TensorScatterOp):
            return TensorScatterOp(
                past_cache=temp_map.get(op.past_cache, op.past_cache),
                update=temp_map.get(op.update, op.update),
                write_indices=(
                    temp_map.get(op.write_indices, op.write_indices)
                    if op.write_indices is not None
                    else None
                ),
                output=temp_map.get(op.output, op.output),
                past_cache_shape=op.past_cache_shape,
                update_shape=op.update_shape,
                output_shape=op.output_shape,
                write_indices_shape=op.write_indices_shape,
                axis=op.axis,
                mode=op.mode,
                dtype=op.dtype,
                write_indices_dtype=op.write_indices_dtype,
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
        if isinstance(op, ShapeOp):
            return ShapeOp(
                input0=temp_map.get(op.input0, op.input0),
                output=temp_map.get(op.output, op.output),
                input_shape=op.input_shape,
                output_shape=op.output_shape,
                values=op.values,
                dtype=op.dtype,
                input_dtype=op.input_dtype,
            )
        if isinstance(op, SizeOp):
            return SizeOp(
                input0=temp_map.get(op.input0, op.input0),
                output=temp_map.get(op.output, op.output),
                input_shape=op.input_shape,
                output_shape=op.output_shape,
                value=op.value,
                dtype=op.dtype,
                input_dtype=op.input_dtype,
            )
        if isinstance(op, NonZeroOp):
            return NonZeroOp(
                input0=temp_map.get(op.input0, op.input0),
                output=temp_map.get(op.output, op.output),
                input_shape=op.input_shape,
                output_shape=op.output_shape,
                dtype=op.dtype,
                input_dtype=op.input_dtype,
            )
        if isinstance(op, NonMaxSuppressionOp):
            return NonMaxSuppressionOp(
                boxes=temp_map.get(op.boxes, op.boxes),
                scores=temp_map.get(op.scores, op.scores),
                max_output_boxes_per_class=CEmitter._map_optional_name(
                    temp_map, op.max_output_boxes_per_class
                ),
                iou_threshold=CEmitter._map_optional_name(
                    temp_map, op.iou_threshold
                ),
                score_threshold=CEmitter._map_optional_name(
                    temp_map, op.score_threshold
                ),
                output=temp_map.get(op.output, op.output),
                boxes_shape=op.boxes_shape,
                scores_shape=op.scores_shape,
                output_shape=op.output_shape,
                center_point_box=op.center_point_box,
                boxes_dtype=op.boxes_dtype,
                output_dtype=op.output_dtype,
                max_output_dtype=op.max_output_dtype,
                max_output_shape=op.max_output_shape,
                iou_threshold_dtype=op.iou_threshold_dtype,
                iou_threshold_shape=op.iou_threshold_shape,
                score_threshold_dtype=op.score_threshold_dtype,
                score_threshold_shape=op.score_threshold_shape,
            )
        if isinstance(op, ExpandOp):
            return ExpandOp(
                input0=temp_map.get(op.input0, op.input0),
                output=temp_map.get(op.output, op.output),
                input_shape=op.input_shape,
                output_shape=op.output_shape,
                input_shape_padded=op.input_shape_padded,
                input_strides=op.input_strides,
                dtype=op.dtype,
                input_dtype=op.input_dtype,
            )
        if isinstance(op, CumSumOp):
            return CumSumOp(
                input0=temp_map.get(op.input0, op.input0),
                axis_input=CEmitter._map_optional_name(temp_map, op.axis_input),
                axis_input_dtype=op.axis_input_dtype,
                axis=op.axis,
                output=temp_map.get(op.output, op.output),
                input_shape=op.input_shape,
                dtype=op.dtype,
                input_dtype=op.input_dtype,
                exclusive=op.exclusive,
                reverse=op.reverse,
            )
        if isinstance(op, RangeOp):
            return RangeOp(
                start=temp_map.get(op.start, op.start),
                limit=temp_map.get(op.limit, op.limit),
                delta=temp_map.get(op.delta, op.delta),
                output=temp_map.get(op.output, op.output),
                output_shape=op.output_shape,
                length=op.length,
                dtype=op.dtype,
                input_dtype=op.input_dtype,
            )
        if isinstance(op, OneHotOp):
            return OneHotOp(
                indices=temp_map.get(op.indices, op.indices),
                depth=temp_map.get(op.depth, op.depth),
                values=temp_map.get(op.values, op.values),
                output=temp_map.get(op.output, op.output),
                axis=op.axis,
                indices_shape=op.indices_shape,
                values_shape=op.values_shape,
                output_shape=op.output_shape,
                depth_dim=op.depth_dim,
                dtype=op.dtype,
                indices_dtype=op.indices_dtype,
                depth_dtype=op.depth_dtype,
            )
        if isinstance(op, SplitOp):
            return SplitOp(
                input0=temp_map.get(op.input0, op.input0),
                outputs=tuple(
                    temp_map.get(name, name) for name in op.outputs
                ),
                input_shape=op.input_shape,
                output_shapes=op.output_shapes,
                axis=op.axis,
                split_sizes=op.split_sizes,
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
                input_dtype=op.input_dtype,
            )
        if isinstance(op, ReshapeOp):
            return ReshapeOp(
                input0=temp_map.get(op.input0, op.input0),
                output=temp_map.get(op.output, op.output),
                input_shape=op.input_shape,
                output_shape=op.output_shape,
                dtype=op.dtype,
                input_dtype=op.input_dtype,
            )
        if isinstance(op, IdentityOp):
            return IdentityOp(
                input0=temp_map.get(op.input0, op.input0),
                output=temp_map.get(op.output, op.output),
                shape=op.shape,
                dtype=op.dtype,
                input_dtype=op.input_dtype,
            )
        if isinstance(op, EyeLikeOp):
            return EyeLikeOp(
                input0=temp_map.get(op.input0, op.input0),
                output=temp_map.get(op.output, op.output),
                output_shape=op.output_shape,
                k=op.k,
                dtype=op.dtype,
                input_dtype=op.input_dtype,
            )
        if isinstance(op, TriluOp):
            return TriluOp(
                input0=temp_map.get(op.input0, op.input0),
                output=temp_map.get(op.output, op.output),
                input_shape=op.input_shape,
                output_shape=op.output_shape,
                upper=op.upper,
                k_value=op.k_value,
                k_input=(
                    temp_map.get(op.k_input, op.k_input)
                    if op.k_input is not None
                    else None
                ),
                k_input_shape=op.k_input_shape,
                k_input_dtype=op.k_input_dtype,
                dtype=op.dtype,
                input_dtype=op.input_dtype,
            )
        if isinstance(op, TileOp):
            return TileOp(
                input0=temp_map.get(op.input0, op.input0),
                output=temp_map.get(op.output, op.output),
                input_shape=op.input_shape,
                output_shape=op.output_shape,
                repeats=op.repeats,
                input_strides=op.input_strides,
                dtype=op.dtype,
                input_dtype=op.input_dtype,
            )
        if isinstance(op, PadOp):
            return PadOp(
                input0=temp_map.get(op.input0, op.input0),
                output=temp_map.get(op.output, op.output),
                input_shape=op.input_shape,
                output_shape=op.output_shape,
                pads_begin=op.pads_begin,
                pads_end=op.pads_end,
                pads_input=(
                    temp_map.get(op.pads_input, op.pads_input)
                    if op.pads_input is not None
                    else None
                ),
                pads_shape=op.pads_shape,
                pads_dtype=op.pads_dtype,
                pads_axis_map=op.pads_axis_map,
                pads_values=op.pads_values,
                axes_input=(
                    temp_map.get(op.axes_input, op.axes_input)
                    if op.axes_input is not None
                    else None
                ),
                axes_shape=op.axes_shape,
                axes_dtype=op.axes_dtype,
                mode=op.mode,
                value=op.value,
                value_input=(
                    temp_map.get(op.value_input, op.value_input)
                    if op.value_input is not None
                    else None
                ),
                value_shape=op.value_shape,
                dtype=op.dtype,
                input_dtype=op.input_dtype,
                input_strides=op.input_strides,
            )
        if isinstance(op, DepthToSpaceOp):
            return DepthToSpaceOp(
                input0=temp_map.get(op.input0, op.input0),
                output=temp_map.get(op.output, op.output),
                input_shape=op.input_shape,
                output_shape=op.output_shape,
                blocksize=op.blocksize,
                mode=op.mode,
                dtype=op.dtype,
                input_dtype=op.input_dtype,
            )
        if isinstance(op, SpaceToDepthOp):
            return SpaceToDepthOp(
                input0=temp_map.get(op.input0, op.input0),
                output=temp_map.get(op.output, op.output),
                input_shape=op.input_shape,
                output_shape=op.output_shape,
                blocksize=op.blocksize,
                dtype=op.dtype,
                input_dtype=op.input_dtype,
            )
        if isinstance(op, SliceOp):
            return SliceOp(
                input0=temp_map.get(op.input0, op.input0),
                output=temp_map.get(op.output, op.output),
                input_shape=op.input_shape,
                output_shape=op.output_shape,
                starts=op.starts,
                steps=op.steps,
                axes=op.axes,
                starts_input=temp_map.get(op.starts_input, op.starts_input)
                if op.starts_input
                else None,
                ends_input=temp_map.get(op.ends_input, op.ends_input)
                if op.ends_input
                else None,
                axes_input=temp_map.get(op.axes_input, op.axes_input)
                if op.axes_input
                else None,
                steps_input=temp_map.get(op.steps_input, op.steps_input)
                if op.steps_input
                else None,
                starts_shape=op.starts_shape,
                ends_shape=op.ends_shape,
                axes_shape=op.axes_shape,
                steps_shape=op.steps_shape,
                starts_dtype=op.starts_dtype,
                ends_dtype=op.ends_dtype,
                axes_dtype=op.axes_dtype,
                steps_dtype=op.steps_dtype,
                dtype=op.dtype,
                input_dtype=op.input_dtype,
            )
        if isinstance(op, ResizeOp):
            return ResizeOp(
                input0=temp_map.get(op.input0, op.input0),
                output=temp_map.get(op.output, op.output),
                input_shape=op.input_shape,
                output_shape=op.output_shape,
                scales=op.scales,
                scales_input=temp_map.get(op.scales_input, op.scales_input)
                if op.scales_input
                else None,
                sizes_input=temp_map.get(op.sizes_input, op.sizes_input)
                if op.sizes_input
                else None,
                roi_input=temp_map.get(op.roi_input, op.roi_input)
                if op.roi_input
                else None,
                axes=op.axes,
                scales_shape=op.scales_shape,
                sizes_shape=op.sizes_shape,
                roi_shape=op.roi_shape,
                scales_dtype=op.scales_dtype,
                sizes_dtype=op.sizes_dtype,
                roi_dtype=op.roi_dtype,
                scales_axes=op.scales_axes,
                sizes_axes=op.sizes_axes,
                roi_axes=op.roi_axes,
                mode=op.mode,
                coordinate_transformation_mode=op.coordinate_transformation_mode,
                nearest_mode=op.nearest_mode,
                cubic_coeff_a=op.cubic_coeff_a,
                exclude_outside=op.exclude_outside,
                extrapolation_value=op.extrapolation_value,
                antialias=op.antialias,
                keep_aspect_ratio_policy=op.keep_aspect_ratio_policy,
                dtype=op.dtype,
            )
        if isinstance(op, GridSampleOp):
            return GridSampleOp(
                input0=temp_map.get(op.input0, op.input0),
                grid=temp_map.get(op.grid, op.grid),
                output=temp_map.get(op.output, op.output),
                input_shape=op.input_shape,
                grid_shape=op.grid_shape,
                output_shape=op.output_shape,
                spatial_rank=op.spatial_rank,
                input_spatial=op.input_spatial,
                output_spatial=op.output_spatial,
                mode=op.mode,
                padding_mode=op.padding_mode,
                align_corners=op.align_corners,
                dtype=op.dtype,
                grid_dtype=op.grid_dtype,
            )
        if isinstance(op, ReduceOp):
            return ReduceOp(
                input0=temp_map.get(op.input0, op.input0),
                output=temp_map.get(op.output, op.output),
                input_shape=op.input_shape,
                output_shape=op.output_shape,
                axes=op.axes,
                axes_input=temp_map.get(op.axes_input, op.axes_input)
                if op.axes_input
                else None,
                axes_input_shape=op.axes_input_shape,
                axes_input_dtype=op.axes_input_dtype,
                keepdims=op.keepdims,
                noop_with_empty_axes=op.noop_with_empty_axes,
                reduce_kind=op.reduce_kind,
                reduce_count=op.reduce_count,
                dtype=op.dtype,
            )
        if isinstance(op, ArgReduceOp):
            return ArgReduceOp(
                input0=temp_map.get(op.input0, op.input0),
                output=temp_map.get(op.output, op.output),
                input_shape=op.input_shape,
                output_shape=op.output_shape,
                axis=op.axis,
                keepdims=op.keepdims,
                select_last_index=op.select_last_index,
                reduce_kind=op.reduce_kind,
                input_dtype=op.input_dtype,
                output_dtype=op.output_dtype,
            )
        if isinstance(op, TopKOp):
            return TopKOp(
                input0=temp_map.get(op.input0, op.input0),
                output_values=temp_map.get(op.output_values, op.output_values),
                output_indices=temp_map.get(op.output_indices, op.output_indices),
                input_shape=op.input_shape,
                output_shape=op.output_shape,
                axis=op.axis,
                k=op.k,
                largest=op.largest,
                sorted=op.sorted,
                input_dtype=op.input_dtype,
                output_values_dtype=op.output_values_dtype,
                output_indices_dtype=op.output_indices_dtype,
            )
        return UnaryOp(
            input0=temp_map.get(op.input0, op.input0),
            output=temp_map.get(op.output, op.output),
            function=op.function,
            shape=op.shape,
            dtype=op.dtype,
        )

    def render_op(self, op: OpBase, ctx: EmitContext) -> str:
        if self._emit_state is None:
            raise CodegenError("Emitter state not initialized")
        state = self._emit_state
        dtype = self._op_output_dtype(op)
        templates = state.templates
        return self._render_op(
            state.model,
            op,
            ctx.op_index,
            array_suffix="",
            loop_vars=(),
            c_type=dtype.c_type,
            zero_literal=dtype.zero_literal,
            min_literal=dtype.min_literal,
            max_literal=dtype.max_literal,
            binary_template=templates["binary"],
            multi_input_template=templates["multi_input"],
            where_template=templates["where"],
            unary_template=templates["unary"],
            clip_template=templates["clip"],
            cast_template=templates["cast"],
            quantize_linear_template=templates["quantize_linear"],
            qlinear_matmul_template=templates["qlinear_matmul"],
            matmul_template=templates["matmul"],
            einsum_template=templates["einsum"],
            gemm_template=templates["gemm"],
            attention_template=templates["attention"],
            rotary_embedding_template=templates["rotary_embedding"],
            conv_template=templates["conv"],
            conv_transpose_template=templates["conv_transpose"],
            avg_pool_template=templates["avg_pool"],
            lp_pool_template=templates["lp_pool"],
            batch_norm_template=templates["batch_norm"],
            lp_norm_template=templates["lp_norm"],
            instance_norm_template=templates["instance_norm"],
            group_norm_template=templates["group_norm"],
            layer_norm_template=templates["layer_norm"],
            mean_variance_norm_template=templates["mean_variance_norm"],
            rms_norm_template=templates["rms_norm"],
            lrn_template=templates["lrn"],
            lstm_template=templates["lstm"],
            adagrad_template=templates["adagrad"],
            softmax_template=templates["softmax"],
            logsoftmax_template=templates["logsoftmax"],
            hardmax_template=templates["hardmax"],
            nllloss_template=templates["nllloss"],
            softmax_cross_entropy_loss_template=templates[
                "softmax_cross_entropy_loss"
            ],
            maxpool_template=templates["maxpool"],
            concat_template=templates["concat"],
            gather_elements_template=templates["gather_elements"],
            gather_template=templates["gather"],
            gather_nd_template=templates["gather_nd"],
            scatter_nd_template=templates["scatter_nd"],
            transpose_template=templates["transpose"],
            reshape_template=templates["reshape"],
            identity_template=templates["identity"],
            eye_like_template=templates["eye_like"],
            trilu_template=templates["trilu"],
            tile_template=templates["tile"],
            pad_template=templates["pad"],
            depth_to_space_template=templates["depth_to_space"],
            space_to_depth_template=templates["space_to_depth"],
            slice_template=templates["slice"],
            slice_dynamic_template=templates["slice_dynamic"],
            resize_template=templates["resize"],
            grid_sample_template=templates["grid_sample"],
            reduce_template=templates["reduce"],
            reduce_dynamic_template=templates["reduce_dynamic"],
            arg_reduce_template=templates["arg_reduce"],
            topk_template=templates["topk"],
            constant_of_shape_template=templates["constant_of_shape"],
            shape_template=templates["shape"],
            size_template=templates["size"],
            nonzero_template=templates["nonzero"],
            nonmax_suppression_template=templates["nonmax_suppression"],
            expand_template=templates["expand"],
            cumsum_template=templates["cumsum"],
            range_template=templates["range"],
            one_hot_template=templates["one_hot"],
            split_template=templates["split"],
            scalar_registry=state.scalar_registry,
            dim_args=state.dim_args,
            tensor_dim_names=state.tensor_dim_names,
        )

    def _render_op(
        self,
        model: LoweredModel,
        op: OpBase,
        index: int,
        *,
        array_suffix: str,
        loop_vars: tuple[str, ...],
        c_type: str,
        zero_literal: str,
        min_literal: str,
        max_literal: str,
        binary_template,
        multi_input_template,
        where_template,
        unary_template,
        clip_template,
        cast_template,
        quantize_linear_template,
        qlinear_matmul_template,
        matmul_template,
        einsum_template,
        gemm_template,
        attention_template,
        rotary_embedding_template,
        conv_template,
        conv_transpose_template,
        avg_pool_template,
        lp_pool_template,
        batch_norm_template,
        lp_norm_template,
        instance_norm_template,
        group_norm_template,
        layer_norm_template,
        mean_variance_norm_template,
        rms_norm_template,
        lrn_template,
        lstm_template,
        adagrad_template,
        softmax_template,
        logsoftmax_template,
        hardmax_template,
        nllloss_template,
        softmax_cross_entropy_loss_template,
        maxpool_template,
        concat_template,
        gather_elements_template,
        gather_template,
        gather_nd_template,
        scatter_nd_template,
        transpose_template,
        reshape_template,
        identity_template,
        eye_like_template,
        trilu_template,
        tile_template,
        pad_template,
        depth_to_space_template,
        space_to_depth_template,
        slice_template,
        slice_dynamic_template,
        resize_template,
        grid_sample_template,
        reduce_template,
        reduce_dynamic_template,
        arg_reduce_template,
        topk_template,
        constant_of_shape_template,
        shape_template,
        size_template,
        nonzero_template,
        nonmax_suppression_template,
        expand_template,
        cumsum_template,
        range_template,
        one_hot_template,
        split_template,
        scalar_registry: ScalarFunctionRegistry | None = None,
        dim_args: str = "",
        tensor_dim_names: Mapping[str, Mapping[int, str]] | None = None,
    ) -> str:
        node_info = model.node_infos[index]
        node_comment = CEmitter._emit_node_comment(node_info, index)
        op_name = self._op_function_name(model, index)
        tensor_dim_names = tensor_dim_names or {}

        def _dim_names_for(name: str) -> Mapping[int, str]:
            return tensor_dim_names.get(name, {})

        def with_node_comment(rendered: str) -> str:
            return f"{node_comment}\n{_format_c_indentation(rendered)}"

        if isinstance(op, BinaryOp):
            params = self._shared_param_map(
                [
                    ("input0", op.input0),
                    ("input1", op.input1),
                    ("output", op.output),
                ]
            )
            scalar_operator = None
            if (
                scalar_registry is not None
                and op.function not in COMPARE_FUNCTIONS
            ):
                scalar_operator = self._scalar_function_name(
                    op.function, op.input_dtype, scalar_registry
                )
            op_spec = binary_op_symbol(
                op.function,
                dtype=op.input_dtype,
                validate_attrs=False,
            )
            if op_spec is None:
                raise CodegenError(
                    f"Unsupported binary operator for rendering: {op.function.value}"
                )
            output_dim_names = _dim_names_for(op.output)
            shape = CEmitter._shape_dim_exprs(op.shape, output_dim_names)
            loop_vars = CEmitter._loop_vars(op.shape)
            output_suffix = self._param_array_suffix(op.shape, output_dim_names)
            input0_suffix = self._param_array_suffix(
                op.input0_shape, _dim_names_for(op.input0)
            )
            input1_suffix = self._param_array_suffix(
                op.input1_shape, _dim_names_for(op.input1)
            )
            input_c_type = op.input_dtype.c_type
            output_c_type = op.dtype.c_type
            param_decls = self._build_param_decls(
                [
                    (params["input0"], input_c_type, input0_suffix, True),
                    (params["input1"], input_c_type, input1_suffix, True),
                    (params["output"], output_c_type, output_suffix, False),
                ]
            )
            common = {
                "model_name": model.name,
                "op_name": op_name,
                "element_count": CEmitter._element_count_expr(shape),
                "array_suffix": output_suffix,
                "shape": shape,
                "loop_vars": loop_vars,
                "input_c_type": input_c_type,
                "output_c_type": output_c_type,
                "zero_literal": zero_literal,
                "dim_args": dim_args,
                "params": param_decls,
            }
            left_expr = CEmitter._broadcast_index_expr(
                params["input0"],
                op.input0_shape,
                op.shape,
                loop_vars,
            )
            right_expr = CEmitter._broadcast_index_expr(
                params["input1"],
                op.input1_shape,
                op.shape,
                loop_vars,
            )
            operator_expr = None
            operator = op_spec.operator
            operator_kind = op.operator_kind
            if scalar_operator is not None:
                operator = scalar_operator
                operator_kind = OperatorKind.FUNC
            if operator_kind == OperatorKind.EXPR:
                operator_expr = op_spec.operator.format(
                    left=left_expr, right=right_expr
                )
            rendered = binary_template.render(
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
            return with_node_comment(rendered)
        if isinstance(op, MultiInputBinaryOp):
            params = self._shared_param_map(
                [
                    *( (f"input{idx}", name) for idx, name in enumerate(op.inputs) ),
                    ("output", op.output),
                ]
            )
            scalar_operator = None
            if (
                scalar_registry is not None
                and op.function not in COMPARE_FUNCTIONS
                and op.function != ScalarFunction.MEAN
            ):
                scalar_operator = self._scalar_function_name(
                    op.function, op.input_dtype, scalar_registry
                )
            op_spec = binary_op_symbol(
                op.function,
                dtype=op.input_dtype,
                validate_attrs=False,
            )
            if op_spec is None:
                raise CodegenError(
                    "Unsupported multi-input operator for rendering: "
                    f"{op.function.value}"
                )
            output_dim_names = _dim_names_for(op.output)
            shape = CEmitter._shape_dim_exprs(op.shape, output_dim_names)
            loop_vars = CEmitter._loop_vars(op.shape)
            array_suffix = self._param_array_suffix(op.shape, output_dim_names)
            input_c_type = op.input_dtype.c_type
            output_c_type = op.dtype.c_type
            input_names = [
                params[f"input{idx}"] for idx in range(len(op.inputs))
            ]
            param_decls = self._build_param_decls(
                [
                    *( (name, input_c_type, array_suffix, True) for name in input_names ),
                    (params["output"], output_c_type, array_suffix, False),
                ]
            )
            common = {
                "model_name": model.name,
                "op_name": op_name,
                "element_count": CEmitter._element_count_expr(shape),
                "array_suffix": array_suffix,
                "shape": shape,
                "loop_vars": loop_vars,
                "input_c_type": input_c_type,
                "output_c_type": output_c_type,
                "zero_literal": zero_literal,
                "dim_args": dim_args,
                "params": param_decls,
            }
            input_exprs = [
                f"{name}" + "".join(f"[{var}]" for var in loop_vars)
                for name in input_names
            ]
            output_expr = f"{params['output']}" + "".join(
                f"[{var}]" for var in loop_vars
            )
            operator = op_spec.operator
            operator_kind = op.operator_kind
            operator_expr = None
            mean_scale = None
            if op.function == ScalarFunction.MEAN:
                operator = "+"
                operator_kind = OperatorKind.INFIX
                mean_scale = len(op.inputs)
            if scalar_operator is not None:
                operator = scalar_operator
                operator_kind = OperatorKind.FUNC
            if operator_kind == OperatorKind.EXPR:
                raise CodegenError(
                    "Multi-input operators do not support expression operators."
                )
            rendered = multi_input_template.render(
                **common,
                inputs=input_names,
                input_exprs=input_exprs,
                output=params["output"],
                output_expr=output_expr,
                operator=operator,
                operator_kind=operator_kind.value,
                operator_expr=operator_expr,
                mean_scale=mean_scale,
                is_mean=op.function == ScalarFunction.MEAN,
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, WhereOp):
            params = self._shared_param_map(
                [
                    ("condition", op.condition),
                    ("input_x", op.input_x),
                    ("input_y", op.input_y),
                    ("output", op.output),
                ]
            )
            output_dim_names = _dim_names_for(op.output)
            output_shape = CEmitter._shape_dim_exprs(
                op.output_shape, output_dim_names
            )
            loop_vars = CEmitter._loop_vars(op.output_shape)
            output_array_suffix = self._param_array_suffix(
                op.output_shape, output_dim_names
            )
            condition_array_suffix = self._param_array_suffix(
                op.condition_shape, _dim_names_for(op.condition)
            )
            x_array_suffix = self._param_array_suffix(
                op.x_shape, _dim_names_for(op.input_x)
            )
            y_array_suffix = self._param_array_suffix(
                op.y_shape, _dim_names_for(op.input_y)
            )
            condition_expr = CEmitter._broadcast_index_expr(
                params["condition"],
                op.condition_shape,
                op.output_shape,
                loop_vars,
            )
            x_expr = CEmitter._broadcast_index_expr(
                params["input_x"], op.x_shape, op.output_shape, loop_vars
            )
            y_expr = CEmitter._broadcast_index_expr(
                params["input_y"], op.y_shape, op.output_shape, loop_vars
            )
            output_expr = f"{params['output']}" + "".join(
                f"[{var}]" for var in loop_vars
            )
            param_decls = self._build_param_decls(
                [
                    (
                        params["condition"],
                        ScalarType.BOOL.c_type,
                        condition_array_suffix,
                        True,
                    ),
                    (params["input_x"], op.dtype.c_type, x_array_suffix, True),
                    (params["input_y"], op.dtype.c_type, y_array_suffix, True),
                    (
                        params["output"],
                        op.dtype.c_type,
                        output_array_suffix,
                        False,
                    ),
                ]
            )
            rendered = where_template.render(
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
                input_c_type=op.dtype.c_type,
                output_c_type=op.dtype.c_type,
                condition_c_type=ScalarType.BOOL.c_type,
                dim_args=dim_args,
                params=param_decls,
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, MatMulOp):
            params = self._shared_param_map(
                [
                    ("input0", op.input0),
                    ("input1", op.input1),
                    ("output", op.output),
                ]
            )
            output_shape = CEmitter._codegen_shape(op.output_shape)
            output_loop_vars = CEmitter._loop_vars(output_shape)
            output_index_expr = f"{params['output']}" + "".join(
                f"[{var}]" for var in output_loop_vars
            )
            batch_rank = len(op.batch_shape)
            batch_vars = output_loop_vars[:batch_rank]
            if op.left_vector and op.right_vector:
                row_var = None
                col_var = None
            elif op.left_vector:
                row_var = None
                col_var = output_loop_vars[-1]
            elif op.right_vector:
                row_var = output_loop_vars[-1]
                col_var = None
            else:
                row_var = output_loop_vars[-2]
                col_var = output_loop_vars[-1]
            input0_index_expr, input1_index_expr = CEmitter._matmul_index_exprs(
                op,
                batch_vars,
                row_var,
                col_var,
                batch_rank,
                input0=params["input0"],
                input1=params["input1"],
            )
            input0_suffix = self._param_array_suffix(op.input0_shape)
            input1_suffix = self._param_array_suffix(op.input1_shape)
            output_suffix = self._param_array_suffix(op.output_shape)
            param_decls = self._build_param_decls(
                [
                    (params["input0"], c_type, input0_suffix, True),
                    (params["input1"], c_type, input1_suffix, True),
                    (params["output"], c_type, output_suffix, False),
                ]
            )
            rendered = matmul_template.render(
                model_name=model.name,
                op_name=op_name,
                input0=params["input0"],
                input1=params["input1"],
                output=params["output"],
                params=param_decls,
                c_type=c_type,
                acc_type=c_type,
                zero_literal=zero_literal,
                input0_suffix=input0_suffix,
                input1_suffix=input1_suffix,
                output_suffix=output_suffix,
                output_loop_vars=output_loop_vars,
                output_loop_bounds=output_shape,
                output_index_expr=output_index_expr,
                input0_index_expr=input0_index_expr,
                input1_index_expr=input1_index_expr,
                m=op.m,
                n=op.n,
                k=op.k,
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, EinsumOp):
            params = self._shared_param_map(
                [
                    *(
                        (f"input{idx}", name)
                        for idx, name in enumerate(op.inputs)
                    ),
                    ("output", op.output),
                ]
            )
            output_dim_names = _dim_names_for(op.output)
            output_shape = CEmitter._shape_dim_exprs(
                op.output_shape, output_dim_names
            )
            output_loop_vars = CEmitter._loop_vars(op.output_shape)
            if output_loop_vars:
                output_expr = f"{params['output']}" + "".join(
                    f"[{var}]" for var in output_loop_vars
                )
            else:
                output_expr = f"{params['output']}[0]"
            input_shapes = op.input_shapes
            input_dim_names = [
                _dim_names_for(name) for name in op.inputs
            ]
            input_suffixes = [
                self._param_array_suffix(shape, dim_names)
                for shape, dim_names in zip(input_shapes, input_dim_names)
            ]
            param_decls = self._build_param_decls(
                [
                    *(
                        (
                            params[f"input{idx}"],
                            op.input_dtype.c_type,
                            input_suffixes[idx],
                            True,
                        )
                        for idx in range(len(op.inputs))
                    ),
                    (
                        params["output"],
                        op.dtype.c_type,
                        self._param_array_suffix(op.output_shape, output_dim_names),
                        False,
                    ),
                ]
            )
            input_loop_vars: tuple[str, ...] = ()
            input_loop_bounds: tuple[str | int, ...] = ()
            reduce_loop_var = "k"
            reduce_loop_bound: str | int | None = None
            input_expr = None
            input0_expr = None
            input1_expr = None
            if op.kind == EinsumKind.REDUCE_ALL:
                input_loop_vars = CEmitter._loop_vars(input_shapes[0])
                input_loop_bounds = tuple(
                    CEmitter._shape_dim_exprs(
                        input_shapes[0], input_dim_names[0]
                    )
                )
                if input_loop_vars:
                    input_expr = f"{params['input0']}" + "".join(
                        f"[{var}]" for var in input_loop_vars
                    )
                else:
                    input_expr = f"{params['input0']}[0]"
            elif op.kind == EinsumKind.SUM_J:
                input_shape_exprs = CEmitter._shape_dim_exprs(
                    input_shapes[0], input_dim_names[0]
                )
                reduce_loop_bound = input_shape_exprs[1]
                input_expr = (
                    f"{params['input0']}"
                    f"[{output_loop_vars[0]}][{reduce_loop_var}]"
                )
            elif op.kind == EinsumKind.TRANSPOSE:
                input_expr = (
                    f"{params['input0']}"
                    f"[{output_loop_vars[1]}][{output_loop_vars[0]}]"
                )
            elif op.kind == EinsumKind.DOT:
                input_shape_exprs = CEmitter._shape_dim_exprs(
                    input_shapes[0], input_dim_names[0]
                )
                reduce_loop_bound = input_shape_exprs[0]
                input0_expr = f"{params['input0']}[{reduce_loop_var}]"
                input1_expr = f"{params['input1']}[{reduce_loop_var}]"
            elif op.kind == EinsumKind.BATCH_MATMUL:
                input_shape_exprs = CEmitter._shape_dim_exprs(
                    input_shapes[0], input_dim_names[0]
                )
                reduce_loop_bound = input_shape_exprs[2]
                input0_expr = (
                    f"{params['input0']}"
                    f"[{output_loop_vars[0]}]"
                    f"[{output_loop_vars[1]}][{reduce_loop_var}]"
                )
                input1_expr = (
                    f"{params['input1']}"
                    f"[{output_loop_vars[0]}]"
                    f"[{reduce_loop_var}][{output_loop_vars[2]}]"
                )
            elif op.kind == EinsumKind.BATCH_DIAGONAL:
                diag_var = output_loop_vars[-1]
                prefix_vars = output_loop_vars[:-1]
                input_expr = f"{params['input0']}" + "".join(
                    f"[{var}]" for var in prefix_vars
                )
                input_expr += f"[{diag_var}][{diag_var}]"
            rendered = einsum_template.render(
                model_name=model.name,
                op_name=op_name,
                params=param_decls,
                dim_args=dim_args,
                kind=op.kind.value,
                output_loop_vars=output_loop_vars,
                output_loop_bounds=output_shape,
                output_expr=output_expr,
                acc_type=op.dtype.c_type,
                zero_literal=zero_literal,
                input_loop_vars=input_loop_vars,
                input_loop_bounds=input_loop_bounds,
                reduce_loop_var=reduce_loop_var,
                reduce_loop_bound=reduce_loop_bound,
                input_expr=input_expr,
                input0_expr=input0_expr,
                input1_expr=input1_expr,
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, GemmOp):
            params = self._shared_param_map(
                [
                    ("input_a", op.input_a),
                    ("input_b", op.input_b),
                    ("input_c", op.input_c),
                    ("output", op.output),
                ]
            )
            input_a_shape = (op.k, op.m) if op.trans_a else (op.m, op.k)
            input_b_shape = (op.n, op.k) if op.trans_b else (op.k, op.n)
            input_a_suffix = self._param_array_suffix(input_a_shape)
            input_b_suffix = self._param_array_suffix(input_b_shape)
            output_suffix = self._param_array_suffix((op.m, op.n))
            c_suffix = (
                self._param_array_suffix(op.c_shape)
                if op.c_shape is not None
                else ""
            )
            param_decls = self._build_param_decls(
                [
                    (params["input_a"], c_type, input_a_suffix, True),
                    (params["input_b"], c_type, input_b_suffix, True),
                    (
                        params["input_c"],
                        c_type,
                        c_suffix,
                        True,
                    )
                    if params["input_c"]
                    else (None, "", "", True),
                    (params["output"], c_type, output_suffix, False),
                ]
            )
            alpha_literal = CEmitter._format_literal(op.dtype, op.alpha)
            beta_literal = CEmitter._format_literal(op.dtype, op.beta)
            if op.c_shape is None:
                c_rank = 0
                c_dim0 = 0
                c_dim1 = 0
            elif len(op.c_shape) == 0:
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
            rendered = gemm_template.render(
                model_name=model.name,
                op_name=op_name,
                input_a=params["input_a"],
                input_b=params["input_b"],
                input_c=params["input_c"],
                output=params["output"],
                params=param_decls,
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
                input_a_suffix=input_a_suffix,
                input_b_suffix=input_b_suffix,
                output_suffix=output_suffix,
                c_suffix=(
                    c_suffix if op.c_shape is not None else None
                ),
                c_rank=c_rank,
                c_dim0=c_dim0,
                c_dim1=c_dim1,
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, AttentionOp):
            if scalar_registry is None:
                raise CodegenError(
                    "Scalar function registry is required for Attention codegen."
                )
            max_fn = self._scalar_function_name(
                ScalarFunction.MAXIMUM, op.dtype, scalar_registry
            )
            if max_fn is None:
                raise CodegenError(
                    "Failed to resolve scalar maximum function for Attention."
                )
            params = self._shared_param_map(
                [
                    ("input_q", op.input_q),
                    ("input_k", op.input_k),
                    ("input_v", op.input_v),
                    ("input_attn_mask", op.input_attn_mask),
                    ("input_past_key", op.input_past_key),
                    ("input_past_value", op.input_past_value),
                    ("input_nonpad_kv_seqlen", op.input_nonpad_kv_seqlen),
                    ("output", op.output),
                    ("output_present_key", op.output_present_key),
                    ("output_present_value", op.output_present_value),
                    ("output_qk_matmul", op.output_qk_matmul),
                ]
            )
            if op.q_rank == 4:
                input_q_shape = (op.batch, op.q_heads, op.q_seq, op.qk_head_size)
            else:
                input_q_shape = (op.batch, op.q_seq, op.q_hidden_size)
            if op.k_rank == 4:
                input_k_shape = (op.batch, op.kv_heads, op.kv_seq, op.qk_head_size)
            else:
                input_k_shape = (op.batch, op.kv_seq, op.k_hidden_size)
            if op.v_rank == 4:
                input_v_shape = (op.batch, op.kv_heads, op.kv_seq, op.v_head_size)
            else:
                input_v_shape = (op.batch, op.kv_seq, op.v_hidden_size)
            if op.output_rank == 4:
                output_shape = (op.batch, op.q_heads, op.q_seq, op.v_head_size)
            else:
                output_shape = (
                    op.batch,
                    op.q_seq,
                    op.q_heads * op.v_head_size,
                )
            present_key_shape = (
                (op.batch, op.kv_heads, op.total_seq, op.qk_head_size)
                if op.output_present_key is not None
                else None
            )
            present_value_shape = (
                (op.batch, op.kv_heads, op.total_seq, op.v_head_size)
                if op.output_present_value is not None
                else None
            )
            qk_matmul_shape = (
                (op.batch, op.q_heads, op.q_seq, op.total_seq)
                if op.output_qk_matmul is not None
                else None
            )
            input_q_suffix = self._param_array_suffix(input_q_shape)
            input_k_suffix = self._param_array_suffix(input_k_shape)
            input_v_suffix = self._param_array_suffix(input_v_shape)
            input_mask_suffix = (
                self._param_array_suffix(op.mask_shape)
                if op.input_attn_mask is not None
                else ""
            )
            input_past_key_suffix = (
                self._param_array_suffix(
                    (op.batch, op.kv_heads, op.past_seq, op.qk_head_size)
                )
                if op.input_past_key is not None
                else ""
            )
            input_past_value_suffix = (
                self._param_array_suffix(
                    (op.batch, op.kv_heads, op.past_seq, op.v_head_size)
                )
                if op.input_past_value is not None
                else ""
            )
            input_nonpad_suffix = (
                self._param_array_suffix((op.batch,))
                if op.input_nonpad_kv_seqlen is not None
                else ""
            )
            output_suffix = self._param_array_suffix(output_shape)
            output_present_key_suffix = (
                self._param_array_suffix(present_key_shape)
                if present_key_shape is not None
                else ""
            )
            output_present_value_suffix = (
                self._param_array_suffix(present_value_shape)
                if present_value_shape is not None
                else ""
            )
            output_qk_matmul_suffix = (
                self._param_array_suffix(qk_matmul_shape)
                if qk_matmul_shape is not None
                else ""
            )
            mask_c_type = "bool" if op.mask_is_bool else c_type
            param_decls = self._build_param_decls(
                [
                    (params["input_q"], c_type, input_q_suffix, True),
                    (params["input_k"], c_type, input_k_suffix, True),
                    (params["input_v"], c_type, input_v_suffix, True),
                    (
                        params["input_attn_mask"],
                        mask_c_type,
                        input_mask_suffix,
                        True,
                    )
                    if params["input_attn_mask"]
                    else (None, "", "", True),
                    (
                        params["input_past_key"],
                        c_type,
                        input_past_key_suffix,
                        True,
                    )
                    if params["input_past_key"]
                    else (None, "", "", True),
                    (
                        params["input_past_value"],
                        c_type,
                        input_past_value_suffix,
                        True,
                    )
                    if params["input_past_value"]
                    else (None, "", "", True),
                    (
                        params["input_nonpad_kv_seqlen"],
                        ScalarType.I64.c_type,
                        input_nonpad_suffix,
                        True,
                    )
                    if params["input_nonpad_kv_seqlen"]
                    else (None, "", "", True),
                    (params["output"], c_type, output_suffix, False),
                    (
                        params["output_present_key"],
                        c_type,
                        output_present_key_suffix,
                        False,
                    )
                    if params["output_present_key"]
                    else (None, "", "", False),
                    (
                        params["output_present_value"],
                        c_type,
                        output_present_value_suffix,
                        False,
                    )
                    if params["output_present_value"]
                    else (None, "", "", False),
                    (
                        params["output_qk_matmul"],
                        c_type,
                        output_qk_matmul_suffix,
                        False,
                    )
                    if params["output_qk_matmul"]
                    else (None, "", "", False),
                ]
            )
            rendered = attention_template.render(
                model_name=model.name,
                op_name=op_name,
                input_q=params["input_q"],
                input_k=params["input_k"],
                input_v=params["input_v"],
                input_attn_mask=params["input_attn_mask"],
                input_past_key=params["input_past_key"],
                input_past_value=params["input_past_value"],
                input_nonpad_kv_seqlen=params["input_nonpad_kv_seqlen"],
                output=params["output"],
                output_present_key=params["output_present_key"],
                output_present_value=params["output_present_value"],
                output_qk_matmul=params["output_qk_matmul"],
                params=param_decls,
                c_type=c_type,
                nonpad_c_type=ScalarType.I64.c_type,
                zero_literal=zero_literal,
                min_literal=min_literal,
                scale_literal=CEmitter._format_floating(op.scale, op.dtype),
                softcap_literal=CEmitter._format_floating(op.softcap, op.dtype),
                one_literal=CEmitter._format_literal(op.dtype, 1),
                max_fn=max_fn,
                exp_fn=CEmitter._math_fn(op.dtype, "expf", "exp"),
                tanh_fn=CEmitter._math_fn(op.dtype, "tanhf", "tanh"),
                is_causal=int(op.is_causal),
                qk_matmul_output_mode=op.qk_matmul_output_mode,
                batch=op.batch,
                q_heads=op.q_heads,
                kv_heads=op.kv_heads,
                q_seq=op.q_seq,
                kv_seq=op.kv_seq,
                total_seq=op.total_seq,
                past_seq=op.past_seq,
                qk_head_size=op.qk_head_size,
                v_head_size=op.v_head_size,
                head_group_size=op.head_group_size,
                q_rank=op.q_rank,
                k_rank=op.k_rank,
                v_rank=op.v_rank,
                output_rank=op.output_rank,
                q_hidden_size=op.q_hidden_size,
                k_hidden_size=op.k_hidden_size,
                v_hidden_size=op.v_hidden_size,
                has_attn_mask=int(op.input_attn_mask is not None),
                mask_rank=op.mask_rank or 0,
                mask_is_bool=int(op.mask_is_bool),
                mask_broadcast_batch=int(op.mask_broadcast_batch),
                mask_broadcast_heads=int(op.mask_broadcast_heads),
                mask_broadcast_q_seq=int(op.mask_broadcast_q_seq),
                mask_q_seq=op.mask_q_seq or 0,
                mask_kv_seq=op.mask_kv_seq or 0,
                input_q_suffix=input_q_suffix,
                input_k_suffix=input_k_suffix,
                input_v_suffix=input_v_suffix,
                input_mask_suffix=input_mask_suffix,
                input_past_key_suffix=input_past_key_suffix,
                input_past_value_suffix=input_past_value_suffix,
                input_nonpad_suffix=input_nonpad_suffix,
                output_suffix=output_suffix,
                output_present_key_suffix=output_present_key_suffix,
                output_present_value_suffix=output_present_value_suffix,
                output_qk_matmul_suffix=output_qk_matmul_suffix,
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, RotaryEmbeddingOp):
            params = self._shared_param_map(
                [
                    ("input0", op.input0),
                    ("cos_cache", op.cos_cache),
                    ("sin_cache", op.sin_cache),
                    ("position_ids", op.position_ids),
                    ("output", op.output),
                ]
            )
            input_suffix = self._param_array_suffix(
                op.input_shape, _dim_names_for(op.input0)
            )
            cos_suffix = self._param_array_suffix(op.cos_shape)
            sin_suffix = self._param_array_suffix(op.sin_shape)
            position_suffix = (
                self._param_array_suffix(op.position_ids_shape)
                if op.position_ids_shape is not None
                else ""
            )
            output_suffix = self._param_array_suffix(
                op.input_shape, _dim_names_for(op.output)
            )
            param_decls = self._build_param_decls(
                [
                    (params["input0"], c_type, input_suffix, True),
                    (params["cos_cache"], c_type, cos_suffix, True),
                    (params["sin_cache"], c_type, sin_suffix, True),
                    (
                        params["position_ids"],
                        op.position_ids_dtype.c_type,
                        position_suffix,
                        True,
                    )
                    if params["position_ids"]
                    else (None, "", "", True),
                    (params["output"], c_type, output_suffix, False),
                ]
            )
            rendered = rotary_embedding_template.render(
                model_name=model.name,
                op_name=op_name,
                input0=params["input0"],
                cos_cache=params["cos_cache"],
                sin_cache=params["sin_cache"],
                position_ids=params["position_ids"],
                output=params["output"],
                params=param_decls,
                c_type=c_type,
                input_suffix=input_suffix,
                cos_suffix=cos_suffix,
                sin_suffix=sin_suffix,
                position_suffix=position_suffix,
                output_suffix=output_suffix,
                batch=op.batch,
                seq_len=op.seq_len,
                num_heads=op.num_heads,
                head_size=op.head_size,
                rotary_dim=op.rotary_dim,
                rotary_dim_half=op.rotary_dim_half,
                input_rank=op.input_rank,
                interleaved=int(op.interleaved),
                has_position_ids=int(op.position_ids is not None),
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, ConvOp):
            params = self._shared_param_map(
                [
                    ("input0", op.input0),
                    ("weights", op.weights),
                    ("bias", op.bias),
                    ("output", op.output),
                ]
            )
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
            input_suffix = self._param_array_suffix(input_shape)
            weight_suffix = self._param_array_suffix(weight_shape)
            bias_suffix = self._param_array_suffix((op.out_channels,))
            output_suffix = self._param_array_suffix(output_shape)
            param_decls = self._build_param_decls(
                [
                    (params["input0"], c_type, input_suffix, True),
                    (params["weights"], c_type, weight_suffix, True),
                    (
                        params["bias"],
                        c_type,
                        bias_suffix,
                        True,
                    )
                    if params["bias"]
                    else (None, "", "", True),
                    (params["output"], c_type, output_suffix, False),
                ]
            )
            rendered = conv_template.render(
                model_name=model.name,
                op_name=op_name,
                input0=params["input0"],
                weights=params["weights"],
                bias=params["bias"],
                output=params["output"],
                params=param_decls,
                c_type=c_type,
                zero_literal=zero_literal,
                input_suffix=input_suffix,
                weight_suffix=weight_suffix,
                bias_suffix=bias_suffix,
                output_suffix=output_suffix,
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
            return with_node_comment(rendered)
        if isinstance(op, ConvTransposeOp):
            params = self._shared_param_map(
                [
                    ("input0", op.input0),
                    ("weights", op.weights),
                    ("bias", op.bias),
                    ("output", op.output),
                ]
            )
            input_shape = (op.batch, op.in_channels, *op.in_spatial)
            weight_shape = (
                op.in_channels,
                op.out_channels // op.group,
                *op.kernel_shape,
            )
            output_shape = (op.batch, op.out_channels, *op.out_spatial)
            in_indices = tuple(f"id{dim}" for dim in range(op.spatial_rank))
            kernel_indices = tuple(
                f"kd{dim}" for dim in range(op.spatial_rank)
            )
            out_indices = tuple(f"od{dim}" for dim in range(op.spatial_rank))
            pad_begin = op.pads[: op.spatial_rank]
            group_in_channels = op.in_channels // op.group
            group_out_channels = op.out_channels // op.group
            input_suffix = self._param_array_suffix(input_shape)
            weight_suffix = self._param_array_suffix(weight_shape)
            bias_suffix = self._param_array_suffix((op.out_channels,))
            output_suffix = self._param_array_suffix(output_shape)
            param_decls = self._build_param_decls(
                [
                    (params["input0"], c_type, input_suffix, True),
                    (params["weights"], c_type, weight_suffix, True),
                    (
                        params["bias"],
                        c_type,
                        bias_suffix,
                        True,
                    )
                    if params["bias"]
                    else (None, "", "", True),
                    (params["output"], c_type, output_suffix, False),
                ]
            )
            rendered = conv_transpose_template.render(
                model_name=model.name,
                op_name=op_name,
                input0=params["input0"],
                weights=params["weights"],
                bias=params["bias"],
                output=params["output"],
                params=param_decls,
                c_type=c_type,
                zero_literal=zero_literal,
                input_suffix=input_suffix,
                weight_suffix=weight_suffix,
                bias_suffix=bias_suffix,
                output_suffix=output_suffix,
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
                in_indices=in_indices,
                kernel_indices=kernel_indices,
                out_indices=out_indices,
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, AveragePoolOp):
            params = self._shared_param_map(
                [("input0", op.input0), ("output", op.output)]
            )
            input_shape = (op.batch, op.channels, op.in_h, op.in_w)
            output_shape = (op.batch, op.channels, op.out_h, op.out_w)
            input_suffix = self._param_array_suffix(input_shape)
            output_suffix = self._param_array_suffix(output_shape)
            param_decls = self._build_param_decls(
                [
                    (params["input0"], c_type, input_suffix, True),
                    (params["output"], c_type, output_suffix, False),
                ]
            )
            rendered = avg_pool_template.render(
                model_name=model.name,
                op_name=op_name,
                input0=params["input0"],
                output=params["output"],
                params=param_decls,
                c_type=c_type,
                zero_literal=zero_literal,
                input_suffix=input_suffix,
                output_suffix=output_suffix,
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
            return with_node_comment(rendered)
        if isinstance(op, LpPoolOp):
            params = self._shared_param_map(
                [("input0", op.input0), ("output", op.output)]
            )
            input_shape = (op.batch, op.channels, op.in_h, op.in_w)
            output_shape = (op.batch, op.channels, op.out_h, op.out_w)
            input_suffix = self._param_array_suffix(input_shape)
            output_suffix = self._param_array_suffix(output_shape)
            param_decls = self._build_param_decls(
                [
                    (params["input0"], c_type, input_suffix, True),
                    (params["output"], c_type, output_suffix, False),
                ]
            )
            rendered = lp_pool_template.render(
                model_name=model.name,
                op_name=op_name,
                input0=params["input0"],
                output=params["output"],
                params=param_decls,
                c_type=c_type,
                input_suffix=input_suffix,
                output_suffix=output_suffix,
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
                p=op.p,
                zero_literal=zero_literal,
                abs_fn=CEmitter._math_fn(op.dtype, "fabsf", "fabs"),
                pow_fn=CEmitter._math_fn(op.dtype, "powf", "pow"),
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, BatchNormOp):
            params = self._shared_param_map(
                [
                    ("input0", op.input0),
                    ("scale", op.scale),
                    ("bias", op.bias),
                    ("mean", op.mean),
                    ("variance", op.variance),
                    ("output", op.output),
                ]
            )
            shape = CEmitter._codegen_shape(op.shape)
            loop_vars = CEmitter._loop_vars(shape)
            input_suffix = self._param_array_suffix(shape)
            output_suffix = self._param_array_suffix(shape)
            scale_suffix = self._param_array_suffix((op.channels,))
            bias_suffix = self._param_array_suffix((op.channels,))
            mean_suffix = self._param_array_suffix((op.channels,))
            variance_suffix = self._param_array_suffix((op.channels,))
            param_decls = self._build_param_decls(
                [
                    (params["input0"], c_type, input_suffix, True),
                    (params["scale"], c_type, scale_suffix, True),
                    (params["bias"], c_type, bias_suffix, True),
                    (params["mean"], c_type, mean_suffix, True),
                    (params["variance"], c_type, variance_suffix, True),
                    (params["output"], c_type, output_suffix, False),
                ]
            )
            rendered = batch_norm_template.render(
                model_name=model.name,
                op_name=op_name,
                input0=params["input0"],
                scale=params["scale"],
                bias=params["bias"],
                mean=params["mean"],
                variance=params["variance"],
                output=params["output"],
                params=param_decls,
                c_type=c_type,
                input_suffix=input_suffix,
                output_suffix=output_suffix,
                scale_suffix=scale_suffix,
                bias_suffix=bias_suffix,
                mean_suffix=mean_suffix,
                variance_suffix=variance_suffix,
                shape=shape,
                loop_vars=loop_vars,
                epsilon_literal=CEmitter._format_floating(op.epsilon, op.dtype),
                sqrt_fn=CEmitter._math_fn(op.dtype, "sqrtf", "sqrt"),
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, LpNormalizationOp):
            params = self._shared_param_map(
                [("input0", op.input0), ("output", op.output)]
            )
            shape = CEmitter._codegen_shape(op.shape)
            array_suffix = self._param_array_suffix(shape)
            param_decls = self._build_param_decls(
                [
                    (params["input0"], c_type, array_suffix, True),
                    (params["output"], c_type, array_suffix, False),
                ]
            )
            rendered = lp_norm_template.render(
                model_name=model.name,
                op_name=op_name,
                input0=params["input0"],
                output=params["output"],
                params=param_decls,
                c_type=c_type,
                array_suffix=array_suffix,
                outer=op.outer,
                axis_size=op.axis_size,
                inner=op.inner,
                p=op.p,
                zero_literal=zero_literal,
                abs_fn=CEmitter._math_fn(op.dtype, "fabsf", "fabs"),
                sqrt_fn=CEmitter._math_fn(op.dtype, "sqrtf", "sqrt"),
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, InstanceNormalizationOp):
            params = self._shared_param_map(
                [
                    ("input0", op.input0),
                    ("scale", op.scale),
                    ("bias", op.bias),
                    ("output", op.output),
                ]
            )
            shape = CEmitter._codegen_shape(op.shape)
            loop_vars = CEmitter._loop_vars(shape)
            input_suffix = self._param_array_suffix(shape)
            output_suffix = self._param_array_suffix(shape)
            scale_suffix = self._param_array_suffix((op.channels,))
            bias_suffix = self._param_array_suffix((op.channels,))
            param_decls = self._build_param_decls(
                [
                    (params["input0"], c_type, input_suffix, True),
                    (params["scale"], c_type, scale_suffix, True),
                    (params["bias"], c_type, bias_suffix, True),
                    (params["output"], c_type, output_suffix, False),
                ]
            )
            rendered = instance_norm_template.render(
                model_name=model.name,
                op_name=op_name,
                input0=params["input0"],
                scale=params["scale"],
                bias=params["bias"],
                output=params["output"],
                params=param_decls,
                c_type=c_type,
                zero_literal=zero_literal,
                input_suffix=input_suffix,
                output_suffix=output_suffix,
                scale_suffix=scale_suffix,
                bias_suffix=bias_suffix,
                shape=shape,
                loop_vars=loop_vars,
                spatial_size=op.spatial_size,
                epsilon_literal=CEmitter._format_floating(op.epsilon, op.dtype),
                sqrt_fn=CEmitter._math_fn(op.dtype, "sqrtf", "sqrt"),
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, GroupNormalizationOp):
            params = self._shared_param_map(
                [
                    ("input0", op.input0),
                    ("scale", op.scale),
                    ("bias", op.bias),
                    ("output", op.output),
                ]
            )
            shape = CEmitter._codegen_shape(op.shape)
            loop_vars = CEmitter._loop_vars(shape)
            input_suffix = self._param_array_suffix(shape)
            output_suffix = self._param_array_suffix(shape)
            scale_suffix = self._param_array_suffix((op.channels,))
            bias_suffix = self._param_array_suffix((op.channels,))
            param_decls = self._build_param_decls(
                [
                    (params["input0"], c_type, input_suffix, True),
                    (params["scale"], c_type, scale_suffix, True),
                    (params["bias"], c_type, bias_suffix, True),
                    (params["output"], c_type, output_suffix, False),
                ]
            )
            rendered = group_norm_template.render(
                model_name=model.name,
                op_name=op_name,
                input0=params["input0"],
                scale=params["scale"],
                bias=params["bias"],
                output=params["output"],
                params=param_decls,
                c_type=c_type,
                zero_literal=zero_literal,
                input_suffix=input_suffix,
                output_suffix=output_suffix,
                scale_suffix=scale_suffix,
                bias_suffix=bias_suffix,
                shape=shape,
                loop_vars=loop_vars,
                num_groups=op.num_groups,
                group_size=op.group_size,
                spatial_size=op.spatial_size,
                epsilon_literal=CEmitter._format_floating(op.epsilon, op.dtype),
                sqrt_fn=CEmitter._math_fn(op.dtype, "sqrtf", "sqrt"),
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, LayerNormalizationOp):
            acc_dtype = (
                ScalarType.F32
                if op.dtype in {ScalarType.F16, ScalarType.F32}
                else op.dtype
            )
            acc_type = acc_dtype.c_type
            acc_zero_literal = CEmitter._format_literal(acc_dtype, 0)
            acc_one_literal = CEmitter._format_literal(acc_dtype, 1)
            acc_epsilon_literal = CEmitter._format_floating(
                op.epsilon, acc_dtype
            )
            acc_sqrt_fn = CEmitter._math_fn(acc_dtype, "sqrtf", "sqrt")
            use_kahan = op.dtype in {ScalarType.F16, ScalarType.F32}
            params = self._shared_param_map(
                [
                    ("input0", op.input0),
                    ("scale", op.scale),
                    ("bias", op.bias),
                    ("output", op.output),
                    ("mean_output", op.mean_output),
                    ("invstd_output", op.invstd_output),
                ]
            )
            shape = CEmitter._codegen_shape(op.shape)
            loop_vars = CEmitter._loop_vars(shape)
            prefix_loop_vars = loop_vars[: op.axis]
            norm_loop_vars = loop_vars[op.axis :]
            scale_index_vars = [
                "0" if dim == 1 else var
                for dim, var in zip(op.scale_shape, norm_loop_vars)
            ]
            bias_index_vars = None
            if op.bias_shape is not None and op.bias is not None:
                bias_index_vars = [
                    "0" if dim == 1 else var
                    for dim, var in zip(op.bias_shape, norm_loop_vars)
                ]
            mean_index_vars = [
                *prefix_loop_vars,
                *("0" for _ in norm_loop_vars),
            ]
            input_suffix = self._param_array_suffix(shape)
            output_suffix = self._param_array_suffix(shape)
            scale_suffix = self._param_array_suffix(op.scale_shape)
            bias_suffix = (
                self._param_array_suffix(op.bias_shape)
                if op.bias_shape is not None
                else ""
            )
            mean_suffix = (
                self._param_array_suffix(
                    op.shape[: op.axis]
                    + (1,) * len(op.normalized_shape)
                )
                if op.mean_output is not None
                else ""
            )
            invstd_suffix = (
                self._param_array_suffix(
                    op.shape[: op.axis]
                    + (1,) * len(op.normalized_shape)
                )
                if op.invstd_output is not None
                else ""
            )
            param_decls = self._build_param_decls(
                [
                    (params["input0"], c_type, input_suffix, True),
                    (params["scale"], c_type, scale_suffix, True),
                    (
                        params["bias"],
                        c_type,
                        bias_suffix,
                        True,
                    )
                    if params["bias"]
                    else (None, "", "", True),
                    (params["output"], c_type, output_suffix, False),
                    (
                        params["mean_output"],
                        c_type,
                        mean_suffix,
                        False,
                    )
                    if params["mean_output"]
                    else (None, "", "", False),
                    (
                        params["invstd_output"],
                        c_type,
                        invstd_suffix,
                        False,
                    )
                    if params["invstd_output"]
                    else (None, "", "", False),
                ]
            )
            rendered = layer_norm_template.render(
                model_name=model.name,
                op_name=op_name,
                input0=params["input0"],
                scale=params["scale"],
                bias=params["bias"],
                output=params["output"],
                mean_output=params["mean_output"],
                invstd_output=params["invstd_output"],
                params=param_decls,
                c_type=c_type,
                zero_literal=zero_literal,
                input_suffix=input_suffix,
                output_suffix=output_suffix,
                scale_suffix=scale_suffix,
                bias_suffix=bias_suffix,
                mean_suffix=mean_suffix,
                invstd_suffix=invstd_suffix,
                prefix_shape=shape[: op.axis],
                norm_shape=shape[op.axis :],
                prefix_loop_vars=prefix_loop_vars,
                norm_loop_vars=norm_loop_vars,
                scale_index_vars=scale_index_vars,
                bias_index_vars=bias_index_vars,
                mean_index_vars=mean_index_vars,
                inner=op.inner,
                acc_type=acc_type,
                acc_zero_literal=acc_zero_literal,
                acc_one_literal=acc_one_literal,
                acc_epsilon_literal=acc_epsilon_literal,
                acc_sqrt_fn=acc_sqrt_fn,
                use_kahan=use_kahan,
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, MeanVarianceNormalizationOp):
            params = self._shared_param_map(
                [("input0", op.input0), ("output", op.output)]
            )
            shape = CEmitter._codegen_shape(op.shape)
            loop_vars = CEmitter._loop_vars(shape)
            input_suffix = self._param_array_suffix(shape)
            output_suffix = self._param_array_suffix(shape)
            param_decls = self._build_param_decls(
                [
                    (params["input0"], c_type, input_suffix, True),
                    (params["output"], c_type, output_suffix, False),
                ]
            )
            rendered = mean_variance_norm_template.render(
                model_name=model.name,
                op_name=op_name,
                input0=params["input0"],
                output=params["output"],
                params=param_decls,
                c_type=c_type,
                zero_literal=zero_literal,
                input_suffix=input_suffix,
                output_suffix=output_suffix,
                shape=shape,
                loop_vars=loop_vars,
                axes=op.axes,
                non_axes=op.non_axes,
                reduce_count=op.reduce_count,
                epsilon_literal=CEmitter._format_floating(op.epsilon, op.dtype),
                sqrt_fn=CEmitter._math_fn(op.dtype, "sqrtf", "sqrt"),
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, RMSNormalizationOp):
            params = self._shared_param_map(
                [
                    ("input0", op.input0),
                    ("scale", op.scale),
                    ("output", op.output),
                ]
            )
            shape = CEmitter._codegen_shape(op.shape)
            loop_vars = CEmitter._loop_vars(shape)
            prefix_loop_vars = loop_vars[: op.axis]
            norm_loop_vars = loop_vars[op.axis :]
            scale_index_vars = [
                "0" if dim == 1 else var
                for dim, var in zip(op.scale_shape, norm_loop_vars)
            ]
            input_suffix = self._param_array_suffix(shape)
            output_suffix = self._param_array_suffix(shape)
            scale_suffix = self._param_array_suffix(op.scale_shape)
            param_decls = self._build_param_decls(
                [
                    (params["input0"], c_type, input_suffix, True),
                    (params["scale"], c_type, scale_suffix, True),
                    (params["output"], c_type, output_suffix, False),
                ]
            )
            rendered = rms_norm_template.render(
                model_name=model.name,
                op_name=op_name,
                input0=params["input0"],
                scale=params["scale"],
                output=params["output"],
                params=param_decls,
                c_type=c_type,
                zero_literal=zero_literal,
                input_suffix=input_suffix,
                output_suffix=output_suffix,
                scale_suffix=scale_suffix,
                prefix_shape=shape[: op.axis],
                norm_shape=shape[op.axis :],
                prefix_loop_vars=prefix_loop_vars,
                norm_loop_vars=norm_loop_vars,
                scale_index_vars=scale_index_vars,
                inner=op.inner,
                epsilon_literal=CEmitter._format_floating(op.epsilon, op.dtype),
                sqrt_fn=CEmitter._math_fn(op.dtype, "sqrtf", "sqrt"),
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, LrnOp):
            params = self._shared_param_map(
                [("input0", op.input0), ("output", op.output)]
            )
            shape = CEmitter._codegen_shape(op.shape)
            loop_vars = CEmitter._loop_vars(shape)
            input_suffix = self._param_array_suffix(shape)
            output_suffix = self._param_array_suffix(shape)
            param_decls = self._build_param_decls(
                [
                    (params["input0"], c_type, input_suffix, True),
                    (params["output"], c_type, output_suffix, False),
                ]
            )
            rendered = lrn_template.render(
                model_name=model.name,
                op_name=op_name,
                input0=params["input0"],
                output=params["output"],
                params=param_decls,
                c_type=c_type,
                input_suffix=input_suffix,
                output_suffix=output_suffix,
                shape=shape,
                channels=op.channels,
                half=op.half,
                loop_vars=loop_vars,
                zero_literal=zero_literal,
                alpha_div_size_literal=CEmitter._format_floating(
                    op.alpha / op.size, op.dtype
                ),
                beta_literal=CEmitter._format_floating(op.beta, op.dtype),
                bias_literal=CEmitter._format_floating(op.bias, op.dtype),
                pow_fn=CEmitter._math_fn(op.dtype, "powf", "pow"),
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, LstmOp):
            params = self._shared_param_map(
                [
                    ("input_x", op.input_x),
                    ("input_w", op.input_w),
                    ("input_r", op.input_r),
                    ("input_b", op.input_b),
                    ("input_sequence_lens", op.input_sequence_lens),
                    ("input_initial_h", op.input_initial_h),
                    ("input_initial_c", op.input_initial_c),
                    ("input_p", op.input_p),
                    ("output_y", op.output_y),
                    ("output_y_h", op.output_y_h),
                    ("output_y_c", op.output_y_c),
                ]
            )
            input_x_shape = (
                (op.seq_length, op.batch_size, op.input_size)
                if op.layout == 0
                else (op.batch_size, op.seq_length, op.input_size)
            )
            w_shape = (op.num_directions, 4 * op.hidden_size, op.input_size)
            r_shape = (op.num_directions, 4 * op.hidden_size, op.hidden_size)
            b_shape = (
                (op.num_directions, 8 * op.hidden_size)
                if op.input_b is not None
                else None
            )
            seq_shape = (op.batch_size,) if op.input_sequence_lens is not None else None
            state_shape = (
                (op.num_directions, op.batch_size, op.hidden_size)
                if op.layout == 0
                else (op.batch_size, op.num_directions, op.hidden_size)
            )
            h_shape = (
                state_shape
                if op.input_initial_h is not None or op.output_y_h is not None
                else None
            )
            c_shape = (
                state_shape
                if op.input_initial_c is not None or op.output_y_c is not None
                else None
            )
            p_shape = (
                (op.num_directions, 3 * op.hidden_size)
                if op.input_p is not None
                else None
            )
            y_shape = (
                (op.seq_length, op.num_directions, op.batch_size, op.hidden_size)
                if op.layout == 0
                else (op.batch_size, op.seq_length, op.num_directions, op.hidden_size)
            )
            param_decls = self._build_param_decls(
                [
                    (
                        params["input_x"],
                        c_type,
                        self._param_array_suffix(input_x_shape),
                        True,
                    ),
                    (
                        params["input_w"],
                        c_type,
                        self._param_array_suffix(w_shape),
                        True,
                    ),
                    (
                        params["input_r"],
                        c_type,
                        self._param_array_suffix(r_shape),
                        True,
                    ),
                    (
                        params["input_b"],
                        c_type,
                        self._param_array_suffix(b_shape),
                        True,
                    )
                    if params["input_b"]
                    else (None, "", "", True),
                    (
                        params["input_sequence_lens"],
                        (op.sequence_lens_dtype or ScalarType.I64).c_type,
                        self._param_array_suffix(seq_shape),
                        True,
                    )
                    if params["input_sequence_lens"]
                    else (None, "", "", True),
                    (
                        params["input_initial_h"],
                        c_type,
                        self._param_array_suffix(h_shape),
                        True,
                    )
                    if params["input_initial_h"]
                    else (None, "", "", True),
                    (
                        params["input_initial_c"],
                        c_type,
                        self._param_array_suffix(c_shape),
                        True,
                    )
                    if params["input_initial_c"]
                    else (None, "", "", True),
                    (
                        params["input_p"],
                        c_type,
                        self._param_array_suffix(p_shape),
                        True,
                    )
                    if params["input_p"]
                    else (None, "", "", True),
                    (
                        params["output_y"],
                        c_type,
                        self._param_array_suffix(y_shape),
                        False,
                    )
                    if params["output_y"]
                    else (None, "", "", False),
                    (
                        params["output_y_h"],
                        c_type,
                        self._param_array_suffix(h_shape),
                        False,
                    )
                    if params["output_y_h"]
                    else (None, "", "", False),
                    (
                        params["output_y_c"],
                        c_type,
                        self._param_array_suffix(c_shape),
                        False,
                    )
                    if params["output_y_c"]
                    else (None, "", "", False),
                ]
            )
            if scalar_registry is None:
                raise CodegenError(
                    "Scalar function registry is required for LSTM codegen."
                )
            activation_functions = tuple(
                self._lstm_activation_function_name(
                    kind,
                    alpha,
                    beta,
                    op.dtype,
                    scalar_registry,
                )
                for kind, alpha, beta in zip(
                    op.activation_kinds,
                    op.activation_alphas,
                    op.activation_betas,
                )
            )
            rendered = lstm_template.render(
                model_name=model.name,
                op_name=op_name,
                input_x=params["input_x"],
                input_w=params["input_w"],
                input_r=params["input_r"],
                input_b=params["input_b"],
                input_sequence_lens=params["input_sequence_lens"],
                input_initial_h=params["input_initial_h"],
                input_initial_c=params["input_initial_c"],
                input_p=params["input_p"],
                output_y=params["output_y"],
                output_y_h=params["output_y_h"],
                output_y_c=params["output_y_c"],
                params=param_decls,
                c_type=c_type,
                seq_c_type=(op.sequence_lens_dtype or ScalarType.I64).c_type,
                zero_literal=zero_literal,
                one_literal=CEmitter._format_literal(op.dtype, 1),
                clip_literal=(
                    CEmitter._format_floating(op.clip, op.dtype)
                    if op.clip is not None
                    else CEmitter._format_literal(op.dtype, 0)
                ),
                use_clip=int(op.clip is not None and op.clip > 0),
                seq_length=op.seq_length,
                batch_size=op.batch_size,
                input_size=op.input_size,
                hidden_size=op.hidden_size,
                num_directions=op.num_directions,
                layout=op.layout,
                direction=op.direction,
                input_forget=op.input_forget,
                activation_functions=activation_functions,
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, AdagradOp):
            params = self._shared_param_map(
                [
                    ("rate", op.rate),
                    ("timestep", op.timestep),
                    *(
                        (f"input{idx}", name)
                        for idx, name in enumerate(op.inputs)
                    ),
                    *(
                        (f"grad{idx}", name)
                        for idx, name in enumerate(op.gradients)
                    ),
                    *(
                        (f"acc{idx}", name)
                        for idx, name in enumerate(op.accumulators)
                    ),
                    *(
                        (f"output{idx}", name)
                        for idx, name in enumerate(op.outputs)
                    ),
                    *(
                        (f"acc_output{idx}", name)
                        for idx, name in enumerate(op.accumulator_outputs)
                    ),
                ]
            )
            rate_suffix = self._param_array_suffix(
                op.rate_shape, _dim_names_for(op.rate)
            )
            timestep_suffix = self._param_array_suffix(
                op.timestep_shape, _dim_names_for(op.timestep)
            )
            param_specs = [
                (params["rate"], op.rate_dtype.c_type, rate_suffix, True),
                (
                    params["timestep"],
                    op.timestep_dtype.c_type,
                    timestep_suffix,
                    True,
                ),
            ]
            tensor_specs = []
            for idx, shape in enumerate(op.output_shapes):
                input_suffix = self._param_array_suffix(
                    op.tensor_shapes[idx], _dim_names_for(op.inputs[idx])
                )
                grad_suffix = self._param_array_suffix(
                    op.tensor_shapes[idx], _dim_names_for(op.gradients[idx])
                )
                acc_suffix = self._param_array_suffix(
                    op.tensor_shapes[idx], _dim_names_for(op.accumulators[idx])
                )
                output_suffix = self._param_array_suffix(
                    op.output_shapes[idx], _dim_names_for(op.outputs[idx])
                )
                acc_output_suffix = self._param_array_suffix(
                    op.output_shapes[idx],
                    _dim_names_for(op.accumulator_outputs[idx]),
                )
                param_specs.extend(
                    [
                        (params[f"input{idx}"], c_type, input_suffix, True),
                        (params[f"grad{idx}"], c_type, grad_suffix, True),
                        (params[f"acc{idx}"], c_type, acc_suffix, True),
                        (params[f"output{idx}"], c_type, output_suffix, False),
                        (
                            params[f"acc_output{idx}"],
                            c_type,
                            acc_output_suffix,
                            False,
                        ),
                    ]
                )
                output_dim_names = _dim_names_for(op.outputs[idx])
                shape_exprs = CEmitter._shape_dim_exprs(
                    shape, output_dim_names
                )
                loop_vars = CEmitter._loop_vars(shape)
                index_suffix = "".join(f"[{var}]" for var in loop_vars)
                tensor_specs.append(
                    {
                        "shape": shape_exprs,
                        "loop_vars": loop_vars,
                        "input_expr": f"{params[f'input{idx}']}{index_suffix}",
                        "grad_expr": f"{params[f'grad{idx}']}{index_suffix}",
                        "acc_expr": f"{params[f'acc{idx}']}{index_suffix}",
                        "output_expr": f"{params[f'output{idx}']}{index_suffix}",
                        "acc_output_expr": f"{params[f'acc_output{idx}']}{index_suffix}",
                    }
                )
            param_decls = self._build_param_decls(param_specs)
            rendered = adagrad_template.render(
                model_name=model.name,
                op_name=op_name,
                rate=params["rate"],
                timestep=params["timestep"],
                params=param_decls,
                c_type=c_type,
                one_literal=CEmitter._format_literal(op.dtype, 1),
                decay_factor_literal=CEmitter._format_floating(
                    op.decay_factor, op.dtype
                ),
                norm_coefficient_literal=CEmitter._format_floating(
                    op.norm_coefficient, op.dtype
                ),
                epsilon_literal=CEmitter._format_floating(op.epsilon, op.dtype),
                sqrt_fn=CEmitter._math_fn(op.dtype, "sqrtf", "sqrt"),
                tensors=tensor_specs,
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, SoftmaxOp):
            if scalar_registry is None:
                raise CodegenError(
                    "Scalar function registry is required for Softmax rendering."
                )
            max_fn = self._scalar_function_name(
                ScalarFunction.MAXIMUM, op.dtype, scalar_registry
            )
            if max_fn is None:
                raise CodegenError(
                    "Failed to resolve scalar maximum function for Softmax."
                )
            params = self._shared_param_map(
                [("input0", op.input0), ("output", op.output)]
            )
            array_suffix = self._param_array_suffix(op.shape)
            param_decls = self._build_param_decls(
                [
                    (params["input0"], c_type, array_suffix, True),
                    (params["output"], c_type, array_suffix, False),
                ]
            )
            rendered = softmax_template.render(
                model_name=model.name,
                op_name=op_name,
                input0=params["input0"],
                output=params["output"],
                params=param_decls,
                c_type=c_type,
                array_suffix=array_suffix,
                outer=op.outer,
                axis_size=op.axis_size,
                inner=op.inner,
                max_fn=max_fn,
                exp_fn=CEmitter._math_fn(op.dtype, "expf", "exp"),
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, LogSoftmaxOp):
            if scalar_registry is None:
                raise CodegenError(
                    "Scalar function registry is required for LogSoftmax rendering."
                )
            max_fn = self._scalar_function_name(
                ScalarFunction.MAXIMUM, op.dtype, scalar_registry
            )
            if max_fn is None:
                raise CodegenError(
                    "Failed to resolve scalar maximum function for LogSoftmax."
                )
            params = self._shared_param_map(
                [("input0", op.input0), ("output", op.output)]
            )
            array_suffix = self._param_array_suffix(op.shape)
            param_decls = self._build_param_decls(
                [
                    (params["input0"], c_type, array_suffix, True),
                    (params["output"], c_type, array_suffix, False),
                ]
            )
            rendered = logsoftmax_template.render(
                model_name=model.name,
                op_name=op_name,
                input0=params["input0"],
                output=params["output"],
                params=param_decls,
                c_type=c_type,
                array_suffix=array_suffix,
                outer=op.outer,
                axis_size=op.axis_size,
                inner=op.inner,
                max_fn=max_fn,
                exp_fn=CEmitter._math_fn(op.dtype, "expf", "exp"),
                log_fn=CEmitter._math_fn(op.dtype, "logf", "log"),
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, HardmaxOp):
            if scalar_registry is None:
                raise CodegenError(
                    "Scalar function registry is required for Hardmax rendering."
                )
            max_fn = self._scalar_function_name(
                ScalarFunction.MAXIMUM, op.dtype, scalar_registry
            )
            if max_fn is None:
                raise CodegenError(
                    "Failed to resolve scalar maximum function for Hardmax."
                )
            params = self._shared_param_map(
                [("input0", op.input0), ("output", op.output)]
            )
            array_suffix = self._param_array_suffix(op.shape)
            param_decls = self._build_param_decls(
                [
                    (params["input0"], c_type, array_suffix, True),
                    (params["output"], c_type, array_suffix, False),
                ]
            )
            rendered = hardmax_template.render(
                model_name=model.name,
                op_name=op_name,
                input0=params["input0"],
                output=params["output"],
                params=param_decls,
                c_type=c_type,
                array_suffix=array_suffix,
                outer=op.outer,
                axis_size=op.axis_size,
                inner=op.inner,
                zero_literal=zero_literal,
                one_literal=CEmitter._format_literal(op.dtype, 1),
                max_fn=max_fn,
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, NegativeLogLikelihoodLossOp):
            acc_dtype = (
                ScalarType.F64
                if op.dtype in {ScalarType.F16, ScalarType.F32}
                else op.dtype
            )
            acc_type = acc_dtype.c_type
            acc_zero_literal = CEmitter._format_literal(acc_dtype, 0)
            acc_one_literal = CEmitter._format_literal(acc_dtype, 1)
            params = self._shared_param_map(
                [
                    ("input0", op.input0),
                    ("target", op.target),
                    ("weight", op.weight),
                    ("output", op.output),
                ]
            )
            input_suffix = self._param_array_suffix(op.input_shape)
            target_suffix = self._param_array_suffix(op.target_shape)
            output_suffix = self._param_array_suffix(op.output_shape)
            weight_suffix = f"[{op.c}]"
            param_decls = self._build_param_decls(
                [
                    (params["input0"], c_type, input_suffix, True),
                    (params["target"], op.target_dtype.c_type, target_suffix, True),
                    (
                        params["weight"],
                        c_type,
                        weight_suffix,
                        True,
                    )
                    if params["weight"]
                    else (None, "", "", True),
                    (params["output"], c_type, output_suffix, False),
                ]
            )
            rendered = nllloss_template.render(
                model_name=model.name,
                op_name=op_name,
                input0=params["input0"],
                target=params["target"],
                weight=params["weight"],
                output=params["output"],
                params=param_decls,
                c_type=c_type,
                target_c_type=op.target_dtype.c_type,
                input_suffix=input_suffix,
                target_suffix=target_suffix,
                output_suffix=output_suffix,
                n=op.n,
                c=op.c,
                d=op.d,
                reduction=op.reduction,
                ignore_index=op.ignore_index,
                zero_literal=zero_literal,
                one_literal=CEmitter._format_literal(op.dtype, 1),
                acc_type=acc_type,
                acc_zero_literal=acc_zero_literal,
                acc_one_literal=acc_one_literal,
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, SoftmaxCrossEntropyLossOp):
            acc_dtype = (
                ScalarType.F64
                if op.dtype in {ScalarType.F16, ScalarType.F32}
                else op.dtype
            )
            if scalar_registry is None:
                raise CodegenError(
                    "Scalar function registry is required for SoftmaxCrossEntropyLoss."
                )
            max_fn = self._scalar_function_name(
                ScalarFunction.MAXIMUM, acc_dtype, scalar_registry
            )
            if max_fn is None:
                raise CodegenError(
                    "Failed to resolve scalar maximum function for SoftmaxCrossEntropyLoss."
                )
            acc_type = acc_dtype.c_type
            acc_zero_literal = CEmitter._format_literal(acc_dtype, 0)
            acc_one_literal = CEmitter._format_literal(acc_dtype, 1)
            acc_exp_fn = CEmitter._math_fn(acc_dtype, "expf", "exp")
            acc_log_fn = CEmitter._math_fn(acc_dtype, "logf", "log")
            params = self._shared_param_map(
                [
                    ("input0", op.input0),
                    ("target", op.target),
                    ("weight", op.weight),
                    ("output", op.output),
                    ("log_prob", op.log_prob),
                ]
            )
            use_ignore_index = int(op.ignore_index is not None)
            ignore_index = op.ignore_index if op.ignore_index is not None else -1
            input_suffix = self._param_array_suffix(op.input_shape)
            target_suffix = self._param_array_suffix(op.target_shape)
            output_suffix = self._param_array_suffix(op.output_shape)
            log_prob_suffix = (
                self._param_array_suffix(op.log_prob_shape)
                if op.log_prob_shape is not None
                else ""
            )
            weight_suffix = f"[{op.c}]"
            param_decls = self._build_param_decls(
                [
                    (params["input0"], c_type, input_suffix, True),
                    (params["target"], op.target_dtype.c_type, target_suffix, True),
                    (
                        params["weight"],
                        c_type,
                        weight_suffix,
                        True,
                    )
                    if params["weight"]
                    else (None, "", "", True),
                    (params["output"], c_type, output_suffix, False),
                    (
                        params["log_prob"],
                        c_type,
                        log_prob_suffix,
                        False,
                    )
                    if params["log_prob"]
                    else (None, "", "", False),
                ]
            )
            rendered = softmax_cross_entropy_loss_template.render(
                model_name=model.name,
                op_name=op_name,
                input0=params["input0"],
                target=params["target"],
                weight=params["weight"],
                output=params["output"],
                log_prob=params["log_prob"],
                params=param_decls,
                c_type=c_type,
                target_c_type=op.target_dtype.c_type,
                input_suffix=input_suffix,
                target_suffix=target_suffix,
                output_suffix=output_suffix,
                log_prob_suffix=log_prob_suffix if op.log_prob_shape is not None else None,
                n=op.n,
                c=op.c,
                d=op.d,
                reduction=op.reduction,
                use_ignore_index=use_ignore_index,
                ignore_index=ignore_index,
                zero_literal=zero_literal,
                one_literal=CEmitter._format_literal(op.dtype, 1),
                acc_type=acc_type,
                acc_zero_literal=acc_zero_literal,
                acc_one_literal=acc_one_literal,
                acc_exp_fn=acc_exp_fn,
                acc_log_fn=acc_log_fn,
                max_fn=max_fn,
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, MaxPoolOp):
            if scalar_registry is None:
                raise CodegenError(
                    "Scalar function registry is required for MaxPool rendering."
                )
            max_fn = self._scalar_function_name(
                ScalarFunction.MAXIMUM, op.dtype, scalar_registry
            )
            if max_fn is None:
                raise CodegenError(
                    "Failed to resolve scalar maximum function for MaxPool."
                )
            params = self._shared_param_map(
                [
                    ("input0", op.input0),
                    ("output", op.output),
                    ("indices", op.indices),
                ]
            )
            input_shape = (op.batch, op.channels, *op.in_spatial)
            output_shape = (op.batch, op.channels, *op.out_spatial)
            indices_c_type = (
                op.indices_dtype.c_type
                if op.indices is not None and op.indices_dtype is not None
                else None
            )
            input_suffix = self._param_array_suffix(input_shape)
            output_suffix = self._param_array_suffix(output_shape)
            indices_suffix = self._param_array_suffix(output_shape)
            param_decls = self._build_param_decls(
                [
                    (params["input0"], c_type, input_suffix, True),
                    (params["output"], c_type, output_suffix, False),
                    (
                        params["indices"],
                        indices_c_type or ScalarType.I64.c_type,
                        indices_suffix,
                        False,
                    )
                    if params["indices"]
                    else (None, "", "", False),
                ]
            )
            rendered = maxpool_template.render(
                model_name=model.name,
                op_name=op_name,
                input0=params["input0"],
                output=params["output"],
                indices=params["indices"],
                params=param_decls,
                c_type=c_type,
                min_literal=min_literal,
                input_suffix=input_suffix,
                output_suffix=output_suffix,
                indices_suffix=indices_suffix,
                indices_c_type=indices_c_type,
                max_fn=max_fn,
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
                storage_order=op.storage_order,
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, ConcatOp):
            input_params = [
                (f"input_{index}", name) for index, name in enumerate(op.inputs)
            ]
            params = self._shared_param_map(
                [*input_params, ("output", op.output)]
            )
            input_names = tuple(
                params[f"input_{index}"] for index in range(len(op.inputs))
            )
            axis = op.axis
            if axis < 0:
                axis += len(op.output_shape)
            outer = CEmitter._element_count(op.output_shape[:axis] or (1,))
            inner = CEmitter._element_count(op.output_shape[axis + 1 :] or (1,))
            axis_sizes = tuple(shape[axis] for shape in op.input_shapes)
            input_suffixes = tuple(
                self._param_array_suffix(shape) for shape in op.input_shapes
            )
            output_suffix = self._param_array_suffix(op.output_shape)
            param_decls = self._build_param_decls(
                [
                    *(
                        (name, c_type, suffix, True)
                        for name, suffix in zip(input_names, input_suffixes)
                    ),
                    (params["output"], c_type, output_suffix, False),
                ]
            )
            rendered = concat_template.render(
                model_name=model.name,
                op_name=op_name,
                inputs=input_names,
                output=params["output"],
                params=param_decls,
                c_type=c_type,
                input_suffixes=input_suffixes,
                output_suffix=output_suffix,
                axis_sizes=axis_sizes,
                input_count=len(op.inputs),
                outer=outer,
                inner=inner,
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, GatherElementsOp):
            params = self._shared_param_map(
                [
                    ("data", op.data),
                    ("indices", op.indices),
                    ("output", op.output),
                ]
            )
            output_shape = CEmitter._codegen_shape(op.output_shape)
            loop_vars = CEmitter._loop_vars(output_shape)
            data_indices = list(loop_vars)
            data_indices[op.axis] = "gather_index"
            data_suffix = self._param_array_suffix(op.data_shape)
            indices_suffix = self._param_array_suffix(op.indices_shape)
            output_suffix = self._param_array_suffix(op.output_shape)
            param_decls = self._build_param_decls(
                [
                    (params["data"], c_type, data_suffix, True),
                    (
                        params["indices"],
                        op.indices_dtype.c_type,
                        indices_suffix,
                        True,
                    ),
                    (params["output"], c_type, output_suffix, False),
                ]
            )
            rendered = gather_elements_template.render(
                model_name=model.name,
                op_name=op_name,
                data=params["data"],
                indices=params["indices"],
                output=params["output"],
                params=param_decls,
                c_type=c_type,
                indices_c_type=op.indices_dtype.c_type,
                data_suffix=data_suffix,
                indices_suffix=indices_suffix,
                output_suffix=output_suffix,
                output_shape=output_shape,
                loop_vars=loop_vars,
                data_indices=data_indices,
                axis_dim=op.data_shape[op.axis],
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, GatherOp):
            params = self._shared_param_map(
                [
                    ("data", op.data),
                    ("indices", op.indices),
                    ("output", op.output),
                ]
            )
            output_shape = CEmitter._codegen_shape(op.output_shape)
            loop_vars = CEmitter._loop_vars(output_shape)
            output_loop_vars = loop_vars if op.output_shape else ()
            indices_rank = len(op.indices_shape)
            if indices_rank == 0:
                indices_indices = ("0",)
            else:
                indices_indices = output_loop_vars[
                    op.axis : op.axis + indices_rank
                ]
            data_indices = [
                *output_loop_vars[: op.axis],
                "gather_index",
                *output_loop_vars[op.axis + indices_rank :],
            ]
            data_suffix = self._param_array_suffix(op.data_shape)
            indices_suffix = self._param_array_suffix(op.indices_shape)
            output_suffix = self._param_array_suffix(op.output_shape)
            param_decls = self._build_param_decls(
                [
                    (params["data"], c_type, data_suffix, True),
                    (
                        params["indices"],
                        op.indices_dtype.c_type,
                        indices_suffix,
                        True,
                    ),
                    (params["output"], c_type, output_suffix, False),
                ]
            )
            rendered = gather_template.render(
                model_name=model.name,
                op_name=op_name,
                data=params["data"],
                indices=params["indices"],
                output=params["output"],
                params=param_decls,
                c_type=c_type,
                indices_c_type=op.indices_dtype.c_type,
                data_suffix=data_suffix,
                indices_suffix=indices_suffix,
                output_suffix=output_suffix,
                output_shape=output_shape,
                loop_vars=loop_vars,
                indices_indices=indices_indices,
                data_indices=data_indices,
                axis_dim=op.data_shape[op.axis],
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, GatherNDOp):
            params = self._shared_param_map(
                [
                    ("data", op.data),
                    ("indices", op.indices),
                    ("output", op.output),
                ]
            )
            indices_dim_names = _dim_names_for(op.indices)
            data_dim_names = _dim_names_for(op.data)
            data_shape = CEmitter._shape_dim_exprs(op.data_shape, data_dim_names)
            indices_shape = CEmitter._shape_dim_exprs(
                op.indices_shape, indices_dim_names
            )
            indices_prefix_shape = indices_shape[:-1]
            indices_prefix_loop_vars = (
                CEmitter._loop_vars(op.indices_shape[:-1])
                if op.indices_shape[:-1]
                else ()
            )
            index_depth = op.indices_shape[-1]
            tail_shape = data_shape[op.batch_dims + index_depth :]
            tail_loop_vars = (
                tuple(f"t{index}" for index in range(len(tail_shape)))
                if tail_shape
                else ()
            )
            output_loop_vars = (*indices_prefix_loop_vars, *tail_loop_vars)
            if output_loop_vars:
                output_index_expr = params["output"] + "".join(
                    f"[{var}]" for var in output_loop_vars
                )
            else:
                output_index_expr = f"{params['output']}[0]"
            data_index_vars = (
                *indices_prefix_loop_vars[: op.batch_dims],
                *tuple(f"index{idx}" for idx in range(index_depth)),
                *tail_loop_vars,
            )
            data_index_expr = params["data"] + "".join(
                f"[{var}]" for var in data_index_vars
            )
            data_suffix = self._param_array_suffix(op.data_shape)
            indices_suffix = self._param_array_suffix(op.indices_shape)
            output_suffix = self._param_array_suffix(op.output_shape)
            param_decls = self._build_param_decls(
                [
                    (params["data"], c_type, data_suffix, True),
                    (
                        params["indices"],
                        op.indices_dtype.c_type,
                        indices_suffix,
                        True,
                    ),
                    (params["output"], c_type, output_suffix, False),
                ]
            )
            rendered = gather_nd_template.render(
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
                batch_dims=op.batch_dims,
                data_shape=data_shape,
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, ScatterNDOp):
            params = self._shared_param_map(
                [
                    ("data", op.data),
                    ("indices", op.indices),
                    ("updates", op.updates),
                    ("output", op.output),
                ]
            )
            output_dim_names = _dim_names_for(op.output)
            indices_dim_names = _dim_names_for(op.indices)
            updates_dim_names = _dim_names_for(op.updates)
            data_dim_names = _dim_names_for(op.data)
            output_shape = CEmitter._shape_dim_exprs(
                op.output_shape, output_dim_names
            )
            data_shape = CEmitter._shape_dim_exprs(op.data_shape, data_dim_names)
            indices_shape = CEmitter._shape_dim_exprs(
                op.indices_shape, indices_dim_names
            )
            output_loop_vars = CEmitter._loop_vars(op.output_shape)
            indices_prefix_shape = indices_shape[:-1]
            indices_prefix_loop_vars = (
                CEmitter._loop_vars(op.indices_shape[:-1])
                if op.indices_shape[:-1]
                else ()
            )
            index_depth = op.indices_shape[-1]
            tail_shape = output_shape[index_depth:]
            tail_loop_vars = (
                tuple(
                    f"t{index}"
                    for index in range(len(op.output_shape[index_depth:]))
                )
                if op.output_shape[index_depth:]
                else ()
            )
            index_vars = tuple(f"index{idx}" for idx in range(index_depth))
            output_index_expr = f"{params['output']}" + "".join(
                f"[{var}]" for var in (*index_vars, *tail_loop_vars)
            )
            updates_index_vars = (*indices_prefix_loop_vars, *tail_loop_vars)
            if not op.updates_shape:
                updates_index_expr = f"{params['updates']}[0]"
            else:
                updates_index_expr = f"{params['updates']}" + "".join(
                    f"[{var}]" for var in updates_index_vars
                )
            data_suffix = self._param_array_suffix(
                op.data_shape, data_dim_names
            )
            indices_suffix = self._param_array_suffix(
                op.indices_shape, indices_dim_names
            )
            updates_suffix = self._param_array_suffix(
                op.updates_shape, updates_dim_names
            )
            output_suffix = self._param_array_suffix(
                op.output_shape, output_dim_names
            )
            param_decls = self._build_param_decls(
                [
                    (params["data"], c_type, data_suffix, True),
                    (
                        params["indices"],
                        op.indices_dtype.c_type,
                        indices_suffix,
                        True,
                    ),
                    (params["updates"], c_type, updates_suffix, True),
                    (params["output"], c_type, output_suffix, False),
                ]
            )
            rendered = scatter_nd_template.render(
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
                reduction=op.reduction,
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, TensorScatterOp):
            param_pairs = [
                ("past_cache", op.past_cache),
                ("update", op.update),
                ("output", op.output),
            ]
            if op.write_indices is not None:
                param_pairs.insert(2, ("write_indices", op.write_indices))
            params = self._shared_param_map(param_pairs)
            output_dim_names = _dim_names_for(op.output)
            update_dim_names = _dim_names_for(op.update)
            past_dim_names = _dim_names_for(op.past_cache)
            write_indices_dim_names = (
                _dim_names_for(op.write_indices) if op.write_indices else None
            )
            output_shape = CEmitter._shape_dim_exprs(
                op.output_shape, output_dim_names
            )
            update_shape = CEmitter._shape_dim_exprs(
                op.update_shape, update_dim_names
            )
            prefix_shape = output_shape[: op.axis]
            prefix_loop_vars = (
                CEmitter._loop_vars(op.output_shape[: op.axis])
                if op.output_shape[: op.axis]
                else ()
            )
            tail_shape = output_shape[op.axis + 1 :]
            tail_loop_vars = (
                tuple(
                    f"t{index}"
                    for index in range(len(op.output_shape[op.axis + 1 :]))
                )
                if op.output_shape[op.axis + 1 :]
                else ()
            )
            output_loop_vars = CEmitter._loop_vars(op.output_shape)
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
            past_suffix = self._param_array_suffix(
                op.past_cache_shape, past_dim_names
            )
            update_suffix = self._param_array_suffix(
                op.update_shape, update_dim_names
            )
            output_suffix = self._param_array_suffix(
                op.output_shape, output_dim_names
            )
            param_decls = [
                (params["past_cache"], c_type, past_suffix, True),
                (params["update"], c_type, update_suffix, True),
            ]
            if op.write_indices is not None and op.write_indices_dtype is not None:
                write_indices_suffix = self._param_array_suffix(
                    op.write_indices_shape or (), write_indices_dim_names
                )
                param_decls.append(
                    (
                        params["write_indices"],
                        op.write_indices_dtype.c_type,
                        write_indices_suffix,
                        True,
                    )
                )
            param_decls.append((params["output"], c_type, output_suffix, False))
            param_decls_rendered = self._build_param_decls(param_decls)
            rendered = tensor_scatter_template.render(
                model_name=model.name,
                op_name=op_name,
                past_cache=params["past_cache"],
                update=params["update"],
                write_indices=(
                    params.get("write_indices") if op.write_indices else None
                ),
                output=params["output"],
                params=param_decls_rendered,
                c_type=c_type,
                output_shape=output_shape,
                output_loop_vars=output_loop_vars,
                prefix_shape=prefix_shape,
                prefix_loop_vars=prefix_loop_vars,
                sequence_dim=update_shape[op.axis],
                sequence_loop_var=sequence_loop_var,
                tail_shape=tail_shape,
                tail_loop_vars=tail_loop_vars,
                output_index_expr=output_index_expr,
                update_index_expr=update_index_expr,
                max_sequence_length=output_shape[op.axis],
                write_indices_present=op.write_indices is not None,
                batch_index_var=prefix_loop_vars[0]
                if prefix_loop_vars
                else "0",
                write_index_var=write_index_var,
                cache_index_var=cache_index_var,
                circular=op.mode == "circular",
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, TransposeOp):
            params = self._shared_param_map(
                [("input0", op.input0), ("output", op.output)]
            )
            output_shape = CEmitter._codegen_shape(op.output_shape)
            loop_vars = CEmitter._loop_vars(output_shape)
            output_suffix = self._param_array_suffix(output_shape)
            input_suffix = self._param_array_suffix(op.input_shape)
            param_decls = self._build_param_decls(
                [
                    (params["input0"], c_type, input_suffix, True),
                    (params["output"], c_type, output_suffix, False),
                ]
            )
            if not op.input_shape:
                input_indices = [loop_vars[0]]
            else:
                input_indices = [None] * len(op.perm)
                for output_axis, input_axis in enumerate(op.perm):
                    input_indices[input_axis] = loop_vars[output_axis]
            rendered = transpose_template.render(
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
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, ReshapeOp):
            params = self._shared_param_map(
                [("input0", op.input0), ("output", op.output)]
            )
            input_suffix = self._param_array_suffix(op.input_shape)
            output_shape = CEmitter._codegen_shape(op.output_shape)
            output_suffix = self._param_array_suffix(op.output_shape)
            param_decls = self._build_param_decls(
                [
                    (params["input0"], c_type, input_suffix, True),
                    (params["output"], c_type, output_suffix, False),
                ]
            )
            loop_vars = CEmitter._loop_vars(op.output_shape)
            rendered = reshape_template.render(
                model_name=model.name,
                op_name=op_name,
                input0=params["input0"],
                output=params["output"],
                params=param_decls,
                c_type=c_type,
                input_suffix=input_suffix,
                output_suffix=output_suffix,
                element_count=CEmitter._element_count(op.output_shape),
                output_shape=output_shape,
                loop_vars=loop_vars,
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, IdentityOp):
            params = self._shared_param_map(
                [("input0", op.input0), ("output", op.output)]
            )
            output_dim_names = _dim_names_for(op.output)
            shape = CEmitter._shape_dim_exprs(op.shape, output_dim_names)
            loop_vars = CEmitter._loop_vars(op.shape)
            output_suffix = self._param_array_suffix(shape, output_dim_names)
            input_suffix = self._param_array_suffix(shape, _dim_names_for(op.input0))
            param_decls = self._build_param_decls(
                [
                    (params["input0"], c_type, input_suffix, True),
                    (params["output"], c_type, output_suffix, False),
                ]
            )
            rendered = identity_template.render(
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
            return with_node_comment(rendered)
        if isinstance(op, EyeLikeOp):
            params = self._shared_param_map(
                [("input0", op.input0), ("output", op.output)]
            )
            output_dim_names = _dim_names_for(op.output)
            shape = CEmitter._shape_dim_exprs(op.output_shape, output_dim_names)
            output_suffix = self._param_array_suffix(op.output_shape, output_dim_names)
            input_suffix = self._param_array_suffix(
                op.output_shape, _dim_names_for(op.input0)
            )
            batch_dims = op.output_shape[:-2]
            batch_size = CEmitter._element_count(batch_dims or (1,))
            param_decls = self._build_param_decls(
                [
                    (params["input0"], c_type, input_suffix, True),
                    (params["output"], c_type, output_suffix, False),
                ]
            )
            rendered = eye_like_template.render(
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
                rows=op.output_shape[-2],
                cols=op.output_shape[-1],
                k=op.k,
                zero_literal=zero_literal,
                one_literal=f"(({c_type})1)",
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, TriluOp):
            param_specs = [("input0", op.input0), ("output", op.output)]
            if op.k_input is not None:
                param_specs.append(("k_input", op.k_input))
            params = self._shared_param_map(param_specs)
            output_dim_names = _dim_names_for(op.output)
            shape = CEmitter._shape_dim_exprs(op.output_shape, output_dim_names)
            output_suffix = self._param_array_suffix(op.output_shape, output_dim_names)
            input_suffix = self._param_array_suffix(
                op.input_shape, _dim_names_for(op.input0)
            )
            k_suffix = ""
            if op.k_input is not None and op.k_input_shape is not None:
                k_suffix = self._param_array_suffix(
                    op.k_input_shape, _dim_names_for(op.k_input)
                )
            batch_dims = op.output_shape[:-2]
            batch_size = CEmitter._element_count(batch_dims or (1,))
            param_decls = [
                (params["input0"], c_type, input_suffix, True),
                (params["output"], c_type, output_suffix, False),
            ]
            if op.k_input is not None and op.k_input_dtype is not None:
                param_decls.append(
                    (
                        params["k_input"],
                        op.k_input_dtype.c_type,
                        k_suffix,
                        True,
                    )
                )
            rendered = trilu_template.render(
                model_name=model.name,
                op_name=op_name,
                input0=params["input0"],
                output=params["output"],
                k_input=params.get("k_input"),
                params=self._build_param_decls(param_decls),
                c_type=c_type,
                k_c_type=(
                    op.k_input_dtype.c_type
                    if op.k_input_dtype is not None
                    else ScalarType.I64.c_type
                ),
                input_suffix=input_suffix,
                output_suffix=output_suffix,
                shape=shape,
                batch_size=batch_size,
                rows=op.output_shape[-2],
                cols=op.output_shape[-1],
                k_value=op.k_value,
                upper=op.upper,
                zero_literal=zero_literal,
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, TileOp):
            params = self._shared_param_map(
                [("input0", op.input0), ("output", op.output)]
            )
            output_dim_names = _dim_names_for(op.output)
            output_shape = CEmitter._shape_dim_exprs(
                op.output_shape, output_dim_names
            )
            loop_vars = CEmitter._loop_vars(op.output_shape)
            input_suffix = self._param_array_suffix(
                op.input_shape, _dim_names_for(op.input0)
            )
            output_suffix = self._param_array_suffix(
                op.output_shape, output_dim_names
            )
            param_decls = self._build_param_decls(
                [
                    (params["input0"], c_type, input_suffix, True),
                    (params["output"], c_type, output_suffix, False),
                ]
            )
            input_index_terms = [
                f"({var} % {dim}) * {stride}"
                for var, dim, stride in zip(
                    loop_vars, op.input_shape, op.input_strides
                )
            ]
            input_index_expr = " + ".join(input_index_terms) or "0"
            rendered = tile_template.render(
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
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, PadOp):
            input_dim_names = _dim_names_for(op.input0)
            output_dim_names = _dim_names_for(op.output)
            input_shape = CEmitter._shape_dim_exprs(
                op.input_shape, input_dim_names
            )
            output_shape = CEmitter._shape_dim_exprs(
                op.output_shape, output_dim_names
            )
            in_loop_vars = CEmitter._loop_vars(op.input_shape)
            out_loop_vars = CEmitter._loop_vars(op.output_shape)
            idx_vars = tuple(f"pad_idx{index}" for index in range(len(op.output_shape)))
            reflect_vars = tuple(
                f"pad_reflect{index}" for index in range(len(op.output_shape))
            )
            pads_c_type = None
            pads_suffix = None
            pads_values = op.pads_values
            if op.pads_input is not None:
                pads_c_type = op.pads_dtype.c_type if op.pads_dtype else "int64_t"
                pads_suffix = (
                    self._param_array_suffix(
                        op.pads_shape or (), _dim_names_for(op.pads_input)
                    )
                    if op.pads_shape is not None
                    else ""
                )
            elif pads_values is not None:
                pads_c_type = "int64_t"

            axes_c_type = None
            axes_suffix = None
            axes_length = None
            if op.axes_input is not None:
                axes_c_type = op.axes_dtype.c_type if op.axes_dtype else "int64_t"
                axes_suffix = (
                    self._param_array_suffix(
                        op.axes_shape or (), _dim_names_for(op.axes_input)
                    )
                    if op.axes_shape is not None
                    else ""
                )
                axes_length = op.axes_shape[0] if op.axes_shape is not None else 0
                pad_begin_exprs = tuple(
                    f"pad_begin[{index}]" for index in range(len(op.output_shape))
                )
            elif op.pads_input is not None:
                if op.pads_axis_map is not None:
                    pad_begin_exprs = tuple(
                        (
                            f"{op.pads_input}[{axis_index}]"
                            if axis_index is not None
                            else "0"
                        )
                        for axis_index in op.pads_axis_map
                    )
                else:
                    pad_begin_exprs = tuple(
                        f"{op.pads_input}[{index}]"
                        for index in range(len(op.output_shape))
                    )
            else:
                pad_begin_exprs = tuple(
                    str(value) for value in (op.pads_begin or ())
                )
            if op.value_input is not None:
                value_suffix = (
                    self._param_array_suffix(
                        op.value_shape or (), _dim_names_for(op.value_input)
                    )
                    if op.value_shape is not None
                    else ""
                )
                pad_value_expr = f"{op.value_input}[0]"
            else:
                value_suffix = None
                pad_value_expr = CEmitter._format_literal(op.dtype, op.value)
            rendered = pad_template.render(
                model_name=model.name,
                op_name=op_name,
                input0=op.input0,
                pads_input=op.pads_input,
                axes_input=op.axes_input,
                value_input=op.value_input,
                output=op.output,
                c_type=c_type,
                pads_c_type=pads_c_type,
                axes_c_type=axes_c_type,
                input_suffix=self._param_array_suffix(
                    op.input_shape, input_dim_names
                ),
                pads_suffix=pads_suffix,
                axes_suffix=axes_suffix,
                value_suffix=value_suffix,
                output_suffix=self._param_array_suffix(
                    op.output_shape, output_dim_names
                ),
                input_shape=input_shape,
                output_shape=output_shape,
                in_loop_vars=in_loop_vars,
                out_loop_vars=out_loop_vars,
                pad_begin_exprs=pad_begin_exprs,
                axes_length=axes_length,
                pads_values=pads_values,
                input_strides=op.input_strides,
                mode=op.mode,
                pad_value_expr=pad_value_expr,
                input0_flat="input_flat",
                base_index="pad_index",
                idx_vars=idx_vars,
                reflect_vars=reflect_vars,
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, DepthToSpaceOp):
            params = self._shared_param_map(
                [("input0", op.input0), ("output", op.output)]
            )
            output_suffix = self._param_array_suffix(op.output_shape)
            input_suffix = self._param_array_suffix(op.input_shape)
            param_decls = self._build_param_decls(
                [
                    (params["input0"], c_type, input_suffix, True),
                    (params["output"], c_type, output_suffix, False),
                ]
            )
            rendered = depth_to_space_template.render(
                model_name=model.name,
                op_name=op_name,
                input0=params["input0"],
                output=params["output"],
                params=param_decls,
                c_type=c_type,
                input_suffix=input_suffix,
                output_suffix=output_suffix,
                batch=op.input_shape[0],
                in_channels=op.input_shape[1],
                out_channels=op.output_shape[1],
                in_h=op.input_shape[2],
                in_w=op.input_shape[3],
                out_h=op.output_shape[2],
                out_w=op.output_shape[3],
                blocksize=op.blocksize,
                mode=op.mode,
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, SpaceToDepthOp):
            params = self._shared_param_map(
                [("input0", op.input0), ("output", op.output)]
            )
            output_suffix = self._param_array_suffix(op.output_shape)
            input_suffix = self._param_array_suffix(op.input_shape)
            param_decls = self._build_param_decls(
                [
                    (params["input0"], c_type, input_suffix, True),
                    (params["output"], c_type, output_suffix, False),
                ]
            )
            rendered = space_to_depth_template.render(
                model_name=model.name,
                op_name=op_name,
                input0=params["input0"],
                output=params["output"],
                params=param_decls,
                c_type=c_type,
                input_suffix=input_suffix,
                output_suffix=output_suffix,
                batch=op.input_shape[0],
                in_channels=op.input_shape[1],
                out_channels=op.output_shape[1],
                in_h=op.input_shape[2],
                in_w=op.input_shape[3],
                out_h=op.output_shape[2],
                out_w=op.output_shape[3],
                blocksize=op.blocksize,
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, SliceOp):
            name_params = self._shared_param_map(
                [
                    ("input0", op.input0),
                    ("starts_input", op.starts_input),
                    ("ends_input", op.ends_input),
                    ("axes_input", op.axes_input),
                    ("steps_input", op.steps_input),
                    ("output", op.output),
                ]
            )
            output_shape = CEmitter._codegen_shape(op.output_shape)
            loop_vars = CEmitter._loop_vars(output_shape)
            if op.starts is not None and op.steps is not None:
                input_indices: list[str] = []
                for start, step, loop_var in zip(op.starts, op.steps, loop_vars):
                    if step == 1:
                        if start == 0:
                            input_indices.append(loop_var)
                        else:
                            input_indices.append(f"{start} + {loop_var}")
                    else:
                        input_indices.append(f"{start} + {step} * {loop_var}")
                input_suffix = self._param_array_suffix(op.input_shape)
                output_suffix = self._param_array_suffix(op.output_shape)
                param_decls = self._build_param_decls(
                    [
                        (name_params["input0"], c_type, input_suffix, True),
                        (name_params["output"], c_type, output_suffix, False),
                    ]
                )
                rendered = slice_template.render(
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
                ).rstrip()
                return with_node_comment(rendered)
            input_suffix = self._param_array_suffix(op.input_shape)
            output_suffix = self._param_array_suffix(op.output_shape)
            params = self._build_param_decls(
                [
                    (name_params["input0"], c_type, input_suffix, True),
                    (
                        name_params["starts_input"],
                        op.starts_dtype.c_type if op.starts_dtype else "",
                        self._param_array_suffix(op.starts_shape)
                        if op.starts_shape
                        else "",
                        True,
                    )
                    if op.starts_input and op.starts_shape and op.starts_dtype
                    else (None, "", "", True),
                    (
                        name_params["ends_input"],
                        op.ends_dtype.c_type if op.ends_dtype else "",
                        self._param_array_suffix(op.ends_shape)
                        if op.ends_shape
                        else "",
                        True,
                    )
                    if op.ends_input and op.ends_shape and op.ends_dtype
                    else (None, "", "", True),
                    (
                        name_params["axes_input"],
                        op.axes_dtype.c_type if op.axes_dtype else "",
                        self._param_array_suffix(op.axes_shape)
                        if op.axes_shape
                        else "",
                        True,
                    )
                    if op.axes_input and op.axes_shape and op.axes_dtype
                    else (None, "", "", True),
                    (
                        name_params["steps_input"],
                        op.steps_dtype.c_type if op.steps_dtype else "",
                        self._param_array_suffix(op.steps_shape)
                        if op.steps_shape
                        else "",
                        True,
                    )
                    if op.steps_input and op.steps_shape and op.steps_dtype
                    else (None, "", "", True),
                    (name_params["output"], c_type, output_suffix, False),
                ]
            )
            input_dims = CEmitter._codegen_shape(op.input_shape)
            rendered = slice_dynamic_template.render(
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
                input_rank=len(op.input_shape),
                starts_len=op.starts_shape[0] if op.starts_shape else 0,
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, ResizeOp):
            name_params = self._shared_param_map(
                [
                    ("input0", op.input0),
                    ("roi_input", op.roi_input),
                    ("scales_input", op.scales_input),
                    ("sizes_input", op.sizes_input),
                    ("output", op.output),
                ]
            )
            input_suffix = self._param_array_suffix(op.input_shape)
            output_suffix = self._param_array_suffix(op.output_shape)
            roi_suffix = None
            scales_suffix = None
            sizes_suffix = None
            roi_c_type = None
            scales_c_type = None
            sizes_c_type = None
            if op.roi_input and op.roi_shape and op.roi_dtype:
                roi_suffix = self._param_array_suffix(op.roi_shape)
                roi_c_type = op.roi_dtype.c_type
            if op.scales_input and op.scales_shape and op.scales_dtype:
                scales_suffix = self._param_array_suffix(op.scales_shape)
                scales_c_type = op.scales_dtype.c_type
            if op.sizes_input and op.sizes_shape and op.sizes_dtype:
                sizes_suffix = self._param_array_suffix(op.sizes_shape)
                sizes_c_type = op.sizes_dtype.c_type
            params = self._build_param_decls(
                [
                    (name_params["input0"], c_type, input_suffix, True),
                    (
                        name_params["roi_input"],
                        roi_c_type or "",
                        roi_suffix or "",
                        True,
                    )
                    if roi_c_type
                    else (None, "", "", True),
                    (
                        name_params["scales_input"],
                        scales_c_type or "",
                        scales_suffix or "",
                        True,
                    )
                    if scales_c_type
                    else (None, "", "", True),
                    (
                        name_params["sizes_input"],
                        sizes_c_type or "",
                        sizes_suffix or "",
                        True,
                    )
                    if sizes_c_type
                    else (None, "", "", True),
                    (name_params["output"], c_type, output_suffix, False),
                ]
            )
            scales_axis_map = None
            if op.scales_input:
                scales_axis_map = (
                    tuple(range(len(op.scales_axes)))
                    if op.scales_axes
                    else op.axes
                )
            sizes_axis_map = None
            if op.sizes_input:
                sizes_axis_map = (
                    tuple(range(len(op.sizes_axes)))
                    if op.sizes_axes
                    else op.axes
                )
            roi_axis_map = None
            if op.roi_input:
                roi_axis_map = (
                    tuple(range(len(op.roi_axes)))
                    if op.roi_axes
                    else op.axes
                )
            rendered = resize_template.render(
                model_name=model.name,
                op_name=op_name,
                params=params,
                input0=name_params["input0"],
                output=name_params["output"],
                c_type=c_type,
                input_suffix=input_suffix,
                output_suffix=output_suffix,
                input_shape=op.input_shape,
                output_shape=op.output_shape,
                rank=len(op.input_shape),
                loop_vars=CEmitter._loop_vars(op.output_shape),
                scales=op.scales,
                scales_input=name_params["scales_input"],
                sizes_input=name_params["sizes_input"],
                roi_input=name_params["roi_input"],
                roi_suffix=roi_suffix,
                scales_suffix=scales_suffix,
                sizes_suffix=sizes_suffix,
                roi_c_type=roi_c_type,
                scales_c_type=scales_c_type,
                sizes_c_type=sizes_c_type,
                axes=op.axes,
                scales_axes=op.scales_axes,
                sizes_axes=op.sizes_axes,
                roi_axes=op.roi_axes,
                scales_axis_map=scales_axis_map,
                sizes_axis_map=sizes_axis_map,
                roi_axis_map=roi_axis_map,
                mode=op.mode,
                coordinate_transformation_mode=op.coordinate_transformation_mode,
                nearest_mode=op.nearest_mode,
                cubic_coeff_a=CEmitter._format_double(op.cubic_coeff_a),
                exclude_outside=op.exclude_outside,
                extrapolation_value=CEmitter._format_double(
                    op.extrapolation_value
                ),
                antialias=op.antialias,
                keep_aspect_ratio_policy=op.keep_aspect_ratio_policy,
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, GridSampleOp):
            input_suffix = self._param_array_suffix(op.input_shape)
            grid_suffix = self._param_array_suffix(op.grid_shape)
            output_suffix = self._param_array_suffix(op.output_shape)
            params = [
                f"const {c_type} {op.input0}{input_suffix}",
                f"const {op.grid_dtype.c_type} {op.grid}{grid_suffix}",
                f"{c_type} {op.output}{output_suffix}",
            ]
            output_loop_vars = CEmitter._loop_vars(op.output_shape)
            rendered = grid_sample_template.render(
                model_name=model.name,
                op_name=op_name,
                params=params,
                input0=op.input0,
                grid=op.grid,
                output=op.output,
                c_type=c_type,
                grid_c_type=op.grid_dtype.c_type,
                input_suffix=input_suffix,
                grid_suffix=grid_suffix,
                output_suffix=output_suffix,
                input_shape=op.input_shape,
                grid_shape=op.grid_shape,
                output_shape=op.output_shape,
                spatial_rank=op.spatial_rank,
                input_spatial=op.input_spatial,
                output_spatial=op.output_spatial,
                output_loop_vars=output_loop_vars,
                mode=op.mode,
                padding_mode=op.padding_mode,
                align_corners=op.align_corners,
                linear_offsets=tuple(
                    itertools.product((0, 1), repeat=op.spatial_rank)
                ),
                cubic_offsets=tuple(
                    itertools.product(range(4), repeat=op.spatial_rank)
                ),
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, ReduceOp) and op.axes_input is None:
            params = self._shared_param_map(
                [("input0", op.input0), ("output", op.output)]
            )
            output_shape = CEmitter._codegen_shape(op.output_shape)
            output_loop_vars = CEmitter._loop_vars(output_shape)
            if not op.input_shape:
                reduce_loop_vars = ("r0",)
                reduce_dims = (1,)
            else:
                reduce_loop_vars = tuple(
                    f"r{idx}" for idx in range(len(op.axes))
                )
                reduce_dims = tuple(op.input_shape[axis] for axis in op.axes)
            if not op.input_shape:
                input_indices = [reduce_loop_vars[0]]
            elif op.keepdims:
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
            value_expr = f"{params['input0']}{input_index_expr}"
            update_expr = None
            init_literal = None
            final_expr = "acc"
            use_kahan = False
            kahan_value_expr = None
            fabs_fn = CEmitter._math_fn(op.dtype, "fabsf", "fabs")
            exp_fn = CEmitter._math_fn(op.dtype, "expf", "exp")
            log_fn = CEmitter._math_fn(op.dtype, "logf", "log")
            sqrt_fn = CEmitter._math_fn(op.dtype, "sqrtf", "sqrt")
            if op.reduce_kind == "sum":
                init_literal = zero_literal
                update_expr = f"acc += {value_expr};"
            elif op.reduce_kind == "mean":
                count_literal = CEmitter._format_literal(
                    op.dtype, op.reduce_count
                )
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
                update_expr = f"acc += {fabs_fn}({value_expr});"
            elif op.reduce_kind == "l2":
                init_literal = zero_literal
                update_expr = f"acc += {value_expr} * {value_expr};"
                final_expr = f"{sqrt_fn}(acc)"
            elif op.reduce_kind == "logsum":
                init_literal = zero_literal
                update_expr = f"acc += {value_expr};"
                final_expr = f"{log_fn}(acc)"
            elif op.reduce_kind == "logsumexp":
                init_literal = zero_literal
                update_expr = f"acc += {exp_fn}({value_expr});"
                final_expr = f"{log_fn}(acc)"
            elif op.reduce_kind == "sumsquare":
                init_literal = zero_literal
                update_expr = f"acc += {value_expr} * {value_expr};"
            else:
                raise CodegenError(
                    f"Unsupported reduce kind {op.reduce_kind}"
                )
            if op.dtype in {ScalarType.F16, ScalarType.F32} and op.reduce_kind in {
                "sum",
                "mean",
                "logsum",
                "logsumexp",
                "l1",
                "l2",
                "sumsquare",
            }:
                use_kahan = True
                if op.reduce_kind == "logsumexp":
                    kahan_value_expr = f"{exp_fn}({value_expr})"
                elif op.reduce_kind == "l1":
                    kahan_value_expr = f"{fabs_fn}({value_expr})"
                elif op.reduce_kind in {"l2", "sumsquare"}:
                    kahan_value_expr = f"{value_expr} * {value_expr}"
                else:
                    kahan_value_expr = value_expr
            input_suffix = self._param_array_suffix(op.input_shape)
            output_suffix = self._param_array_suffix(op.output_shape)
            param_decls = self._build_param_decls(
                [
                    (params["input0"], c_type, input_suffix, True),
                    (params["output"], c_type, output_suffix, False),
                ]
            )
            rendered = reduce_template.render(
                model_name=model.name,
                op_name=op_name,
                input0=params["input0"],
                output=params["output"],
                params=param_decls,
                c_type=c_type,
                input_suffix=input_suffix,
                output_suffix=output_suffix,
                output_shape=output_shape,
                output_loop_vars=output_loop_vars,
                reduce_loop_vars=reduce_loop_vars,
                reduce_dims=reduce_dims,
                output_index_expr=output_index_expr,
                init_literal=init_literal,
                zero_literal=zero_literal,
                update_expr=update_expr,
                final_expr=final_expr,
                use_kahan=use_kahan,
                kahan_value_expr=kahan_value_expr,
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, ArgReduceOp):
            params = self._shared_param_map(
                [("input0", op.input0), ("output", op.output)]
            )
            output_shape = CEmitter._codegen_shape(op.output_shape)
            output_loop_vars = CEmitter._loop_vars(output_shape)
            reduce_var = "r0"
            reduce_dim = op.input_shape[op.axis]
            if op.keepdims:
                input_indices = [
                    reduce_var if axis == op.axis else output_loop_vars[axis]
                    for axis in range(len(op.input_shape))
                ]
            else:
                kept_axes = [
                    axis
                    for axis in range(len(op.input_shape))
                    if axis != op.axis
                ]
                input_indices = [
                    reduce_var
                    if axis == op.axis
                    else output_loop_vars[kept_axes.index(axis)]
                    for axis in range(len(op.input_shape))
                ]
            init_indices = [
                "0" if axis == op.axis else input_indices[axis]
                for axis in range(len(op.input_shape))
            ]
            input_index_expr = "".join(f"[{var}]" for var in input_indices)
            init_index_expr = "".join(f"[{var}]" for var in init_indices)
            output_index_expr = "".join(
                f"[{var}]" for var in output_loop_vars
            )
            if op.reduce_kind == "max":
                compare_op = ">=" if op.select_last_index else ">"
            elif op.reduce_kind == "min":
                compare_op = "<=" if op.select_last_index else "<"
            else:
                raise CodegenError(
                    f"Unsupported arg reduce kind {op.reduce_kind}"
                )
            input_suffix = self._param_array_suffix(op.input_shape)
            output_suffix = self._param_array_suffix(op.output_shape)
            param_decls = self._build_param_decls(
                [
                    (params["input0"], op.input_dtype.c_type, input_suffix, True),
                    (params["output"], op.output_dtype.c_type, output_suffix, False),
                ]
            )
            rendered = arg_reduce_template.render(
                model_name=model.name,
                op_name=op_name,
                input0=params["input0"],
                output=params["output"],
                params=param_decls,
                input_c_type=op.input_dtype.c_type,
                output_c_type=op.output_dtype.c_type,
                input_suffix=input_suffix,
                output_suffix=output_suffix,
                output_shape=output_shape,
                output_loop_vars=output_loop_vars,
                reduce_var=reduce_var,
                reduce_dim=reduce_dim,
                input_index_expr=input_index_expr,
                init_index_expr=init_index_expr,
                output_index_expr=output_index_expr,
                compare_op=compare_op,
                dim_args=dim_args,
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, TopKOp):
            params = self._shared_param_map(
                [
                    ("input0", op.input0),
                    ("output_values", op.output_values),
                    ("output_indices", op.output_indices),
                ]
            )
            output_shape = CEmitter._codegen_shape(op.output_shape)
            outer_shape = tuple(
                dim for axis, dim in enumerate(output_shape) if axis != op.axis
            )
            outer_loop_vars = CEmitter._loop_vars(outer_shape)
            reduce_var = "r0"
            k_var = "k0"
            input_indices: list[str] = []
            output_indices: list[str] = []
            outer_index = 0
            for axis in range(len(op.input_shape)):
                if axis == op.axis:
                    input_indices.append(reduce_var)
                    output_indices.append(k_var)
                else:
                    input_indices.append(outer_loop_vars[outer_index])
                    output_indices.append(outer_loop_vars[outer_index])
                    outer_index += 1
            input_index_expr = "".join(f"[{var}]" for var in input_indices)
            output_index_expr = "".join(f"[{var}]" for var in output_indices)
            compare_expr = (
                "(a > b) || ((a == b) && (ai < bi))"
                if op.largest
                else "(a < b) || ((a == b) && (ai < bi))"
            )
            input_suffix = self._param_array_suffix(op.input_shape)
            output_suffix = self._param_array_suffix(op.output_shape)
            param_decls = self._build_param_decls(
                [
                    (params["input0"], op.input_dtype.c_type, input_suffix, True),
                    (
                        params["output_values"],
                        op.output_values_dtype.c_type,
                        output_suffix,
                        False,
                    ),
                    (
                        params["output_indices"],
                        op.output_indices_dtype.c_type,
                        output_suffix,
                        False,
                    ),
                ]
            )
            rendered = topk_template.render(
                model_name=model.name,
                op_name=op_name,
                input0=params["input0"],
                output_values=params["output_values"],
                output_indices=params["output_indices"],
                params=param_decls,
                input_c_type=op.input_dtype.c_type,
                output_values_c_type=op.output_values_dtype.c_type,
                output_indices_c_type=op.output_indices_dtype.c_type,
                input_suffix=input_suffix,
                output_suffix=output_suffix,
                output_shape=output_shape,
                outer_shape=outer_shape,
                outer_loop_vars=outer_loop_vars,
                reduce_var=reduce_var,
                k_var=k_var,
                axis_dim=op.input_shape[op.axis],
                k=op.k,
                input_index_expr=input_index_expr,
                output_index_expr=output_index_expr,
                compare_expr=compare_expr,
                dim_args=dim_args,
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, ReduceOp):
            name_params = self._shared_param_map(
                [
                    ("input0", op.input0),
                    ("axes_input", op.axes_input),
                    ("output", op.output),
                ]
            )
            output_shape = CEmitter._codegen_shape(op.output_shape)
            output_loop_vars = CEmitter._loop_vars(output_shape)
            input_shape = CEmitter._codegen_shape(op.input_shape)
            input_loop_vars = CEmitter._loop_vars(input_shape)
            axes_shape = op.axes_input_shape or ()
            axes_count = 1
            for dim in axes_shape:
                if dim == 0:
                    axes_count = 0
                    break
                axes_count *= dim
            axes_c_type = (
                op.axes_input_dtype.c_type
                if op.axes_input_dtype
                else ScalarType.I64.c_type
            )
            input_indices = "".join(f"[{var}]" for var in input_loop_vars)
            output_indices = "".join(
                f"[out_indices[{idx}]]" for idx in range(len(output_shape))
            )
            output_loop_index_expr = "".join(
                f"[{var}]" for var in output_loop_vars
            )
            value_expr = f"{name_params['input0']}{input_indices}"
            update_expr = None
            init_literal = None
            post_expr = None
            fabs_fn = CEmitter._math_fn(op.dtype, "fabsf", "fabs")
            exp_fn = CEmitter._math_fn(op.dtype, "expf", "exp")
            log_fn = CEmitter._math_fn(op.dtype, "logf", "log")
            sqrt_fn = CEmitter._math_fn(op.dtype, "sqrtf", "sqrt")
            if op.reduce_kind == "sum":
                init_literal = zero_literal
                update_expr = f"*out_ptr += {value_expr};"
            elif op.reduce_kind == "mean":
                init_literal = zero_literal
                update_expr = f"*out_ptr += {value_expr};"
                post_expr = "*out_ptr = *out_ptr / reduce_count;"
            elif op.reduce_kind == "max":
                init_literal = min_literal
                update_expr = f"if ({value_expr} > *out_ptr) *out_ptr = {value_expr};"
            elif op.reduce_kind == "min":
                init_literal = max_literal
                update_expr = f"if ({value_expr} < *out_ptr) *out_ptr = {value_expr};"
            elif op.reduce_kind == "prod":
                init_literal = CEmitter._format_literal(op.dtype, 1)
                update_expr = f"*out_ptr *= {value_expr};"
            elif op.reduce_kind == "l1":
                init_literal = zero_literal
                update_expr = f"*out_ptr += {fabs_fn}({value_expr});"
            elif op.reduce_kind == "l2":
                init_literal = zero_literal
                update_expr = f"*out_ptr += {value_expr} * {value_expr};"
                post_expr = f"*out_ptr = {sqrt_fn}(*out_ptr);"
            elif op.reduce_kind == "logsum":
                init_literal = zero_literal
                update_expr = f"*out_ptr += {value_expr};"
                post_expr = f"*out_ptr = {log_fn}(*out_ptr);"
            elif op.reduce_kind == "logsumexp":
                init_literal = zero_literal
                update_expr = f"*out_ptr += {exp_fn}({value_expr});"
                post_expr = f"*out_ptr = {log_fn}(*out_ptr);"
            elif op.reduce_kind == "sumsquare":
                init_literal = zero_literal
                update_expr = f"*out_ptr += {value_expr} * {value_expr};"
            else:
                raise CodegenError(
                    f"Unsupported reduce kind {op.reduce_kind}"
                )
            input_suffix = self._param_array_suffix(op.input_shape)
            output_suffix = self._param_array_suffix(op.output_shape)
            axes_suffix = (
                self._param_array_suffix(op.axes_input_shape)
                if op.axes_input_shape
                else ""
            )
            params = self._build_param_decls(
                [
                    (name_params["input0"], c_type, input_suffix, True),
                    (
                        name_params["axes_input"],
                        axes_c_type,
                        axes_suffix,
                        True,
                    )
                    if name_params["axes_input"]
                    else (None, "", "", True),
                    (name_params["output"], c_type, output_suffix, False),
                ]
            )
            rendered = reduce_dynamic_template.render(
                model_name=model.name,
                op_name=op_name,
                params=params,
                input0=name_params["input0"],
                axes_input=name_params["axes_input"],
                output=name_params["output"],
                c_type=c_type,
                axes_c_type=axes_c_type,
                input_shape=input_shape,
                output_shape=output_shape,
                input_loop_vars=input_loop_vars,
                output_loop_vars=output_loop_vars,
                output_index_expr=output_indices,
                output_loop_index_expr=output_loop_index_expr,
                input_index_expr=input_indices,
                init_literal=init_literal,
                update_expr=update_expr,
                post_expr=post_expr,
                keepdims=op.keepdims,
                noop_with_empty_axes=op.noop_with_empty_axes,
                axes_count=axes_count,
                reduce_mask_vars=tuple(
                    f"reduce_mask_{idx}"
                    for idx in range(len(input_shape))
                ),
                output_rank=len(output_shape),
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, ConstantOfShapeOp):
            params = self._shared_param_map(
                [("input0", op.input0), ("output", op.output)]
            )
            shape = CEmitter._codegen_shape(op.shape)
            loop_vars = CEmitter._loop_vars(shape)
            array_suffix = self._param_array_suffix(shape)
            input_suffix = self._param_array_suffix(op.input_shape)
            param_decls = self._build_param_decls(
                [
                    (params["input0"], op.input_dtype.c_type, input_suffix, True),
                    (params["output"], c_type, array_suffix, False),
                ]
            )
            rendered = constant_of_shape_template.render(
                model_name=model.name,
                op_name=op_name,
                input0=params["input0"],
                output=params["output"],
                params=param_decls,
                input_c_type=op.input_dtype.c_type,
                c_type=c_type,
                input_suffix=input_suffix,
                array_suffix=array_suffix,
                shape=shape,
                loop_vars=loop_vars,
                value_literal=CEmitter._format_literal(op.dtype, op.value),
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, ShapeOp):
            params = self._shared_param_map(
                [("input0", op.input0), ("output", op.output)]
            )
            input_suffix = self._param_array_suffix(op.input_shape)
            output_suffix = self._param_array_suffix(op.output_shape)
            param_decls = self._build_param_decls(
                [
                    (params["input0"], op.input_dtype.c_type, input_suffix, True),
                    (params["output"], c_type, output_suffix, False),
                ]
            )
            rendered = shape_template.render(
                model_name=model.name,
                op_name=op_name,
                input0=params["input0"],
                output=params["output"],
                params=param_decls,
                input_c_type=op.input_dtype.c_type,
                c_type=c_type,
                input_suffix=input_suffix,
                output_suffix=output_suffix,
                values=[
                    CEmitter._format_literal(op.dtype, value)
                    for value in op.values
                ],
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, SizeOp):
            params = self._shared_param_map(
                [("input0", op.input0), ("output", op.output)]
            )
            input_suffix = self._param_array_suffix(op.input_shape)
            output_suffix = self._param_array_suffix(op.output_shape)
            param_decls = self._build_param_decls(
                [
                    (params["input0"], op.input_dtype.c_type, input_suffix, True),
                    (params["output"], c_type, output_suffix, False),
                ]
            )
            rendered = size_template.render(
                model_name=model.name,
                op_name=op_name,
                input0=params["input0"],
                output=params["output"],
                params=param_decls,
                input_c_type=op.input_dtype.c_type,
                c_type=c_type,
                input_suffix=input_suffix,
                output_suffix=output_suffix,
                value=CEmitter._format_literal(op.dtype, op.value),
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, NonZeroOp):
            params = self._shared_param_map(
                [("input0", op.input0), ("output", op.output)]
            )
            input_dim_names = _dim_names_for(op.input0)
            output_dim_names = _dim_names_for(op.output)
            input_shape = CEmitter._shape_dim_exprs(
                op.input_shape, input_dim_names
            )
            loop_vars = CEmitter._loop_vars(op.input_shape)
            input_suffix = self._param_array_suffix(
                op.input_shape, input_dim_names
            )
            output_suffix = self._param_array_suffix(
                op.output_shape, output_dim_names
            )
            param_decls = self._build_param_decls(
                [
                    (params["input0"], op.input_dtype.c_type, input_suffix, True),
                    (params["output"], c_type, output_suffix, False),
                ]
            )
            input_expr = f"{params['input0']}" + "".join(
                f"[{var}]" for var in loop_vars
            )
            rendered = nonzero_template.render(
                model_name=model.name,
                op_name=op_name,
                input0=params["input0"],
                output=params["output"],
                params=param_decls,
                input_c_type=op.input_dtype.c_type,
                output_c_type=c_type,
                input_suffix=input_suffix,
                output_suffix=output_suffix,
                input_shape=input_shape,
                loop_vars=loop_vars,
                input_expr=input_expr,
                zero_literal=op.input_dtype.zero_literal,
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, NonMaxSuppressionOp):
            if scalar_registry is None:
                raise CodegenError(
                    "Scalar function registry is required for NonMaxSuppression."
                )
            min_fn = self._scalar_function_name(
                ScalarFunction.MINIMUM, op.boxes_dtype, scalar_registry
            )
            max_fn = self._scalar_function_name(
                ScalarFunction.MAXIMUM, op.boxes_dtype, scalar_registry
            )
            if min_fn is None or max_fn is None:
                raise CodegenError(
                    "Failed to resolve scalar min/max functions for NonMaxSuppression."
                )
            params = self._shared_param_map(
                [
                    ("boxes", op.boxes),
                    ("scores", op.scores),
                    ("max_output_boxes_per_class", op.max_output_boxes_per_class),
                    ("iou_threshold", op.iou_threshold),
                    ("score_threshold", op.score_threshold),
                    ("output", op.output),
                ]
            )
            boxes_suffix = self._param_array_suffix(
                op.boxes_shape, _dim_names_for(op.boxes)
            )
            scores_suffix = self._param_array_suffix(
                op.scores_shape, _dim_names_for(op.scores)
            )
            output_suffix = self._param_array_suffix(
                op.output_shape, _dim_names_for(op.output)
            )
            max_output_suffix = (
                self._param_array_suffix(
                    op.max_output_shape,
                    _dim_names_for(op.max_output_boxes_per_class or ""),
                )
                if op.max_output_shape is not None
                else ""
            )
            iou_threshold_suffix = (
                self._param_array_suffix(
                    op.iou_threshold_shape,
                    _dim_names_for(op.iou_threshold or ""),
                )
                if op.iou_threshold_shape is not None
                else ""
            )
            score_threshold_suffix = (
                self._param_array_suffix(
                    op.score_threshold_shape,
                    _dim_names_for(op.score_threshold or ""),
                )
                if op.score_threshold_shape is not None
                else ""
            )
            param_decls = self._build_param_decls(
                [
                    (params["boxes"], op.boxes_dtype.c_type, boxes_suffix, True),
                    (params["scores"], op.boxes_dtype.c_type, scores_suffix, True),
                    (
                        params["max_output_boxes_per_class"],
                        op.max_output_dtype.c_type if op.max_output_dtype else "",
                        max_output_suffix,
                        True,
                    )
                    if params["max_output_boxes_per_class"]
                    else (None, "", "", True),
                    (
                        params["iou_threshold"],
                        (
                            op.iou_threshold_dtype.c_type
                            if op.iou_threshold_dtype
                            else ""
                        ),
                        iou_threshold_suffix,
                        True,
                    )
                    if params["iou_threshold"]
                    else (None, "", "", True),
                    (
                        params["score_threshold"],
                        (
                            op.score_threshold_dtype.c_type
                            if op.score_threshold_dtype
                            else ""
                        ),
                        score_threshold_suffix,
                        True,
                    )
                    if params["score_threshold"]
                    else (None, "", "", True),
                    (params["output"], op.output_dtype.c_type, output_suffix, False),
                ]
            )
            rendered = nonmax_suppression_template.render(
                model_name=model.name,
                op_name=op_name,
                boxes=params["boxes"],
                scores=params["scores"],
                max_output_boxes_per_class=params["max_output_boxes_per_class"],
                iou_threshold=params["iou_threshold"],
                score_threshold=params["score_threshold"],
                output=params["output"],
                params=param_decls,
                input_c_type=op.boxes_dtype.c_type,
                output_c_type=op.output_dtype.c_type,
                compute_type=op.boxes_dtype.c_type,
                output_capacity=op.output_shape[0],
                num_batches=op.boxes_shape[0],
                num_boxes=op.boxes_shape[1],
                num_classes=op.scores_shape[1],
                center_point_box=op.center_point_box,
                min_fn=min_fn,
                max_fn=max_fn,
                iou_threshold_default=op.boxes_dtype.zero_literal,
                score_threshold_default=op.boxes_dtype.zero_literal,
                score_threshold_enabled=op.score_threshold is not None,
                dim_args=dim_args,
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, ExpandOp):
            params = self._shared_param_map(
                [("input0", op.input0), ("output", op.output)]
            )
            output_dim_names = _dim_names_for(op.output)
            output_shape = CEmitter._shape_dim_exprs(
                op.output_shape, output_dim_names
            )
            loop_vars = CEmitter._loop_vars(op.output_shape)
            input_suffix = self._param_array_suffix(
                op.input_shape, _dim_names_for(op.input0)
            )
            output_suffix = self._param_array_suffix(
                op.output_shape, output_dim_names
            )
            param_decls = self._build_param_decls(
                [
                    (params["input0"], c_type, input_suffix, True),
                    (params["output"], c_type, output_suffix, False),
                ]
            )
            input_index_terms = [
                f"{loop_var} * {stride}"
                for loop_var, input_dim, stride in zip(
                    loop_vars, op.input_shape_padded, op.input_strides
                )
                if input_dim != 1
            ]
            input_index_expr = (
                " + ".join(input_index_terms) if input_index_terms else "0"
            )
            rendered = expand_template.render(
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
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, CumSumOp):
            params = self._unique_param_map(
                [
                    ("input0", op.input0),
                    ("axis_input", op.axis_input),
                    ("output", op.output),
                ]
            )
            input_dim_names = _dim_names_for(op.input0)
            output_dim_names = _dim_names_for(op.output)
            axis_c_type = (
                op.axis_input_dtype.c_type if op.axis_input_dtype else "int64_t"
            )
            rendered = cumsum_template.render(
                model_name=model.name,
                op_name=op_name,
                input0=params["input0"],
                axis_input=params["axis_input"],
                axis_c_type=axis_c_type,
                axis_suffix=self._param_array_suffix(()),
                axis_literal=op.axis,
                output=params["output"],
                c_type=c_type,
                input_suffix=self._param_array_suffix(
                    op.input_shape, input_dim_names
                ),
                output_suffix=self._param_array_suffix(
                    op.input_shape, output_dim_names
                ),
                input_shape=CEmitter._shape_dim_exprs(
                    op.input_shape, input_dim_names
                ),
                rank=len(op.input_shape),
                exclusive=op.exclusive,
                reverse=op.reverse,
                dim_args=dim_args,
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, RangeOp):
            params = self._shared_param_map(
                [
                    ("start", op.start),
                    ("limit", op.limit),
                    ("delta", op.delta),
                    ("output", op.output),
                ]
            )
            scalar_suffix = self._param_array_suffix(())
            output_suffix = self._param_array_suffix(op.output_shape)
            param_decls = self._build_param_decls(
                [
                    (params["start"], c_type, scalar_suffix, True),
                    (params["limit"], c_type, scalar_suffix, True),
                    (params["delta"], c_type, scalar_suffix, True),
                    (params["output"], c_type, output_suffix, False),
                ]
            )
            rendered = range_template.render(
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
                length=op.length,
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, OneHotOp):
            params = self._shared_param_map(
                [
                    ("indices", op.indices),
                    ("depth", op.depth),
                    ("values", op.values),
                    ("output", op.output),
                ]
            )
            output_dim_names = _dim_names_for(op.output)
            indices_dim_names = _dim_names_for(op.indices)
            values_dim_names = _dim_names_for(op.values)
            output_shape = CEmitter._codegen_shape(op.output_shape)
            loop_vars = CEmitter._loop_vars(output_shape)
            indices_indices = tuple(
                var for idx, var in enumerate(loop_vars) if idx != op.axis
            )
            if not indices_indices:
                indices_indices = ("0",)
            output_suffix = self._param_array_suffix(
                op.output_shape, output_dim_names
            )
            indices_suffix = self._param_array_suffix(
                op.indices_shape, indices_dim_names
            )
            values_suffix = self._param_array_suffix(
                op.values_shape, values_dim_names
            )
            depth_suffix = self._param_array_suffix(())
            param_decls = self._build_param_decls(
                [
                    (
                        params["indices"],
                        op.indices_dtype.c_type,
                        indices_suffix,
                        True,
                    ),
                    (
                        params["depth"],
                        op.depth_dtype.c_type,
                        depth_suffix,
                        True,
                    ),
                    (params["values"], c_type, values_suffix, True),
                    (params["output"], c_type, output_suffix, False),
                ]
            )
            rendered = one_hot_template.render(
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
                axis_index=loop_vars[op.axis],
                depth_dim=op.depth_dim,
                indices_c_type=op.indices_dtype.c_type,
                c_type=c_type,
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, SplitOp):
            output_params = [
                (f"output_{index}", name)
                for index, name in enumerate(op.outputs)
            ]
            params = self._shared_param_map(
                [("input0", op.input0), *output_params]
            )
            output_names = tuple(
                params[f"output_{index}"]
                for index in range(len(op.outputs))
            )
            output_suffixes = tuple(
                self._param_array_suffix(
                    shape, _dim_names_for(name)
                )
                for name, shape in zip(output_names, op.output_shapes)
            )
            outer = 1
            for dim in op.input_shape[: op.axis]:
                outer *= dim
            inner = 1
            for dim in op.input_shape[op.axis + 1 :]:
                inner *= dim
            input_suffix = self._param_array_suffix(
                op.input_shape, _dim_names_for(op.input0)
            )
            param_decls = self._build_param_decls(
                [
                    (params["input0"], c_type, input_suffix, True),
                    *(
                        (name, c_type, suffix, False)
                        for name, suffix in zip(output_names, output_suffixes)
                    ),
                ]
            )
            rendered = split_template.render(
                model_name=model.name,
                op_name=op_name,
                input0=params["input0"],
                outputs=output_names,
                params=param_decls,
                output_suffixes=output_suffixes,
                c_type=c_type,
                input_suffix=input_suffix,
                axis_sizes=op.split_sizes,
                axis_total=op.input_shape[op.axis],
                outer=outer,
                inner=inner,
                output_count=len(output_names),
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, CastOp):
            params = self._shared_param_map(
                [("input0", op.input0), ("output", op.output)]
            )
            output_dim_names = _dim_names_for(op.output)
            shape = CEmitter._shape_dim_exprs(op.shape, output_dim_names)
            loop_vars = CEmitter._loop_vars(op.shape)
            array_suffix = self._param_array_suffix(op.shape, output_dim_names)
            param_decls = self._build_param_decls(
                [
                    (params["input0"], op.input_dtype.c_type, array_suffix, True),
                    (params["output"], op.dtype.c_type, array_suffix, False),
                ]
            )
            rendered = cast_template.render(
                model_name=model.name,
                op_name=op_name,
                input0=params["input0"],
                output=params["output"],
                params=param_decls,
                input_c_type=op.input_dtype.c_type,
                output_c_type=op.dtype.c_type,
                array_suffix=array_suffix,
                shape=shape,
                loop_vars=loop_vars,
                dim_args=dim_args,
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, QuantizeLinearOp):
            if scalar_registry is None:
                raise CodegenError(
                    "Scalar function registry is required for QuantizeLinear."
                )
            params = self._shared_param_map(
                [
                    ("input0", op.input0),
                    ("scale", op.scale),
                    ("zero_point", op.zero_point),
                    ("output", op.output),
                ]
            )
            output_dim_names = _dim_names_for(op.output)
            shape = CEmitter._shape_dim_exprs(op.input_shape, output_dim_names)
            loop_vars = CEmitter._loop_vars(op.input_shape)
            input_suffix = self._param_array_suffix(
                op.input_shape, _dim_names_for(op.input0)
            )
            scale_shape = (
                ()
                if op.axis is None
                else (op.input_shape[op.axis],)
            )
            scale_suffix = self._param_array_suffix(
                scale_shape, _dim_names_for(op.scale)
            )
            zero_point_suffix = self._param_array_suffix(
                scale_shape, _dim_names_for(op.zero_point or "")
            )
            param_decls = self._build_param_decls(
                [
                    (params["input0"], op.input_dtype.c_type, input_suffix, True),
                    (params["scale"], op.scale_dtype.c_type, scale_suffix, True),
                    (
                        params["zero_point"],
                        op.dtype.c_type,
                        zero_point_suffix,
                        True,
                    )
                    if params["zero_point"]
                    else (None, "", "", True),
                    (params["output"], op.dtype.c_type, input_suffix, False),
                ]
            )
            compute_type = "double" if op.input_dtype == ScalarType.F64 else "float"
            compute_dtype = (
                ScalarType.F64
                if compute_type == "double"
                else ScalarType.F32
            )
            max_fn = self._scalar_function_name(
                ScalarFunction.MAXIMUM, compute_dtype, scalar_registry
            )
            min_fn = self._scalar_function_name(
                ScalarFunction.MINIMUM, compute_dtype, scalar_registry
            )
            if max_fn is None or min_fn is None:
                raise CodegenError(
                    "Failed to resolve scalar min/max functions for QuantizeLinear."
                )
            round_fn = CEmitter._math_fn(
                op.input_dtype, "nearbyintf", "nearbyint"
            )
            scale_index = "0" if op.axis is None else loop_vars[op.axis]
            input_expr = f"{params['input0']}" + "".join(
                f"[{var}]" for var in loop_vars
            )
            output_expr = f"{params['output']}" + "".join(
                f"[{var}]" for var in loop_vars
            )
            scale_expr = f"{params['scale']}[{scale_index}]"
            if params["zero_point"]:
                zero_expr = f"{params['zero_point']}[{scale_index}]"
            else:
                zero_expr = "0"
            rendered = quantize_linear_template.render(
                model_name=model.name,
                op_name=op_name,
                input0=params["input0"],
                scale=params["scale"],
                zero_point=params["zero_point"],
                output=params["output"],
                params=param_decls,
                compute_type=compute_type,
                input_c_type=op.input_dtype.c_type,
                output_c_type=op.dtype.c_type,
                shape=shape,
                loop_vars=loop_vars,
                input_expr=input_expr,
                scale_expr=scale_expr,
                zero_expr=zero_expr,
                output_expr=output_expr,
                round_fn=round_fn,
                min_literal=op.dtype.min_literal,
                max_literal=op.dtype.max_literal,
                min_fn=min_fn,
                max_fn=max_fn,
                dim_args=dim_args,
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, QLinearMatMulOp):
            if scalar_registry is None:
                raise CodegenError(
                    "Scalar function registry is required for QLinearMatMul."
                )
            params = self._shared_param_map(
                [
                    ("input0", op.input0),
                    ("input0_scale", op.input0_scale),
                    ("input0_zero_point", op.input0_zero_point),
                    ("input1", op.input1),
                    ("input1_scale", op.input1_scale),
                    ("input1_zero_point", op.input1_zero_point),
                    ("output_scale", op.output_scale),
                    ("output_zero_point", op.output_zero_point),
                    ("output", op.output),
                ]
            )
            output_shape = CEmitter._codegen_shape(op.output_shape)
            output_loop_vars = CEmitter._loop_vars(output_shape)
            output_index_expr = f"{params['output']}" + "".join(
                f"[{var}]" for var in output_loop_vars
            )
            batch_rank = len(op.batch_shape)
            batch_vars = output_loop_vars[:batch_rank]
            if op.left_vector and op.right_vector:
                row_var = None
                col_var = None
            elif op.left_vector:
                row_var = None
                col_var = output_loop_vars[-1]
            elif op.right_vector:
                row_var = output_loop_vars[-1]
                col_var = None
            else:
                row_var = output_loop_vars[-2]
                col_var = output_loop_vars[-1]
            input0_index_expr, input1_index_expr = CEmitter._matmul_index_exprs(
                op,
                batch_vars,
                row_var,
                col_var,
                batch_rank,
                input0=params["input0"],
                input1=params["input1"],
            )
            input0_suffix = self._param_array_suffix(op.input0_shape)
            input1_suffix = self._param_array_suffix(op.input1_shape)
            input0_scale_suffix = self._param_array_suffix(
                op.input0_scale_shape
            )
            input1_scale_suffix = self._param_array_suffix(
                op.input1_scale_shape
            )
            output_scale_suffix = self._param_array_suffix(
                op.output_scale_shape
            )
            input0_zero_suffix = self._param_array_suffix(op.input0_zero_shape)
            input1_zero_suffix = self._param_array_suffix(op.input1_zero_shape)
            output_zero_suffix = self._param_array_suffix(op.output_zero_shape)
            output_suffix = self._param_array_suffix(op.output_shape)
            param_decls = self._build_param_decls(
                [
                    (
                        params["input0"],
                        op.input0_dtype.c_type,
                        input0_suffix,
                        True,
                    ),
                    (
                        params["input0_scale"],
                        op.input0_scale_dtype.c_type,
                        input0_scale_suffix,
                        True,
                    ),
                    (
                        params["input0_zero_point"],
                        op.input0_dtype.c_type,
                        input0_zero_suffix,
                        True,
                    ),
                    (
                        params["input1"],
                        op.input1_dtype.c_type,
                        input1_suffix,
                        True,
                    ),
                    (
                        params["input1_scale"],
                        op.input1_scale_dtype.c_type,
                        input1_scale_suffix,
                        True,
                    ),
                    (
                        params["input1_zero_point"],
                        op.input1_dtype.c_type,
                        input1_zero_suffix,
                        True,
                    ),
                    (
                        params["output_scale"],
                        op.output_scale_dtype.c_type,
                        output_scale_suffix,
                        True,
                    ),
                    (
                        params["output_zero_point"],
                        op.dtype.c_type,
                        output_zero_suffix,
                        True,
                    ),
                    (
                        params["output"],
                        op.dtype.c_type,
                        output_suffix,
                        False,
                    ),
                ]
            )
            compute_dtype = (
                ScalarType.F64
                if ScalarType.F64
                in {
                    op.input0_scale_dtype,
                    op.input1_scale_dtype,
                    op.output_scale_dtype,
                }
                else ScalarType.F32
            )
            compute_type = (
                "double" if compute_dtype == ScalarType.F64 else "float"
            )
            max_fn = self._scalar_function_name(
                ScalarFunction.MAXIMUM, compute_dtype, scalar_registry
            )
            min_fn = self._scalar_function_name(
                ScalarFunction.MINIMUM, compute_dtype, scalar_registry
            )
            if max_fn is None or min_fn is None:
                raise CodegenError(
                    "Failed to resolve scalar min/max functions for QLinearMatMul."
                )
            round_fn = CEmitter._math_fn(
                compute_dtype, "nearbyintf", "nearbyint"
            )
            scale_index = "0"
            rendered = qlinear_matmul_template.render(
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
                output_c_type=op.dtype.c_type,
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
                k=op.k,
                round_fn=round_fn,
                min_literal=op.dtype.min_literal,
                max_literal=op.dtype.max_literal,
                min_fn=min_fn,
                max_fn=max_fn,
                dim_args=dim_args,
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, ClipOp):
            if scalar_registry is None:
                raise CodegenError(
                    "Scalar function registry is required for Clip rendering."
                )
            min_fn = self._scalar_function_name(
                ScalarFunction.MINIMUM, op.dtype, scalar_registry
            )
            max_fn = self._scalar_function_name(
                ScalarFunction.MAXIMUM, op.dtype, scalar_registry
            )
            if min_fn is None or max_fn is None:
                raise CodegenError(
                    "Failed to resolve scalar min/max functions for Clip."
                )
            params = self._shared_param_map(
                [
                    ("input0", op.input0),
                    ("input_min", op.input_min),
                    ("input_max", op.input_max),
                    ("output", op.output),
                ]
            )
            output_dim_names = _dim_names_for(op.output)
            output_shape = CEmitter._shape_dim_exprs(
                op.output_shape, output_dim_names
            )
            loop_vars = CEmitter._loop_vars(op.output_shape)
            input_expr = CEmitter._broadcast_index_expr(
                params["input0"],
                op.input_shape,
                op.output_shape,
                loop_vars,
            )
            min_expr = (
                CEmitter._broadcast_index_expr(
                    params["input_min"],
                    op.min_shape,
                    op.output_shape,
                    loop_vars,
                )
                if op.input_min is not None
                else op.dtype.min_literal
            )
            max_expr = (
                CEmitter._broadcast_index_expr(
                    params["input_max"],
                    op.max_shape,
                    op.output_shape,
                    loop_vars,
                )
                if op.input_max is not None
                else op.dtype.max_literal
            )
            input_suffix = self._param_array_suffix(
                op.input_shape, _dim_names_for(op.input0)
            )
            min_suffix = (
                self._param_array_suffix(
                    op.min_shape, _dim_names_for(op.input_min)
                )
                if op.min_shape is not None
                else ""
            )
            max_suffix = (
                self._param_array_suffix(
                    op.max_shape, _dim_names_for(op.input_max)
                )
                if op.max_shape is not None
                else ""
            )
            output_suffix = self._param_array_suffix(
                op.output_shape, output_dim_names
            )
            param_decls = self._build_param_decls(
                [
                    (params["input0"], op.dtype.c_type, input_suffix, True),
                    (
                        params["input_min"],
                        op.dtype.c_type,
                        min_suffix,
                        True,
                    )
                    if params["input_min"]
                    else (None, "", "", True),
                    (
                        params["input_max"],
                        op.dtype.c_type,
                        max_suffix,
                        True,
                    )
                    if params["input_max"]
                    else (None, "", "", True),
                    (params["output"], op.dtype.c_type, output_suffix, False),
                ]
            )
            rendered = clip_template.render(
                model_name=model.name,
                op_name=op_name,
                input0=params["input0"],
                input_min=params["input_min"],
                input_max=params["input_max"],
                output=params["output"],
                params=param_decls,
                input_c_type=op.dtype.c_type,
                output_c_type=op.dtype.c_type,
                input_suffix=input_suffix,
                min_suffix=min_suffix,
                max_suffix=max_suffix,
                output_suffix=output_suffix,
                shape=output_shape,
                loop_vars=loop_vars,
                input_expr=input_expr,
                min_expr=min_expr,
                max_expr=max_expr,
                min_fn=min_fn,
                max_fn=max_fn,
                dim_args=dim_args,
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, UnaryOp):
            params = self._shared_param_map(
                [("input0", op.input0), ("output", op.output)]
            )
            scalar_operator = None
            if scalar_registry is not None:
                scalar_operator = self._scalar_function_name(
                    op.function, op.dtype, scalar_registry, params=op.params
                )
            output_dim_names = _dim_names_for(op.output)
            shape = CEmitter._shape_dim_exprs(op.shape, output_dim_names)
            loop_vars = CEmitter._loop_vars(op.shape)
            array_suffix = self._param_array_suffix(op.shape, output_dim_names)
            param_decls = self._build_param_decls(
                [
                    (params["input0"], op.input_dtype.c_type, array_suffix, True),
                    (params["output"], op.dtype.c_type, array_suffix, False),
                ]
            )
            operator_symbol = unary_op_symbol(op.function, dtype=op.dtype)
            if op.function == ScalarFunction.ISINF and len(op.params) == 2:
                detect_negative, detect_positive = op.params
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
            elif op.function in {ScalarFunction.ISINF, ScalarFunction.ISNAN}:
                operator_symbol = (
                    "isinf" if op.function == ScalarFunction.ISINF else "isnan"
                )
            if operator_symbol is None and scalar_operator is None:
                raise CodegenError(
                    f"Unsupported unary operator for rendering: {op.function.value}"
                )
            common = {
                "model_name": model.name,
                "op_name": op_name,
                "element_count": CEmitter._element_count_expr(shape),
                "array_suffix": array_suffix,
                "shape": shape,
                "loop_vars": loop_vars,
                "input_c_type": op.input_dtype.c_type,
                "output_c_type": op.dtype.c_type,
                "zero_literal": zero_literal,
                "dim_args": dim_args,
                "params": param_decls,
            }
            rendered = unary_template.render(
                **common,
                input0=params["input0"],
                output=params["output"],
                operator=scalar_operator or operator_symbol,
            ).rstrip()
            return with_node_comment(rendered)
        raise CodegenError(f"Unsupported op for rendering: {type(op).__name__}")

    @staticmethod
    def _op_output(
        op: BinaryOp
        | MultiInputBinaryOp
        | WhereOp
        | UnaryOp
        | ClipOp
        | CastOp
        | QuantizeLinearOp
        | MatMulOp
        | EinsumOp
        | GemmOp
        | AttentionOp
        | ConvOp
        | ConvTransposeOp
        | AveragePoolOp
        | LpPoolOp
        | BatchNormOp
        | LpNormalizationOp
        | InstanceNormalizationOp
        | GroupNormalizationOp
        | LayerNormalizationOp
        | MeanVarianceNormalizationOp
        | RMSNormalizationOp
        | LrnOp
        | LstmOp
        | SoftmaxOp
        | LogSoftmaxOp
        | HardmaxOp
        | NegativeLogLikelihoodLossOp
        | SoftmaxCrossEntropyLossOp
        | MaxPoolOp
        | ConcatOp
        | GatherElementsOp
        | GatherOp
        | GatherNDOp
        | ScatterNDOp
        | TensorScatterOp
        | TransposeOp
        | ReshapeOp
        | IdentityOp
        | EyeLikeOp
        | TriluOp
        | TileOp
        | PadOp
        | DepthToSpaceOp
        | SpaceToDepthOp
        | ResizeOp
        | GridSampleOp
        | ReduceOp
        | ArgReduceOp
        | TopKOp
        | ConstantOfShapeOp
        | ShapeOp
        | SizeOp
        | ExpandOp
        | CumSumOp
        | RangeOp
        | OneHotOp
        | SplitOp,
    ) -> str:
        if isinstance(op, SplitOp):
            return op.outputs[0]
        if isinstance(op, TopKOp):
            return op.output_values
        return op.output

    @staticmethod
    def _op_inputs(
        op: BinaryOp
        | MultiInputBinaryOp
        | WhereOp
        | UnaryOp
        | ClipOp
        | CastOp
        | QuantizeLinearOp
        | MatMulOp
        | EinsumOp
        | GemmOp
        | AttentionOp
        | ConvOp
        | ConvTransposeOp
        | AveragePoolOp
        | LpPoolOp
        | BatchNormOp
        | LpNormalizationOp
        | InstanceNormalizationOp
        | GroupNormalizationOp
        | LayerNormalizationOp
        | MeanVarianceNormalizationOp
        | RMSNormalizationOp
        | LrnOp
        | LstmOp
        | SoftmaxOp
        | LogSoftmaxOp
        | HardmaxOp
        | NegativeLogLikelihoodLossOp
        | SoftmaxCrossEntropyLossOp
        | MaxPoolOp
        | ConcatOp
        | GatherElementsOp
        | GatherOp
        | GatherNDOp
        | ScatterNDOp
        | TensorScatterOp
        | TransposeOp
        | ReshapeOp
        | IdentityOp
        | EyeLikeOp
        | TriluOp
        | TileOp
        | PadOp
        | DepthToSpaceOp
        | SpaceToDepthOp
        | ResizeOp
        | GridSampleOp
        | ReduceOp
        | ArgReduceOp
        | TopKOp
        | ConstantOfShapeOp
        | ShapeOp
        | SizeOp
        | ExpandOp
        | CumSumOp
        | RangeOp
        | OneHotOp
        | SplitOp,
    ) -> tuple[tuple[str, tuple[int, ...]], ...]:
        if isinstance(op, BinaryOp):
            return (
                (op.input0, op.input0_shape),
                (op.input1, op.input1_shape),
            )
        if isinstance(op, MultiInputBinaryOp):
            return tuple((name, op.shape) for name in op.inputs)
        if isinstance(op, EinsumOp):
            return tuple(
                (name, shape)
                for name, shape in zip(op.inputs, op.input_shapes)
            )
        if isinstance(op, UnaryOp):
            return ((op.input0, op.shape),)
        if isinstance(op, LpNormalizationOp):
            return ((op.input0, op.shape),)
        if isinstance(op, InstanceNormalizationOp):
            return (
                (op.input0, op.shape),
                (op.scale, (op.channels,)),
                (op.bias, (op.channels,)),
            )
        if isinstance(op, GroupNormalizationOp):
            return (
                (op.input0, op.shape),
                (op.scale, (op.channels,)),
                (op.bias, (op.channels,)),
            )
        if isinstance(op, LayerNormalizationOp):
            inputs = [(op.input0, op.shape), (op.scale, op.scale_shape)]
            if op.bias is not None and op.bias_shape is not None:
                inputs.append((op.bias, op.bias_shape))
            return tuple(inputs)
        if isinstance(op, MeanVarianceNormalizationOp):
            return ((op.input0, op.shape),)
        if isinstance(op, RMSNormalizationOp):
            return ((op.input0, op.shape), (op.scale, op.scale_shape))
        if isinstance(op, ClipOp):
            inputs = [(op.input0, op.input_shape)]
            if op.input_min is not None and op.min_shape is not None:
                inputs.append((op.input_min, op.min_shape))
            if op.input_max is not None and op.max_shape is not None:
                inputs.append((op.input_max, op.max_shape))
            return tuple(inputs)
        if isinstance(op, CastOp):
            return ((op.input0, op.shape),)
        if isinstance(op, NonZeroOp):
            return ((op.input0, op.input_shape),)
        if isinstance(op, NonMaxSuppressionOp):
            inputs = [
                (op.boxes, op.boxes_shape),
                (op.scores, op.scores_shape),
            ]
            if (
                op.max_output_boxes_per_class is not None
                and op.max_output_shape is not None
            ):
                inputs.append(
                    (op.max_output_boxes_per_class, op.max_output_shape)
                )
            if (
                op.iou_threshold is not None
                and op.iou_threshold_shape is not None
            ):
                inputs.append((op.iou_threshold, op.iou_threshold_shape))
            if (
                op.score_threshold is not None
                and op.score_threshold_shape is not None
            ):
                inputs.append(
                    (op.score_threshold, op.score_threshold_shape)
                )
            return tuple(inputs)
        if isinstance(op, QuantizeLinearOp):
            scale_shape = (
                ()
                if op.axis is None
                else (op.input_shape[op.axis],)
            )
            inputs = [(op.input0, op.input_shape), (op.scale, scale_shape)]
            if op.zero_point is not None:
                inputs.append((op.zero_point, scale_shape))
            return tuple(inputs)
        if isinstance(op, IdentityOp):
            return ((op.input0, op.shape),)
        if isinstance(op, EyeLikeOp):
            return ((op.input0, op.output_shape),)
        if isinstance(op, TriluOp):
            inputs = [(op.input0, op.input_shape)]
            if op.k_input is not None and op.k_input_shape is not None:
                inputs.append((op.k_input, op.k_input_shape))
            return tuple(inputs)
        if isinstance(op, GridSampleOp):
            return ((op.input0, op.input_shape), (op.grid, op.grid_shape))
        if isinstance(op, PadOp):
            inputs = [(op.input0, op.input_shape)]
            if op.pads_input is not None and op.pads_shape is not None:
                inputs.append((op.pads_input, op.pads_shape))
            if op.axes_input is not None and op.axes_shape is not None:
                inputs.append((op.axes_input, op.axes_shape))
            if op.value_input is not None and op.value_shape is not None:
                inputs.append((op.value_input, op.value_shape))
            return tuple(inputs)
        if isinstance(op, ScatterNDOp):
            return ((op.data, op.data_shape),)
        if isinstance(op, TensorScatterOp):
            inputs = [
                (op.past_cache, op.past_cache_shape),
                (op.update, op.update_shape),
            ]
            if op.write_indices is not None and op.write_indices_shape is not None:
                inputs.append((op.write_indices, op.write_indices_shape))
            return tuple(inputs)
        if isinstance(op, CumSumOp):
            return ((op.input0, op.input_shape),)
        if isinstance(op, RangeOp):
            return ((op.start, ()), (op.limit, ()), (op.delta, ()))
        if isinstance(op, OneHotOp):
            return (
                (op.indices, op.indices_shape),
                (op.depth, ()),
                (op.values, op.values_shape),
            )
        if isinstance(op, SplitOp):
            return ((op.input0, op.input_shape),)
        if isinstance(op, TopKOp):
            return ((op.input0, op.input_shape),)
        return ()

    def _propagate_tensor_dim_names(
        self,
        resolved_ops: Sequence[
            BinaryOp
            | MultiInputBinaryOp
            | WhereOp
            | UnaryOp
            | ClipOp
            | CastOp
            | QuantizeLinearOp
            | MatMulOp
            | EinsumOp
            | GemmOp
            | AttentionOp
            | ConvOp
            | ConvTransposeOp
            | AveragePoolOp
            | LpPoolOp
            | BatchNormOp
            | LpNormalizationOp
            | InstanceNormalizationOp
            | GroupNormalizationOp
            | LayerNormalizationOp
            | MeanVarianceNormalizationOp
            | RMSNormalizationOp
            | LrnOp
            | LstmOp
            | SoftmaxOp
            | LogSoftmaxOp
            | HardmaxOp
            | NegativeLogLikelihoodLossOp
            | SoftmaxCrossEntropyLossOp
            | MaxPoolOp
            | ConcatOp
            | GatherElementsOp
            | GatherOp
            | GatherNDOp
            | TransposeOp
            | ReshapeOp
            | IdentityOp
            | EyeLikeOp
            | TriluOp
            | TileOp
            | PadOp
            | DepthToSpaceOp
            | SpaceToDepthOp
            | ResizeOp
            | GridSampleOp
            | ReduceOp
            | ArgReduceOp
            | TopKOp
            | ConstantOfShapeOp
            | ShapeOp
            | SizeOp
            | NonZeroOp
            | NonMaxSuppressionOp
            | ExpandOp
            | RangeOp
            | OneHotOp
            | SplitOp
        ],
        tensor_dim_names: dict[str, dict[int, str]],
    ) -> None:
        for op in resolved_ops:
            for output_name, output_shape, _ in self._op_outputs(op):
                if output_name in tensor_dim_names:
                    continue
                for input_name, input_shape in self._op_inputs(op):
                    dim_names = tensor_dim_names.get(input_name)
                    if dim_names and input_shape == output_shape:
                        tensor_dim_names[output_name] = dict(dim_names)
                        break

    @staticmethod
    def _op_outputs(
        op: BinaryOp
        | MultiInputBinaryOp
        | WhereOp
        | UnaryOp
        | ClipOp
        | CastOp
        | QuantizeLinearOp
        | MatMulOp
        | EinsumOp
        | GemmOp
        | AttentionOp
        | ConvOp
        | ConvTransposeOp
        | AveragePoolOp
        | LpPoolOp
        | BatchNormOp
        | LpNormalizationOp
        | InstanceNormalizationOp
        | GroupNormalizationOp
        | LayerNormalizationOp
        | MeanVarianceNormalizationOp
        | RMSNormalizationOp
        | LrnOp
        | LstmOp
        | SoftmaxOp
        | LogSoftmaxOp
        | HardmaxOp
        | NegativeLogLikelihoodLossOp
        | SoftmaxCrossEntropyLossOp
        | MaxPoolOp
        | ConcatOp
        | GatherElementsOp
        | GatherOp
        | GatherNDOp
        | ScatterNDOp
        | TensorScatterOp
        | TransposeOp
        | ReshapeOp
        | IdentityOp
        | EyeLikeOp
        | TriluOp
        | TileOp
        | PadOp
        | DepthToSpaceOp
        | SpaceToDepthOp
        | ResizeOp
        | GridSampleOp
        | ReduceOp
        | ArgReduceOp
        | TopKOp
        | ConstantOfShapeOp
        | ShapeOp
        | SizeOp
        | NonZeroOp
        | NonMaxSuppressionOp
        | ExpandOp
        | RangeOp
        | OneHotOp
        | SplitOp,
    ) -> tuple[tuple[str, tuple[int, ...], str], ...]:
        if isinstance(op, AttentionOp):
            outputs: list[tuple[str, tuple[int, ...], str]] = [
                (op.output, CEmitter._op_output_shape(op), op.dtype)
            ]
            if op.output_present_key is not None:
                outputs.append(
                    (
                        op.output_present_key,
                        (op.batch, op.kv_heads, op.total_seq, op.qk_head_size),
                        op.dtype,
                    )
                )
            if op.output_present_value is not None:
                outputs.append(
                    (
                        op.output_present_value,
                        (op.batch, op.kv_heads, op.total_seq, op.v_head_size),
                        op.dtype,
                    )
                )
            if op.output_qk_matmul is not None:
                outputs.append(
                    (
                        op.output_qk_matmul,
                        (op.batch, op.q_heads, op.q_seq, op.total_seq),
                        op.dtype,
                    )
            )
            return tuple(outputs)
        if isinstance(op, LstmOp):
            outputs: list[tuple[str, tuple[int, ...], str]] = []
            if op.output_y is not None:
                if op.layout == 0:
                    y_shape = (
                        op.seq_length,
                        op.num_directions,
                        op.batch_size,
                        op.hidden_size,
                    )
                else:
                    y_shape = (
                        op.batch_size,
                        op.seq_length,
                        op.num_directions,
                        op.hidden_size,
                    )
                outputs.append((op.output_y, y_shape, op.dtype))
            if op.output_y_h is not None:
                outputs.append(
                    (
                        op.output_y_h,
                        (op.num_directions, op.batch_size, op.hidden_size),
                        op.dtype,
                    )
                )
            if op.output_y_c is not None:
                outputs.append(
                    (
                        op.output_y_c,
                        (op.num_directions, op.batch_size, op.hidden_size),
                        op.dtype,
                    )
                )
            return tuple(outputs)
        if isinstance(op, AdagradOp):
            outputs = [
                (name, shape, op.dtype)
                for name, shape in zip(op.outputs, op.output_shapes)
            ]
            outputs.extend(
                (name, shape, op.dtype)
                for name, shape in zip(
                    op.accumulator_outputs, op.output_shapes
                )
            )
            return tuple(outputs)
        if isinstance(op, SoftmaxCrossEntropyLossOp):
            outputs = [(op.output, op.output_shape, op.dtype)]
            if op.log_prob is not None and op.log_prob_shape is not None:
                outputs.append((op.log_prob, op.log_prob_shape, op.dtype))
            return tuple(outputs)
        if isinstance(op, LayerNormalizationOp):
            outputs: list[tuple[str, tuple[int, ...], str]] = [
                (op.output, op.shape, op.dtype)
            ]
            if op.mean_output is not None:
                mean_shape = op.shape[: op.axis] + (1,) * len(op.normalized_shape)
                outputs.append((op.mean_output, mean_shape, op.dtype))
            if op.invstd_output is not None:
                invstd_shape = op.shape[: op.axis] + (1,) * len(op.normalized_shape)
                outputs.append((op.invstd_output, invstd_shape, op.dtype))
            return tuple(outputs)
        if isinstance(op, MaxPoolOp):
            outputs = [(op.output, CEmitter._op_output_shape(op), op.dtype)]
            if op.indices is not None and op.indices_dtype is not None:
                outputs.append(
                    (op.indices, CEmitter._op_output_shape(op), op.indices_dtype)
                )
            return tuple(outputs)
        if isinstance(op, SplitOp):
            return tuple(
                (name, shape, op.dtype)
                for name, shape in zip(op.outputs, op.output_shapes)
            )
        if isinstance(op, ArgReduceOp):
            return ((op.output, CEmitter._op_output_shape(op), op.output_dtype),)
        if isinstance(op, TopKOp):
            return (
                (
                    op.output_values,
                    CEmitter._op_output_shape(op),
                    op.output_values_dtype,
                ),
                (
                    op.output_indices,
                    CEmitter._op_output_shape(op),
                    op.output_indices_dtype,
                ),
            )
        if isinstance(op, NonMaxSuppressionOp):
            return ((op.output, op.output_shape, op.output_dtype),)
        return ((op.output, CEmitter._op_output_shape(op), op.dtype),)

    @staticmethod
    def _op_output_shape(
        op: BinaryOp
        | MultiInputBinaryOp
        | WhereOp
        | UnaryOp
        | ClipOp
        | CastOp
        | QuantizeLinearOp
        | QLinearMatMulOp
        | MatMulOp
        | EinsumOp
        | GemmOp
        | AttentionOp
        | ConvOp
        | AveragePoolOp
        | BatchNormOp
        | LpNormalizationOp
        | InstanceNormalizationOp
        | GroupNormalizationOp
        | LayerNormalizationOp
        | MeanVarianceNormalizationOp
        | RMSNormalizationOp
        | LrnOp
        | LstmOp
        | SoftmaxOp
        | LogSoftmaxOp
        | HardmaxOp
        | NegativeLogLikelihoodLossOp
        | SoftmaxCrossEntropyLossOp
        | MaxPoolOp
        | ConcatOp
        | GatherElementsOp
        | GatherOp
        | GatherNDOp
        | TransposeOp
        | ReshapeOp
        | IdentityOp
        | EyeLikeOp
        | TriluOp
        | TileOp
        | SliceOp
        | ResizeOp
        | GridSampleOp
        | ReduceOp
        | ArgReduceOp
        | TopKOp
        | ConstantOfShapeOp
        | ShapeOp
        | SizeOp
        | NonZeroOp
        | NonMaxSuppressionOp
        | ExpandOp
        | CumSumOp
        | RangeOp
        | OneHotOp
        | SplitOp
        | PadOp,
    ) -> tuple[int, ...]:
        if isinstance(op, BinaryOp):
            return op.shape
        if isinstance(op, MultiInputBinaryOp):
            return op.shape
        if isinstance(op, WhereOp):
            return op.output_shape
        if isinstance(op, UnaryOp):
            return op.shape
        if isinstance(op, ClipOp):
            return op.output_shape
        if isinstance(op, QuantizeLinearOp):
            return op.input_shape
        if isinstance(op, CastOp):
            return op.shape
        if isinstance(op, QLinearMatMulOp):
            return op.output_shape
        if isinstance(op, MatMulOp):
            return op.output_shape
        if isinstance(op, EinsumOp):
            return op.output_shape
        if isinstance(op, GemmOp):
            return (op.m, op.n)
        if isinstance(op, ConvOp):
            return (op.batch, op.out_channels, *op.out_spatial)
        if isinstance(op, ConvTransposeOp):
            return (op.batch, op.out_channels, *op.out_spatial)
        if isinstance(op, AveragePoolOp):
            return (op.batch, op.channels, op.out_h, op.out_w)
        if isinstance(op, LpPoolOp):
            return (op.batch, op.channels, op.out_h, op.out_w)
        if isinstance(op, BatchNormOp):
            return op.shape
        if isinstance(
            op,
            (
                LpNormalizationOp,
                InstanceNormalizationOp,
                GroupNormalizationOp,
                LayerNormalizationOp,
                MeanVarianceNormalizationOp,
                RMSNormalizationOp,
            ),
        ):
            return op.shape
        if isinstance(op, LrnOp):
            return op.shape
        if isinstance(op, SoftmaxOp):
            return op.shape
        if isinstance(op, LogSoftmaxOp):
            return op.shape
        if isinstance(op, HardmaxOp):
            return op.shape
        if isinstance(op, NegativeLogLikelihoodLossOp):
            return op.output_shape
        if isinstance(op, SoftmaxCrossEntropyLossOp):
            return op.output_shape
        if isinstance(op, MaxPoolOp):
            return (op.batch, op.channels, *op.out_spatial)
        if isinstance(op, ConcatOp):
            return op.output_shape
        if isinstance(op, GatherElementsOp):
            return op.output_shape
        if isinstance(op, GatherOp):
            return op.output_shape
        if isinstance(op, GatherNDOp):
            return op.output_shape
        if isinstance(op, ScatterNDOp):
            return op.output_shape
        if isinstance(op, TensorScatterOp):
            return op.output_shape
        if isinstance(op, TransposeOp):
            return op.output_shape
        if isinstance(op, ReshapeOp):
            return op.output_shape
        if isinstance(op, IdentityOp):
            return op.shape
        if isinstance(op, EyeLikeOp):
            return op.output_shape
        if isinstance(op, TriluOp):
            return op.output_shape
        if isinstance(op, TileOp):
            return op.output_shape
        if isinstance(op, PadOp):
            return op.output_shape
        if isinstance(op, DepthToSpaceOp):
            return op.output_shape
        if isinstance(op, SpaceToDepthOp):
            return op.output_shape
        if isinstance(op, SliceOp):
            return op.output_shape
        if isinstance(op, ResizeOp):
            return op.output_shape
        if isinstance(op, GridSampleOp):
            return op.output_shape
        if isinstance(op, ReduceOp):
            return op.output_shape
        if isinstance(op, ArgReduceOp):
            return op.output_shape
        if isinstance(op, TopKOp):
            return op.output_shape
        if isinstance(op, ConstantOfShapeOp):
            return op.shape
        if isinstance(op, ShapeOp):
            return op.output_shape
        if isinstance(op, SizeOp):
            return op.output_shape
        if isinstance(op, NonZeroOp):
            return op.output_shape
        if isinstance(op, NonMaxSuppressionOp):
            return op.output_shape
        if isinstance(op, ExpandOp):
            return op.output_shape
        if isinstance(op, CumSumOp):
            return op.input_shape
        if isinstance(op, RangeOp):
            return op.output_shape
        if isinstance(op, OneHotOp):
            return op.output_shape
        if op.output_rank == 3:
            return (op.batch, op.q_seq, op.q_heads * op.v_head_size)
        return (op.batch, op.q_heads, op.q_seq, op.v_head_size)

    @staticmethod
    def _op_output_dtype(
        op: BinaryOp
        | MultiInputBinaryOp
        | WhereOp
        | UnaryOp
        | ClipOp
        | CastOp
        | QuantizeLinearOp
        | MatMulOp
        | EinsumOp
        | GemmOp
        | AttentionOp
        | ConvOp
        | ConvTransposeOp
        | AveragePoolOp
        | LpPoolOp
        | BatchNormOp
        | LpNormalizationOp
        | InstanceNormalizationOp
        | GroupNormalizationOp
        | LayerNormalizationOp
        | MeanVarianceNormalizationOp
        | RMSNormalizationOp
        | SoftmaxOp
        | LogSoftmaxOp
        | HardmaxOp
        | AdagradOp
        | NegativeLogLikelihoodLossOp
        | SoftmaxCrossEntropyLossOp
        | MaxPoolOp
        | ConcatOp
        | GatherElementsOp
        | GatherOp
        | GatherNDOp
        | TransposeOp
        | ReshapeOp
        | IdentityOp
        | EyeLikeOp
        | TriluOp
        | TileOp
        | ResizeOp
        | GridSampleOp
        | ReduceOp
        | ArgReduceOp
        | ConstantOfShapeOp
        | ShapeOp
        | SizeOp
        | NonZeroOp
        | NonMaxSuppressionOp
        | ExpandOp
        | CumSumOp
        | RangeOp
        | OneHotOp
        | SplitOp
        | PadOp,
    ) -> ScalarType:
        if isinstance(op, ArgReduceOp):
            return op.output_dtype
        if isinstance(op, TopKOp):
            return op.output_values_dtype
        if isinstance(op, NonMaxSuppressionOp):
            return op.output_dtype
        return op.dtype

    @staticmethod
    def _codegen_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
        if not shape:
            return (1,)
        return tuple(max(1, dim) if isinstance(dim, int) else dim for dim in shape)

    @staticmethod
    def _array_suffix(shape: tuple[int, ...]) -> str:
        shape = CEmitter._codegen_shape(shape)
        return "".join(f"[{dim}]" for dim in shape)

    def _param_array_suffix(
        self,
        shape: tuple[int, ...],
        dim_names: Mapping[int, str] | None = None,
    ) -> str:
        shape = CEmitter._codegen_shape(shape)
        dim_names = dim_names or {}
        if not self._restrict_arrays:
            return "".join(
                f"[{dim_names.get(index, dim)}]"
                for index, dim in enumerate(shape)
            )
        first, *rest = shape
        first_dim = dim_names.get(0, first)
        rest_dims = "".join(
            f"[{dim_names.get(index + 1, dim)}]"
            for index, dim in enumerate(rest)
        )
        return f"[restrict {first_dim}]{rest_dims}"

    @staticmethod
    def _format_dim_args(dim_order: Sequence[str]) -> list[str]:
        return [f"int {dim_name}" for dim_name in dim_order]

    @staticmethod
    def _format_dim_args_prefix(dim_order: Sequence[str]) -> str:
        if not dim_order:
            return ""
        return ", ".join(f"int {dim_name}" for dim_name in dim_order) + ", "

    def _build_variable_dim_names(
        self,
        model: LoweredModel,
        variable_dim_inputs: Mapping[int, Mapping[int, str]] | None,
        variable_dim_outputs: Mapping[int, Mapping[int, str]] | None,
    ) -> tuple[
        list[str],
        dict[int, dict[int, str]],
        dict[int, dict[int, str]],
        dict[str, int],
    ]:
        variable_dim_inputs = variable_dim_inputs or {}
        variable_dim_outputs = variable_dim_outputs or {}
        dim_order: list[str] = []
        dim_vars: dict[tuple[str, int, int], str] = {}
        dim_values: dict[str, int] = {}
        reserved_names = set(model.input_names) | set(model.output_names)
        used_names = set(reserved_names)
        dim_aliases: dict[str, str] = {}

        def _unique_dim_name(dim_name: str) -> str:
            if dim_name in dim_aliases:
                return dim_aliases[dim_name]
            base_name = dim_name
            if base_name in used_names:
                base_name = f"{dim_name}_dim"
            candidate = base_name
            counter = 1
            while candidate in used_names:
                counter += 1
                candidate = f"{base_name}{counter}"
            dim_aliases[dim_name] = candidate
            used_names.add(candidate)
            return candidate

        def _register_dim(
            kind: str,
            tensor_index: int,
            dim_index: int,
            dim_name: str,
            dim_value: int,
        ) -> str:
            key = (kind, tensor_index, dim_index)
            dim_name = _unique_dim_name(dim_name)
            if key not in dim_vars:
                dim_vars[key] = dim_name
                if dim_name not in dim_order:
                    dim_order.append(dim_name)
                dim_values.setdefault(dim_name, dim_value)
            else:
                if dim_values[dim_name] != dim_value:
                    raise CodegenError(
                        "Variable dimension values must be consistent, "
                        f"got {dim_values[dim_name]} and {dim_value} for {dim_name}"
                    )
            return dim_name

        def _build_dim_names(
            kind: str,
            tensor_index: int,
            shape: tuple[int, ...],
            variable_dims: Mapping[int, Mapping[int, str]],
        ) -> dict[int, str]:
            dim_names: dict[int, str] = {}
            for dim_index, dim_name in variable_dims.get(tensor_index, {}).items():
                if dim_index < 0 or dim_index >= len(shape):
                    raise CodegenError(
                        f"Variable {kind} dim {dim_index} is out of range for shape {shape}"
                    )
                dim_names[dim_index] = _register_dim(
                    kind,
                    tensor_index,
                    dim_index,
                    dim_name,
                    shape[dim_index],
                )
            return dim_names

        input_dim_names: dict[int, dict[int, str]] = {}
        for index, shape in enumerate(model.input_shapes):
            dim_names = _build_dim_names(
                "input", index, shape, variable_dim_inputs
            )
            if dim_names:
                input_dim_names[index] = dim_names

        output_dim_names: dict[int, dict[int, str]] = {}
        for index, shape in enumerate(model.output_shapes):
            dim_names = _build_dim_names(
                "output", index, shape, variable_dim_outputs
            )
            if dim_names:
                output_dim_names[index] = dim_names

        return dim_order, input_dim_names, output_dim_names, dim_values

    @staticmethod
    def _shape_dim_exprs(
        shape: tuple[int, ...],
        dim_names: Mapping[int, str] | None,
    ) -> tuple[str | int, ...]:
        dim_names = dim_names or {}
        if not shape:
            shape = (1,)
        return tuple(
            dim_names.get(index, dim) for index, dim in enumerate(shape)
        )

    @staticmethod
    def _element_count_expr(shape_exprs: Sequence[str | int]) -> str:
        if not shape_exprs:
            return "1"
        return " * ".join(str(dim) for dim in shape_exprs)

    def _build_tensor_dim_names(
        self,
        model: LoweredModel,
        input_dim_names: Mapping[int, Mapping[int, str]],
        output_dim_names: Mapping[int, Mapping[int, str]],
    ) -> dict[str, dict[int, str]]:
        dim_names: dict[str, dict[int, str]] = {}
        for index, name in enumerate(model.input_names):
            if index in input_dim_names:
                dim_names[name] = dict(input_dim_names[index])
        for index, name in enumerate(model.output_names):
            if index in output_dim_names:
                dim_names[name] = dict(output_dim_names[index])
        return dim_names

    @staticmethod
    def _loop_vars(shape: tuple[int, ...]) -> tuple[str, ...]:
        shape = CEmitter._codegen_shape(shape)
        return tuple(f"i{index}" for index in range(len(shape)))

    @staticmethod
    def _broadcast_index_expr(
        name: str,
        input_shape: tuple[int, ...],
        output_shape: tuple[int, ...],
        loop_vars: tuple[str, ...],
    ) -> str:
        if not input_shape:
            return f"{name}[0]"
        output_shape = CEmitter._codegen_shape(output_shape)
        if len(output_shape) != len(loop_vars):
            raise CodegenError("Loop variables must match output shape rank")
        offset = len(output_shape) - len(input_shape)
        indices: list[str] = []
        for input_dim, dim_size in enumerate(input_shape):
            output_dim = input_dim + offset
            if dim_size == 1:
                indices.append("[0]")
            else:
                indices.append(f"[{loop_vars[output_dim]}]")
        return f"{name}{''.join(indices)}"

    @staticmethod
    def _element_count(shape: tuple[int, ...]) -> int:
        if not shape:
            return 1
        count = 1
        for dim in shape:
            if dim < 0:
                raise CodegenError("Dynamic dims are not supported")
            if dim == 0:
                return 0
            count *= dim
        return count

    @staticmethod
    def _matmul_index_exprs(
        op: MatMulOp,
        batch_vars: tuple[str, ...],
        row_var: str | None,
        col_var: str | None,
        batch_rank: int,
        *,
        input0: str | None = None,
        input1: str | None = None,
    ) -> tuple[str, str]:
        def batch_indices(
            batch_shape: tuple[int, ...], actual_rank: int
        ) -> list[str]:
            if actual_rank == 0:
                return []
            offset = batch_rank - actual_rank
            indices: list[str] = []
            for idx in range(actual_rank):
                dim = batch_shape[offset + idx]
                var = batch_vars[offset + idx]
                indices.append("0" if dim == 1 else var)
            return indices

        if op.left_vector:
            input0_indices = ["k"]
        else:
            input0_batch_rank = len(op.input0_shape) - 2
            input0_indices = batch_indices(
                op.input0_batch_shape, input0_batch_rank
            )
            input0_indices.append(row_var if row_var is not None else "0")
            input0_indices.append("k")
        if op.right_vector:
            input1_indices = ["k"]
        else:
            input1_batch_rank = len(op.input1_shape) - 2
            input1_indices = batch_indices(
                op.input1_batch_shape, input1_batch_rank
            )
            input1_indices.append("k")
            input1_indices.append(col_var if col_var is not None else "0")
        input0_index_expr = f"{input0 or op.input0}" + "".join(
            f"[{index}]" for index in input0_indices
        )
        input1_index_expr = f"{input1 or op.input1}" + "".join(
            f"[{index}]" for index in input1_indices
        )
        return input0_index_expr, input1_index_expr

    def _emit_testbench(
        self,
        model: LoweredModel,
        testbench_template,
        *,
        testbench_inputs: Mapping[str, tuple[float | int | bool, ...]] | None = None,
        dim_order: Sequence[str],
        dim_values: Mapping[str, int],
        weight_data_filename: str,
    ) -> str:
        input_counts = tuple(
            self._element_count(shape) for shape in model.input_shapes
        )
        testbench_inputs = testbench_inputs or {}
        inputs = []
        for name, shape, count, dtype in zip(
            model.input_names, model.input_shapes, input_counts, model.input_dtypes
        ):
            codegen_shape = self._codegen_shape(shape)
            loop_shape = (1,) if not shape else shape
            loop_vars = self._loop_vars(loop_shape)
            if dtype in {ScalarType.F16, ScalarType.F32}:
                random_expr = "rng_next_float()"
            elif dtype == ScalarType.F64:
                random_expr = "rng_next_double()"
            elif dtype == ScalarType.BOOL:
                random_expr = "((rng_next_u64() & 1ull) != 0)"
            else:
                random_expr = f"({dtype.c_type})rng_next_i64()"
            constant_values = testbench_inputs.get(name)
            constant_name = None
            constant_lines = None
            if constant_values is not None:
                constant_name = f"{name}_testbench_data"
                if constant_values:
                    constant_lines = [
                        self._format_value(value, dtype)
                        for value in constant_values
                    ]
                else:
                    constant_lines = [self._format_value(0, dtype)]
            inputs.append(
                {
                    "name": name,
                    "shape": loop_shape,
                    "shape_literal": ",".join(str(dim) for dim in shape),
                    "count": count,
                    "array_suffix": self._array_suffix(codegen_shape),
                    "array_index_expr": "".join(
                        f"[{var}]" for var in loop_vars
                    ),
                    "loop_vars": loop_vars,
                    "rank": len(loop_shape),
                    "index_expr": self._index_expr(loop_shape, loop_vars),
                    "dtype": dtype,
                    "c_type": dtype.c_type,
                    "random_expr": random_expr,
                    "print_format": self._print_format(dtype),
                    "print_cast": self._print_cast(dtype),
                    "constant_name": constant_name,
                    "constant_lines": constant_lines,
                }
            )
        outputs = []
        for name, shape, dtype in zip(
            model.output_names, model.output_shapes, model.output_dtypes
        ):
            codegen_shape = self._codegen_shape(shape)
            loop_shape = (1,) if not shape else shape
            output_loop_vars = self._loop_vars(loop_shape)
            outputs.append(
                {
                    "name": name,
                    "shape": loop_shape,
                    "shape_literal": ",".join(str(dim) for dim in shape),
                    "count": self._element_count(shape),
                    "array_suffix": self._array_suffix(codegen_shape),
                    "array_index_expr": "".join(
                        f"[{var}]" for var in output_loop_vars
                    ),
                    "loop_vars": output_loop_vars,
                    "rank": len(loop_shape),
                    "index_expr": self._index_expr(loop_shape, output_loop_vars),
                    "dtype": dtype,
                    "c_type": dtype.c_type,
                    "print_format": self._print_format(dtype),
                    "print_cast": self._print_cast(dtype),
                }
            )
        rendered = testbench_template.render(
            model_name=model.name,
            dim_args=[
                {"name": dim_name, "value": dim_values[dim_name]}
                for dim_name in dim_order
            ],
            inputs=inputs,
            outputs=outputs,
            weight_data_filename=weight_data_filename,
        ).rstrip()
        return _format_c_indentation(rendered)

    @staticmethod
    def _testbench_requires_math(
        model: LoweredModel,
        testbench_inputs: Mapping[str, tuple[float | int | bool, ...]] | None,
    ) -> bool:
        if not testbench_inputs:
            return False
        dtype_map = dict(zip(model.input_names, model.input_dtypes))
        float_dtypes = {ScalarType.F16, ScalarType.F32, ScalarType.F64}
        for name, values in testbench_inputs.items():
            if dtype_map.get(name) not in float_dtypes:
                continue
            for value in values:
                if not math.isfinite(float(value)):
                    return True
        return False

    def _partition_constants(
        self, constants: tuple[ConstTensor, ...]
    ) -> tuple[tuple[ConstTensor, ...], tuple[ConstTensor, ...]]:
        if self._large_weight_threshold <= 0:
            return constants, ()
        inline: list[ConstTensor] = []
        large: list[ConstTensor] = []
        for const in constants:
            if self._element_count(const.shape) > self._large_weight_threshold:
                large.append(const)
            else:
                inline.append(const)
        return tuple(inline), tuple(large)

    @staticmethod
    def _weight_data_filename(model: LoweredModel) -> str:
        return f"{model.name}.bin"

    def _emit_weight_loader(
        self, model: LoweredModel, large_constants: tuple[ConstTensor, ...]
    ) -> str:
        lines = [f"_Bool {model.name}_load(const char *path) {{"]
        if not large_constants:
            lines.append("    (void)path;")
            lines.append("    return 1;")
            lines.append("}")
            return _format_c_indentation("\n".join(lines))
        lines.append("    FILE *file = fopen(path, \"rb\");")
        lines.append("    if (!file) {")
        lines.append("        return 0;")
        lines.append("    }")
        lines.append(
            f"    _Bool ok = {model.name}_load_file(file);"
        )
        lines.append("    fclose(file);")
        lines.append("    return ok;")
        lines.append("}")
        lines.append("")
        lines.append(f"static _Bool {model.name}_load_file(FILE *file) {{")
        for const in large_constants:
            shape = self._codegen_shape(const.shape)
            loop_vars = self._loop_vars(shape)
            for depth, var in enumerate(loop_vars):
                lines.append(
                    f"    for (idx_t {var} = 0; {var} < {shape[depth]}; ++{var}) {{"
                )
            index_expr = "".join(f"[{var}]" for var in loop_vars)
            zero_index = "[0]" * len(shape)
            lines.append(
                f"        if (fread(&{const.name}{index_expr}, "
                f"sizeof({const.name}{zero_index}), 1, file) != 1) {{"
            )
            lines.append("            return 0;")
            lines.append("        }")
            for _ in loop_vars[::-1]:
                lines.append("    }")
        lines.append("    return 1;")
        lines.append("}")
        return _format_c_indentation("\n".join(lines))

    def _emit_constant_definitions(
        self,
        constants: tuple[ConstTensor, ...],
        *,
        storage_prefix: str = "static const",
    ) -> str:
        if not constants:
            return ""
        lines: list[str] = []
        for index, const in enumerate(constants, start=1):
            lines.append(self._emit_constant_comment(const, index))
            c_type = const.dtype.c_type
            shape = self._codegen_shape(const.shape)
            array_suffix = self._array_suffix(shape)
            values = [
                self._format_weight_value(value, const.dtype)
                for value in const.data
            ]
            lines.append(
                f"{storage_prefix} {c_type} {const.name}{array_suffix} = {{"
            )
            if values:
                if (
                    self._truncate_weights_after is not None
                    and len(values) > self._truncate_weights_after
                ):
                    truncated_lines, _, _, _ = (
                        self._emit_initializer_lines_truncated(
                            values, shape, self._truncate_weights_after
                        )
                    )
                    lines.extend(truncated_lines)
                else:
                    lines.extend(self._emit_initializer_lines(values, shape))
            lines.append("};")
            lines.append("")
        if lines and not lines[-1]:
            lines.pop()
        return "\n".join(lines)

    def _emit_constant_declarations(
        self, constants: tuple[ConstTensor, ...]
    ) -> str:
        if not constants:
            return ""
        lines = []
        for index, const in enumerate(constants, start=1):
            lines.append(self._emit_constant_comment(const, index))
            c_type = const.dtype.c_type
            array_suffix = self._array_suffix(const.shape)
            lines.append(f"extern const {c_type} {const.name}{array_suffix};")
        return "\n".join(lines)

    def _emit_constant_storage_definitions(
        self,
        constants: tuple[ConstTensor, ...],
        *,
        storage_prefix: str = "static",
    ) -> str:
        if not constants:
            return ""
        lines: list[str] = []
        for index, const in enumerate(constants, start=1):
            lines.append(self._emit_constant_comment(const, index))
            c_type = const.dtype.c_type
            array_suffix = self._array_suffix(const.shape)
            lines.append(f"{storage_prefix} {c_type} {const.name}{array_suffix};")
            lines.append("")
        if lines and not lines[-1]:
            lines.pop()
        return "\n".join(lines)

    def collect_weight_data(
        self, constants: tuple[ConstTensor, ...]
    ) -> bytes | None:
        _, large_constants = self._partition_constants(constants)
        if not large_constants:
            return None
        chunks: list[bytes] = []
        for const in large_constants:
            array = np.asarray(const.data, dtype=const.dtype.np_dtype)
            chunks.append(array.tobytes(order="C"))
        return b"".join(chunks)

    @staticmethod
    def _index_expr(shape: tuple[int, ...], loop_vars: tuple[str, ...]) -> str:
        shape = CEmitter._codegen_shape(shape)
        if len(shape) != len(loop_vars):
            raise CodegenError("Loop variables must match shape rank")
        if not shape:
            return "0"
        expr = loop_vars[0]
        for dim, var in zip(shape[1:], loop_vars[1:]):
            expr = f"({expr} * {dim} + {var})"
        return expr

    @staticmethod
    def _format_float(value: float) -> str:
        if math.isnan(value):
            return "NAN"
        if math.isinf(value):
            return "-INFINITY" if value < 0 else "INFINITY"
        formatted = f"{value:.9g}"
        if "e" not in formatted and "E" not in formatted and "." not in formatted:
            formatted = f"{formatted}.0"
        return f"{formatted}f"

    @staticmethod
    def _format_float16(value: float) -> str:
        return f"(_Float16){CEmitter._format_float(value)}"

    @staticmethod
    def _format_double(value: float) -> str:
        if math.isnan(value):
            return "NAN"
        if math.isinf(value):
            return "-INFINITY" if value < 0 else "INFINITY"
        formatted = f"{value:.17g}"
        if "e" not in formatted and "E" not in formatted and "." not in formatted:
            formatted = f"{formatted}.0"
        return formatted

    @staticmethod
    def _format_float32_hex(value: float) -> str:
        bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
        sign = "-" if (bits >> 31) else ""
        exponent = (bits >> 23) & 0xFF
        mantissa = bits & 0x7FFFFF
        if exponent == 0 and mantissa == 0:
            return f"{sign}0x0.0p+0"
        if exponent == 0xFF:
            if mantissa == 0:
                return f"{sign}INFINITY"
            return "NAN"
        if exponent == 0:
            shift = mantissa.bit_length() - 1
            exponent_val = shift - 149
            fraction = (mantissa - (1 << shift)) << (24 - shift)
        else:
            exponent_val = exponent - 127
            fraction = mantissa << 1
        return f"{sign}0x1.{fraction:06x}p{exponent_val:+d}"

    @staticmethod
    def _format_float64_hex(value: float) -> str:
        bits = struct.unpack("<Q", struct.pack("<d", float(value)))[0]
        sign = "-" if (bits >> 63) else ""
        exponent = (bits >> 52) & 0x7FF
        mantissa = bits & 0xFFFFFFFFFFFFF
        if exponent == 0 and mantissa == 0:
            return f"{sign}0x0.0p+0"
        if exponent == 0x7FF:
            if mantissa == 0:
                return f"{sign}INFINITY"
            return "NAN"
        if exponent == 0:
            shift = mantissa.bit_length() - 1
            exponent_val = shift - 1074
            fraction = (mantissa - (1 << shift)) << (52 - shift)
        else:
            exponent_val = exponent - 1023
            fraction = mantissa
        return f"{sign}0x1.{fraction:013x}p{exponent_val:+d}"

    @staticmethod
    def _format_floating(value: float, dtype: ScalarType) -> str:
        if dtype == ScalarType.F64:
            return CEmitter._format_double(value)
        if dtype == ScalarType.F16:
            return CEmitter._format_float16(value)
        return CEmitter._format_float(value)

    @staticmethod
    def _math_fn(dtype: ScalarType, float_name: str, double_name: str) -> str:
        if dtype == ScalarType.F64:
            return double_name
        return float_name

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
    def _format_uint(value: int, bits: int, max_macro: str) -> str:
        max_value = 2**bits - 1
        if value == max_value:
            return max_macro
        return str(int(value))

    @staticmethod
    def _format_literal(dtype: ScalarType, value: float | int | bool) -> str:
        if dtype == ScalarType.F16:
            return CEmitter._format_float16(float(value))
        if dtype == ScalarType.F32:
            return CEmitter._format_float(float(value))
        if dtype == ScalarType.F64:
            return CEmitter._format_double(float(value))
        if dtype == ScalarType.BOOL:
            return "true" if bool(value) else "false"
        if dtype == ScalarType.U64:
            return CEmitter._format_uint(int(value), 64, "UINT64_MAX")
        if dtype == ScalarType.U32:
            return CEmitter._format_uint(int(value), 32, "UINT32_MAX")
        if dtype == ScalarType.U16:
            return CEmitter._format_uint(int(value), 16, "UINT16_MAX")
        if dtype == ScalarType.U8:
            return CEmitter._format_uint(int(value), 8, "UINT8_MAX")
        if dtype == ScalarType.I64:
            return CEmitter._format_int64(int(value))
        if dtype == ScalarType.I32:
            return CEmitter._format_int(int(value), 32, "INT32_MIN")
        if dtype == ScalarType.I16:
            return CEmitter._format_int(int(value), 16, "INT16_MIN")
        if dtype == ScalarType.I8:
            return CEmitter._format_int(int(value), 8, "INT8_MIN")
        raise CodegenError(f"Unsupported dtype {dtype.onnx_name}")

    def _format_value(self, value: float | int | bool, dtype: ScalarType) -> str:
        if dtype == ScalarType.F16:
            return self._format_float16(float(value))
        if dtype == ScalarType.F32:
            return self._format_float(float(value))
        if dtype == ScalarType.F64:
            return self._format_double(float(value))
        if dtype == ScalarType.BOOL:
            return "true" if bool(value) else "false"
        if dtype == ScalarType.U64:
            return self._format_uint(int(value), 64, "UINT64_MAX")
        if dtype == ScalarType.U32:
            return self._format_uint(int(value), 32, "UINT32_MAX")
        if dtype == ScalarType.U16:
            return self._format_uint(int(value), 16, "UINT16_MAX")
        if dtype == ScalarType.U8:
            return self._format_uint(int(value), 8, "UINT8_MAX")
        if dtype == ScalarType.I64:
            return self._format_int64(int(value))
        if dtype == ScalarType.I32:
            return self._format_int(int(value), 32, "INT32_MIN")
        if dtype == ScalarType.I16:
            return self._format_int(int(value), 16, "INT16_MIN")
        if dtype == ScalarType.I8:
            return self._format_int(int(value), 8, "INT8_MIN")
        raise CodegenError(f"Unsupported dtype {dtype.onnx_name}")

    def _format_weight_value(
        self, value: float | int | bool, dtype: ScalarType
    ) -> str:
        if dtype == ScalarType.F16:
            formatted = self._format_float32_hex(float(value))
            if formatted == "NAN" or formatted.endswith("INFINITY"):
                return f"(_Float16){formatted}"
            return f"(_Float16){formatted}f"
        if dtype == ScalarType.F32:
            formatted = self._format_float32_hex(float(value))
            if formatted == "NAN" or formatted.endswith("INFINITY"):
                return formatted
            return f"{formatted}f"
        if dtype == ScalarType.F64:
            return self._format_float64_hex(float(value))
        if dtype == ScalarType.BOOL:
            return "true" if bool(value) else "false"
        if dtype == ScalarType.U64:
            return self._format_uint(int(value), 64, "UINT64_MAX")
        if dtype == ScalarType.U32:
            return self._format_uint(int(value), 32, "UINT32_MAX")
        if dtype == ScalarType.U16:
            return self._format_uint(int(value), 16, "UINT16_MAX")
        if dtype == ScalarType.U8:
            return self._format_uint(int(value), 8, "UINT8_MAX")
        if dtype == ScalarType.I64:
            return self._format_int64(int(value))
        if dtype == ScalarType.I32:
            return self._format_int(int(value), 32, "INT32_MIN")
        if dtype == ScalarType.I16:
            return self._format_int(int(value), 16, "INT16_MIN")
        if dtype == ScalarType.I8:
            return self._format_int(int(value), 8, "INT8_MIN")
        raise CodegenError(f"Unsupported dtype {dtype.onnx_name}")

    @staticmethod
    def _emit_initializer_lines(
        values: Sequence[str],
        shape: tuple[int, ...],
        indent: str = "    ",
        per_line: int = 8,
    ) -> list[str]:
        if len(shape) == 1:
            lines: list[str] = []
            for index in range(0, len(values), per_line):
                chunk = ", ".join(values[index : index + per_line])
                lines.append(f"{indent}{chunk},")
            if lines:
                lines[-1] = lines[-1].rstrip(",")
            return lines
        sub_shape = shape[1:]
        sub_size = prod(sub_shape)
        lines = []
        for index in range(shape[0]):
            start = index * sub_size
            end = start + sub_size
            lines.append(f"{indent}{{")
            lines.extend(
                CEmitter._emit_initializer_lines(
                    values[start:end],
                    sub_shape,
                    indent + "    ",
                    per_line,
                )
            )
            lines.append(f"{indent}}},")
        if lines:
            lines[-1] = lines[-1].rstrip(",")
        return lines

    @staticmethod
    def _emit_initializer_lines_truncated(
        values: Sequence[str],
        shape: tuple[int, ...],
        truncate_after: int,
        indent: str = "    ",
        per_line: int = 8,
        start_index: int = 0,
        emitted: int = 0,
    ) -> tuple[list[str], int, int, bool]:
        if len(shape) == 1:
            items: list[str] = []
            truncated = False
            index = start_index
            for _ in range(shape[0]):
                if emitted >= truncate_after:
                    items.append("...")
                    truncated = True
                    break
                items.append(values[index])
                index += 1
                emitted += 1
            lines: list[str] = []
            for item_index in range(0, len(items), per_line):
                chunk = ", ".join(items[item_index : item_index + per_line])
                lines.append(f"{indent}{chunk},")
            if lines:
                lines[-1] = lines[-1].rstrip(",")
            return lines, index, emitted, truncated
        sub_shape = shape[1:]
        lines: list[str] = []
        index = start_index
        truncated = False
        for _ in range(shape[0]):
            lines.append(f"{indent}{{")
            sub_lines, index, emitted, sub_truncated = (
                CEmitter._emit_initializer_lines_truncated(
                    values,
                    sub_shape,
                    truncate_after,
                    indent + "    ",
                    per_line,
                    index,
                    emitted,
                )
            )
            lines.extend(sub_lines)
            lines.append(f"{indent}}},")
            if sub_truncated:
                truncated = True
                break
        if lines:
            lines[-1] = lines[-1].rstrip(",")
        return lines, index, emitted, truncated

    @staticmethod
    def _print_format(dtype: ScalarType) -> str:
        if dtype == ScalarType.F16:
            return "\\\"%a\\\""
        if dtype == ScalarType.F32:
            return "\\\"%a\\\""
        if dtype == ScalarType.F64:
            return "\\\"%a\\\""
        if dtype == ScalarType.BOOL:
            return "%d"
        if dtype == ScalarType.U64:
            return "%llu"
        if dtype == ScalarType.U32:
            return "%u"
        if dtype == ScalarType.U16:
            return "%hu"
        if dtype == ScalarType.U8:
            return "%hhu"
        if dtype == ScalarType.I64:
            return "%lld"
        if dtype == ScalarType.I32:
            return "%d"
        if dtype == ScalarType.I16:
            return "%hd"
        if dtype == ScalarType.I8:
            return "%hhd"
        raise CodegenError(f"Unsupported dtype {dtype.onnx_name}")

    @staticmethod
    def _print_cast(dtype: ScalarType) -> str:
        if dtype.is_float:
            return "(double)"
        if dtype == ScalarType.BOOL:
            return "(int)"
        if dtype == ScalarType.U64:
            return "(unsigned long long)"
        if dtype in {ScalarType.U32, ScalarType.U16, ScalarType.U8}:
            return "(unsigned int)"
        if dtype == ScalarType.I64:
            return "(long long)"
        if dtype in {ScalarType.I32, ScalarType.I16, ScalarType.I8}:
            return "(int)"
        raise CodegenError(f"Unsupported dtype {dtype.onnx_name}")


def _format_multiline_value(value: str | None) -> list[str]:
    if not value:
        return ["  n/a"]
    lines = value.splitlines() or [""]
    return [f"  {line}" if line else "  " for line in lines]

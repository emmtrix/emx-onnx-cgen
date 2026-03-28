from __future__ import annotations

from dataclasses import dataclass
import math
from math import prod
from pathlib import Path
import re
import struct
from typing import Mapping, Sequence

from jinja2 import (
    Environment,
    FileSystemLoader,
    PackageLoader,
    Template,
    select_autoescape,
)
import numpy as np
from onnx import GraphProto

from ..errors import CodegenError
from ..testbench_output_format import parse_testbench_output_format
from ..sequence_shape_hints import SequenceElementShapeHint
from ..ir.op_base import (
    OpBase,
    CodegenDim,
    EmitContext,
)
from ..ir.op_context import OpContext
from ..ir.model import SequenceType, TensorType, ValueType
from ..ir.ops import (
    ArgReduceOp,
    AttentionOp,
    AveragePoolOp,
    BatchNormOp,
    BinaryOp,
    BlackmanWindowOp,
    CastOp,
    ClipOp,
    CompressOp,
    ConcatOp,
    ConcatFromSequenceOp,
    ConstantOfShapeOp,
    Col2ImOp,
    ConvOp,
    ConvIntegerOp,
    ConvTransposeOp,
    DeformConvOp,
    DetOp,
    CumSumOp,
    DFTOp,
    DepthToSpaceOp,
    DequantizeLinearOp,
    EinsumOp,
    ExpandOp,
    EyeLikeOp,
    GatherElementsOp,
    GatherNDOp,
    GatherOp,
    GemmOp,
    GruOp,
    AffineGridOp,
    GridSampleOp,
    GroupNormalizationOp,
    HammingWindowOp,
    HannWindowOp,
    IfOptionalSequenceConstOp,
    HardmaxOp,
    IdentityOp,
    InstanceNormalizationOp,
    LayerNormalizationOp,
    LogSoftmaxOp,
    LpNormalizationOp,
    LpPoolOp,
    LrnOp,
    RnnOp,
    LstmOp,
    MatMulOp,
    MaxPoolOp,
    MeanVarianceNormalizationOp,
    NegativeLogLikelihoodLossOp,
    NonMaxSuppressionOp,
    NonZeroOp,
    OneHotOp,
    QuantizeLinearOp,
    QLinearAddOp,
    QLinearMulOp,
    QLinearMatMulOp,
    QLinearSoftmaxOp,
    LoopSequenceInsertOp,
    LoopSequenceMapOp,
    RangeOp,
    SequenceAtOp,
    SequenceConstructOp,
    SequenceEmptyOp,
    SequenceEraseOp,
    SequenceIdentityOp,
    SequenceInsertOp,
    SequenceLengthOp,
    ReduceOp,
    ReshapeOp,
    ResizeOp,
    RMSNormalizationOp,
    RoiAlignOp,
    ShapeOp,
    SizeOp,
    SliceOp,
    SoftmaxCrossEntropyLossOp,
    SoftmaxOp,
    SpaceToDepthOp,
    SplitOp,
    SplitToSequenceOp,
    TreeEnsembleClassifierOp,
    TileOp,
    CenterCropPadOp,
    TopKOp,
    TransposeOp,
    TriluOp,
    UniqueOp,
    UnaryOp,
    WhereOp,
)
from shared.scalar_functions import (
    ScalarFunction,
    ScalarFunctionKey,
    ScalarFunctionRegistry,
)
from shared.fft_kernel_registry import FFTKernelRegistry
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
    is_sequence: bool = False


@dataclass(frozen=True)
class ModelHeader:
    generator: str
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
    codegen_settings: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class LoweredModel:
    name: str
    input_names: tuple[str, ...]
    input_optional_names: tuple[str | None, ...]
    input_shapes: tuple[tuple[int, ...], ...]
    input_dtypes: tuple[ScalarType, ...]
    input_types: tuple[ValueType, ...]
    output_names: tuple[str, ...]
    output_optional_names: tuple[str | None, ...]
    output_shapes: tuple[tuple[int, ...], ...]
    output_dtypes: tuple[ScalarType, ...]
    output_types: tuple[ValueType, ...]
    constants: tuple[ConstTensor, ...]
    ops: tuple[OpBase, ...]
    node_infos: tuple[NodeInfo, ...]
    header: ModelHeader
    op_context: OpContext


@dataclass
class _EmitState:
    model: LoweredModel
    templates: dict[str, Template]
    scalar_registry: ScalarFunctionRegistry
    fft_kernel_registry: FFTKernelRegistry
    dim_args: str
    tensor_dim_names: Mapping[str, Mapping[int, CodegenDim]]
    sequence_shape_hints: Mapping[str, SequenceElementShapeHint]
    sequence_dim_max_sizes: Mapping[str, int]
    op_context: OpContext
    value_name_map: Mapping[str, str]


class CEmitter:
    def __init__(
        self,
        template_dir: Path | None,
        *,
        restrict_arrays: bool = True,
        fp32_accumulation_strategy: str = "simple",
        fp16_accumulation_strategy: str = "fp32",
        truncate_weights_after: int | None = None,
        large_temp_threshold_bytes: int = 1024,
        large_weight_threshold: int = 1024,
        replicate_ort_bugs: bool = False,
        sequence_element_shapes: Mapping[str, SequenceElementShapeHint] | None = None,
    ) -> None:
        loader = (
            FileSystemLoader(str(template_dir))
            if template_dir is not None
            else PackageLoader("emx_onnx_cgen", "templates")
        )
        self._env = Environment(
            loader=loader,
            autoescape=select_autoescape(enabled_extensions=()),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self._restrict_arrays = restrict_arrays
        if fp32_accumulation_strategy not in {"simple", "fp64"}:
            raise CodegenError("fp32_accumulation_strategy must be 'simple' or 'fp64'")
        self._fp32_accumulation_strategy = fp32_accumulation_strategy
        if fp16_accumulation_strategy not in {"simple", "fp32"}:
            raise CodegenError("fp16_accumulation_strategy must be 'simple' or 'fp32'")
        self._fp16_accumulation_strategy = fp16_accumulation_strategy
        if truncate_weights_after is not None and truncate_weights_after < 1:
            raise CodegenError("truncate_weights_after must be >= 1")
        self._truncate_weights_after = truncate_weights_after
        if large_temp_threshold_bytes < 0:
            raise CodegenError("large_temp_threshold_bytes must be >= 0")
        self._large_temp_threshold_bytes = large_temp_threshold_bytes
        if large_weight_threshold < 0:
            raise CodegenError("large_weight_threshold must be >= 0")
        self._large_weight_threshold = large_weight_threshold
        self._replicate_ort_bugs = replicate_ort_bugs
        self._sequence_element_shapes = dict(sequence_element_shapes or {})
        self._emit_state: _EmitState | None = None

    def _setup_template_resolvers(
        self,
        scalar_registry: ScalarFunctionRegistry,
        fft_kernel_registry: FFTKernelRegistry,
    ) -> None:
        def scalar_fn(
            function: ScalarFunction,
            dtype: ScalarType,
            params: tuple[float, ...] = (),
        ) -> str | None:
            return CEmitter._scalar_function_name(
                function, dtype, scalar_registry, params=params
            )

        def fft_kernel(dtype: ScalarType, fft_length: int) -> str:
            return fft_kernel_registry.request(dtype=dtype, fft_length=fft_length)

        self._env.globals["scalar_fn"] = scalar_fn
        self._env.globals["fft_kernel"] = fft_kernel
        self._env.globals["SF"] = ScalarFunction
        self._env.globals["ST"] = ScalarType

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

    def _accumulation_dtype(self, dtype: ScalarType) -> ScalarType:
        if dtype == ScalarType.F32:
            return (
                ScalarType.F32
                if self._fp32_accumulation_strategy == "simple"
                else ScalarType.F64
            )
        if dtype in {ScalarType.F16, ScalarType.BF16}:
            return (
                dtype
                if self._fp16_accumulation_strategy == "simple"
                else ScalarType.F32
            )
        return dtype

    def _ctx_name(self, name: str) -> str:
        if self._emit_state is None:
            raise CodegenError("Emitter state not initialized")
        return self._emit_state.value_name_map.get(name, name)

    def _ctx_shape(self, name: str) -> tuple[int, ...]:
        if self._emit_state is None:
            raise CodegenError("Emitter state not initialized")
        return self._emit_state.op_context.shape(self._ctx_name(name))

    def _ctx_dtype(self, name: str) -> ScalarType:
        if self._emit_state is None:
            raise CodegenError("Emitter state not initialized")
        return self._emit_state.op_context.dtype(self._ctx_name(name))

    def _ctx_sequence_elem_type(self, name: str) -> TensorType:
        if self._emit_state is None:
            raise CodegenError("Emitter state not initialized")
        value = self._emit_state.op_context.find_value(self._ctx_name(name))
        if not isinstance(value.type, SequenceType):
            raise CodegenError(f"Expected sequence value for '{name}'")
        return value.type.elem

    @staticmethod
    def _merge_sequence_storage_shape(
        left: tuple[int, ...], right: tuple[int, ...]
    ) -> tuple[int, ...]:
        if left == (1,) and right != (1,):
            return right
        if right == (1,) and left != (1,):
            return left
        if len(left) != len(right):
            return right
        return tuple(max(a, b) for a, b in zip(left, right))

    def _sequence_shape_source_name(
        self,
        sequence_name: str,
        *,
        _visited: set[str] | None = None,
    ) -> str | None:
        if self._emit_state is None:
            return None
        if _visited is None:
            _visited = set()
        if sequence_name in _visited:
            return None
        _visited.add(sequence_name)
        producer = self._emit_state.op_context.producer(self._ctx_name(sequence_name))
        if producer is None or not producer.inputs:
            return None
        if producer.op_type in {"Identity", "SequenceErase", "SequenceInsert"}:
            src_name = producer.inputs[0]
            src_value = self._emit_state.op_context.find_value(self._ctx_name(src_name))
            return src_name if isinstance(src_value.type, SequenceType) else None
        if producer.op_type == "Loop" and producer.inputs:
            if self._loop_sequence_map_output_kind(sequence_name) == "shape":
                return None
            trip_count_producer = self._emit_state.op_context.producer(
                self._ctx_name(producer.inputs[0])
            )
            if (
                trip_count_producer is not None
                and trip_count_producer.op_type == "SequenceLength"
                and trip_count_producer.inputs
            ):
                src_name = trip_count_producer.inputs[0]
                src_value = self._emit_state.op_context.find_value(
                    self._ctx_name(src_name)
                )
                if isinstance(src_value.type, SequenceType):
                    return src_name
        if producer.op_type != "SequenceMap":
            return None
        output_rank = len(self._ctx_sequence_elem_type(sequence_name).shape)
        for src_name in producer.inputs:
            src_value = self._emit_state.op_context.find_value(self._ctx_name(src_name))
            if not isinstance(src_value.type, SequenceType):
                continue
            if len(src_value.type.elem.shape) == output_rank:
                return src_name
        return None

    @staticmethod
    def _loop_sequence_insert_capacity(body: object) -> int | None:
        if not isinstance(body, GraphProto):
            return None
        seq_insert_nodes = [
            node
            for node in body.node
            if node.op_type == "SequenceInsert" and len(node.input) >= 2
        ]
        if len(seq_insert_nodes) != 1:
            return None
        slice_name = seq_insert_nodes[0].input[1]
        slice_node = next(
            (
                node
                for node in body.node
                if node.op_type == "Slice"
                and len(node.output) == 1
                and node.output[0] == slice_name
                and node.input
            ),
            None,
        )
        if slice_node is None:
            return None
        const_name = slice_node.input[0]
        const_node = next(
            (
                node
                for node in body.node
                if node.op_type == "Constant"
                and len(node.output) == 1
                and node.output[0] == const_name
            ),
            None,
        )
        if const_node is None:
            return None
        value_attr = next(
            (attr for attr in const_node.attribute if attr.name == "value"), None
        )
        if value_attr is None or len(value_attr.t.dims) != 1:
            return None
        return int(value_attr.t.dims[0])

    def _loop_sequence_map_output_kind(self, sequence_name: str) -> str | None:
        if self._emit_state is None:
            return None
        for op in self._emit_state.model.ops:
            if not isinstance(op, LoopSequenceMapOp):
                continue
            for out_name, kind in zip(op.output_sequences, op.output_kinds):
                if out_name == sequence_name:
                    return kind
        return None

    def _sequence_storage_shape(
        self,
        sequence_name: str,
        *,
        _visited: set[str] | None = None,
    ) -> tuple[int, ...]:
        elem = self._ctx_sequence_elem_type(sequence_name)
        explicit_hint = (
            self._emit_state.sequence_shape_hints.get(sequence_name)
            if self._emit_state is not None
            else self._sequence_element_shapes.get(sequence_name)
        )
        if explicit_hint is not None:
            return explicit_hint.max_shape
        if elem.shape != ():
            resolved_shape: list[int] = []
            if self._emit_state is not None:
                symbol_sizes = self._emit_state.sequence_dim_max_sizes
            else:
                symbol_sizes = {}
            for axis, dim in enumerate(elem.shape):
                if dim >= 0:
                    resolved_shape.append(dim)
                    continue
                dim_param = (
                    elem.dim_params[axis] if axis < len(elem.dim_params) else None
                )
                if dim_param and dim_param in symbol_sizes:
                    resolved_shape.append(symbol_sizes[dim_param])
                    continue
                if dim_param:
                    resolved_shape.append(CodegenDim(dim_param).expected_size)
                    continue
                resolved_shape.append(
                    CodegenDim(f"{sequence_name}_dim_{axis}").expected_size
                )
            return tuple(resolved_shape)
        if _visited is None:
            _visited = set()
        if sequence_name in _visited:
            return (1,)
        if self._emit_state is None:
            raise CodegenError("Emitter state not initialized")
        mapped_name = self._ctx_name(sequence_name)
        producer = self._emit_state.op_context.producer(mapped_name)
        if producer is None or producer.op_type != "SequenceInsert":
            shape_source = self._sequence_shape_source_name(
                sequence_name, _visited=_visited
            )
            if shape_source is not None:
                return self._sequence_storage_shape(shape_source, _visited=_visited)
        _visited.add(sequence_name)
        if producer is not None and producer.op_type == "Loop":
            loop_capacity = self._loop_sequence_insert_capacity(
                producer.attrs.get("body")
            )
            if loop_capacity is not None:
                return (loop_capacity,)
        inferred = (1,)
        if producer is not None and producer.op_type == "SequenceInsert":
            in_sequence = producer.inputs[0]
            tensor = producer.inputs[1]
            prev_shape = self._sequence_storage_shape(in_sequence, _visited=_visited)
            tensor_shape = self._ctx_shape(tensor)
            inferred = self._merge_sequence_storage_shape(prev_shape, tensor_shape)
        elif producer is not None and producer.op_type == "SequenceErase":
            inferred = self._sequence_storage_shape(
                producer.inputs[0], _visited=_visited
            )
        for node in self._emit_state.op_context.nodes:
            if node.op_type == "Loop" and sequence_name in node.inputs[2:]:
                loop_capacity = self._loop_sequence_insert_capacity(
                    node.attrs.get("body")
                )
                if loop_capacity is not None:
                    inferred = self._merge_sequence_storage_shape(
                        inferred, (loop_capacity,)
                    )
                continue
            if not node.inputs or node.inputs[0] != mapped_name:
                continue
            if node.op_type != "SequenceInsert" or len(node.inputs) < 2:
                continue
            tensor_shape = self._ctx_shape(node.inputs[1])
            inferred = self._merge_sequence_storage_shape(inferred, tensor_shape)
            if node.outputs and node.outputs[0]:
                downstream = self._sequence_storage_shape(
                    node.outputs[0], _visited=_visited
                )
                inferred = self._merge_sequence_storage_shape(inferred, downstream)
        return inferred

    def _build_sequence_dim_max_sizes(
        self,
        model: LoweredModel,
        sequence_shape_hints: Mapping[str, SequenceElementShapeHint],
    ) -> dict[str, int]:
        symbol_sizes: dict[str, int] = {}
        for name, value_type in zip(model.input_names, model.input_types):
            if not isinstance(value_type, SequenceType):
                continue
            hint = sequence_shape_hints.get(name)
            if hint is None:
                continue
            for axis, dim_hint in enumerate(hint.dims):
                if axis >= len(value_type.elem.dim_params):
                    continue
                dim_param = value_type.elem.dim_params[axis]
                if not dim_param:
                    continue
                current = symbol_sizes.get(dim_param)
                if current is None or dim_hint.max_size > current:
                    symbol_sizes[dim_param] = dim_hint.max_size
        return symbol_sizes

    def _sequence_dynamic_axes(
        self, sequence_name: str, *, _visited: set[str] | None = None
    ) -> tuple[int, ...]:
        hint = (
            self._emit_state.sequence_shape_hints.get(sequence_name)
            if self._emit_state is not None
            else self._sequence_element_shapes.get(sequence_name)
        )
        if hint is not None:
            return hint.dynamic_axes
        if self._emit_state is None:
            return ()
        value = self._emit_state.op_context.find_value(self._ctx_name(sequence_name))
        if not isinstance(value.type, SequenceType):
            return ()
        if _visited is None:
            _visited = set()
        if sequence_name in _visited:
            return ()
        shape_source = self._sequence_shape_source_name(
            sequence_name, _visited=_visited
        )
        if shape_source is not None:
            return self._sequence_dynamic_axes(shape_source, _visited=_visited)
        direct_axes = tuple(
            axis
            for axis, (dim, dim_param) in enumerate(
                zip(value.type.elem.shape, value.type.elem.dim_params)
            )
            if dim < 0 or dim_param is not None
        )
        if direct_axes:
            return direct_axes
        producer = self._emit_state.op_context.producer(self._ctx_name(sequence_name))
        if producer is not None and producer.op_type == "Loop":
            loop_capacity = self._loop_sequence_insert_capacity(
                producer.attrs.get("body")
            )
            if loop_capacity is not None:
                return (0,)
        _visited.add(sequence_name)
        if producer is None:
            return ()
        if (
            producer.op_type in {"SequenceInsert", "SequenceErase", "Identity"}
            and producer.inputs
        ):
            src_name = producer.inputs[0]
            return self._sequence_dynamic_axes(src_name, _visited=_visited)
        return ()

    @staticmethod
    def _sequence_dim_array_name(name: str, axis: int) -> str:
        return f"{name}__dim_{axis}"

    def _sequence_dim_arg_names(self, name: str) -> list[str]:
        return [
            self._sequence_dim_array_name(name, axis)
            for axis in self._sequence_dynamic_axes(name)
        ]

    def _derived(self, op: OpBase, key: str) -> object:
        if self._emit_state is None:
            raise CodegenError("Emitter state not initialized")
        return self._emit_state.op_context.require_derived(op, key)

    def _maybe_derived(self, op: OpBase, key: str) -> object | None:
        if self._emit_state is None:
            raise CodegenError("Emitter state not initialized")
        value = self._emit_state.op_context.get_derived(op, key, None)
        return value

    @staticmethod
    def _build_param_decls(
        specs: Sequence[tuple[str | None, str, str, bool]],
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
            decls.append(f"{const_prefix}{info['c_type']} {name}{info['suffix']}")
        return decls

    @staticmethod
    def _dft_stockham_stage_plan(
        fft_length: int,
    ) -> tuple[tuple[str, int, int], ...]:
        stages: list[tuple[str, int, int]] = []
        m = 1
        while m < fft_length:
            remainder = fft_length // m
            if remainder % 4 == 0:
                stage_span = fft_length // (4 * m)
                stages.append(("radix4", m, stage_span))
                m *= 4
                continue
            if remainder % 2 == 0:
                stage_span = fft_length // (2 * m)
                stages.append(("radix2", m, stage_span))
                m *= 2
                continue
            return ()
        return tuple(stages)

    @staticmethod
    def _dft_twiddle_table(
        fft_length: int,
        *,
        inverse: bool,
        dtype: ScalarType,
    ) -> tuple[tuple[str, str], ...]:
        sign = 1.0 if inverse else -1.0
        twiddles: list[tuple[str, str]] = []
        for index in range(fft_length):
            angle = sign * 2.0 * math.pi * (index / fft_length)
            real = math.cos(angle)
            imag = math.sin(angle)
            if abs(real) < 1e-15:
                real = 0.0
            if abs(imag) < 1e-15:
                imag = 0.0
            twiddles.append(
                (
                    CEmitter._format_literal(dtype, real),
                    CEmitter._format_literal(dtype, imag),
                )
            )
        return tuple(twiddles)

    @staticmethod
    def _op_names(op: OpBase) -> tuple[str, ...]:
        names = (*op.input_names, *op.output_names)
        if any(not isinstance(name, str) for name in names):
            raise CodegenError(f"{op.kind} inputs/outputs must be tuple[str, ...]")
        return names

    def _build_name_map(self, model: LoweredModel) -> dict[str, str]:
        used: set[str] = set()
        name_map: dict[str, str] = {}
        constant_names = {const.name for const in model.constants}
        names = [model.name]
        names.extend(model.input_names)
        names.extend(model.output_names)
        names.extend(name for name in model.input_optional_names if name is not None)
        names.extend(name for name in model.output_optional_names if name is not None)
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
    def _map_optional_name(name_map: dict[str, str], name: str | None) -> str | None:
        if name is None:
            return None
        return name_map.get(name, name)

    @staticmethod
    def _map_op_names(op: OpBase, name_map: dict[str, str]) -> OpBase:
        return op.remap_names(name_map)

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
            input_names=tuple(name_map.get(name, name) for name in model.input_names),
            input_optional_names=tuple(
                name_map.get(name, name) if name is not None else None
                for name in model.input_optional_names
            ),
            input_shapes=model.input_shapes,
            input_dtypes=model.input_dtypes,
            input_types=model.input_types,
            output_names=tuple(name_map.get(name, name) for name in model.output_names),
            output_optional_names=tuple(
                name_map.get(name, name) if name is not None else None
                for name in model.output_optional_names
            ),
            output_shapes=model.output_shapes,
            output_dtypes=model.output_dtypes,
            output_types=model.output_types,
            constants=constants,
            ops=ops,
            node_infos=model.node_infos,
            header=model.header,
            op_context=model.op_context,
        )
        return sanitized, name_map

    def _sanitize_model_names(self, model: LoweredModel) -> LoweredModel:
        return self._sanitize_model_names_with_map(model)[0]

    @staticmethod
    def _copy_derived(
        op_context: OpContext,
        source_ops: Sequence[OpBase],
        target_ops: Sequence[OpBase],
    ) -> None:
        for source_op, target_op in zip(source_ops, target_ops):
            op_context.copy_derived(source_op, target_op)

    @staticmethod
    def _build_value_name_map(
        name_map: Mapping[str, str],
        temp_name_map: Mapping[str, str],
    ) -> dict[str, str]:
        reverse_name_map = {
            sanitized: original for original, sanitized in name_map.items()
        }
        value_name_map = dict(reverse_name_map)
        for sanitized_name, temp_name in temp_name_map.items():
            original_name = reverse_name_map.get(sanitized_name, sanitized_name)
            value_name_map[temp_name] = original_name
        return value_name_map

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

    @staticmethod
    def _sanitize_testbench_optional_inputs(
        testbench_optional_inputs: Mapping[str, bool] | None,
        name_map: Mapping[str, str],
    ) -> Mapping[str, bool] | None:
        if not testbench_optional_inputs:
            return None
        return {
            name_map.get(name, name): value
            for name, value in testbench_optional_inputs.items()
        }

    @staticmethod
    def _sanitize_testbench_outputs(
        testbench_outputs: Mapping[str, np.ndarray | list[np.ndarray]] | None,
        name_map: Mapping[str, str],
    ) -> Mapping[str, np.ndarray | list[np.ndarray]] | None:
        if not testbench_outputs:
            return None
        return {
            name_map.get(name, name): value for name, value in testbench_outputs.items()
        }

    def _load_templates(self, emit_testbench: bool) -> dict[str, Template]:
        try:
            templates = {
                "binary": self._env.get_template("binary_op.c.j2"),
                "multi_input": self._env.get_template("multi_input_op.c.j2"),
                "where": self._env.get_template("where_op.c.j2"),
                "unary": self._env.get_template("unary_op.c.j2"),
                "clip": self._env.get_template("clip_op.c.j2"),
                "cast": self._env.get_template("cast_op.c.j2"),
                "quantize_linear": self._env.get_template("quantize_linear_op.c.j2"),
                "dequantize_linear": self._env.get_template(
                    "dequantize_linear_op.c.j2"
                ),
                "dynamic_quantize_linear": self._env.get_template(
                    "dynamic_quantize_linear_op.c.j2"
                ),
                "qlinear_add": self._env.get_template("qlinear_add_op.c.j2"),
                "qlinear_mul": self._env.get_template("qlinear_mul_op.c.j2"),
                "qlinear_matmul": self._env.get_template("qlinear_matmul_op.c.j2"),
                "qlinear_avg_pool": self._env.get_template(
                    "qlinear_average_pool_op.c.j2"
                ),
                "qlinear_softmax": self._env.get_template("qlinear_softmax_op.c.j2"),
                "qlinear_conv": self._env.get_template("qlinear_conv_op.c.j2"),
                "matmul_integer": self._env.get_template("matmul_integer_op.c.j2"),
                "matmul_nbits": self._env.get_template("matmul_nbits_op.c.j2"),
                "matmul_bnb4": self._env.get_template("matmul_bnb4_op.c.j2"),
                "matmul": self._env.get_template("matmul_op.c.j2"),
                "fused_matmul": self._env.get_template("fused_matmul_op.c.j2"),
                "einsum": self._env.get_template("einsum_op.c.j2"),
                "gemm": self._env.get_template("gemm_op.c.j2"),
                "qgemm": self._env.get_template("qgemm_op.c.j2"),
                "attention": self._env.get_template("attention_op.c.j2"),
                "rotary_embedding": self._env.get_template("rotary_embedding_op.c.j2"),
                "conv": self._env.get_template("conv_op.c.j2"),
                "conv_integer": self._env.get_template("conv_integer_op.c.j2"),
                "col2im": self._env.get_template("col2im_op.c.j2"),
                "conv_transpose": self._env.get_template("conv_transpose_op.c.j2"),
                "deform_conv": self._env.get_template("deform_conv_op.c.j2"),
                "avg_pool": self._env.get_template("average_pool_op.c.j2"),
                "lp_pool": self._env.get_template("lp_pool_op.c.j2"),
                "batch_norm": self._env.get_template("batch_norm_op.c.j2"),
                "lp_norm": self._env.get_template("lp_normalization_op.c.j2"),
                "instance_norm": self._env.get_template(
                    "instance_normalization_op.c.j2"
                ),
                "group_norm": self._env.get_template("group_normalization_op.c.j2"),
                "layer_norm": self._env.get_template("layer_normalization_op.c.j2"),
                "mean_variance_norm": self._env.get_template(
                    "mean_variance_normalization_op.c.j2"
                ),
                "rms_norm": self._env.get_template("rms_normalization_op.c.j2"),
                "lrn": self._env.get_template("lrn_op.c.j2"),
                "gru": self._env.get_template("gru_op.c.j2"),
                "rnn": self._env.get_template("rnn_op.c.j2"),
                "lstm": self._env.get_template("lstm_op.c.j2"),
                "dynamic_quantize_lstm": self._env.get_template(
                    "dynamic_quantize_lstm_op.c.j2"
                ),
                "adam": self._env.get_template("adam_op.c.j2"),
                "adagrad": self._env.get_template("adagrad_op.c.j2"),
                "momentum": self._env.get_template("momentum_op.c.j2"),
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
                "nhwc_maxpool": self._env.get_template("nhwc_maxpool_op.c.j2"),
                "maxunpool": self._env.get_template("maxunpool_op.c.j2"),
                "roi_align": self._env.get_template("roi_align_op.c.j2"),
                "concat": self._env.get_template("concat_op.c.j2"),
                "concat_from_sequence": self._env.get_template(
                    "concat_from_sequence_op.c.j2"
                ),
                "compress": self._env.get_template("compress_op.c.j2"),
                "gather_elements": self._env.get_template("gather_elements_op.c.j2"),
                "gather": self._env.get_template("gather_op.c.j2"),
                "gather_block_quantized": self._env.get_template(
                    "gather_block_quantized_op.c.j2"
                ),
                "gather_nd": self._env.get_template("gather_nd_op.c.j2"),
                "scatter": self._env.get_template("scatter_op.c.j2"),
                "scatter_elements": self._env.get_template("scatter_elements_op.c.j2"),
                "scatter_nd": self._env.get_template("scatter_nd_op.c.j2"),
                "tensor_scatter": self._env.get_template("tensor_scatter_op.c.j2"),
                "transpose": self._env.get_template("transpose_op.c.j2"),
                "reshape": self._env.get_template("reshape_op.c.j2"),
                "identity": self._env.get_template("identity_op.c.j2"),
                "bernoulli": self._env.get_template("bernoulli_op.c.j2"),
                "random_uniform": self._env.get_template("random_uniform_op.c.j2"),
                "dropout": self._env.get_template("dropout_op.c.j2"),
                "eye_like": self._env.get_template("eye_like_op.c.j2"),
                "trilu": self._env.get_template("trilu_op.c.j2"),
                "tile": self._env.get_template("tile_op.c.j2"),
                "center_crop_pad": self._env.get_template("center_crop_pad_op.c.j2"),
                "pad": self._env.get_template("pad_op.c.j2"),
                "depth_to_space": self._env.get_template("depth_to_space_op.c.j2"),
                "space_to_depth": self._env.get_template("space_to_depth_op.c.j2"),
                "slice": self._env.get_template("slice_op.c.j2"),
                "slice_dynamic": self._env.get_template("slice_op_dynamic.c.j2"),
                "resize": self._env.get_template("resize_op.c.j2"),
                "grid_sample": self._env.get_template("grid_sample_op.c.j2"),
                "affine_grid": self._env.get_template("affine_grid_op.c.j2"),
                "reduce": self._env.get_template("reduce_op.c.j2"),
                "reduce_dynamic": self._env.get_template("reduce_op_dynamic.c.j2"),
                "arg_reduce": self._env.get_template("arg_reduce_op.c.j2"),
                "det": self._env.get_template("det_op.c.j2"),
                "array_feature_extractor": self._env.get_template(
                    "array_feature_extractor_op.c.j2"
                ),
                "topk": self._env.get_template("topk_op.c.j2"),
                "constant_of_shape": self._env.get_template(
                    "constant_of_shape_op.c.j2"
                ),
                "shape": self._env.get_template("shape_op.c.j2"),
                "size": self._env.get_template("size_op.c.j2"),
                "optional_has_element": self._env.get_template(
                    "optional_has_element_op.c.j2"
                ),
                "optional_get_element": self._env.get_template(
                    "optional_get_element_op.c.j2"
                ),
                "nonzero": self._env.get_template("nonzero_op.c.j2"),
                "unique": self._env.get_template("unique_op.c.j2"),
                "nonmax_suppression": self._env.get_template(
                    "nonmax_suppression_op.c.j2"
                ),
                "expand": self._env.get_template("expand_op.c.j2"),
                "cumsum": self._env.get_template("cumsum_op.c.j2"),
                "stft": self._env.get_template("stft_op.c.j2"),
                "dft": self._env.get_template("dft_op.c.j2"),
                "range": self._env.get_template("range_op.c.j2"),
                "loop_range": self._env.get_template("loop_range_op.c.j2"),
                "loop_sequence_insert": self._env.get_template(
                    "loop_sequence_insert_op.c.j2"
                ),
                "blackman_window": self._env.get_template("blackman_window_op.c.j2"),
                "hamming_window": self._env.get_template("hamming_window_op.c.j2"),
                "hann_window": self._env.get_template("hann_window_op.c.j2"),
                "mel_weight_matrix": self._env.get_template(
                    "mel_weight_matrix_op.c.j2"
                ),
                "one_hot": self._env.get_template("one_hot_op.c.j2"),
                "tfidf_vectorizer": self._env.get_template("tfidf_vectorizer_op.c.j2"),
                "string_concat": self._env.get_template("string_concat_op.c.j2"),
                "string_normalizer": self._env.get_template(
                    "string_normalizer_op.c.j2"
                ),
                "label_encoder": self._env.get_template("label_encoder_op.c.j2"),
                "string_split": self._env.get_template("string_split_op.c.j2"),
                "tree_ensemble": self._env.get_template("tree_ensemble_op.c.j2"),
                "tree_ensemble_classifier": self._env.get_template(
                    "tree_ensemble_classifier_op.c.j2"
                ),
                "split": self._env.get_template("split_op.c.j2"),
                "split_to_sequence": self._env.get_template(
                    "split_to_sequence_op.c.j2"
                ),
                "reverse_sequence": self._env.get_template("reverse_sequence_op.c.j2"),
                "sequence_at": self._env.get_template("sequence_at_op.c.j2"),
                "sequence_construct": self._env.get_template(
                    "sequence_construct_op.c.j2"
                ),
                "sequence_empty": self._env.get_template("sequence_empty_op.c.j2"),
                "sequence_erase": self._env.get_template("sequence_erase_op.c.j2"),
                "sequence_insert": self._env.get_template("sequence_insert_op.c.j2"),
                "sequence_identity": self._env.get_template(
                    "sequence_identity_op.c.j2"
                ),
                "sequence_length": self._env.get_template("sequence_length_op.c.j2"),
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
        testbench_output_format: str = "json",
        testbench_inputs: Mapping[str, tuple[float | int | bool, ...]] | None = None,
        testbench_outputs: Mapping[str, np.ndarray | list[np.ndarray]] | None = None,
        testbench_optional_inputs: Mapping[str, bool] | None = None,
        variable_dim_inputs: Mapping[int, Mapping[int, str]] | None = None,
        variable_dim_outputs: Mapping[int, Mapping[int, str]] | None = None,
    ) -> str:
        original_model = model
        model, name_map = self._sanitize_model_names_with_map(model)
        self._copy_derived(model.op_context, original_model.ops, model.ops)
        testbench_inputs = self._sanitize_testbench_inputs(testbench_inputs, name_map)
        testbench_outputs = self._sanitize_testbench_outputs(
            testbench_outputs, name_map
        )
        testbench_optional_inputs = self._sanitize_testbench_optional_inputs(
            testbench_optional_inputs, name_map
        )
        inline_constants, large_constants = self._partition_constants(model.constants)
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
            name_map,
        )
        dim_args = self._format_dim_args_prefix(dim_order)
        self._env.globals["dim_args"] = dim_args
        templates = self._load_templates(emit_testbench)
        scalar_registry = ScalarFunctionRegistry()
        fft_kernel_registry = FFTKernelRegistry(
            literal_formatter=CEmitter._format_literal
        )
        self._setup_template_resolvers(scalar_registry, fft_kernel_registry)
        testbench_template = templates.get("testbench")
        initial_name_map = self._build_value_name_map(name_map, {})
        sequence_shape_hints = {
            name_map.get(name, name): hint
            for name, hint in self._sequence_element_shapes.items()
        }
        sequence_dim_max_sizes = self._build_sequence_dim_max_sizes(
            model, sequence_shape_hints
        )
        self._emit_state = _EmitState(
            model=model,
            templates=templates,
            scalar_registry=scalar_registry,
            fft_kernel_registry=fft_kernel_registry,
            dim_args=dim_args,
            tensor_dim_names=tensor_dim_names,
            sequence_shape_hints=sequence_shape_hints,
            sequence_dim_max_sizes=sequence_dim_max_sizes,
            op_context=model.op_context,
            value_name_map=initial_name_map,
        )
        reserved_names = {
            model.name,
            *model.input_names,
            *model.output_names,
            *(name for name in model.input_optional_names if name is not None),
            *(name for name in model.output_optional_names if name is not None),
            *(const.name for const in model.constants),
        }
        temp_buffers = self._temp_buffers(model, reserved_names=reserved_names)
        temp_name_map = {
            original: buffer.name for original, buffer in temp_buffers.items()
        }
        self._copy_temp_buffer_dim_names(tensor_dim_names, temp_name_map)
        heap_temp_include = {
            "#include <stdlib.h>"
            for temp in temp_buffers.values()
            if self._temp_buffer_uses_heap(
                temp,
                tensor_dim_names.get(temp.name),
            )
        }
        resolved_ops = [self._resolve_op(op, temp_name_map) for op in model.ops]
        self._copy_derived(model.op_context, model.ops, resolved_ops)
        value_name_map = self._build_value_name_map(name_map, temp_name_map)
        self._emit_state.value_name_map = value_name_map
        self._propagate_tensor_dim_names(resolved_ops, tensor_dim_names)
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
        emitted_testbench = None
        if emit_testbench and testbench_template is not None:
            emitted_testbench = self._emit_testbench(
                model,
                testbench_template,
                testbench_output_format=testbench_output_format,
                testbench_inputs=testbench_inputs,
                testbench_outputs=testbench_outputs,
                testbench_optional_inputs=testbench_optional_inputs,
                input_dim_names=input_dim_names,
                output_dim_names=output_dim_names,
                dim_order=dim_order,
                dim_values=dim_values,
                weight_data_filename=self._weight_data_filename(model),
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
        fft_kernel_functions = fft_kernel_registry.render()
        fft_kernel_include_lines = (
            fft_kernel_registry.include_lines() if fft_kernel_functions else []
        )
        fft_kernel_includes = {
            line for line in fft_kernel_include_lines if line.startswith("#include ")
        }
        fft_kernel_preamble = [
            line
            for line in fft_kernel_include_lines
            if not line.startswith("#include ")
        ]
        testbench_math_include = set()
        if emit_testbench and self._testbench_requires_math(model, testbench_inputs):
            testbench_math_include.add("#include <math.h>")
        includes = self._collect_includes(
            original_model,
            list(original_model.ops),
            emit_testbench=emit_testbench,
            extra_includes=(
                scalar_includes
                | fft_kernel_includes
                | testbench_math_include
                | heap_temp_include
            ),
            needs_weight_loader=bool(large_constants),
        )
        sections = [
            self._emit_header_comment(model.header),
            "",
            *includes,
            "",
            self._emit_index_type_define(),
            self._emit_unused_define(),
            self._emit_node_function_define(),
            self._emit_string_max_len_define(),
            self._emit_sequence_max_len_define(),
        ]
        float8_typedefs = CEmitter._emit_float8_typedefs(
            {
                *original_model.input_dtypes,
                *original_model.output_dtypes,
                *(c.dtype for c in original_model.constants),
            }
        )
        if float8_typedefs:
            sections.append(float8_typedefs)
        if scalar_preamble:
            sections.extend(("", *scalar_preamble))
        if fft_kernel_preamble:
            sections.extend(("", *fft_kernel_preamble))
        sections.append("")
        constants_section = self._emit_constant_declarations(inline_constants)
        if constants_section:
            sections.extend((constants_section.rstrip(), ""))
        storage_declarations = self._emit_constant_storage_declarations(large_constants)
        if storage_declarations:
            sections.extend((storage_declarations.rstrip(), ""))
        constants_section = self._emit_constant_definitions(
            inline_constants, storage_prefix="const"
        )
        if constants_section:
            sections.extend((constants_section.rstrip(), ""))
        large_constants_section = self._emit_constant_storage_definitions(
            large_constants, storage_prefix=""
        )
        if large_constants_section:
            sections.extend((large_constants_section.rstrip(), ""))
        if scalar_functions:
            sections.extend(("\n".join(scalar_functions), ""))
        if fft_kernel_functions:
            sections.extend(("\n".join(fft_kernel_functions), ""))
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
        if emitted_testbench is not None:
            sections.extend(
                (
                    "",
                    emitted_testbench,
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
        testbench_output_format: str = "json",
        testbench_inputs: Mapping[str, tuple[float | int | bool, ...]] | None = None,
        testbench_outputs: Mapping[str, np.ndarray | list[np.ndarray]] | None = None,
        testbench_optional_inputs: Mapping[str, bool] | None = None,
        variable_dim_inputs: Mapping[int, Mapping[int, str]] | None = None,
        variable_dim_outputs: Mapping[int, Mapping[int, str]] | None = None,
    ) -> tuple[str, str]:
        original_model = model
        model, name_map = self._sanitize_model_names_with_map(model)
        self._copy_derived(model.op_context, original_model.ops, model.ops)
        testbench_inputs = self._sanitize_testbench_inputs(testbench_inputs, name_map)
        testbench_outputs = self._sanitize_testbench_outputs(
            testbench_outputs, name_map
        )
        testbench_optional_inputs = self._sanitize_testbench_optional_inputs(
            testbench_optional_inputs, name_map
        )
        inline_constants, large_constants = self._partition_constants(model.constants)
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
            name_map,
        )
        dim_args = self._format_dim_args_prefix(dim_order)
        self._env.globals["dim_args"] = dim_args
        templates = self._load_templates(emit_testbench)
        scalar_registry = ScalarFunctionRegistry()
        fft_kernel_registry = FFTKernelRegistry(
            literal_formatter=CEmitter._format_literal
        )
        self._setup_template_resolvers(scalar_registry, fft_kernel_registry)
        testbench_template = templates.get("testbench")
        initial_name_map = self._build_value_name_map(name_map, {})
        sequence_shape_hints = {
            name_map.get(name, name): hint
            for name, hint in self._sequence_element_shapes.items()
        }
        sequence_dim_max_sizes = self._build_sequence_dim_max_sizes(
            model, sequence_shape_hints
        )
        self._emit_state = _EmitState(
            model=model,
            templates=templates,
            scalar_registry=scalar_registry,
            fft_kernel_registry=fft_kernel_registry,
            dim_args=dim_args,
            tensor_dim_names=tensor_dim_names,
            sequence_shape_hints=sequence_shape_hints,
            sequence_dim_max_sizes=sequence_dim_max_sizes,
            op_context=model.op_context,
            value_name_map=initial_name_map,
        )
        reserved_names = {
            model.name,
            *model.input_names,
            *model.output_names,
            *(name for name in model.input_optional_names if name is not None),
            *(name for name in model.output_optional_names if name is not None),
            *(const.name for const in model.constants),
        }
        temp_buffers = self._temp_buffers(model, reserved_names=reserved_names)
        temp_name_map = {
            original: buffer.name for original, buffer in temp_buffers.items()
        }
        self._copy_temp_buffer_dim_names(tensor_dim_names, temp_name_map)
        heap_temp_include = {
            "#include <stdlib.h>"
            for temp in temp_buffers.values()
            if self._temp_buffer_uses_heap(
                temp,
                tensor_dim_names.get(temp.name),
            )
        }
        resolved_ops = [self._resolve_op(op, temp_name_map) for op in model.ops]
        self._copy_derived(model.op_context, model.ops, resolved_ops)
        value_name_map = self._build_value_name_map(name_map, temp_name_map)
        self._emit_state.value_name_map = value_name_map
        self._propagate_tensor_dim_names(resolved_ops, tensor_dim_names)
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
        emitted_testbench = None
        if emit_testbench and testbench_template is not None:
            emitted_testbench = self._emit_testbench(
                model,
                testbench_template,
                testbench_output_format=testbench_output_format,
                testbench_inputs=testbench_inputs,
                testbench_outputs=testbench_outputs,
                testbench_optional_inputs=testbench_optional_inputs,
                input_dim_names=input_dim_names,
                output_dim_names=output_dim_names,
                dim_order=dim_order,
                dim_values=dim_values,
                weight_data_filename=self._weight_data_filename(model),
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
        fft_kernel_functions = fft_kernel_registry.render()
        fft_kernel_include_lines = (
            fft_kernel_registry.include_lines() if fft_kernel_functions else []
        )
        fft_kernel_includes = {
            line for line in fft_kernel_include_lines if line.startswith("#include ")
        }
        fft_kernel_preamble = [
            line
            for line in fft_kernel_include_lines
            if not line.startswith("#include ")
        ]
        testbench_math_include = set()
        if emit_testbench and self._testbench_requires_math(model, testbench_inputs):
            testbench_math_include.add("#include <math.h>")
        includes = self._collect_includes(
            original_model,
            list(original_model.ops),
            emit_testbench=emit_testbench,
            extra_includes=(
                scalar_includes
                | fft_kernel_includes
                | testbench_math_include
                | heap_temp_include
            ),
            needs_weight_loader=bool(large_constants),
        )
        sections = [
            self._emit_header_comment(model.header),
            "",
            *includes,
            "",
            self._emit_index_type_define(),
            self._emit_unused_define(),
            self._emit_node_function_define(),
            self._emit_string_max_len_define(),
            self._emit_sequence_max_len_define(),
        ]
        float8_typedefs = CEmitter._emit_float8_typedefs(
            {
                *original_model.input_dtypes,
                *original_model.output_dtypes,
                *(c.dtype for c in original_model.constants),
            }
        )
        if float8_typedefs:
            sections.append(float8_typedefs)
        if scalar_preamble:
            sections.extend(("", *scalar_preamble))
        if fft_kernel_preamble:
            sections.extend(("", *fft_kernel_preamble))
        sections.append("")
        constants_section = self._emit_constant_declarations(inline_constants)
        if constants_section:
            sections.extend((constants_section.rstrip(), ""))
        storage_declarations = self._emit_constant_storage_declarations(large_constants)
        if storage_declarations:
            sections.extend((storage_declarations.rstrip(), ""))
        large_constants_section = self._emit_constant_storage_definitions(
            large_constants, storage_prefix=""
        )
        if large_constants_section:
            sections.extend((large_constants_section.rstrip(), ""))
        if scalar_functions:
            sections.extend(("\n".join(scalar_functions), ""))
        if fft_kernel_functions:
            sections.extend(("\n".join(fft_kernel_functions), ""))
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
        if emitted_testbench is not None:
            sections.extend(
                (
                    "",
                    emitted_testbench,
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
        data_sections.extend((self._emit_unused_define(), ""))
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

    def emit_testbench(
        self,
        model: LoweredModel,
        *,
        testbench_output_format: str = "json",
        testbench_inputs: Mapping[str, tuple[float | int | bool, ...]] | None = None,
        testbench_outputs: Mapping[str, np.ndarray | list[np.ndarray]] | None = None,
        testbench_optional_inputs: Mapping[str, bool] | None = None,
        variable_dim_inputs: Mapping[int, Mapping[int, str]] | None = None,
        variable_dim_outputs: Mapping[int, Mapping[int, str]] | None = None,
    ) -> str:
        original_model = model
        model, name_map = self._sanitize_model_names_with_map(model)
        self._copy_derived(model.op_context, original_model.ops, model.ops)
        testbench_inputs = self._sanitize_testbench_inputs(testbench_inputs, name_map)
        testbench_outputs = self._sanitize_testbench_outputs(
            testbench_outputs, name_map
        )
        testbench_optional_inputs = self._sanitize_testbench_optional_inputs(
            testbench_optional_inputs, name_map
        )
        dim_order, _in_dims, _out_dims, dim_values = self._build_variable_dim_names(
            model,
            variable_dim_inputs,
            variable_dim_outputs,
        )
        dim_args = self._format_dim_args_prefix(dim_order)
        self._env.globals["dim_args"] = dim_args
        templates = self._load_templates(True)
        testbench_template = templates.get("testbench")
        if testbench_template is None:
            raise CodegenError("Failed to load testbench template")
        scalar_registry = ScalarFunctionRegistry()
        fft_kernel_registry = FFTKernelRegistry(
            literal_formatter=CEmitter._format_literal
        )
        self._setup_template_resolvers(scalar_registry, fft_kernel_registry)
        tensor_dim_names = self._build_tensor_dim_names(model, {}, {}, name_map)
        initial_name_map = self._build_value_name_map(name_map, {})
        sequence_shape_hints = {
            name_map.get(name, name): hint
            for name, hint in self._sequence_element_shapes.items()
        }
        sequence_dim_max_sizes = self._build_sequence_dim_max_sizes(
            model, sequence_shape_hints
        )
        self._emit_state = _EmitState(
            model=model,
            templates=templates,
            scalar_registry=scalar_registry,
            fft_kernel_registry=fft_kernel_registry,
            dim_args=dim_args,
            tensor_dim_names=tensor_dim_names,
            sequence_shape_hints=sequence_shape_hints,
            sequence_dim_max_sizes=sequence_dim_max_sizes,
            op_context=model.op_context,
            value_name_map=initial_name_map,
        )
        rendered = self._emit_testbench(
            model,
            testbench_template,
            testbench_output_format=testbench_output_format,
            testbench_inputs=testbench_inputs,
            testbench_outputs=testbench_outputs,
            testbench_optional_inputs=testbench_optional_inputs,
            input_dim_names=_in_dims,
            output_dim_names=_out_dims,
            dim_order=dim_order,
            dim_values=dim_values,
            weight_data_filename=self._weight_data_filename(model),
        )
        if not rendered.endswith("\n"):
            rendered += "\n"
        return rendered

    @staticmethod
    def _emit_header_comment(header: ModelHeader) -> str:
        lines: list[str] = [header.generator, ""]
        lines.append("Codegen settings:")
        if header.codegen_settings:
            for key, value in header.codegen_settings:
                lines.append(f"  {key}: {value}")
        else:
            lines.append("  n/a")
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
        comment_lines.extend(f" * {line}" if line else " *" for line in lines)
        comment_lines.append(" */")
        return "\n".join(comment_lines)

    @staticmethod
    def _format_node_attr_value(value: object) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        if isinstance(value, str):
            return value
        if isinstance(value, (list, tuple)):
            rendered = ", ".join(
                CEmitter._format_node_attr_value(item) for item in value
            )
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
            "Inputs: " + (", ".join(node_info.inputs) if node_info.inputs else "n/a"),
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
        comment_lines.extend(f" * {line}" if line else " *" for line in lines)
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
        comment_lines.extend(f" * {line}" if line else " *" for line in lines)
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
        if CEmitter._constants_need_math(constants):
            includes.append("#include <math.h>")
        return includes

    @staticmethod
    def _constants_need_math(constants: tuple[ConstTensor, ...]) -> bool:
        float_dtypes = {
            ScalarType.F16,
            ScalarType.BF16,
            ScalarType.F32,
            ScalarType.F64,
        }
        for const in constants:
            if const.dtype not in float_dtypes:
                continue
            for value in const.data:
                if not math.isfinite(float(value)):
                    return True
        return False

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
            ScalarFunction.BINARIZER,
            ScalarFunction.CEIL,
            ScalarFunction.CELU,
            ScalarFunction.COS,
            ScalarFunction.COSH,
            ScalarFunction.DIV,
            ScalarFunction.ELU,
            ScalarFunction.ERF,
            ScalarFunction.EXP,
            ScalarFunction.FLOOR,
            ScalarFunction.FMAX,
            ScalarFunction.FMIN,
            ScalarFunction.GELU,
            ScalarFunction.GELU_TANH,
            ScalarFunction.HARDSIGMOID,
            ScalarFunction.HARDSWISH,
            ScalarFunction.LLRINT,
            ScalarFunction.LOG,
            ScalarFunction.FMOD,
            ScalarFunction.REMAINDER,
            ScalarFunction.LEAKY_RELU,
            ScalarFunction.MISH,
            ScalarFunction.MUL,
            ScalarFunction.NEARBYINT,
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
            if dtype in {ScalarType.BF16, ScalarType.F32, ScalarType.F64}:
                scalar_function = (
                    ScalarFunction.MAXIMUM
                    if function == ScalarFunction.MAXIMUM
                    else ScalarFunction.MINIMUM
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

    def _rnn_activation_function_name(
        self,
        kind: int,
        alpha: float,
        beta: float,
        dtype: ScalarType,
        registry: ScalarFunctionRegistry,
    ) -> str:
        spec = _LSTM_ACTIVATION_SPECS.get(kind)
        if spec is None:
            raise CodegenError(f"Unsupported RNN activation kind for codegen: {kind}")
        function, param_count = spec
        if param_count == 0:
            params = ()
        elif param_count == 1:
            params = (alpha,)
        else:
            params = (alpha, beta)
        name = self._scalar_function_name(function, dtype, registry, params=params)
        if name is None:
            raise CodegenError(
                f"Failed to resolve scalar function for RNN activation kind {kind}"
            )
        return name

    @staticmethod
    def _collect_includes(
        model: LoweredModel,
        resolved_ops: list[OpBase],
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

        model_dtypes: set[ScalarType] = {
            *model.input_dtypes,
            *model.output_dtypes,
            *(const.dtype for const in model.constants),
        }

        ctx = model.op_context
        for op in resolved_ops:
            includes.update(op.required_includes(ctx))
            model_dtypes.update(op.extra_model_dtypes(ctx))

        model_dtypes.update(op.resolved_output_dtype(ctx) for op in resolved_ops)
        if CEmitter._needs_stdint(model_dtypes, (), has_resize=False):
            includes.add("#include <stdint.h>")
        if ScalarType.BOOL in model_dtypes:
            includes.add("#include <stdbool.h>")
        if CEmitter._constants_need_math(model.constants):
            includes.add("#include <math.h>")
        ordered_includes = (
            "#include <stdint.h>",
            "#include <stdio.h>",
            "#include <stddef.h>",
            "#include <stdbool.h>",
            "#include <stdlib.h>",
            "#include <math.h>",
            "#include <float.h>",
            "#include <limits.h>",
            "#include <ctype.h>",
            "#include <string.h>",
            "#include <strings.h>",
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
    def _emit_float8_typedefs(model_dtypes: set[ScalarType]) -> str | None:
        _FLOAT8_TYPEDEFS = {
            ScalarType.F4E2M1: "typedef uint8_t emx_float4e2m1_t;",
            ScalarType.F8E4M3FN: "typedef uint8_t emx_float8e4m3fn_t;",
            ScalarType.F8E4M3FNUZ: "typedef uint8_t emx_float8e4m3fnuz_t;",
            ScalarType.F8E5M2: "typedef uint8_t emx_float8e5m2_t;",
            ScalarType.F8E5M2FNUZ: "typedef uint8_t emx_float8e5m2fnuz_t;",
            ScalarType.F8E8M0FNU: "typedef uint8_t emx_float8e8m0fnu_t;",
        }
        lines = [
            _FLOAT8_TYPEDEFS[dt]
            for dt in sorted(_FLOAT8_TYPEDEFS, key=lambda d: d.value)
            if dt in model_dtypes
        ]
        return "\n".join(lines) if lines else None

    @staticmethod
    def _emit_unused_define() -> str:
        return "\n".join(
            (
                "#ifndef EMX_UNUSED",
                "#if defined(__GNUC__) || defined(__clang__)",
                "#define EMX_UNUSED __attribute__((unused))",
                "#else",
                "#define EMX_UNUSED",
                "#endif",
                "#endif",
            )
        )

    @staticmethod
    def _emit_node_function_define() -> str:
        return "\n".join(
            (
                "#ifndef EMX_NODE_FN",
                "#define EMX_NODE_FN static inline",
                "#endif",
            )
        )

    @staticmethod
    def _emit_string_max_len_define() -> str:
        return "\n".join(
            (
                "#ifndef EMX_STRING_MAX_LEN",
                "#define EMX_STRING_MAX_LEN 256",
                "#endif",
            )
        )

    @staticmethod
    def _emit_sequence_max_len_define() -> str:
        return "\n".join(
            (
                "#ifndef EMX_SEQUENCE_MAX_LEN",
                "#define EMX_SEQUENCE_MAX_LEN 32",
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
            | DequantizeLinearOp
            | QLinearAddOp
            | QLinearMulOp
            | QLinearMatMulOp
            | QLinearSoftmaxOp
            | MatMulOp
            | EinsumOp
            | GemmOp
            | AttentionOp
            | ConvOp
            | ConvIntegerOp
            | ConvTransposeOp
            | Col2ImOp
            | DeformConvOp
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
            | GruOp
            | RnnOp
            | LstmOp
            | SoftmaxOp
            | LogSoftmaxOp
            | HardmaxOp
            | NegativeLogLikelihoodLossOp
            | SoftmaxCrossEntropyLossOp
            | MaxPoolOp
            | RoiAlignOp
            | ConcatOp
            | ConcatFromSequenceOp
            | CompressOp
            | GatherElementsOp
            | GatherOp
            | GatherNDOp
            | TransposeOp
            | ReshapeOp
            | IdentityOp
            | EyeLikeOp
            | TriluOp
            | TileOp
            | CenterCropPadOp
            | DepthToSpaceOp
            | SpaceToDepthOp
            | SliceOp
            | ResizeOp
            | GridSampleOp
            | AffineGridOp
            | ReduceOp
            | DetOp
            | ArgReduceOp
            | TopKOp
            | ConstantOfShapeOp
            | ShapeOp
            | SizeOp
            | NonZeroOp
            | UniqueOp
            | NonMaxSuppressionOp
            | ExpandOp
            | CumSumOp
            | DFTOp
            | RangeOp
            | BlackmanWindowOp
            | HammingWindowOp
            | HannWindowOp
            | OneHotOp
            | SplitOp
            | SequenceAtOp
            | SequenceConstructOp
            | SequenceEraseOp
            | SequenceInsertOp
            | SequenceLengthOp
        ],
        temp_buffers: tuple[TempBuffer, ...],
        *,
        dim_order: Sequence[str],
        input_dim_names: Mapping[int, Mapping[int, CodegenDim]],
        output_dim_names: Mapping[int, Mapping[int, CodegenDim]],
    ) -> str:
        signature = self._entrypoint_signature(
            model,
            dim_order=dim_order,
            input_dim_names=input_dim_names,
            output_dim_names=output_dim_names,
        )
        lines = [f"void {model.name}({signature}) {{"]
        heap_temp_names: list[str] = []
        for temp in temp_buffers:
            c_type = temp.dtype.c_type
            dim_names = self.dim_names_for(temp.name)
            if temp.is_sequence:
                suffix = self._param_array_suffix(
                    temp.shape,
                    dim_names,
                    dtype=temp.dtype,
                )
                storage = (
                    "static "
                    if not dim_names
                    and self._buffer_size_bytes(
                        temp.shape,
                        temp.dtype,
                        dim_names=dim_names,
                        is_sequence=temp.is_sequence,
                    )
                    > self._large_temp_threshold_bytes
                    else ""
                )
                lines.append(
                    f"    {storage}{c_type} {temp.name}[EMX_SEQUENCE_MAX_LEN]{suffix};"
                )
                lines.append(f"    idx_t {temp.name}__count = 0;")
                for axis in self._sequence_dynamic_axes(temp.name):
                    lines.append(
                        "    "
                        f"idx_t {self._sequence_dim_array_name(temp.name, axis)}"
                        "[EMX_SEQUENCE_MAX_LEN] = {0};"
                    )
            elif self._temp_buffer_uses_heap(temp, dim_names):
                lines.extend(
                    f"    {line}"
                    for line in self._heap_temp_buffer_lines(temp, dim_names)
                )
                heap_temp_names.append(temp.name)
            else:
                storage = ""
                if (
                    not dim_names
                    and self._temp_buffer_size_bytes(temp)
                    > self._large_temp_threshold_bytes
                ):
                    storage = "static "
                lines.append(
                    f"    {storage}{c_type} {temp.name}{self._param_array_suffix(temp.shape, dim_names, dtype=temp.dtype)};"
                )
        sequence_temp_count_names = {
            f"{temp.name}__count" for temp in temp_buffers if temp.is_sequence
        }
        for index, op in enumerate(resolved_ops):
            op_name = self._op_function_name(model, index)
            op_args = list(op.call_args())
            if isinstance(op, SequenceIdentityOp):
                op_args = [
                    op.input_sequence,
                    f"{op.input_sequence}__count",
                    op.output_sequence,
                    f"{op.output_sequence}__count",
                    *self._sequence_dim_arg_names(op.input_sequence),
                    *self._sequence_dim_arg_names(op.output_sequence),
                    *(
                        [op.input_present, op.output_present]
                        if op.input_present is not None
                        and op.output_present is not None
                        else []
                    ),
                ]
            elif isinstance(op, SequenceInsertOp):
                op_args = [
                    op.input_sequence,
                    f"{op.input_sequence}__count",
                    *self._sequence_dim_arg_names(op.input_sequence),
                    op.tensor,
                    *([op.position] if op.position is not None else []),
                    op.output_sequence,
                    f"{op.output_sequence}__count",
                    *self._sequence_dim_arg_names(op.output_sequence),
                ]
            elif isinstance(op, SequenceEraseOp):
                op_args = [
                    op.input_sequence,
                    f"{op.input_sequence}__count",
                    *self._sequence_dim_arg_names(op.input_sequence),
                    *([op.position] if op.position is not None else []),
                    op.output_sequence,
                    f"{op.output_sequence}__count",
                    *self._sequence_dim_arg_names(op.output_sequence),
                ]
            elif isinstance(op, LoopSequenceMapOp):
                op_args = [op.trip_count, op.cond]
                for name in op.input_sequences:
                    op_args.extend(
                        [name, f"{name}__count", *self._sequence_dim_arg_names(name)]
                    )
                for name in op.input_tensors:
                    op_args.append(name)
                for name in op.output_sequences:
                    op_args.extend(
                        [
                            name,
                            f"{name}__count",
                            *self._sequence_dim_arg_names(name),
                        ]
                    )
            elif isinstance(op, SplitToSequenceOp):
                op_args = [op.input0]
                if op.split is not None:
                    op_args.append(op.split)
                op_args.extend(
                    [
                        op.output_sequence,
                        f"{op.output_sequence}__count",
                        *self._sequence_dim_arg_names(op.output_sequence),
                    ]
                )
            elif isinstance(op, LoopSequenceInsertOp):
                op_args = [
                    op.trip_count,
                    op.cond,
                    op.input_sequence,
                    f"{op.input_sequence}__count",
                ]
                if op.input_sequence_present is not None:
                    op_args.append(op.input_sequence_present)
                op_args.extend(
                    [
                        op.output_sequence,
                        f"{op.output_sequence}__count",
                        *self._sequence_dim_arg_names(op.output_sequence),
                    ]
                )
            output_count_names: set[str] = set()
            if isinstance(op, SequenceIdentityOp):
                output_count_names.add(f"{op.output_sequence}__count")
            elif isinstance(op, LoopSequenceMapOp):
                output_count_names.update(
                    f"{name}__count" for name in op.output_sequences
                )
            elif isinstance(
                op,
                (
                    SequenceConstructOp,
                    SequenceEmptyOp,
                    SequenceEraseOp,
                    SequenceInsertOp,
                    IfOptionalSequenceConstOp,
                    SplitToSequenceOp,
                    LoopSequenceInsertOp,
                ),
            ):
                output_count_names.add(f"{op.output_sequence}__count")
            if output_count_names:
                for arg_index, arg in enumerate(op_args):
                    if arg in output_count_names and arg in sequence_temp_count_names:
                        op_args[arg_index] = f"&{arg}"
            args = [*dim_order, *op_args]
            call = ", ".join(args)
            lines.append(f"    {op_name}({call});")
        for temp_name in reversed(heap_temp_names):
            lines.append(f"    free({temp_name});")
        lines.append("}")
        return "\n".join(lines)

    def _entrypoint_signature(
        self,
        model: LoweredModel,
        *,
        dim_order: Sequence[str],
        input_dim_names: Mapping[int, Mapping[int, CodegenDim]],
        output_dim_names: Mapping[int, Mapping[int, CodegenDim]],
    ) -> str:
        params: list[str] = []
        optional_flags = self._optional_input_flag_map(model)
        output_optional_flags = self._optional_output_flag_map(model)
        if dim_order:
            params.extend(self._format_dim_args(dim_order))
        for index, (name, shape, dtype, value_type) in enumerate(
            zip(
                model.input_names,
                model.input_shapes,
                model.input_dtypes,
                model.input_types,
            )
        ):
            if isinstance(value_type, SequenceType):
                sequence_shape = self._sequence_storage_shape(name)
                elem_suffix = self._param_array_suffix(
                    sequence_shape,
                    dtype=dtype,
                )
                params.append(
                    f"const {dtype.c_type} {name}[EMX_SEQUENCE_MAX_LEN]{elem_suffix}"
                )
                params.append(f"idx_t {name}__count")
                for axis in self._sequence_dynamic_axes(name):
                    params.append(
                        "const idx_t "
                        f"{self._sequence_dim_array_name(name, axis)}[EMX_SEQUENCE_MAX_LEN]"
                    )
                optional_flag = optional_flags.get(name)
                if optional_flag is not None:
                    params.append(f"_Bool {optional_flag}")
                continue
            params.append(
                f"const {dtype.c_type} {name}"
                f"{self._param_array_suffix(shape, input_dim_names.get(index), use_restrict=True, dtype=dtype)}"
            )
            optional_flag = optional_flags.get(name)
            if optional_flag is not None:
                params.append(f"_Bool {optional_flag}")
        for index, (name, shape, dtype, value_type) in enumerate(
            zip(
                model.output_names,
                model.output_shapes,
                model.output_dtypes,
                model.output_types,
            )
        ):
            if isinstance(value_type, SequenceType):
                sequence_shape = self._sequence_storage_shape(name)
                elem_suffix = self._param_array_suffix(
                    sequence_shape,
                    dtype=dtype,
                )
                params.append(
                    f"{dtype.c_type} {name}[EMX_SEQUENCE_MAX_LEN]{elem_suffix}"
                )
                params.append(f"idx_t *{name}__count")
                for axis in self._sequence_dynamic_axes(name):
                    params.append(
                        f"idx_t {self._sequence_dim_array_name(name, axis)}[EMX_SEQUENCE_MAX_LEN]"
                    )
                optional_flag = output_optional_flags.get(name)
                if optional_flag is not None:
                    params.append(f"_Bool *{optional_flag}")
                continue
            params.append(
                f"{dtype.c_type} {name}"
                f"{self._param_array_suffix(shape, output_dim_names.get(index), use_restrict=True, dtype=dtype)}"
            )
        return ", ".join(params)

    def emit_testbench_declarations(
        self,
        model: LoweredModel,
        *,
        variable_dim_inputs: Mapping[int, Mapping[int, str]] | None = None,
        variable_dim_outputs: Mapping[int, Mapping[int, str]] | None = None,
    ) -> str:
        original_model = model
        model, name_map = self._sanitize_model_names_with_map(model)
        self._copy_derived(model.op_context, original_model.ops, model.ops)
        dim_order, input_dim_names, output_dim_names, _dim_values = (
            self._build_variable_dim_names(
                model, variable_dim_inputs, variable_dim_outputs
            )
        )
        signature = self._entrypoint_signature(
            model,
            dim_order=dim_order,
            input_dim_names=input_dim_names,
            output_dim_names=output_dim_names,
        )
        return "\n".join(
            [
                f"_Bool {model.name}_load(const char *path);",
                f"void {model.name}({signature});",
            ]
        )

    @staticmethod
    def _buffer_size_bytes(
        shape: tuple[int, ...],
        dtype: ScalarType,
        *,
        dim_names: Mapping[int, CodegenDim] | None = None,
        is_sequence: bool = False,
    ) -> int:
        element_count = 1
        for dim in CEmitter._shape_with_expected_sizes(shape, dim_names):
            element_count *= dim
        if is_sequence:
            element_count *= 32
        return element_count * dtype.np_dtype.itemsize

    def _local_array_storage(
        self,
        shape: tuple[int, ...],
        dtype: ScalarType,
        *,
        dim_names: Mapping[int, CodegenDim] | None = None,
        is_sequence: bool = False,
    ) -> str:
        if dim_names:
            return ""
        if (
            self._buffer_size_bytes(
                shape,
                dtype,
                dim_names=dim_names,
                is_sequence=is_sequence,
            )
            > self._large_temp_threshold_bytes
        ):
            return "static "
        return ""

    @staticmethod
    def _temp_buffer_size_bytes(temp: TempBuffer) -> int:
        return CEmitter._buffer_size_bytes(
            temp.shape, temp.dtype, is_sequence=temp.is_sequence
        )

    def _temp_buffer_uses_heap(
        self,
        temp: TempBuffer,
        dim_names: Mapping[int, CodegenDim] | None = None,
    ) -> bool:
        if temp.is_sequence:
            return False
        return (
            self._buffer_size_bytes(
                temp.shape,
                temp.dtype,
                dim_names=dim_names,
                is_sequence=temp.is_sequence,
            )
            > self._large_temp_threshold_bytes
        )

    def _heap_temp_buffer_lines(
        self,
        temp: TempBuffer,
        dim_names: Mapping[int, CodegenDim] | None = None,
    ) -> list[str]:
        dim_names = dim_names or {}
        shape = self._codegen_shape(temp.shape)
        shape_exprs = [
            self._dim_expr(dim_names.get(index, dim)) for index, dim in enumerate(shape)
        ]
        suffix = ""
        if len(shape_exprs) > 1:
            suffix = "".join(f"[{expr}]" for expr in shape_exprs[1:])
        if temp.dtype == ScalarType.STRING:
            suffix += "[EMX_STRING_MAX_LEN]"
        c_type = temp.dtype.c_type
        if len(shape_exprs) <= 1:
            decl = f"{c_type} *{temp.name}"
            alloc = (
                f"malloc(sizeof(*{temp.name}) * {shape_exprs[0] if shape_exprs else 1})"
            )
        else:
            decl = f"{c_type} (*{temp.name}){suffix}"
            alloc = f"malloc(sizeof(*{temp.name}) * {shape_exprs[0]})"
        return [
            f"{decl} = {alloc};",
            f"if ({temp.name} == NULL) {{",
            "        return;",
            "    }",
        ]

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
        sequence_output_names = {
            output_name
            for op in model.ops
            if isinstance(
                op,
                (
                    SequenceConstructOp,
                    SequenceEmptyOp,
                    SequenceEraseOp,
                    SequenceInsertOp,
                    SplitToSequenceOp,
                    LoopSequenceInsertOp,
                ),
            )
            for output_name, _, _ in self._op_outputs(op)
        }
        if len(intermediates) == 1:
            name, shape, dtype = intermediates[0]
            temp_name = allocate_temp_name(f"tmp0_{name}")
            return {
                name: TempBuffer(
                    name=temp_name,
                    shape=shape,
                    dtype=dtype,
                    is_sequence=name in sequence_output_names,
                )
            }
        return {
            name: TempBuffer(
                name=allocate_temp_name(f"tmp{index}_{name}"),
                shape=shape,
                dtype=dtype,
                is_sequence=name in sequence_output_names,
            )
            for index, (name, shape, dtype) in enumerate(intermediates)
        }

    @staticmethod
    def _resolve_op(op: OpBase, temp_map: dict[str, str]) -> OpBase:
        return op.remap_names(temp_map)

    def render_op(self, op: OpBase, ctx: EmitContext) -> str:
        return op.emit(self, ctx)

    def require_emit_state(self) -> _EmitState:
        if self._emit_state is None:
            raise CodegenError("Emitter state not initialized")
        return self._emit_state

    def op_function_name(self, model: LoweredModel, index: int) -> str:
        return self._op_function_name(model, index)

    def with_node_comment(
        self,
        model: LoweredModel,
        index: int,
        rendered: str,
    ) -> str:
        node_info = model.node_infos[index]
        node_comment = CEmitter._emit_node_comment(node_info, index)
        return f"{node_comment}\n{_format_c_indentation(rendered)}"

    def shared_param_map(
        self,
        items: list[tuple[str, str | None]],
    ) -> dict[str, str]:
        return self._shared_param_map(items)

    def ctx_shape(self, name: str) -> tuple[int, ...]:
        return self._ctx_shape(name)

    def ctx_dtype(self, name: str) -> ScalarType:
        return self._ctx_dtype(name)

    def derived(self, op: OpBase, key: str) -> object:
        return self._derived(op, key)

    def param_array_suffix(
        self,
        shape: tuple[int, ...],
        dim_names: Mapping[int, str] | None = None,
        *,
        dtype: ScalarType | None = None,
    ) -> str:
        return self._param_array_suffix(shape, dim_names, dtype=dtype)

    def build_param_decls(
        self,
        params: list[tuple[str | None, str, str, bool]],
    ) -> tuple[str, ...]:
        return self._build_param_decls(params)

    def accumulation_dtype(self, dtype: ScalarType) -> ScalarType:
        return self._accumulation_dtype(dtype)

    def format_literal(self, dtype: ScalarType, value: float | int | bool) -> str:
        return CEmitter._format_literal(dtype, value)

    def format_floating(self, value: float, dtype: ScalarType) -> str:
        return CEmitter._format_floating(value, dtype)

    def dim_names_for(self, name: str) -> Mapping[int, str]:
        if self._emit_state is None:
            return {}
        tensor_dim_names = self._emit_state.tensor_dim_names or {}
        return tensor_dim_names.get(name, {})

    def dim_args_str(self) -> str:
        if self._emit_state is None:
            return ""
        return self._emit_state.dim_args

    def rnn_activation_function_name(
        self, kind: str, alpha: float, beta: float, dtype: ScalarType
    ) -> str:
        if self._emit_state is None:
            raise CodegenError("Emitter state not initialized")
        sr = self._emit_state.scalar_registry
        if sr is None:
            raise CodegenError(
                "Scalar function registry is required for RNN activation codegen."
            )
        return self._rnn_activation_function_name(kind, alpha, beta, dtype, sr)

    def scalar_registry(self):
        if self._emit_state is None:
            return None
        return self._emit_state.scalar_registry

    @property
    def replicate_ort_bugs(self) -> bool:
        return self._replicate_ort_bugs

    def op_output_dtype(self, op: OpBase) -> ScalarType:
        return self._op_output_dtype(op)

    def unique_param_map(
        self, items: list[tuple[str, str | None]]
    ) -> dict[str, str | None]:
        return self._unique_param_map(items)

    def optional_input_flag_map(self, model) -> dict[str, str]:
        return self._optional_input_flag_map(model)

    def ctx_sequence_elem_type(self, name: str):
        return self._ctx_sequence_elem_type(name)

    def sequence_storage_shape(self, name: str) -> tuple[int, ...]:
        return self._sequence_storage_shape(name)

    def sequence_dynamic_axes(self, name: str) -> tuple[int, ...]:
        return self._sequence_dynamic_axes(name)

    def sequence_dim_array_name(self, name: str, axis: int) -> str:
        return self._sequence_dim_array_name(name, axis)

    def format_value(self, value: float | int | bool, dtype: ScalarType) -> str:
        return self._format_value(value, dtype)

    def dft_stockham_stage_plan(self, fft_length: int):
        return self._dft_stockham_stage_plan(fft_length)

    def dft_twiddle_table(self, fft_length: int, *, inverse: bool, dtype: ScalarType):
        return self._dft_twiddle_table(fft_length, inverse=inverse, dtype=dtype)

    def format_c_string_literal(self, value: str) -> str:
        return self._format_c_string_literal(value)

    def format_double(self, value: float) -> str:
        return self._format_double(value)

    def emit_initializer_lines(self, values, shape: tuple[int, ...]) -> list[str]:
        return self._emit_initializer_lines(values, shape)

    def index_expr(self, shape: tuple[int, ...], loop_vars: tuple[str, ...]) -> str:
        return self._index_expr(shape, loop_vars)

    def format_c_indentation(self, text: str) -> str:
        return _format_c_indentation(text)

    def scalar_fn(
        self,
        function: ScalarFunction,
        dtype: ScalarType,
        params: tuple[float, ...] = (),
    ) -> str | None:
        if self._emit_state is None:
            raise CodegenError("Emitter state not initialized")
        sr = self._emit_state.scalar_registry
        if sr is None:
            return None
        return CEmitter._scalar_function_name(function, dtype, sr, params=params)

    def maybe_derived(self, op: OpBase, key: str) -> object | None:
        return self._maybe_derived(op, key)

    def emit_generic_op(self, op: OpBase, ctx: EmitContext) -> str:
        if self._emit_state is None:
            if isinstance(op, TreeEnsembleClassifierOp):
                return ""
            raise CodegenError("Emitter state not initialized")
        state = self._emit_state
        return self._render_op(state, op, ctx.op_index)

    def _render_reduce_like_op(
        self,
        *,
        state: _EmitState,
        op: OpBase,
        op_name: str,
        c_type: str,
        zero_literal: str,
        min_literal: str,
        max_literal: str,
    ) -> str | None:
        model = state.model
        templates = state.templates
        if isinstance(op, ReduceOp) and op.axes_input is None:
            input_shape = self._ctx_shape(op.input0)
            output_shape_raw = self._ctx_shape(op.output)
            axes = self._derived(op, "axes")
            output_dtype = self._ctx_dtype(op.output)
            acc_dtype = self._accumulation_dtype(output_dtype)
            params = self._shared_param_map(
                [("input0", op.input0), ("output", op.output)]
            )
            output_shape = CEmitter._codegen_shape(output_shape_raw)
            output_loop_vars = CEmitter._loop_vars(output_shape)
            if not input_shape:
                reduce_loop_vars = ("r0",)
                reduce_dims = (1,)
            else:
                reduce_loop_vars = tuple(f"r{idx}" for idx in range(len(axes)))
                reduce_dims = tuple(input_shape[axis] for axis in axes)
            if not input_shape:
                input_indices = [reduce_loop_vars[0]]
            elif op.keepdims:
                input_indices = [
                    (
                        reduce_loop_vars[axes.index(axis)]
                        if axis in axes
                        else output_loop_vars[axis]
                    )
                    for axis in range(len(input_shape))
                ]
            else:
                kept_axes = [
                    axis for axis in range(len(input_shape)) if axis not in axes
                ]
                input_indices = [
                    (
                        reduce_loop_vars[axes.index(axis)]
                        if axis in axes
                        else output_loop_vars[kept_axes.index(axis)]
                    )
                    for axis in range(len(input_shape))
                ]
            input_index_expr = "".join(f"[{var}]" for var in input_indices)
            output_index_expr = "".join(f"[{var}]" for var in output_loop_vars)
            value_expr = f"{params['input0']}{input_index_expr}"
            update_expr = None
            init_literal = None
            final_expr = "acc"
            fabs_fn = self._scalar_function_name(
                ScalarFunction.ABS, acc_dtype, self._emit_state.scalar_registry
            )
            exp_fn = self._scalar_function_name(
                ScalarFunction.EXP, acc_dtype, self._emit_state.scalar_registry
            )
            log_fn = self._scalar_function_name(
                ScalarFunction.LOG, acc_dtype, self._emit_state.scalar_registry
            )
            sqrt_fn = self._scalar_function_name(
                ScalarFunction.SQRT, acc_dtype, self._emit_state.scalar_registry
            )
            if op.reduce_kind == "sum":
                init_literal = CEmitter._format_literal(acc_dtype, 0)
                update_expr = f"acc += {value_expr};"
            elif op.reduce_kind == "mean":
                count_literal = CEmitter._format_literal(acc_dtype, op.reduce_count)
                init_literal = CEmitter._format_literal(acc_dtype, 0)
                update_expr = f"acc += {value_expr};"
                final_expr = f"acc / {count_literal}"
            elif op.reduce_kind == "max":
                init_literal = acc_dtype.min_literal
                update_expr = f"if ({value_expr} > acc) acc = {value_expr};"
            elif op.reduce_kind == "min":
                init_literal = acc_dtype.max_literal
                update_expr = f"if ({value_expr} < acc) acc = {value_expr};"
            elif op.reduce_kind == "prod":
                init_literal = CEmitter._format_literal(acc_dtype, 1)
                update_expr = f"acc *= {value_expr};"
            elif op.reduce_kind == "l1":
                init_literal = CEmitter._format_literal(acc_dtype, 0)
                update_expr = f"acc += {fabs_fn}({value_expr});"
            elif op.reduce_kind == "l2":
                init_literal = CEmitter._format_literal(acc_dtype, 0)
                update_expr = f"acc += {value_expr} * {value_expr};"
                final_expr = f"{sqrt_fn}(acc)"
            elif op.reduce_kind == "logsum":
                init_literal = CEmitter._format_literal(acc_dtype, 0)
                update_expr = f"acc += {value_expr};"
                final_expr = f"{log_fn}(acc)"
            elif op.reduce_kind == "logsumexp":
                init_literal = CEmitter._format_literal(acc_dtype, 0)
                update_expr = f"acc += {exp_fn}({value_expr});"
                final_expr = f"{log_fn}(acc)"
            elif op.reduce_kind == "sumsquare":
                init_literal = CEmitter._format_literal(acc_dtype, 0)
                update_expr = f"acc += {value_expr} * {value_expr};"
            else:
                raise CodegenError(f"Unsupported reduce kind {op.reduce_kind}")
            input_suffix = self._param_array_suffix(input_shape)
            output_suffix = self._param_array_suffix(output_shape_raw)
            param_decls = self._build_param_decls(
                [
                    (params["input0"], c_type, input_suffix, True),
                    (params["output"], c_type, output_suffix, False),
                ]
            )
            return (
                templates["reduce"]
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
                    output_loop_vars=output_loop_vars,
                    reduce_loop_vars=reduce_loop_vars,
                    reduce_dims=reduce_dims,
                    output_index_expr=output_index_expr,
                    init_literal=init_literal,
                    acc_type=acc_dtype.c_type,
                    acc_zero_literal=CEmitter._format_literal(acc_dtype, 0),
                    update_expr=update_expr,
                    final_expr=final_expr,
                    use_kahan=False,
                    kahan_value_expr=None,
                )
                .rstrip()
            )
        if isinstance(op, ReduceOp):
            name_params = self._shared_param_map(
                [
                    ("input0", op.input0),
                    ("axes_input", op.axes_input),
                    ("output", op.output),
                ]
            )
            input_shape_raw = self._ctx_shape(op.input0)
            output_shape_raw = self._ctx_shape(op.output)
            output_shape = CEmitter._codegen_shape(output_shape_raw)
            output_loop_vars = CEmitter._loop_vars(output_shape)
            input_shape = CEmitter._codegen_shape(input_shape_raw)
            input_loop_vars = CEmitter._loop_vars(input_shape)
            axes_shape = (
                self._ctx_shape(op.axes_input) if op.axes_input is not None else ()
            )
            axes_count = 1
            for dim in axes_shape:
                if dim == 0:
                    axes_count = 0
                    break
                axes_count *= dim
            axes_c_type = (
                self._ctx_dtype(op.axes_input).c_type
                if op.axes_input is not None
                else ScalarType.I64.c_type
            )
            input_indices = "".join(f"[{var}]" for var in input_loop_vars)
            output_indices = "".join(
                f"[out_indices[{idx}]]" for idx in range(len(output_shape))
            )
            output_loop_index_expr = "".join(f"[{var}]" for var in output_loop_vars)
            value_expr = f"{name_params['input0']}{input_indices}"
            update_expr = None
            init_literal = None
            post_expr = None
            reduce_dtype = self._ctx_dtype(op.output)
            fabs_fn = self._scalar_function_name(
                ScalarFunction.ABS, reduce_dtype, self._emit_state.scalar_registry
            )
            exp_fn = self._scalar_function_name(
                ScalarFunction.EXP, reduce_dtype, self._emit_state.scalar_registry
            )
            log_fn = self._scalar_function_name(
                ScalarFunction.LOG, reduce_dtype, self._emit_state.scalar_registry
            )
            sqrt_fn = self._scalar_function_name(
                ScalarFunction.SQRT, reduce_dtype, self._emit_state.scalar_registry
            )
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
                init_literal = CEmitter._format_literal(reduce_dtype, 1)
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
                raise CodegenError(f"Unsupported reduce kind {op.reduce_kind}")
            input_suffix = self._param_array_suffix(input_shape_raw)
            output_suffix = self._param_array_suffix(output_shape_raw)
            axes_suffix = self._param_array_suffix(axes_shape) if axes_shape else ""
            params = self._build_param_decls(
                [
                    (name_params["input0"], c_type, input_suffix, True),
                    (
                        (
                            name_params["axes_input"],
                            axes_c_type,
                            axes_suffix,
                            True,
                        )
                        if name_params["axes_input"]
                        else (None, "", "", True)
                    ),
                    (name_params["output"], c_type, output_suffix, False),
                ]
            )
            return (
                templates["reduce_dynamic"]
                .render(
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
                        f"reduce_mask_{idx}" for idx in range(len(input_shape))
                    ),
                    output_rank=len(output_shape),
                )
                .rstrip()
            )
        return None

    def _render_op(
        self,
        state: _EmitState,
        op: OpBase,
        index: int,
    ) -> str:
        model = state.model
        dtype = self._op_output_dtype(op)
        c_type = dtype.c_type
        zero_literal = dtype.zero_literal
        min_literal = dtype.min_literal
        max_literal = dtype.max_literal
        node_info = model.node_infos[index]
        node_comment = CEmitter._emit_node_comment(node_info, index)
        op_name = self._op_function_name(model, index)
        tensor_dim_names = state.tensor_dim_names or {}

        def _dim_names_for(name: str) -> Mapping[int, str]:
            return tensor_dim_names.get(name, {})

        def with_node_comment(rendered: str) -> str:
            return f"{node_comment}\n{_format_c_indentation(rendered)}"

        reduce_like_rendered = self._render_reduce_like_op(
            state=state,
            op=op,
            op_name=op_name,
            c_type=c_type,
            zero_literal=zero_literal,
            min_literal=min_literal,
            max_literal=max_literal,
        )
        if reduce_like_rendered is not None:
            return with_node_comment(reduce_like_rendered)

        raise CodegenError(f"Unsupported op for rendering: {type(op).__name__}")

    def _op_inputs(
        self,
        op: OpBase,
    ) -> tuple[tuple[str, tuple[int, ...]], ...]:
        return op.c_op_inputs(self)

    def _propagate_tensor_dim_names(
        self,
        resolved_ops: Sequence[OpBase],
        tensor_dim_names: dict[str, dict[int, str]],
    ) -> None:
        for op in resolved_ops:
            for output_name, output_shape, _ in self._op_outputs(op):
                propagated = dict(tensor_dim_names.get(output_name, {}))
                if isinstance(op, TransposeOp):
                    input_dim_names = tensor_dim_names.get(op.input0, {})
                    for output_axis, input_axis in enumerate(op.perm):
                        dim_ref = input_dim_names.get(input_axis)
                        if dim_ref is not None:
                            propagated.setdefault(output_axis, dim_ref)
                for input_name, input_shape in self._op_inputs(op):
                    dim_names = tensor_dim_names.get(input_name)
                    if not dim_names:
                        continue
                    if input_shape == output_shape:
                        for axis, dim_ref in dim_names.items():
                            propagated.setdefault(axis, dim_ref)
                        continue
                    shared_rank = min(len(input_shape), len(output_shape))
                    for axis in range(shared_rank):
                        dim_ref = dim_names.get(axis)
                        if dim_ref is None:
                            continue
                        if input_shape[axis] == output_shape[axis]:
                            propagated.setdefault(axis, dim_ref)
                if propagated:
                    tensor_dim_names[output_name] = propagated

    def _op_outputs(
        self,
        op: OpBase,
    ) -> tuple[tuple[str, tuple[int, ...], ScalarType], ...]:
        return op.c_op_outputs(self)

    def _op_output_shape(self, op: OpBase) -> tuple[int, ...]:
        return op.computed_output_shape(self)

    def _op_output_dtype(self, op: OpBase) -> ScalarType:
        return op.computed_output_dtype(self)

    @staticmethod
    def _codegen_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
        if not shape:
            return (1,)
        return tuple(max(1, dim) if isinstance(dim, int) else dim for dim in shape)

    @staticmethod
    def _array_suffix(shape: tuple[int, ...], dtype: ScalarType | None = None) -> str:
        shape = CEmitter._codegen_shape(shape)
        suffix = "".join(f"[{dim}]" for dim in shape)
        if dtype == ScalarType.STRING:
            suffix += "[EMX_STRING_MAX_LEN]"
        return suffix

    @staticmethod
    def _dim_expr(dim: int | CodegenDim) -> str:
        return str(dim)

    @staticmethod
    def _dim_expected_size(dim: int | CodegenDim) -> int:
        if isinstance(dim, CodegenDim):
            return dim.expected_size
        return int(dim)

    @staticmethod
    def _shape_with_expected_sizes(
        shape: tuple[int, ...],
        dim_names: Mapping[int, CodegenDim] | None = None,
    ) -> tuple[int, ...]:
        dim_names = dim_names or {}
        return tuple(
            dim_names.get(index, dim).expected_size if index in dim_names else dim
            for index, dim in enumerate(shape)
        )

    def _param_array_suffix(
        self,
        shape: tuple[int, ...],
        dim_names: Mapping[int, CodegenDim] | None = None,
        *,
        use_restrict: bool = False,
        dtype: ScalarType | None = None,
    ) -> str:
        shape = CEmitter._codegen_shape(shape)
        dim_names = dim_names or {}
        if not (self._restrict_arrays and use_restrict):
            suffix = "".join(
                f"[{self._dim_expr(dim_names.get(index, dim))}]"
                for index, dim in enumerate(shape)
            )
            if dtype == ScalarType.STRING:
                suffix += "[EMX_STRING_MAX_LEN]"
            return suffix
        first, *rest = shape
        first_dim = self._dim_expr(dim_names.get(0, first))
        rest_dims = "".join(
            f"[{self._dim_expr(dim_names.get(index + 1, dim))}]"
            for index, dim in enumerate(rest)
        )
        suffix = f"[restrict {first_dim}]{rest_dims}"
        if dtype == ScalarType.STRING:
            suffix += "[EMX_STRING_MAX_LEN]"
        return suffix

    @staticmethod
    def _format_dim_args(dim_order: Sequence[str]) -> list[str]:
        return [f"int {dim_name}" for dim_name in dim_order]

    @staticmethod
    def _format_dim_args_prefix(dim_order: Sequence[str]) -> str:
        if not dim_order:
            return ""
        return ", ".join(f"int {dim_name}" for dim_name in dim_order) + ", "

    @staticmethod
    def _optional_input_flag_map(model: LoweredModel) -> dict[str, str]:
        return {
            name: flag
            for name, flag in zip(model.input_names, model.input_optional_names)
            if flag is not None
        }

    @staticmethod
    def _optional_output_flag_map(model: LoweredModel) -> dict[str, str]:
        return {
            name: flag
            for name, flag in zip(model.output_names, model.output_optional_names)
            if flag is not None
        }

    def _build_variable_dim_names(
        self,
        model: LoweredModel,
        variable_dim_inputs: Mapping[int, Mapping[int, str]] | None,
        variable_dim_outputs: Mapping[int, Mapping[int, str]] | None,
    ) -> tuple[
        list[str],
        dict[int, dict[int, CodegenDim]],
        dict[int, dict[int, CodegenDim]],
        dict[str, int],
    ]:
        variable_dim_inputs = variable_dim_inputs or {}
        variable_dim_outputs = variable_dim_outputs or {}
        dim_order: list[str] = []
        dim_vars: dict[tuple[str, int, int], CodegenDim] = {}
        dim_values: dict[str, int] = {}
        reserved_names = set(model.input_names) | set(model.output_names)
        reserved_names.update(
            name for name in model.input_optional_names if name is not None
        )
        reserved_names.update(
            name for name in model.output_optional_names if name is not None
        )
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
        ) -> CodegenDim:
            key = (kind, tensor_index, dim_index)
            dim_name = _unique_dim_name(dim_name)
            expected_size = (
                dim_value if dim_value > 1 else CodegenDim(dim_name).expected_size
            )
            if key not in dim_vars:
                dim_ref = CodegenDim(dim_name, expected_size)
                dim_vars[key] = dim_ref
                if dim_name not in dim_order:
                    dim_order.append(dim_name)
                dim_values.setdefault(dim_name, expected_size)
            else:
                expected_size = max(dim_values[dim_name], expected_size)
                dim_values[dim_name] = expected_size
                dim_vars[key] = CodegenDim(dim_name, expected_size)
            return dim_vars[key]

        def _build_dim_names(
            kind: str,
            tensor_index: int,
            shape: tuple[int, ...],
            variable_dims: Mapping[int, Mapping[int, str]],
        ) -> dict[int, CodegenDim]:
            dim_names: dict[int, CodegenDim] = {}
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

        input_dim_names: dict[int, dict[int, CodegenDim]] = {}
        for index, (shape, value_type) in enumerate(
            zip(model.input_shapes, model.input_types)
        ):
            if isinstance(value_type, SequenceType):
                continue
            dim_names = _build_dim_names("input", index, shape, variable_dim_inputs)
            if dim_names:
                input_dim_names[index] = dim_names

        output_dim_names: dict[int, dict[int, CodegenDim]] = {}
        for index, (shape, value_type) in enumerate(
            zip(model.output_shapes, model.output_types)
        ):
            if isinstance(value_type, SequenceType):
                continue
            dim_names = _build_dim_names("output", index, shape, variable_dim_outputs)
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
        return tuple(dim_names.get(index, dim) for index, dim in enumerate(shape))

    @staticmethod
    def _element_count_expr(shape_exprs: Sequence[str | int]) -> str:
        if not shape_exprs:
            return "1"
        return " * ".join(str(dim) for dim in shape_exprs)

    def _build_tensor_dim_names(
        self,
        model: LoweredModel,
        input_dim_names: Mapping[int, Mapping[int, CodegenDim]],
        output_dim_names: Mapping[int, Mapping[int, CodegenDim]],
        name_map: Mapping[str, str] | None = None,
    ) -> dict[str, dict[int, CodegenDim]]:
        dim_names: dict[str, dict[int, CodegenDim]] = {}
        symbol_dims: dict[str, CodegenDim] = {}

        def _tensor_type(value_type: ValueType) -> TensorType | None:
            if isinstance(value_type, TensorType):
                return value_type
            if isinstance(value_type, SequenceType):
                return value_type.elem
            return None

        def _remember_symbol_dims(
            value_types: Sequence[ValueType],
            per_tensor_dim_names: Mapping[int, Mapping[int, CodegenDim]],
        ) -> None:
            for index, value_type in enumerate(value_types):
                tensor_type = _tensor_type(value_type)
                if tensor_type is None:
                    continue
                tensor_dim_names = per_tensor_dim_names.get(index, {})
                for axis, dim_param in enumerate(tensor_type.dim_params):
                    dim_ref = tensor_dim_names.get(axis)
                    if dim_param and dim_ref is not None:
                        symbol_dims.setdefault(dim_param, dim_ref)

        _remember_symbol_dims(model.input_types, input_dim_names)
        _remember_symbol_dims(model.output_types, output_dim_names)

        for index, name in enumerate(model.input_names):
            if index in input_dim_names:
                dim_names[name] = dict(input_dim_names[index])
        for index, name in enumerate(model.output_names):
            if index in output_dim_names:
                dim_names[name] = dict(output_dim_names[index])
        for value in (
            model.op_context.graph.inputs
            + model.op_context.graph.values
            + model.op_context.graph.outputs
        ):
            tensor_type = _tensor_type(value.type)
            if tensor_type is None:
                continue
            tensor_dim_names = {
                axis: symbol_dims[dim_param]
                for axis, dim_param in enumerate(tensor_type.dim_params)
                if dim_param and dim_param in symbol_dims
            }
            if not tensor_dim_names:
                continue
            mapped_name = (
                name_map.get(value.name, value.name) if name_map else value.name
            )
            dim_names.setdefault(mapped_name, {}).update(tensor_dim_names)
        return dim_names

    @staticmethod
    def _copy_temp_buffer_dim_names(
        tensor_dim_names: dict[str, dict[int, CodegenDim]],
        temp_name_map: Mapping[str, str],
    ) -> None:
        for original_name, temp_name in temp_name_map.items():
            dim_names = tensor_dim_names.get(original_name)
            if dim_names:
                tensor_dim_names[temp_name] = dict(dim_names)

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
        batch_vars: tuple[str, ...],
        row_var: str | None,
        col_var: str | None,
        batch_rank: int,
        *,
        input0: str,
        input1: str,
        left_vector: bool,
        right_vector: bool,
        input0_shape: tuple[int, ...],
        input1_shape: tuple[int, ...],
        input0_batch_shape: tuple[int, ...],
        input1_batch_shape: tuple[int, ...],
    ) -> tuple[str, str]:
        def batch_indices(batch_shape: tuple[int, ...], actual_rank: int) -> list[str]:
            if actual_rank == 0:
                return []
            offset = batch_rank - actual_rank
            indices: list[str] = []
            for idx in range(actual_rank):
                dim = batch_shape[offset + idx]
                var = batch_vars[offset + idx]
                indices.append("0" if dim == 1 else var)
            return indices

        if left_vector:
            input0_indices = ["k"]
        else:
            input0_batch_rank = len(input0_shape) - 2
            input0_indices = batch_indices(input0_batch_shape, input0_batch_rank)
            input0_indices.append(row_var if row_var is not None else "0")
            input0_indices.append("k")
        if right_vector:
            input1_indices = ["k"]
        else:
            input1_batch_rank = len(input1_shape) - 2
            input1_indices = batch_indices(input1_batch_shape, input1_batch_rank)
            input1_indices.append("k")
            input1_indices.append(col_var if col_var is not None else "0")
        input0_index_expr = f"{input0}" + "".join(
            f"[{index}]" for index in input0_indices
        )
        input1_index_expr = f"{input1}" + "".join(
            f"[{index}]" for index in input1_indices
        )
        return input0_index_expr, input1_index_expr

    def _emit_testbench(
        self,
        model: LoweredModel,
        testbench_template,
        *,
        testbench_output_format: str,
        testbench_inputs: Mapping[str, tuple[float | int | bool, ...]] | None = None,
        testbench_outputs: Mapping[str, np.ndarray | list[np.ndarray]] | None = None,
        testbench_optional_inputs: Mapping[str, bool] | None = None,
        input_dim_names: Mapping[int, Mapping[int, CodegenDim]],
        output_dim_names: Mapping[int, Mapping[int, CodegenDim]],
        dim_order: Sequence[str],
        dim_values: Mapping[str, int],
        weight_data_filename: str,
    ) -> str:
        scalar_registry = self._emit_state.scalar_registry

        def concrete_shape_for_testbench(
            shape: tuple[int, ...],
            dim_names: Mapping[int, CodegenDim] | None,
            *,
            actual_shape: tuple[int, ...] | None = None,
        ) -> tuple[int, ...]:
            dim_names = dim_names or {}
            resolved: list[int] = []
            for axis, dim in enumerate(shape):
                if actual_shape is not None and axis < len(actual_shape):
                    actual_dim = int(actual_shape[axis])
                    if actual_dim >= 0 and (dim < 0 or axis in dim_names):
                        resolved.append(actual_dim)
                        continue
                dim_ref = dim_names.get(axis)
                if dim_ref is not None:
                    resolved.append(dim_values.get(dim_ref.name, dim_ref.expected_size))
                elif dim < 0:
                    resolved.append(CodegenDim(f"dim_{axis}").expected_size)
                else:
                    resolved.append(dim)
            return tuple(resolved)

        testbench_inputs = testbench_inputs or {}
        testbench_optional_inputs = testbench_optional_inputs or {}
        observed_dim_values: dict[str, int] = {}
        rng_requires_u64 = False
        rng_requires_float = False
        rng_requires_double = False
        rng_requires_i64 = False
        inputs = []
        for index, (name, shape, dtype, optional_flag, value_type) in enumerate(
            zip(
                model.input_names,
                model.input_shapes,
                model.input_dtypes,
                model.input_optional_names,
                model.input_types,
            )
        ):
            json_name = self._ctx_name(name)
            dim_names = input_dim_names.get(index) or self.dim_names_for(name)
            is_sequence_input = isinstance(value_type, SequenceType)
            constant_values = testbench_inputs.get(name)
            sequence_dim_axes = (
                self._sequence_dynamic_axes(name) if is_sequence_input else ()
            )
            sequence_items = (
                [np.asarray(item) for item in constant_values]
                if is_sequence_input and isinstance(constant_values, list)
                else None
            )
            input_actual_shape: tuple[int, ...] | None = None
            if isinstance(constant_values, np.ndarray):
                if is_sequence_input and constant_values.ndim >= 1:
                    input_actual_shape = tuple(
                        int(dim) for dim in constant_values.shape[1:]
                    )
                else:
                    input_actual_shape = tuple(
                        int(dim) for dim in constant_values.shape
                    )
            if input_actual_shape is not None and dim_names:
                for axis, actual_dim in enumerate(input_actual_shape):
                    dim_ref = dim_names.get(axis)
                    if dim_ref is not None:
                        observed_dim_values[dim_ref.name] = max(
                            observed_dim_values.get(dim_ref.name, 0),
                            int(actual_dim),
                        )
                        dim_values[dim_ref.name] = observed_dim_values[dim_ref.name]
            if is_sequence_input:
                concrete_codegen_shape = self._sequence_storage_shape(name)
            else:
                concrete_codegen_shape = concrete_shape_for_testbench(
                    shape,
                    dim_names,
                    actual_shape=input_actual_shape,
                )
            runtime_shape = tuple(
                (
                    dim_names[axis].name
                    if dim_names is not None and axis in dim_names
                    else concrete_codegen_shape[axis]
                )
                for axis in range(len(concrete_codegen_shape))
            )
            input_file_shape = tuple(
                (
                    f"*{dim_names[axis].name}__ptr"
                    if dim_names is not None and axis in dim_names
                    else concrete_codegen_shape[axis]
                )
                for axis in range(len(concrete_codegen_shape))
            )
            if is_sequence_input and isinstance(constant_values, np.ndarray):
                target_shape = (int(constant_values.shape[0]), *concrete_codegen_shape)
                if constant_values.shape != target_shape:
                    normalized = np.zeros(target_shape, dtype=constant_values.dtype)
                    if constant_values.ndim == len(target_shape):
                        overlap = (slice(0, target_shape[0]),) + tuple(
                            slice(0, min(cur_dim, tgt_dim))
                            for cur_dim, tgt_dim in zip(
                                constant_values.shape[1:], concrete_codegen_shape
                            )
                        )
                        normalized[overlap] = constant_values[overlap]
                    constant_values = normalized
            elif is_sequence_input and sequence_items is not None:
                constant_values = None
            loop_shape = CEmitter._codegen_shape(concrete_codegen_shape)
            if is_sequence_input:
                if (
                    isinstance(constant_values, np.ndarray)
                    and constant_values.ndim >= 1
                ):
                    loop_shape = (int(constant_values.shape[0]), *loop_shape)
                else:
                    loop_shape = (32, *loop_shape)
                runtime_shape = (32, *runtime_shape)
                input_file_shape = (32, *input_file_shape)
            elif not runtime_shape:
                runtime_shape = (1,)
                input_file_shape = (1,)
            loop_vars = self._loop_vars(loop_shape)
            if constant_values is None:
                if dtype in {ScalarType.F16, ScalarType.BF16, ScalarType.F32}:
                    rng_requires_u64 = True
                    rng_requires_float = True
                elif dtype == ScalarType.F64:
                    rng_requires_u64 = True
                    rng_requires_double = True
                elif dtype == ScalarType.BOOL:
                    rng_requires_u64 = True
                    pass
                elif dtype == ScalarType.STRING:
                    pass
                elif dtype.is_typedef_float:
                    rng_requires_u64 = True
                    rng_requires_i64 = True
                else:
                    rng_requires_u64 = True
                    rng_requires_i64 = True
            if dtype in {ScalarType.F16, ScalarType.BF16, ScalarType.F32}:
                random_expr = "rng_next_float()"
            elif dtype == ScalarType.F64:
                random_expr = "rng_next_double()"
            elif dtype == ScalarType.BOOL:
                random_expr = "((rng_next_u64() & 1ull) != 0)"
            elif dtype == ScalarType.STRING:
                random_expr = '""'
            elif dtype.is_typedef_float:
                mask = "0x0F" if dtype.is_float4 else "0xFF"
                random_expr = f"({dtype.c_type})(rng_next_i64() & {mask})"
            else:
                random_expr = f"({dtype.c_type})rng_next_i64()"
            constant_name = None
            constant_lines = None
            if constant_values is not None:
                constant_name = f"{name}_testbench_data"
                if isinstance(constant_values, np.ndarray):
                    flat_values = constant_values.reshape(-1).tolist()
                else:
                    flat_values = list(constant_values)
                formatted_values = [
                    self._format_value(value, dtype) for value in flat_values
                ]
                if (
                    formatted_values
                    and dtype == ScalarType.STRING
                    and isinstance(constant_values, np.ndarray)
                    and constant_values.ndim > 0
                ):
                    constant_lines = self._emit_initializer_lines(
                        formatted_values,
                        tuple(int(dim) for dim in constant_values.shape),
                    )
                elif formatted_values:
                    constant_lines = self._emit_initializer_lines(
                        formatted_values,
                        (len(formatted_values),),
                    )
                else:
                    storage_shape = self._codegen_shape(concrete_codegen_shape)
                    default_value = "" if dtype == ScalarType.STRING else 0
                    initializer_shape = (
                        storage_shape
                        if dtype == ScalarType.STRING
                        else (self._element_count(storage_shape),)
                    )
                    constant_lines = self._emit_initializer_lines(
                        [
                            self._format_value(default_value, dtype)
                            for _ in range(self._element_count(storage_shape))
                        ],
                        initializer_shape,
                    )
            optional_present = (
                testbench_optional_inputs.get(name, True)
                if optional_flag is not None
                else None
            )
            input_array_suffix = (
                f"[EMX_SEQUENCE_MAX_LEN]{self._array_suffix(concrete_codegen_shape, dtype)}"
                if is_sequence_input
                else self._array_suffix(concrete_codegen_shape, dtype)
            )
            input_storage = self._local_array_storage(
                concrete_codegen_shape,
                dtype,
                is_sequence=is_sequence_input,
            )
            inputs.append(
                {
                    "name": name,
                    "shape": loop_shape,
                    "runtime_shape": runtime_shape,
                    "input_file_shape": input_file_shape,
                    "shape_literal": ",".join(
                        str(dim) for dim in concrete_codegen_shape
                    ),
                    "count": self._element_count(concrete_codegen_shape),
                    "array_suffix": input_array_suffix,
                    "array_index_expr": "".join(f"[{var}]" for var in loop_vars),
                    "loop_vars": loop_vars,
                    "rank": len(loop_shape),
                    "index_expr": self._index_expr(loop_shape, loop_vars),
                    "dtype": dtype,
                    "is_string": dtype == ScalarType.STRING,
                    "c_type": dtype.c_type,
                    "random_expr": random_expr,
                    "print_format": self._testbench_print_format(dtype),
                    "print_cast": self._testbench_print_cast(dtype, scalar_registry),
                    "print_suffix": self._testbench_print_suffix(dtype),
                    "constant_name": constant_name,
                    "constant_lines": constant_lines,
                    "json_name": json_name,
                    "optional_flag_name": optional_flag,
                    "optional_present": optional_present,
                    "is_sequence": is_sequence_input,
                    "runtime_dim_reads": tuple(
                        {"axis": axis, "name": dim_ref.name}
                        for axis, dim_ref in sorted((dim_names or {}).items())
                    )
                    if not is_sequence_input
                    else (),
                    "count_name": f"{name}__count",
                    "count_value": loop_shape[0] if is_sequence_input else 1,
                    "storage": input_storage,
                    "sequence_dim_arrays": tuple(
                        {
                            "axis": axis,
                            "name": self._sequence_dim_array_name(name, axis),
                            "default_value": concrete_codegen_shape[axis],
                        }
                        for axis in sequence_dim_axes
                    ),
                }
            )
        outputs = []
        for index, (name, shape, dtype, value_type, optional_flag) in enumerate(
            zip(
                model.output_names,
                model.output_shapes,
                model.output_dtypes,
                model.output_types,
                model.output_optional_names,
            )
        ):
            json_name = self._ctx_name(name)
            dim_names = output_dim_names.get(index) or self.dim_names_for(name)
            output_values = testbench_outputs.get(name) if testbench_outputs else None
            sequence_dim_axes = (
                self._sequence_dynamic_axes(name)
                if isinstance(value_type, SequenceType)
                else ()
            )
            output_actual_shape: tuple[int, ...] | None = None
            if isinstance(output_values, np.ndarray):
                output_actual_shape = tuple(int(dim) for dim in output_values.shape)
            if output_actual_shape is not None and dim_names:
                for axis, actual_dim in enumerate(output_actual_shape):
                    dim_ref = dim_names.get(axis)
                    if dim_ref is not None:
                        observed_dim_values[dim_ref.name] = max(
                            observed_dim_values.get(dim_ref.name, 0),
                            int(actual_dim),
                        )
                        dim_values[dim_ref.name] = observed_dim_values[dim_ref.name]
            if isinstance(value_type, SequenceType):
                concrete_codegen_shape = self._sequence_storage_shape(name)
            else:
                concrete_codegen_shape = concrete_shape_for_testbench(
                    shape,
                    dim_names,
                    actual_shape=output_actual_shape,
                )
            is_sequence_output = isinstance(value_type, SequenceType)
            loop_shape = (1,) if not concrete_codegen_shape else concrete_codegen_shape
            if is_sequence_output:
                loop_shape = (32, *loop_shape)
            output_loop_vars = self._loop_vars(loop_shape)
            output_array_suffix = (
                f"[EMX_SEQUENCE_MAX_LEN]{self._array_suffix(concrete_codegen_shape, dtype)}"
                if is_sequence_output
                else self._array_suffix(concrete_codegen_shape, dtype)
            )
            output_storage = self._local_array_storage(
                concrete_codegen_shape,
                dtype,
                is_sequence=is_sequence_output,
            )
            outputs.append(
                {
                    "name": name,
                    "shape": loop_shape,
                    "shape_literal": ",".join(
                        str(dim) for dim in concrete_codegen_shape
                    ),
                    "count": self._element_count(concrete_codegen_shape),
                    "array_suffix": output_array_suffix,
                    "array_index_expr": "".join(f"[{var}]" for var in output_loop_vars),
                    "loop_vars": output_loop_vars,
                    "rank": len(loop_shape),
                    "index_expr": self._index_expr(loop_shape, output_loop_vars),
                    "dtype": dtype,
                    "is_string": dtype == ScalarType.STRING,
                    "c_type": dtype.c_type,
                    "print_format": self._testbench_print_format(dtype),
                    "print_cast": self._testbench_print_cast(dtype, scalar_registry),
                    "print_suffix": self._testbench_print_suffix(dtype),
                    "json_name": json_name,
                    "is_sequence": is_sequence_output,
                    "count_name": f"{name}__count",
                    "optional_flag_name": optional_flag,
                    "storage": output_storage,
                    "sequence_dim_arrays": tuple(
                        {
                            "axis": axis,
                            "name": self._sequence_dim_array_name(name, axis),
                            "value": concrete_codegen_shape[axis],
                        }
                        for axis in sequence_dim_axes
                    ),
                    "sequence_item_shape_dims": tuple(
                        {
                            "axis": axis,
                            "name": (
                                self._sequence_dim_array_name(name, axis)
                                if axis in sequence_dim_axes
                                else None
                            ),
                            "value": concrete_codegen_shape[axis],
                        }
                        for axis in range(len(concrete_codegen_shape))
                    ),
                }
            )
        try:
            parsed_output_format = parse_testbench_output_format(
                testbench_output_format
            )
        except ValueError as exc:
            raise CodegenError(str(exc)) from exc
        rendered = testbench_template.render(
            model_name=model.name,
            testbench_output_format=parsed_output_format.kind,
            testbench_emmtrix_ulp_tag=(
                parsed_output_format.emmtrix_ulp_tag
                if parsed_output_format.kind == "txt-emmtrix"
                else None
            ),
            rng_requires_u64=rng_requires_u64,
            rng_requires_float=rng_requires_float,
            rng_requires_double=rng_requires_double,
            rng_requires_i64=rng_requires_i64,
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
        testbench_inputs: (
            Mapping[str, tuple[float | int | bool, ...] | np.ndarray] | None
        ),
    ) -> bool:
        if not testbench_inputs:
            return False
        dtype_map = dict(zip(model.input_names, model.input_dtypes))
        float_dtypes = {ScalarType.F16, ScalarType.BF16, ScalarType.F32, ScalarType.F64}
        for name, values in testbench_inputs.items():
            if dtype_map.get(name) not in float_dtypes:
                continue
            if isinstance(values, np.ndarray):
                flat_values = values.reshape(-1).tolist()
            elif isinstance(values, list):
                flat_values = [
                    item
                    for tensor in values
                    for item in np.asarray(tensor).reshape(-1).tolist()
                ]
            else:
                flat_values = values
            for value in flat_values:
                if not math.isfinite(float(value)):
                    return True
        return False

    def _partition_constants(
        self, constants: tuple[ConstTensor, ...]
    ) -> tuple[tuple[ConstTensor, ...], tuple[ConstTensor, ...]]:
        if self._large_weight_threshold <= 0:
            return constants, ()
        sorted_constants = sorted(
            enumerate(constants),
            key=lambda item: (
                (
                    0
                    if item[1].dtype == ScalarType.STRING
                    else self._element_count(item[1].shape)
                    * item[1].dtype.np_dtype.itemsize
                ),
                item[0],
            ),
        )
        inline_set: set[ConstTensor] = set()
        total_bytes = 0
        for _, const in sorted_constants:
            if const.dtype == ScalarType.STRING:
                inline_set.add(const)
                continue
            const_bytes = (
                self._element_count(const.shape) * const.dtype.np_dtype.itemsize
            )
            if total_bytes + const_bytes <= self._large_weight_threshold:
                inline_set.add(const)
                total_bytes += const_bytes
        inline: list[ConstTensor] = []
        large: list[ConstTensor] = []
        for const in constants:
            if const in inline_set:
                inline.append(const)
            else:
                large.append(const)
        return tuple(inline), tuple(large)

    @staticmethod
    def _weight_data_filename(model: LoweredModel) -> str:
        return f"{model.name}.bin"

    def _emit_weight_loader(
        self, model: LoweredModel, large_constants: tuple[ConstTensor, ...]
    ) -> str:
        lines = []
        if not large_constants:
            lines.append(f"_Bool {model.name}_load(const char *path) {{")
            lines.append("    (void)path;")
            lines.append("    return 1;")
            lines.append("}")
            return _format_c_indentation("\n".join(lines))
        lines.append(f"static _Bool {model.name}_load_file(FILE *file);")
        lines.append("")
        lines.append(f"_Bool {model.name}_load(const char *path) {{")
        lines.append('    FILE *file = fopen(path, "rb");')
        lines.append("    if (!file) {")
        lines.append(
            '        fprintf(stderr, "Failed to open weight file: %s\\n", path);'
        )
        lines.append("        return 0;")
        lines.append("    }")
        lines.append(f"    _Bool ok = {model.name}_load_file(file);")
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
            array_suffix = self._array_suffix(shape, const.dtype)
            values = [
                self._format_weight_value(value, const.dtype) for value in const.data
            ]
            lines.append(
                f"{storage_prefix} EMX_UNUSED {c_type} {const.name}{array_suffix} = {{"
            )
            if values:
                if (
                    self._truncate_weights_after is not None
                    and len(values) > self._truncate_weights_after
                ):
                    truncated_lines, _, _, _ = self._emit_initializer_lines_truncated(
                        values, shape, self._truncate_weights_after
                    )
                    lines.extend(truncated_lines)
                else:
                    lines.extend(self._emit_initializer_lines(values, shape))
            lines.append("};")
            lines.append("")
        if lines and not lines[-1]:
            lines.pop()
        return "\n".join(lines)

    def _emit_constant_declarations(self, constants: tuple[ConstTensor, ...]) -> str:
        if not constants:
            return ""
        lines = []
        for index, const in enumerate(constants, start=1):
            c_type = const.dtype.c_type
            array_suffix = self._array_suffix(const.shape, const.dtype)
            lines.append(f"extern const {c_type} {const.name}{array_suffix};")
        return "\n".join(lines)

    def _emit_constant_storage_declarations(
        self, constants: tuple[ConstTensor, ...]
    ) -> str:
        if not constants:
            return ""
        lines = []
        for index, const in enumerate(constants, start=1):
            c_type = const.dtype.c_type
            array_suffix = self._array_suffix(const.shape, const.dtype)
            lines.append(f"extern {c_type} {const.name}{array_suffix};")
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
        prefix = f"{storage_prefix} " if storage_prefix else ""
        for index, const in enumerate(constants, start=1):
            lines.append(self._emit_constant_comment(const, index))
            c_type = const.dtype.c_type
            array_suffix = self._array_suffix(const.shape, const.dtype)
            lines.append(f"{prefix}{c_type} {const.name}{array_suffix};")
            lines.append("")
        if lines and not lines[-1]:
            lines.pop()
        return "\n".join(lines)

    def collect_weight_data(self, constants: tuple[ConstTensor, ...]) -> bytes | None:
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
    def _format_bfloat16(value: float) -> str:
        return f"(__bf16){CEmitter._format_float(value)}"

    @staticmethod
    def _format_float8(value: float, dtype: ScalarType) -> str:
        import ml_dtypes  # noqa: F811

        _ML_DTYPE_MAP = {
            ScalarType.F4E2M1: ml_dtypes.float4_e2m1fn,
            ScalarType.F8E4M3FN: ml_dtypes.float8_e4m3fn,
            ScalarType.F8E4M3FNUZ: ml_dtypes.float8_e4m3fnuz,
            ScalarType.F8E5M2: ml_dtypes.float8_e5m2,
            ScalarType.F8E5M2FNUZ: ml_dtypes.float8_e5m2fnuz,
            ScalarType.F8E8M0FNU: ml_dtypes.float8_e8m0fnu,
        }
        import numpy as np

        f8_dtype = _ML_DTYPE_MAP[dtype]
        f8_val = np.array(value, dtype=np.float32).astype(f8_dtype)
        byte_val = f8_val.view(np.uint8).item()
        return f"0x{byte_val:02X}u"

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
        if dtype == ScalarType.BF16:
            return CEmitter._format_bfloat16(value)
        if dtype.is_typedef_float:
            return CEmitter._format_float8(value, dtype)
        return CEmitter._format_float(value)

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
    def _format_c_string_literal(value: str) -> str:
        escaped_parts: list[str] = []
        for byte in value.encode("utf-8"):
            if byte == 0x5C:
                escaped_parts.append("\\\\")
            elif byte == 0x22:
                escaped_parts.append('\\"')
            elif byte == 0x0A:
                escaped_parts.append("\\n")
            elif byte == 0x0D:
                escaped_parts.append("\\r")
            elif byte == 0x09:
                escaped_parts.append("\\t")
            elif 0x20 <= byte <= 0x7E:
                escaped_parts.append(chr(byte))
            else:
                escaped_parts.append(f"\\x{byte:02X}")
        return f'"{"".join(escaped_parts)}"'

    @staticmethod
    def _format_literal(dtype: ScalarType, value: float | int | bool) -> str:
        if dtype == ScalarType.F16:
            return CEmitter._format_float16(float(value))
        if dtype == ScalarType.BF16:
            return CEmitter._format_bfloat16(float(value))
        if dtype.is_typedef_float:
            return CEmitter._format_float8(float(value), dtype)
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
        if dtype == ScalarType.U4:
            return CEmitter._format_uint(int(value), 4, "15")
        if dtype == ScalarType.U2:
            return CEmitter._format_uint(int(value), 2, "3")
        if dtype == ScalarType.I64:
            return CEmitter._format_int64(int(value))
        if dtype == ScalarType.I32:
            return CEmitter._format_int(int(value), 32, "INT32_MIN")
        if dtype == ScalarType.I16:
            return CEmitter._format_int(int(value), 16, "INT16_MIN")
        if dtype == ScalarType.I8:
            return CEmitter._format_int(int(value), 8, "INT8_MIN")
        if dtype == ScalarType.I4:
            return CEmitter._format_int(int(value), 4, "-8")
        if dtype == ScalarType.I2:
            return CEmitter._format_int(int(value), 2, "-2")
        raise CodegenError(f"Unsupported dtype {dtype.onnx_name}")

    def _format_value(self, value: float | int | bool, dtype: ScalarType) -> str:
        if dtype == ScalarType.F16:
            return self._format_float16(float(value))
        if dtype == ScalarType.BF16:
            return self._format_bfloat16(float(value))
        if dtype.is_typedef_float:
            return self._format_float8(float(value), dtype)
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
        if dtype == ScalarType.U4:
            return self._format_uint(int(value), 4, "15")
        if dtype == ScalarType.U2:
            return self._format_uint(int(value), 2, "3")
        if dtype == ScalarType.I64:
            return self._format_int64(int(value))
        if dtype == ScalarType.I32:
            return self._format_int(int(value), 32, "INT32_MIN")
        if dtype == ScalarType.I16:
            return self._format_int(int(value), 16, "INT16_MIN")
        if dtype == ScalarType.I8:
            return self._format_int(int(value), 8, "INT8_MIN")
        if dtype == ScalarType.I4:
            return self._format_int(int(value), 4, "-8")
        if dtype == ScalarType.I2:
            return self._format_int(int(value), 2, "-2")
        if dtype == ScalarType.STRING:
            if isinstance(value, bytes):
                return self._format_c_string_literal(value.decode("utf-8"))
            return self._format_c_string_literal(str(value))
        raise CodegenError(f"Unsupported dtype {dtype.onnx_name}")

    def _format_weight_value(
        self, value: float | int | bool | bytes | str, dtype: ScalarType
    ) -> str:
        if dtype == ScalarType.F16:
            formatted = self._format_float32_hex(float(value))
            if formatted == "NAN" or formatted.endswith("INFINITY"):
                return f"(_Float16){formatted}"
            return f"(_Float16){formatted}f"
        if dtype == ScalarType.BF16:
            formatted = self._format_float32_hex(float(value))
            if formatted == "NAN" or formatted.endswith("INFINITY"):
                return f"(__bf16){formatted}"
            return f"(__bf16){formatted}f"
        if dtype.is_typedef_float:
            return self._format_float8(float(value), dtype)
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
        if dtype == ScalarType.U4:
            return self._format_uint(int(value), 4, "15")
        if dtype == ScalarType.U2:
            return self._format_uint(int(value), 2, "3")
        if dtype == ScalarType.I64:
            return self._format_int64(int(value))
        if dtype == ScalarType.I32:
            return self._format_int(int(value), 32, "INT32_MIN")
        if dtype == ScalarType.I16:
            return self._format_int(int(value), 16, "INT16_MIN")
        if dtype == ScalarType.I8:
            return self._format_int(int(value), 8, "INT8_MIN")
        if dtype == ScalarType.I4:
            return self._format_int(int(value), 4, "-8")
        if dtype == ScalarType.I2:
            return self._format_int(int(value), 2, "-2")
        if dtype == ScalarType.STRING:
            if isinstance(value, bytes):
                decoded = value.decode("utf-8")
            else:
                decoded = str(value)
            return self._format_c_string_literal(decoded)
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
        if dtype in {ScalarType.F16, ScalarType.BF16}:
            return '\\"%a\\"'
        if dtype == ScalarType.F32:
            return '\\"%a\\"'
        if dtype == ScalarType.F64:
            return '\\"%a\\"'
        if dtype.is_typedef_float:
            return "%hhu"
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
        if dtype in {ScalarType.U4, ScalarType.U2}:
            return "%u"
        if dtype == ScalarType.I64:
            return "%lld"
        if dtype == ScalarType.I32:
            return "%d"
        if dtype == ScalarType.I16:
            return "%hd"
        if dtype == ScalarType.I8:
            return "%hhd"
        if dtype in {ScalarType.I4, ScalarType.I2}:
            return "%d"
        if dtype == ScalarType.STRING:
            return '"%s"'
        raise CodegenError(f"Unsupported dtype {dtype.onnx_name}")

    @staticmethod
    def _print_cast(dtype: ScalarType) -> str:
        if dtype.is_typedef_float:
            return "(unsigned int)"
        if dtype.is_float:
            return "(double)"
        if dtype == ScalarType.BOOL:
            return "(int)"
        if dtype == ScalarType.U64:
            return "(unsigned long long)"
        if dtype in {
            ScalarType.U32,
            ScalarType.U16,
            ScalarType.U8,
            ScalarType.U4,
            ScalarType.U2,
        }:
            return "(unsigned int)"
        if dtype == ScalarType.I64:
            return "(long long)"
        if dtype in {
            ScalarType.I32,
            ScalarType.I16,
            ScalarType.I8,
            ScalarType.I4,
            ScalarType.I2,
        }:
            return "(int)"
        if dtype == ScalarType.STRING:
            return ""
        raise CodegenError(f"Unsupported dtype {dtype.onnx_name}")

    def _testbench_print_format(self, dtype: ScalarType) -> str:
        if dtype.is_typedef_float:
            return '\\"%a\\"'
        return self._print_format(dtype)

    def _testbench_print_cast(
        self, dtype: ScalarType, scalar_registry: ScalarFunctionRegistry
    ) -> str:
        if dtype.is_typedef_float:
            fn_name = scalar_registry.request(
                ScalarFunctionKey(
                    function=ScalarFunction.CONVERT_FROM_BOOL,
                    return_type=dtype,
                )
            )
            return f"(double){fn_name}("
        return self._print_cast(dtype)

    @staticmethod
    def _testbench_print_suffix(dtype: ScalarType) -> str:
        if dtype.is_typedef_float:
            return ")"
        return ""


def _format_multiline_value(value: str | None) -> list[str]:
    if not value:
        return ["  n/a"]
    lines = value.splitlines() or [""]
    return [f"  {line}" if line else "  " for line in lines]

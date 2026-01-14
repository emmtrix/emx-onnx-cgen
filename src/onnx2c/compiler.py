from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Mapping

import numpy as np
import onnx

from shared.scalar_types import ScalarType

from .codegen.c_emitter import (
    AttentionOp,
    AveragePoolOp,
    BatchNormOp,
    LpNormalizationOp,
    InstanceNormalizationOp,
    GroupNormalizationOp,
    LayerNormalizationOp,
    MeanVarianceNormalizationOp,
    RMSNormalizationOp,
    BinaryOp,
    CastOp,
    ClipOp,
    CEmitter,
    ConstTensor,
    ConvOp,
    ConcatOp,
    ConstantOfShapeOp,
    GemmOp,
    GatherOp,
    GatherElementsOp,
    ExpandOp,
    RangeOp,
    LrnOp,
    LstmOp,
    LogSoftmaxOp,
    NegativeLogLikelihoodLossOp,
    NodeInfo,
    SplitOp,
    SoftmaxCrossEntropyLossOp,
    LoweredModel,
    ModelHeader,
    MatMulOp,
    MaxPoolOp,
    ReduceOp,
    ArgReduceOp,
    ReshapeOp,
    ResizeOp,
    SoftmaxOp,
    ShapeOp,
    SliceOp,
    TransposeOp,
    UnaryOp,
    WhereOp,
)
from .dtypes import dtype_info
from .errors import CodegenError, ShapeInferenceError, UnsupportedOpError
from .ir.model import Graph, Value
from .lowering.attention import AttentionSpec, resolve_attention_spec
from .lowering.average_pool import (
    lower_average_pool,
    lower_global_average_pool,
)
from .lowering.batch_normalization import lower_batch_normalization
from .lowering.cast import lower_cast
from .lowering.concat import lower_concat
from .lowering.common import (
    ensure_supported_dtype,
    node_dtype,
    shape_product,
    value_dtype,
    value_shape,
)
from .lowering.conv import ConvSpec, resolve_conv_spec
from .lowering.constant_of_shape import lower_constant_of_shape
from .lowering.dropout import lower_dropout
from .lowering.flatten import lower_flatten
from .lowering.gather import lower_gather
from .lowering.gather_elements import lower_gather_elements
from .lowering.gemm import resolve_gemm_spec, validate_gemm_bias_shape
from .lowering.lrn import LrnSpec, resolve_lrn_spec
from .lowering.logsoftmax import lower_logsoftmax
from .lowering import group_normalization as _group_normalization  # noqa: F401
from .lowering import instance_normalization as _instance_normalization  # noqa: F401
from .lowering import layer_normalization as _layer_normalization  # noqa: F401
from .lowering import lp_normalization as _lp_normalization  # noqa: F401
from .lowering import mean_variance_normalization as _mean_variance_normalization  # noqa: F401
from .lowering.negative_log_likelihood_loss import (
    lower_negative_log_likelihood_loss,
)
from .lowering.expand import lower_expand
from .lowering.range import lower_range
from .lowering.split import lower_split
from .lowering.softmax_cross_entropy_loss import (
    lower_softmax_cross_entropy_loss,
)
from .lowering.matmul import lower_matmul
from .lowering.maxpool import MaxPoolSpec, resolve_maxpool_spec
from .lowering.reduce import (
    REDUCE_KIND_BY_OP,
    REDUCE_OUTPUTS_FLOAT_ONLY,
)
from .lowering import arg_reduce as _arg_reduce  # noqa: F401
from .lowering.reshape import lower_reshape
from .lowering.resize import lower_resize
from .lowering.slice import lower_slice
from .lowering.squeeze import lower_squeeze
from .lowering import depth_space as _depth_space  # noqa: F401
from .lowering import eye_like as _eye_like  # noqa: F401
from .lowering import identity as _identity  # noqa: F401
from .lowering import tile as _tile  # noqa: F401
from .lowering.shape import lower_shape
from .lowering.size import lower_size
from .lowering.softmax import lower_softmax
from .lowering.transpose import lower_transpose
from .lowering.unsqueeze import lower_unsqueeze
from .lowering.where import lower_where
from .lowering.elementwise import (
    lower_celu,
    lower_clip,
    lower_isinf,
    lower_isnan,
    lower_shrink,
    lower_swish,
)
from .lowering import rms_normalization as _rms_normalization  # noqa: F401
from .lowering.registry import get_lowering_registry, resolve_dispatch
from .onnx_import import import_onnx
from .ops import (
    BINARY_OP_TYPES,
    COMPARE_FUNCTIONS,
    UNARY_OP_TYPES,
    binary_op_symbol,
    unary_op_symbol,
    validate_unary_attrs,
)
from shared.scalar_functions import ScalarFunction, ScalarFunctionError
from .runtime.evaluator import Evaluator


@dataclass(frozen=True)
class CompilerOptions:
    template_dir: Path
    model_name: str = "model"
    emit_testbench: bool = False
    command_line: str | None = None
    model_checksum: str | None = None
    restrict_arrays: bool = True
    testbench_inputs: Mapping[str, np.ndarray] | None = None


class Compiler:
    def __init__(self, options: CompilerOptions | None = None) -> None:
        if options is None:
            options = CompilerOptions(template_dir=Path("templates"))
        self._options = options
        self._emitter = CEmitter(
            options.template_dir, restrict_arrays=options.restrict_arrays
        )

    def compile(self, model: onnx.ModelProto) -> str:
        graph = import_onnx(model)
        testbench_inputs = self._resolve_testbench_inputs(graph)
        variable_dim_inputs, variable_dim_outputs = self._collect_variable_dims(
            graph
        )
        lowered = self._lower_model(model, graph)
        return self._emitter.emit_model(
            lowered,
            emit_testbench=self._options.emit_testbench,
            testbench_inputs=testbench_inputs,
            variable_dim_inputs=variable_dim_inputs,
            variable_dim_outputs=variable_dim_outputs,
        )

    def compile_with_data_file(self, model: onnx.ModelProto) -> tuple[str, str]:
        graph = import_onnx(model)
        testbench_inputs = self._resolve_testbench_inputs(graph)
        variable_dim_inputs, variable_dim_outputs = self._collect_variable_dims(
            graph
        )
        lowered = self._lower_model(model, graph)
        return self._emitter.emit_model_with_data_file(
            lowered,
            emit_testbench=self._options.emit_testbench,
            testbench_inputs=testbench_inputs,
            variable_dim_inputs=variable_dim_inputs,
            variable_dim_outputs=variable_dim_outputs,
        )

    @staticmethod
    def _collect_variable_dims(
        graph: Graph,
    ) -> tuple[dict[int, dict[int, str]], dict[int, dict[int, str]]]:
        def collect(values: tuple[Value, ...]) -> dict[int, dict[int, str]]:
            dim_map: dict[int, dict[int, str]] = {}
            for index, value in enumerate(values):
                dims = {
                    dim_index: dim_param
                    for dim_index, dim_param in enumerate(
                        value.type.dim_params
                    )
                    if dim_param
                }
                if dims:
                    dim_map[index] = dims
            return dim_map

        return collect(graph.inputs), collect(graph.outputs)

    def _lower_model(self, model: onnx.ModelProto, graph: Graph) -> LoweredModel:
        constants = _lowered_constants(graph)
        self._validate_graph(graph)
        (
            input_names,
            input_shapes,
            input_dtypes,
            output_names,
            output_shapes,
            output_dtypes,
        ) = self._collect_io_specs(graph)
        ops, node_infos = self._lower_nodes(graph)
        header = self._build_header(model, graph)
        return LoweredModel(
            name=self._options.model_name,
            input_names=input_names,
            input_shapes=input_shapes,
            input_dtypes=input_dtypes,
            output_names=output_names,
            output_shapes=output_shapes,
            output_dtypes=output_dtypes,
            constants=constants,
            ops=tuple(ops),
            node_infos=tuple(node_infos),
            header=header,
        )

    def _resolve_testbench_inputs(
        self, graph: Graph
    ) -> Mapping[str, tuple[float | int | bool, ...]] | None:
        if not self._options.testbench_inputs:
            return None
        input_specs = {value.name: value for value in graph.inputs}
        unknown_inputs = sorted(
            name
            for name in self._options.testbench_inputs
            if name not in input_specs
        )
        if unknown_inputs:
            raise CodegenError(
                "Testbench inputs include unknown inputs: "
                + ", ".join(unknown_inputs)
            )
        resolved: dict[str, tuple[float | int | bool, ...]] = {}
        for name, values in self._options.testbench_inputs.items():
            if not isinstance(values, np.ndarray):
                raise CodegenError(
                    f"Testbench input {name} must be a numpy array"
                )
            input_value = input_specs[name]
            dtype = value_dtype(graph, name)
            info = dtype_info(dtype)
            expected_shape = input_value.type.shape
            expected_count = shape_product(expected_shape)
            array = values.astype(info.np_dtype, copy=False)
            if array.size != expected_count:
                raise CodegenError(
                    "Testbench input "
                    f"{name} has {array.size} elements, expected {expected_count}"
                )
            array = array.reshape(expected_shape)
            resolved[name] = tuple(array.ravel().tolist())
        return resolved

    def _validate_graph(self, graph: Graph) -> None:
        if not graph.outputs:
            raise UnsupportedOpError("Graph must have at least one output")
        if not graph.nodes:
            raise UnsupportedOpError("Graph must contain at least one node")
        for value in graph.outputs:
            element_count = shape_product(value.type.shape)
            if element_count <= 0:
                raise ShapeInferenceError("Output shape must be fully defined")

    def _collect_io_specs(
        self, graph: Graph
    ) -> tuple[
        tuple[str, ...],
        tuple[tuple[int, ...], ...],
        tuple[ScalarType, ...],
        tuple[str, ...],
        tuple[tuple[int, ...], ...],
        tuple[ScalarType, ...],
    ]:
        input_names = tuple(value.name for value in graph.inputs)
        input_shapes = tuple(value.type.shape for value in graph.inputs)
        input_dtypes = tuple(
            value_dtype(graph, value.name) for value in graph.inputs
        )
        output_names = tuple(value.name for value in graph.outputs)
        output_shapes = tuple(value.type.shape for value in graph.outputs)
        output_dtypes = tuple(
            value_dtype(graph, value.name) for value in graph.outputs
        )
        return (
            input_names,
            input_shapes,
            input_dtypes,
            output_names,
            output_shapes,
            output_dtypes,
        )

    def _lower_nodes(
        self, graph: Graph
    ) -> tuple[
        list[
            BinaryOp
            | UnaryOp
            | ClipOp
            | CastOp
            | MatMulOp
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
            | NegativeLogLikelihoodLossOp
            | SoftmaxCrossEntropyLossOp
            | MaxPoolOp
            | ConcatOp
            | GatherElementsOp
            | GatherOp
            | TransposeOp
            | ConstantOfShapeOp
            | ReshapeOp
            | SliceOp
            | ResizeOp
            | ReduceOp
            | ArgReduceOp
            | ShapeOp
            | ExpandOp
            | RangeOp
            | SplitOp
        ],
        list[NodeInfo],
    ]:
        ops: list[
            BinaryOp
            | UnaryOp
            | ClipOp
            | CastOp
            | MatMulOp
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
            | NegativeLogLikelihoodLossOp
            | SoftmaxCrossEntropyLossOp
            | MaxPoolOp
            | ConcatOp
            | GatherElementsOp
            | GatherOp
            | TransposeOp
            | ConstantOfShapeOp
            | ReshapeOp
            | SliceOp
            | ResizeOp
            | ReduceOp
            | ArgReduceOp
            | ShapeOp
            | ExpandOp
            | RangeOp
            | SplitOp
            | WhereOp
        ] = []
        node_infos: list[NodeInfo] = []
        for node in graph.nodes:
            lowering = resolve_dispatch(
                node.op_type,
                get_lowering_registry(),
                binary_types=BINARY_OP_TYPES,
                unary_types=UNARY_OP_TYPES,
                binary_fallback=lambda: _lower_binary_unary,
                unary_fallback=lambda: _lower_binary_unary,
            )
            ops.append(lowering(graph, node))
            node_infos.append(
                NodeInfo(
                    op_type=node.op_type,
                    inputs=tuple(node.inputs),
                    outputs=tuple(node.outputs),
                    attrs=dict(node.attrs),
                )
            )
        return ops, node_infos

    def _build_header(self, model: onnx.ModelProto, graph: Graph) -> ModelHeader:
        metadata_props = tuple(
            (prop.key, prop.value) for prop in model.metadata_props
        )
        opset_imports = tuple(
            (opset.domain, opset.version) for opset in model.opset_import
        )
        checksum = self._options.model_checksum
        if checksum is None:
            checksum = hashlib.sha256(model.SerializeToString()).hexdigest()
        return ModelHeader(
            generator="Generated by emmtrix ONNX to C Compiler (emx-onnx2c)",
            command_line=self._options.command_line,
            model_checksum=checksum,
            model_name=self._options.model_name,
            graph_name=model.graph.name or None,
            description=model.doc_string or None,
            graph_description=model.graph.doc_string or None,
            producer_name=model.producer_name or None,
            producer_version=model.producer_version or None,
            domain=model.domain or None,
            model_version=model.model_version or None,
            ir_version=model.ir_version or None,
            opset_imports=opset_imports,
            metadata_props=metadata_props,
            input_count=len(graph.inputs),
            output_count=len(graph.outputs),
            node_count=len(graph.nodes),
            initializer_count=len(graph.initializers),
        )

    def run(
        self, model: onnx.ModelProto, feeds: Mapping[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        graph = import_onnx(model)
        evaluator = Evaluator(graph)
        return evaluator.run(feeds)


def _lowered_constants(graph: Graph) -> tuple[ConstTensor, ...]:
    constants: list[ConstTensor] = []
    for initializer in graph.initializers:
        dtype = ensure_supported_dtype(initializer.type.dtype)
        constants.append(
            ConstTensor(
                name=initializer.name,
                shape=initializer.type.shape,
                data=tuple(
                    dtype.np_dtype.type(value)
                    for value in initializer.data.ravel()
                ),
                dtype=dtype,
            )
        )
    return tuple(constants)


def _lower_binary_unary(graph: Graph, node: Node) -> BinaryOp | UnaryOp:
    if node.op_type == "BitShift":
        if len(node.inputs) != 2 or len(node.outputs) != 1:
            raise UnsupportedOpError("BitShift must have 2 inputs and 1 output")
        direction_attr = node.attrs.get("direction", "LEFT")
        if isinstance(direction_attr, bytes):
            direction = direction_attr.decode()
        else:
            direction = str(direction_attr)
        if direction not in {"LEFT", "RIGHT"}:
            raise UnsupportedOpError(
                "BitShift direction must be LEFT or RIGHT"
            )
        op_dtype = node_dtype(graph, node, *node.inputs, *node.outputs)
        if not op_dtype.is_integer:
            raise UnsupportedOpError("BitShift expects integer inputs")
        function = (
            ScalarFunction.BITWISE_LEFT_SHIFT
            if direction == "LEFT"
            else ScalarFunction.BITWISE_RIGHT_SHIFT
        )
        op_spec = binary_op_symbol(function, node.attrs, dtype=op_dtype)
        if op_spec is None:
            raise UnsupportedOpError("Unsupported op BitShift")
        output_shape = value_shape(graph, node.outputs[0], node)
        return BinaryOp(
            input0=node.inputs[0],
            input1=node.inputs[1],
            output=node.outputs[0],
            function=function,
            operator_kind=op_spec.kind,
            shape=output_shape,
            dtype=op_dtype,
            input_dtype=op_dtype,
        )
    if node.op_type == "Mod":
        fmod = int(node.attrs.get("fmod", 0))
        if fmod not in {0, 1}:
            raise UnsupportedOpError("Mod only supports fmod=0 or fmod=1")
        function = (
            ScalarFunction.FMOD if fmod == 1 else ScalarFunction.REMAINDER
        )
    else:
        try:
            function = ScalarFunction.from_onnx_op(node.op_type)
        except ScalarFunctionError as exc:
            raise UnsupportedOpError(
                f"Unsupported op {node.op_type}"
            ) from exc
    validate_unary_attrs(node.op_type, node.attrs)
    if function in COMPARE_FUNCTIONS:
        input_dtype = node_dtype(graph, node, *node.inputs)
        output_dtype = value_dtype(graph, node.outputs[0], node)
        op_spec = binary_op_symbol(function, node.attrs, dtype=input_dtype)
        if op_spec is None:
            raise UnsupportedOpError(f"Unsupported op {node.op_type}")
        if len(node.inputs) != 2 or len(node.outputs) != 1:
            raise UnsupportedOpError(
                f"{node.op_type} must have 2 inputs and 1 output"
            )
        if output_dtype != ScalarType.BOOL:
            raise UnsupportedOpError(
                f"{node.op_type} expects bool output, got {output_dtype.onnx_name}"
            )
        output_shape = value_shape(graph, node.outputs[0], node)
        return BinaryOp(
            input0=node.inputs[0],
            input1=node.inputs[1],
            output=node.outputs[0],
            function=function,
            operator_kind=op_spec.kind,
            shape=output_shape,
            dtype=output_dtype,
            input_dtype=input_dtype,
        )
    op_dtype = node_dtype(graph, node, *node.inputs, *node.outputs)
    op_spec = binary_op_symbol(function, node.attrs, dtype=op_dtype)
    unary_symbol = unary_op_symbol(function, dtype=op_dtype)
    if op_spec is None and unary_symbol is None:
        raise UnsupportedOpError(f"Unsupported op {node.op_type}")
    if op_spec is not None:
        if len(node.inputs) != 2 or len(node.outputs) != 1:
            raise UnsupportedOpError(
                f"{node.op_type} must have 2 inputs and 1 output"
            )
        output_shape = value_shape(graph, node.outputs[0], node)
        return BinaryOp(
            input0=node.inputs[0],
            input1=node.inputs[1],
            output=node.outputs[0],
            function=function,
            operator_kind=op_spec.kind,
            shape=output_shape,
            dtype=op_dtype,
            input_dtype=op_dtype,
        )
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        raise UnsupportedOpError(f"{node.op_type} must have 1 input and 1 output")
    output_shape = value_shape(graph, node.outputs[0], node)
    return UnaryOp(
        input0=node.inputs[0],
        output=node.outputs[0],
        function=function,
        shape=output_shape,
        dtype=op_dtype,
        input_dtype=op_dtype,
        params=(),
    )

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, Template, select_autoescape

from ..errors import CodegenError
from ..dtypes import dtype_info


def _format_c_indentation(source: str, *, indent: str = "    ") -> str:
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
        open_count = stripped.count("{")
        close_count = stripped.count("}")
        if stripped.startswith("}"):
            close_count = max(close_count - 1, 0)
        indent_level += open_count - close_count
        indent_level = max(indent_level, 0)
    return "\n".join(formatted_lines)


@dataclass(frozen=True)
class BinaryOp:
    input0: str
    input1: str
    output: str
    operator: str
    operator_kind: str
    shape: tuple[int, ...]
    dtype: str
    input_dtype: str


@dataclass(frozen=True)
class WhereOp:
    condition: str
    input_x: str
    input_y: str
    output: str
    condition_shape: tuple[int, ...]
    x_shape: tuple[int, ...]
    y_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class NodeInfo:
    op_type: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    attrs: dict[str, object]


@dataclass(frozen=True)
class UnaryOp:
    input0: str
    output: str
    operator: str
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class CastOp:
    input0: str
    output: str
    shape: tuple[int, ...]
    input_dtype: str
    dtype: str


@dataclass(frozen=True)
class MatMulOp:
    input0: str
    input1: str
    output: str
    input0_shape: tuple[int, ...]
    input1_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    batch_shape: tuple[int, ...]
    input0_batch_shape: tuple[int, ...]
    input1_batch_shape: tuple[int, ...]
    m: int
    n: int
    k: int
    left_vector: bool
    right_vector: bool
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
    input_attn_mask: str | None
    input_past_key: str | None
    input_past_value: str | None
    input_nonpad_kv_seqlen: str | None
    output: str
    output_present_key: str | None
    output_present_value: str | None
    output_qk_matmul: str | None
    batch: int
    q_heads: int
    kv_heads: int
    q_seq: int
    kv_seq: int
    total_seq: int
    past_seq: int
    qk_head_size: int
    v_head_size: int
    q_hidden_size: int | None
    k_hidden_size: int | None
    v_hidden_size: int | None
    scale: float
    is_causal: bool
    softcap: float
    qk_matmul_output_mode: int
    q_rank: int
    k_rank: int
    v_rank: int
    output_rank: int
    mask_shape: tuple[int, ...] | None
    mask_is_bool: bool
    mask_rank: int | None
    mask_broadcast_batch: bool
    mask_broadcast_heads: bool
    mask_broadcast_q_seq: bool
    mask_q_seq: int | None
    mask_kv_seq: int | None
    head_group_size: int
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
class LogSoftmaxOp:
    input0: str
    output: str
    outer: int
    axis_size: int
    inner: int
    axis: int
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class NegativeLogLikelihoodLossOp:
    input0: str
    target: str
    weight: str | None
    output: str
    input_shape: tuple[int, ...]
    target_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    n: int
    c: int
    d: int
    reduction: str
    ignore_index: int
    dtype: str
    target_dtype: str


@dataclass(frozen=True)
class SoftmaxCrossEntropyLossOp:
    input0: str
    target: str
    weight: str | None
    output: str
    log_prob: str | None
    input_shape: tuple[int, ...]
    target_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    log_prob_shape: tuple[int, ...] | None
    n: int
    c: int
    d: int
    reduction: str
    ignore_index: int | None
    dtype: str
    target_dtype: str


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
class LstmOp:
    input_x: str
    input_w: str
    input_r: str
    input_b: str | None
    input_sequence_lens: str | None
    input_initial_h: str | None
    input_initial_c: str | None
    input_p: str | None
    output_y: str | None
    output_y_h: str | None
    output_y_c: str | None
    seq_length: int
    batch_size: int
    input_size: int
    hidden_size: int
    num_directions: int
    direction: str
    layout: int
    input_forget: int
    clip: float | None
    activation_kinds: tuple[int, ...]
    activation_alphas: tuple[float, ...]
    activation_betas: tuple[float, ...]
    dtype: str
    sequence_lens_dtype: str | None


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
class GatherElementsOp:
    data: str
    indices: str
    output: str
    axis: int
    data_shape: tuple[int, ...]
    indices_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    dtype: str
    indices_dtype: str


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
class ResizeOp:
    input0: str
    output: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    scales: tuple[float, ...]
    scales_input: str | None
    sizes_input: str | None
    roi_input: str | None
    axes: tuple[int, ...]
    scales_shape: tuple[int, ...] | None
    sizes_shape: tuple[int, ...] | None
    roi_shape: tuple[int, ...] | None
    scales_dtype: str | None
    sizes_dtype: str | None
    roi_dtype: str | None
    scales_axes: tuple[int, ...] | None
    sizes_axes: tuple[int, ...] | None
    roi_axes: tuple[int, ...] | None
    mode: str
    coordinate_transformation_mode: str
    nearest_mode: str
    cubic_coeff_a: float
    exclude_outside: bool
    extrapolation_value: float
    antialias: bool
    keep_aspect_ratio_policy: str
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
class ShapeOp:
    input0: str
    output: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    values: tuple[int, ...]
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
    input_dtypes: tuple[str, ...]
    output_names: tuple[str, ...]
    output_shapes: tuple[tuple[int, ...], ...]
    output_dtypes: tuple[str, ...]
    constants: tuple[ConstTensor, ...]
    ops: tuple[
        BinaryOp
        | WhereOp
        | UnaryOp
        | CastOp
        | MatMulOp
        | GemmOp
        | AttentionOp
        | ConvOp
        | AveragePoolOp
        | BatchNormOp
        | LrnOp
        | LstmOp
        | SoftmaxOp
        | LogSoftmaxOp
        | NegativeLogLikelihoodLossOp
        | SoftmaxCrossEntropyLossOp
        | MaxPoolOp
        | ConcatOp
        | GatherElementsOp
        | TransposeOp
        | ReshapeOp
        | ResizeOp
        | ReduceOp
        | ConstantOfShapeOp
        | ShapeOp,
        ...,
    ]
    node_infos: tuple[NodeInfo, ...]
    header: ModelHeader


class CEmitter:
    def __init__(self, template_dir: Path, *, restrict_arrays: bool = True) -> None:
        self._env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=select_autoescape(enabled_extensions=()),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self._restrict_arrays = restrict_arrays

    def _load_templates(self, emit_testbench: bool) -> dict[str, Template]:
        try:
            templates = {
                "binary": self._env.get_template("binary_op.c.j2"),
                "where": self._env.get_template("where_op.c.j2"),
                "unary": self._env.get_template("unary_op.c.j2"),
                "cast": self._env.get_template("cast_op.c.j2"),
                "matmul": self._env.get_template("matmul_op.c.j2"),
                "gemm": self._env.get_template("gemm_op.c.j2"),
                "attention": self._env.get_template("attention_op.c.j2"),
                "conv": self._env.get_template("conv_op.c.j2"),
                "avg_pool": self._env.get_template("average_pool_op.c.j2"),
                "batch_norm": self._env.get_template("batch_norm_op.c.j2"),
                "lrn": self._env.get_template("lrn_op.c.j2"),
                "lstm": self._env.get_template("lstm_op.c.j2"),
                "softmax": self._env.get_template("softmax_op.c.j2"),
                "logsoftmax": self._env.get_template("logsoftmax_op.c.j2"),
                "nllloss": self._env.get_template(
                    "negative_log_likelihood_loss_op.c.j2"
                ),
                "softmax_cross_entropy_loss": self._env.get_template(
                    "softmax_cross_entropy_loss_op.c.j2"
                ),
                "maxpool": self._env.get_template("maxpool_op.c.j2"),
                "concat": self._env.get_template("concat_op.c.j2"),
                "gather_elements": self._env.get_template("gather_elements_op.c.j2"),
                "transpose": self._env.get_template("transpose_op.c.j2"),
                "reshape": self._env.get_template("reshape_op.c.j2"),
                "resize": self._env.get_template("resize_op.c.j2"),
                "reduce": self._env.get_template("reduce_op.c.j2"),
                "constant_of_shape": self._env.get_template(
                    "constant_of_shape_op.c.j2"
                ),
                "shape": self._env.get_template("shape_op.c.j2"),
            }
            if emit_testbench:
                templates["testbench"] = self._env.get_template("testbench.c.j2")
        except Exception as exc:  # pragma: no cover - template load failure
            raise CodegenError("Failed to load C template") from exc
        return templates

    def emit_model(self, model: LoweredModel, *, emit_testbench: bool = False) -> str:
        templates = self._load_templates(emit_testbench)
        binary_template = templates["binary"]
        where_template = templates["where"]
        unary_template = templates["unary"]
        cast_template = templates["cast"]
        matmul_template = templates["matmul"]
        gemm_template = templates["gemm"]
        attention_template = templates["attention"]
        conv_template = templates["conv"]
        avg_pool_template = templates["avg_pool"]
        batch_norm_template = templates["batch_norm"]
        lrn_template = templates["lrn"]
        lstm_template = templates["lstm"]
        softmax_template = templates["softmax"]
        logsoftmax_template = templates["logsoftmax"]
        nllloss_template = templates["nllloss"]
        softmax_cross_entropy_loss_template = templates["softmax_cross_entropy_loss"]
        maxpool_template = templates["maxpool"]
        concat_template = templates["concat"]
        gather_elements_template = templates["gather_elements"]
        transpose_template = templates["transpose"]
        reshape_template = templates["reshape"]
        resize_template = templates["resize"]
        reduce_template = templates["reduce"]
        constant_of_shape_template = templates["constant_of_shape"]
        shape_template = templates["shape"]
        testbench_template = templates.get("testbench")
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
                c_type=dtype_info(op.dtype).c_type,
                zero_literal=dtype_info(op.dtype).zero_literal,
                min_literal=dtype_info(op.dtype).min_literal,
                max_literal=dtype_info(op.dtype).max_literal,
                binary_template=binary_template,
                where_template=where_template,
                unary_template=unary_template,
                cast_template=cast_template,
                matmul_template=matmul_template,
                gemm_template=gemm_template,
                attention_template=attention_template,
                conv_template=conv_template,
                avg_pool_template=avg_pool_template,
                batch_norm_template=batch_norm_template,
                lrn_template=lrn_template,
                lstm_template=lstm_template,
                softmax_template=softmax_template,
                logsoftmax_template=logsoftmax_template,
                nllloss_template=nllloss_template,
                softmax_cross_entropy_loss_template=softmax_cross_entropy_loss_template,
                maxpool_template=maxpool_template,
                concat_template=concat_template,
                gather_elements_template=gather_elements_template,
                transpose_template=transpose_template,
                reshape_template=reshape_template,
                resize_template=resize_template,
                reduce_template=reduce_template,
                constant_of_shape_template=constant_of_shape_template,
                shape_template=shape_template,
            )
            for index, op in enumerate(resolved_ops)
        )
        wrapper_fn = self._emit_model_wrapper(
            model,
            resolved_ops,
            tuple(temp_buffers.values()),
        )
        includes = self._collect_includes(
            model, resolved_ops, emit_testbench=emit_testbench
        )
        sections = [self._emit_header_comment(model.header), "", *includes, ""]
        constants_section = self._emit_constant_definitions(model.constants)
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

    def emit_model_with_data_file(
        self, model: LoweredModel, *, emit_testbench: bool = False
    ) -> tuple[str, str]:
        templates = self._load_templates(emit_testbench)
        binary_template = templates["binary"]
        where_template = templates["where"]
        unary_template = templates["unary"]
        cast_template = templates["cast"]
        matmul_template = templates["matmul"]
        gemm_template = templates["gemm"]
        attention_template = templates["attention"]
        conv_template = templates["conv"]
        avg_pool_template = templates["avg_pool"]
        batch_norm_template = templates["batch_norm"]
        lrn_template = templates["lrn"]
        lstm_template = templates["lstm"]
        softmax_template = templates["softmax"]
        logsoftmax_template = templates["logsoftmax"]
        nllloss_template = templates["nllloss"]
        softmax_cross_entropy_loss_template = templates["softmax_cross_entropy_loss"]
        maxpool_template = templates["maxpool"]
        concat_template = templates["concat"]
        gather_elements_template = templates["gather_elements"]
        transpose_template = templates["transpose"]
        reshape_template = templates["reshape"]
        resize_template = templates["resize"]
        reduce_template = templates["reduce"]
        constant_of_shape_template = templates["constant_of_shape"]
        shape_template = templates["shape"]
        testbench_template = templates.get("testbench")
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
                c_type=dtype_info(op.dtype).c_type,
                zero_literal=dtype_info(op.dtype).zero_literal,
                min_literal=dtype_info(op.dtype).min_literal,
                max_literal=dtype_info(op.dtype).max_literal,
                binary_template=binary_template,
                where_template=where_template,
                unary_template=unary_template,
                cast_template=cast_template,
                matmul_template=matmul_template,
                gemm_template=gemm_template,
                attention_template=attention_template,
                conv_template=conv_template,
                avg_pool_template=avg_pool_template,
                batch_norm_template=batch_norm_template,
                lrn_template=lrn_template,
                lstm_template=lstm_template,
                softmax_template=softmax_template,
                logsoftmax_template=logsoftmax_template,
                nllloss_template=nllloss_template,
                softmax_cross_entropy_loss_template=softmax_cross_entropy_loss_template,
                maxpool_template=maxpool_template,
                concat_template=concat_template,
                gather_elements_template=gather_elements_template,
                transpose_template=transpose_template,
                reshape_template=reshape_template,
                resize_template=resize_template,
                reduce_template=reduce_template,
                constant_of_shape_template=constant_of_shape_template,
                shape_template=shape_template,
            )
            for index, op in enumerate(resolved_ops)
        )
        wrapper_fn = self._emit_model_wrapper(
            model,
            resolved_ops,
            tuple(temp_buffers.values()),
        )
        includes = self._collect_includes(
            model, resolved_ops, emit_testbench=emit_testbench
        )
        sections = [self._emit_header_comment(model.header), "", *includes, ""]
        constants_section = self._emit_constant_declarations(model.constants)
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
        main_rendered = "\n".join(sections)
        if not main_rendered.endswith("\n"):
            main_rendered += "\n"
        data_includes = self._collect_constant_includes(model.constants)
        data_sections = [self._emit_header_comment(model.header), ""]
        if data_includes:
            data_sections.extend((*data_includes, ""))
        else:
            data_sections.append("")
        data_constants = self._emit_constant_definitions(
            model.constants, storage_prefix="const"
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
    def _collect_constant_includes(constants: tuple[ConstTensor, ...]) -> list[str]:
        if not constants:
            return []
        includes: list[str] = []
        dtypes = {const.dtype for const in constants}
        if any(
            dtype
            in {
                "int64",
                "int32",
                "int16",
                "int8",
                "uint64",
                "uint32",
                "uint16",
                "uint8",
            }
            for dtype in dtypes
        ):
            includes.append("#include <stdint.h>")
        if "bool" in dtypes:
            includes.append("#include <stdbool.h>")
        return includes

    @staticmethod
    def _collect_includes(
        model: LoweredModel,
        resolved_ops: list[
            BinaryOp
            | WhereOp
            | UnaryOp
            | CastOp
            | MatMulOp
            | GemmOp
            | AttentionOp
            | ConvOp
            | AveragePoolOp
            | BatchNormOp
            | LrnOp
            | LstmOp
            | SoftmaxOp
            | LogSoftmaxOp
            | NegativeLogLikelihoodLossOp
            | SoftmaxCrossEntropyLossOp
            | MaxPoolOp
            | ConcatOp
            | GatherElementsOp
            | TransposeOp
            | ReshapeOp
            | ResizeOp
            | ReduceOp
            | ConstantOfShapeOp
            | ShapeOp
        ],
        *,
        emit_testbench: bool,
    ) -> list[str]:
        includes: set[str] = {"#include <stddef.h>"}
        if emit_testbench:
            includes.update({"#include <stdio.h>", "#include <stdint.h>"})
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
            (nll_target_dtypes, sce_target_dtypes),
            has_resize=any(isinstance(op, ResizeOp) for op in resolved_ops),
        ):
            includes.add("#include <stdint.h>")
        if "bool" in model_dtypes:
            includes.add("#include <stdbool.h>")
        if any(
            isinstance(op, UnaryOp) and op.operator in {"llabs", "abs"}
            for op in resolved_ops
        ):
            includes.add("#include <stdlib.h>")
        if CEmitter._needs_math(resolved_ops):
            includes.add("#include <math.h>")
        if CEmitter._needs_limits(resolved_ops):
            includes.add("#include <limits.h>")
        if any(isinstance(op, (ConcatOp, ReshapeOp)) for op in resolved_ops):
            includes.add("#include <string.h>")
        ordered_includes = (
            "#include <stddef.h>",
            "#include <stdio.h>",
            "#include <stdint.h>",
            "#include <stdbool.h>",
            "#include <stdlib.h>",
            "#include <math.h>",
            "#include <limits.h>",
            "#include <string.h>",
        )
        return [include for include in ordered_includes if include in includes]

    @staticmethod
    def _needs_stdint(
        model_dtypes: set[str],
        targets: tuple[set[str], ...],
        *,
        has_resize: bool,
    ) -> bool:
        integer_dtypes = {
            "int64",
            "int32",
            "int16",
            "int8",
            "uint64",
            "uint32",
            "uint16",
            "uint8",
        }
        if any(dtype in integer_dtypes for dtype in model_dtypes):
            return True
        if any(
            dtype in {"int64", "int32"}
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
            | CastOp
            | MatMulOp
            | GemmOp
            | AttentionOp
            | ConvOp
            | AveragePoolOp
            | BatchNormOp
            | LrnOp
            | LstmOp
            | SoftmaxOp
            | LogSoftmaxOp
            | NegativeLogLikelihoodLossOp
            | SoftmaxCrossEntropyLossOp
            | MaxPoolOp
            | ConcatOp
            | GatherElementsOp
            | TransposeOp
            | ReshapeOp
            | ResizeOp
            | ReduceOp
            | ConstantOfShapeOp
            | ShapeOp
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
        if any(
            isinstance(op, UnaryOp) and op.operator in math_ops
            for op in resolved_ops
        ):
            return True
        if any(
            isinstance(op, BinaryOp) and op.operator in binary_math_ops
            for op in resolved_ops
        ):
            return True
        if any(
            isinstance(
                op,
                (
                    AttentionOp,
                    BatchNormOp,
                    LrnOp,
                    LstmOp,
                    SoftmaxOp,
                    LogSoftmaxOp,
                    SoftmaxCrossEntropyLossOp,
                    ResizeOp,
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
            and op.dtype in {"float", "double"}
            for op in resolved_ops
        ):
            return True
        if any(
            isinstance(op, MaxPoolOp) and op.dtype in {"float", "double"}
            for op in resolved_ops
        ):
            return True
        return False

    @staticmethod
    def _needs_limits(
        resolved_ops: list[
            BinaryOp
            | UnaryOp
            | CastOp
            | MatMulOp
            | GemmOp
            | AttentionOp
            | ConvOp
            | AveragePoolOp
            | BatchNormOp
            | LrnOp
            | LstmOp
            | SoftmaxOp
            | LogSoftmaxOp
            | NegativeLogLikelihoodLossOp
            | SoftmaxCrossEntropyLossOp
            | MaxPoolOp
            | ConcatOp
            | GatherElementsOp
            | TransposeOp
            | ReshapeOp
            | ResizeOp
            | ReduceOp
            | ConstantOfShapeOp
            | ShapeOp
        ],
    ) -> bool:
        if any(
            isinstance(op, ReduceOp)
            and op.reduce_kind in {"min", "max"}
            and op.dtype in {"int64", "int32", "int16", "int8"}
            for op in resolved_ops
        ):
            return True
        if any(
            isinstance(op, MaxPoolOp)
            and op.dtype in {"int64", "int32", "int16", "int8"}
            for op in resolved_ops
        ):
            return True
        return False

    def _emit_model_wrapper(
        self,
        model: LoweredModel,
        resolved_ops: list[
            BinaryOp
            | WhereOp
            | UnaryOp
            | CastOp
            | MatMulOp
            | GemmOp
            | AttentionOp
            | ConvOp
            | AveragePoolOp
            | BatchNormOp
            | LrnOp
            | LstmOp
            | SoftmaxOp
            | LogSoftmaxOp
            | NegativeLogLikelihoodLossOp
            | SoftmaxCrossEntropyLossOp
            | MaxPoolOp
            | ConcatOp
            | GatherElementsOp
            | ReshapeOp
            | ResizeOp
            | ReduceOp
            | ConstantOfShapeOp
        ],
        temp_buffers: tuple[TempBuffer, ...],
    ) -> str:
        params = [
            f"const {dtype_info(dtype).c_type} {name}"
            f"{self._param_array_suffix(shape)}"
            for name, shape, dtype in zip(
                model.input_names, model.input_shapes, model.input_dtypes
            )
        ]
        for name, shape, dtype in zip(
            model.output_names, model.output_shapes, model.output_dtypes
        ):
            output_type = dtype_info(dtype).c_type
            params.append(
                f"{output_type} {name}{self._param_array_suffix(shape)}"
            )
        signature = ", ".join(params)
        lines = [f"void {model.name}({signature}) {{"]
        for temp in temp_buffers:
            c_type = dtype_info(temp.dtype).c_type
            lines.append(
                f"    {c_type} {temp.name}{self._array_suffix(temp.shape)};"
            )
        for index, op in enumerate(resolved_ops):
            call = self._build_op_call(op)
            lines.append(f"    {model.name}_op{index}({call});")
        lines.append("}")
        return "\n".join(lines)

    @staticmethod
    def _build_op_call(
        op: BinaryOp
        | WhereOp
        | UnaryOp
        | CastOp
        | MatMulOp
        | GemmOp
        | AttentionOp
        | ConvOp
        | AveragePoolOp
        | BatchNormOp
        | LrnOp
        | LstmOp
        | SoftmaxOp
        | LogSoftmaxOp
        | NegativeLogLikelihoodLossOp
        | SoftmaxCrossEntropyLossOp
        | MaxPoolOp
        | ConcatOp
        | GatherElementsOp
        | TransposeOp
        | ReshapeOp
        | ResizeOp
        | ReduceOp
        | ConstantOfShapeOp
        | ShapeOp,
    ) -> str:
        if isinstance(op, BinaryOp):
            return f"{op.input0}, {op.input1}, {op.output}"
        if isinstance(op, WhereOp):
            return (
                f"{op.condition}, {op.input_x}, {op.input_y}, {op.output}"
            )
        if isinstance(op, MatMulOp):
            return f"{op.input0}, {op.input1}, {op.output}"
        if isinstance(op, GemmOp):
            if op.input_c is None:
                return f"{op.input_a}, {op.input_b}, {op.output}"
            return f"{op.input_a}, {op.input_b}, {op.input_c}, {op.output}"
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
            return ", ".join(call_parts)
        if isinstance(op, ConvOp):
            if op.bias is None:
                return f"{op.input0}, {op.weights}, {op.output}"
            return f"{op.input0}, {op.weights}, {op.bias}, {op.output}"
        if isinstance(op, AveragePoolOp):
            return f"{op.input0}, {op.output}"
        if isinstance(op, BatchNormOp):
            return (
                f"{op.input0}, {op.scale}, {op.bias}, "
                f"{op.mean}, {op.variance}, {op.output}"
            )
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
            return ", ".join(call_parts)
        if isinstance(op, (SoftmaxOp, LogSoftmaxOp)):
            return f"{op.input0}, {op.output}"
        if isinstance(op, NegativeLogLikelihoodLossOp):
            call_parts = [op.input0, op.target]
            if op.weight is not None:
                call_parts.append(op.weight)
            call_parts.append(op.output)
            return ", ".join(call_parts)
        if isinstance(op, SoftmaxCrossEntropyLossOp):
            call_parts = [op.input0, op.target]
            if op.weight is not None:
                call_parts.append(op.weight)
            call_parts.append(op.output)
            if op.log_prob is not None:
                call_parts.append(op.log_prob)
            return ", ".join(call_parts)
        if isinstance(op, GatherElementsOp):
            return f"{op.data}, {op.indices}, {op.output}"
        if isinstance(op, ConcatOp):
            return ", ".join((*op.inputs, op.output))
        if isinstance(op, ConstantOfShapeOp):
            return f"{op.input0}, {op.output}"
        if isinstance(op, ShapeOp):
            return f"{op.input0}, {op.output}"
        if isinstance(op, ReshapeOp):
            return f"{op.input0}, {op.output}"
        if isinstance(op, ResizeOp):
            call_parts = [op.input0]
            if op.roi_input is not None:
                call_parts.append(op.roi_input)
            if op.scales_input is not None:
                call_parts.append(op.scales_input)
            if op.sizes_input is not None:
                call_parts.append(op.sizes_input)
            call_parts.append(op.output)
            return ", ".join(call_parts)
        return f"{op.input0}, {op.output}"

    def _temp_buffers(self, model: LoweredModel) -> dict[str, TempBuffer]:
        output_names = set(model.output_names)
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
            return {name: TempBuffer(name="tmp", shape=shape, dtype=dtype)}
        return {
            name: TempBuffer(name=f"tmp{index}", shape=shape, dtype=dtype)
            for index, (name, shape, dtype) in enumerate(intermediates)
        }

    @staticmethod
    def _resolve_op(
        op: BinaryOp
        | WhereOp
        | UnaryOp
        | CastOp
        | MatMulOp
        | GemmOp
        | AttentionOp
        | ConvOp
        | AveragePoolOp
        | BatchNormOp
        | LrnOp
        | LstmOp
        | SoftmaxOp
        | LogSoftmaxOp
        | NegativeLogLikelihoodLossOp
        | SoftmaxCrossEntropyLossOp
        | MaxPoolOp
        | ConcatOp
        | GatherElementsOp
        | TransposeOp
        | ReshapeOp
        | ResizeOp
        | ReduceOp
        | ConstantOfShapeOp
        | ShapeOp,
        temp_map: dict[str, str],
    ) -> (
        BinaryOp
        | WhereOp
        | UnaryOp
        | CastOp
        | MatMulOp
        | GemmOp
        | AttentionOp
        | ConvOp
        | AveragePoolOp
        | BatchNormOp
        | LrnOp
        | LstmOp
        | SoftmaxOp
        | LogSoftmaxOp
        | NegativeLogLikelihoodLossOp
        | SoftmaxCrossEntropyLossOp
        | MaxPoolOp
        | ConcatOp
        | GatherElementsOp
        | TransposeOp
        | ReshapeOp
        | ResizeOp
        | ReduceOp
        | ConstantOfShapeOp
        | ShapeOp
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
        if isinstance(op, CastOp):
            return CastOp(
                input0=temp_map.get(op.input0, op.input0),
                output=temp_map.get(op.output, op.output),
                shape=op.shape,
                input_dtype=op.input_dtype,
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
                dtype=op.dtype,
                target_dtype=op.target_dtype,
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

    def _render_op(
        self,
        model: LoweredModel,
        op: BinaryOp
        | WhereOp
        | UnaryOp
        | CastOp
        | MatMulOp
        | GemmOp
        | AttentionOp
        | ConvOp
        | AveragePoolOp
        | BatchNormOp
        | LrnOp
        | LstmOp
        | SoftmaxOp
        | LogSoftmaxOp
        | NegativeLogLikelihoodLossOp
        | SoftmaxCrossEntropyLossOp
        | MaxPoolOp
        | ConcatOp
        | GatherElementsOp
        | TransposeOp
        | ReshapeOp
        | ResizeOp
        | ReduceOp
        | ConstantOfShapeOp
        | ShapeOp,
        index: int,
        *,
        array_suffix: str,
        loop_vars: tuple[str, ...],
        c_type: str,
        zero_literal: str,
        min_literal: str,
        max_literal: str,
        binary_template,
        where_template,
        unary_template,
        cast_template,
        matmul_template,
        gemm_template,
        attention_template,
        conv_template,
        avg_pool_template,
        batch_norm_template,
        lrn_template,
        lstm_template,
        softmax_template,
        logsoftmax_template,
        nllloss_template,
        softmax_cross_entropy_loss_template,
        maxpool_template,
        concat_template,
        gather_elements_template,
        transpose_template,
        reshape_template,
        resize_template,
        reduce_template,
        constant_of_shape_template,
        shape_template,
    ) -> str:
        node_comment = CEmitter._emit_node_comment(model.node_infos[index], index)

        def with_node_comment(rendered: str) -> str:
            return f"{node_comment}\n{_format_c_indentation(rendered)}"

        if isinstance(op, BinaryOp):
            shape = CEmitter._codegen_shape(op.shape)
            loop_vars = CEmitter._loop_vars(shape)
            array_suffix = self._param_array_suffix(shape)
            input_c_type = dtype_info(op.input_dtype).c_type
            output_c_type = dtype_info(op.dtype).c_type
            common = {
                "model_name": model.name,
                "op_name": f"{model.name}_op{index}",
                "element_count": CEmitter._element_count(shape),
                "array_suffix": array_suffix,
                "shape": shape,
                "loop_vars": loop_vars,
                "input_c_type": input_c_type,
                "output_c_type": output_c_type,
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
            rendered = binary_template.render(
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
            return with_node_comment(rendered)
        if isinstance(op, WhereOp):
            output_shape = CEmitter._codegen_shape(op.output_shape)
            loop_vars = CEmitter._loop_vars(output_shape)
            loop_indents = CEmitter._loop_indents(output_shape)
            inner_indent = CEmitter._inner_indent(output_shape)
            output_array_suffix = self._param_array_suffix(output_shape)
            condition_array_suffix = self._param_array_suffix(op.condition_shape)
            x_array_suffix = self._param_array_suffix(op.x_shape)
            y_array_suffix = self._param_array_suffix(op.y_shape)
            condition_expr = CEmitter._broadcast_index_expr(
                op.condition, op.condition_shape, op.output_shape, loop_vars
            )
            x_expr = CEmitter._broadcast_index_expr(
                op.input_x, op.x_shape, op.output_shape, loop_vars
            )
            y_expr = CEmitter._broadcast_index_expr(
                op.input_y, op.y_shape, op.output_shape, loop_vars
            )
            output_expr = f"{op.output}" + "".join(
                f"[{var}]" for var in loop_vars
            )
            rendered = where_template.render(
                model_name=model.name,
                op_name=f"{model.name}_op{index}",
                output_shape=output_shape,
                loop_vars=loop_vars,
                loop_indents=loop_indents,
                inner_indent=inner_indent,
                condition=op.condition,
                input_x=op.input_x,
                input_y=op.input_y,
                output=op.output,
                condition_array_suffix=condition_array_suffix,
                x_array_suffix=x_array_suffix,
                y_array_suffix=y_array_suffix,
                output_array_suffix=output_array_suffix,
                condition_expr=condition_expr,
                x_expr=x_expr,
                y_expr=y_expr,
                output_expr=output_expr,
                input_c_type=dtype_info(op.dtype).c_type,
                output_c_type=dtype_info(op.dtype).c_type,
                condition_c_type=dtype_info("bool").c_type,
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, MatMulOp):
            output_shape = CEmitter._codegen_shape(op.output_shape)
            output_loop_vars = CEmitter._loop_vars(output_shape)
            output_index_expr = f"{op.output}" + "".join(
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
            )
            rendered = matmul_template.render(
                model_name=model.name,
                op_name=f"{model.name}_op{index}",
                input0=op.input0,
                input1=op.input1,
                output=op.output,
                c_type=c_type,
                acc_type=c_type,
                zero_literal=zero_literal,
                input0_suffix=self._param_array_suffix(op.input0_shape),
                input1_suffix=self._param_array_suffix(op.input1_shape),
                output_suffix=self._param_array_suffix(op.output_shape),
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
            rendered = gemm_template.render(
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
                input_a_suffix=self._param_array_suffix(input_a_shape),
                input_b_suffix=self._param_array_suffix(input_b_shape),
                output_suffix=self._param_array_suffix((op.m, op.n)),
                c_suffix=(
                    self._param_array_suffix(op.c_shape)
                    if op.c_shape is not None
                    else None
                ),
                c_rank=c_rank,
                c_dim0=c_dim0,
                c_dim1=c_dim1,
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, AttentionOp):
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
            rendered = attention_template.render(
                model_name=model.name,
                op_name=f"{model.name}_op{index}",
                input_q=op.input_q,
                input_k=op.input_k,
                input_v=op.input_v,
                input_attn_mask=op.input_attn_mask,
                input_past_key=op.input_past_key,
                input_past_value=op.input_past_value,
                input_nonpad_kv_seqlen=op.input_nonpad_kv_seqlen,
                output=op.output,
                output_present_key=op.output_present_key,
                output_present_value=op.output_present_value,
                output_qk_matmul=op.output_qk_matmul,
                c_type=c_type,
                nonpad_c_type=dtype_info("int64").c_type,
                zero_literal=zero_literal,
                min_literal=min_literal,
                scale_literal=CEmitter._format_floating(op.scale, op.dtype),
                softcap_literal=CEmitter._format_floating(op.softcap, op.dtype),
                one_literal=CEmitter._format_literal(op.dtype, 1),
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
                input_q_suffix=self._param_array_suffix(input_q_shape),
                input_k_suffix=self._param_array_suffix(input_k_shape),
                input_v_suffix=self._param_array_suffix(input_v_shape),
                input_mask_suffix=(
                    self._param_array_suffix(op.mask_shape)
                    if op.input_attn_mask is not None
                    else ""
                ),
                input_past_key_suffix=(
                    self._param_array_suffix(
                        (op.batch, op.kv_heads, op.past_seq, op.qk_head_size)
                    )
                    if op.input_past_key is not None
                    else ""
                ),
                input_past_value_suffix=(
                    self._param_array_suffix(
                        (op.batch, op.kv_heads, op.past_seq, op.v_head_size)
                    )
                    if op.input_past_value is not None
                    else ""
                ),
                input_nonpad_suffix=(
                    self._param_array_suffix((op.batch,))
                    if op.input_nonpad_kv_seqlen is not None
                    else ""
                ),
                output_suffix=self._param_array_suffix(output_shape),
                output_present_key_suffix=(
                    self._param_array_suffix(present_key_shape)
                    if present_key_shape is not None
                    else ""
                ),
                output_present_value_suffix=(
                    self._param_array_suffix(present_value_shape)
                    if present_value_shape is not None
                    else ""
                ),
                output_qk_matmul_suffix=(
                    self._param_array_suffix(qk_matmul_shape)
                    if qk_matmul_shape is not None
                    else ""
                ),
            ).rstrip()
            return with_node_comment(rendered)
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
            rendered = conv_template.render(
                model_name=model.name,
                op_name=f"{model.name}_op{index}",
                input0=op.input0,
                weights=op.weights,
                bias=op.bias,
                output=op.output,
                c_type=c_type,
                zero_literal=zero_literal,
                input_suffix=self._param_array_suffix(input_shape),
                weight_suffix=self._param_array_suffix(weight_shape),
                bias_suffix=self._param_array_suffix((op.out_channels,)),
                output_suffix=self._param_array_suffix(output_shape),
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
        if isinstance(op, AveragePoolOp):
            input_shape = (op.batch, op.channels, op.in_h, op.in_w)
            output_shape = (op.batch, op.channels, op.out_h, op.out_w)
            rendered = avg_pool_template.render(
                model_name=model.name,
                op_name=f"{model.name}_op{index}",
                input0=op.input0,
                output=op.output,
                c_type=c_type,
                zero_literal=zero_literal,
                input_suffix=self._param_array_suffix(input_shape),
                output_suffix=self._param_array_suffix(output_shape),
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
        if isinstance(op, BatchNormOp):
            shape = CEmitter._codegen_shape(op.shape)
            loop_vars = CEmitter._loop_vars(shape)
            rendered = batch_norm_template.render(
                model_name=model.name,
                op_name=f"{model.name}_op{index}",
                input0=op.input0,
                scale=op.scale,
                bias=op.bias,
                mean=op.mean,
                variance=op.variance,
                output=op.output,
                c_type=c_type,
                input_suffix=self._param_array_suffix(shape),
                output_suffix=self._param_array_suffix(shape),
                scale_suffix=self._param_array_suffix((op.channels,)),
                bias_suffix=self._param_array_suffix((op.channels,)),
                mean_suffix=self._param_array_suffix((op.channels,)),
                variance_suffix=self._param_array_suffix((op.channels,)),
                shape=shape,
                loop_vars=loop_vars,
                epsilon_literal=CEmitter._format_floating(op.epsilon, op.dtype),
                sqrt_fn=CEmitter._math_fn(op.dtype, "sqrtf", "sqrt"),
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, LrnOp):
            shape = CEmitter._codegen_shape(op.shape)
            loop_vars = CEmitter._loop_vars(shape)
            rendered = lrn_template.render(
                model_name=model.name,
                op_name=f"{model.name}_op{index}",
                input0=op.input0,
                output=op.output,
                c_type=c_type,
                input_suffix=self._param_array_suffix(shape),
                output_suffix=self._param_array_suffix(shape),
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
            h_shape = (
                (op.num_directions, op.batch_size, op.hidden_size)
                if op.input_initial_h is not None
                else None
            )
            c_shape = (
                (op.num_directions, op.batch_size, op.hidden_size)
                if op.input_initial_c is not None
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
            rendered = lstm_template.render(
                model_name=model.name,
                op_name=f"{model.name}_op{index}",
                input_x=op.input_x,
                input_w=op.input_w,
                input_r=op.input_r,
                input_b=op.input_b,
                input_sequence_lens=op.input_sequence_lens,
                input_initial_h=op.input_initial_h,
                input_initial_c=op.input_initial_c,
                input_p=op.input_p,
                output_y=op.output_y,
                output_y_h=op.output_y_h,
                output_y_c=op.output_y_c,
                c_type=c_type,
                seq_c_type=dtype_info(op.sequence_lens_dtype or "int64").c_type,
                zero_literal=zero_literal,
                one_literal=CEmitter._format_literal(op.dtype, 1),
                clip_literal=(
                    CEmitter._format_floating(op.clip, op.dtype)
                    if op.clip is not None
                    else CEmitter._format_literal(op.dtype, 0)
                ),
                use_clip=int(op.clip is not None and op.clip > 0),
                input_suffix=self._param_array_suffix(input_x_shape),
                w_suffix=self._param_array_suffix(w_shape),
                r_suffix=self._param_array_suffix(r_shape),
                b_suffix=self._param_array_suffix(b_shape) if b_shape else None,
                seq_suffix=self._param_array_suffix(seq_shape) if seq_shape else None,
                h_suffix=self._param_array_suffix(h_shape) if h_shape else None,
                c_suffix=self._param_array_suffix(c_shape) if c_shape else None,
                p_suffix=self._param_array_suffix(p_shape) if p_shape else None,
                y_suffix=self._param_array_suffix(y_shape) if op.output_y else None,
                y_h_suffix=(
                    self._param_array_suffix((op.num_directions, op.batch_size, op.hidden_size))
                    if op.output_y_h
                    else None
                ),
                y_c_suffix=(
                    self._param_array_suffix((op.num_directions, op.batch_size, op.hidden_size))
                    if op.output_y_c
                    else None
                ),
                seq_length=op.seq_length,
                batch_size=op.batch_size,
                input_size=op.input_size,
                hidden_size=op.hidden_size,
                num_directions=op.num_directions,
                layout=op.layout,
                direction=op.direction,
                input_forget=op.input_forget,
                activation_kinds=op.activation_kinds,
                activation_alphas=tuple(
                    CEmitter._format_floating(value, op.dtype)
                    for value in op.activation_alphas
                ),
                activation_betas=tuple(
                    CEmitter._format_floating(value, op.dtype)
                    for value in op.activation_betas
                ),
                exp_fn=CEmitter._math_fn(op.dtype, "expf", "exp"),
                tanh_fn=CEmitter._math_fn(op.dtype, "tanhf", "tanh"),
                log1p_fn=CEmitter._math_fn(op.dtype, "log1pf", "log1p"),
                fabs_fn=CEmitter._math_fn(op.dtype, "fabsf", "fabs"),
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, SoftmaxOp):
            rendered = softmax_template.render(
                model_name=model.name,
                op_name=f"{model.name}_op{index}",
                input0=op.input0,
                output=op.output,
                c_type=c_type,
                array_suffix=self._param_array_suffix(op.shape),
                outer=op.outer,
                axis_size=op.axis_size,
                inner=op.inner,
                exp_fn=CEmitter._math_fn(op.dtype, "expf", "exp"),
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, LogSoftmaxOp):
            rendered = logsoftmax_template.render(
                model_name=model.name,
                op_name=f"{model.name}_op{index}",
                input0=op.input0,
                output=op.output,
                c_type=c_type,
                array_suffix=self._param_array_suffix(op.shape),
                outer=op.outer,
                axis_size=op.axis_size,
                inner=op.inner,
                exp_fn=CEmitter._math_fn(op.dtype, "expf", "exp"),
                log_fn=CEmitter._math_fn(op.dtype, "logf", "log"),
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, NegativeLogLikelihoodLossOp):
            rendered = nllloss_template.render(
                model_name=model.name,
                op_name=f"{model.name}_op{index}",
                input0=op.input0,
                target=op.target,
                weight=op.weight,
                output=op.output,
                c_type=c_type,
                target_c_type=dtype_info(op.target_dtype).c_type,
                input_suffix=self._param_array_suffix(op.input_shape),
                target_suffix=self._param_array_suffix(op.target_shape),
                output_suffix=self._param_array_suffix(op.output_shape),
                n=op.n,
                c=op.c,
                d=op.d,
                reduction=op.reduction,
                ignore_index=op.ignore_index,
                zero_literal=zero_literal,
                one_literal=CEmitter._format_literal(op.dtype, 1),
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, SoftmaxCrossEntropyLossOp):
            use_ignore_index = int(op.ignore_index is not None)
            ignore_index = op.ignore_index if op.ignore_index is not None else -1
            rendered = softmax_cross_entropy_loss_template.render(
                model_name=model.name,
                op_name=f"{model.name}_op{index}",
                input0=op.input0,
                target=op.target,
                weight=op.weight,
                output=op.output,
                log_prob=op.log_prob,
                c_type=c_type,
                target_c_type=dtype_info(op.target_dtype).c_type,
                input_suffix=self._param_array_suffix(op.input_shape),
                target_suffix=self._param_array_suffix(op.target_shape),
                output_suffix=self._param_array_suffix(op.output_shape),
                log_prob_suffix=(
                    self._param_array_suffix(op.log_prob_shape)
                    if op.log_prob_shape is not None
                    else None
                ),
                n=op.n,
                c=op.c,
                d=op.d,
                reduction=op.reduction,
                use_ignore_index=use_ignore_index,
                ignore_index=ignore_index,
                zero_literal=zero_literal,
                one_literal=CEmitter._format_literal(op.dtype, 1),
                exp_fn=CEmitter._math_fn(op.dtype, "expf", "exp"),
                log_fn=CEmitter._math_fn(op.dtype, "logf", "log"),
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, MaxPoolOp):
            input_shape = (op.batch, op.channels, *op.in_spatial)
            output_shape = (op.batch, op.channels, *op.out_spatial)
            rendered = maxpool_template.render(
                model_name=model.name,
                op_name=f"{model.name}_op{index}",
                input0=op.input0,
                output=op.output,
                c_type=c_type,
                min_literal=min_literal,
                input_suffix=self._param_array_suffix(input_shape),
                output_suffix=self._param_array_suffix(output_shape),
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
            return with_node_comment(rendered)
        if isinstance(op, ConcatOp):
            axis = op.axis
            if axis < 0:
                axis += len(op.output_shape)
            outer = CEmitter._element_count(op.output_shape[:axis] or (1,))
            inner = CEmitter._element_count(op.output_shape[axis + 1 :] or (1,))
            axis_sizes = tuple(shape[axis] for shape in op.input_shapes)
            rendered = concat_template.render(
                model_name=model.name,
                op_name=f"{model.name}_op{index}",
                inputs=op.inputs,
                output=op.output,
                c_type=c_type,
                input_suffixes=tuple(
                    self._param_array_suffix(shape) for shape in op.input_shapes
                ),
                output_suffix=self._param_array_suffix(op.output_shape),
                axis_sizes=axis_sizes,
                input_count=len(op.inputs),
                outer=outer,
                inner=inner,
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, GatherElementsOp):
            output_shape = CEmitter._codegen_shape(op.output_shape)
            loop_vars = CEmitter._loop_vars(output_shape)
            loop_indents = CEmitter._loop_indents(output_shape)
            inner_indent = CEmitter._inner_indent(output_shape)
            data_indices = list(loop_vars)
            data_indices[op.axis] = "gather_index"
            rendered = gather_elements_template.render(
                model_name=model.name,
                op_name=f"{model.name}_op{index}",
                data=op.data,
                indices=op.indices,
                output=op.output,
                c_type=c_type,
                indices_c_type=dtype_info(op.indices_dtype).c_type,
                data_suffix=self._param_array_suffix(op.data_shape),
                indices_suffix=self._param_array_suffix(op.indices_shape),
                output_suffix=self._param_array_suffix(op.output_shape),
                output_shape=output_shape,
                loop_vars=loop_vars,
                loop_indents=loop_indents,
                inner_indent=inner_indent,
                data_indices=data_indices,
                axis_dim=op.data_shape[op.axis],
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, TransposeOp):
            output_shape = CEmitter._codegen_shape(op.output_shape)
            loop_vars = CEmitter._loop_vars(output_shape)
            output_suffix = self._param_array_suffix(output_shape)
            input_suffix = self._param_array_suffix(op.input_shape)
            if not op.input_shape:
                input_indices = [loop_vars[0]]
            else:
                input_indices = [None] * len(op.perm)
                for output_axis, input_axis in enumerate(op.perm):
                    input_indices[input_axis] = loop_vars[output_axis]
            rendered = transpose_template.render(
                model_name=model.name,
                op_name=f"{model.name}_op{index}",
                input0=op.input0,
                output=op.output,
                c_type=c_type,
                input_suffix=input_suffix,
                output_suffix=output_suffix,
                output_shape=output_shape,
                loop_vars=loop_vars,
                input_indices=input_indices,
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, ReshapeOp):
            rendered = reshape_template.render(
                model_name=model.name,
                op_name=f"{model.name}_op{index}",
                input0=op.input0,
                output=op.output,
                c_type=c_type,
                input_suffix=self._param_array_suffix(op.input_shape),
                output_suffix=self._param_array_suffix(op.output_shape),
                element_count=CEmitter._element_count(op.output_shape),
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, ResizeOp):
            input_suffix = self._param_array_suffix(op.input_shape)
            output_suffix = self._param_array_suffix(op.output_shape)
            params = [f"const {c_type} {op.input0}{input_suffix}"]
            roi_suffix = None
            scales_suffix = None
            sizes_suffix = None
            roi_c_type = None
            scales_c_type = None
            sizes_c_type = None
            if op.roi_input and op.roi_shape and op.roi_dtype:
                roi_suffix = self._param_array_suffix(op.roi_shape)
                roi_c_type = dtype_info(op.roi_dtype).c_type
                params.append(
                    f"const {roi_c_type} {op.roi_input}{roi_suffix}"
                )
            if op.scales_input and op.scales_shape and op.scales_dtype:
                scales_suffix = self._param_array_suffix(op.scales_shape)
                scales_c_type = dtype_info(op.scales_dtype).c_type
                params.append(
                    f"const {scales_c_type} {op.scales_input}{scales_suffix}"
                )
            if op.sizes_input and op.sizes_shape and op.sizes_dtype:
                sizes_suffix = self._param_array_suffix(op.sizes_shape)
                sizes_c_type = dtype_info(op.sizes_dtype).c_type
                params.append(
                    f"const {sizes_c_type} {op.sizes_input}{sizes_suffix}"
                )
            params.append(f"{c_type} {op.output}{output_suffix}")
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
                op_name=f"{model.name}_op{index}",
                params=params,
                input0=op.input0,
                output=op.output,
                c_type=c_type,
                input_suffix=input_suffix,
                output_suffix=output_suffix,
                input_shape=op.input_shape,
                output_shape=op.output_shape,
                rank=len(op.input_shape),
                loop_vars=CEmitter._loop_vars(op.output_shape),
                scales=op.scales,
                scales_input=op.scales_input,
                sizes_input=op.sizes_input,
                roi_input=op.roi_input,
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
        if isinstance(op, ReduceOp):
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
            value_expr = f"{op.input0}{input_index_expr}"
            update_expr = None
            init_literal = None
            final_expr = "acc"
            fabs_fn = CEmitter._math_fn(op.dtype, "fabsf", "fabs")
            exp_fn = CEmitter._math_fn(op.dtype, "expf", "exp")
            log_fn = CEmitter._math_fn(op.dtype, "logf", "log")
            sqrt_fn = CEmitter._math_fn(op.dtype, "sqrtf", "sqrt")
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
            rendered = reduce_template.render(
                model_name=model.name,
                op_name=f"{model.name}_op{index}",
                input0=op.input0,
                output=op.output,
                c_type=c_type,
                input_suffix=self._param_array_suffix(op.input_shape),
                output_suffix=self._param_array_suffix(op.output_shape),
                output_shape=output_shape,
                output_loop_vars=output_loop_vars,
                reduce_loop_vars=reduce_loop_vars,
                reduce_dims=reduce_dims,
                output_index_expr=output_index_expr,
                init_literal=init_literal,
                update_expr=update_expr,
                final_expr=final_expr,
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, ConstantOfShapeOp):
            shape = CEmitter._codegen_shape(op.shape)
            loop_vars = CEmitter._loop_vars(shape)
            array_suffix = self._param_array_suffix(shape)
            rendered = constant_of_shape_template.render(
                model_name=model.name,
                op_name=f"{model.name}_op{index}",
                input0=op.input0,
                output=op.output,
                input_c_type=dtype_info(op.input_dtype).c_type,
                c_type=c_type,
                input_suffix=self._param_array_suffix(op.input_shape),
                array_suffix=array_suffix,
                shape=shape,
                loop_vars=loop_vars,
                value_literal=CEmitter._format_literal(op.dtype, op.value),
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, ShapeOp):
            rendered = shape_template.render(
                model_name=model.name,
                op_name=f"{model.name}_op{index}",
                input0=op.input0,
                output=op.output,
                input_c_type=dtype_info(op.input_dtype).c_type,
                c_type=c_type,
                input_suffix=self._param_array_suffix(op.input_shape),
                output_suffix=self._param_array_suffix(op.output_shape),
                values=[
                    CEmitter._format_literal(op.dtype, value)
                    for value in op.values
                ],
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, CastOp):
            shape = CEmitter._codegen_shape(op.shape)
            loop_vars = CEmitter._loop_vars(shape)
            array_suffix = self._param_array_suffix(shape)
            rendered = cast_template.render(
                model_name=model.name,
                op_name=f"{model.name}_op{index}",
                input0=op.input0,
                output=op.output,
                input_c_type=dtype_info(op.input_dtype).c_type,
                output_c_type=dtype_info(op.dtype).c_type,
                array_suffix=array_suffix,
                shape=shape,
                loop_vars=loop_vars,
            ).rstrip()
            return with_node_comment(rendered)
        if isinstance(op, UnaryOp):
            shape = CEmitter._codegen_shape(op.shape)
            loop_vars = CEmitter._loop_vars(shape)
            array_suffix = self._param_array_suffix(shape)
            common = {
                "model_name": model.name,
                "op_name": f"{model.name}_op{index}",
                "element_count": CEmitter._element_count(shape),
                "array_suffix": array_suffix,
                "shape": shape,
                "loop_vars": loop_vars,
                "c_type": c_type,
                "zero_literal": zero_literal,
            }
            rendered = unary_template.render(
                **common,
                input0=op.input0,
                output=op.output,
                operator=op.operator,
            ).rstrip()
            return with_node_comment(rendered)
        raise CodegenError(f"Unsupported op for rendering: {type(op).__name__}")

    @staticmethod
    def _op_output(
        op: BinaryOp
        | WhereOp
        | UnaryOp
        | CastOp
        | MatMulOp
        | GemmOp
        | AttentionOp
        | ConvOp
        | AveragePoolOp
        | BatchNormOp
        | LrnOp
        | LstmOp
        | SoftmaxOp
        | LogSoftmaxOp
        | NegativeLogLikelihoodLossOp
        | SoftmaxCrossEntropyLossOp
        | MaxPoolOp
        | ConcatOp
        | GatherElementsOp
        | TransposeOp
        | ReshapeOp
        | ResizeOp
        | ReduceOp
        | ConstantOfShapeOp
        | ShapeOp,
    ) -> str:
        return op.output

    @staticmethod
    def _op_outputs(
        op: BinaryOp
        | WhereOp
        | UnaryOp
        | CastOp
        | MatMulOp
        | GemmOp
        | AttentionOp
        | ConvOp
        | AveragePoolOp
        | BatchNormOp
        | LrnOp
        | LstmOp
        | SoftmaxOp
        | LogSoftmaxOp
        | NegativeLogLikelihoodLossOp
        | SoftmaxCrossEntropyLossOp
        | MaxPoolOp
        | ConcatOp
        | GatherElementsOp
        | TransposeOp
        | ReshapeOp
        | ResizeOp
        | ReduceOp
        | ConstantOfShapeOp
        | ShapeOp,
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
        if isinstance(op, SoftmaxCrossEntropyLossOp):
            outputs = [(op.output, op.output_shape, op.dtype)]
            if op.log_prob is not None and op.log_prob_shape is not None:
                outputs.append((op.log_prob, op.log_prob_shape, op.dtype))
            return tuple(outputs)
        return ((op.output, CEmitter._op_output_shape(op), op.dtype),)

    @staticmethod
    def _op_output_shape(
        op: BinaryOp
        | WhereOp
        | UnaryOp
        | CastOp
        | MatMulOp
        | GemmOp
        | AttentionOp
        | ConvOp
        | AveragePoolOp
        | BatchNormOp
        | LrnOp
        | LstmOp
        | SoftmaxOp
        | LogSoftmaxOp
        | NegativeLogLikelihoodLossOp
        | SoftmaxCrossEntropyLossOp
        | MaxPoolOp
        | ConcatOp
        | GatherElementsOp
        | TransposeOp
        | ReshapeOp
        | ResizeOp
        | ReduceOp
        | ConstantOfShapeOp,
    ) -> tuple[int, ...]:
        if isinstance(op, BinaryOp):
            return op.shape
        if isinstance(op, WhereOp):
            return op.output_shape
        if isinstance(op, UnaryOp):
            return op.shape
        if isinstance(op, CastOp):
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
        if isinstance(op, LogSoftmaxOp):
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
        if isinstance(op, TransposeOp):
            return op.output_shape
        if isinstance(op, ReshapeOp):
            return op.output_shape
        if isinstance(op, ResizeOp):
            return op.output_shape
        if isinstance(op, ReduceOp):
            return op.output_shape
        if isinstance(op, ConstantOfShapeOp):
            return op.shape
        if isinstance(op, ShapeOp):
            return op.output_shape
        if op.output_rank == 3:
            return (op.batch, op.q_seq, op.q_heads * op.v_head_size)
        return (op.batch, op.q_heads, op.q_seq, op.v_head_size)

    @staticmethod
    def _op_output_dtype(
        op: BinaryOp
        | WhereOp
        | UnaryOp
        | MatMulOp
        | GemmOp
        | AttentionOp
        | ConvOp
        | AveragePoolOp
        | BatchNormOp
        | SoftmaxOp
        | LogSoftmaxOp
        | NegativeLogLikelihoodLossOp
        | SoftmaxCrossEntropyLossOp
        | MaxPoolOp
        | ConcatOp
        | GatherElementsOp
        | TransposeOp
        | ReshapeOp
        | ResizeOp
        | ReduceOp
        | ConstantOfShapeOp
        | ShapeOp,
    ) -> str:
        return op.dtype

    @staticmethod
    def _codegen_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
        if not shape:
            return (1,)
        return shape

    @staticmethod
    def _array_suffix(shape: tuple[int, ...]) -> str:
        shape = CEmitter._codegen_shape(shape)
        return "".join(f"[{dim}]" for dim in shape)

    def _param_array_suffix(self, shape: tuple[int, ...]) -> str:
        shape = CEmitter._codegen_shape(shape)
        if not self._restrict_arrays:
            return "".join(f"[{dim}]" for dim in shape)
        first, *rest = shape
        return f"[restrict {first}]" + "".join(f"[{dim}]" for dim in rest)

    @staticmethod
    def _loop_vars(shape: tuple[int, ...]) -> tuple[str, ...]:
        shape = CEmitter._codegen_shape(shape)
        return tuple(f"i{index}" for index in range(len(shape)))

    @staticmethod
    def _element_count(shape: tuple[int, ...]) -> int:
        shape = CEmitter._codegen_shape(shape)
        count = 1
        for dim in shape:
            if dim <= 0:
                raise CodegenError("Dynamic or zero dims are not supported")
            count *= dim
        return count

    @staticmethod
    def _matmul_index_exprs(
        op: MatMulOp,
        batch_vars: tuple[str, ...],
        row_var: str | None,
        col_var: str | None,
        batch_rank: int,
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
        input0_index_expr = f"{op.input0}" + "".join(
            f"[{index}]" for index in input0_indices
        )
        input1_index_expr = f"{op.input1}" + "".join(
            f"[{index}]" for index in input1_indices
        )
        return input0_index_expr, input1_index_expr

    def _emit_testbench(self, model: LoweredModel, testbench_template) -> str:
        input_counts = tuple(
            self._element_count(shape) for shape in model.input_shapes
        )
        inputs = []
        for name, shape, count, dtype in zip(
            model.input_names, model.input_shapes, input_counts, model.input_dtypes
        ):
            info = dtype_info(dtype)
            codegen_shape = self._codegen_shape(shape)
            loop_vars = self._loop_vars(codegen_shape)
            if dtype == "float":
                random_expr = "rng_next_float()"
            elif dtype == "double":
                random_expr = "rng_next_double()"
            elif dtype == "bool":
                random_expr = "((rng_next_u64() & 1ull) != 0)"
            else:
                random_expr = f"({info.c_type})rng_next_i64()"
            inputs.append(
                {
                    "name": name,
                    "shape": codegen_shape,
                    "shape_literal": ",".join(str(dim) for dim in shape),
                    "count": count,
                    "array_suffix": self._array_suffix(codegen_shape),
                    "loop_vars": loop_vars,
                    "rank": len(codegen_shape),
                    "index_expr": self._index_expr(codegen_shape, loop_vars),
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
            codegen_shape = self._codegen_shape(shape)
            output_loop_vars = self._loop_vars(codegen_shape)
            outputs.append(
                {
                    "name": name,
                    "shape": codegen_shape,
                    "shape_literal": ",".join(str(dim) for dim in shape),
                    "count": self._element_count(codegen_shape),
                    "array_suffix": self._array_suffix(codegen_shape),
                    "loop_vars": output_loop_vars,
                    "rank": len(codegen_shape),
                    "index_expr": self._index_expr(codegen_shape, output_loop_vars),
                    "dtype": dtype,
                    "c_type": output_info.c_type,
                    "print_format": self._print_format(dtype),
                    "print_cast": self._print_cast(dtype),
                }
            )
        rendered = testbench_template.render(
            model_name=model.name,
            inputs=inputs,
            outputs=outputs,
        ).rstrip()
        return _format_c_indentation(rendered)

    def _emit_constant_definitions(
        self,
        constants: tuple[ConstTensor, ...],
        *,
        storage_prefix: str = "static const",
    ) -> str:
        if not constants:
            return ""
        lines: list[str] = []
        for const in constants:
            c_type = dtype_info(const.dtype).c_type
            array_suffix = self._array_suffix(const.shape)
            values = [
                self._format_value(value, const.dtype) for value in const.data
            ]
            lines.append(
                f"{storage_prefix} {c_type} {const.name}{array_suffix} = {{"
            )
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

    def _emit_constant_declarations(
        self, constants: tuple[ConstTensor, ...]
    ) -> str:
        if not constants:
            return ""
        lines = []
        for const in constants:
            c_type = dtype_info(const.dtype).c_type
            array_suffix = self._array_suffix(const.shape)
            lines.append(f"extern const {c_type} {const.name}{array_suffix};")
        return "\n".join(lines)

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
        formatted = f"{value:.9g}"
        if "e" not in formatted and "E" not in formatted and "." not in formatted:
            formatted = f"{formatted}.0"
        return f"{formatted}f"

    @staticmethod
    def _format_double(value: float) -> str:
        formatted = f"{value:.17g}"
        if "e" not in formatted and "E" not in formatted and "." not in formatted:
            formatted = f"{formatted}.0"
        return formatted

    @staticmethod
    def _format_floating(value: float, dtype: str) -> str:
        if dtype == "double":
            return CEmitter._format_double(value)
        return CEmitter._format_float(value)

    @staticmethod
    def _math_fn(dtype: str, float_name: str, double_name: str) -> str:
        if dtype == "double":
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
    def _format_literal(dtype: str, value: float | int | bool) -> str:
        if dtype == "float":
            return CEmitter._format_float(float(value))
        if dtype == "double":
            return CEmitter._format_double(float(value))
        if dtype == "bool":
            return "true" if bool(value) else "false"
        if dtype == "uint64":
            return CEmitter._format_uint(int(value), 64, "UINT64_MAX")
        if dtype == "uint32":
            return CEmitter._format_uint(int(value), 32, "UINT32_MAX")
        if dtype == "uint16":
            return CEmitter._format_uint(int(value), 16, "UINT16_MAX")
        if dtype == "uint8":
            return CEmitter._format_uint(int(value), 8, "UINT8_MAX")
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
        if dtype == "double":
            return self._format_double(float(value))
        if dtype == "bool":
            return "true" if bool(value) else "false"
        if dtype == "uint64":
            return self._format_uint(int(value), 64, "UINT64_MAX")
        if dtype == "uint32":
            return self._format_uint(int(value), 32, "UINT32_MAX")
        if dtype == "uint16":
            return self._format_uint(int(value), 16, "UINT16_MAX")
        if dtype == "uint8":
            return self._format_uint(int(value), 8, "UINT8_MAX")
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
        if dtype == "double":
            return "%.17g"
        if dtype == "bool":
            return "%d"
        if dtype == "uint64":
            return "%llu"
        if dtype == "uint32":
            return "%u"
        if dtype == "uint16":
            return "%hu"
        if dtype == "uint8":
            return "%hhu"
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
        if dtype in {"float", "double"}:
            return "(double)"
        if dtype == "bool":
            return "(int)"
        if dtype == "uint64":
            return "(unsigned long long)"
        if dtype in {"uint32", "uint16", "uint8"}:
            return "(unsigned int)"
        if dtype == "int64":
            return "(long long)"
        if dtype in {"int32", "int16", "int8"}:
            return "(int)"
        raise CodegenError(f"Unsupported dtype {dtype}")


def _format_multiline_value(value: str | None) -> list[str]:
    if not value:
        return ["  n/a"]
    lines = value.splitlines() or [""]
    return [f"  {line}" if line else "  " for line in lines]

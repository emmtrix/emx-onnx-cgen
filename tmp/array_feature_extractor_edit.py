from pathlib import Path

root = Path(r"D:\Stripf\emmtrix\git\emx-onnx-cgen")

path = root / "src/emx_onnx_cgen/ir/ops/misc.py"
text = path.read_text(encoding="utf-8")
anchor = """
@dataclass(frozen=True)
class GatherNDOp(RenderableOpBase):
"""
insert = """
@dataclass(frozen=True)
class ArrayFeatureExtractorOp(RenderableOpBase):
    __io_inputs__ = (\"data\", \"indices\")
    __io_outputs__ = (\"output\",)
    data: str
    indices: str
    output: str

    def validate(self, ctx: OpContext) -> None:
        data_shape = ctx.shape(self.data)
        if not data_shape:
            raise ShapeInferenceError(\"ArrayFeatureExtractor does not support scalar input\")
        indices_dtype = ctx.dtype(self.indices)
        if indices_dtype not in {ScalarType.I32, ScalarType.I64}:
            raise UnsupportedOpError(
                \"ArrayFeatureExtractor indices must be int32 or int64, \"
                f\"got {indices_dtype.onnx_name}\"
            )
        indices_shape = ctx.shape(self.indices)
        if any(dim < 0 for dim in indices_shape):
            raise UnsupportedOpError(
                \"ArrayFeatureExtractor does not support dynamic indices shapes\"
            )

    def infer_types(self, ctx: OpContext) -> None:
        data_dtype = ctx.dtype(self.data)
        try:
            output_dtype = ctx.dtype(self.output)
        except ShapeInferenceError:
            ctx.set_dtype(self.output, data_dtype)
            output_dtype = data_dtype
        if output_dtype != data_dtype:
            raise UnsupportedOpError(
                \"ArrayFeatureExtractor expects output dtype \"
                f\"{data_dtype.onnx_name}, got {output_dtype.onnx_name}\"
            )

    def infer_shapes(self, ctx: OpContext) -> None:
        data_shape = ctx.shape(self.data)
        indices_shape = ctx.shape(self.indices)
        feature_count = _shape_product(indices_shape)
        output_shape = (
            (1, feature_count)
            if len(data_shape) == 1
            else (*data_shape[:-1], feature_count)
        )
        try:
            expected = ctx.shape(self.output)
        except ShapeInferenceError:
            expected = None
        if expected is not None and expected != output_shape:
            raise ShapeInferenceError(
                \"ArrayFeatureExtractor output shape must be \"
                f\"{output_shape}, got {expected}\"
            )
        ctx.set_shape(self.output, output_shape)

    def emit(self, emitter: \"Emitter\", ctx: \"EmitContext\") -> str:
        state = emitter.require_emit_state()
        model = state.model
        op_name = emitter.op_function_name(model, ctx.op_index)
        c_type = emitter.ctx_dtype(self.output).c_type
        params = emitter.shared_param_map(
            [
                (\"data\", self.data),
                (\"indices\", self.indices),
                (\"output\", self.output),
            ]
        )
        data_shape_raw = emitter.ctx_shape(self.data)
        indices_shape_raw = emitter.ctx_shape(self.indices)
        output_shape_raw = emitter.ctx_shape(self.output)
        data_shape = CEmitterCompat.shape_dim_exprs(
            data_shape_raw, emitter.dim_names_for(self.data)
        )
        indices_shape = CEmitterCompat.shape_dim_exprs(
            indices_shape_raw, emitter.dim_names_for(self.indices)
        )
        output_shape = CEmitterCompat.shape_dim_exprs(
            output_shape_raw, emitter.dim_names_for(self.output)
        )
        prefix_shape = output_shape[:-1]
        prefix_loop_vars = CEmitterCompat.loop_vars(prefix_shape)
        feature_var = \"feature_idx\"
        indices_coord_vars = tuple(f\"indices_i{idx}\" for idx in range(len(indices_shape)))
        output_indices = (*prefix_loop_vars, feature_var)
        data_indices = (
            (*prefix_loop_vars[: len(data_shape_raw) - 1], \"gather_index\")
            if len(data_shape_raw) > 1
            else (\"gather_index\",)
        )
        data_suffix = emitter.param_array_suffix(data_shape_raw)
        indices_suffix = emitter.param_array_suffix(indices_shape_raw)
        output_suffix = emitter.param_array_suffix(output_shape_raw)
        indices_dtype = emitter.ctx_dtype(self.indices)
        param_decls = emitter.build_param_decls(
            [
                (params[\"data\"], c_type, data_suffix, True),
                (params[\"indices\"], indices_dtype.c_type, indices_suffix, True),
                (params[\"output\"], c_type, output_suffix, False),
            ]
        )
        rendered = (
            state.templates[\"array_feature_extractor\"]
            .render(
                model_name=model.name,
                op_name=op_name,
                data=params[\"data\"],
                indices=params[\"indices\"],
                output=params[\"output\"],
                params=param_decls,
                prefix_shape=prefix_shape,
                prefix_loop_vars=prefix_loop_vars,
                output_feature_dim=output_shape[-1],
                feature_var=feature_var,
                indices_shape=indices_shape,
                indices_coord_vars=indices_coord_vars,
                data_indices=data_indices,
                output_indices=output_indices,
                last_axis_dim=data_shape[-1],
            )
            .rstrip()
        )
        return emitter.with_node_comment(model, ctx.op_index, rendered)


"""
if anchor not in text:
    raise SystemExit("Anchor for ArrayFeatureExtractorOp not found in misc.py")
text = text.replace(anchor, insert + anchor, 1)
path.write_text(text, encoding="utf-8")

path = root / "src/emx_onnx_cgen/ir/ops/__init__.py"
text = path.read_text(encoding="utf-8")
text = text.replace("    EyeLikeOp,\n", "    EyeLikeOp,\n    ArrayFeatureExtractorOp,\n", 1)
text = text.replace("    \"EyeLikeOp\",\n", "    \"EyeLikeOp\",\n    \"ArrayFeatureExtractorOp\",\n", 1)
path.write_text(text, encoding="utf-8")

path = root / "src/emx_onnx_cgen/lowering/__init__.py"
text = path.read_text(encoding="utf-8")
text = text.replace("_LOWERING_MODULES = [\n", "_LOWERING_MODULES = [\n    \"array_feature_extractor\",\n", 1)
path.write_text(text, encoding="utf-8")

(root / "src/emx_onnx_cgen/lowering/array_feature_extractor.py").write_text(
    '''from __future__ import annotations

from ..errors import ShapeInferenceError, UnsupportedOpError
from ..ir.context import GraphContext
from ..ir.model import Graph, Node
from ..ir.ops import ArrayFeatureExtractorOp
from .common import value_shape
from .registry import register_lowering


@register_lowering("ArrayFeatureExtractor")
def lower_array_feature_extractor(
    graph: Graph, node: Node
) -> ArrayFeatureExtractorOp:
    if len(node.inputs) != 2 or len(node.outputs) != 1:
        raise UnsupportedOpError(
            "ArrayFeatureExtractor must have 2 inputs and 1 output"
        )
    data_name, indices_name = node.inputs
    output_name = node.outputs[0]
    data_shape = value_shape(graph, data_name, node)
    if not data_shape:
        raise ShapeInferenceError("ArrayFeatureExtractor does not support scalar input")
    indices_shape = value_shape(graph, indices_name, node)
    if any(dim < 0 for dim in indices_shape):
        raise UnsupportedOpError(
            "ArrayFeatureExtractor does not support dynamic indices shapes"
        )
    feature_count = 1
    for dim in indices_shape:
        feature_count *= dim
    output_shape = (
        (1, feature_count) if len(data_shape) == 1 else (*data_shape[:-1], feature_count)
    )
    if isinstance(graph, GraphContext):
        graph.set_shape(output_name, output_shape)
    return ArrayFeatureExtractorOp(
        data=data_name,
        indices=indices_name,
        output=output_name,
    )
''',
    encoding="utf-8",
)

path = root / "src/emx_onnx_cgen/codegen/c_emitter.py"
text = path.read_text(encoding="utf-8")
text = text.replace("    EyeLikeOp,\n", "    EyeLikeOp,\n    ArrayFeatureExtractorOp,\n", 1)
text = text.replace(
    '                "expand": self._env.get_template("expand_op.c.j2"),\n',
    '                "expand": self._env.get_template("expand_op.c.j2"),\n                "array_feature_extractor": self._env.get_template(\n                    "array_feature_extractor_op.c.j2"\n                ),\n',
    1,
)
text = text.replace("            | ExpandOp\n", "            | ExpandOp\n            | ArrayFeatureExtractorOp\n", 1)
path.write_text(text, encoding="utf-8")

(root / "src/emx_onnx_cgen/templates/array_feature_extractor_op.c.j2").write_text(
    '''EMX_NODE_FN void {{ op_name }}({{ dim_args }}{{ params | join(', ') }}) {
{% for dim in prefix_shape %}
for (idx_t {{ prefix_loop_vars[loop.index0] }} = 0; {{ prefix_loop_vars[loop.index0] }} < {{ dim }}; ++{{ prefix_loop_vars[loop.index0] }}) {
{% endfor %}
for (idx_t {{ feature_var }} = 0; {{ feature_var }} < {{ output_feature_dim }}; ++{{ feature_var }}) {
{% if indices_shape %}
    idx_t indices_linear = {{ feature_var }};
{% for coord_var in indices_coord_vars | reverse %}
    idx_t {{ coord_var }} = indices_linear % {{ indices_shape[(indices_coord_vars | length - 1) - loop.index0] }};
    indices_linear /= {{ indices_shape[(indices_coord_vars | length - 1) - loop.index0] }};
{% endfor %}
    idx_t gather_index = {{ indices }}{% for coord_var in indices_coord_vars %}[{{ coord_var }}]{% endfor %};
{% else %}
    idx_t gather_index = {{ indices }}[0];
{% endif %}
    if (gather_index < 0) {
        gather_index += {{ last_axis_dim }};
    }
    {{ output }}{% for idx in output_indices %}[{{ idx }}]{% endfor %} = {{ data }}{% for idx in data_indices %}[{{ idx }}]{% endfor %};
}
{% for _ in prefix_shape %}
}
{% endfor %}
}
''',
    encoding="utf-8",
)

path = root / "tests/test_ops.py"
text = path.read_text(encoding="utf-8")
anchor = "\n\ndef _make_gathernd_model(\n"
insert = '''


def _make_array_feature_extractor_model(
    *,
    data_shape: list[int],
    indices_shape: list[int],
    data_dtype: int = TensorProto.FLOAT,
    indices_dtype: int = TensorProto.INT64,
) -> onnx.ModelProto:
    if len(data_shape) < 1:
        raise ValueError("ArrayFeatureExtractor requires rank >= 1 data")
    feature_count = 1
    for dim in indices_shape:
        feature_count *= dim
    output_shape = (
        [1, feature_count] if len(data_shape) == 1 else [*data_shape[:-1], feature_count]
    )
    data_input = helper.make_tensor_value_info("data", data_dtype, data_shape)
    indices_input = helper.make_tensor_value_info("indices", indices_dtype, indices_shape)
    output = helper.make_tensor_value_info("out", data_dtype, output_shape)
    node = helper.make_node(
        "ArrayFeatureExtractor",
        inputs=["data", "indices"],
        outputs=[output.name],
        domain="ai.onnx.ml",
    )
    graph = helper.make_graph(
        [node],
        "array_feature_extractor_graph",
        [data_input, indices_input],
        [output],
    )
    model = helper.make_model(
        graph,
        producer_name="onnx2c",
        opset_imports=[
            helper.make_operatorsetid("", 13),
            helper.make_operatorsetid("ai.onnx.ml", 1),
        ],
    )
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model
'''
if anchor not in text:
    raise SystemExit("Anchor for test_ops helper not found")
text = text.replace(anchor, insert + anchor, 1)
path.write_text(text, encoding="utf-8")

path = root / "tests/test_golden_ops.py"
text = path.read_text(encoding="utf-8")
text = text.replace("    _make_attention_model,\n", "    _make_attention_model,\n    _make_array_feature_extractor_model,\n", 1)
anchor = '    (\n        "gatherelements",\n'
insert = '''    (
        "arrayfeatureextractor",
        "array_feature_extractor",
        lambda: _make_array_feature_extractor_model(data_shape=[3, 4], indices_shape=[2]),
    ),
'''
if anchor not in text:
    raise SystemExit("Anchor for test_golden_ops case not found")
text = text.replace(anchor, insert + anchor, 1)
path.write_text(text, encoding="utf-8")

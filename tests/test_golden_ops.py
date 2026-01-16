from __future__ import annotations

from pathlib import Path
from typing import Callable

import onnx
from onnx import TensorProto, helper

from golden_utils import assert_golden
from emx_onnx_cgen import Compiler
from test_ops import (
    _make_arg_reduce_model,
    _make_batchnorm_model,
    _make_cast_model,
    _make_constant_of_shape_model,
    _make_conv_model,
    _make_cumsum_model,
    _make_expand_model,
    _make_gather_elements_model,
    _make_gather_model,
    _make_gridsample_model,
    _make_layer_normalization_model,
    _make_lp_normalization_model,
    _make_lstm_model,
    _make_maxpool_model,
    _make_mean_variance_normalization_model,
    _make_operator_model,
    _make_pad_model,
    _make_range_model,
    _make_reduce_model,
    _make_reshape_model,
    _make_resize_model,
    _make_rms_normalization_model,
    _make_shape_model,
    _make_size_model,
    _make_slice_model,
    _make_split_model,
    _make_tile_model,
    _make_eye_like_model,
    _make_instance_normalization_model,
    _make_group_normalization_model,
    _make_global_max_pool_model,
    _make_hardmax_model,
    _make_mish_model,
)


def _make_where_model() -> onnx.ModelProto:
    condition = helper.make_tensor_value_info(
        "condition", TensorProto.BOOL, [2, 3]
    )
    input_x = helper.make_tensor_value_info(
        "x", TensorProto.FLOAT, [2, 3]
    )
    input_y = helper.make_tensor_value_info(
        "y", TensorProto.FLOAT, [2, 3]
    )
    output = helper.make_tensor_value_info(
        "out", TensorProto.FLOAT, [2, 3]
    )
    node = helper.make_node(
        "Where", inputs=["condition", "x", "y"], outputs=[output.name]
    )
    graph = helper.make_graph(
        [node], "where_graph", [condition, input_x, input_y], [output]
    )
    model = helper.make_model(
        graph,
        producer_name="onnx2c",
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model


def _make_clip_model() -> onnx.ModelProto:
    input_info = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [2, 3]
    )
    min_info = helper.make_tensor_value_info("min", TensorProto.FLOAT, [])
    max_info = helper.make_tensor_value_info("max", TensorProto.FLOAT, [])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 3])
    min_tensor = helper.make_tensor("min", TensorProto.FLOAT, dims=[], vals=[0.0])
    max_tensor = helper.make_tensor("max", TensorProto.FLOAT, dims=[], vals=[6.0])
    node = helper.make_node(
        "Clip", inputs=["input", "min", "max"], outputs=[output.name]
    )
    graph = helper.make_graph(
        [node],
        "clip_graph",
        [input_info, min_info, max_info],
        [output],
        initializer=[min_tensor, max_tensor],
    )
    model = helper.make_model(
        graph,
        producer_name="onnx2c",
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model


def _make_matmul_model() -> onnx.ModelProto:
    return _make_operator_model(
        op_type="MatMul",
        input_shapes=[[2, 3], [3, 4]],
        output_shape=[2, 4],
        dtype=TensorProto.FLOAT,
        opset=13,
    )


def _make_gemm_model() -> onnx.ModelProto:
    return _make_operator_model(
        op_type="Gemm",
        input_shapes=[[2, 3], [3, 4], [2, 4]],
        output_shape=[2, 4],
        dtype=TensorProto.FLOAT,
        attrs={"alpha": 1.0, "beta": 1.0},
        opset=13,
    )


def _make_attention_model() -> onnx.ModelProto:
    return _make_operator_model(
        op_type="Attention",
        input_shapes=[[1, 2, 3, 4], [1, 2, 5, 4], [1, 2, 5, 4]],
        output_shape=[1, 2, 3, 4],
        dtype=TensorProto.FLOAT,
        attrs={},
        opset=23,
    )


def _make_average_pool_model() -> onnx.ModelProto:
    input_shape = [1, 1, 4, 4]
    output_shape = [1, 1, 2, 2]
    input_info = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, input_shape
    )
    output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, output_shape
    )
    node = helper.make_node(
        "AveragePool",
        inputs=["input"],
        outputs=[output.name],
        kernel_shape=[2, 2],
        strides=[2, 2],
    )
    graph = helper.make_graph(
        [node], "average_pool_graph", [input_info], [output]
    )
    model = helper.make_model(
        graph,
        producer_name="onnx2c",
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model


def _make_softmax_model() -> onnx.ModelProto:
    return _make_operator_model(
        op_type="Softmax",
        input_shapes=[[2, 3]],
        output_shape=[2, 3],
        dtype=TensorProto.FLOAT,
        attrs={"axis": 1},
        opset=13,
    )


def _make_logsoftmax_model() -> onnx.ModelProto:
    return _make_operator_model(
        op_type="LogSoftmax",
        input_shapes=[[2, 3]],
        output_shape=[2, 3],
        dtype=TensorProto.FLOAT,
        attrs={"axis": 1},
        opset=13,
    )


def _make_negative_log_likelihood_loss_model() -> onnx.ModelProto:
    input_info = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [2, 3]
    )
    target_info = helper.make_tensor_value_info(
        "target", TensorProto.INT64, [2]
    )
    output = helper.make_tensor_value_info("loss", TensorProto.FLOAT, [])
    node = helper.make_node(
        "NegativeLogLikelihoodLoss",
        inputs=["input", "target"],
        outputs=[output.name],
        reduction="mean",
    )
    graph = helper.make_graph(
        [node],
        "nllloss_graph",
        [input_info, target_info],
        [output],
    )
    model = helper.make_model(
        graph,
        producer_name="onnx2c",
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model


def _make_softmax_cross_entropy_loss_model() -> onnx.ModelProto:
    input_info = helper.make_tensor_value_info(
        "scores", TensorProto.FLOAT, [2, 3]
    )
    target_info = helper.make_tensor_value_info(
        "labels", TensorProto.INT64, [2]
    )
    output = helper.make_tensor_value_info("loss", TensorProto.FLOAT, [])
    log_prob = helper.make_tensor_value_info(
        "log_prob", TensorProto.FLOAT, [2, 3]
    )
    node = helper.make_node(
        "SoftmaxCrossEntropyLoss",
        inputs=["scores", "labels"],
        outputs=[output.name, log_prob.name],
        reduction="mean",
    )
    graph = helper.make_graph(
        [node],
        "softmax_cross_entropy_loss_graph",
        [input_info, target_info],
        [output, log_prob],
    )
    model = helper.make_model(
        graph,
        producer_name="onnx2c",
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model


def _make_lrn_model() -> onnx.ModelProto:
    return _make_operator_model(
        op_type="LRN",
        input_shapes=[[1, 3, 4, 4]],
        output_shape=[1, 3, 4, 4],
        dtype=TensorProto.FLOAT,
        attrs={"size": 3, "alpha": 0.0001, "beta": 0.75, "bias": 1.0},
        opset=13,
    )


def _make_concat_model() -> onnx.ModelProto:
    return _make_operator_model(
        op_type="Concat",
        input_shapes=[[1, 2, 3], [1, 2, 1]],
        output_shape=[1, 2, 4],
        dtype=TensorProto.FLOAT,
        attrs={"axis": 2},
        opset=13,
    )


def _make_transpose_model() -> onnx.ModelProto:
    return _make_operator_model(
        op_type="Transpose",
        input_shapes=[[2, 3, 4]],
        output_shape=[4, 2, 3],
        dtype=TensorProto.FLOAT,
        attrs={"perm": [2, 0, 1]},
        opset=13,
    )


def _make_identity_model() -> onnx.ModelProto:
    return _make_operator_model(
        op_type="Identity",
        input_shapes=[[2, 3]],
        output_shape=[2, 3],
        dtype=TensorProto.FLOAT,
        opset=13,
    )


def _make_depth_to_space_model() -> onnx.ModelProto:
    return _make_operator_model(
        op_type="DepthToSpace",
        input_shapes=[[1, 4, 2, 2]],
        output_shape=[1, 1, 4, 4],
        dtype=TensorProto.FLOAT,
        attrs={"blocksize": 2, "mode": "DCR"},
        opset=13,
    )


def _make_space_to_depth_model() -> onnx.ModelProto:
    return _make_operator_model(
        op_type="SpaceToDepth",
        input_shapes=[[1, 1, 4, 4]],
        output_shape=[1, 4, 2, 2],
        dtype=TensorProto.FLOAT,
        attrs={"blocksize": 2},
        opset=13,
    )


def _make_binary_mul_model() -> onnx.ModelProto:
    return _make_operator_model(
        op_type="Mul",
        input_shapes=[[2, 3], [2, 3]],
        output_shape=[2, 3],
        dtype=TensorProto.FLOAT,
        opset=13,
    )


def _make_multi_input_sum_model() -> onnx.ModelProto:
    return _make_operator_model(
        op_type="Sum",
        input_shapes=[[2, 3], [2, 3], [2, 3]],
        output_shape=[2, 3],
        dtype=TensorProto.FLOAT,
        opset=13,
    )


def _make_unary_tanh_model() -> onnx.ModelProto:
    return _make_operator_model(
        op_type="Tanh",
        input_shapes=[[2, 3]],
        output_shape=[2, 3],
        dtype=TensorProto.FLOAT,
        opset=13,
    )


def _make_lstm_golden_model() -> onnx.ModelProto:
    return _make_lstm_model(
        seq_length=1,
        batch_size=1,
        input_size=2,
        hidden_size=3,
        dtype=TensorProto.FLOAT,
        include_optional_inputs=False,
        include_y=True,
        include_y_h=True,
        include_y_c=True,
    )


def _make_reduce_mean_model() -> onnx.ModelProto:
    return _make_reduce_model(
        op_type="ReduceMean",
        input_shape=[2, 3, 4],
        output_shape=[2, 1, 4],
        axes=[1],
        keepdims=1,
        dtype=TensorProto.FLOAT,
        opset=18,
    )


def _make_argmax_model() -> onnx.ModelProto:
    return _make_arg_reduce_model(
        op_type="ArgMax",
        input_shape=[2, 3, 4],
        output_shape=[2, 1, 4],
        axis=1,
        keepdims=1,
        select_last_index=0,
        dtype=TensorProto.FLOAT,
        opset=13,
    )


def _compile_and_assert_golden(model: onnx.ModelProto, filename: str) -> None:
    compiler = Compiler()
    generated = compiler.compile(model)
    golden_path = Path(__file__).parent / "golden" / filename
    assert_golden(generated, golden_path)


def _make_test_case(model_fn: Callable[[], onnx.ModelProto], filename: str) -> Callable[[], None]:
    def _test() -> None:
        _compile_and_assert_golden(model_fn(), filename)

    return _test


# Each entry is (class_name, op_name, model_factory). class_name must be a single
# token without underscores; filenames and test names are generated as:
#   test_op_<class_name>_<op_name> and op_<class_name>_<op_name>.c
OP_GOLDEN_CASES = [
    ("binary", "mul", _make_binary_mul_model),
    ("multiinputbinary", "sum", _make_multi_input_sum_model),
    ("where", "where", _make_where_model),
    ("unary", "tanh", _make_unary_tanh_model),
    ("clip", "clip", _make_clip_model),
    ("cast", "cast", _make_cast_model),
    ("matmul", "matmul", _make_matmul_model),
    ("gemm", "gemm", _make_gemm_model),
    ("attention", "attention", _make_attention_model),
    ("conv", "conv", _make_conv_model),
    ("averagepool", "average_pool", _make_average_pool_model),
    ("softmax", "softmax", _make_softmax_model),
    ("logsoftmax", "logsoftmax", _make_logsoftmax_model),
    ("hardmax", "hardmax", _make_hardmax_model),
    (
        "negativeloglikelihoodloss",
        "negative_log_likelihood_loss",
        _make_negative_log_likelihood_loss_model,
    ),
    (
        "softmaxcrossentropyloss",
        "softmax_cross_entropy_loss",
        _make_softmax_cross_entropy_loss_model,
    ),
    ("batchnorm", "batch_normalization", lambda: _make_batchnorm_model()[0]),
    (
        "lpnormalization",
        "lp_normalization",
        lambda: _make_lp_normalization_model(input_shape=[2, 3], axis=-1, p=1),
    ),
    (
        "instancenormalization",
        "instance_normalization",
        lambda: _make_instance_normalization_model(input_shape=[1, 3, 2, 2]),
    ),
    (
        "groupnormalization",
        "group_normalization",
        lambda: _make_group_normalization_model(
            input_shape=[1, 4, 2, 2], num_groups=2
        ),
    ),
    (
        "layernormalization",
        "layer_normalization",
        lambda: _make_layer_normalization_model(input_shape=[2, 3, 4], axis=1),
    ),
    (
        "meanvariancenormalization",
        "mean_variance_normalization",
        lambda: _make_mean_variance_normalization_model(
            input_shape=[2, 3, 4], axes=[-1]
        ),
    ),
    (
        "rmsnormalization",
        "rms_normalization",
        lambda: _make_rms_normalization_model(input_shape=[2, 3, 4], axis=-1),
    ),
    ("lrn", "lrn", _make_lrn_model),
    ("lstm", "lstm", _make_lstm_golden_model),
    (
        "maxpool",
        "maxpool",
        lambda: _make_maxpool_model(
            input_shape=[1, 1, 4, 4],
            kernel_shape=[2, 2],
            strides=[2, 2],
            pads=[0, 0, 0, 0],
            ceil_mode=0,
        ),
    ),
    ("globalmaxpool", "global_max_pool", _make_global_max_pool_model),
    ("concat", "concat", _make_concat_model),
    (
        "gatherelements",
        "gather_elements",
        lambda: _make_gather_elements_model(
            data_shape=[2, 3], indices_shape=[2, 3], axis=0
        ),
    ),
    (
        "gather",
        "gather",
        lambda: _make_gather_model(
            data_shape=[3, 2], indices_shape=[2], axis=0
        ),
    ),
    ("transpose", "transpose", _make_transpose_model),
    ("reshape", "reshape", _make_reshape_model),
    ("identity", "identity", _make_identity_model),
    (
        "eyelike",
        "eye_like",
        lambda: _make_eye_like_model(input_shape=[3, 3], dtype=TensorProto.FLOAT),
    ),
    (
        "tile",
        "tile",
        lambda: _make_tile_model(
            input_shape=[2, 3], repeats=[2, 1], dtype=TensorProto.FLOAT
        ),
    ),
    (
        "pad",
        "pad",
        lambda: _make_pad_model(
            input_shape=[2, 3], pads=[0, 1, 0, 1], value=0.0, dtype=TensorProto.FLOAT
        ),
    ),
    ("depthtospace", "depth_to_space", _make_depth_to_space_model),
    ("spacetodepth", "space_to_depth", _make_space_to_depth_model),
    ("slice", "slice", _make_slice_model),
    ("resize", "resize", _make_resize_model),
    (
        "gridsample",
        "grid_sample",
        lambda: _make_gridsample_model(
            input_shape=[1, 1, 2, 2],
            grid_shape=[1, 2, 2, 2],
            output_shape=[1, 1, 2, 2],
        ),
    ),
    ("reduce", "reduce_mean", _make_reduce_mean_model),
    ("argreduce", "arg_max", _make_argmax_model),
    ("constantofshape", "constant_of_shape", _make_constant_of_shape_model),
    ("shape", "shape", lambda: _make_shape_model(input_shape=[2, 3, 4])),
    ("size", "size", lambda: _make_size_model(input_shape=[2, 3, 4])),
    (
        "expand",
        "expand",
        lambda: _make_expand_model(
            input_shape=[1, 3], target_shape=[2, 3], dtype=TensorProto.FLOAT
        ),
    ),
    (
        "cumsum",
        "cumsum",
        lambda: _make_cumsum_model(
            input_shape=[2, 3], axis=1, dtype=TensorProto.FLOAT
        ),
    ),
    (
        "range",
        "range",
        lambda: _make_range_model(start=0, limit=4, delta=1, dtype=TensorProto.INT64),
    ),
    ("mish", "mish", _make_mish_model),
    (
        "split",
        "split",
        lambda: _make_split_model(
            input_shape=[2, 6], split_sizes=[2, 2, 2], axis=1, dtype=TensorProto.FLOAT
        ),
    ),
]

for op_class, op_name, model_fn in OP_GOLDEN_CASES:
    test_name = f"test_op_{op_class}_{op_name}"
    filename = f"op_{op_class}_{op_name}.c"
    test_fn = _make_test_case(model_fn, filename)
    test_fn.__name__ = test_name
    globals()[test_name] = test_fn

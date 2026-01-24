# Error frequency

| Error message | Count | Histogram |
| --- | --- | --- |
| Out of tolerance | 38 | ██████████████████████████████ |
| Unsupported value type '*' for '*'. Hint: export the model with tensor inputs/outputs. | 36 | ████████████████████████████ |
| Unsupported elem_type 8 (STRING) for tensor '*'. | 32 | █████████████████████████ |
| Unsupported elem_type 17 (FLOAT8E4M3FN) for tensor '*'. | 22 | █████████████████ |
| Unsupported elem_type 19 (FLOAT8E5M2) for tensor '*'. | 20 | ████████████████ |
| Unsupported elem_type 18 (FLOAT8E4M3FNUZ) for tensor '*'. | 18 | ██████████████ |
| Unsupported elem_type 20 (FLOAT8E5M2FNUZ) for tensor '*'. | 18 | ██████████████ |
| Unsupported elem_type 21 (UINT4) for tensor '*'. | 17 | █████████████ |
| Unsupported elem_type 22 (INT4) for tensor '*'. | 17 | █████████████ |
| Unsupported elem_type 25 (UINT2) for tensor '*'. | 17 | █████████████ |
| Unsupported elem_type 26 (INT2) for tensor '*'. | 17 | █████████████ |
| Unsupported elem_type 23 (FLOAT4E2M1) for tensor '*'. | 14 | ███████████ |
| Unsupported op ImageDecoder | 9 | ███████ |
| '*' object has no attribute '*' | 8 | ██████ |
| Dropout supports only the data input and 1 or 2 outputs | 8 | ██████ |
| tuple index out of range | 8 | ██████ |
| Unsupported op TfIdfVectorizer | 7 | ██████ |
| AveragePool has unsupported attributes | 6 | █████ |
| Unsupported elem_type 16 (BFLOAT16) for tensor '*'. | 6 | █████ |
| Unsupported op CenterCropPad | 6 | █████ |
| Unsupported op DFT | 6 | █████ |
| Unsupported op ScatterElements | 6 | █████ |
| Unsupported op Unique | 6 | █████ |
| Iteration of zero-sized operands is not enabled | 5 | ████ |
| Unsupported op Col2Im | 5 | ████ |
| Unsupported op DequantizeLinear | 5 | ████ |
| Unsupported op If | 5 | ████ |
| Unsupported elem_type 24 (FLOAT8E8M0) for tensor '*'. | 4 | ███ |
| Unsupported op AffineGrid | 4 | ███ |
| Unsupported op Compress | 4 | ███ |
| Unsupported op DeformConv | 4 | ███ |
| Unsupported op GRU | 4 | ███ |
| Unsupported op OptionalHasElement | 4 | ███ |
| Unsupported op RNN | 4 | ███ |
| AveragePool supports auto_pad=NOTSET only | 3 | ██ |
| Elu only supports alpha=1.0 | 3 | ██ |
| HardSigmoid only supports alpha=0.2 | 3 | ██ |
| LeakyRelu only supports alpha=0.01 | 3 | ██ |
| Unsupported op Bernoulli | 3 | ██ |
| Unsupported op DynamicQuantizeLinear | 3 | ██ |
| Unsupported op Loop | 3 | ██ |
| Unsupported op Momentum | 3 | ██ |
| Unsupported op RandomUniformLike | 3 | ██ |
| Unsupported op RoiAlign | 3 | ██ |
| name '*' is not defined | 3 | ██ |
| 
Arrays are not equal

Mismatched elements: 1 / 6 (16.7%)
Max absolute difference among violations: 108
Max relative difference among violations: 5.4
 ACTUAL: array([[  41,  -12,   -9],
       [   1,  -75, -128]], dtype=int8)
 DESIRED: array([[ 41, -12,  -9],
       [  1, -75,  20]], dtype=int8) | 2 | ██ |
| AveragePool supports ceil_mode=0 only | 2 | ██ |
| BatchNormalization must have 5 inputs and 1 output | 2 | ██ |
| Failed to build testbench. | 2 | ██ |
| Gelu only supports approximate=none | 2 | ██ |
| LpPool expects 2D kernel_shape | 2 | ██ |
| LpPool supports auto_pad=NOTSET only | 2 | ██ |
| Pow expects matching dtypes, got float, int32 | 2 | ██ |
| Pow expects matching dtypes, got float, int64 | 2 | ██ |
| QuantizeLinear block_size is not supported | 2 | ██ |
| Selu only supports alpha=1.6732632423543772 | 2 | ██ |
| ThresholdedRelu only supports alpha=1.0 | 2 | ██ |
| Tile repeats input must be a constant initializer | 2 | ██ |
| Unsupported op Adam | 2 | ██ |
| Unsupported op BitwiseNot | 2 | ██ |
| Unsupported op BlackmanWindow | 2 | ██ |
| Unsupported op ConvInteger | 2 | ██ |
| Unsupported op Det | 2 | ██ |
| Unsupported op Gradient | 2 | ██ |
| Unsupported op HammingWindow | 2 | ██ |
| Unsupported op HannWindow | 2 | ██ |
| Unsupported op MaxUnpool | 2 | ██ |
| Unsupported op ReverseSequence | 2 | ██ |
| Unsupported op STFT | 2 | ██ |
| Unsupported op Scan | 2 | ██ |
| Unsupported op Scatter | 2 | ██ |
| Unsupported op TreeEnsemble | 2 | ██ |
| 
Arrays are not equal

Mismatched elements: 1 / 6 (16.7%)
Max absolute difference among violations: 1
Max relative difference among violations: 0.00558659
 ACTUAL: array([153, 255,   0,  26, 221, 178], dtype=uint8)
 DESIRED: array([153, 255,   0,  26, 221, 179], dtype=uint8) | 1 | █ |
| 
Arrays are not equal

Mismatched elements: 2 / 12 (16.7%)
Max absolute difference among violations: 1
Max relative difference among violations: 3.05203723e-05
 ACTUAL: array([32767, 32703, 32768, 32766, 32768, 32766, 32769, 32765, 65535,
           0, 65535,     0], dtype=uint16)
 DESIRED: array([32767, 32703, 32769, 32765, 32768, 32766, 32769, 32765, 65535,
           0, 65535,     0], dtype=uint16) | 1 | █ |
| 
Arrays are not equal

Mismatched elements: 4 / 12 (33.3%)
Max absolute difference among violations: 11
Max relative difference among violations: 0.09401709
 ACTUAL: array([[[ -86, -128, -128],
        [ 115,   39, -121]],
...
 DESIRED: array([[[ -86,  117,  120],
        [ 115,   39, -121]],
... | 1 | █ |
| 
Arrays are not equal

Mismatched elements: 4 / 12 (33.3%)
Max absolute difference among violations: 12
Max relative difference among violations: 0.10344828
 ACTUAL: array([[[ -86, -128, -128],
        [ 115,   39, -121]],
...
 DESIRED: array([[[ -86,  116,  119],
        [ 115,   39, -121]],
... | 1 | █ |
| AveragePool supports 2D/3D inputs only | 1 | █ |
| Broadcasting mismatch for shapes: (2, 3, 4), (3,) | 1 | █ |
| Broadcasting mismatch for shapes: (2, 3, 4, 5), (3,) | 1 | █ |
| Broadcasting mismatch for shapes: (2, 3, 4, 5, 6), (3,) | 1 | █ |
| ConvTranspose output shape must be fully defined and non-negative | 1 | █ |
| Dropout mask output is not supported | 1 | █ |
| Dynamic dim for tensor '*' | 1 | █ |
| Graph must contain at least one node | 1 | █ |
| LpPool supports dilations=1 only | 1 | █ |
| Max must have at least 2 inputs | 1 | █ |
| Mean must have at least 2 inputs | 1 | █ |
| Min must have at least 2 inputs | 1 | █ |
| Pad value input must be a scalar | 1 | █ |
| Pow expects matching dtypes, got float, uint32 | 1 | █ |
| Pow expects matching dtypes, got float, uint64 | 1 | █ |
| ReduceMax does not support dtype bool | 1 | █ |
| ReduceMin does not support dtype bool | 1 | █ |
| Sum must have at least 2 inputs | 1 | █ |
| Unsupported op ArrayFeatureExtractor | 1 | █ |
| Unsupported op Binarizer | 1 | █ |
| Unsupported op MatMulInteger | 1 | █ |
| Unsupported op MelWeightMatrix | 1 | █ |
| Unsupported op OptionalGetElement | 1 | █ |
| Unsupported op QLinearConv | 1 | █ |
| Unsupported op Upsample | 1 | █ |
| onnxruntime failed to run onnx-org/onnx/backend/test/data/node/test_adagrad/model.onnx: [ONNXRuntimeError] : 1 : FAIL : Fatal error: ai.onnx.preview.training:Adagrad(-1) is not a registered function/op | 1 | █ |
| onnxruntime failed to run onnx-org/onnx/backend/test/data/node/test_adagrad_multiple/model.onnx: [ONNXRuntimeError] : 1 : FAIL : Fatal error: ai.onnx.preview.training:Adagrad(-1) is not a registered function/op | 1 | █ |

## Local ONNX file support histogram

### Error frequency

| Error message | Count | Histogram |
| --- | --- | --- |
| Out of tolerance | 2 | ██████████████████████████████ |
| Unsupported LSTM direction b'*' | 2 | ██████████████████████████████ |
| Unsupported op QLinearAdd | 2 | ██████████████████████████████ |
| 
Arrays are not equal

Mismatched elements: 288 / 432 (66.7%)
Max absolute difference among violations: 288
Max relative difference among violations: 288.
 ACTUAL: array([[[[  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11],
         [ 12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23],
         [ 24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35],...
 DESIRED: array([[[[  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11],
         [ 12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23],
         [ 24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35],... | 1 | ███████████████ |
| Gemm bias input must be broadcastable to output shape, got (2,) vs (2, 4) | 1 | ███████████████ |
| onnx-reference failed to run onnx2c-org/test/local_ops/test_lstm_all_outputs/model.onnx: Unable to find output name '*' in ['*', '*', '*', '*', '*', '*'], proto is
ir_version: 5
producer_name: "backend-test"
graph {
  node {
    input: "X"
    input: "W"
    input: "R"
    output: "Y"
    output: "Y_h"
    output: "Y_c"
    op_type: "LSTM"
    attribute {
      name: "hidden_size"
      i: 4
      type: INT
    }
  }
  name: "test_lstm_all_outputs"
  input {
    name: "X"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
            dim_value: 2
          }
          dim {
            dim_value: 3
          }
          dim {
            dim_value: 3
          }
        }
      }
    }
  }
  input {
    name: "W"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
            dim_value: 1
          }
          dim {
            dim_value: 16
          }
          dim {
            dim_value: 3
          }
        }
      }
    }
  }
  input {
    name: "R"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
            dim_value: 1
          }
          dim {
            dim_value: 16
          }
          dim {
            dim_value: 4
          }
        }
      }
    }
  }
  output {
    name: "Y"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
            dim_value: 2
          }
          dim {
            dim_value: 1
          }
          dim {
            dim_value: 3
          }
          dim {
            dim_value: 4
          }
        }
      }
    }
  }
  output {
    name: "Y_h"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
            dim_value: 1
          }
          dim {
            dim_value: 3
          }
          dim {
            dim_value: 4
          }
        }
      }
    }
  }
  output {
    name: "Y_c"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
            dim_value: 1
          }
          dim {
            dim_value: 3
          }
          dim {
            dim_value: 4
          }
        }
      }
    }
  }
}
opset_import {
  version: 11
}
 | 1 | ███████████████ |
| onnx-reference failed to run onnx2c-org/test/local_ops/test_lstm_y_c/model.onnx: Unable to find output name '*' in ['*', '*', '*', '*', '*'], proto is
ir_version: 5
producer_name: "backend-test"
graph {
  node {
    input: "X"
    input: "W"
    input: "R"
    output: "Y_h"
    output: ""
    output: "Y_c"
    op_type: "LSTM"
    attribute {
      name: "hidden_size"
      i: 4
      type: INT
    }
  }
  name: "test_lstm_y_c"
  input {
    name: "X"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
            dim_value: 2
          }
          dim {
            dim_value: 3
          }
          dim {
            dim_value: 3
          }
        }
      }
    }
  }
  input {
    name: "W"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
            dim_value: 1
          }
          dim {
            dim_value: 16
          }
          dim {
            dim_value: 3
          }
        }
      }
    }
  }
  input {
    name: "R"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
            dim_value: 1
          }
          dim {
            dim_value: 16
          }
          dim {
            dim_value: 4
          }
        }
      }
    }
  }
  output {
    name: "Y_h"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
            dim_value: 2
          }
          dim {
            dim_value: 1
          }
          dim {
            dim_value: 3
          }
          dim {
            dim_value: 4
          }
        }
      }
    }
  }
  output {
    name: "Y_c"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
            dim_value: 1
          }
          dim {
            dim_value: 3
          }
          dim {
            dim_value: 4
          }
        }
      }
    }
  }
}
opset_import {
  version: 11
}
 | 1 | ███████████████ |
| onnx-reference failed to run onnx2c-org/test/local_ops/test_nodes_out_of_order/model.onnx: Unable to find input '*' in known results ['*', '*'], self.rt_inits_ has [], feed_inputs has ['*']. | 1 | ███████████████ |
| onnx-reference failed to run onnx2c-org/test/local_ops/test_pad_constant_default/model.onnx: Unable to find input '*' in known results ['*', '*'], self.rt_inits_ has [], feed_inputs has ['*']. | 1 | ███████████████ |
| onnx-reference failed to run onnx2c-org/test/local_ops/test_pad_constant_input/model.onnx: Unable to find input '*' in known results ['*', '*'], self.rt_inits_ has [], feed_inputs has ['*']. | 1 | ███████████████ |
| onnx-reference failed to run onnx2c-org/test/local_ops/test_pad_edge/model.onnx: Unable to find input '*' in known results ['*', '*'], self.rt_inits_ has [], feed_inputs has ['*']. | 1 | ███████████████ |
| onnx-reference failed to run onnx2c-org/test/local_ops/test_pad_edge_allaxes/model.onnx: Unable to find input '*' in known results ['*', '*'], self.rt_inits_ has [], feed_inputs has ['*']. | 1 | ███████████████ |
| onnx-reference failed to run onnx2c-org/test/local_ops/test_pad_reflect_allaxes/model.onnx: Unable to find input '*' in known results ['*', '*'], self.rt_inits_ has [], feed_inputs has ['*']. | 1 | ███████████████ |
| onnx-reference failed to run onnx2c-org/test/local_ops/test_pad_reflect_nopadding/model.onnx: Unable to find input '*' in known results ['*', '*'], self.rt_inits_ has [], feed_inputs has ['*']. | 1 | ███████████████ |
| onnx-reference failed to run onnx2c-org/test/local_ops/test_resize_downsample_sizes_linear_1D/model.onnx: Unable to find input '*' in known results ['*', '*'], self.rt_inits_ has [], feed_inputs has ['*']. | 1 | ███████████████ |
| onnx-reference failed to run onnx2c-org/test/local_ops/test_resize_downsample_sizes_linear_1D_align/model.onnx: Unable to find input '*' in known results ['*', '*'], self.rt_inits_ has [], feed_inputs has ['*']. | 1 | ███████████████ |
| onnx-reference failed to run onnx2c-org/test/local_ops/test_slice_end_INT64_MAX/model.onnx: Unable to find input '*' in known results ['*', '*'], self.rt_inits_ has [], feed_inputs has ['*']. | 1 | ███████████████ |

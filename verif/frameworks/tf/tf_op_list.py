# Copyright (c) 2025, ARM Limited.
# SPDX-License-Identifier: Apache-2.0
import tensorflow as tf  # noqa: E402
from frameworks.test_gen_utils import QuantType  # noqa: E402
from frameworks.tf.arg_gen import ArgGen  # noqa: E402
from frameworks.tf.tensor_gen import TGen  # noqa: E402
from frameworks.tf.test_builder import TBuilder  # noqa: E402

# Lists of different data types
TYPE_F = [tf.float32]
TYPE_I = [tf.int32]
TYPE_FI = [tf.float32, tf.int32]
TYPE_B = [tf.bool]
TYPE_FIB = [tf.float32, tf.int32, tf.bool]
TYPE_H = [tf.float16]
TYPE_FH = [tf.float32, tf.float16]
TYPE_FHI = [tf.float32, tf.float16, tf.int32]
TYPE_FHIB = [tf.float32, tf.float16, tf.int32, tf.bool]

# The list of operator tests
# Each dictionary entry for an op is a dictionary with the following required members:
#   'operands': tuple (number_of_placeholder_tensors, number_of_constant_tensors)
#   'build_fcn: tuple (Test builder function, Tensor generator function,
#                      Argument generator function)
#   'types': list of Tensorflow types that should be tested for this op
#               OR
#            a dictionary of {'framework_name': [type_list] } for cases where only
#            a subset of the types should be tested in each framework.  This can also
#            be used to restrict an operator to a particular framework.
#
# And optional members:
#   'template':      boolean (indicates that this is a templated op which gets further
#                    processing in createDynamicOpLists)
#   'bias':          boolean indicating that there is a bias component to be generated
#   'qtypes':        List of QuantType quantized types to generate for this op
#   'rank':          tuple (lowest rank, highest rank). Dimension range of input tensor.
#   'custom_shapes': List of custom shapes for specific operators

TF_OP_LIST = {
    "add": {
        "operands": (2, 0),
        "build_fcn": (TBuilder.Add, TGen.tgBFuzz, ArgGen.agNone),
        "types": {
            "tf": TYPE_FI,
            "tflite": list(
                TYPE_FI + [QuantType.ALL_U8, QuantType.ALL_I8, QuantType.ALL_I16]
            ),
        },
    },
    "sub": {
        "operands": (2, 0),
        "build_fcn": (TBuilder.Sub, TGen.tgBFuzz, ArgGen.agNone),
        "types": {
            "tf": TYPE_FI,
            "tflite": list(TYPE_FI + [QuantType.ALL_U8, QuantType.ALL_I8]),
            # QuantType.ALL_I16 fail in TFLite conversion
        },
    },
    "mul": {
        "operands": (2, 0),
        "build_fcn": (TBuilder.Mul, TGen.tgBFuzz, ArgGen.agNone),
        "types": {
            "tf": TYPE_FI,
            "tflite": list(
                TYPE_FI + [QuantType.ALL_U8, QuantType.ALL_I8, QuantType.ALL_I16]
            ),
        },
    },
    "floor_div": {
        "operands": (2, 0),
        "build_fcn": (TBuilder.FloorDiv, TGen.tgDiv, ArgGen.agNone),
        "types": TYPE_FI,
    },
    "exp": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.Exp, TGen.tgBasic, ArgGen.agNone),
        "types": TYPE_F,
    },
    "rcp": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.Rcp, TGen.tgBasic, ArgGen.agNone),
        "types": TYPE_F,
    },
    "relu": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.Relu, TGen.tgBasic, ArgGen.agNone),
        "types": {
            "tf": TYPE_F,
            "tflite": list(
                TYPE_F + [QuantType.ALL_U8, QuantType.ALL_I8, QuantType.ALL_I16]
            ),
        },
    },
    "relu1": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.Relu1, TGen.tgBasic, ArgGen.agNone),
        "types": {
            "tf": [],
            "tflite": list(TYPE_F + [QuantType.ALL_U8, QuantType.ALL_I8]),
        },
    },
    "relu0To1": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.Relu0To1, TGen.tgBasic, ArgGen.agNone),
        "types": {
            "tf": [],
            "tflite": list(TYPE_F + [QuantType.ALL_U8, QuantType.ALL_I8]),
        },
    },
    "relu6": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.Relu6, TGen.tgBasic, ArgGen.agNone),
        "types": {
            "tf": TYPE_F,
            "tflite": list(
                TYPE_F + [QuantType.ALL_U8, QuantType.ALL_I8, QuantType.ALL_I16]
            ),
        },
    },
    "leaky_relu": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.LeakyRelu, TGen.tgBasic, ArgGen.agFloat),
        "types": {
            "tf": TYPE_F,
            "tflite": list(
                TYPE_F + [QuantType.ALL_U8, QuantType.ALL_I8, QuantType.ALL_I16]
            ),
        },
    },
    "prelu": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.Prelu, TGen.tgBasic, ArgGen.agNone),
        "types": {
            "tflite": list(TYPE_F + [QuantType.ALL_U8, QuantType.ALL_I8]),
        },
    },
    "gelu": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.Gelu, TGen.tgBasicPositive, ArgGen.agNone),
        "types": {
            # Need compiler support for tf.Erf.
            # "tf": TYPE_F,
            "tflite": list(
                # Only float32, int8 and uint8 supported currently
                TYPE_F
                + [QuantType.ALL_U8, QuantType.ALL_I8]
            ),
        },
    },
    "concat": {
        "operands": (2, 0),
        "build_fcn": (TBuilder.Concat, TGen.tgBasic, ArgGen.agAxes),
        "types": TYPE_FI,
        "rank": (0, 4),
        "custom_shapes": {
            "custom_shape_only": False,
            "shape_list": [()],
        },
    },
    "bitwise_and": {
        "operands": (2, 0),
        "build_fcn": (TBuilder.BitwiseAnd, TGen.tgBFuzz, ArgGen.agNone),
        "types": {"tf": TYPE_I},  # Not supported in TF Lite
    },
    "bitwise_or": {
        "operands": (2, 0),
        "build_fcn": (TBuilder.BitwiseOr, TGen.tgBFuzz, ArgGen.agNone),
        "types": {"tf": TYPE_I},  # Not supported in TF Lite
    },
    "bitwise_not": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.BitwiseNot, TGen.tgBFuzz, ArgGen.agNone),
        "types": {"tf": TYPE_I},  # Not supported in TF Lite
    },
    "bitwise_xor": {
        "operands": (2, 0),
        "build_fcn": (TBuilder.BitwiseXor, TGen.tgBFuzz, ArgGen.agNone),
        "types": {"tf": TYPE_I},  # Not supported in TF Lite
    },
    "logical_and": {
        "operands": (2, 0),
        "build_fcn": (TBuilder.LogicalAnd, TGen.tgBFuzz, ArgGen.agNone),
        "types": TYPE_B,
    },
    "logical_or": {
        "operands": (2, 0),
        "build_fcn": (TBuilder.LogicalOr, TGen.tgBFuzz, ArgGen.agNone),
        "types": TYPE_B,
    },
    "logical_not": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.LogicalNot, TGen.tgBFuzz, ArgGen.agNone),
        "types": TYPE_B,
    },
    "reduce_any": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.ReduceAny, TGen.tgBasic, ArgGen.agAxesListKeepdims),
        "types": TYPE_B,
    },
    "reduce_all": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.ReduceAll, TGen.tgBasic, ArgGen.agAxesListKeepdims),
        "types": TYPE_B,
    },
    "reduce_min": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.ReduceMin, TGen.tgBasic, ArgGen.agAxesListKeepdims),
        "types": {
            "tf": TYPE_FI,
            "tflite": list(TYPE_FI + [QuantType.ALL_U8, QuantType.ALL_I8]),
        },
    },
    "reduce_max": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.ReduceMax, TGen.tgBasic, ArgGen.agAxesListKeepdims),
        "types": {
            "tf": TYPE_FI,
            "tflite": list(TYPE_FI + [QuantType.ALL_U8, QuantType.ALL_I8]),
        },
    },
    "reduce_sum": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.ReduceSum, TGen.tgBasic, ArgGen.agAxesListKeepdims),
        "types": {
            "tf": TYPE_F,
            # v2 converter doesn't recognize quantized reduce_sum
            # "tflite": list(TYPE_F + [QuantType.ALL_U8, QuantType.ALL_I8]),
            "tflite": TYPE_F,
        },
    },
    "reduce_mean": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.ReduceMean, TGen.tgBasic, ArgGen.agAxesListKeepdims),
        "types": {
            "tf": TYPE_F,
            "tflite": list(
                TYPE_F + [QuantType.ALL_U8, QuantType.ALL_I8, QuantType.ALL_I16]
            ),
        },
    },
    "reduce_product": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.ReduceProduct, TGen.tgBasic, ArgGen.agAxesListKeepdims),
        "types": TYPE_F,
    },
    "min": {
        "operands": (2, 0),
        "build_fcn": (TBuilder.Min, TGen.tgBFuzz, ArgGen.agNone),
        "types": TYPE_FI,
    },
    "max": {
        "operands": (2, 0),
        "build_fcn": (TBuilder.Max, TGen.tgBFuzz, ArgGen.agNone),
        "types": TYPE_FI,
    },
    "pow": {
        "operands": (2, 0),
        "build_fcn": (TBuilder.Pow, TGen.tgPow, ArgGen.agNone),
        # Technically, integer is supported, but only for positive exponents.
        # Needs a random argument generator.
        "types": TYPE_F,
    },
    "abs": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.Abs, TGen.tgBasic, ArgGen.agNone),
        "types": TYPE_F,
    },
    "ceil": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.Ceil, TGen.tgBasic, ArgGen.agNone),
        "types": TYPE_F,
    },
    "floor": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.Floor, TGen.tgBasic, ArgGen.agNone),
        "types": TYPE_F,
    },
    "log": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.Log, TGen.tgBasic, ArgGen.agNone),
        "types": TYPE_F,
    },
    "negate": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.Negate, TGen.tgBasic, ArgGen.agNone),
        "types": TYPE_F,
    },
    "rsqrt": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.Rsqrt, TGen.tgBasicPositive, ArgGen.agNone),
        "types": {
            "tf": TYPE_F,
            "tflite": list(TYPE_F + [QuantType.ALL_I8]),
        },
    },
    "sign": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.Sign, TGen.tgBasic, ArgGen.agNone),
        "types": {
            "tf": TYPE_F,
        },
    },
    "sigmoid": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.Sigmoid, TGen.tgBasic, ArgGen.agNone),
        "types": {
            "tf": TYPE_F,
            "tflite": list(
                TYPE_F + [QuantType.ALL_U8, QuantType.ALL_I8, QuantType.ALL_I16]
            ),
        },
    },
    "tanh": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.Tanh, TGen.tgBasic, ArgGen.agNone),
        "types": {
            "tf": TYPE_F,
            "tflite": list(
                TYPE_F + [QuantType.ALL_U8, QuantType.ALL_I8, QuantType.ALL_I16]
            ),
        },
    },
    "erf": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.Erf, TGen.tgBasic, ArgGen.agNone),
        "types": {
            "tf": TYPE_F,
        },
    },
    "sin": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.Sin, TGen.tgBasic, ArgGen.agNone),
        "types": TYPE_F,
    },
    "cos": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.Cos, TGen.tgBasic, ArgGen.agNone),
        "types": TYPE_F,
    },
    "atan2": {
        "operands": (2, 0),
        "build_fcn": (TBuilder.Atan2, TGen.tgBasic, ArgGen.agNone),
        "types": {
            "tflite": TYPE_F,
        },
    },
    "square": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.Square, TGen.tgBasic, ArgGen.agNone),
        "types": TYPE_F,
    },
    "squared_difference": {
        "operands": (2, 0),
        "build_fcn": (TBuilder.SquaredDifference, TGen.tgBFuzz, ArgGen.agNone),
        "types": {
            "tf": TYPE_F,
            "tflite": list(TYPE_FI + [QuantType.ALL_I8]),
        },
    },
    "equal": {
        "operands": (2, 0),
        "build_fcn": (TBuilder.Equal, TGen.tgBFuzz, ArgGen.agNone),
        "types": TYPE_FI,
    },
    "greater_equal": {
        "operands": (2, 0),
        "build_fcn": (TBuilder.GreaterEqual, TGen.tgBFuzz, ArgGen.agNone),
        "types": TYPE_FI,
    },
    "greater": {
        "operands": (2, 0),
        "build_fcn": (TBuilder.Greater, TGen.tgBFuzz, ArgGen.agNone),
        "types": TYPE_FI,
    },
    "less": {
        "operands": (2, 0),
        "build_fcn": (TBuilder.Less, TGen.tgBFuzz, ArgGen.agNone),
        "types": TYPE_FI,
    },
    "less_equal": {
        "operands": (2, 0),
        "build_fcn": (TBuilder.LessEqual, TGen.tgBFuzz, ArgGen.agNone),
        "types": TYPE_FI,
    },
    "conv2d_TEMPLATE": {
        "operands": (1, 1),
        "build_fcn": (TBuilder.Conv2d, TGen.tgConv2d, ArgGen.agConv2d),
        "types": {
            "tf": [tf.float32],
            "tflite": [
                tf.float32,
                QuantType.CONV_U8_U8,
                QuantType.CONV_I8_I8,
                QuantType.CONV_I16_I8,
            ],
        },
        "template": True,
    },
    "conv2d_relu_TEMPLATE": {
        "operands": (1, 2),
        "build_fcn": (TBuilder.Conv2dRelu, TGen.tgConv2d, ArgGen.agNone),
        "types": {
            "tf": [tf.float32],
            "tflite": [
                tf.float32,
                QuantType.CONV_U8_U8,
                QuantType.CONV_I8_I8,
                QuantType.CONV_I16_I8,
            ],
        },
        "template": True,
    },
    "conv2d_relu6_TEMPLATE": {
        "operands": (1, 2),
        "build_fcn": (TBuilder.Conv2dRelu6, TGen.tgConv2d, ArgGen.agNone),
        "types": {
            "tf": [tf.float32],
            "tflite": [
                tf.float32,
                QuantType.CONV_U8_U8,
                QuantType.CONV_I8_I8,
                QuantType.CONV_I16_I8,
            ],
        },
        "template": True,
    },
    "conv2d_relu_n1_to_1_TEMPLATE": {
        "operands": (1, 2),
        "build_fcn": (TBuilder.Conv2dReluN1To1, TGen.tgConv2d, ArgGen.agNone),
        "types": {
            "tf": [tf.float32],
            "tflite": [
                tf.float32,
                QuantType.CONV_U8_U8,
                QuantType.CONV_I8_I8,
                QuantType.CONV_I16_I8,
            ],
        },
        "template": True,
    },
    # This test is converted as:
    # tfl.conv2d(){fused_activation_function="NONE"} + tfl.tanh()
    # TODO: anyway to generate tfl.conv2d(){fused_activation_function="TANH"}?
    "conv2d_tanh_TEMPLATE": {
        "operands": (1, 2),
        "build_fcn": (TBuilder.Conv2dTanh, TGen.tgConv2d, ArgGen.agNone),
        "types": {
            "tf": [tf.float32],
            "tflite": [
                tf.float32,
                QuantType.CONV_U8_U8,
                QuantType.CONV_I8_I8,
                QuantType.CONV_I16_I8,
            ],
        },
        "template": True,
    },
    "conv2d_bias_TEMPLATE": {
        "operands": (1, 2),
        "build_fcn": (TBuilder.Conv2dWithBias, TGen.tgConv2d, ArgGen.agConv2d),
        "types": {
            "tf": [tf.float32],
            "tflite": [
                tf.float32,
                QuantType.CONV_U8_U8,
                QuantType.CONV_I8_I8,
                QuantType.CONV_I16_I8,
            ],
        },
        "bias": True,
        "template": True,
    },
    "conv3d_TEMPLATE": {
        "operands": (1, 1),
        "build_fcn": (TBuilder.Conv3d, TGen.tgConv3d, ArgGen.agConv3d),
        "types": {
            "tf": [tf.float32],
            "tflite": [
                tf.float32,
                QuantType.CONV_U8_U8,
                QuantType.CONV_I8_I8,
                # Quantization to 16x8-bit not yet supported by tflite.
            ],
        },
        "template": True,
        "rank": (1, 5),
    },
    "conv3d_bias_TEMPLATE": {
        "operands": (1, 2),
        "build_fcn": (TBuilder.Conv3dWithBias, TGen.tgConv3d, ArgGen.agConv3d),
        "types": {
            "tf": [tf.float32],
            "tflite": [
                tf.float32,
                QuantType.CONV_U8_U8,
                QuantType.CONV_I8_I8,
                # Quantization to 16x8-bit not yet supported by tflite.
            ],
        },
        "bias": True,
        "template": True,
        "rank": (1, 5),
    },
    "depthwise_conv2d_TEMPLATE": {
        "operands": (1, 1),
        "build_fcn": (
            TBuilder.DepthwiseConv2d,
            TGen.tgDepthwiseConv2d,
            ArgGen.agDepthwiseConv2d,
        ),
        "types": {
            "tf": [tf.float32],
            "tflite": [
                tf.float32,
                QuantType.CONV_U8_U8,
                QuantType.CONV_I8_I8,
                QuantType.CONV_I16_I8,
            ],
        },
        "template": True,
    },
    "depthwise_conv2d_bias_TEMPLATE": {
        "operands": (1, 2),
        "build_fcn": (
            TBuilder.DepthwiseConv2dWithBias,
            TGen.tgDepthwiseConv2d,
            ArgGen.agDepthwiseConv2d,
        ),
        "types": {
            "tf": [tf.float32],
            "tflite": [
                tf.float32,
                QuantType.CONV_U8_U8,
                QuantType.CONV_I8_I8,
                QuantType.CONV_I16_I8,
            ],
        },
        "bias": True,
        "template": True,
    },
    "transpose_conv2d_TEMPLATE": {
        "operands": (1, 1),
        "build_fcn": (
            TBuilder.TransposeConv2d,
            TGen.tgTransposeConv2d,
            ArgGen.agTransposeConv2d,
        ),
        "types": {
            "tf": [tf.float32],
            "tflite": [
                tf.float32,
                QuantType.CONV_U8_U8,
                QuantType.CONV_I8_I8,
                QuantType.CONV_I16_I8,
            ],
        },
        "template": True,
    },
    "argmax": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.Argmax, TGen.tgBasic, ArgGen.agAxes),
        "types": {"tf": TYPE_F},
    },
    "avg_pool2d": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.AvgPool2d, TGen.tgPooling, ArgGen.agPooling),
        "types": {
            "tf": TYPE_F,
            "tflite": list(
                TYPE_F + [QuantType.ALL_U8, QuantType.ALL_I8, QuantType.ALL_I16]
            ),
        },
    },
    "max_pool2d": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.MaxPool2d, TGen.tgPooling, ArgGen.agPooling),
        "types": {
            "tf": TYPE_F,
            "tflite": list(TYPE_F + [QuantType.ALL_U8, QuantType.ALL_I8]),
            # ALL_I16 not supported yet
            # In tensorflow/compiler/mlir/lite/ir/tfl_ops.td,
            # QI16 is missing from MaxPoolOperandAndResultConstraints
            # If adding QI16 back this test can run through.
        },
    },
    "reshape": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.Reshape, TGen.tgBasic, ArgGen.agReshape),
        "types": TYPE_FI,
    },
    "transpose": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.Transpose, TGen.tgBasic, ArgGen.agTranspose),
        "types": TYPE_FI,
    },
    "slice": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.Slice, TGen.tgBasic, ArgGen.agSlice),
        "types": TYPE_FI,
    },
    "strided_slice": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.StridedSlice, TGen.tgBasic, ArgGen.agStridedSlice),
        "types": TYPE_FI,
    },
    "select": {
        "operands": (3, 0),
        "build_fcn": (TBuilder.Select, TGen.tgSelect, ArgGen.agNone),
        "types": TYPE_FI,
    },
    "addn": {
        "operands": (4, 0),
        "build_fcn": (TBuilder.Addn, TGen.tgBasic, ArgGen.agNone),
        "types": TYPE_FI,
    },
    "concatv2": {
        "operands": (4, 0),
        "build_fcn": (TBuilder.Concatv2, TGen.tgBasic, ArgGen.agAxes),
        "types": TYPE_FI,
        "rank": (0, 4),
        "custom_shapes": {
            "custom_shape_only": False,
            "shape_list": [()],
        },
    },
    "stack": {
        "operands": (4, 0),
        "build_fcn": (TBuilder.Stack, TGen.tgBasic, ArgGen.agStack),
        "types": TYPE_FI,
    },
    "unstack": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.Unstack, TGen.tgPooling, ArgGen.agAxes),
        "types": TYPE_F,
    },
    "mirrorpad": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.MirrorPad, TGen.tgBasic, ArgGen.agMirrorPad),
        "types": TYPE_FI,
    },
    "pad": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.Pad, TGen.tgBasic, ArgGen.agPad),
        "types": {
            "tf": TYPE_F,
            "tflite": list(TYPE_F + [QuantType.ALL_U8, QuantType.ALL_I8]),
        },
    },
    "expand_dims": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.ExpandDims, TGen.tgBasic, ArgGen.agStack),
        "types": TYPE_FI,
    },
    "shape": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.Shape, TGen.tgBasic, ArgGen.agNone),
        "types": TYPE_FI,
    },
    "rank": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.Rank, TGen.tgBasic, ArgGen.agNone),
        "types": TYPE_FI,
    },
    "fill": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.Fill, TGen.tgBasic, ArgGen.agFill),
        "types": TYPE_FI,
    },
    "elu": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.Elu, TGen.tgBasic, ArgGen.agNone),
        "types": TYPE_F,
    },
    "softmax": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.Softmax, TGen.tgBasic, ArgGen.agNone),
        "types": {
            "tf": TYPE_F,
            "tflite": list(
                TYPE_F + [QuantType.ALL_U8, QuantType.ALL_I8, QuantType.ALL_I16]
            ),
        },
    },
    "log_softmax": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.LogSoftmax, TGen.tgBasic, ArgGen.agNone),
        "types": TYPE_F,
    },
    "dynamic_linear": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.DynamicLinear, TGen.tgBasic, ArgGen.agNone),
        "types": {
            "tf": [],
            "tflite": list(TYPE_F),
        },
        "custom_shapes": {
            "custom_shape_only": True,
            "shape_list": [(14, 19)],
        },
        # number of operands of tuples which spcifies which dim to set to None
        # In this case, we have 1 input. So we have 1 tuple
        # We're setting the first input's first dim to None
        "dynamic_shape_dim": [
            (0,),
        ],
    },
    "matmul": {
        "operands": (2, 0),
        "build_fcn": (TBuilder.MatMul, TGen.tgMatmul, ArgGen.agNone),
        "types": {
            "tf": TYPE_F,
            "tflite": list(
                TYPE_F
                + [QuantType.ALL_U8, QuantType.ALL_I8]
                # 16 bits matmul fail to convert
            ),
        },
    },
    "add_scalar": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.AddScalar, TGen.tgBasic, ArgGen.agNone),
        "types": TYPE_F,
    },
    "add_1d": {
        "operands": (2, 0),
        "build_fcn": (TBuilder.Add1d, TGen.tgBasic, ArgGen.agNone),
        "types": TYPE_F,
    },
    "split": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.Split, TGen.tgBasic, ArgGen.agSplit),
        "types": TYPE_FI,
    },
    "tile": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.Tile, TGen.tgBasic, ArgGen.agTile),
        "types": TYPE_FI,
    },
    "reverse": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.Reverse, TGen.tgBasic, ArgGen.agAxes),
        "types": {"tf": TYPE_FI},
    },
    "gather": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.Gather, TGen.tgBasic, ArgGen.agGather),
        "types": TYPE_FI,
    },
    "gather_nd": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.GatherNd, TGen.tgBasic, ArgGen.agGatherND),
        "types": TYPE_FI,
    },
    "scatter_nd": {
        "operands": (1, 2),
        "build_fcn": (TBuilder.ScatterNd, TGen.tgScatterND, ArgGen.agNone),
        "types": TYPE_FI,
    },
    "space_to_batch": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.SpaceToBatch, TGen.tgBasic, ArgGen.agSpaceToBatch),
        "types": TYPE_F,
    },
    "batch_to_space": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.BatchToSpace, TGen.tgBasic, ArgGen.agBatchToSpace),
        "types": TYPE_F,
    },
    "space_to_depth": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.SpaceToDepth, TGen.tgBasic, ArgGen.agSpaceToDepth),
        "types": TYPE_F,
    },
    "depth_to_space": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.DepthToSpace, TGen.tgBasic, ArgGen.agDepthToSpace),
        "types": TYPE_F,
    },
    "one_hot": {
        "operands": (3, 1),
        "build_fcn": (TBuilder.OneHot, TGen.tgOneHot, ArgGen.agOneHot),
        "types": TYPE_FI,
    },
    "fakequant": {
        "operands": (1, 0),
        "build_fcn": (
            TBuilder.Fakequant,
            TGen.tgBasic,
            ArgGen.agFakequant,
        ),
        "types": {"tf": TYPE_F},
    },
    "resize": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.Resize, TGen.tgPooling, ArgGen.agResize),
        "types": {
            "tf": TYPE_F,
            "tflite": list(
                TYPE_F + [QuantType.ALL_U8, QuantType.ALL_I8, QuantType.ALL_I16]
            ),
        },
        "custom_shapes": {
            "custom_shape_only": False,
            "shape_list": [(3, 1, 1, 7)],
        },
    },
    "left_shift": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.LeftShift, TGen.tgBasic, ArgGen.agShift),
        "types": {"tf": [tf.int32, tf.int16, tf.int8]},
    },
    "right_shift": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.RightShift, TGen.tgBasic, ArgGen.agShift),
        "types": {"tf": [tf.int32, tf.int16, tf.int8]},
    },
    "while": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.While, TGen.tgBasic, ArgGen.agNone),
        "types": {
            "tflite": list(TYPE_F),
        },
    },
    "lstm": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.LSTM, TGen.tgRecurrent, ArgGen.agNone),
        "types": {
            "tflite": [
                tf.float32,
                # tf.int32
            ]
        },
    },
    "lstm_stateful": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.SLSTM, TGen.tgRecurrent, ArgGen.agNone),
        "types": {
            "tflite": [
                tf.float32,
            ]
        },
        "num_variables": 2,
    },
    "gru": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.GRU, TGen.tgRecurrent, ArgGen.agNone),
        "types": {
            "tflite": [
                tf.float32,
                # tf.int32
            ]
        },
    },
    "rnn": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.RNN, TGen.tgRecurrent, ArgGen.agNone),
        "types": {
            "tflite": [
                tf.float32,
            ]
        },
    },
    "callonce": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.CallOnce, TGen.tgBasic, ArgGen.agNone),
        "types": {
            "tflite": [tf.float32],
        },
        "custom_shapes": {
            "custom_shape_only": True,
            "shape_list": [(1,)],
        },
    },
    "rfft2d": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.RFFT2d, TGen.tgRFFT2d, ArgGen.agRFFT2d),
        "types": {
            "tflite": TYPE_F,
        },
    },
    "real": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.Real, TGen.tgComplexComponents, ArgGen.agNone),
        "types": {
            "tflite": [tf.complex64],
        },
    },
    "imag": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.Imag, TGen.tgComplexComponents, ArgGen.agNone),
        "types": {
            "tflite": [tf.complex64],
        },
    },
    "broadcastto": {
        "operands": (1, 1),
        "build_fcn": (TBuilder.BroadcastTo, TGen.tgBroadcastTo, ArgGen.agNone),
        "types": {
            "tf": TYPE_FIB,
        },
    },
    "reduce_max_special_fp": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.ReduceMax, TGen.tgReduce, ArgGen.agAxesListKeepdims),
        "types": {
            "tf": TYPE_F,
            # This test generates special floating number such as nan and inf that cannot be quantized.
        },
        "custom_shapes": {
            "custom_shape_only": True,
            "shape_list": [(14, 19)],
        },
    },
    "reduce_min_special_fp": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.ReduceMin, TGen.tgReduce, ArgGen.agAxesListKeepdims),
        "types": {
            "tf": TYPE_F,
            # This test generates special floating number such as nan and inf that cannot be quantized.
        },
        "custom_shapes": {
            "custom_shape_only": True,
            "shape_list": [(14, 19)],
        },
    },
}

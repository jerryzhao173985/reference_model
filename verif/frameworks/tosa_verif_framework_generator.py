#!/usr/bin/env python3
# Copyright (c) 2020-2022, ARM Limited.
# SPDX-License-Identifier: Apache-2.0
import argparse
import os
import re
import traceback

import numpy as np

#  Level | Level for Humans | Level Description
# -------|------------------|------------------------------------
#  0     | DEBUG            | [Default] Print all messages
#  1     | INFO             | Filter out INFO messages
#  2     | WARNING          | Filter out INFO & WARNING messages
#  3     | ERROR            | Filter out all messages
# Filter tensorflow debug message except errors
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Flake8 E402 - ignore imports not at top of file to allow os.environ setting
import tensorflow as tf  # noqa: E402
from frameworks.write_test_json import write_test_json  # noqa: E402
from frameworks.arg_gen import ArgGen  # noqa: E402
from frameworks.tensor_gen import TGen  # noqa: E402
from frameworks.test_builder import TBuilder  # noqa: E402
from frameworks.test_gen_utils import (  # noqa: E402
    QuantType,
    get_tf_dtype,
    get_shape_str,
)  # noqa: E402
from tensorflow.lite.python.interpreter import OpResolverType  # noqa: E402

# All of the supported frameworks
ALL_FRAMEWORKS = ["tf", "tflite"]

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
#   'template': boolean (indicates that this is a templated op which gets further
#               processing in createDynamicOpLists)
#   'bias':     boolean indicating that there is a bias component to be generated
#   'qtypes':   List of QuantType quantized types to generate for this op

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
    "concat": {
        "operands": (2, 0),
        "build_fcn": (TBuilder.Concat, TGen.tgBasic, ArgGen.agAxes),
        "types": TYPE_FI,
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
        "types": {"tf": TYPE_B},
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
        "build_fcn": (TBuilder.Pow, TGen.tgBFuzz, ArgGen.agNone),
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
        "build_fcn": (TBuilder.Rsqrt, TGen.tgBasic, ArgGen.agNone),
        "types": TYPE_F,
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
    "square": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.Square, TGen.tgBasic, ArgGen.agNone),
        "types": TYPE_F,
    },
    "squared_difference": {
        "operands": (2, 0),
        "build_fcn": (TBuilder.SquaredDifference, TGen.tgBFuzz, ArgGen.agNone),
        "types": TYPE_F,
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
    "pad": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.Pad, TGen.tgBasic, ArgGen.agPad),
        "types": TYPE_F,
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
        "operands": (1, 0),
        "build_fcn": (TBuilder.ScatterNd, TGen.tgBasic, ArgGen.agScatterND),
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
    "resize_nearest": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.ResizeNearest, TGen.tgPooling, ArgGen.agNone),
        "types": {
            "tf": TYPE_F,
            "tflite": list(
                TYPE_F + [QuantType.ALL_U8, QuantType.ALL_I8, QuantType.ALL_I16]
            ),
        },
    },
    "resize_bilinear": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.ResizeBilinear, TGen.tgPooling, ArgGen.agNone),
        "types": {
            "tf": TYPE_F,
            "tflite": list(
                TYPE_F + [QuantType.ALL_U8, QuantType.ALL_I8, QuantType.ALL_I16]
            ),
        },
    },
    "left_shift": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.LeftShift, TGen.tgBasic, ArgGen.agShift),
        "types": {"tf": [tf.int32]},
    },
    "right_shift": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.RightShift, TGen.tgBasic, ArgGen.agShift),
        "types": {
            "tf": [
                tf.int32,
            ]
        },
    },
}

# Shapes to be tested; default can be overwritten
shape_list = [
    (1,),
    (64,),
    (14, 19),
    (13, 21, 3),
    (1, 4, 4, 4),
    (1, 8, 4, 17),
    (1, 4, 8, 19),
    (1, 32, 32, 8),
    (1, 7, 7, 9),
]


def gen_rand_shapes(args):
    """Overwrite the global shape list with a new list of random shapes"""
    global shape_list

    rng = np.random.default_rng(args.random_seed)

    # Don't let things get too big... cap the maximum volume, but let
    # an individual dimension be 1..47
    max_total_volume = 32 * 32 * 4

    shape_list = []
    # Only iterate over ranks 2, 3, and 4
    for rank in range(2, 5):
        for n in range(args.random_shapes):
            new_shape = rng.integers(1, 48, size=rank)

            # Set the batch dimension on 4D objects to 1
            if rank == 4:
                new_shape[0] = 1

            # Limit the total shape volume and throw out any
            # shapes that wouldn't leave at least size=2 in some non-batch dimension
            volume = 1
            skip_shape = False
            for i in range(rank):

                volume *= new_shape[i]

                # Reduce the shape, while it's larger than the maximum volume
                while volume > max_total_volume:
                    new_shape[i] = new_shape[i] // 2
                    volume = volume // 2

                    # Now an untenable dimension size?  Skip this one.
                    if new_shape[i] < 1:
                        skip_shape = True

            if not skip_shape:
                shape_list.append(tuple(new_shape))


# Construct, run and save a whole tensorflow tf.function to a protobuf file
# or convert to .tflite if it's quantized unit test
def run_unit_test(
    op_name,
    args,
    test_dir,
    curr_shape,
    addl_args,
    dtype,
    excluded_framework_list,
    quantized_inference_dtype,
    result_name,
    seed,
):

    try:
        op = TF_OP_LIST[op_name]
        op_fcn, tensor_gen_fcn, arg_gen_fcn = op["build_fcn"]

        # Get and seed a random number generator for this test
        rng = np.random.default_rng(seed)

        # return placeholders=(str: name, np.array: value)
        # consts=(str: name, np.array: value)
        placeholders, consts = tensor_gen_fcn(op, curr_shape, dtype, rng)

        # if test doesn't have any placeholders/consts, terminated
        if len(placeholders) == 0 and len(consts) == 0:
            return True

        if not args.quiet:
            print("   {}              ".format(test_dir))

        try:
            os.mkdir(test_dir)
        except FileExistsError:
            pass

        const_nodes = [value for name, value in consts]

        num_placeholders = len(placeholders)
        # if test is quantized, create tensor quantization metadata info for
        # each input tensor, based on different quantized type
        if quantized_inference_dtype:
            is_quantized = True
            # TODO: support INT8 IFM x INT4 weight later
            if quantized_inference_dtype == QuantType.ALL_U8:
                qzero = [128] * num_placeholders
                numpy_dtype = [np.uint8] * num_placeholders
                tflite_inference_dtype = tf.uint8
            elif quantized_inference_dtype == QuantType.ALL_I8:
                qzero = [0] * num_placeholders
                numpy_dtype = [np.int8] * num_placeholders
                tflite_inference_dtype = tf.int8
            elif quantized_inference_dtype == QuantType.ALL_I16:
                qzero = [0] * num_placeholders
                numpy_dtype = [np.int16] * num_placeholders
                tflite_inference_dtype = tf.int16
            elif quantized_inference_dtype == QuantType.CONV_U8_U8:
                assert (
                    num_placeholders == 1
                ), "Unsupported number of placeholders for Convolution: {}".format(
                    num_placeholders
                )
                qzero = [128] * num_placeholders
                if num_placeholders == 2:
                    numpy_dtype = [np.uint8, np.uint8]
                else:
                    numpy_dtype = [np.uint8, np.uint8, np.int32]
                tflite_inference_dtype = tf.uint8
            elif quantized_inference_dtype == QuantType.CONV_I8_I8:
                assert (
                    num_placeholders == 1
                ), "Unsupported number of placeholders for Convolution: {}".format(
                    num_placeholders
                )
                qzero = [0] * num_placeholders
                if num_placeholders == 2:
                    numpy_dtype = [np.int8, np.int8]
                else:
                    numpy_dtype = [np.int8, np.int8, np.int32]
                tflite_inference_dtype = tf.int8
            elif quantized_inference_dtype == QuantType.CONV_I16_I8:
                assert (
                    num_placeholders == 1
                ), "Unsupported number of placeholders for Convolution: {}".format(
                    num_placeholders
                )
                if num_placeholders == 2:
                    qzero = [0, 0]
                    numpy_dtype = [np.int16, np.int8]
                else:
                    qzero = [0, 0, 0]
                    numpy_dtype = [
                        np.int16,
                        np.int8,
                        np.int64,
                    ]  # np.int64 to represent 40 bits accumulator
                tflite_inference_dtype = tf.int16
            else:
                raise Exception(
                    "Unsupported fakequant dtype: {}".format(quantized_inference_dtype)
                )

        else:
            is_quantized = False

        tf_model_filename = None
        tf_result_npy_filename = None
        tf_result_name = None

        tflite_model_filename = None
        tflite_result_npy_filename = None
        tflite_result_name = None

        placeholder_names = []
        placeholder_vals = []
        placeholder_signatures = ()
        placeholder_npy_filenames = []
        placeholder_shapes = []

        for idx, (name, val) in enumerate(placeholders):
            placeholder_names.append(name)
            placeholder_signatures = placeholder_signatures + (
                tf.TensorSpec(shape=val.shape, dtype=val.dtype, name=name),
            )
            placeholder_npy_filenames.append("{}.npy".format(name.split(":")[0]))
            placeholder_shapes.append(val.shape)

        # Get test builder class
        fcn_node = op_fcn(*const_nodes, *addl_args, result_name)
        concrete_function = tf.function(input_signature=placeholder_signatures)(
            fcn_node.eval
        ).get_concrete_function()

        if is_quantized:

            assert dtype is tf.float32, "quantized test must come from float32 graph"

            # 1. Quantize float placeholder npy to quantized to feed the graph
            for idx, (name, val) in enumerate(placeholders):

                # we use np.amin()/np.amax() to determine dynamic range
                # for quantized test
                zeropoint = 0
                scale = 1.0
                if numpy_dtype[idx] != np.int64:
                    qmin = np.iinfo(numpy_dtype[idx]).min
                    qmax = np.iinfo(numpy_dtype[idx]).max
                    num_bits = np.iinfo(numpy_dtype[idx]).bits
                # 40 bit is represented as np.int64
                else:
                    num_bits = 40
                    qmin = -(1 << num_bits)
                    qmax = (1 << num_bits) - 1

                min_val = np.amin(val)
                max_val = np.amax(val)

                # for single value tensor, we set scale equal to the abs(value),
                # and fix zeropoint to 128
                # if val > 0, it'll be represented as 129,
                #    where val = (129 - 128) * val
                # if val < 0, it'll be represented as 127,
                #    where val = (127 - 128) * (-val)
                # if val == 0, it'll be represted as 128, with range [-128.0, 128.0]
                # and let quantized 1 represent the value
                # also adjust effective min/max consequently
                if max_val == min_val:
                    if max_val != 0:
                        scale = abs(max_val)
                    else:
                        scale = 1.0
                    min_val = float(qmin - qzero[idx]) * scale
                    max_val = float(qmax - qzero[idx]) * scale
                else:
                    scale = (max_val - min_val) / float(qmax - qmin)
                    zeropoint = int(round((-min_val) / scale)) + qmin

                # run through tf.fakequant first to assure quantization error aligned
                fakequant_val = tf.quantization.fake_quant_with_min_max_args(
                    val,
                    min=min_val,
                    max=max_val,
                    num_bits=num_bits,
                    name="gen_quant_npy",
                )

                quant_val = np.round(fakequant_val / scale).astype(np.int32) + zeropoint

                # very few unit tests after TF hash may/2020, this quantized
                # value for some reason exceed [0, 255] range
                saved_val = np.clip(quant_val, qmin, qmax).astype(numpy_dtype[idx])

                # saved all quantized tensor as np.int32
                # since TOSA numpy Cpp API only supports int32
                np.save(
                    os.path.join(test_dir, placeholder_npy_filenames[idx]),
                    saved_val.astype(np.int32),
                    False,
                )

                placeholder_vals.append(tf.convert_to_tensor(saved_val))

            # 2. Convert the model to quantized TFLite flatbuffer
            module = tf.Module()
            converter = tf.lite.TFLiteConverter.from_concrete_functions(
                [concrete_function], module
            )
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.experimental_new_converter = True

            # use MLIR-based post-quantizer
            converter.experimental_new_quantizer = True

            flag = (
                tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8  # noqa: E501
            )
            if tflite_inference_dtype == tf.int16:
                converter.target_spec.supported_ops = [flag]

            def input_stats():
                for i in range(0, args.num_samples):
                    a = [
                        TGen.getRand(shape, tf.float32, rng)
                        for shape in placeholder_shapes
                    ]
                    yield a

            converter.representative_dataset = input_stats
            converter.inference_input_type = tflite_inference_dtype
            converter.inference_output_type = tflite_inference_dtype

            tflite_model = converter.convert()

            tflite_model_filename = "model.tflite"

            # Write out converted model to disk
            with open(os.path.join(test_dir, tflite_model_filename), "wb") as f:
                f.write(tflite_model)

        else:  # is_quantized is False

            # 1. Saved out numpy array directly
            for idx, (name, val) in enumerate(placeholders):
                placeholder_vals.append(tf.convert_to_tensor(val))
                np.save(
                    os.path.join(test_dir, placeholder_npy_filenames[idx]), val, False
                )

            # 2.a Saved out .pb if framework includes tensorflow
            if "tf" not in excluded_framework_list:
                # Write out graph as protobuf to disk
                tf_model_filename = "model.pb"
                tf.io.write_graph(
                    concrete_function.graph, test_dir, tf_model_filename, True
                )

            # 2.b Saved out .tflite if framework includes tflite
            if "tflite" not in excluded_framework_list:
                # Convert the model to TFLite flatbuffer
                module = tf.Module()
                converter = tf.lite.TFLiteConverter.from_concrete_functions(
                    [concrete_function], module
                )

                converter.experimental_new_converter = True

                # Even it's non-quantized int32 test, this needs to be set to tf.float32
                converter.inference_input_type = tf.float32
                converter.inference_output_type = tf.float32
                tflite_model = converter.convert()

                # Write out converted model to disk
                tflite_model_filename = "model.tflite"
                with open(os.path.join(test_dir, tflite_model_filename), "wb") as f:
                    f.write(tflite_model)

        # Get TF reference result if .pb is specified
        if tf_model_filename:
            tf_result_npy_filename = "tf_result.npy"
            tf_result = concrete_function(*placeholder_vals)
            np.save(os.path.join(test_dir, tf_result_npy_filename), tf_result, False)

            tf_result_name = result_name

        # Get TFLite inference result if .tflite is specified
        if tflite_model_filename:
            tflite_result_npy_filename = "tflite_result.npy"

            ops_with_optimized_only_kernel = ["elu", "ceil", "gather"]

            if args.tflite_kernel_mode == "optimized" or (
                op_name in ops_with_optimized_only_kernel
            ):
                interpreter = tf.lite.Interpreter(
                    model_path=os.path.join(test_dir, tflite_model_filename)
                )
            elif args.tflite_kernel_mode == "reference":
                interpreter = tf.lite.Interpreter(
                    model_path=os.path.join(test_dir, tflite_model_filename),
                    experimental_op_resolver_type=OpResolverType.BUILTIN_REF,
                )
            else:
                assert 0, "unknown tflite interpreter mode {}".format(
                    args.tflite_kernel_mode
                )
            interpreter.allocate_tensors()

            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            assert len(input_details) == len(
                placeholder_vals
            ), "number of placeholder mismatch"

            for idx, val in enumerate(placeholder_vals):
                interpreter.set_tensor(input_details[idx]["index"], val.numpy())

            interpreter.invoke()
            tflite_result = interpreter.get_tensor(output_details[0]["index"])

            np.save(
                os.path.join(test_dir, tflite_result_npy_filename), tflite_result, False
            )

            # Result tensor name would change after converting to TFLite flatbuffer
            # Overwrite the information from TFLite models directly.
            # Assume single result tensor now
            tflite_result_name = output_details[0]["name"]

        # Write out test descriptor
        write_test_json(
            filename=os.path.join(test_dir, "test.json"),
            tf_model_filename=tf_model_filename,
            tf_result_npy_filename=tf_result_npy_filename,
            tf_result_name=tf_result_name,
            tflite_model_filename=tflite_model_filename,
            tflite_result_npy_filename=tflite_result_npy_filename,
            tflite_result_name=tflite_result_name,
            ifm_name=placeholder_names,
            ifm_file=placeholder_npy_filenames,
            ifm_shape=placeholder_shapes,
            framework_exclusions=excluded_framework_list,
            quantized=is_quantized,
        )
    except Exception as e:
        msg = "Error running task: {}".format(e)
        print(msg)
        print(
            "".join(
                traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)
            )
        )
        return False
    return True


def build_const_net(
    args,
    curr_shape,
    op_name,
    dtype,
    excluded_framework_list,
    quantized_inference_dtype,
    result_name,
    seed,
    rng,
    filter,
    unit_test_args,
):

    if quantized_inference_dtype:
        quant_dtype = get_tf_dtype(quantized_inference_dtype)
        test_dir = "test_{}_{}".format(op_name, get_shape_str(curr_shape, quant_dtype))
    else:
        test_dir = "test_{}_{}".format(op_name, get_shape_str(curr_shape, dtype))
    test_dir = os.path.join(args.output_dir, test_dir)

    # If the operator has an additional function to generate arguments, call it
    # here and iterate through the argument list that it generates
    op = TF_OP_LIST[op_name]
    op_fcn, tensor_gen_fcn, arg_gen_fcn = op["build_fcn"]

    addl_args_tuple = arg_gen_fcn(op, curr_shape, rng)
    for desc, addl_args in addl_args_tuple:
        # Only filter on the full test_name, not the output directory
        _, test_name = os.path.split(test_dir + desc)
        if not filter or filter.search(test_name):
            unit_test_args.append(
                [
                    op_name,
                    args,
                    test_dir + desc,
                    curr_shape,
                    addl_args,
                    dtype,
                    excluded_framework_list,
                    quantized_inference_dtype,
                    result_name,
                    seed,
                ]
            )


# python hash is not reproducible, create hash for our purpose
def op_name_hash(op_name):
    result = 0xDEADBEEF
    for ch in op_name:
        if result & 1:
            result = (ord(ch) << 24) ^ (result >> 1) ^ 0x82608EDB
        else:
            result = (ord(ch) << 24) ^ (result >> 1)

    return result


def generate_op_tests(args, op_name, shape_list, result_name, filter, unit_test_args):

    if not args.quiet:
        print(
            "Generating tests for {}                                        ".format(
                op_name
            )
        )

    op = TF_OP_LIST[op_name]

    # Seed the RNG so that we get the same random tests for each test each time
    # If the number of tests for a given generation function changes, the tests
    # for that operator may also change accordingly, but this will at least keep
    # down churn across operators.

    bounded_hash_val = (args.random_seed + op_name_hash(op_name)) % np.iinfo(
        np.int32
    ).max
    rng = np.random.default_rng(bounded_hash_val)

    # this is a dictionary with 'tf' and 'tflite' as key
    # and value being the data types we want to test under these framework

    if isinstance(op["types"], dict):
        try:
            tf_dtypes = op["types"]["tf"]
        except KeyError:
            tf_dtypes = []
        try:
            tflite_dtypes = op["types"]["tflite"]
        except KeyError:
            tflite_dtypes = []
    elif isinstance(op["types"], list):
        tf_dtypes = op["types"]
        tflite_dtypes = op["types"]

    tf_nonquantized_dtypes = tf_dtypes  # tf doesn't support quantized data types
    tflite_quantized_dtypes = []
    tflite_nonquantized_dtypes = []
    for dtype in tflite_dtypes:
        if isinstance(dtype, QuantType):
            tflite_quantized_dtypes.append(dtype)
        else:
            tflite_nonquantized_dtypes.append(dtype)

    nonquantized_dtypes_set = set(tf_nonquantized_dtypes).union(
        set(tflite_nonquantized_dtypes)
    )
    nonquantized_dtypes = list(nonquantized_dtypes_set)
    quantized_dtypes = tflite_quantized_dtypes

    # populate non quantized unit test arguments
    for dtype in nonquantized_dtypes:

        excluded_framework_set = set(ALL_FRAMEWORKS)
        if dtype in tf_nonquantized_dtypes:
            excluded_framework_set.remove("tf")
        if dtype in tflite_nonquantized_dtypes:
            excluded_framework_set.remove("tflite")
        excluded_framework_list = list(excluded_framework_set)

        for curr_shape in shape_list:
            build_const_net(
                args,
                curr_shape,
                op_name,
                dtype,
                excluded_framework_list,
                None,
                result_name,
                bounded_hash_val,
                rng,
                filter,
                unit_test_args,
            )

    # populate quantized unit test arguments
    # must exclude 'tf' and source dtype being tf.float32
    for dtype in quantized_dtypes:
        for curr_shape in shape_list:
            build_const_net(
                args,
                curr_shape,
                op_name,
                tf.float32,
                ["tf"],
                dtype,
                result_name,
                bounded_hash_val,
                rng,
                filter,
                unit_test_args,
            )

    return unit_test_args


def createDynamicOpLists():
    """The templated operators are conv2d-style operators with a number of kernel
    sizes.  Since the operator is unchanged, we generate the range of kernel
    sizes here in this loop and remove the original templates from the list.

    This could be expanded to non-conv2d-style operators in the future."""

    # Dynamically create op lists for convolutions with a list of kernel sizes
    KERNELS = [
        [1, 1],
        [3, 3],
        [5, 5],
    ]

    TEMPLATE_LIST = [
        "conv2d",
        "conv2d_bias",
        "conv2d_relu",
        "conv2d_relu6",
        "conv2d_relu_n1_to_1",
        "conv2d_tanh",
        "depthwise_conv2d",
        "depthwise_conv2d_bias",
        "transpose_conv2d",
    ]

    for t in TEMPLATE_LIST:
        for k in KERNELS:
            testName = "{}_{}x{}".format(t, k[0], k[1])
            TF_OP_LIST[testName] = TF_OP_LIST["{}_TEMPLATE".format(t)].copy()
            TF_OP_LIST[testName]["filter"] = k
            TF_OP_LIST[testName]["template"] = False

    # Delete any templates after having created any dynamic ops
    # This is a two-pass operation because it's bad practice to delete
    # keys from dictionaries while iterating
    keyList = []
    for k in TF_OP_LIST:
        try:
            if TF_OP_LIST[k]["template"]:
                keyList.append(k)
                continue
        except KeyError:
            pass

    for k in keyList:
        del TF_OP_LIST[k]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", dest="random_seed", default=42, type=int, help="Random seed"
    )
    parser.add_argument(
        "--random-shapes",
        dest="random_shapes",
        default=0,
        type=int,
        help=(
            "Use N random shapes of each rank for generating tests,"
            "seeded with random seed"
        ),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        default=".",
        type=str,
        help="Test output directory path prefix",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        dest="quiet",
        default=False,
        action="store_true",
        help="Do not print test names",
    )
    parser.add_argument(
        "-j", "--jobs", dest="jobs", type=int, default=1, help="Number of parallel jobs"
    )
    parser.add_argument(
        "-m",
        "--tflite-kernel-mode",
        dest="tflite_kernel_mode",
        type=str,
        choices=["reference", "optimized"],
        default="reference",
        help="TFLite interpreter kernel mode",
    )
    parser.add_argument(
        "--num-samples",
        dest="num_samples",
        default=200,
        type=int,
        help="Number of input samples for post-training quantization",
    )
    parser.add_argument(
        "--filter",
        dest="filter",
        default="",
        type=str,
        help="Filter test names by this expression",
    )
    args = parser.parse_args()

    # Turn the filter into a re object if present
    filter = None
    if args.filter != "":
        filter = re.compile(args.filter)

    # Autodetect CPU count
    if args.jobs <= 0:
        args.jobs = os.cpu_count()

    # Disable TF info messages
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    try:
        os.makedirs(args.output_dir)
    except FileExistsError:
        pass

    if args.random_shapes:
        gen_rand_shapes(args)

    # Build dynamic ops
    createDynamicOpLists()

    # Generate the test list and arguments to run_unit_test()
    unit_test_args = []

    for op in TF_OP_LIST:
        generate_op_tests(args, op, shape_list, "result", filter, unit_test_args)

    errors = 0
    for t in unit_test_args:
        if not run_unit_test(*t):
            errors = errors + 1

    if not args.quiet:
        print("\nAll tasks done - with {} errors".format(errors))

    return 1 if errors else 0


if __name__ == "__main__":
    exit(main())

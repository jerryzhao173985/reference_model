# Copyright (c) 2025, ARM Limited.
# SPDX-License-Identifier: Apache-2.0
import torch
from frameworks.shape_list import shape_list_conv2d
from frameworks.shape_list import shape_list_linear
from frameworks.shape_list import shape_list_matmul_2d
from frameworks.shape_list import shape_list_matmul_3d
from frameworks.torch.arg_gen import ArgGen
from frameworks.torch.tensor_gen import TGen
from frameworks.torch.test_builder import TBuilder

# Lists of different data types
TYPE_F = [torch.float32]
TYPE_I = [torch.int32]
TYPE_FI = [torch.float32, torch.int32]
TYPE_B = [torch.bool]
TYPE_FIB = [torch.float32, torch.int32, torch.bool]
TYPE_H = [torch.float16]
TYPE_FH = [torch.float32, torch.float16]
TYPE_FHI = [torch.float32, torch.float16, torch.int32]
TYPE_FHIB = [torch.float32, torch.float16, torch.int32, torch.bool]

# The list of operator tests
# Each dictionary entry for an op is a dictionary with the following required members:
#   'operands': tuple (number_of_placeholder_tensors, number_of_constant_tensors)
#   'build_fcn: tuple (Test builder function, Tensor generator function,
#                      Argument generator function)
#   'types': list of Torch types that should be tested for this op
#
# And optional members:
#   'template':      boolean (indicates that this is a templated op which gets further
#                    processing in createDynamicOpLists)
#   'bias':          boolean indicating that there is a bias component to be generated
#   'qtypes':        List of QuantType quantized types to generate for this op
#   'rank':          tuple (lowest rank, highest rank). Dimension range of input tensor.
#   'custom_shapes': List of custom shapes for specific operators

TORCH_OP_LIST = {
    "log": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.Log, TGen.tgBasic, ArgGen.agNone),
        "types": TYPE_F,
    },
    "exp": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.Exp, TGen.tgBasic, ArgGen.agNone),
        "types": TYPE_F,
    },
    "neg": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.Neg, TGen.tgBasic, ArgGen.agNone),
        "types": TYPE_F,
    },
    "floor": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.Floor, TGen.tgBasic, ArgGen.agNone),
        "types": TYPE_F,
    },
    "rsqrt": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.Rsqrt, TGen.tgBasicPositive, ArgGen.agNone),
        "types": TYPE_F,
    },
    "bitwise_not": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.BitwiseNot, TGen.tgBFuzz, ArgGen.agNone),
        "types": TYPE_I,
    },
    "ceil": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.Ceil, TGen.tgBasic, ArgGen.agNone),
        "types": TYPE_F,
    },
    "rcp": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.Rcp, TGen.tgBasic, ArgGen.agNone),
        "types": TYPE_F,
    },
    "minimum": {
        "operands": (2, 0),
        "build_fcn": (TBuilder.Minimum, TGen.tgBFuzz, ArgGen.agNone),
        "types": TYPE_FI,
    },
    "maximum": {
        "operands": (2, 0),
        "build_fcn": (TBuilder.Maximum, TGen.tgBFuzz, ArgGen.agNone),
        "types": TYPE_FI,
    },
    "logical_or": {
        "operands": (2, 0),
        "build_fcn": (TBuilder.LogicalOr, TGen.tgBFuzz, ArgGen.agNone),
        "types": TYPE_B,
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
    "bitwise_and": {
        "operands": (2, 0),
        "build_fcn": (TBuilder.BitwiseAnd, TGen.tgBFuzz, ArgGen.agNone),
        "types": TYPE_I,
    },
    "bitwise_or": {
        "operands": (2, 0),
        "build_fcn": (TBuilder.BitwiseOr, TGen.tgBFuzz, ArgGen.agNone),
        "types": TYPE_I,
    },
    "bitwise_xor": {
        "operands": (2, 0),
        "build_fcn": (TBuilder.BitwiseXor, TGen.tgBFuzz, ArgGen.agNone),
        "types": TYPE_I,
    },
    "add": {
        "operands": (2, 0),
        "build_fcn": (TBuilder.Add, TGen.tgBFuzz, ArgGen.agNone),
        "types": TYPE_FI,
    },
    "sub": {
        "operands": (2, 0),
        "build_fcn": (TBuilder.Sub, TGen.tgBFuzz, ArgGen.agNone),
        "types": TYPE_FI,
    },
    "mul": {
        "operands": (2, 0),
        "build_fcn": (TBuilder.Mul, TGen.tgBFuzz, ArgGen.agNone),
        "types": TYPE_FI,
    },
    "div": {
        "operands": (2, 0),
        "build_fcn": (TBuilder.Div, TGen.tgBFuzz, ArgGen.agNone),
        "types": TYPE_FI,
    },
    "reduce_mean": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.ReduceMean, TGen.tgBasic, ArgGen.agNone),
        "types": TYPE_F,
    },
    "reduce_sum": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.ReduceSum, TGen.tgBasic, ArgGen.agNone),
        "types": TYPE_F,
    },
    "reduce_any": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.ReduceAny, TGen.tgBasic, ArgGen.agNone),
        "types": TYPE_B,
    },
    "reduce_all": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.ReduceAll, TGen.tgBasic, ArgGen.agNone),
        "types": TYPE_B,
    },
    "squeeze": {
        "operands": (1, 0),
        "build_fcn": [TBuilder.Squeeze, TGen.tgBasic, ArgGen.agNone],
        "types": TYPE_F,
    },
    "squeeze_dim": {
        "operands": (1, 0),
        "build_fcn": [TBuilder.SqueezeDim, TGen.tgBasic, ArgGen.agSqueezeDim],
        "types": TYPE_F,
    },
    "matmul": {
        "operands": (2, 0),
        "build_fcn": (TBuilder.MatMul, TGen.tgMatmul, ArgGen.agNone),
        "types": TYPE_F,
    },
    "mm": {
        "operands": (2, 0),
        "build_fcn": (TBuilder.Mm, TGen.tgMm, ArgGen.agNone),
        "types": TYPE_F,
        "custom_shapes": {
            "custom_shape_only": True,
            "shape_list": shape_list_matmul_2d,
        },
    },
    "bmm": {
        "operands": (2, 0),
        "build_fcn": (TBuilder.Bmm, TGen.tgMm, ArgGen.agNone),
        "types": TYPE_F,
        "custom_shapes": {
            "custom_shape_only": True,
            "shape_list": shape_list_matmul_3d,
        },
    },
    "linear": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.Linear, TGen.tgBasic, ArgGen.agNone),
        "types": TYPE_F,
        "custom_shapes": {
            "custom_shape_only": True,
            "shape_list": shape_list_linear,
        },
    },
    "fill_int": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.FillInt, TGen.tgBasic, ArgGen.agFill),
        "types": TYPE_I,
    },
    "fill_float": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.FillFloat, TGen.tgBasic, ArgGen.agFill),
        "types": TYPE_F,
    },
    "tanh": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.Tanh, TGen.tgBasic, ArgGen.agNone),
        "types": TYPE_F,
    },
    "sigmoid": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.Sigmoid, TGen.tgBasic, ArgGen.agNone),
        "types": TYPE_F,
    },
    "relu": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.Relu, TGen.tgBasic, ArgGen.agNone),
        "types": TYPE_F,
    },
    "leaky_relu": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.LeakyRelu, TGen.tgBasic, ArgGen.agFloat),
        "types": TYPE_F,
    },
    "gelu": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.Gelu, TGen.tgBasic, ArgGen.agNone),
        "types": TYPE_F,
    },
    "abs": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.Abs, TGen.tgBasic, ArgGen.agNone),
        "types": TYPE_F,
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
    "avg_pool2d_TEMPLATE": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.AvgPool2d, TGen.tgPooling, ArgGen.agPooling),
        "types": TYPE_F,
        "template": True,
        "custom_shapes": {
            "custom_shape_only": True,
            "shape_list": shape_list_conv2d,
        },
    },
    "adaptive_avgpool2d": {
        "operands": (1, 0),
        "build_fcn": (
            TBuilder.AdaptiveAvgPool2d,
            TGen.tgPooling,
            ArgGen.agAdaptivePooling,
        ),
        "types": TYPE_F,
        "custom_shapes": {
            "custom_shape_only": True,
            "shape_list": shape_list_conv2d,
        },
    },
    "conv2d_TEMPLATE": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.Conv2d, TGen.tgConv2d, ArgGen.agConv2d),
        "types": TYPE_F,
        "template": True,
        "custom_shapes": {
            "custom_shape_only": True,
            "shape_list": shape_list_conv2d,
        },
    },
    "conv2d_bias_TEMPLATE": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.Conv2dWithBias, TGen.tgConv2d, ArgGen.agConv2d),
        "types": TYPE_F,
        "bias": True,
        "template": True,
        "custom_shapes": {
            "custom_shape_only": True,
            "shape_list": shape_list_conv2d,
        },
    },
    "maxpool2d_TEMPLATE": {
        "operands": (1, 0),
        "build_fcn": (TBuilder.MaxPool2d, TGen.tgPooling, ArgGen.agPooling),
        "types": TYPE_F,
        "template": True,
        "custom_shapes": {
            "custom_shape_only": True,
            "shape_list": shape_list_conv2d,
        },
    },
}

#!/usr/bin/env python3

# Copyright (c) 2020-2021, ARM Limited.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import numpy as np
import argparse
import sys
import re
import os
import subprocess
import shlex
import json
import glob
import math
import queue
import threading
import traceback
import math
import itertools
from copy import deepcopy

from enum import IntEnum, Enum, unique
from tosa_ref_run import TosaReturnCode

# Include the ../thirdparty/serialization_lib/python directory in PYTHONPATH
parent_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(
    os.path.join(parent_dir, "..", "thirdparty", "serialization_lib", "python")
)
import tosa_serializer as ts
from tosa_serializer import *
import tosa
from tosa_error_if import ErrorIf

# Convenience variables to the flatc-generated types that should be enums, but aren't
DType = tosa.DType.DType()
Op = tosa.Op.Op()
ResizeMode = tosa.ResizeMode.ResizeMode()


def product(shape):
    value = 1
    for n in shape:
        value *= n
    return value

class TosaQuantGen:
    """QuantizedInfo random generator helper functions.  Specify with 'qgen': in the operator defintion"""

    def __init__(self):
        pass

    @staticmethod
    def getQinfo(testGen, dtype, error_name=None):

        if dtype == DType.INT8:
            return testGen.randInt(-128, 128)
        elif dtype == DType.UINT8:
            return testGen.randInt(0, 256)
        elif error_name in [ErrorIf.InputZeroPointNotZero, ErrorIf.WeightZeroPointNotZero, ErrorIf.OutputZeroPointNotZero]:
            zero_point = testGen.randInt(-128, 128)
            if zero_point == 0:
                zero_point = 1
            return zero_point
        return 0

    @staticmethod
    def qgUnary(testGen, op, dtype, error_name=None):
        qinfo = ts.TosaSerializerQuantInfo()
        if error_name == ErrorIf.InputZeroPointNotZero:
            qinfo.UnaryQuantInfo(
                TosaQuantGen.getQinfo(testGen, dtype, error_name), TosaQuantGen.getQinfo(testGen, dtype)
            )
        elif error_name == ErrorIf.OutputZeroPointNotZero:
            qinfo.UnaryQuantInfo(
                TosaQuantGen.getQinfo(testGen, dtype), TosaQuantGen.getQinfo(testGen, dtype, error_name)
            )
        else:
            qinfo.UnaryQuantInfo(
                TosaQuantGen.getQinfo(testGen, dtype), TosaQuantGen.getQinfo(testGen, dtype)
            )
        return qinfo

    @staticmethod
    def qgConv(testGen, op, dtype_or_dtypeList, error_name=None):
        qinfo = ts.TosaSerializerQuantInfo()
        if isinstance(dtype_or_dtypeList, list):
            # a list of [input, weights, accumulator] dtypes
            dtypeList = dtype_or_dtypeList
        else:
            # an int, [input, weights, accumulator] dtypes are the same
            dtypeList = [dtype_or_dtypeList] * 3

        if error_name == ErrorIf.InputZeroPointNotZero:
            input_zp = TosaQuantGen.getQinfo(testGen, dtypeList[0], error_name)
            weights_zp = TosaQuantGen.getQinfo(testGen, dtypeList[1])
        elif error_name == ErrorIf.WeightZeroPointNotZero:
            input_zp = TosaQuantGen.getQinfo(testGen, dtypeList[0])
            weights_zp = TosaQuantGen.getQinfo(testGen, dtypeList[1], error_name)
        else:
            input_zp = TosaQuantGen.getQinfo(testGen, dtypeList[0])
            weights_zp = TosaQuantGen.getQinfo(testGen, dtypeList[1])

        qinfo.ConvQuantInfo(input_zp, weights_zp)
        return qinfo

    @staticmethod
    def qgMatmul(testGen, op, dtype, error_name=None):
        qinfo = ts.TosaSerializerQuantInfo()
        if error_name == ErrorIf.InputZeroPointNotZero:
            qinfo.MatMulQuantInfo(
                TosaQuantGen.getQinfo(testGen, dtype, error_name), TosaQuantGen.getQinfo(testGen, dtype, error_name)
        )
        else:
            qinfo.MatMulQuantInfo(
                TosaQuantGen.getQinfo(testGen, dtype), TosaQuantGen.getQinfo(testGen, dtype)
            )
        return qinfo

    @staticmethod
    def qgPad(testGen, op, dtype, error_name=None):
        qinfo = ts.TosaSerializerQuantInfo()
        if error_name == ErrorIf.InputZeroPointNotZero:
            qinfo.PadQuantInfo(TosaQuantGen.getQinfo(testGen, dtype, error_name))
        else:
            qinfo.PadQuantInfo(TosaQuantGen.getQinfo(testGen, dtype))
        return qinfo

    @staticmethod
    def computeMultiplierAndShift(scaleFp, scale32):
        # Derived from computeMultiplierAndShiftTosaScale32
        # Provide a floating-point scaling factor and the scale32 parameter
        # to compute the multiplier and shift

        if scale32:
            scaleBits = 31
        else:
            scaleBits = 15

        m, shift = math.frexp(scaleFp)

        if scaleFp < 0.0:
            m = -m

        multiplier = round(m * (1 << scaleBits))
        assert multiplier <= (1 << scaleBits)

        if multiplier == (1 << scaleBits):
            multiplier = multiplier // 2
            shift = shift + 1

        shift = (-shift) + scaleBits
        #print('scalefp {} scaleBits {} m {} mult {} shift {}'.format(scaleFp, scaleBits, m, multiplier, shift))

        # Adjust multiplier such that shift is in allowed value range.
        if shift == 0:
            multiplier = multiplier // 4
            shift = shift + 2
        elif shift == 1:
            multiplier = multiplier // 2
            shift = shift + 1
        elif shift == 63:
            multiplier = multiplier * 2
            shift = shift - 1

        assert multiplier <= (1 << scaleBits)
        assert shift >= 2 and shift <= 62

        return multiplier, shift


class TosaTensorGen:
    """Tensor generators create a shape list for the placeholder and const tensor
    data operands for the operator.  The actual random data is generated separately for each test."""

    def __init__(self):
        pass

    @staticmethod
    def tgBasic(testGen, opName, rank, error_name=None):
        pl, const = opName["operands"]
        shape = testGen.makeShape(rank)

        # Constrict the overall size of the shape when creating ERROR_IF tests
        if error_name:
            shape = TosaErrorIfArgGen.eiRestrictDimensions(shape)

        shape_list = []
        for i in range(pl + const):
            shape_list.append(shape.copy())

            if error_name == ErrorIf.RankMismatch:
                if rank == 1 and i != 1:
                    shape = testGen.makeShape(rank + testGen.rng.choice([1, 2, 3]))
                elif i != 1:
                    shape = testGen.makeShape(rank + testGen.rng.choice([-1, 1]))

        return shape_list

    @staticmethod
    def tgNHWC(testGen, opName, rank, error_name=None):
        pl, const = opName["operands"]

        if error_name != ErrorIf.WrongRank:
            assert rank == 4

        shape = testGen.makeShape(rank)

        # Constrict the batch size?
        if testGen.args.max_batch_size:
            shape[0] = (shape[0] % testGen.args.max_batch_size) + 1

        # Constrict the overall size of the shape when creating ERROR_IF tests
        if error_name:
            shape = TosaErrorIfArgGen.eiRestrictDimensions(shape)

        shape_list = []
        for i in range(pl + const):
            shape_list.append(shape.copy())

        return shape_list

    @staticmethod
    def tgScatter(testGen, opName, rank, error_name=None):
        pl, const = opName["operands"]

        assert pl == 2
        assert const == 0
        assert rank == 3

        values_in_shape = testGen.makeShape(rank)

        # ignore max batch size if target shape is set
        if testGen.args.max_batch_size and not testGen.args.target_shapes:
            values_in_shape[0] = (values_in_shape[0] % testGen.args.max_batch_size) + 1

        W = testGen.randInt(
            testGen.args.tensor_shape_range[0], testGen.args.tensor_shape_range[1]
        )
        # Constrict W if one dimension is too large to keep tensor size reasonable
        if max(values_in_shape) > 5000:
            W = testGen.randInt(0, 16)

        input_shape = [values_in_shape[0], W, values_in_shape[2]]

        shape_list = []
        shape_list.append(values_in_shape.copy())
        shape_list.append(input_shape.copy())

        return shape_list

    @staticmethod
    def tgBroadcastFuzz(testGen, op, rank, error_name=None):
        shape = testGen.makeShape(rank)

        pl, const = op["operands"]

        shape_list = []

        # Choose one of the inputs to broadcast
        bcast_idx = testGen.randInt(0, pl + const)
        for i in range(pl + const):
            shape_bcast = shape.copy()

            if error_name == ErrorIf.RankMismatch:
                bcast_idx = -1 # Turn off broadcast because we are not testing it
                if rank == 1 and i != 1:
                    shape_bcast = testGen.makeShape(rank + testGen.rng.choice([1, 2, 3]))
                elif i != 1:
                    shape_bcast = testGen.makeShape(rank + testGen.rng.choice([-1, 1]))

            # If the chosen input, pick a random index to broadcast
            if i == bcast_idx:
                fuzz_idx = testGen.randInt(0, rank)
                shape_bcast[fuzz_idx] = 1

            shape_list.append(shape_bcast)

        return shape_list

    @staticmethod
    def tgConv2D(testGen, op, rank, error_name=None):
        pl, const = op["operands"]

        assert rank == 4

        # IFM dimensions are NHWC
        ifm_shape = testGen.makeShape(rank)

        # Constrict the batch size?
        if testGen.args.max_batch_size:
            ifm_shape[0] = (ifm_shape[0] % testGen.args.max_batch_size) + 1

        # Get the filter height/width from the operator parameters
        filter_hw = op["filter"]

        # Generate a random OFM depth
        ofm_depth = testGen.makeShape(1)[0]

        # The filter dimensions are OHWI
        filter_shape = np.asarray([ofm_depth, filter_hw[0], filter_hw[1], ifm_shape[3]])

        # The bias is OC
        bias_shape = np.asarray([ofm_depth])

        return [ifm_shape, filter_shape, bias_shape]

    @staticmethod
    def tgConv3D(testGen, op, rank, error_name=None):
        pl, const = op["operands"]

        assert rank == 5

        # IFM dimensions are NDHWC
        ifm_shape = testGen.makeShape(rank)

        # Constrict the batch size?
        if testGen.args.max_batch_size:
            ifm_shape[0] = (ifm_shape[0] % testGen.args.max_batch_size) + 1

        # Get the filter depth/height/width from the operator parameters
        filter_dhw = op["filter"]

        # Generate a random OFM channel
        ofm_channel = testGen.makeShape(1)[0]

        # The filter dimensions are ODHWI
        filter_shape = np.asarray(
            [ofm_channel, filter_dhw[0], filter_dhw[1], filter_dhw[2], ifm_shape[4]]
        )

        # The bias is OC
        bias_shape = np.asarray([ofm_channel])

        return [ifm_shape, filter_shape, bias_shape]

    @staticmethod
    def tgTransposeConv2D(testGen, op, rank, error_name=None):
        pl, const = op["operands"]

        assert rank == 4

        # IFM dimensions are NHWC
        ifm_shape = testGen.makeShape(rank)

        # Constrict the batch size?
        if testGen.args.max_batch_size:
            ifm_shape[0] = (ifm_shape[0] % testGen.args.max_batch_size) + 1

        # Get the filter height/width from the operator parameters
        filter_hw = op["filter"]

        # Generate a random OFM depth
        ofm_depth = testGen.makeShape(1)[0]

        # The filter dimensions are OHWI
        filter_shape = np.asarray([ofm_depth, filter_hw[0], filter_hw[1], ifm_shape[3]])

        # The bias is OC
        bias_shape = np.asarray([ofm_depth])

        return [ifm_shape, filter_shape, bias_shape]

    @staticmethod
    def tgDepthwiseConv2D(testGen, op, rank, error_name=None):
        pl, const = op["operands"]

        assert rank == 4
        assert pl == 1 and const == 2

        # IFM dimensions are NHWC
        ifm_shape = testGen.makeShape(rank)

        # Constrict the batch size?
        if testGen.args.max_batch_size:
            ifm_shape[0] = (ifm_shape[0] % testGen.args.max_batch_size) + 1

        # Get the filter height/width from the operator parameters
        # Filter is KH, HW, C, M
        filter_hw = op["filter"]

        # Generate a random OFM depth, but don't let it get too big because
        # the output depth is M * C
        filter_m = (
            testGen.makeShape(1)[0] % (testGen.args.tensor_shape_range[1] // 4)
        ) + 1

        # The filter dimensions are HWCM
        filter_shape = np.asarray([filter_hw[0], filter_hw[1], ifm_shape[3], filter_m])

        # The bias is M * C
        bias_shape = np.asarray([ifm_shape[3] * filter_m])

        return [ifm_shape, filter_shape, bias_shape]

    @staticmethod
    def tgFullyConnected(testGen, op, rank, error_name=None):
        pl, const = op["operands"]

        if error_name != ErrorIf.WrongRank:
            assert rank == 2

        input_shape = testGen.makeShape(rank)

        # Constrict the overall size of the shape when creating ERROR_IF tests
        if error_name:
            shape = TosaErrorIfArgGen.eiRestrictDimensions(shape)

        filter_oc = testGen.rng.integers(
            low=testGen.args.tensor_shape_range[0],
            high=testGen.args.tensor_shape_range[1],
            size=1,
        )[0]
        filter_shape = np.asarray([filter_oc, input_shape[1]])

        bias_shape = np.asarray([filter_oc])

        return [input_shape, filter_shape, bias_shape]

    @staticmethod
    def tgMatmul(testGen, op, rank, error_name=None):
        pl, const = op["operands"]

        if error_name != ErrorIf.WrongRank:
            assert rank == 3
        assert pl == 2 and const == 0

        a_shape = testGen.makeShape(rank)

        # Constrict the overall size of the shape when creating ERROR_IF tests
        if error_name:
            shape = TosaErrorIfArgGen.eiRestrictDimensions(shape)

        # Get a random number for b_oc even if target shape is defined
        b_oc = np.int32(
            testGen.rng.integers(
                low=testGen.args.tensor_shape_range[0],
                high=testGen.args.tensor_shape_range[1],
                size=1,
            )
        )[0]
        # If N or H is large let b_oc be 1 to reduce output tensor size
        if max(a_shape) > 1000:
            b_oc = 1

        b_shape = np.asarray([a_shape[0], a_shape[2], b_oc])
        return [a_shape, b_shape]

    @staticmethod
    def tgConcat(testGen, opName, rank, error_name=None):
        pl, const = opName["operands"]
        shape = testGen.makeShape(rank)

        # Create extra tensors to concat.
        # Take into account value of pl when getting maximum number of concats
        num_tensors = testGen.randInt(0, 4)
        shape_list = []
        for i in range(pl + const + num_tensors):
            if error_name == ErrorIf.ConcatInputRankMismatch and i != 0:
                remove = testGen.rng.choice([True, False])
                wrongShape = shape.copy()

                if remove and len(shape) > 1:
                    wrongShape = wrongShape[1:]
                else:
                    wrongShape = list(wrongShape)
                    wrongShape.append(testGen.rng.integers(1, 10))

                shape_list.append(wrongShape)
            else:
                shape_list.append(shape.copy())

        return shape_list

    @staticmethod
    def tgConcatConstInput(testGen, shapeList, axis, error_name=None):
        if error_name in [ErrorIf.AxisSmallerZero, ErrorIf.AxisLargerRank, ErrorIf.ConcatInputRankMismatch]:
            return shapeList

        # Split concat shape along axis to allow for multiple const inputs
        # without making too many large tensors
        if len(shapeList) == 2 or shapeList[0][axis] < len(shapeList):
            # If axis can't be split we still need to invalidate other dimensions
            if error_name == ErrorIf.ConcatInputDimMismatch:
                for shape in shapeList[1:]:
                    # Negative test shapeLists are created individually for each test,
                    # so no need to copy the shape before altering it.
                    shape[(axis + 1) % len(shape)] += testGen.rng.integers(5, 10)
            return shapeList

        # Create copy of shape we are going to split (so we don't alter shapeList)
        shape = shapeList[0].copy()
        # Add original shape as first input
        new_shapeList = [shape.copy()]
        length_on_axis = shape[axis]
        remaining_length = length_on_axis
        for i in range(len(shapeList) - 2):
            # Calculate split on axis and remaining value
            split_shape_val = int(shape[axis] / 2)
            remaining_length = remaining_length - split_shape_val

            # Append new shape, and set remaining shape
            shape[axis] = split_shape_val
            new_shapeList.append(shape.copy())

            # invalidate dimensions
            if error_name == ErrorIf.ConcatInputDimMismatch:
                shape[(axis + 1) % len(shape)] += testGen.rng.integers(5, 10)
            else:
                shape[axis] = remaining_length

            if i == len(shapeList) - 3:
                new_shapeList.append(shape.copy())

        return new_shapeList


class TosaArgGen:
    """Argument generators create exhaustive or random lists of attributes for operators that take
    attributes or other parameters.  The return value is a list of (descriptive_name, [arglist])
    tuples where the descriptive_name is appended to the test name and the arglist is expanded
    as arguments to the operator build function."""

    def __init__(self):
        pass

    @staticmethod
    def agNone(testGen, opName, shapeList, dtype, error_name=None):
        """A trivial argument generator for operators that don't take any
        non-tensor arguments"""
        return [("", [])]

    @staticmethod
    def agAxis(testGen, opName, shapeList, dtype, error_name=None):
        """Build the axis argument for operators that take a single axis"""
        axes = []
        shape = shapeList[0]

        if error_name == ErrorIf.AxisSmallerZero:
            small_axis = testGen.rng.integers(-5, 0)
            axes.append(("axis{}".format(small_axis), [small_axis]))
        elif error_name == ErrorIf.AxisLargerRank:
            large_axis = testGen.rng.integers(len(shape) + 1, len(shape) + 10)
            axes.append(("axis{}".format(large_axis), [large_axis]))
        else:
            for a in range(0, len(shape)):
                axes.append(("axis{}".format(a), [a]))

        return axes

    @staticmethod
    def agConv(testGen, opName, shapeList, dtype, error_name=None):
        arg_list = []

        ifm_shape = shapeList[0]
        filter_shape = shapeList[1]
        # determine the kernel shape from the operator name (e.g. "conv2d_3x3" => [3,3])
        k = [int(x) for x in opName.split("_")[-1].split("x")]

        # Check the rank
        rank = 5 if opName.startswith("conv3d") else 4
        assert len(ifm_shape) == rank
        assert len(filter_shape) == rank

        # kernel rank omits batch and channels
        k_rank = rank - 2

        # Generate comprehensive argument lists
        p_vals = [x for x in range(0, testGen.args.max_conv_padding + 1)]
        paddings = {x for x in itertools.product(*([p_vals] * k_rank * 2))}
        s_vals = [x for x in range(1, testGen.args.max_conv_stride + 1)]
        strides = {x for x in itertools.product(*([s_vals] * k_rank))}
        d_vals = [x for x in range(1, testGen.args.max_conv_dilation + 1)]
        dilations = {x for x in itertools.product(*([d_vals] * k_rank))}

        # add some oversize argument values
        if max(ifm_shape) < 64:
            bigPadding = 9
            paddings.update({x for x in itertools.product(*([[0, bigPadding]] * (k_rank * 2)))})
        bigStride = 8
        strides.update({x for x in itertools.product(*([[1, bigStride]] * k_rank))})
        bigDilation = 7
        dilations.update({x for x in itertools.product(*([[1, bigDilation]] * k_rank))})

        # There are too many parameter combinations, so generate them sparsely
        # To get a variety of parameter combinations sparsity should not be a multiple of 2, 3 or 5
        sparsity = len(paddings) * len(strides) * len(dilations) // 100 + 1
        if sparsity < 13:
            sparsity = 1
        while sparsity % 2 == 0 or sparsity % 3 == 0 or sparsity % 5 == 0:
            sparsity += 1
        n = 0
        for s in sorted(list(strides)):
            for p in sorted(list(paddings)):
                for d in sorted(list(dilations)):
                    if (n % sparsity == 0
                        # padding must not exceed the kernel size ?
                        # and p[0] < k[0] and p[1] < k[0] and p[2] < k[1] and p[3] < k[1]
                        # and (k_rank < 3 or (p[4] < k[2] and p[5] < k[2]))
                        # the padded shape must exceed the kernel size
                        and (ifm_shape[1] + p[0] + p[1]) > k[0] and (ifm_shape[2] + p[2] + p[3]) > k[1]
                        and (k_rank < 3 or ((ifm_shape[3] + p[4] + p[5]) > k[2]))
                        # the padded shape must exceed the dilation
                        and (ifm_shape[1] + p[0] + p[1]) > d[0] and (ifm_shape[2] + p[2] + p[3]) > d[1]
                        and (k_rank < 3 or ((ifm_shape[3] + p[4] + p[5]) > d[2]))
                    ):
                        arg_list.append(
                            (
                                "st{}_pad{}_dilat{}".format(
                                    "".join([str(x) for x in s]),
                                    "".join([str(x) for x in p]),
                                    "".join([str(x) for x in d]),
                                ),
                                [s, p, d],
                            )
                        )
                    n += 1

        return arg_list

    @staticmethod
    def agTransposeConv2D(testGen, opName, shapeList, dtype, error_name=None):
        arg_list = []

        ifm_shape = shapeList[0]
        filter_shape = shapeList[1]

        # Must be rank 4
        assert len(ifm_shape) == 4
        assert len(filter_shape) == 4

        # Generate comprehensive argument lists
        p_vals = [x for x in range(0, testGen.args.max_conv_padding + 1)]
        paddings = {x for x in itertools.product(*([p_vals] * 2))}
        s_vals = [x for x in range(1, testGen.args.max_conv_stride + 1)]
        strides = {x for x in itertools.product(*([s_vals] * 2))}
        d_vals = [x for x in range(1, testGen.args.max_conv_dilation + 1)]
        dilations = {x for x in itertools.product(*([d_vals] * 2))}

        # add some oversize argument values
        if max(ifm_shape) < 64:
            bigPadding = 9
            paddings.update({x for x in itertools.product(*([[0, bigPadding]] * 2))})
        bigStride = 8
        strides.update({x for x in itertools.product(*([[1, bigStride]] * 2))})
        bigDilation = 7
        dilations.update({x for x in itertools.product(*([[1, bigDilation]] * 2))})

        # There are too many parameter combinations, so generate them sparsely
        # To get a variety of parameter combinations sparsity should not be a multiple of 2, 3 or 5
        sparsity = len(paddings) * len(strides) * len(dilations) // 100 + 1
        if sparsity < 13:
            sparsity = 1
        while sparsity % 2 == 0 or sparsity % 3 == 0 or sparsity % 5 == 0:
            sparsity += 1
        n = 0
        for s in sorted(list(strides)):
            for p in sorted(list(paddings)):
                for d in sorted(list(dilations)):
                    if n % sparsity == 0:
                        # Determine the output shape
                        oh = (
                            ifm_shape[1]
                            - filter_shape[1]
                            - (filter_shape[1] - 1) * (d[0] - 1)
                            + 2 * p[0]
                        ) // s[0] + 1
                        ow = (
                            ifm_shape[2]
                            - filter_shape[2]
                            - (filter_shape[2] - 1) * (d[1] - 1)
                            + 2 * p[1]
                        ) // s[1] + 1
                        os = [ifm_shape[0], oh, ow, filter_shape[0]]
                        arg_list.append(
                            (
                                "st{}_pad{}_dilat{}_os{}".format(
                                    "".join([str(x) for x in s]),
                                    "".join([str(x) for x in p]),
                                    "".join([str(x) for x in d]),
                                    "x".join([str(x) for x in os]),
                                ),
                                [s, p, d, os],
                            )
                        )
                    n += 1

        return arg_list

    @staticmethod
    def agPad(testGen, opName, shapeList, dtype, error_name=None):
        arg_list = []
        rank = len(shapeList[0])

        # Exhaustively test combinations of padding on each side of each dimension
        # - the range of padding values is defined by pad_min and pad_max
        # - for padding >9, the name format needs to be more distinctive
        pad_min, pad_max = 0, 1
        pad_values = [x for x in range(pad_min, pad_max + 1)]
        if error_name == ErrorIf.PadSmallerZero:
            pad_values = [x for x in range(-2, 0)]
        axis_pad_values = [x for x in itertools.product(pad_values, pad_values)]
        shape_pad_values = itertools.product(*([axis_pad_values] * rank))

        if dtype in [DType.BOOL, DType.INT8, DType.INT16, DType.INT32]:
            pad_const_int = testGen.getRandNumberDType(dtype)
            pad_const_fp = 0
        elif dtype == DType.FLOAT:
            pad_const_int = 0
            pad_const_fp = testGen.getRandNumberDType(dtype)
        else:
            return []

        for paddings in shape_pad_values:
            name = "pad"
            for r in range(rank):
                before, after = paddings[r]
                name = f"{name}{before}{after}"
            arg_list.append((name, [np.array(paddings), pad_const_int, pad_const_fp]))

        return arg_list

    @staticmethod
    def agPooling(testGen, opName, shapeList, dtype, error_name=None):
        arg_list = []

        shape = shapeList[0]
        if error_name != ErrorIf.WrongRank:
            assert len(shape) == 4

        # Generate comprehensive argument lists
        p_vals = [x for x in range(0, testGen.args.max_pooling_padding + 1)]
        paddings = {x for x in itertools.product(*([p_vals] * 4))}
        s_vals = [x for x in range(1, testGen.args.max_pooling_stride + 1)]
        strides = {x for x in itertools.product(*([s_vals] * 2))}
        k_vals = [x for x in range(2, testGen.args.max_pooling_kernel + 2)]
        kernels = {x for x in itertools.product(*([k_vals] * 2))}

        # add some oversize argument values
        bigStride = 7
        strides.update({x for x in itertools.product(*([[1, bigStride]] * 2))})
        bigKernel = 6
        kernels.update({x for x in itertools.product(*([[2, bigKernel]] * 2))})
        if max(shape) < 64:
            # padding must be less than the kernel size
            bigPadding = bigKernel - 1
            paddings.update({x for x in itertools.product(*([[0, bigPadding]] * 4))})

        # There are too many parameter combinations, so generate them sparsely
        sparsity = len(paddings) * len(strides) * len(kernels) // 500 + 1
        n = 0
        for s in sorted(list(strides)):
            for p in sorted(list(paddings)):
                for k in sorted(list(kernels)):
                    if error_name in [ErrorIf.StrideSmallerOne, ErrorIf.KernelSmallerOne, ErrorIf.PadSmallerZero, ErrorIf.PadLargerEqualKernel]:
                        sNew, pNew, kNew = TosaErrorIfArgGen.eiPoolingErrorIf(testGen, error_name, s, p, k)
                        if None not in [sNew, pNew, kNew] and n % sparsity == 0:
                            arg_list.append(
                                (
                                    "st{}_kern{}_pad{}".format(
                                        "".join([str(x) for x in sNew]),
                                        "".join([str(x) for x in kNew]),
                                        "".join([str(x) for x in pNew]),
                                    ),
                                    [sNew, pNew, kNew],
                                )
                            )
                    elif (n % sparsity == 0
                        # padding must not exceed the kernel size
                        and p[0] < k[0] and p[1] < k[0] and p[2] < k[1] and p[3] < k[1]
                        # the padded shape must exceed the kernel size
                        and (shape[1] + p[0] + p[1]) > k[0] and (shape[2] + p[2] + p[3]) > k[1]
                    ):
                        arg_list.append(
                            (
                                "st{}_kern{}_pad{}".format(
                                    "".join([str(x) for x in s]),
                                    "".join([str(x) for x in k]),
                                    "".join([str(x) for x in p]),
                                ),
                                [s, p, k],
                            )
                        )
                    n += 1

        return arg_list

    @staticmethod
    def agCast(testGen, opName, shapeList, inDtype, error_name=None):
        arg_list = []

        # Enumerate the output types here
        if error_name == ErrorIf.WrongOutputType:
            dtypeList = TosaErrorIfArgGen.eiCastErrorIf(testGen, inDtype)
        elif inDtype == DType.INT8:
            dtypeList = [DType.BOOL, DType.INT16, DType.INT32, DType.FLOAT]
        elif inDtype == DType.INT16:
            dtypeList = [DType.BOOL, DType.INT8, DType.INT32, DType.FLOAT]
        elif inDtype == DType.INT32:
            dtypeList = [DType.BOOL, DType.INT8, DType.INT16, DType.FLOAT]
        elif inDtype == DType.BOOL:
            dtypeList = [DType.INT8, DType.INT16, DType.INT32]
        elif inDtype == DType.FLOAT:
            dtypeList = [DType.INT8, DType.INT16, DType.INT32]
        elif error_name == ErrorIf.WrongInputType:
            # Pick some potentially correct output type for incorrect input type
            dtypeList = [DType.BOOL, DType.INT8, DType.INT16, DType.FLOAT]
        else:
            raise Exception("Unexpected input dtype: {}".format(inDtype))

        for dtype in dtypeList:
            arg_list.append(("out{}".format(DTypeNames[dtype]), [dtype]))

        return arg_list

    @staticmethod
    def agRescale(testGen, opName, shapeList, inDtype, error_name=None):
        arg_list = []

        # Enumerate the output types here
        for dtype in [DType.UINT8, DType.INT8, DType.INT16, DType.INT32]:
            if dtype in [DType.UINT8, DType.INT8] and error_name == ErrorIf.OutputZeroPointNotZero:
                continue
            if inDtype == DType.UINT8 and dtype != DType.INT8 and error_name != ErrorIf.WrongOutputType:
                # The only output dtype for UINT8 is INT8, skip all other combinations
                continue
            if inDtype != DType.INT8 and dtype == DType.UINT8 and error_name != ErrorIf.WrongOutputType:
                # The only input dtype for UINT8 is INT8, skip all other combinations
                continue
            if error_name == ErrorIf.WrongOutputType and not TosaErrorIfArgGen.eiRescaleWrongOutputType(inDtype, dtype):
                continue

            for scale32 in [False, True]:
                if error_name == ErrorIf.ScaleTrue and scale32 == False:
                    continue
                elif error_name == ErrorIf.ScaleNotTrue and scale32 == True:
                    continue
                for double_round in [False, True]:
                    if error_name == ErrorIf.ScaleNotTrue and double_round == False:
                        continue
                    for per_channel in [False, True]:

                        if inDtype == DType.INT48 and scale32 and error_name != ErrorIf.ScaleTrue:
                            # Illegal condition.  Must be scale32=False
                            continue
                        if double_round and not scale32 and error_name != ErrorIf.ScaleNotTrue:
                            # Illegal condition.  ERROR_IF(!scale32 && double_round)
                            continue

                        arg_list.append(
                            (
                                "out{}_sc{}_dr{}_pc{}".format(
                                    DTypeNames[dtype],
                                    int(scale32),
                                    int(double_round),
                                    int(per_channel),
                                ),
                                [dtype, scale32, double_round, per_channel],
                            )
                        )

        return arg_list

    @staticmethod
    def agMul(testGen, opName, shapeList, dtype, error_name=None):
        arg_list = []

        if dtype is DType.INT32:
            for p in range(testGen.args.num_rand_permutations):

                shift = testGen.randInt(0, 32)

                arg_list.append(("perm{}_shift{}".format(p, shift), [shift]))
        else:
            arg_list.append(("perm0_shift0", [0]))

        return arg_list

    @staticmethod
    def agArithmeticRightShift(testGen, opName, shapeList, dtype, error_name=None):
        arg_list = []

        arg_list.append(("roundTrue", [True]))
        arg_list.append(("roundFalse", [False]))

        return arg_list

    # Helper function for reshape.  Gets some factors of a larger number.
    @staticmethod
    def getFactors(val, start=1):
        factors = []

        for i in range(start, int(np.sqrt(val)) + 1):
            if (val % i) == 0:
                factors.append(i)

        return factors

    @staticmethod
    def agReshape(testGen, opName, shapeList, dtype, error_name=None):
        arg_list = []

        origShape = shapeList[0]

        totalElements = 1
        for s in origShape:
            totalElements *= s

        # This code is NOT fast.  Fortunately, the numbers are fairly small.
        factors = TosaArgGen.getFactors(totalElements)

        for p in range(testGen.args.num_rand_permutations):
            newRank = testGen.randInt(1, 7)
            if len(factors) < newRank:
                continue

            found = True
            # escape_counter breaks while loop if it continues on for too long
            escape_counter = 0
            while found:
                newShape = []
                # Generate newShape ensuring it isn't a duplicate
                remainingElements = totalElements
                shuffledFactors = testGen.rng.permutation(factors)
                for i in range(1, newRank):
                    # pick rank-1 factors
                    newShape.append(shuffledFactors[0])
                    remainingElements = remainingElements // shuffledFactors[0]
                    shuffledFactors = testGen.rng.permutation(
                        TosaArgGen.getFactors(remainingElements)
                    )
                newShape.append(remainingElements)

                # Toss in a -1 sometimes
                minusOne = testGen.randInt(0, newRank * 4)
                if minusOne < newRank:
                    newShape[minusOne] = -1

                # Check for duplicates
                found = False
                for name, other_shape in arg_list:
                    if other_shape[0] == newShape:
                        found = True
                        break

                escape_counter += 1
                if escape_counter >= 100:
                    break

                if not found:
                    arg_list.append(("perm{}_rank{}".format(p, newRank), [newShape]))

        return arg_list

    @staticmethod
    def agTranspose(testGen, opName, shapeList, dtype, error_name=None):
        arg_list = []

        ifm_shape = shapeList[0]


        if error_name == ErrorIf.IndexOutsideBounds:
            incorrect_large_index = range(len(ifm_shape)+1, 2*len(ifm_shape)+1)
            incorrect_small_index = range(-len(ifm_shape), 0)
            permutations = [p for p in itertools.permutations(incorrect_large_index)]
            permutations.extend([p for p in itertools.permutations(incorrect_small_index)])
        elif error_name == ErrorIf.IndexUsedTwice:
            # Create list with a duplicated index
            perm_range = list(range(len(ifm_shape)))
            index_choice = testGen.rng.choice(range(len(perm_range)))
            perm_range[(index_choice + 1) % len(perm_range)] = perm_range[index_choice]
            permutations = [p for p in itertools.permutations(perm_range)]


        else:
            # Get all permutations
            permutations = [p for p in itertools.permutations(range(len(ifm_shape)))]

        # Limit to possible permutations from shape dimension or argument setting
        limit = min(len(permutations), testGen.args.num_rand_permutations)

        # Get random permutation generator that uses all permutations
        random_permutations = testGen.rng.permutation(permutations)

        # Create list of required amount of permutations
        arg_list = [
            ("perm{}".format(p), [random_permutations[p].tolist()])
            for p in range(limit)
        ]
        return arg_list

    @staticmethod
    def agSlice(testGen, opName, shapeList, dtype, error_name=None):
        arg_list = []

        ifm_shape = shapeList[0]
        rank = len(ifm_shape)

        for p in range(testGen.args.num_rand_permutations):
            start = []
            size = []

            valid = True

            for i in range(rank):
                if ifm_shape[i] > 1:
                    start.append(testGen.randInt(0, ifm_shape[i]))
                    size.append(testGen.randInt(0, ifm_shape[i] - start[i]))

                    # Invalid slice size?
                    if size[i] == 0:
                        valid = False
                else:
                    start.append(0)
                    size.append(1)

            if valid:
                # If ERROR_IF test required then incorrect start, size will be returned
                start, size = TosaErrorIfArgGen.eiSliceErrorIf(testGen, error_name, ifm_shape, start, size)
                arg_list.append(("perm{}".format(p), [start, size]))
        return arg_list

    @staticmethod
    def agTile(testGen, opName, shapeList, dtype, error_name=None):
        arg_list = []

        ifm_shape = shapeList[0]
        rank = len(ifm_shape)

        for p in range(testGen.args.num_rand_permutations):

            # Pick a few random, but small multiple values
            # because otherwise this has a tendency to generate
            # enormous tensors
            multiples = []
            for i in range(rank):
                if ifm_shape[i] > 1000:
                    # Multiple of 1 if ifm_shape dimension is large to reduce tensor size
                    multiples.append(1)
                elif max(ifm_shape) > 1000:
                    multiples.append(2)
                else:
                    multiples.append(testGen.randInt(1, 4))
            arg_list.append(("perm{}".format(p), [multiples]))

        return arg_list

    @staticmethod
    def agResize(testGen, opName, shapeList, dtype, error_name=None):
        arg_list = []

        ifm_shape = shapeList[0]
        for mode in [ResizeMode.NEAREST, ResizeMode.BILINEAR]:

            # Exclude illegal {mode, type} configurations.  Pick legal output types
            if mode == ResizeMode.NEAREST and dtype == DType.INT8:
                outputDTypeList = [DType.INT8]
            elif mode == ResizeMode.NEAREST and dtype == DType.INT16:
                outputDTypeList = [DType.INT16]
            elif mode == ResizeMode.BILINEAR and dtype == DType.INT8:
                outputDTypeList = [DType.INT32]
            elif mode == ResizeMode.BILINEAR and dtype == DType.INT16:
                outputDTypeList = [DType.INT48]
            elif dtype == DType.FLOAT:
                outputDTypeList = [DType.FLOAT]
            elif error_name == ErrorIf.WrongInputType:
                # If an incorrect input type is used then we set a 'correct'
                # output type to avoid other errors
                outputDTypeList = [DType.INT8, DType.INT16, DType.INT32]
            else:
                continue

            for outputDType in outputDTypeList:
                for perm in range(testGen.args.num_rand_permutations):
                    # Randomly generate legal output dimensions and shift
                    # and then compute the stride and offset based on them
                    # A output_dim of 1 will cause offset to exceed allowed range
                    # so minimum value 2 produced below
                    output_dims = [testGen.randInt(1) + 1, testGen.randInt(1) + 1]
                    while ((float(ifm_shape[1]) / float(output_dims[0])) >= 16):
                        output_dims[0] += 1
                    while ((float(ifm_shape[2]) / float(output_dims[1])) >= 16):
                        output_dims[1] += 1

                    in_center_h = (ifm_shape[1] - 1) / 2.0
                    in_center_w = (ifm_shape[2] - 1) / 2.0
                    out_center_h = (output_dims[0] - 1) / 2.0
                    out_center_w = (output_dims[1] - 1) / 2.0

                    fp_stride_y = float(ifm_shape[1]) / float(output_dims[0])
                    fp_stride_x = float(ifm_shape[2]) / float(output_dims[1])
                    fp_offset_y = in_center_h - fp_stride_y * out_center_h
                    fp_offset_x = in_center_w - fp_stride_x * out_center_w

                    if outputDType == DType.FLOAT:
                        shift = 0
                        stride = [0, 0]
                        offset = [0, 0]
                        stride_fp = [fp_stride_y, fp_stride_x]
                        offset_fp = [fp_offset_y, fp_offset_x]

                        if error_name is not None:
                            shift, stride, stride_fp, offset, offset_fp, outputDTypeNew = TosaErrorIfArgGen.eiResizeErrorIf(
                                testGen,
                                error_name,
                                mode,
                                dtype,
                                shapeList,
                                outputDType,
                                shift,
                                stride,
                                stride_fp,
                                offset,
                                offset_fp
                            )
                        else:
                            outputDTypeNew = outputDType

                        arg_list.append(
                            (
                                "mode{}_odim{}x{}_out{}_st{:.2f}x{:.2f}_off{:.2f}x{:.2f}".format(
                                    "N" if mode == ResizeMode.NEAREST else "B",
                                    output_dims[0],
                                    output_dims[1],
                                    testGen.typeStr(outputDTypeNew),
                                    stride_fp[0],
                                    stride_fp[1],
                                    offset_fp[0],
                                    offset_fp[1],
                                ),
                                [
                                    mode,
                                    stride,
                                    offset,
                                    shift,
                                    stride_fp,
                                    offset_fp,
                                    output_dims,
                                    dtype,
                                    outputDTypeNew,
                                ],
                            )
                        )
                    else:
                        shift = testGen.randInt(1,12)
                        # Now search for a shift value (1 to 11) that will produce
                        # a valid and predictable resize operation
                        count = 0
                        while (count < 12):
                            unit = float(1 << shift)
                            stride_y = int(round(fp_stride_y * unit))
                            stride_x = int(round(fp_stride_x * unit))
                            offset_y = int(round(fp_offset_y * unit))
                            offset_x = int(round(fp_offset_x * unit))

                            if (
                                stride_y >= (16 << shift)
                                or stride_x >= (16 << shift)
                                or offset_y >= (16 << shift)
                                or offset_x >= (16 << shift)
                                or offset_y <= (-16 << shift)
                                or offset_x <= (-16 << shift)
                            ):
                                # Change the shift value and check again
                                count += 1
                                shift = (shift % 11) + 1
                                continue

                            def RESIZE_REQUIRE_CALC(length_in, length_out, stride, offset, shift):
                                # Perform the pseudo loop to look for out of bounds
                                for pos in range(0,length_out):
                                    a = pos * stride + offset
                                    ia = a >> shift
                                    ia0 = max(ia, 0)
                                    ia1 = min(ia+1, length_in-1)
                                    if ia0 > ia1:
                                        # Found a problem value
                                        break
                                return ia0, ia1

                            iy0, iy1 = RESIZE_REQUIRE_CALC(ifm_shape[1], output_dims[0], stride_y, offset_y, shift)
                            ix0, ix1 = RESIZE_REQUIRE_CALC(ifm_shape[2], output_dims[1], stride_x, offset_x, shift)
                            if ix0 > ix1 or iy0 > iy1:
                                # Change the shift value and check again
                                count += 1
                                shift = (shift % 11) + 1
                                continue
                            break

                        if count >= 12:
                            # Couldn't find a good set of values for this test, skip it
                            continue

                        stride = [stride_y, stride_x]
                        offset = [offset_y, offset_x]

                        stride_fp = [0.0, 0.0]
                        offset_fp = [0.0, 0.0]

                        if error_name is not None:
                            shift, stride, stride_fp, offset, offset_fp, outputDTypeNew = TosaErrorIfArgGen.eiResizeErrorIf(
                                testGen,
                                error_name,
                                mode,
                                dtype,
                                shapeList,
                                outputDType,
                                shift,
                                stride,
                                stride_fp,
                                offset,
                                offset_fp
                            )
                        else:
                            outputDTypeNew = outputDType

                        arg_list.append(
                            (
                                "mode{}_shift{}_odim{}x{}_out{}_st{}x{}_off{}x{}".format(
                                    "N" if mode == ResizeMode.NEAREST else "B",
                                    shift,
                                    output_dims[0],
                                    output_dims[1],
                                    testGen.typeStr(outputDTypeNew),
                                    stride[0],
                                    stride[1],
                                    offset[0],
                                    offset[1],
                                ),
                                [
                                    mode,
                                    stride,
                                    offset,
                                    shift,
                                    stride_fp,
                                    offset_fp,
                                    output_dims,
                                    dtype,
                                    outputDTypeNew,
                                ],
                            )
                        )

        return arg_list

    @staticmethod
    def agTable(testGen, opName, shapeList, dtype, error_name=None):
        arg_list = []

        if dtype == DType.INT8:
            table = np.int32(
                testGen.rng.integers(low=-128, high=128, size=[256])
            ).tolist()
        else:  # INT16
            table = np.int32(
                testGen.rng.integers(low=-32768, high=32768, size=[513])
            ).tolist()

        arg_list.append(
            (
                "",
                [table],
            )
        )
        return arg_list

    def agCondIf(testGen, opName, shapeList, dtype, error_name=None):
        # CondIf generates the condition values here.
        # Convert to tensors in the build function, along with the
        # then and else blocks
        arg_list = []

        for c in [False, True]:
            arg_list.append(("cond{}".format(int(c)), [c]))

        return arg_list

    def agWhileLoop(testGen, opName, shapeList, dtype, error_name=None):
        # While loop: 0 iterations, 1, more than 1
        arg_list = []

        for iter in [0, 1, 4]:
            arg_list.append(("iter{}".format(iter), [iter]))

        return arg_list

class TosaErrorIfArgGen:

    @staticmethod
    def eiResizeErrorIf(testGen, error_name, mode, dtype, shapeList, outputDType, shift, stride, stride_fp, offset, offset_fp):

        if outputDType == DType.FLOAT:
            if error_name == ErrorIf.StrideSmallerEqualZero:
                stride_fp  = testGen.rng.random(size=[2]) - 2
            elif error_name == ErrorIf.ShiftNotZero:
                shift = testGen.rng.integers(1, 5)
            elif error_name == ErrorIf.StrideLargerDimension:
                shape = shapeList[0]
                transform_height = testGen.rng.choice([False, True])
                if transform_height:
                    stride_fp[0] = shape[1] + testGen.rng.integers(1, 10)
                else:
                    stride_fp[1] = shape[2] + testGen.rng.integers(1, 10)
        else:
            if error_name == ErrorIf.StrideSmallerEqualZero:
                stride = np.int16(testGen.rng.integers(-1, 1, size=[2]))
            elif error_name == ErrorIf.ShiftSmallerOne:
                shift = testGen.rng.integers(-3, 1)
                if shift <= 0:
                    stride = [(16 >> -shift) - 1, (16 >> -shift) - 1] # avoids other ERROR_IF checks
                    offset = [(16 >> -shift) - 1, (16 >> -shift) - 1] # avoids other ERROR_IF checks
                else:
                    stride = [(16 << shift) - 1, (16 << shift) - 1] # avoids other ERROR_IF checks
                    offset = [(16 << shift) - 1, (16 << shift) - 1] # avoids other ERROR_IF checks
            elif error_name == ErrorIf.ShiftLargerEleven:
                shift = np.int16(testGen.rng.integers(12, 15))
            elif error_name == ErrorIf.StrideLargerDimension:
                shape = shapeList[0]
                transform_height = testGen.rng.choice([False, True])
                if transform_height:
                    stride[0] = shape[1] + testGen.rng.integers(1, 10)
                else:
                    stride[1] = shape[2] + testGen.rng.integers(1, 10)
            elif error_name == ErrorIf.StrideLargerEqualMax:
                stride = [(16 << shift) + 1, (16 << shift) + 1]
            elif error_name == ErrorIf.OffsetLargerEqualMax:
                offset = [(16 << shift) + 1, (16 << shift) + 1]
            elif error_name == ErrorIf.OffsetSmallerEqualMin:
                offset = [(-16 << shift) - 1, (-16 << shift) - 1]


        if error_name == ErrorIf.WrongOutputType:
            if mode == ResizeMode.NEAREST and dtype == DType.INT8:
                incorrect_types = (DType.INT4, DType.INT16, DType.INT32, DType.INT48, DType.FLOAT)
            elif mode == ResizeMode.NEAREST and dtype == DType.INT16:
                incorrect_types = (DType.INT4, DType.INT8, DType.INT32, DType.INT48, DType.FLOAT)
            elif mode == ResizeMode.BILINEAR and dtype == DType.INT8:
                incorrect_types = (DType.INT4, DType.INT8, DType.INT16, DType.INT48, DType.FLOAT)
            elif mode == ResizeMode.BILINEAR and dtype == DType.INT16:
                incorrect_types = (DType.INT4, DType.INT8, DType.INT16, DType.INT32, DType.FLOAT)
            elif dtype == DType.FLOAT:
                incorrect_types = (DType.INT4, DType.INT8, DType.INT16, DType.INT32, DType.INT48)
            outputDType = testGen.rng.choice(a=incorrect_types)

        return shift, stride, stride_fp, offset, offset_fp, outputDType


    @staticmethod
    def eiPoolingErrorIf(testGen, error_name, stride, pad, kernel):
        if (error_name == ErrorIf.StrideSmallerOne
            # padding must not exceed the kernel size
            and pad[0] < kernel[0] and pad[1] < kernel[0] and pad[2] < kernel[1] and pad[3] < kernel[1]):
            wrongStride = (testGen.rng.choice([0, -1, -2, -3]), testGen.rng.choice([0, -1, -2, -3]))
            return wrongStride, pad, kernel
        elif error_name == ErrorIf.PadSmallerZero:
            wrongPad = (testGen.rng.choice([-1, -2, -3]),
                        testGen.rng.choice([-1, -2, -3]),
                        testGen.rng.choice([-1, -2, -3]),
                        testGen.rng.choice([-1, -2, -3]))
            return stride, wrongPad, kernel
        elif error_name == ErrorIf.KernelSmallerOne:
            wrongKernel = (testGen.rng.choice([0, -1, -2, -3]), testGen.rng.choice([0, -1, -2, -3]))
            return stride, pad, wrongKernel
        elif error_name == ErrorIf.PadLargerEqualKernel:
            wrongPad = (testGen.rng.choice([kernel[0], kernel[0]+1, kernel[0]+2]),
                        testGen.rng.choice([kernel[0], kernel[0]+1, kernel[0]+2]),
                        testGen.rng.choice([kernel[1], kernel[1]+1, kernel[1]+2]),
                        testGen.rng.choice([kernel[1], kernel[1]+1, kernel[1]+2]))
            return stride, wrongPad, kernel
        else:
            return None, None, None


    @staticmethod
    def eiRescaleWrongOutputType(input_dtype, output_dtype):
        if input_dtype == DType.INT8:
            if output_dtype not in [DType.UINT8, DType.INT8, DType.INT16, DType.INT32]:
                return True
        if input_dtype in [DType.INT16, DType.INT32]:
            if output_dtype not in [DType.INT8, DType.INT16, DType.INT32]:
                return True
        elif input_dtype == DType.INT48:
            if output_dtype not in [DType.INT8, DType.INT16, DType.INT32]:
                return True
        elif input_dtype == DType.UINT8:
            if output_dtype != DType.INT8:
                return True
        return False


    @staticmethod
    def eiInvalidateInputOutputList(testGen, error_name, input_list, output_list):
        # Mess up input/output tensors for ERROR_IF checks
        if error_name == "WrongInputList":
            add_input = testGen.rng.choice([True, False])
            if add_input:
                input_list.append('eiDummyInput')
            else:
                input_list = input_list[:-1]
        if error_name == "WrongOutputList":
            add_output = testGen.rng.choice([True, False])
            if add_output:
                output_list.append('eiDummyOutput')
            else:
                output_list = []
        return input_list, output_list

    @staticmethod
    def eiRestrictDimensions(shape, max_dim=32, max_items=100000):
        """Restrict the dimensions and overall size of a shape to max_dim and max_items."""
        new_shape = [min(d, max_dim) for d in shape] if max(shape) > max_dim else shape
        while product(new_shape) > max_items:
            new_shape = [max(d - 1, 1) for d in new_shape]
        return new_shape

    def eiSliceErrorIf(testGen, error_name, input_shape, start, size):
        if error_name == ErrorIf.StartSmallerZero:
            newStart = []
            for i in range(len(input_shape)):
                newStart.append(testGen.rng.choice([-3, -2, -1]))
            return newStart, size
        elif error_name == ErrorIf.SizeSmallerEqualZero:
            newSize = []
            for i in range(len(input_shape)):
                newSize.append(testGen.rng.choice([-3, -2, -1, 0]))
            return start, newSize
        elif error_name == ErrorIf.StartSizeOutsideBounds:
            newStart, newSize = [], []
            for i in range(len(input_shape)):
                newStart.append(input_shape[i]-1)
                newSize.append(testGen.rng.choice([2, 3, 4]))
            return newStart, newSize
        elif error_name == ErrorIf.InputSizeStartLengthMismatch:
            remove = testGen.rng.choice([True, False])
            if remove:
                newStart = start[1:]
                newSize = size[1:]
            else:
                newStart = start
                newStart.append(1)
                newSize = size
                newSize.append(1)
            return newStart, newSize
        else:
            return start, size

    @staticmethod
    def eiCastErrorIf(testGen, input_dtype):
        if input_dtype in [DType.BOOL, DType.FLOAT]:
            outputDType = [DType.BOOL, DType.INT48, DType.FLOAT]
        elif input_dtype in [DType.INT8, DType.INT16, DType.INT32]:
            outputDType = [DType.INT48]
        else:
            assert True, f"input_dtype ({input_dtype}) not supported"
        return outputDType


class TosaErrorValidator:

    @staticmethod
    def evValidateErrorIfs(serializer, validator_fcns, error_name, **kwargs):
        # Check ERROR_IF statements

        for val_fcn in validator_fcns:
            val_result = val_fcn(True, **kwargs)

            validator_name = val_result['error_name']
            error_result = val_result['error_result']
            error_reason = val_result['error_reason']

            if error_result:
                if error_name == validator_name:
                    serializer.setExpectedReturnCode(2, error_reason)
                else:
                    print(f"Multiple ERROR_IF checks hit \nError required: {error_name}, Error_produced: {validator_name}")
                    return None # Return None to delete test if wrong ERROR_IF is hit
            else:
                if error_name == validator_name:
                    print(f"No ERROR_IF hit for {error_name}")
                    return None

    @staticmethod
    def evWrongInputType(check=False, **kwargs):
        all_dtypes = {DType.BOOL, DType.INT4, DType.INT8, DType.INT16, DType.INT32, DType.INT48, DType.FLOAT}

        # Find the unsupported input data types
        assert 'op' in kwargs
        op = kwargs['op']
        input_dtypes = op['types']

        allowed_input_dtypes = {t[0] if isinstance(t, list) else t for t in input_dtypes}
        wrong_input_dtypes = list(all_dtypes - allowed_input_dtypes)

        if op['op'] == Op.CLAMP:
            wrong_input_dtypes.remove(DType.INT48)

        error_name = ErrorIf.WrongInputType
        param_reqs = {"rank": None, "dtype": wrong_input_dtypes, "shape": None}
        error_result = False
        error_reason = "Input data type not supported for this operator"

        if check:
            input_dtype = kwargs['input_dtype']
            if op['op'] == Op.FULLY_CONNECTED:
                if input_dtype not in allowed_input_dtypes:
                    error_result = True
            elif input_dtype not in input_dtypes:
                error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs
        }
        return info_dict

    @staticmethod
    def evWrongOutputType(check=False, **kwargs):
        error_name = ErrorIf.WrongOutputType
        param_reqs = {"rank": None, "dtype": None, "shape": None}
        error_result = False
        error_reason = "Output data type not supported for this configuration of operator"

        if check:
            input_dtype = kwargs['input_dtype']
            output_dtype = kwargs['output_dtype']
            op = kwargs['op']

            if op['op'] == Op.RESIZE:
                mode = kwargs['mode']
                if (
                    (mode == ResizeMode.NEAREST and input_dtype == DType.INT8 and output_dtype != DType.INT8) or
                    (mode == ResizeMode.NEAREST and input_dtype == DType.INT16 and output_dtype != DType.INT16) or
                    (mode == ResizeMode.BILINEAR and input_dtype == DType.INT8 and output_dtype != DType.INT32) or
                    (mode == ResizeMode.BILINEAR and input_dtype == DType.INT16 and output_dtype != DType.INT48) or
                    (input_dtype == DType.FLOAT and output_dtype != DType.FLOAT)
                ):
                    error_result = True

            elif op['op'] == Op.RESCALE:
                if input_dtype == DType.INT8:
                    if output_dtype not in [DType.UINT8, DType.INT8, DType.INT16, DType.INT32]:
                        error_result = True
                if input_dtype in [DType.INT16, DType.INT32]:
                    if output_dtype not in [DType.INT8, DType.INT16, DType.INT32]:
                        error_result = True
                elif input_dtype == DType.INT48:
                    if output_dtype not in [DType.INT8, DType.INT16, DType.INT32]:
                        error_result = True
                elif input_dtype == DType.UINT8:
                    if output_dtype != DType.INT8:
                        error_result = True

            elif op['op'] in [Op.FULLY_CONNECTED, Op.MATMUL]:
                if (
                    (input_dtype == DType.INT8 and output_dtype != DType.INT32) or
                    (input_dtype == DType.INT16 and output_dtype != DType.INT48) or
                    (input_dtype == DType.FLOAT and output_dtype != DType.FLOAT)
                ):
                    error_result = True

            elif op['op'] == Op.ARGMAX:
                if input_dtype in [DType.INT8, DType.INT16, DType.FLOAT] and output_dtype != DType.INT32:
                    error_result = True

            elif op['op'] == Op.MUL:
                if input_dtype != DType.FLOAT and output_dtype != DType.INT32:
                    error_result = True
                elif input_dtype == DType.FLOAT and output_dtype != DType.FLOAT:
                    error_result = True

            elif op['op'] == Op.TABLE:
                if input_dtype == DType.INT8 and output_dtype != DType.INT8:
                    error_result = True
                elif input_dtype == DType.INT16 and output_dtype != DType.INT32:
                    error_result = True

            elif op['op'] in [Op.EQUAL, Op.GREATER_EQUAL, Op.GREATER]:
                if output_dtype != DType.BOOL:
                    error_result = True

            elif op['op'] == Op.CAST:
                if (
                    (input_dtype == DType.BOOL and output_dtype not in [DType.INT8, DType.INT16, DType.INT32])
                    or (input_dtype == DType.INT8 and output_dtype not in [DType.BOOL, DType.INT16, DType.INT32, DType.FLOAT])
                    or (input_dtype == DType.INT16 and output_dtype not in [DType.BOOL, DType.INT8, DType.INT32, DType.FLOAT])
                    or (input_dtype == DType.INT32 and output_dtype not in [DType.BOOL, DType.INT8, DType.INT16, DType.FLOAT])
                    or (input_dtype == DType.FLOAT and output_dtype not in [DType.INT8, DType.INT16, DType.INT32])
                ):
                    error_result = True

            else:
                if output_dtype != input_dtype:
                    error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs
        }
        return info_dict

    @staticmethod
    def evWrongRank(check=False, **kwargs):
        all_ranks = (1, 2, 3, 4, 5)

        # Make a list of incorrect ranks
        assert 'op' in kwargs
        op = kwargs['op']
        rmin, rmax = op['rank']
        rank_range = range(rmin, rmax + 1)
        incorrect_ranks = list(set(all_ranks) - set(rank_range))
        # Remove small incorrect ranks to avoid index errors
        incorrect_ranks = [rank for rank in incorrect_ranks if rank > rmin]
        # Set minimum incorrect rank to 3 to avoid index error
        if op['op'] in [Op.RESIZE]:
            incorrect_ranks = [3, 5]
        if op['op'] in [Op.TRANSPOSE]:
            incorrect_ranks = [7, 8]

        error_name = ErrorIf.WrongRank
        param_reqs = {"rank": incorrect_ranks, "dtype": None, "shape": None}
        error_result = False
        error_reason = "Rank not supported for this operator"

        if check:
            input_shape = kwargs['input_shape']

            if op['op'] in [Op.RESIZE, Op.AVG_POOL2D, Op.MAX_POOL2D] and len(input_shape) != 4:
                error_result = True
            elif op['op'] == Op.FULLY_CONNECTED and len(input_shape) != 2:
                error_result = True
            elif op['op'] == Op.MATMUL and len(input_shape) != 3:
                error_result = True
            else:
                if len(input_shape) not in rank_range:
                    error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs
        }
        return info_dict

    @staticmethod
    def evWrongInputList(check=False, **kwargs):
        error_name = ErrorIf.WrongInputList
        param_reqs = {"rank": None, "dtype": None, "shape": None}
        error_result = False
        error_reason = "Op input list does not match expected input"

        if check:
            op = kwargs['op']
            input_list = kwargs['input_list']
            num_operands = kwargs['num_operands']
            if op['op'] in [Op.SCATTER, Op.GATHER]:
                # SCATTER/GATHER add an indices input tensor in their build functions
                num_operands += 1
            if len(input_list) != num_operands:
                error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs
        }
        return info_dict

    @staticmethod
    def evWrongOutputList(check=False, **kwargs):
        error_name = ErrorIf.WrongOutputList
        param_reqs = {"rank": None, "dtype": None, "shape": None}
        error_result = False
        error_reason = "Op output list does not match expected output"

        if check:
            output_list = kwargs['output_list']
            # Note this will be incorrect if an operator returns more than one output
            if len(output_list) != 1:
                error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs
        }
        return info_dict

    @staticmethod
    def evMaxDimExceeded(check=False, **kwargs):
        error_name = ErrorIf.MaxDimExceeded
        param_reqs = {
            "rank": [4,4],
            "dtype": [DType.INT8],
            "shape": [[1, 16584, 5, 1], [1, 2, 16499, 4]]
            }
        error_result = False
        error_reason = "At least one maximum dimension is larger than 16384"

        if check:
            input_shape = kwargs['input_shape']
            output_shape = kwargs['output_shape'] # Note this is just (OH, OW)
            if ((input_shape[1] > 16384) or
                (input_shape[2] > 16384) or
                (output_shape[0] > 16384) or
                (output_shape[1] > 16384)):
                error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs
        }
        return info_dict

    @staticmethod
    def evBatchMismatch(check=False, **kwargs):
        error_name = ErrorIf.BatchMismatch
        param_reqs = {"rank": [4,4], "dtype": None, "shape": None}
        error_result = False
        error_reason = "Input batch size not equal to output batch size"

        assert 'op' in kwargs
        op = kwargs['op']
        rmin, rmax = op['rank']
        rank_range = range(rmin, rmax + 1)

        if check:
            input_shape = kwargs['input_shape']
            output_shape = kwargs['result_tensor'].shape # Note this is just (N, OH, OW, C)

            if (len(input_shape) in rank_range) and (input_shape[0] != output_shape[0]):
                error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs
        }
        return info_dict

    @staticmethod
    def evChannelMismatch(check=False, **kwargs):
        error_name = ErrorIf.ChannelMismatch
        param_reqs = {"rank": [4,4], "dtype": None, "shape": None}
        error_result = False
        error_reason = "Input channel size not equal to output channel size"

        assert 'op' in kwargs
        op = kwargs['op']
        rmin, rmax = op['rank']
        rank_range = range(rmin, rmax + 1)

        if check:
            input_shape = kwargs['input_shape']
            output_shape = kwargs['result_tensor'].shape # Note this is just (N, OH, OW, C)
            if (len(input_shape) in rank_range) and (input_shape[3] != output_shape[3]):
                error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs
        }
        return info_dict

    @staticmethod
    def evStrideSmallerEqualZero(check=False, **kwargs):
        error_name = ErrorIf.StrideSmallerEqualZero
        param_reqs = {"rank": None, "dtype": None, "shape": None}
        error_result = False
        error_reason = "Stride value smaller than or equal zero"

        if check:
            input_dtype = kwargs['input_dtype']
            output_dtype = kwargs['output_dtype']
            if input_dtype != DType.FLOAT and output_dtype == DType.FLOAT:
                stride = kwargs['stride'] # Work around wrong input/output type tests
            elif output_dtype == DType.FLOAT:
                stride = kwargs['stride_fp']
            elif input_dtype == DType.FLOAT and output_dtype != DType.FLOAT:
                stride = kwargs['stride_fp'] # Work around wrong input/output type tests
            else:
                stride = kwargs['stride']

            if min(stride) <= 0:
                error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs
        }
        return info_dict

    @staticmethod
    def evStrideLargerEqualMax(check=False, **kwargs):
        error_name = ErrorIf.StrideLargerEqualMax
        param_reqs = {"rank": None, "dtype": [DType.INT8, DType.INT16], "shape": None}
        error_result = False
        error_reason = "Stride value larger than or equal to maximum value"

        if check:
            shift = kwargs['shift']
            input_dtype = kwargs['input_dtype']
            stride = kwargs['stride']
            if input_dtype in [DType.INT8, DType.INT16]:
                if shift >= 0 and (stride[0] >= (16 << shift) or stride[1] >= (16 << shift)):
                    error_result = True
                elif shift < 0 and (stride[0] >= (16 >> -shift) or stride[1] >= (16 >> -shift)):
                    error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs
        }
        return info_dict


    @staticmethod
    def evStrideLargerDimension(check=False, **kwargs):
        error_name = ErrorIf.StrideLargerDimension
        param_reqs = {"rank": None, "dtype": [DType.FLOAT], "shape": None}
        error_result = False
        error_reason = "Stride value larger than or equal to H/W dimension"

        if check:
            shape = kwargs['input_shape']
            input_dtype = kwargs['input_dtype']
            stride = kwargs['stride_fp']

            if input_dtype == DType.FLOAT and (stride[0] > shape[1]) or (stride[1] > shape[2]):
                error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs
        }
        return info_dict


    @staticmethod
    def evOffsetSmallerEqualMin(check=False, **kwargs):
        error_name = ErrorIf.OffsetSmallerEqualMin
        param_reqs = {"rank": None, "dtype": [DType.INT8, DType.INT16], "shape": None}
        error_result = False
        error_reason = "Offset value smaller than or equal to minimum value"

        if check:
            shift = kwargs['shift']
            output_dtype = kwargs['output_dtype']
            if output_dtype == DType.FLOAT:
                offset = kwargs['offset_fp']
            else:
                offset = kwargs['offset']

            if shift >= 0 and (offset[0] <= (-16 << shift) or offset[1] <= (-16 << shift)):
                error_result = True
            elif shift < 0 and (offset[0] <= (-16 >> -shift) or offset[1] <= (-16 >> -shift)):
                error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs
        }
        return info_dict

    @staticmethod
    def evOffsetLargerEqualMax(check=False, **kwargs):
        error_name = ErrorIf.OffsetLargerEqualMax
        param_reqs = {"rank": None, "dtype": [DType.INT8, DType.INT16], "shape": None}
        error_result = False
        error_reason = "Offset value larger than or equal to maximum value"

        if check:
            shift = kwargs['shift']
            output_dtype = kwargs['output_dtype']
            if output_dtype == DType.FLOAT:
                offset = kwargs['offset_fp']
            else:
                offset = kwargs['offset']

            if shift >= 0:
                if offset[0] >= (16 << shift) or offset[1] >= (16 << shift):
                    error_result = True

            if shift >= 0 and (offset[0] >= (16 << shift) or offset[1] >= (16 << shift)):
                error_result = True
            elif shift < 0 and (offset[0] >= (16 >> -shift) or offset[1] >= (16 >> -shift)):
                error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs
        }
        return info_dict

    @staticmethod
    def evShiftNotZero(check=False, **kwargs):
        error_name = ErrorIf.ShiftNotZero
        param_reqs = {"rank": None, "dtype": [DType.FLOAT], "shape": None}
        error_result = False
        error_reason = "Shift value must be zero for float input"

        if check:
            shift = kwargs['shift']
            input_dtype = kwargs['input_dtype']
            output_dtype = kwargs['output_dtype']
            if input_dtype == DType.FLOAT and output_dtype == DType.FLOAT and shift != 0:
                error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs
        }
        return info_dict


    @staticmethod
    def evShiftSmallerOne(check=False, **kwargs):
        error_name = ErrorIf.ShiftSmallerOne
        param_reqs = {"rank": None, "dtype": [DType.INT8, DType.INT16], "shape": None}
        error_result = False
        error_reason = "Shift value smaller than one"

        if check:
            shift = kwargs['shift']
            input_dtype = kwargs['input_dtype']
            output_dtype = kwargs['output_dtype']
            if shift < 1 and input_dtype != DType.FLOAT and output_dtype != DType.FLOAT:
                error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs
        }
        return info_dict

    @staticmethod
    def evShiftLargerEleven(check=False, **kwargs):
        error_name = ErrorIf.ShiftLargerEleven
        param_reqs = {"rank": None, "dtype": [DType.INT8, DType.INT16], "shape": None}
        error_result = False
        error_reason = "Shift value larger than eleven"

        if check:
            shift = kwargs['shift']
            if shift > 11:
                error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs
        }
        return info_dict


    @staticmethod
    def evRankMismatch(check=False, **kwargs):
        error_name = ErrorIf.RankMismatch
        param_reqs = {"rank": None, "dtype": None, "shape": None}
        error_result = False
        error_reason = "Input Rank does not match output rank"

        if check:
            input1_shape = kwargs['input1'].shape
            input2_shape = kwargs['input2'].shape
            output_shape = kwargs['result_tensor'].shape
            if (len(input1_shape) != len(output_shape)) or (len(input2_shape) != len(output_shape)):
                error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs
        }
        return info_dict

    @staticmethod
    def evInputZeroPointNotZero(check=False, **kwargs):
        op = kwargs['op']
        inputDtypes = op['types'].copy()
        # If inputDtypes is a list then only the first two elements are INT8 inputs
        if isinstance(inputDtypes, list):
            inputDtypes = inputDtypes[2:]

        if DType.INT8 in inputDtypes:
            inputDtypes.remove(DType.INT8)
        if DType.UINT8 in inputDtypes:
            inputDtypes.remove(DType.UINT8)

        error_name = ErrorIf.InputZeroPointNotZero
        param_reqs = {
            "rank": None,
            "dtype": inputDtypes,
            "shape": None
            }
        error_result = False
        error_reason = "Input DType not INT8 and zero point not 0"

        if check:
            input_dtype = kwargs['input_dtype']
            if isinstance(kwargs['qinfo'], tuple):
                qinfo = kwargs['qinfo']
                input_zero_point = qinfo[0]
            else:
                # For use: qinfo.ints[0][1] = input_zp, qinfo.ints[1][1] = output_zp
                qinfo = kwargs['qinfo'].ints
                input_zero_point = qinfo[0][1]

            if op['op'] == Op.MATMUL:
                input1_dtype = kwargs['input_dtype']
                input2_dtype = kwargs['input2_dtype']
                qinfo = kwargs['qinfo'].ints
                input1_zero_point = qinfo[0][1]
                input2_zero_point = qinfo[1][1]
                if (input1_dtype != DType.INT8 and input1_zero_point != 0) or (input2_dtype != DType.INT8 and input2_zero_point != 0):
                    error_result = True
            else:
                if input_dtype not in [DType.INT8, DType.UINT8] and input_zero_point != 0:
                    error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs
        }
        return info_dict


    @staticmethod
    def evWeightZeroPointNotZero(check=False, **kwargs):
        op = kwargs['op']

        # exclude inputs with INT8 weights
        inputDtypes = [t for t in op['types']
                       if not isinstance(t, list) or t[1] != DType.INT8]

        error_name = ErrorIf.WeightZeroPointNotZero
        param_reqs = {
            "rank": None,
            "dtype": inputDtypes,
            "shape": None
            }
        error_result = False
        error_reason = "Weight DType not INT8 and zero point not 0"

        if check:
            weight_dtype = kwargs['weight_dtype']
            # For use: qinfo.ints[0][1] = input_zp, qinfo.ints[1][1] = weight_zp
            qinfo = kwargs['qinfo'].ints
            weight_zero_point = qinfo[1][1]
            if weight_dtype != DType.INT8 and weight_zero_point != 0:
                error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs
        }
        return info_dict


    @staticmethod
    def evOutputZeroPointNotZero(check=False, **kwargs):
        op = kwargs['op']
        inputDtypes = op['types'].copy()
        if DType.INT8 in inputDtypes:
            inputDtypes.remove(DType.INT8)
        if DType.UINT8 in inputDtypes:
            inputDtypes.remove(DType.UINT8)

        error_name = ErrorIf.OutputZeroPointNotZero
        param_reqs = {
            "rank": None,
            "dtype": inputDtypes,
            "shape": None
            }
        error_result = False
        error_reason = "Output DType not INT8 and zero point not 0"

        if check:
            input_dtype = kwargs['input_dtype']
            output_dtype = kwargs['output_dtype']
            if isinstance(kwargs['qinfo'], tuple):
                qinfo = kwargs['qinfo']
                output_zero_point = qinfo[1]
            else:
                # For use: qinfo.ints[0][1] = input_zp, qinfo.ints[1][1] = output_zp
                qinfo = kwargs['qinfo'].ints
                output_zero_point = qinfo[1][1]
            if op['op'] == Op.AVG_POOL2D:
                if input_dtype != DType.INT8 and output_zero_point != 0:
                    error_result = True
            elif output_dtype not in [DType.INT8, DType.UINT8] and output_zero_point != 0:
                error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs
        }
        return info_dict

    @staticmethod
    def evAxisSmallerZero(check=False, **kwargs):
        error_name = ErrorIf.AxisSmallerZero
        param_reqs = {"rank": None, "dtype": None, "shape": None}
        error_result = False
        error_reason = "Axis smaller than zero"

        if check:
            axis = kwargs['axis']
            if axis < 0:
                error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs
        }
        return info_dict


    @staticmethod
    def evAxisLargerRank(check=False, **kwargs):
        error_name = ErrorIf.AxisLargerRank
        param_reqs = {"rank": None, "dtype": None, "shape": None}
        error_result = False
        error_reason = "Axis larger than rank"

        if check:
            axis = kwargs['axis']
            shape = kwargs['input_shape']
            if axis > len(shape):
                error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs
        }
        return info_dict


    @staticmethod
    def evShapeOfAxisNotOne(check=False, **kwargs):
        error_name = ErrorIf.ShapeOfAxisNotOne
        param_reqs = {"rank": None, "dtype": None, "shape": None}
        error_result = False
        error_reason = "shape[axis] is not equal to 1"

        if check:
            axis = kwargs['axis']
            shape = kwargs['output_shape']
            if (0 <= axis < len(shape)) and shape[axis] != 1:
                error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs
        }
        return info_dict


    @staticmethod
    def evPadSmallerZero(check=False, **kwargs):
        error_name = ErrorIf.PadSmallerZero
        param_reqs = {"rank": None, "dtype": None, "shape": None}
        error_result = False
        error_reason = "At least one pad is smaller than zero"

        if check:
            op = kwargs['op']
            pad = kwargs['pad']
            if op['op'] == Op.PAD:
                for padding in pad:
                    if min(padding) < 0:
                        error_result = True
            else:
                if min(pad) < 0:
                    error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs
        }
        return info_dict


    @staticmethod
    def evPadLargerEqualKernel(check=False, **kwargs):
        error_name = ErrorIf.PadLargerEqualKernel
        param_reqs = {"rank": None, "dtype": None, "shape": None}
        error_result = False
        error_reason = "At least one pad is larger than kernel dimension"

        if check:
            pad = kwargs['pad']
            kernel = kwargs['kernel']
            if min(pad) > 0 and min(kernel) > 1:
                if pad[0] >= kernel[0] or pad[1] >= kernel[0] or pad[2] >= kernel[1] or pad[3] >= kernel[1]:
                    error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs
        }
        return info_dict

    @staticmethod
    def evPoolingOutputShapeMismatch(check=False, **kwargs):
        error_name = ErrorIf.PoolingOutputShapeMismatch
        param_reqs = {"rank": None, "dtype": None, "shape": None}
        error_result = False
        error_reason = "Mismatch between output shape provided and expected output shape"

        if check:
            pad = kwargs['pad']
            pad_top, pad_bottom, pad_left, pad_right = pad[0], pad[1], pad[2], pad[3]

            kernel = kwargs['kernel']
            kernel_y, kernel_x = kernel[0], kernel[1]

            input_shape = kwargs['input_shape']
            IH, IW = input_shape[1], input_shape[2]

            output_shape = kwargs['output_shape']
            OH, OW = output_shape[1], output_shape[2]

            stride = kwargs['stride']
            stride_y, stride_x = stride[0], stride[1]

            # calculate correct height, width dimensions
            if stride_x != 0 and stride_y != 0:
                y_correct = (IH + pad_top + pad_bottom + stride_y - kernel_y) // stride_y
                x_correct = (IW + pad_left + pad_right + stride_x - kernel_x) // stride_x

            # ensure parameters are valid
            params_valid = (min(kernel) >= 1 and min(stride) >= 1 and min(pad) >= 0
                and not (pad[0] >= kernel[0] or pad[1] >= kernel[0] or pad[2] >= kernel[1] or pad[3] >= kernel[1]))

            if params_valid and (OH != y_correct or OW != x_correct):
                error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs
        }
        return info_dict

    @staticmethod
    def evArgmaxOutputShapeMismatch(check=False, **kwargs):
        error_name = ErrorIf.ArgmaxOutputShapeMismatch
        param_reqs = {"rank": [2,4], "dtype": None, "shape": None}
        error_result = False
        error_reason = "Mismatch between output shape provided and expected output shape"

        if check:
            output_shape = kwargs['output_shape']
            input_shape = kwargs['input_shape']
            axis = kwargs['axis']

            dimension_match = True
            axis_shift = 0

            # Check that rank is correct before trying to check dimensions
            if (len(input_shape) - 1) == len(output_shape):
                for i in range(len(input_shape)):
                    if i == axis:
                        axis_shift = 1
                        continue
                    if input_shape[i] != output_shape[i - axis_shift]:
                        dimension_match = False

                if not dimension_match:
                    error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs
        }
        return info_dict

    @staticmethod
    def evArgmaxOutputRankMismatch(check=False, **kwargs):
        error_name = ErrorIf.ArgmaxOutputRankMismatch
        param_reqs = {"rank": None, "dtype": None, "shape": None}
        error_result = False
        error_reason = "Mismatch between output shape provided and expected output shape"

        if check:
            output_shape = kwargs['output_shape']
            input_shape = kwargs['input_shape']
            axis = kwargs['axis']
            valid_params = axis >= 0 and axis < len(input_shape)

            if valid_params and (len(input_shape) - 1) != len(output_shape):
                error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs
        }
        return info_dict


    @staticmethod
    def evKernelSmallerOne(check=False, **kwargs):
        error_name = ErrorIf.KernelSmallerOne
        param_reqs = {"rank": None, "dtype": None, "shape": None}
        error_result = False
        error_reason = "At least one kernel dimension is smaller than zero"

        if check:
            kernel = kwargs['kernel']
            if min(kernel) < 1:
                error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs
        }
        return info_dict

    @staticmethod
    def evStrideSmallerOne(check=False, **kwargs):
        error_name = ErrorIf.StrideSmallerOne
        param_reqs = {"rank": None, "dtype": None, "shape": None}
        error_result = False
        error_reason = "At least one stride dimension is smaller than zero"

        if check:
            stride = kwargs['stride']
            if min(stride) < 1:
                error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs
        }
        return info_dict

    @staticmethod
    def evScaleTrue(check=False, **kwargs):
        error_name = ErrorIf.ScaleTrue
        param_reqs = {"rank": None, "dtype": [DType.INT48], "shape": None}
        error_result = False
        error_reason = "Scale set to true but input type is INT48"

        if check:
            input_dtype = kwargs['input_dtype']
            scale32 = kwargs['scale32']
            if scale32 and input_dtype == DType.INT48:
                error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs
        }
        return info_dict

    @staticmethod
    def evScaleNotTrue(check=False, **kwargs):
        error_name = ErrorIf.ScaleNotTrue
        param_reqs = {"rank": None, "dtype": None, "shape": None}
        error_result = False
        error_reason = "Scale set to false but double round set to true"

        if check:
            scale32 = kwargs['scale32']
            double_round = kwargs['double_round']
            if not scale32 and double_round:
                error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs
        }
        return info_dict

    @staticmethod
    def evTensorSizeInputOutputMismatch(check=False, **kwargs):
        error_name = ErrorIf.TensorSizeInputOutputMismatch
        param_reqs = {"rank": None, "dtype": None, "shape": None}
        error_result = False
        error_reason = "Input tensor size does not match output tensor size"

        if check:
            input_shape = kwargs['input_shape']
            output_shape = kwargs['output_shape']
            input_size = np.prod(input_shape)
            output_size = np.prod(output_shape)
            if input_size != output_size:
                error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs
        }
        return info_dict

    @staticmethod
    def evStartSmallerZero(check=False, **kwargs):
        error_name = ErrorIf.StartSmallerZero
        param_reqs = {"rank": None, "dtype": None, "shape": None}
        error_result = False
        error_reason = "Starting point smaller than zero"

        if check:
            input_shape = kwargs['input_shape']
            start = kwargs['start']
            rank = len(input_shape)
            if len(start) == rank:
                for index in range(rank):
                    if start[index] < 0:
                        error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs
        }
        return info_dict


    @staticmethod
    def evSizeSmallerEqualZero(check=False, **kwargs):
        error_name = ErrorIf.SizeSmallerEqualZero
        param_reqs = {"rank": None, "dtype": None, "shape": None}
        error_result = False
        error_reason = "Size smaller than or equal to zero"

        if check:
            input_shape = kwargs['input_shape']
            size = kwargs['size']
            rank = len(input_shape)
            if len(size) == rank:
                for index in range(rank):
                    if size[index] <= 0:
                        error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs
        }
        return info_dict


    @staticmethod
    def evStartSizeOutsideBounds(check=False, **kwargs):
        error_name = ErrorIf.StartSizeOutsideBounds
        param_reqs = {"rank": None, "dtype": None, "shape": None}
        error_result = False
        error_reason = "starting point plus size larger than input dimension"

        if check:
            input_shape = kwargs['input_shape']
            start = kwargs['start']
            size = kwargs['size']
            rank = len(input_shape)
            if len(start) == rank and len(size) == rank:
                for index in range(rank):
                    if start[index] + size[index] > input_shape[index]:
                        error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs
        }
        return info_dict


    @staticmethod
    def evSizeOutputShapeMismatch(check=False, **kwargs):
        error_name = ErrorIf.SizeOutputShapeMismatch
        param_reqs = {"rank": None, "dtype": None, "shape": None}
        error_result = False
        error_reason = "Size does not match output dimension"

        if check:
            input_shape = kwargs['input_shape']
            output_shape = kwargs['output_shape']
            size = kwargs['size']
            rank = len(input_shape)
            if len(size) == rank:
                for index in range(rank):
                    if size[index] != output_shape[index]:
                        error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs
        }
        return info_dict

    @staticmethod
    def evInputSizeStartLengthMismatch(check=False, **kwargs):
        error_name = ErrorIf.InputSizeStartLengthMismatch
        param_reqs = {"rank": None, "dtype": None, "shape": None}
        error_result = False
        error_reason = "rank of input not equal to length of start or size"

        if check:
            input_shape = kwargs['input_shape']
            start = kwargs['start']
            size = kwargs['size']
            rank = len(input_shape)
            if rank != len(start) or rank != len(size):
                error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs
        }
        return info_dict

    @staticmethod
    def evIndexOutsideBounds(check=False, **kwargs):
        error_name = ErrorIf.IndexOutsideBounds
        param_reqs = {"rank": None, "dtype": None, "shape": None}
        error_result = False
        error_reason = "Index outside of allowed bounds"

        if check:
            input_shape = kwargs['input_shape']
            perms = kwargs['perms']
            rank = len(input_shape)

            for index in perms:
                if index < 0 or index > rank:
                    error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs
        }
        return info_dict

    @staticmethod
    def evIndexUsedTwice(check=False, **kwargs):
        error_name = ErrorIf.IndexUsedTwice
        param_reqs = {"rank": [2,4], "dtype": None, "shape": None}
        error_result = False
        error_reason = "Index used multiple times"

        if check:
            input_shape = kwargs['input_shape']
            perms = kwargs['perms']
            rank = len(input_shape)

            unique_indices = []
            for index in perms:
                if index in unique_indices:
                    error_result = True
                else:
                    unique_indices.append(index)

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs
        }
        return info_dict

    @staticmethod
    def evMaxSmallerMin(check=False, **kwargs):
        error_name = ErrorIf.MaxSmallerMin
        param_reqs = {"rank": [2,4], "dtype": None, "shape": None}
        error_result = False
        error_reason = "Max value smaller than min value"

        if check:
            max_val = kwargs['max_val']
            min_val = kwargs['min_val']
            if max_val < min_val:
                error_result = True


        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs
        }
        return info_dict

    @staticmethod
    def evConcatInputRankMismatch(check=False, **kwargs):
        error_name = ErrorIf.ConcatInputRankMismatch
        param_reqs = {"rank": [2,4], "dtype": None, "shape": None}
        error_result = False
        error_reason = "Input ranks are not identical"

        if check:
            inputs = kwargs['inputs']
            input_shape = kwargs['input_shape']
            for input in inputs:
                if len(input.shape) != len(input_shape):
                    error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs
        }
        return info_dict

    @staticmethod
    def evConcatInputDimMismatch(check=False, **kwargs):
        error_name = ErrorIf.ConcatInputDimMismatch
        param_reqs = {"rank": [2,4], "dtype": None, "shape": None}
        error_result = False
        error_reason = "Input dimensions differ on too many axes"

        if check:
            inputs = kwargs['inputs']
            input_shape = kwargs['input_shape']
            axis = kwargs['axis']

            # Ensure rank is valid before checking dims.
            valid_rank = True
            for input in inputs:
                if len(input.shape) != len(input_shape):
                    valid_rank = False

            if valid_rank:
                for input in inputs:
                    for i, dim in enumerate(input.shape):
                        if dim != input_shape[i] and axis != i:
                            error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs
        }
        return info_dict

    @staticmethod
    def evConcatShapeSumMismatch(check=False, **kwargs):
        error_name = ErrorIf.ConcatShapeSumMismatch
        param_reqs = {"rank": [2,4], "dtype": None, "shape": None}
        error_result = False
        error_reason = "Sum of dimensions on axis not equal to output dimension"

        if check:
            inputs = kwargs['inputs']
            input_shape = kwargs['input_shape']
            output_shape = kwargs['output_shape']
            axis = kwargs['axis']

            # Ensure rank is valid before checking dims.
            valid_params = True
            for input in inputs:
                if len(input.shape) != len(input_shape):
                    valid_params = False
            if axis < 0 or axis > len(input_shape):
                valid_params = False

            if valid_params:
                axis_dim_sum = 0
                for input in inputs:
                    axis_dim_sum += input.shape[axis]

                if axis_dim_sum != output_shape[axis]:
                    error_result = True


        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs
        }
        return info_dict

    @staticmethod
    def evInputListThenGraphMismatch(check=False, **kwargs):
        error_name = ErrorIf.CondIfInputListThenGraphMismatch
        param_reqs = {"rank": None, "dtype": None, "shape": None}
        error_result = False
        error_reason = "Input list shape does not match then-graph shape"

        if check:
            a = kwargs['a']
            b = kwargs['b']
            basicBlocks = kwargs['basicBlocks']
            then_block = basicBlocks[1]
            then_inputs = then_block.inputs
            then_tens = then_block.tensors
            if (a.shape != then_tens[then_inputs[0]].shape) or (b.shape != then_tens[then_inputs[1]].shape):
                error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs
        }
        return info_dict


    @staticmethod
    def evInputListElseGraphMismatch(check=False, **kwargs):
        error_name = ErrorIf.CondIfInputListElseGraphMismatch
        param_reqs = {"rank": None, "dtype": None, "shape": None}
        error_result = False
        error_reason = "Input list shape does not match else-graph shape"

        if check:
            a = kwargs['a']
            b = kwargs['b']
            basicBlocks = kwargs['basicBlocks']
            else_block = basicBlocks[2]
            else_inputs = else_block.inputs
            else_tens = else_block.tensors
            if (a.shape != else_tens[else_inputs[0]].shape) or (b.shape != else_tens[else_inputs[1]].shape):
                error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs
        }
        return info_dict


    @staticmethod
    def evOutputListThenGraphMismatch(check=False, **kwargs):
        error_name = ErrorIf.CondIfOutputListThenGraphMismatch
        param_reqs = {"rank": None, "dtype": None, "shape": None}
        error_result = False
        error_reason = "Output list shape does not match then-graph shape"

        if check:
            basicBlocks = kwargs['basicBlocks']
            cond_block = basicBlocks[0]
            cond_outputs = cond_block.outputs
            cond_tens = cond_block.tensors
            then_block = basicBlocks[1]
            then_outputs = then_block.outputs
            then_tens = then_block.tensors
            if then_tens[then_outputs[0]].shape != cond_tens[cond_outputs[0]].shape:
                error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs
        }
        return info_dict


    @staticmethod
    def evOutputListElseGraphMismatch(check=False, **kwargs):
        error_name = ErrorIf.CondIfOutputListElseGraphMismatch
        param_reqs = {"rank": None, "dtype": None, "shape": None}
        error_result = False
        error_reason = "Output list shape does not match else-graph shape"

        if check:
            basicBlocks = kwargs['basicBlocks']
            cond_block = basicBlocks[0]
            cond_outputs = cond_block.outputs
            cond_tens = cond_block.tensors
            else_block = basicBlocks[2]
            else_outputs = else_block.outputs
            else_tens = else_block.tensors
            if else_tens[else_outputs[0]].shape != cond_tens[cond_outputs[0]].shape:
                error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs
        }
        return info_dict


    @staticmethod
    def evInputListOutputListMismatch(check=False, **kwargs):
        error_name = ErrorIf.InputListOutputListMismatch
        param_reqs = {"rank": None, "dtype": None, "shape": None}
        error_result = False
        error_reason = "Input list does not match output list"

        if check:
            basicBlocks = kwargs['basicBlocks']
            while_block = basicBlocks[0]
            while_inputs = while_block.inputs
            while_outputs = while_block.outputs
            while_tens = while_block.tensors
            if while_tens[while_inputs[1]].shape != while_tens[while_outputs[0]].shape:
                error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs
        }
        return info_dict


    @staticmethod
    def evInputListCondGraphMismatch(check=False, **kwargs):
        error_name = ErrorIf.InputListCondGraphMismatch
        param_reqs = {"rank": None, "dtype": None, "shape": None}
        error_result = False
        error_reason = "Input list does not match cond graph"

        if check:
            basicBlocks = kwargs['basicBlocks']
            while_block = basicBlocks[0]
            while_inputs = while_block.inputs
            while_tens = while_block.tensors
            cond_block = basicBlocks[1]
            cond_inputs = cond_block.inputs
            cond_tens = cond_block.tensors
            if ((while_tens[while_inputs[0]].shape != cond_tens[cond_inputs[0]].shape) or
                (while_tens[while_inputs[1]].shape != cond_tens[cond_inputs[2]].shape)):
                error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs
        }
        return info_dict


    @staticmethod
    def evInputListBodyGraphInputMismatch(check=False, **kwargs):
        error_name = ErrorIf.InputListBodyGraphInputMismatch
        param_reqs = {"rank": None, "dtype": None, "shape": None}
        error_result = False
        error_reason = "Input list does not match body graph input"

        if check:
            basicBlocks = kwargs['basicBlocks']
            while_block = basicBlocks[0]
            while_inputs = while_block.inputs
            while_tens = while_block.tensors
            body_block = basicBlocks[2]
            body_outputs = body_block.inputs
            body_tens = body_block.tensors
            if ((while_tens[while_inputs[0]].shape != body_tens[body_outputs[0]].shape) or
                (while_tens[while_inputs[1]].shape != body_tens[body_outputs[2]].shape)):
                error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs
        }
        return info_dict


    @staticmethod
    def evInputListBodyGraphOutputMismatch(check=False, **kwargs):
        error_name = ErrorIf.InputListBodyGraphOutputMismatch
        param_reqs = {"rank": None, "dtype": None, "shape": None}
        error_result = False
        error_reason = "Input list does not match body graph output"

        if check:
            basicBlocks = kwargs['basicBlocks']
            while_block = basicBlocks[0]
            while_inputs = while_block.inputs
            while_tens = while_block.tensors
            body_block = basicBlocks[2]
            body_outputs = body_block.outputs
            body_tens = body_block.tensors
            if ((while_tens[while_inputs[0]].shape != body_tens[body_outputs[0]].shape) or
                (while_tens[while_inputs[1]].shape != body_tens[body_outputs[2]].shape)):
                error_result = True
        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs
        }
        return info_dict


    @staticmethod
    def evCondGraphOutputNotMatchingBool(check=False, **kwargs):
        error_name = ErrorIf.CondGraphOutputNotMatchingBool
        param_reqs = {"rank": None, "dtype": None, "shape": None}
        error_result = False
        error_reason = "Cond graph output is not a match list of booleans"

        if check:
            basicBlocks = kwargs['basicBlocks']
            cond_block = basicBlocks[1]
            cond_outputs = cond_block.outputs
            cond_tens = cond_block.tensors
            if cond_tens[cond_outputs[0]].dtype != DType.BOOL:
                error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs
        }
        return info_dict


class TosaInvalidValidator:

    @staticmethod
    def ivWrongDataTypeOrModeResize(**kwargs):
        input_dtype = kwargs["input_dtype"]
        args = kwargs["args"]
        mode = args[0]
        stride = args[1]
        stride_fp = args[4]
        output_dtype = args[8]

        if mode == ResizeMode.BILINEAR:
            # Invalid output data type / Invalid input datatype
            return (
                not (input_dtype == DType.INT8 and output_dtype == DType.INT32) or
                not (input_dtype == DType.INT16 and output_dtype == DType.INT48) or
                not (input_dtype == DType.FLOAT and output_dtype == DType.FLOAT) or
                (input_dtype not in [DType.INT8, DType.INT32, DType.FLOAT])
            )
        elif mode == ResizeMode.NEAREST:
            # Invalid output data type / Invalid input datatype
            return (
                (input_dtype != output_dtype) or
                (input_dtype not in [DType.INT8, DType.INT32, DType.FLOAT])
            )
        else:
            # Invalid resize mode
            return True

    @staticmethod
    def ivBadStride(**kwargs):
        input_dtype = kwargs["input_dtype"]
        args = kwargs["args"]
        stride_x = args[1][0]
        stride_y = args[1][1]
        stride_fp_x = args[4][0]
        stride_fp_y = args[4][1]

        if input_dtype == DType.FLOAT:
            if stride_fp_x <= 0 or stride_fp_y <= 0:
                # Negative or zero stride
                return True
        else:
            if stride_x <= 0 or stride_y <= 0:
                # Negative or zero stride
                return True
        return False


    @staticmethod
    def ivHeightWidthSmallerZero(**kwargs):
        opName = kwargs['opName']

        inputShapes = kwargs['shapeList']
        input = inputShapes[0]
        if not opName.endswith("pool2d"):
            filter = inputShapes[1]

        args = kwargs['args']
        strides = args[0]
        padding = args[1]
        dilations = args[2]
        if opName.endswith("pool2d"):
            kernel = args[2]

        if opName.startswith('conv2d'):
            h = (
                input[1]
                - filter[1]
                - (filter[1] - 1) * (dilations[0] - 1)
                + padding[0]
                + padding[1]
            ) // strides[0] + 1

            w = (
                input[2]
                - filter[2]
                - (filter[2] - 1) * (dilations[1] - 1)
                + padding[2]
                + padding[3]
            ) // strides[1] + 1
        elif opName.startswith("depthwise_conv2d"):
            h = (
                input[1]
                - filter[0]
                - (filter[0] - 1) * (dilations[0] - 1)
                + padding[0]
                + padding[1]
            ) // strides[0] + 1

            w = (
                input[2]
                - filter[1]
                - (filter[1] - 1) * (dilations[1] - 1)
                + padding[2]
                + padding[3]
            ) // strides[1] + 1
        elif opName.endswith("pool2d"):
            h = (input[1] + padding[0] + padding[1] + strides[0] - kernel[0]) // strides[0]
            w = (input[2] + padding[2] + padding[3] + strides[1] - kernel[1]) // strides[1]
        else:
            assert False, "Unrecognized Op"

        if h <= 0 or w <= 0:
            # Invalid parameter combination
            return True
        return False

    @staticmethod
    def ivNonPositiveOutputShape(**kwargs):
        args = kwargs['args']
        output_shape = args[3]
        if output_shape[1] <= 0 or output_shape[2] <= 0:
            # Negative output shape
            return True
        return False



class TosaTestGen:
    # Maximum rank of tensor supported by test generator.
    TOSA_TENSOR_MAX_RANK = 6

    def __init__(self, args):
        self.args = args
        self.basePath = args.output_dir
        self.random_seed = args.random_seed
        self.ser = None
        self.rng = np.random.default_rng(self.random_seed)
        self.createDynamicOpLists()
        self.initOpListDefaults()
        self.quantGen = TosaQuantGen()
        # Force makeShape to do a specific starting shape
        self.targetted_shape = None

    def createSerializer(self, opName, testPath):
        self.testPath = os.path.join(opName, testPath)

        fullPath = os.path.join(self.basePath, self.testPath)
        os.makedirs(fullPath, exist_ok=True)
        self.ser = ts.TosaSerializer(fullPath)

    def getSerializer(self):
        return self.ser

    def serialize(self, testName):
        with open(
            os.path.join(self.basePath, self.testPath, "{}.tosa".format(testName)), "wb"
        ) as fd:
            fd.write(self.ser.serialize())

        with open(os.path.join(self.basePath, self.testPath, "desc.json"), "w") as fd:
            fd.write(self.ser.writeJson("{}.tosa".format(testName)))

    def resetRNG(self, seed=None):
        if seed == None:
            seed = self.random_seed + 1
        self.rng = np.random.default_rng(seed)

    def getRandTensor(self, shape, dtype):
        if dtype == DType.BOOL:
            np_dt = np.bool
            return np.bool_(self.rng.choice(a=[False, True], size=shape))
        # TOSA specific INT4 weight range from -7 to 7
        elif dtype == DType.INT4:
            return np.int32(self.rng.integers(low=-7, high=8, size=shape))
        elif dtype == DType.INT8:
            return np.int32(self.rng.integers(low=-128, high=128, size=shape))
        elif dtype == DType.UINT8:
            return np.int32(self.rng.integers(low=0, high=256, size=shape))
        elif dtype == DType.INT16:
            return np.int32(self.rng.integers(low=-32768, high=32768, size=shape))
        elif dtype == DType.INT32:
            return np.int32(
                self.rng.integers(low=-(1 << 31), high=(1 << 31), size=shape)
            )
        elif dtype == DType.INT48:
            return np.int64(
                self.rng.integers(low=-(1 << 47), high=(1 << 47), size=shape)
            )
        elif dtype == DType.FLOAT:
            return np.float32(self.rng.random(size=shape))
        else:
            raise Exception("Unrecognized Dtype: {}".format(dtype))

    def buildPlaceholderTensors(self, shape_list, dtype_list):
        placeholders = []

        assert len(shape_list) == len(dtype_list)

        for idx, shape in enumerate(shape_list):
            arr = self.getRandTensor(shape, dtype_list[idx])
            placeholders.append(self.ser.addPlaceholder(shape, dtype_list[idx], arr))

        return placeholders

    def buildConstTensors(self, shape_list, dtype_list):
        consts = []

        assert len(shape_list) == len(dtype_list)

        for idx, shape in enumerate(shape_list):
            arr = self.getRandTensor(shape, dtype_list[idx])
            consts.append(self.ser.addConst(shape, dtype_list[idx], arr))

        return consts

    def makeShape(self, rank):
        if self.targetted_shape:
            return np.int32(self.targetted_shape)
        return np.int32(
            self.rng.integers(
                low=self.args.tensor_shape_range[0],
                high=self.args.tensor_shape_range[1],
                size=rank,
            )
        )

    def setTargetShape(self, shape):
        self.targetted_shape = shape

    def randInt(self, low=0, high=256):
        return np.int32(self.rng.integers(low=low, high=high, size=1))[0]

    def getRandNumberDType(self, dtype):
        if dtype == DType.FLOAT:
            return self.rng.random()
        elif dtype == DType.BOOL:
            return self.rng.choice([False, True])
        # TOSA specific INT4 weight range from -7 to 7
        elif dtype == DType.INT4:
            low, high = (-7, 8)
        elif dtype == DType.INT8:
            low, high = (-128, 128)
        elif dtype == DType.INT16:
            low, high = (-32768, 32768)
        elif dtype == DType.INT32:
            low, high = (-(1 << 31), (1 << 31))
        elif dtype == DType.INT48:
            low, high = (-(1 << 47), (1 << 47))
            # Special size
            return np.int64(self.rng.integers(low, high, size=1))[0]
        else:
            raise Exception("Unknown dtype: {}".format(dtype))

        return np.int32(self.rng.integers(low, high, size=1))[0]

    def shapeStr(self, shape):

        sStr = []
        # Convert to strings
        for i in shape:
            sStr.append(str(i))

        return "x".join(sStr)

    def typeStr(self, t):
        if isinstance(t, list):
            assert len(t) >= 2
            return "{}x{}".format(self.typeStr(t[0]), self.typeStr(t[1]))
        else:
            if t == DType.BOOL:
                return "b"
            elif t == DType.INT4:
                return "i4"
            elif t == DType.INT8:
                return "i8"
            elif t == DType.UINT8:
                return "u8"
            elif t == DType.INT16:
                return "i16"
            elif t == DType.INT32:
                return "i32"
            elif t == DType.INT48:
                return "i48"
            elif t == DType.FLOAT:
                return "float"
            else:
                raise Exception("Unknown dtype, cannot convert to string: {}".format(t))

    def typeWidth(self, t):
        """ Get the datatype width for integer types"""
        if t == DType.INT4:
            return 4
        elif t == DType.INT8:
            return 8
        elif t == DType.UINT8:
            return 8
        elif t == DType.INT16:
            return 16
        elif t == DType.INT32:
            return 32
        elif t == DType.INT48:
            return 48
        elif t == DType.FLOAT:
            return 32
        elif t == DType.BOOL:
            return 1
        else:
            raise Exception("Unknown dtype, cannot convert to string: {}".format(t))

    # Argument generators
    # Returns a list of tuples (stringDescriptor, [build_fcn_arg_list])
    # Where the string descriptor is used to generate the test name and
    # The build_fcn_arg_list is expanded and passed to the operator test
    # build function

    def build_unary(self, op, a, validator_fcns=None, error_name=None, qinfo=None):
        result_tens = OutputShaper.unaryOp(self.ser, self.rng, a, error_name)

        # build_placeholder returns an int, ABS/other ops does not
        if isinstance(op, int):
            self.ser.addOperator(op, a.name, result_tens.name, None, qinfo)
            return result_tens
        elif op['op'] == Op.IDENTITY:
            self.ser.addOperator(op['op'], a.name, result_tens.name, None, qinfo)
            return result_tens

        # Ensure new output type has correct qinfo
        if error_name == ErrorIf.WrongOutputType:
            if result_tens.dtype not in [DType.INT8, DType.UINT8]:
                qinfo = ts.TosaSerializerQuantInfo()
                qinfo.UnaryQuantInfo(
                TosaQuantGen.getQinfo(self, a.dtype), TosaQuantGen.getQinfo(self, result_tens.dtype)
                )

        # Invalidate Input/Output list for error if checks.
        input_list = [a.name]
        output_list = [result_tens.name]
        pCount, cCount = op["operands"]
        num_operands = pCount + cCount
        input_list, output_list = TosaErrorIfArgGen.eiInvalidateInputOutputList(self, error_name, input_list, output_list)

        TosaErrorValidator.evValidateErrorIfs(
            self.ser,
            validator_fcns,
            error_name,
            op=op,
            input_dtype=a.dtype,
            output_dtype=result_tens.dtype,
            qinfo = qinfo,
            result_tensor = result_tens,
            input_list=input_list,
            output_list=output_list,
            num_operands=num_operands,
        )

        self.ser.addOperator(op['op'], input_list, output_list, None, qinfo)
        return result_tens

    def build_binary_broadcast(self, op, a, b, validator_fcns, error_name=None):
        result_tens = OutputShaper.binaryBroadcastOp(self.ser, self.rng, a, b, error_name)


        # Invalidate Input/Output list for error if checks.
        input_list = [a.name, b.name]
        output_list = [result_tens.name]
        pCount, cCount = op["operands"]
        num_operands = pCount + cCount
        input_list, output_list = TosaErrorIfArgGen.eiInvalidateInputOutputList(self, error_name, input_list, output_list)

        TosaErrorValidator.evValidateErrorIfs(
            self.ser,
            validator_fcns,
            error_name,
            op=op,
            input1 = a,
            input2 = b,
            input_dtype = a.dtype,
            output_dtype = result_tens.dtype,
            result_tensor = result_tens,
            input_list=input_list,
            output_list=output_list,
            num_operands=num_operands,
        )

        self.ser.addOperator(op['op'], input_list, output_list)
        return result_tens

    def build_binary_nonbroadcast(self, op, a, b, validator_fcns=None, error_name=None):
        result_tens = OutputShaper.binaryNonBroadcastOp(self.ser, a, b)
        self.ser.addOperator(op['op'], [a.name, b.name], [result_tens.name])
        return result_tens

    def build_arithmetic_right_shift(self, op, a, b, round, validator_fcns=None, error_name=None):
        result_tens = OutputShaper.binaryBroadcastOp(self.ser, self.rng, a, b, error_name)

        # Invalidate Input/Output list for error if checks.
        input_list = [a.name, b.name]
        output_list = [result_tens.name]
        pCount, cCount = op["operands"]
        num_operands = pCount + cCount
        input_list, output_list = TosaErrorIfArgGen.eiInvalidateInputOutputList(self, error_name, input_list, output_list)

        TosaErrorValidator.evValidateErrorIfs(
            self.ser,
            validator_fcns,
            error_name,
            op=op,
            input1 = a,
            input2 = b,
            input_dtype = a.dtype,
            output_dtype = result_tens.dtype,
            result_tensor = result_tens,
            input_list=input_list,
            output_list=output_list,
            num_operands=num_operands,
        )

        attr = ts.TosaSerializerAttribute()
        attr.ArithmeticRightShiftAttribute(round)

        self.ser.addOperator(op['op'], input_list, output_list, attr)
        return result_tens

    def build_mul(self, op, a, b, shift, validator_fcns=None, error_name=None):
        result_tens = OutputShaper.binaryBroadcastOp(self.ser, self.rng, a, b, error_name)

        # Special for multiply:
        # Force the result to INT32 for INT types
        if a.dtype != DType.FLOAT:
            result_tens.setDtype(DType.INT32)
        if error_name == ErrorIf.WrongOutputType:
            all_dtypes = [DType.INT8, DType.INT16, DType.INT48]
            outputDType = self.rng.choice(all_dtypes)
            result_tens.setDtype(outputDType)

        # Invalidate Input/Output list for error if checks.
        input_list = [a.name, b.name]
        output_list = [result_tens.name]
        pCount, cCount = op["operands"]
        num_operands = pCount + cCount
        input_list, output_list = TosaErrorIfArgGen.eiInvalidateInputOutputList(self, error_name, input_list, output_list)

        TosaErrorValidator.evValidateErrorIfs(
            self.ser,
            validator_fcns,
            error_name,
            op=op,
            input1 = a,
            input2 = b,
            input_dtype = a.dtype,
            output_dtype = result_tens.dtype,
            result_tensor = result_tens,
            input_list=input_list,
            output_list=output_list,
            num_operands=num_operands,
        )

        attr = ts.TosaSerializerAttribute()
        attr.MulAttribute(shift)

        self.ser.addOperator(op['op'], input_list, output_list, attr)
        return result_tens

    def build_table(self, op, a, table, validator_fcns=None, error_name=None):
        result_tens = OutputShaper.tableOp(self.ser, self.rng, a, error_name)

        attr = ts.TosaSerializerAttribute()
        attr.TableAttribute(table)

        # Invalidate Input/Output list for error if checks.
        input_list = [a.name]
        output_list = [result_tens.name]
        pCount, cCount = op["operands"]
        num_operands = pCount + cCount
        input_list, output_list = TosaErrorIfArgGen.eiInvalidateInputOutputList(self, error_name, input_list, output_list)

        TosaErrorValidator.evValidateErrorIfs(
            self.ser,
            validator_fcns,
            error_name,
            op=op,
            input_shape = a.shape,
            input_dtype = a.dtype,
            output_dtype = result_tens.dtype,
            result_tensor = result_tens,
            input_list=input_list,
            output_list=output_list,
            num_operands=num_operands,
        )

        self.ser.addOperator(op['op'], input_list, output_list, attr)

        return result_tens

    def build_select(self, op, cond, a, b, validator_fcns=None, error_name=None):
        result_tens = OutputShaper.selectOp(self.ser, self.rng, cond, a, b, error_name)

        # Invalidate Input/Output list for error if checks.
        input_list = [cond.name, a.name, b.name]
        output_list = [result_tens.name]
        pCount, cCount = op["operands"]
        num_operands = pCount + cCount
        input_list, output_list = TosaErrorIfArgGen.eiInvalidateInputOutputList(self, error_name, input_list, output_list)

        TosaErrorValidator.evValidateErrorIfs(
            self.ser,
            validator_fcns,
            error_name,
            op=op,
            input_shape = a.shape,
            input_dtype = a.dtype,
            output_dtype = result_tens.dtype,
            result_tensor = result_tens,
            input_list=input_list,
            output_list=output_list,
            num_operands=num_operands,
        )

        self.ser.addOperator(op['op'], input_list, output_list,)
        return result_tens

    def build_comparison(self, op, a, b, validator_fcns=None, error_name=None):
        result_tens = OutputShaper.binaryComparisonOp(self.ser, self.rng, a, b, error_name)

        # Invalidate Input/Output list for error if checks.
        input_list = [a.name, b.name]
        output_list = [result_tens.name]
        pCount, cCount = op["operands"]
        num_operands = pCount + cCount
        input_list, output_list = TosaErrorIfArgGen.eiInvalidateInputOutputList(self, error_name, input_list, output_list)

        TosaErrorValidator.evValidateErrorIfs(
            self.ser,
            validator_fcns,
            error_name,
            op=op,
            input_shape = a.shape,
            input_dtype = a.dtype,
            output_shape = result_tens.shape,
            output_dtype = result_tens.dtype,
            result_tensor = result_tens,
            input_list=input_list,
            output_list=output_list,
            num_operands=num_operands,
        )

        self.ser.addOperator(op['op'], input_list, output_list,)
        return result_tens

    def build_argmax(self, op, a, axis, validator_fcns, error_name):
        result_tens = OutputShaper.argmaxOp(self.ser, self.rng, a, axis, error_name)

        # Invalidate Input/Output list for error if checks.
        input_list = [a.name]
        output_list = [result_tens.name]
        pCount, cCount = op["operands"]
        num_operands = pCount + cCount
        input_list, output_list = TosaErrorIfArgGen.eiInvalidateInputOutputList(self, error_name, input_list, output_list)

        TosaErrorValidator.evValidateErrorIfs(
            self.ser,
            validator_fcns,
            error_name,
            op=op,
            axis=axis,
            input_shape = a.shape,
            input_dtype = a.dtype,
            output_shape = result_tens.shape,
            output_dtype = result_tens.dtype,
            result_tensor = result_tens,
            input_list=input_list,
            output_list=output_list,
            num_operands=num_operands,
        )

        attr = ts.TosaSerializerAttribute()
        attr.AxisAttribute(axis)

        self.ser.addOperator(op['op'], input_list, output_list, attr)
        return result_tens

    def build_pool2d(self, op, input, stride, pad, kernel, validator_fcns=None, error_name=None, qinfo=None):
        result_tens = OutputShaper.pool2dOp(self.ser, self.rng, input, kernel, stride, pad, error_name)

        # Ensure new output type has correct qinfo
        if error_name == ErrorIf.WrongInputType:
            if input.dtype not in [DType.INT8, DType.UINT8]:
                qinfo = ts.TosaSerializerQuantInfo()
                qinfo.UnaryQuantInfo(
                TosaQuantGen.getQinfo(self, input.dtype), TosaQuantGen.getQinfo(self, result_tens.dtype)
                )

        # Invalidate Input/Output list for error if checks.
        input_list = [input.name]
        output_list = [result_tens.name]
        pCount, cCount = op["operands"]
        num_operands = pCount + cCount
        input_list, output_list = TosaErrorIfArgGen.eiInvalidateInputOutputList(self, error_name, input_list, output_list)

        TosaErrorValidator.evValidateErrorIfs(
            self.ser,
            validator_fcns,
            error_name,
            op=op,
            input_shape=input.shape,
            input_dtype=input.dtype,
            output_shape=result_tens.shape,
            output_dtype=result_tens.dtype,
            kernel=kernel,
            stride=stride,
            pad=pad,
            qinfo = qinfo,
            result_tensor = result_tens,
            input_list=input_list,
            output_list=output_list,
            num_operands=num_operands,
        )

        attr = ts.TosaSerializerAttribute()
        attr.PoolAttribute(kernel, stride, pad)

        self.ser.addOperator(op['op'], input_list, output_list, attr, qinfo)
        return result_tens

    def build_conv2d(self, op, ifm, filter, bias, strides, padding, dilations, qinfo):
        assert len(padding) == 4
        result_tens = OutputShaper.conv2dOp(
            self.ser, ifm, filter, strides, padding, dilations
        )

        attr = ts.TosaSerializerAttribute()
        attr.ConvAttribute(padding, strides, dilations)

        self.ser.addOperator(
            op['op'], [ifm.name, filter.name, bias.name], [result_tens.name], attr, qinfo
        )
        return result_tens

    def build_conv3d(self, op, ifm, filter, bias, strides, padding, dilations, qinfo):
        assert len(padding) == 6
        result_tens = OutputShaper.conv3dOp(
            self.ser, ifm, filter, strides, padding, dilations
        )

        attr = ts.TosaSerializerAttribute()
        attr.ConvAttribute(padding, strides, dilations)

        self.ser.addOperator(
            op['op'], [ifm.name, filter.name, bias.name], [result_tens.name], attr, qinfo
        )
        return result_tens

    def build_transpose_conv2d(
        self, op, ifm, filter, bias, stride, outpad, dilation, output_shape, qinfo
    ):
        assert len(outpad) == 2
        result_tens = OutputShaper.transposeConv2DOp(self.ser, ifm, output_shape)

        attr = ts.TosaSerializerAttribute()
        attr.TransposeConvAttribute(outpad, stride, dilation, output_shape)

        self.ser.addOperator(
            op['op'], [ifm.name, filter.name, bias.name], [result_tens.name], attr, qinfo
        )
        return result_tens

    def build_depthwise_conv2d(
        self, op, ifm, filter, bias, strides, padding, dilations, qinfo
    ):
        result_tens = OutputShaper.depthwiseConv2dOp(
            self.ser, ifm, filter, strides, padding, dilations
        )

        attr = ts.TosaSerializerAttribute()
        attr.ConvAttribute(padding, strides, dilations)

        self.ser.addOperator(
            op['op'], [ifm.name, filter.name, bias.name], [result_tens.name], attr, qinfo
        )
        return result_tens

    def build_fully_connected(self, op, ifm, filter, bias, validator_fcns=None, error_name=None, qinfo=None):
        result_tens = OutputShaper.fullyConnectedOp(self.ser, self.rng, ifm, filter, error_name)

        # Invalidate Input/Output list for error if checks.
        input_list = [ifm.name, filter.name, bias.name]
        output_list = [result_tens.name]
        pCount, cCount = op["operands"]
        num_operands = pCount + cCount
        input_list, output_list = TosaErrorIfArgGen.eiInvalidateInputOutputList(self, error_name, input_list, output_list)

        TosaErrorValidator.evValidateErrorIfs(
            self.ser,
            validator_fcns,
            error_name,
            op=op,
            input_shape=ifm.shape,
            input_dtype=ifm.dtype,
            weight_dtype=filter.dtype,
            output_shape=result_tens.shape,
            output_dtype=result_tens.dtype,
            qinfo = qinfo,
            result_tensor = result_tens,
            input_list=input_list,
            output_list=output_list,
            num_operands=num_operands,
        )

        self.ser.addOperator(
            op['op'], input_list, output_list, None, qinfo
        )
        return result_tens

    def build_matmul(self, op, a, b, validator_fcns=None, error_name=None, qinfo=None):
        result_tens = OutputShaper.matmulOp(self.ser, self.rng, a, b, error_name)

        # Invalidate Input/Output list for error if checks.
        input_list = [a.name, b.name]
        output_list = [result_tens.name]
        pCount, cCount = op["operands"]
        num_operands = pCount + cCount
        input_list, output_list = TosaErrorIfArgGen.eiInvalidateInputOutputList(self, error_name, input_list, output_list)

        TosaErrorValidator.evValidateErrorIfs(
            self.ser,
            validator_fcns,
            error_name,
            op=op,
            input_shape=a.shape,
            input_dtype=a.dtype,
            input2_shape=b.shape,
            input2_dtype=b.dtype,
            output_shape=result_tens.shape,
            output_dtype=result_tens.dtype,
            qinfo = qinfo,
            result_tensor = result_tens,
            input_list=input_list,
            output_list=output_list,
            num_operands=num_operands,
        )

        self.ser.addOperator(op['op'], input_list, output_list, None, qinfo)
        return result_tens

    def build_reduce(self, op, a, axis, validator_fcns, error_name=None):
        result_tens = OutputShaper.reduceOp(self.ser, self.rng, a, axis, error_name)

        # Invalidate Input/Output list for error if checks.
        input_list = [a.name]
        output_list = [result_tens.name]
        pCount, cCount = op["operands"]
        num_operands = pCount + cCount
        input_list, output_list = TosaErrorIfArgGen.eiInvalidateInputOutputList(self, error_name, input_list, output_list)

        TosaErrorValidator.evValidateErrorIfs(
            self.ser,
            validator_fcns,
            error_name,
            op=op,
            axis = axis,
            input_shape = a.shape,
            output_shape = result_tens.shape,
            input_dtype = a.dtype,
            output_dtype = result_tens.dtype,
            result_tensor = result_tens,
            input_list=input_list,
            output_list=output_list,
            num_operands=num_operands,
        )

        attr = ts.TosaSerializerAttribute()
        attr.AxisAttribute(axis)

        self.ser.addOperator(op['op'], input_list, output_list, attr)
        return result_tens

    def build_clamp(self, op, a, validator_fcns=None, error_name=None):
        result_tens = OutputShaper.unaryOp(self.ser, self.rng, a, error_name)

        v = [self.getRandNumberDType(a.dtype), self.getRandNumberDType(a.dtype)]

        if error_name == ErrorIf.MaxSmallerMin:
            # Make sure the numbers are different to invoke this error
            while v[0] == v[1]:
                v = [self.getRandNumberDType(a.dtype), self.getRandNumberDType(a.dtype)]
            max_val = min(v)
            min_val = max(v)
        else:
            max_val = max(v)
            min_val = min(v)

        # Invalidate Input/Output list for error if checks.
        input_list = [a.name]
        output_list = [result_tens.name]
        pCount, cCount = op["operands"]
        num_operands = pCount + cCount
        input_list, output_list = TosaErrorIfArgGen.eiInvalidateInputOutputList(self, error_name, input_list, output_list)

        TosaErrorValidator.evValidateErrorIfs(
            self.ser,
            validator_fcns,
            error_name,
            op=op,
            max_val=max_val,
            min_val=min_val,
            input_shape = a.shape,
            output_shape = result_tens.shape,
            input_dtype = a.dtype,
            output_dtype = result_tens.dtype,
            result_tensor = result_tens,
            input_list=input_list,
            output_list=output_list,
            num_operands=num_operands,
        )

        attr = ts.TosaSerializerAttribute()
        if a.dtype == DType.FLOAT:
            attr.ClampAttribute(0, 0, min_val, max_val)
        else:
            attr.ClampAttribute(min_val, max_val, 0, 0)

        self.ser.addOperator(op['op'], input_list, output_list, attr)
        return result_tens

    def build_leaky_relu(self, op, a, validator_fcns=None, error_name=None):
        result_tens = OutputShaper.unaryOp(self.ser, self.rng, a, error_name)
        attr = ts.TosaSerializerAttribute()

        attr.LeakyReluAttribute(self.getRandNumberDType(DType.FLOAT))

        self.ser.addOperator(op['op'], [a.name], [result_tens.name], attr)
        return result_tens

    # Needs an additional type/input
    def build_prelu(self, op, a, validator_fcns=None, error_name=None):
        result_tens = OutputShaper.unaryOp(self.ser, self.rng, a, error_name)

        self.ser.addOperator(op['op'], [a.name], [result_tens.name])
        return result_tens

    def build_sigmoid(self, op, a, validator_fcns=None, error_name=None):
        result_tens = OutputShaper.unaryOp(self.ser, self.rng, a, error_name)

        # Invalidate Input/Output list for error if checks.
        input_list = [a.name]
        output_list = [result_tens.name]
        pCount, cCount = op["operands"]
        num_operands = pCount + cCount
        input_list, output_list = TosaErrorIfArgGen.eiInvalidateInputOutputList(self, error_name, input_list, output_list)

        TosaErrorValidator.evValidateErrorIfs(
            self.ser,
            validator_fcns,
            error_name,
            op=op,
            input_shape = a.shape,
            output_shape = result_tens.shape,
            input_dtype = a.dtype,
            output_dtype = result_tens.dtype,
            result_tensor = result_tens,
            input_list=input_list,
            output_list=output_list,
            num_operands=num_operands,
        )

        self.ser.addOperator(op['op'], input_list, output_list)
        return result_tens

    def build_tanh(self, op, a, validator_fcns=None, error_name=None):
        result_tens = OutputShaper.unaryOp(self.ser, self.rng, a, error_name)

        # Invalidate Input/Output list for error if checks.
        input_list = [a.name]
        output_list = [result_tens.name]
        pCount, cCount = op["operands"]
        num_operands = pCount + cCount
        input_list, output_list = TosaErrorIfArgGen.eiInvalidateInputOutputList(self, error_name, input_list, output_list)

        TosaErrorValidator.evValidateErrorIfs(
            self.ser,
            validator_fcns,
            error_name,
            op=op,
            input_shape = a.shape,
            output_shape = result_tens.shape,
            input_dtype = a.dtype,
            output_dtype = result_tens.dtype,
            result_tensor = result_tens,
            input_list=input_list,
            output_list=output_list,
            num_operands=num_operands,
        )

        self.ser.addOperator(op['op'], input_list, output_list)
        return result_tens

    def build_concat(self, op, *a, validator_fcns=None, error_name=None):
        if error_name != ErrorIf.WrongInputType:
            assert type(a[-1]) == int

        # To store variable length list of input tensors we need to store axis along with it
        axis = a[-1]
        a = a[:-1]

        result_tens = OutputShaper.concatOp(self.ser, self.rng, axis, *a, error_name=error_name)

        input_tensor_names = []
        for tensor in a:
            input_tensor_names.append(tensor.name)

        # Invalidate Input/Output list for error if checks.
        input_list = input_tensor_names
        output_list = [result_tens.name]
        pCount, cCount = op["operands"]
        num_operands = pCount + cCount
        input_list, output_list = TosaErrorIfArgGen.eiInvalidateInputOutputList(self, error_name, input_list, output_list)

        TosaErrorValidator.evValidateErrorIfs(
            self.ser,
            validator_fcns,
            error_name,
            op=op,
            axis=axis,
            input_shape = a[0].shape,
            output_shape = result_tens.shape,
            input_dtype = a[0].dtype,
            output_dtype = result_tens.dtype,
            inputs=a,
            result_tensor = result_tens,
            input_list=input_list,
            output_list=output_list,
            num_operands=num_operands,
        )

        attr = ts.TosaSerializerAttribute()
        attr.AxisAttribute(axis)


        self.ser.addOperator(op['op'], input_list, output_list, attr)
        return result_tens

    def build_pad(self, op, a, padding, pad_const_int, pad_const_float, validator_fcns=None, error_name=None, qinfo=None):
        result_tens = OutputShaper.padOp(self.ser, self.rng, a, padding, error_name)

        attr = ts.TosaSerializerAttribute()
        attr.PadAttribute(padding.flatten(), pad_const_int, pad_const_float)

        # Invalidate Input/Output list for error if checks.
        input_list = [a.name]
        output_list = [result_tens.name]
        pCount, cCount = op["operands"]
        num_operands = pCount + cCount
        input_list, output_list = TosaErrorIfArgGen.eiInvalidateInputOutputList(self, error_name, input_list, output_list)

        TosaErrorValidator.evValidateErrorIfs(
            self.ser,
            validator_fcns,
            error_name,
            op=op,
            input_shape = a.shape,
            output_shape = result_tens.shape,
            input_dtype = a.dtype,
            output_dtype = result_tens.dtype,
            pad=padding,
            qinfo=qinfo,
            result_tensor = result_tens,
            input_list=input_list,
            output_list=output_list,
            num_operands=num_operands,
        )

        self.ser.addOperator(
            op['op'], input_list, output_list, attr, qinfo
        )
        return result_tens

    def build_reshape(self, op, a, newShape, validator_fcns=None, error_name=None):
        result_tens = OutputShaper.reshapeOp(self.ser, self.rng, a, newShape, error_name)

        # Invalidate Input/Output list for error if checks.
        input_list = [a.name]
        output_list = [result_tens.name]
        pCount, cCount = op["operands"]
        num_operands = pCount + cCount
        input_list, output_list = TosaErrorIfArgGen.eiInvalidateInputOutputList(self, error_name, input_list, output_list)

        TosaErrorValidator.evValidateErrorIfs(
            self.ser,
            validator_fcns,
            error_name,
            op=op,
            input_shape = a.shape,
            output_shape = result_tens.shape,
            input_dtype = a.dtype,
            output_dtype = result_tens.dtype,
            result_tensor = result_tens,
            input_list=input_list,
            output_list=output_list,
            num_operands=num_operands,
        )

        attr = ts.TosaSerializerAttribute()
        attr.ReshapeAttribute(newShape)

        self.ser.addOperator(op['op'], input_list, output_list, attr)
        return result_tens

    def build_reverse(self, op, a, axis, validator_fcns=None, error_name=None):
        result_tens = OutputShaper.unaryOp(self.ser, self.rng, a, error_name)

        # Invalidate Input/Output list for error if checks.
        input_list = [a.name]
        output_list = [result_tens.name]
        pCount, cCount = op["operands"]
        num_operands = pCount + cCount
        input_list, output_list = TosaErrorIfArgGen.eiInvalidateInputOutputList(self, error_name, input_list, output_list)

        TosaErrorValidator.evValidateErrorIfs(
            self.ser,
            validator_fcns,
            error_name,
            op=op,
            axis=axis,
            input_shape = a.shape,
            output_shape = result_tens.shape,
            input_dtype = a.dtype,
            output_dtype = result_tens.dtype,
            result_tensor = result_tens,
            input_list=input_list,
            output_list=output_list,
            num_operands=num_operands,
        )

        attr = ts.TosaSerializerAttribute()
        attr.AxisAttribute(axis)

        self.ser.addOperator(op['op'], input_list, output_list, attr)
        return result_tens

    def build_transpose(self, op, a, perms, validator_fcns=None, error_name=None):
        result_tens = OutputShaper.transposeOp(self.ser, self.rng, a, perms, error_name)

        attr = ts.TosaSerializerAttribute()
        attr.TransposeAttribute(perms)

        # Invalidate Input/Output list for error if checks.
        input_list = [a.name]
        output_list = [result_tens.name]
        pCount, cCount = op["operands"]
        num_operands = pCount + cCount
        input_list, output_list = TosaErrorIfArgGen.eiInvalidateInputOutputList(self, error_name, input_list, output_list)

        TosaErrorValidator.evValidateErrorIfs(
            self.ser,
            validator_fcns,
            error_name,
            op=op,
            input_shape = a.shape,
            output_shape = result_tens.shape,
            perms=perms,
            input_dtype = a.dtype,
            output_dtype = result_tens.dtype,
            result_tensor = result_tens,
            input_list=input_list,
            output_list=output_list,
            num_operands=num_operands,
        )


        self.ser.addOperator(op['op'], input_list, output_list, attr)
        return result_tens

    def build_slice(self, op, a, start, size, validator_fcns=None, error_name=None):
        result_tens = OutputShaper.sliceOp(self.ser, self.rng, a, start, size, error_name)

        # Invalidate Input/Output list for error if checks.
        input_list = [a.name]
        output_list = [result_tens.name]
        pCount, cCount = op["operands"]
        num_operands = pCount + cCount
        input_list, output_list = TosaErrorIfArgGen.eiInvalidateInputOutputList(self, error_name, input_list, output_list)

        TosaErrorValidator.evValidateErrorIfs(
            self.ser,
            validator_fcns,
            error_name,
            op=op,
            input_shape = a.shape,
            output_shape = result_tens.shape,
            input_dtype = a.dtype,
            output_dtype = result_tens.dtype,
            start=start,
            size=size,
            result_tensor = result_tens,
            input_list=input_list,
            output_list=output_list,
            num_operands=num_operands,
        )

        attr = ts.TosaSerializerAttribute()
        attr.SliceAttribute(start, size)

        self.ser.addOperator(op['op'], input_list, output_list, attr)
        return result_tens

    def build_tile(self, op, a, multiples, validator_fcns=None, error_name=None):
        result_tens = OutputShaper.tileOp(self.ser, self.rng, a, multiples, error_name)

        # Invalidate Input/Output list for error if checks.
        input_list = [a.name]
        output_list = [result_tens.name]
        pCount, cCount = op["operands"]
        num_operands = pCount + cCount
        input_list, output_list = TosaErrorIfArgGen.eiInvalidateInputOutputList(self, error_name, input_list, output_list)

        TosaErrorValidator.evValidateErrorIfs(
            self.ser,
            validator_fcns,
            error_name,
            op=op,
            input_shape = a.shape,
            output_shape = result_tens.shape,
            input_dtype = a.dtype,
            output_dtype = result_tens.dtype,
            result_tensor = result_tens,
            input_list=input_list,
            output_list=output_list,
            num_operands=num_operands,
        )

        attr = ts.TosaSerializerAttribute()
        attr.TileAttribute(multiples)

        self.ser.addOperator(op['op'], input_list, output_list, attr)
        return result_tens

    def build_gather(self, op, values, validator_fcns=None, error_name=None):

        # Create a new indicies tensor
        # here with data that doesn't exceed the dimensions of the values tensor

        K = values.shape[1]  # K
        W = self.randInt(
            self.args.tensor_shape_range[0], self.args.tensor_shape_range[1]
        )  # W
        indicies_arr = np.int32(
            self.rng.integers(low=0, high=K, size=[values.shape[0], W])
        )  # (N, W)
        indicies = self.ser.addConst(indicies_arr.shape, DType.INT32, indicies_arr)

        result_tens = OutputShaper.gatherOp(self.ser, self.rng, values, indicies, error_name)

        # Invalidate Input/Output list for error if checks.
        input_list = [values.name, indicies.name]
        output_list = [result_tens.name]
        pCount, cCount = op["operands"]
        num_operands = pCount + cCount
        input_list, output_list = TosaErrorIfArgGen.eiInvalidateInputOutputList(self, error_name, input_list, output_list)

        TosaErrorValidator.evValidateErrorIfs(
            self.ser,
            validator_fcns,
            error_name,
            op=op,
            input_shape = values.shape,
            output_shape = result_tens.shape,
            input_dtype = values.dtype,
            output_dtype = result_tens.dtype,
            result_tensor = result_tens,
            input_list=input_list,
            output_list=output_list,
            num_operands=num_operands,
        )

        self.ser.addOperator(op['op'], input_list, output_list)

        return result_tens

    def build_scatter(self, op, values_in, input, validator_fcns=None, error_name=None):

        # Create a new indicies tensor
        # here with data that doesn't exceed the dimensions of the values_in tensor

        K = values_in.shape[1]  # K
        W = input.shape[1]  # W
        indicies_arr = np.int32(
            self.rng.integers(low=0, high=K, size=[values_in.shape[0], W])
        )  # (N, W)
        indicies = self.ser.addConst(indicies_arr.shape, DType.INT32, indicies_arr)

        result_tens = OutputShaper.scatterOp(self.ser, self.rng, values_in, indicies, input, error_name)

        # Invalidate Input/Output list for error if checks.
        input_list = [values_in.name, indicies.name, input.name]
        output_list = [result_tens.name]
        pCount, cCount = op["operands"]
        num_operands = pCount + cCount
        input_list, output_list = TosaErrorIfArgGen.eiInvalidateInputOutputList(self, error_name, input_list, output_list)

        TosaErrorValidator.evValidateErrorIfs(
            self.ser,
            validator_fcns,
            error_name,
            op=op,
            input_shape = input.shape,
            output_shape = result_tens.shape,
            input_dtype = input.dtype,
            output_dtype = result_tens.dtype,
            result_tensor = result_tens,
            input_list=input_list,
            output_list=output_list,
            num_operands=num_operands,
        )

        self.ser.addOperator(op['op'], input_list, output_list)

        return result_tens


    def build_resize(
        self,
        op,
        input,
        mode,
        stride,
        offset,
        shift,
        stride_fp,
        offset_fp,
        output_dims,
        input_dtype,
        output_dtype,
        validator_fcns,
        error_name = None,
    ):
        result_tens = OutputShaper.resizeOp(
            self.ser,
            self.rng,
            input,
            mode,
            stride,
            offset,
            shift,
            stride_fp,
            offset_fp,
            output_dims,
            input_dtype,
            output_dtype,
            error_name
        )

        # Invalidate Input/Output list for error if checks.
        input_list = [input.name]
        output_list = [result_tens.name]
        pCount, cCount = op["operands"]
        num_operands = pCount + cCount
        input_list, output_list = TosaErrorIfArgGen.eiInvalidateInputOutputList(self, error_name, input_list, output_list)

        TosaErrorValidator.evValidateErrorIfs(
            self.ser,
            validator_fcns,
            error_name,
            op=op,
            mode=mode,
            shift=shift,
            input_dtype=input_dtype,
            output_dtype=output_dtype,
            input_shape=input.shape,
            output_shape=output_dims,
            offset=offset,
            offset_fp=offset_fp,
            stride=stride,
            stride_fp=stride_fp,
            input_list=input_list,
            output_list=output_list,
            result_tensor=result_tens,
            num_operands=num_operands,
        )

        attr = ts.TosaSerializerAttribute()

        attr.ResizeAttribute(
            output_dims, stride, offset, shift, stride_fp, offset_fp, mode
        )

        self.ser.addOperator(op['op'], input_list, output_list, attr)
        return result_tens

    def build_identityn(self, op, val, val2, validator_fcns=None, error_name=None):
        result_tens = OutputShaper.unaryOp(self.ser, self.rng, val, error_name)
        result_tens2 = OutputShaper.unaryOp(self.ser, self.rng, val2, error_name)
        self.ser.addOperator(
            op, [val.name, val2.name], [result_tens.name, result_tens2.name]
        )
        return result_tens

    def build_const(self, op, val, validator_fcns=None, error_name=None):
        self.ser.addOutputTensor(val)
        return val

    # Type Conversion
    def build_cast(self, op, val, out_dtype, validator_fcns=None, error_name=None):
        result_tens = OutputShaper.typeConversionOp(self.ser, self.rng, val, out_dtype, error_name)

        # Invalidate Input/Output list for error if checks.
        input_list = [val.name]
        output_list = [result_tens.name]
        pCount, cCount = op["operands"]
        num_operands = pCount + cCount
        input_list, output_list = TosaErrorIfArgGen.eiInvalidateInputOutputList(self, error_name, input_list, output_list)

        TosaErrorValidator.evValidateErrorIfs(
            self.ser,
            validator_fcns,
            error_name,
            op=op,
            input_shape = val.shape,
            output_shape = result_tens.shape,
            input_dtype = val.dtype,
            output_dtype = result_tens.dtype,
            result_tensor = result_tens,
            input_list=input_list,
            output_list=output_list,
            num_operands=num_operands,
        )

        self.ser.addOperator(op['op'], input_list, output_list)
        return result_tens

    def build_rescale(self, op, val, out_dtype, scale32, double_round, per_channel, validator_fcns, error_name):
        result_tens = OutputShaper.typeConversionOp(self.ser, self.rng, val, out_dtype, error_name)

        if per_channel:
            nc = val.shape[-1]
        else:
            nc = 1

        in_type_width = self.typeWidth(val.dtype)
        out_type_width = self.typeWidth(out_dtype)

        if val.dtype == DType.INT8:
            input_zp = self.randInt(-128, 128)
            in_type_width = in_type_width + 1
        elif val.dtype == DType.UINT8:
            input_zp = self.randInt(0, 256)
            in_type_width = in_type_width + 1
        elif error_name == ErrorIf.InputZeroPointNotZero:
            input_zp = self.randInt(-128, 128)
            if input_zp == 0:
                input_zp = input_zp + self.rng.integers(1, 10)
            in_type_width = in_type_width + 1
        else:
            input_zp = 0

        if out_dtype == DType.INT8:
            output_zp = self.randInt(-128, 128)
            out_type_width = out_type_width + 1
        elif out_dtype == DType.UINT8:
            output_zp = self.randInt(0, 256)
            out_type_width = out_type_width + 1
        elif error_name == ErrorIf.OutputZeroPointNotZero:
            output_zp = self.randInt(-128, 128)
            if output_zp == 0:
                output_zp = output_zp + self.rng.integers(1, 10)
            out_type_width = out_type_width + 1
        else:
            output_zp = 0

        # Calculate scale based on:
        # scale = a *(2^output_width)/(2^input_width))

        a = np.float32(self.rng.random(size=[nc]))
        scale_arr = a * np.float32((1 << out_type_width) / (1 << in_type_width))

        if scale32:
            pass
            # Cap the scaling at 2^31 - 1 for scale32
            scale_arr = np.clip(scale_arr, 1.0 / (1 << 31), (1 << 31) - 1)
        else:
            # Cap the scaling at 2^15 - 1 for scale16
            scale_arr = np.clip(scale_arr, 1.0 / (1 << 31), 32767.0)

        # print('{} {} -> {}'.format(out_type_width, in_type_width, scale_arr))

        multiplier_arr = np.int32(np.zeros(shape=[nc]))
        shift_arr = np.int32(np.zeros(shape=[nc]))

        for i in range(nc):
            multiplier_arr[i], shift_arr[i] = TosaQuantGen.computeMultiplierAndShift(
                scale_arr[i], scale32
            )

        # print('multiplier {} shift {} inzp {} outzp {}'.format(multiplier_arr, shift_arr, input_zp, output_zp))

        # Invalidate Input/Output list for error if checks.
        input_list = [val.name]
        output_list = [result_tens.name]
        pCount, cCount = op["operands"]
        num_operands = pCount + cCount
        input_list, output_list = TosaErrorIfArgGen.eiInvalidateInputOutputList(self, error_name, input_list, output_list)

        qinfo = (input_zp, output_zp)
        TosaErrorValidator.evValidateErrorIfs(
            self.ser,
            validator_fcns,
            error_name,
            op=op,
            input_dtype=val.dtype,
            output_dtype=out_dtype,
            input_shape=val.shape,
            qinfo=qinfo,
            scale32 = scale32,
            double_round = double_round,
            input_list=input_list,
            output_list=output_list,
            result_tensor=result_tens,
            num_operands=num_operands,
        )

        attr = ts.TosaSerializerAttribute()
        attr.RescaleAttribute(
            input_zp,
            output_zp,
            multiplier_arr,
            shift_arr,
            scale32,
            double_round,
            per_channel,
        )

        self.ser.addOperator(op['op'], input_list, output_list, attr)
        return result_tens

    def build_cond_if_const(self, op, then_tens, else_tens, cond, validator_fcns=None, error_name=None):
        # For cond_if with constants, we're supplied with then/else tensors that we ignore
        # (except for the generated shap) and the condition.  Build Then/Else blocks
        # and fill them with const nodes for the body.

        # Condition tensor
        cond_tens = self.ser.addConst([], DType.BOOL, [cond])

        # Make then/else tensors
        out_shape = then_tens.shape

        # Create an incorrect output shape for error_if tests
        if error_name in [ErrorIf.CondIfOutputListThenGraphMismatch, ErrorIf.CondIfOutputListElseGraphMismatch]:
            incorrect_shape = deepcopy(then_tens.shape)
            for i in range(len(incorrect_shape)):
                incorrect_shape[i] = incorrect_shape[i] + self.rng.choice([-3, -2, 2, 3])
            incorrect_arr = np.int32(self.rng.integers(0, 256, size=incorrect_shape))

        then_arr = np.int32(self.rng.integers(0, 256, size=out_shape))
        else_arr = np.int32(self.rng.integers(0, 256, size=out_shape))

        # And the result tensor based on any of the outputs
        result_tens = self.ser.addOutput(out_shape, DType.INT32)

        # Create the attribute with the names of the then/else blocks
        then_block = "THEN_BLOCK"
        else_block = "ELSE_BLOCK"
        attr = ts.TosaSerializerAttribute()
        attr.CondIfAttribute(then_block, else_block)

        # Finally, build the op and the two blocks
        self.ser.addOperator(op['op'], [cond_tens.name], [result_tens.name], attr)

        self.ser.startBasicBlock(then_block)
        # Build the actual then/else tensors inside their blocks
        if error_name == ErrorIf.CondIfOutputListThenGraphMismatch:
            then_tens = self.ser.addConst(incorrect_shape, DType.INT32, incorrect_arr)
        else:
            then_tens = self.ser.addConst(out_shape, DType.INT32, then_arr)
        self.ser.addOutputTensor(then_tens)

        self.ser.startBasicBlock(else_block)
        if error_name == ErrorIf.CondIfOutputListElseGraphMismatch:
            else_tens = self.ser.addConst(incorrect_shape, DType.INT32, incorrect_arr)
        else:
            else_tens = self.ser.addConst(out_shape, DType.INT32, else_arr)
        self.ser.addOutputTensor(else_tens)

        TosaErrorValidator.evValidateErrorIfs(
            self.ser,
            validator_fcns,
            error_name,
            op=op,
            basicBlocks=self.ser.basicBlocks
        )

        return result_tens

    def build_cond_if_binary(self, op, a, b, cond, validator_fcns=None, error_name=None):
        # For cond_if with a binary op in the then/else blocks, take a and b and
        # alternately add or subtract them based on the condition

        # Condition tensor
        cond_tens = self.ser.addConst([], DType.BOOL, [cond])

        result_tens = self.ser.addOutput(a.shape, a.dtype)

        # Create the attribute with the names of the then/else blocks
        then_block = "THEN_BLOCK"
        else_block = "ELSE_BLOCK"
        attr = ts.TosaSerializerAttribute()
        attr.CondIfAttribute(then_block, else_block)

        if error_name in [ErrorIf.CondIfInputListThenGraphMismatch, ErrorIf.CondIfInputListElseGraphMismatch,
                          ErrorIf.CondIfOutputListElseGraphMismatch, ErrorIf.CondIfOutputListThenGraphMismatch]:
            incorrect_shape = a.shape.copy()
            for i in range(len(incorrect_shape)):
                incorrect_shape[i] += self.rng.choice([-3, -2, 2, 3])
            incorrect_block_input = deepcopy(a)
            incorrect_block_input.shape = incorrect_shape


        # Finally, build the op and the two blocks
        self.ser.addOperator(
            op['op'], [cond_tens.name, a.name, b.name], [result_tens.name], attr
        )

        if a.dtype in (DType.FLOAT, DType.INT32):
            then_op, else_op = Op.ADD, Op.SUB
        elif a.dtype in (DType.INT8, DType.INT16):
            then_op, else_op = Op.LOGICAL_RIGHT_SHIFT, Op.LOGICAL_LEFT_SHIFT
        else:
            assert False, f"No tests for DType: {a.dtype}"

        for block, op in ((then_block, then_op), (else_block, else_op)):
            self.ser.startBasicBlock(block)
            if ((error_name == ErrorIf.CondIfInputListThenGraphMismatch and block == then_block) or
                (error_name == ErrorIf.CondIfInputListElseGraphMismatch and block == else_block)):
                self.ser.addInputTensor(incorrect_block_input)
                self.ser.addInputTensor(b)
                tens = self.ser.addOutput(a.shape, a.dtype)
            elif ((error_name == ErrorIf.CondIfOutputListThenGraphMismatch and block == then_block) or
                (error_name == ErrorIf.CondIfOutputListElseGraphMismatch and block == else_block)):
                self.ser.addInputTensor(a)
                self.ser.addInputTensor(b)
                tens = self.ser.addOutput(incorrect_block_input.shape, a.dtype)
            else:
                self.ser.addInputTensor(a)
                self.ser.addInputTensor(b)
                tens = self.ser.addOutput(a.shape, a.dtype)
            self.ser.addOperator(op, [a.name, b.name], [tens.name])

        TosaErrorValidator.evValidateErrorIfs(
            self.ser,
            validator_fcns,
            error_name,
            op=op,
            a=a,
            b=b,
            basicBlocks=self.ser.basicBlocks
        )

        return result_tens

    def build_while_loop(self, op, a, iter_val, validator_fcns=None, error_name=None):
        iter = self.ser.addPlaceholder([], DType.INT32, [np.int32(iter_val)])

        cond_block = "COND_BLOCK"
        body_block = "BODY_BLOCK"

        attr = ts.TosaSerializerAttribute()
        attr.WhileLoopAttribute(cond_block, body_block)

        # Accumulator tensor
        # acc = self.ser.addOutput(a.shape, a.dtype)
        acc_init_val = np.int32(np.zeros(a.shape))
        acc = self.ser.addPlaceholder(a.shape, a.dtype, acc_init_val)

        # Intermediate/output tensors for everything going through the loop
        iter_out = self.ser.addIntermediate(iter.shape, iter.dtype)
        a_out = self.ser.addIntermediate(a.shape, a.dtype)
        if error_name == ErrorIf.InputListOutputListMismatch:
            incorrect_acc = deepcopy(acc)
            for i in range(len(incorrect_acc.shape)):
                incorrect_acc.shape[i] += self.rng.choice([-3, -2, 2, 3])
            acc_out = self.ser.addIntermediate(incorrect_acc.shape, acc.dtype)
        else:
            acc_out = self.ser.addIntermediate(acc.shape, acc.dtype)

        # While_loop operator
        self.ser.addOperator(
            op['op'],
            [iter.name, a.name, acc.name],
            [iter_out.name, a_out.name, acc_out.name],
            attr,
        )
        self.ser.addOutputTensor(acc_out)

        if error_name in [ErrorIf.InputListCondGraphMismatch, ErrorIf.InputListBodyGraphInputMismatch, ErrorIf.InputListBodyGraphOutputMismatch]:
            incorrect_iter = deepcopy(iter)
            for i in range(len(incorrect_iter.shape)):
                incorrect_iter.shape[i] += self.rng.choice([-3, -2, 2, 3])
            if len(incorrect_iter.shape) == 0:
                incorrect_iter.shape.append(self.rng.choice([-3, -2, 2, 3]))

            incorrect_acc = deepcopy(acc)
            for i in range(len(incorrect_acc.shape)):
                incorrect_acc.shape[i] += self.rng.choice([-3, -2, 2, 3])

        # COND block (input: iter, output: cond_tens )
        self.ser.startBasicBlock(cond_block)
        if error_name == ErrorIf.InputListCondGraphMismatch:
            self.ser.addInputTensor(incorrect_iter)
            self.ser.addInputTensor(a)
            self.ser.addInputTensor(incorrect_acc)
        else:
            self.ser.addInputTensor(iter)
            self.ser.addInputTensor(a)
            self.ser.addInputTensor(acc)
        zero_tens = self.ser.addConst([], DType.INT32, [np.int32(0)])

        if error_name == ErrorIf.CondGraphOutputNotMatchingBool:
            cond_tens = self.ser.addOutput([], self.rng.choice([DType.INT8, DType.INT32, DType.FLOAT]))
        else:
            cond_tens = self.ser.addOutput([], DType.BOOL)

        self.ser.addOperator(Op.GREATER, [iter.name, zero_tens.name], [cond_tens.name])

        # BODY block (input: a, acc, iter, output: a, acc, iter)
        # Note that local intermediate tensors need to be declared here for the outputs
        self.ser.startBasicBlock(body_block)
        if error_name == ErrorIf.InputListBodyGraphInputMismatch:
            self.ser.addInputTensor(incorrect_iter)
            self.ser.addInputTensor(a)
            self.ser.addInputTensor(incorrect_acc)
        else:
            self.ser.addInputTensor(iter)
            self.ser.addInputTensor(a)
            self.ser.addInputTensor(acc)

        one_tens = self.ser.addConst([], DType.INT32, [np.int32(1)])

        if error_name == ErrorIf.InputListBodyGraphOutputMismatch:
            iter_body_out = self.ser.addIntermediate(incorrect_iter.shape, incorrect_iter.dtype)
            acc_body_out = self.ser.addIntermediate(incorrect_acc.shape, incorrect_acc.dtype)
        else:
            iter_body_out = self.ser.addIntermediate(iter.shape, iter.dtype)
            acc_body_out = self.ser.addIntermediate(acc.shape, acc.dtype)

        self.ser.addOperator(Op.ADD, [a.name, acc.name], [acc_body_out.name])
        self.ser.addOperator(Op.SUB, [iter.name, one_tens.name], [iter_body_out.name])
        self.ser.addOutputTensor(iter_body_out)
        self.ser.addOutputTensor(a)
        self.ser.addOutputTensor(acc_body_out)

        TosaErrorValidator.evValidateErrorIfs(
            self.ser,
            validator_fcns,
            error_name,
            op=op,
            basicBlocks=self.ser.basicBlocks
        )

        return acc_out

    def create_filter_lists(self, op, shapeFilter, rankFilter, dtypeFilter, testType, validator=None):
        # Create a default testing rank range, 1-4 inclusive to keep test sizes reasonably small.
        default_test_rank_range = range(1, 5)
        if not shapeFilter:
            shapeFilter = [None]

        # Calculate the filters based on what is requested and what the operator allows
        rmin, rmax = op["rank"]
        if rankFilter is not None:
            cleanRankFilter = []
            # Ensure rankFilter values are allowed by operator
            for rank in rankFilter:
                if rank >= rmin and rank <= rmax:
                    cleanRankFilter.append(rank)
        elif rankFilter is None and shapeFilter[0] is None:
            # Ensure default behaviour is bounded by default range or by operator,
            # whichever is the smaller range of ranks.
            opRankRange = range(rmin, rmax + 1)
            cleanRankFilter = opRankRange if len(opRankRange) <= len(default_test_rank_range) else default_test_rank_range
        else:
            cleanRankFilter = range(rmin, rmax + 1)

        dtypes = op["types"]

        if dtypeFilter is not None:
            cleanDtypeFilter = []
            # Create list of operator dtypes filtered by requested dtypes
            for dtype in dtypes:
                if dtype in dtypeFilter or (isinstance(dtype, list) and dtype[0] in dtypeFilter):
                    cleanDtypeFilter.append(dtype)
        else:
            cleanDtypeFilter = dtypes

        if testType == 'positive':
            filterDict = {
                'shapeFilter': shapeFilter,
                'rankFilter': cleanRankFilter,
                'dtypeFilter': cleanDtypeFilter
            }
            return filterDict
        elif testType == 'negative':
            if validator is not None:
                validator_info = validator(check=False, op=op)
            else:
                return None

            error_arguments = validator_info['param_reqs']

            #Set parameters as required
            if error_arguments['rank'] != None:
                rankFilter = error_arguments['rank']
            else:
                rankFilter = cleanRankFilter

            if error_arguments['dtype'] != None:
                dtypeFilter = error_arguments['dtype']
            else:
                dtypeFilter = cleanDtypeFilter

            if error_arguments['shape'] != None:
                shapeFilter = error_arguments['shape']
            else:
                shapeFilter = shapeFilter[:2] # Reduce number of shapes to keep test numbers small

            filterDict = {
                'shapeFilter': shapeFilter,
                'rankFilter': rankFilter,
                'dtypeFilter': dtypeFilter
            }
            return filterDict


    def genOpTestList(
        self, opName, shapeFilter=[None], rankFilter=None, dtypeFilter=None, testType='positive'
    ):

        try:
            op = self.TOSA_OP_LIST[opName]
        except KeyError as e:
            raise Exception("Cannot find op with name {}".format(opName))

        # Initialize a new random number generator
        self.rng = np.random.default_rng(self.random_seed)

        build_fcn, tgen_fcn, agen_fcn = op["build_fcn"]

        # Test list consists of a tuple of:
        # (opName, testNameStr, dtype, shapeList, argumentsList)
        testList = []
        if testType == 'negative' and "error_if_validators" in op:
            error_if_validators = op["error_if_validators"]
        else:
            error_if_validators = [None]

        for validator in error_if_validators:
            if validator is not None:
                error_name = validator(check=False, op=op)['error_name']
            else:
                error_name = None

            filterDict = self.create_filter_lists(op, shapeFilter, rankFilter, dtypeFilter, testType, validator)
            if filterDict == None:
                return []
            cleanRankFilter = filterDict['rankFilter']
            cleanDtypeFilter = filterDict['dtypeFilter']
            cleanShapeFilter = filterDict['shapeFilter']
            #print(f"Filters: S {shapeFilter}, R {cleanRankFilter}, T {cleanDtypeFilter}")

            for r in cleanRankFilter:
                if opName.startswith("conv3d"):
                    assert r == 5, "conv3d test must have input rank == 5"
                for t in cleanDtypeFilter:
                    for shape in cleanShapeFilter:
                        # Filter out by rank
                        if shape is not None and len(shape) != r:
                            continue
                        self.setTargetShape(shape)
                        shapeList = tgen_fcn(self, op, r, error_name)

                        shapeStr = self.shapeStr(shapeList[0])
                        typeStr = self.typeStr(t)

                        # Argument lists consists of tuples of the (str, []) string representation and the build function argument list
                        argList = []
                        if agen_fcn:
                            argList = agen_fcn(self, opName, shapeList, t, error_name)
                        else:
                            argList = [("", [])]

                        for argStr, args in argList:
                            if testType == 'positive':
                                if argStr:
                                    testStr = "{}_{}_{}_{}".format(
                                        opName, shapeStr, typeStr, argStr
                                    )
                                else:
                                    testStr = "{}_{}_{}".format(opName, shapeStr, typeStr)
                            elif testType == 'negative':
                                if argStr:
                                    testStr = "{}_ERRORIF_{}_{}_{}_{}".format(
                                        opName, error_name, shapeStr, typeStr, argStr
                                    )
                                else:
                                    testStr = "{}_ERRORIF_{}_{}_{}".format(opName, error_name, shapeStr, typeStr)

                            testList.append((opName, testStr, t, error_name, shapeList, args))

        if testType == 'positive':
            # Remove tests which are expected to fail but don't correlate to a ERROR_IF statement
            if "invalid_test_validators" in op:
                invalid_test_validators = op["invalid_test_validators"]
                clean_testList = []
                for test in testList:
                    for validator_fcn in invalid_test_validators:
                        remove_test = False
                        if validator_fcn(opName=test[0], input_dtype=test[2], shapeList=test[4], args=test[5]):
                            remove_test = True
                    if not remove_test:
                        clean_testList.append(test)
                testList = clean_testList

        return testList


    def serializeTest(self, opName, testStr, dtype_or_dtypeList, error_name, shapeList, testArgs):
        try:
            op = self.TOSA_OP_LIST[opName]
        except KeyError as e:
            raise Exception("Cannot find op with name {}".format(opName))

        # Create a serializer
        self.createSerializer(opName, testStr)

        build_fcn, tgen_fcn, agen_fcn = op["build_fcn"]
        if "error_if_validators" in op:
            error_if_validators = op["error_if_validators"]
        else:
            error_if_validators = None

        pCount, cCount = op["operands"]
        num_operands = pCount + cCount

        if isinstance(dtype_or_dtypeList, list):
            dtypeList = dtype_or_dtypeList
        elif op["op"] == Op.CONCAT:
            dtypeList = [dtype_or_dtypeList] * len(shapeList)
        else:
            dtypeList = [dtype_or_dtypeList] * (num_operands)

        if op["op"] != Op.CONCAT:
            assert (
                len(shapeList) == num_operands
            ), "shapeList length {} must match number of operands {}".format(
                len(shapeList), num_operands
            )
            assert (
                len(dtypeList) == num_operands
            ), "dtypeList length {} must match number of operands {}".format(
                len(dtypeList), num_operands
            )

        try:
            qgen = op["qgen"]
        except KeyError:
            qgen = None

        # Build the random tensor operands and the test
        tens = []

        tens = self.generate_tensors(op, dtypeList, shapeList, testArgs, error_name)

        if qgen is not None:
            qinfo = qgen(self, op, dtype_or_dtypeList, error_name)
        else:
            qinfo = None

        try:
            if error_if_validators is None:
                if qinfo is not None:
                    resultName = build_fcn(self, op, *tens, *testArgs, qinfo)
                else:
                    resultName = build_fcn(self, op, *tens, *testArgs)
            else:
                if qinfo is not None:
                    resultName = build_fcn(self, op, *tens, *testArgs, validator_fcns=error_if_validators, error_name=error_name, qinfo=qinfo)
                else:
                    resultName = build_fcn(self, op, *tens, *testArgs, validator_fcns=error_if_validators, error_name=error_name)
        except TypeError as e:
            print(
                "build_fcn: {}\nTensors: {}\nArgs: {}\n".format(
                    build_fcn, tens, testArgs
                )
            )
            raise e

        if resultName is None:
            print("Invalid ERROR_IF tests created")

        # Save the serialized test
        self.serialize("test")


    def generate_tensors(self, op, dtypeList, shapeList, testArgs, error_name=None):
        pCount, cCount = op["operands"]

        tens = []
        if (op["op"] == Op.ADD or op["op"] == Op.SUB) and dtypeList[0] == DType.INT32 and error_name == None:
            # Make sure the operation does not cause value saturation - where
            # the number wraps due to limited number of bits to store the answer
            assert (
                pCount == 2 and cCount == 0
            ), "Op.ADD / Op.SUB must have 2 placeholders, 0 consts"
            placeholders = []
            add = (op["op"] == Op.ADD)
            a_arr = self.getRandTensor(shapeList[0], dtypeList[0])
            b_arr = self.getRandTensor(shapeList[1], dtypeList[1])
            if add:
                res_arr = np.add(a_arr, b_arr, dtype=np.int64)
            else:
                res_arr = np.subtract(a_arr, b_arr, dtype=np.int64)

            # Work out the saturation limits
            max_i32 = (1 << 31)-1
            min_i32 = -(1 << 31)
            max_arr = np.full(shapeList[1], max_i32)
            min_arr = np.full(shapeList[1], min_i32)

            # Find how much values exceed the maximum/minimums
            sat_max_arr = np.maximum(res_arr - max_arr, 0)
            sat_min_arr = np.minimum(res_arr - min_arr, 0)

            if not add:
                # Swap saturation values and negate values as we need to perform opposite operations
                sat_max_arr, sat_min_arr = -sat_min_arr, -sat_max_arr

            # Create new array of unsaturated values by clipping values as needed
            b_unsat_arr = b_arr
            if (sat_max_arr != 0).any():
                # Clip values that cause saturation
                b_unsat_arr = np.subtract(b_unsat_arr, sat_max_arr, dtype=np.int32)
                # Reduce axes in unsaturated tensor to match original tensor
                for axis, dim in enumerate(b_arr.shape):
                    if dim != b_unsat_arr.shape[axis]:
                        assert ( dim == 1 ), "Op.ADD / SUB dimension must be 1 or matching to be broadcastable"
                        b_unsat_arr = np.amin(b_unsat_arr, axis=axis, keepdims=True)

            if (sat_min_arr != 0).any():
                # Clip values that cause saturation
                b_unsat_arr = np.subtract(b_unsat_arr, sat_min_arr, dtype=np.int32)
                # Reduce axes in unsaturated tensor to match original tensor
                for axis, dim in enumerate(b_arr.shape):
                    if dim != b_unsat_arr.shape[axis]:
                        assert ( dim == 1 ), "Op.ADD / SUB dimension must be 1 or matching to be broadcastable"
                        b_unsat_arr = np.amax(b_unsat_arr, axis=axis, keepdims=True)

            placeholders.append(
                self.ser.addPlaceholder(shapeList[0], dtypeList[0], a_arr)
            )
            placeholders.append(
                self.ser.addPlaceholder(shapeList[1], dtypeList[1], b_unsat_arr)
            )

            tens.extend(placeholders)
        elif (op["op"] == Op.COND_IF or op["op"] == Op.WHILE_LOOP) and dtypeList[0] == DType.INT32:
            # Limit input tensors with cond_if_binary or while_loop to stop
            # saturation of add/sub ops
            pRemain = pCount
            placeholders = []
            for idx, shape in  enumerate(shapeList[:]):
                arr = self.getRandTensor(shapeList[idx], DType.INT16)
                if pRemain > 0:
                    placeholders.append(self.ser.addPlaceholder(shape, dtypeList[idx], arr))
                    pRemain -= 1
                else:
                    placeholders.append(self.ser.addConst(shape, dtypeList[idx], arr))

            tens.extend(placeholders)
        elif op["op"] == Op.ARITHMETIC_RIGHT_SHIFT:
            # Force value of operand[1] to be within [0, num_bits]
            assert (
                pCount == 2 and cCount == 0
            ), "Op.ArithmeticRightShift must have 2 placeholders, 0 consts"

            placeholders = []
            for idx, shape in enumerate(shapeList[:]):
                if idx == 1:
                    if dtypeList[idx] == DType.INT8:
                        arr = np.int32(self.rng.integers(low=0, high=8, size=shape))
                    elif dtypeList[idx] == DType.INT16:
                        arr = np.int32(self.rng.integers(low=0, high=16, size=shape))
                    elif dtypeList[idx] == DType.INT32:
                        arr = np.int32(self.rng.integers(low=0, high=32, size=shape))
                    elif error_name == ErrorIf.WrongInputType:
                        arr = np.int32(self.rng.integers(low=0, high=8, size=shape))
                    else:
                        raise Exception("OpArithmeticRightShift: invalid input dtype")
                else:
                    arr = self.getRandTensor(shape, dtypeList[idx])
                placeholders.append(self.ser.addPlaceholder(shape, dtypeList[idx], arr))

            tens.extend(placeholders)
        elif op["op"] == Op.SELECT:
            # Set datatype of condition tensor to boolean
            dtypeList[0] = DType.BOOL
            tens.extend(
                self.buildPlaceholderTensors(shapeList[0:pCount], dtypeList[0:pCount])
            )
            tens.extend(self.buildConstTensors(shapeList[pCount:], dtypeList[pCount:]))
        elif op["op"] == Op.INTDIV  and error_name == None:
            assert (
                pCount == 2 and cCount == 0
            ), "Op.INTDIV must have 2 placeholders, 0 consts"

            placeholders = []

            # Two invalid cases for Op.INTDIV:
            # 1. divisor == 0
            # 2. dividend == -(1<<31) and divisor == -1
            while True:
                dividend_arr = self.getRandTensor(shapeList[0], dtypeList[0])
                divisor_arr = self.getRandTensor(shapeList[1], dtypeList[1])

                if (divisor_arr == 0).any():
                    continue

                if (dividend_arr == -(2 ** 31)).any() and (divisor_arr == -1).any():
                    continue

                break

            placeholders.append(
                self.ser.addPlaceholder(shapeList[0], dtypeList[0], dividend_arr)
            )
            placeholders.append(
                self.ser.addPlaceholder(shapeList[1], dtypeList[1], divisor_arr)
            )

            tens.extend(placeholders)
        elif op["op"] == Op.MUL:
            assert (
                pCount == 2 and cCount == 0
            ), "Op.MUL must have 2 placeholders, 0 consts"

            if dtypeList[0] == DType.FLOAT:
                tens.extend(self.buildPlaceholderTensors(shapeList[:], dtypeList[:]))
            else:
                placeholders = []

                # Make sure multiply result in int32 range
                shift = testArgs[0]
                if dtypeList[0] == DType.INT8:
                    num_bits = 8
                elif dtypeList[0] == DType.INT16:
                    num_bits = 16
                elif dtypeList[0] == DType.INT32:
                    num_bits = 32
                elif error_name == ErrorIf.WrongInputType:
                    num_bits = 8
                else:
                    raise Exception("OpMul: invalid input dtype")

                for idx, shape in enumerate(shapeList[:]):
                    low = -(2 ** (num_bits - 1))
                    high = (2 ** (num_bits - 1)) - 1

                    a_arr = np.int32(
                        self.rng.integers(low=low, high=high, size=shapeList[0])
                    )
                    b_arr = np.int32(
                        self.rng.integers(low=low, high=high, size=shapeList[1])
                    )

                i = 0
                while True:

                    a_arr_64 = a_arr.astype(np.int64)
                    b_arr_64 = b_arr.astype(np.int64)

                    if shift > 0:
                        rounding = 1 << (shift - 1)
                        result_arr = ((a_arr_64 * b_arr_64) + rounding) >> shift
                    else:
                        result_arr = a_arr_64 * b_arr_64

                    if (result_arr > -(2 ** 31)).all() and (
                        result_arr <= ((2 ** 31) - 1)
                    ).all():
                        break

                    i = i + 1
                    a_arr = a_arr // 2
                    b_arr = b_arr // 2

                placeholders.append(
                    self.ser.addPlaceholder(shapeList[0], dtypeList[0], a_arr)
                )
                placeholders.append(
                    self.ser.addPlaceholder(shapeList[1], dtypeList[1], b_arr)
                )

                tens.extend(placeholders)
        elif op["op"] == Op.CONCAT:
            count = len(shapeList) - self.args.num_const_inputs_concat
            if count < 1:
                count = 1
            if self.args.num_const_inputs_concat == 0:
                count = len(shapeList)

            # Ensure axis is an int
            testArgs[0] = int(testArgs[0])

            shapeList = TosaTensorGen.tgConcatConstInput(self, shapeList, testArgs[0], error_name)

            tens.extend(
                self.buildPlaceholderTensors(shapeList[0:count], dtypeList[0:count])
            )
            tens.extend(self.buildConstTensors(shapeList[count:], dtypeList[count:]))
        else:
            tens.extend(
                self.buildPlaceholderTensors(shapeList[0:pCount], dtypeList[0:pCount])
            )
            tens.extend(self.buildConstTensors(shapeList[pCount:], dtypeList[pCount:]))

        return tens

    def createDynamicOpLists(self):

        # Dynamically create op lists for convolutions with a list of kernel sizes
        KERNELS_2D = [[1, 1], [2, 2], [3, 3], [5, 5], [3, 1], [1, 3]]

        for k in KERNELS_2D:
            testName = "conv2d_{}x{}".format(k[0], k[1])
            self.TOSA_OP_LIST[testName] = self.TOSA_OP_LIST["conv2d_TEMPLATE"].copy()
            self.TOSA_OP_LIST[testName]["filter"] = k
            self.TOSA_OP_LIST[testName]["template"] = False

            testName = "depthwise_conv2d_{}x{}".format(k[0], k[1])
            self.TOSA_OP_LIST[testName] = self.TOSA_OP_LIST[
                "depthwise_conv2d_TEMPLATE"
            ].copy()
            self.TOSA_OP_LIST[testName]["filter"] = k
            self.TOSA_OP_LIST[testName]["template"] = False

            testName = "transpose_conv2d_{}x{}".format(k[0], k[1])
            self.TOSA_OP_LIST[testName] = self.TOSA_OP_LIST[
                "transpose_conv2d_TEMPLATE"
            ].copy()
            self.TOSA_OP_LIST[testName]["filter"] = k
            self.TOSA_OP_LIST[testName]["template"] = False

        KERNELS_3D = [[1, 1, 1], [2, 1, 1], [1, 2, 1], [1, 1, 2]]
        for k in KERNELS_3D:
            testName = "conv3d_{}x{}x{}".format(k[0], k[1], k[2])
            self.TOSA_OP_LIST[testName] = self.TOSA_OP_LIST["conv3d_TEMPLATE"].copy()
            self.TOSA_OP_LIST[testName]["filter"] = k
            self.TOSA_OP_LIST[testName]["template"] = False

        # Delete any templates after having created any dynamic ops
        # This is a two-pass operation because it's bad practice to delete
        # keys from dictionaries while iterating
        keyList = []
        for k in self.TOSA_OP_LIST:
            try:
                if self.TOSA_OP_LIST[k]["template"] == True:
                    keyList.append(k)
                    continue
            except KeyError:
                pass

        for k in keyList:
            del self.TOSA_OP_LIST[k]

    def initOpListDefaults(self):
        """Fill in default fields for ops if they aren't already specified.
        Look for missing required fields (datastructure linting)."""
        for op in self.TOSA_OP_LIST:

            # Required fields
            try:
                pl, c = self.TOSA_OP_LIST[op]["operands"]
            except (KeyError, ValueError, TypeError):
                raise Exception(
                    "Op {} is missing a valid operand tuple in TOSA_OP_LIST".format(op)
                )

            try:
                fcn, tgen, arggen = self.TOSA_OP_LIST[op]["build_fcn"]
            except (KeyError, ValueError, TypeError):
                raise Exception(
                    "Op {} is missing a valid build_fcn tuple in TOSA_OP_LIST".format(
                        op
                    )
                )

            try:
                types = self.TOSA_OP_LIST[op]["types"]
            except KeyError as e:
                raise Exception(
                    "Op {} is missing a valid type list in TOSA_OP_LIST".format(op)
                )

            try:
                opcode = self.TOSA_OP_LIST[op]["op"]
            except KeyError as e:
                raise Exception(
                    "Op {} is missing the Op field in TOSA_OP_LIST".format(op)
                )

            # Put in default rank range, if missing
            try:
                rank = self.TOSA_OP_LIST[op]["rank"]
            except KeyError:
                self.TOSA_OP_LIST[op]["rank"] = self.DEFAULT_RANK_RANGE

    # Tensor operator list
    #  'op': op name
    #  'operands': tuple of (placeholder, const) operands
    #  'rank': optional, restricts rank to tuple inclusive of (min, max),
    #    if not specified, defaults to (1, 4)
    #  'build_fcn': tuple of the function to (build_operator(), TensorGen function, ArgGen enum)
    #  'types': array of datatypes to be tested
    TYPE_FP = [DType.FLOAT]

    TYPE_INT = [DType.INT8, DType.INT16, DType.INT32]  # Excludes INT4
    TYPE_INT_FP = [DType.INT8, DType.INT16, DType.INT32, DType.FLOAT]  # Excludes INT4

    TYPE_BOOL = [DType.BOOL]
    TYPE_FI32 = [DType.FLOAT, DType.INT32]
    TYPE_FIB = [DType.FLOAT, DType.INT8, DType.INT16, DType.INT32, DType.BOOL]
    TYPE_FI16 = [DType.FLOAT, DType.INT16]

    TYPE_NARROW_INT_FP = [DType.INT8, DType.INT16, DType.FLOAT]

    TYPE_CONV = [
        [DType.INT8, DType.INT4, DType.INT32],
        [DType.INT8, DType.INT8, DType.INT32],
        [DType.INT16, DType.INT8, DType.INT48],
        DType.FLOAT,
    ]

    DEFAULT_RANK_RANGE = (1, TOSA_TENSOR_MAX_RANK)

    TOSA_OP_LIST = {
        # Tensor operators
        "argmax": {
            "op": Op.ARGMAX,
            "operands": (1, 0),
            "rank": (1, 4),
            "build_fcn": (build_argmax, TosaTensorGen.tgBasic, TosaArgGen.agAxis),
            "types": TYPE_NARROW_INT_FP,
            "error_if_validators": (TosaErrorValidator.evAxisSmallerZero, TosaErrorValidator.evAxisLargerRank, TosaErrorValidator.evArgmaxOutputRankMismatch,
            TosaErrorValidator.evArgmaxOutputShapeMismatch, TosaErrorValidator.evWrongRank, TosaErrorValidator.evWrongInputType,
            TosaErrorValidator.evWrongOutputType, TosaErrorValidator.evWrongInputList, TosaErrorValidator.evWrongOutputList)
        },
        "avg_pool2d": {
            "op": Op.AVG_POOL2D,
            "operands": (1, 0),
            "rank": (4, 4),
            "build_fcn": (build_pool2d, TosaTensorGen.tgNHWC, TosaArgGen.agPooling),
            "qgen": TosaQuantGen.qgUnary,
            "types": TYPE_NARROW_INT_FP,
            "invalid_test_validators": (TosaInvalidValidator.ivHeightWidthSmallerZero,),
            "error_if_validators": (TosaErrorValidator.evKernelSmallerOne, TosaErrorValidator.evStrideSmallerOne, TosaErrorValidator.evPadSmallerZero,
            TosaErrorValidator.evWrongRank, TosaErrorValidator.evWrongInputType, TosaErrorValidator.evWrongOutputType, TosaErrorValidator.evWrongInputList,
            TosaErrorValidator.evWrongOutputList, TosaErrorValidator.evInputZeroPointNotZero, TosaErrorValidator.evOutputZeroPointNotZero,
            TosaErrorValidator.evPadLargerEqualKernel, TosaErrorValidator.evPoolingOutputShapeMismatch)
        },
        # Templated operator.  Filled in by createDynamicOpLists
        "conv2d_TEMPLATE": {
            "op": Op.CONV2D,
            "operands": (1, 2),
            "rank": (4, 4),
            "build_fcn": (build_conv2d, TosaTensorGen.tgConv2D, TosaArgGen.agConv),
            "qgen": TosaQuantGen.qgConv,
            "types": TYPE_CONV,
            "invalid_test_validators": (TosaInvalidValidator.ivHeightWidthSmallerZero,),
            "template": True,
        },
        # Templated operator.  Filled in by createDynamicOpLists
        "conv3d_TEMPLATE": {
            "op": Op.CONV3D,
            "operands": (1, 2),
            "rank": (5, 5),
            "build_fcn": (build_conv3d, TosaTensorGen.tgConv3D, TosaArgGen.agConv),
            "qgen": TosaQuantGen.qgConv,
            "types": TYPE_CONV,
            "template": True,
        },
        # Templated operator.  Filled in by createDynamicOpLists
        "depthwise_conv2d_TEMPLATE": {
            "op": Op.DEPTHWISE_CONV2D,
            "operands": (1, 2),
            "filter": [1, 1],
            "rank": (4, 4),
            "build_fcn": (
                build_depthwise_conv2d,
                TosaTensorGen.tgDepthwiseConv2D,
                TosaArgGen.agConv,
            ),
            "qgen": TosaQuantGen.qgConv,
            "types": TYPE_CONV,
            "invalid_test_validators": (TosaInvalidValidator.ivHeightWidthSmallerZero,),
            "template": True,
        },
        "fully_connected": {
            "op": Op.FULLY_CONNECTED,
            "operands": (1, 2),
            "rank": (2, 2),
            "build_fcn": (build_fully_connected, TosaTensorGen.tgFullyConnected, None),
            "qgen": TosaQuantGen.qgConv,
            "types": TYPE_CONV,
            "error_if_validators": (TosaErrorValidator.evInputZeroPointNotZero, TosaErrorValidator.evWeightZeroPointNotZero, TosaErrorValidator.evWrongRank,
            TosaErrorValidator.evWrongInputType, TosaErrorValidator.evWrongOutputType, TosaErrorValidator.evWrongInputList, TosaErrorValidator.evWrongOutputList)
        },
        "matmul": {
            "op": Op.MATMUL,
            "operands": (2, 0),
            "rank": (3, 3),
            "build_fcn": (build_matmul, TosaTensorGen.tgMatmul, None),
            "qgen": TosaQuantGen.qgMatmul,
            "types": TYPE_NARROW_INT_FP,
            "error_if_validators": (TosaErrorValidator.evInputZeroPointNotZero, TosaErrorValidator.evWrongRank, TosaErrorValidator.evWrongInputType,
            TosaErrorValidator.evWrongOutputType, TosaErrorValidator.evWrongInputList, TosaErrorValidator.evWrongOutputList)
        },
        "max_pool2d": {
            "op": Op.MAX_POOL2D,
            "operands": (1, 0),
            "rank": (4, 4),
            "build_fcn": (build_pool2d, TosaTensorGen.tgNHWC, TosaArgGen.agPooling),
            "types": TYPE_NARROW_INT_FP,
            "invalid_test_validators": (TosaInvalidValidator.ivHeightWidthSmallerZero,),
            "error_if_validators": (TosaErrorValidator.evKernelSmallerOne, TosaErrorValidator.evStrideSmallerOne, TosaErrorValidator.evPadSmallerZero,
            TosaErrorValidator.evWrongRank, TosaErrorValidator.evWrongInputType, TosaErrorValidator.evWrongOutputType, TosaErrorValidator.evWrongInputList,
            TosaErrorValidator.evWrongOutputList, TosaErrorValidator.evPadLargerEqualKernel, TosaErrorValidator.evPoolingOutputShapeMismatch)
        },
        # Templated operator.  Filled in by createDynamicOpLists
        "transpose_conv2d_TEMPLATE": {
            "op": Op.TRANSPOSE_CONV2D,
            "operands": (1, 2),
            "rank": (4, 4),
            "build_fcn": (
                build_transpose_conv2d,
                TosaTensorGen.tgTransposeConv2D,
                TosaArgGen.agTransposeConv2D,
            ),
            "qgen": TosaQuantGen.qgConv,
            "types": TYPE_CONV,
            "invalid_test_validators": (TosaInvalidValidator.ivNonPositiveOutputShape,),
            "template": True,
        },
        # Activation functions
        "clamp": {
            "op": Op.CLAMP,
            "operands": (1, 0),
            "build_fcn": (build_clamp, TosaTensorGen.tgBasic, None),
            "types": TYPE_NARROW_INT_FP,
            "error_if_validators": (TosaErrorValidator.evMaxSmallerMin, TosaErrorValidator.evWrongInputType, TosaErrorValidator.evWrongOutputType,
            TosaErrorValidator.evWrongInputList, TosaErrorValidator.evWrongOutputList)
        },
        "sigmoid": {
            "op": Op.SIGMOID,
            "operands": (1, 0),
            "build_fcn": (build_sigmoid, TosaTensorGen.tgBasic, None),
            "types": TYPE_FP,
            "error_if_validators": (TosaErrorValidator.evWrongInputType, TosaErrorValidator.evWrongOutputType, TosaErrorValidator.evWrongInputList,
            TosaErrorValidator.evWrongOutputList)
        },
        "tanh": {
            "op": Op.TANH,
            "operands": (1, 0),
            "build_fcn": (build_tanh, TosaTensorGen.tgBasic, None),
            "types": TYPE_FP,
            "error_if_validators": (TosaErrorValidator.evWrongInputType, TosaErrorValidator.evWrongOutputType, TosaErrorValidator.evWrongInputList,
            TosaErrorValidator.evWrongOutputList)
        },
        # Elementwise Binary Operators
        "add": {
            "op": Op.ADD,
            "operands": (2, 0),
            "build_fcn": (build_binary_broadcast, TosaTensorGen.tgBroadcastFuzz, None),
            "types": TYPE_FI32,
            "error_if_validators": (TosaErrorValidator.evRankMismatch, TosaErrorValidator.evWrongInputType, TosaErrorValidator.evWrongOutputType,
            TosaErrorValidator.evWrongInputList, TosaErrorValidator.evWrongOutputList)
        },
        "arithmetic_right_shift": {
            "op": Op.ARITHMETIC_RIGHT_SHIFT,
            "operands": (2, 0),
            "build_fcn": (
                build_arithmetic_right_shift,
                TosaTensorGen.tgBroadcastFuzz,
                TosaArgGen.agArithmeticRightShift,
            ),
            "types": TYPE_INT,
            "error_if_validators": (TosaErrorValidator.evWrongInputType, TosaErrorValidator.evWrongOutputType, TosaErrorValidator.evWrongInputList,
            TosaErrorValidator.evWrongOutputList)
        },
        "bitwise_and": {
            "op": Op.BITWISE_AND,
            "operands": (2, 0),
            "build_fcn": (build_binary_broadcast, TosaTensorGen.tgBroadcastFuzz, None),
            "types": TYPE_INT,
            "error_if_validators": (TosaErrorValidator.evRankMismatch, TosaErrorValidator.evWrongInputType, TosaErrorValidator.evWrongOutputType,
            TosaErrorValidator.evWrongInputList, TosaErrorValidator.evWrongOutputList)
        },
        "bitwise_or": {
            "op": Op.BITWISE_OR,
            "operands": (2, 0),
            "build_fcn": (build_binary_broadcast, TosaTensorGen.tgBroadcastFuzz, None),
            "types": TYPE_INT,
            "error_if_validators": (TosaErrorValidator.evRankMismatch, TosaErrorValidator.evWrongInputType, TosaErrorValidator.evWrongOutputType,
            TosaErrorValidator.evWrongInputList, TosaErrorValidator.evWrongOutputList)
        },
        "bitwise_xor": {
            "op": Op.BITWISE_XOR,
            "operands": (2, 0),
            "build_fcn": (build_binary_broadcast, TosaTensorGen.tgBroadcastFuzz, None),
            "types": TYPE_INT,
            "error_if_validators": (TosaErrorValidator.evRankMismatch, TosaErrorValidator.evWrongInputType, TosaErrorValidator.evWrongOutputType,
            TosaErrorValidator.evWrongInputList, TosaErrorValidator.evWrongOutputList)
        },
        "intdiv": {
            "op": Op.INTDIV,
            "operands": (2, 0),
            "build_fcn": (build_binary_broadcast, TosaTensorGen.tgBroadcastFuzz, None),
            "types": [DType.INT32],
            "error_if_validators": (TosaErrorValidator.evRankMismatch, TosaErrorValidator.evWrongInputType, TosaErrorValidator.evWrongOutputType,
            TosaErrorValidator.evWrongInputList, TosaErrorValidator.evWrongOutputList)
        },
        "logical_and": {
            "op": Op.LOGICAL_AND,
            "operands": (2, 0),
            "build_fcn": (build_binary_broadcast, TosaTensorGen.tgBroadcastFuzz, None),
            "types": TYPE_BOOL,
            "error_if_validators": (TosaErrorValidator.evRankMismatch, TosaErrorValidator.evWrongInputType, TosaErrorValidator.evWrongOutputType,
            TosaErrorValidator.evWrongInputList, TosaErrorValidator.evWrongOutputList)
        },
        "logical_left_shift": {
            "op": Op.LOGICAL_LEFT_SHIFT,
            "operands": (2, 0),
            "build_fcn": (build_binary_broadcast, TosaTensorGen.tgBroadcastFuzz, None),
            "types": TYPE_INT,
            "error_if_validators": (TosaErrorValidator.evRankMismatch, TosaErrorValidator.evWrongInputType, TosaErrorValidator.evWrongOutputType,
            TosaErrorValidator.evWrongInputList, TosaErrorValidator.evWrongOutputList)
        },
        "logical_right_shift": {
            "op": Op.LOGICAL_RIGHT_SHIFT,
            "operands": (2, 0),
            "build_fcn": (build_binary_broadcast, TosaTensorGen.tgBroadcastFuzz, None),
            "types": TYPE_INT,
            "error_if_validators": (TosaErrorValidator.evRankMismatch, TosaErrorValidator.evWrongInputType, TosaErrorValidator.evWrongOutputType,
            TosaErrorValidator.evWrongInputList, TosaErrorValidator.evWrongOutputList)
        },
        "logical_or": {
            "op": Op.LOGICAL_OR,
            "operands": (2, 0),
            "build_fcn": (build_binary_broadcast, TosaTensorGen.tgBroadcastFuzz, None),
            "types": TYPE_BOOL,
            "error_if_validators": (TosaErrorValidator.evRankMismatch, TosaErrorValidator.evWrongInputType, TosaErrorValidator.evWrongOutputType,
            TosaErrorValidator.evWrongInputList, TosaErrorValidator.evWrongOutputList)
        },
        "logical_xor": {
            "op": Op.LOGICAL_XOR,
            "operands": (2, 0),
            "build_fcn": (build_binary_broadcast, TosaTensorGen.tgBroadcastFuzz, None),
            "types": TYPE_BOOL,
            "error_if_validators": (TosaErrorValidator.evRankMismatch, TosaErrorValidator.evWrongInputType, TosaErrorValidator.evWrongOutputType,
            TosaErrorValidator.evWrongInputList, TosaErrorValidator.evWrongOutputList)
        },
        "maximum": {
            "op": Op.MAXIMUM,
            "operands": (2, 0),
            "build_fcn": (build_binary_broadcast, TosaTensorGen.tgBroadcastFuzz, None),
            "types": TYPE_FI32,
            "error_if_validators": (TosaErrorValidator.evRankMismatch, TosaErrorValidator.evWrongInputType, TosaErrorValidator.evWrongOutputType,
            TosaErrorValidator.evWrongInputList, TosaErrorValidator.evWrongOutputList)
        },
        "minimum": {
            "op": Op.MINIMUM,
            "operands": (2, 0),
            "build_fcn": (build_binary_broadcast, TosaTensorGen.tgBroadcastFuzz, None),
            "types": TYPE_FI32,
            "error_if_validators": (TosaErrorValidator.evRankMismatch, TosaErrorValidator.evWrongInputType, TosaErrorValidator.evWrongOutputType,
            TosaErrorValidator.evWrongInputList, TosaErrorValidator.evWrongOutputList)
        },
        "mul": {
            "op": Op.MUL,
            "operands": (2, 0),
            "build_fcn": (build_mul, TosaTensorGen.tgBroadcastFuzz, TosaArgGen.agMul),
            "types": TYPE_INT_FP,
            "error_if_validators": (TosaErrorValidator.evWrongInputType, TosaErrorValidator.evWrongOutputType, TosaErrorValidator.evWrongInputList,
            TosaErrorValidator.evWrongOutputList)
        },
        "pow": {
            "op": Op.POW,
            "operands": (2, 0),
            "build_fcn": (build_binary_broadcast, TosaTensorGen.tgBasic, None),
            "types": TYPE_FP,
            "error_if_validators": (TosaErrorValidator.evRankMismatch, TosaErrorValidator.evWrongInputType, TosaErrorValidator.evWrongOutputType,
            TosaErrorValidator.evWrongInputList, TosaErrorValidator.evWrongOutputList)
        },
        "sub": {
            "op": Op.SUB,
            "operands": (2, 0),
            "build_fcn": (build_binary_broadcast, TosaTensorGen.tgBroadcastFuzz, None),
            "types": TYPE_FI32,
            "error_if_validators": (TosaErrorValidator.evRankMismatch, TosaErrorValidator.evWrongInputType, TosaErrorValidator.evWrongOutputType,
            TosaErrorValidator.evWrongInputList, TosaErrorValidator.evWrongOutputList)
        },
        "table": {
            "op": Op.TABLE,
            # Use the automatic generation functions to create the input array
            # but create the table tensor in the build function, as it may be
            # a different type from the input
            "operands": (1, 0),
            "build_fcn": (build_table, TosaTensorGen.tgBasic, TosaArgGen.agTable),
            "types": [DType.INT8, DType.INT16],
            "error_if_validators": (TosaErrorValidator.evWrongInputType, TosaErrorValidator.evWrongOutputType, TosaErrorValidator.evWrongInputList,
            TosaErrorValidator.evWrongOutputList)
        },
        # Elementwise Unary operators
        "abs": {
            "op": Op.ABS,
            "operands": (1, 0),
            "build_fcn": (build_unary, TosaTensorGen.tgBasic, None),
            "types": TYPE_FI32,
            "error_if_validators": (TosaErrorValidator.evWrongInputType, TosaErrorValidator.evWrongOutputType,
            TosaErrorValidator.evWrongInputList, TosaErrorValidator.evWrongOutputList)
        },
        "bitwise_not": {
            "op": Op.BITWISE_NOT,
            "operands": (1, 0),
            "build_fcn": (build_unary, TosaTensorGen.tgBasic, None),
            "types": TYPE_INT,
            "error_if_validators": (TosaErrorValidator.evWrongInputType, TosaErrorValidator.evWrongOutputType,
            TosaErrorValidator.evWrongInputList, TosaErrorValidator.evWrongOutputList)
        },
        "ceil": {
            "op": Op.CEIL,
            "operands": (1, 0),
            "build_fcn": (build_unary, TosaTensorGen.tgBasic, None),
            "types": TYPE_FP,
            "error_if_validators": (TosaErrorValidator.evWrongInputType, TosaErrorValidator.evWrongOutputType,
            TosaErrorValidator.evWrongInputList, TosaErrorValidator.evWrongOutputList)
        },
        "clz": {
            "op": Op.CLZ,
            "operands": (1, 0),
            "build_fcn": (build_unary, TosaTensorGen.tgBasic, None),
            "types": [DType.INT32],
            "error_if_validators": (TosaErrorValidator.evWrongInputType, TosaErrorValidator.evWrongOutputType,
            TosaErrorValidator.evWrongInputList, TosaErrorValidator.evWrongOutputList)
        },
        "exp": {
            "op": Op.EXP,
            "operands": (1, 0),
            "build_fcn": (build_unary, TosaTensorGen.tgBasic, None),
            "types": TYPE_FP,
            "error_if_validators": (TosaErrorValidator.evWrongInputType, TosaErrorValidator.evWrongOutputType,
            TosaErrorValidator.evWrongInputList, TosaErrorValidator.evWrongOutputList)
        },
        "floor": {
            "op": Op.FLOOR,
            "operands": (1, 0),
            "build_fcn": (build_unary, TosaTensorGen.tgBasic, None),
            "types": TYPE_FP,
            "error_if_validators": (TosaErrorValidator.evWrongInputType, TosaErrorValidator.evWrongOutputType,
            TosaErrorValidator.evWrongInputList, TosaErrorValidator.evWrongOutputList)
        },
        "log": {
            "op": Op.LOG,
            "operands": (1, 0),
            "build_fcn": (build_unary, TosaTensorGen.tgBasic, None),
            "types": TYPE_FP,
            "error_if_validators": (TosaErrorValidator.evWrongInputType, TosaErrorValidator.evWrongOutputType,
            TosaErrorValidator.evWrongInputList, TosaErrorValidator.evWrongOutputList)
        },
        "logical_not": {
            "op": Op.LOGICAL_NOT,
            "operands": (1, 0),
            "build_fcn": (build_unary, TosaTensorGen.tgBasic, None),
            "types": TYPE_BOOL,
            "error_if_validators": (TosaErrorValidator.evWrongInputType, TosaErrorValidator.evWrongOutputType,
            TosaErrorValidator.evWrongInputList, TosaErrorValidator.evWrongOutputList)
        },
        "negate": {
            "op": Op.NEGATE,
            "operands": (1, 0),
            "build_fcn": (build_unary, TosaTensorGen.tgBasic, None),
            "qgen": TosaQuantGen.qgUnary,
            "types": TYPE_INT_FP,
            "error_if_validators": (TosaErrorValidator.evInputZeroPointNotZero, TosaErrorValidator.evOutputZeroPointNotZero,
            TosaErrorValidator.evWrongInputType, TosaErrorValidator.evWrongOutputType, TosaErrorValidator.evWrongInputList,
            TosaErrorValidator.evWrongOutputList)
        },
        "reciprocal": {
            "op": Op.RECIPROCAL,
            "operands": (1, 0),
            "build_fcn": (build_unary, TosaTensorGen.tgBasic, None),
            "types": TYPE_FP,
            "error_if_validators": (TosaErrorValidator.evWrongInputType, TosaErrorValidator.evWrongOutputType,
            TosaErrorValidator.evWrongInputList, TosaErrorValidator.evWrongOutputList)
        },
        "rsqrt": {
            "op": Op.RSQRT,
            "operands": (1, 0),
            "build_fcn": (build_unary, TosaTensorGen.tgBasic, None),
            "types": TYPE_FP,
            "error_if_validators": (TosaErrorValidator.evWrongInputType, TosaErrorValidator.evWrongOutputType,
            TosaErrorValidator.evWrongInputList, TosaErrorValidator.evWrongOutputList)
        },
        # Elementwise Ternary operators
        "select": {
            "op": Op.SELECT,
            "operands": (3, 0),
            "build_fcn": (build_select, TosaTensorGen.tgBroadcastFuzz, None),
            "types": TYPE_FIB,
            "error_if_validators": (TosaErrorValidator.evWrongInputType, TosaErrorValidator.evWrongOutputType,
            TosaErrorValidator.evWrongInputList, TosaErrorValidator.evWrongOutputList)
        },
        # Comparison operators
        "equal": {
            "op": Op.EQUAL,
            "operands": (2, 0),
            "build_fcn": (build_comparison, TosaTensorGen.tgBroadcastFuzz, None),
            "types": TYPE_FI32,
            "error_if_validators": (TosaErrorValidator.evWrongInputType, TosaErrorValidator.evWrongOutputType,
            TosaErrorValidator.evWrongInputList, TosaErrorValidator.evWrongOutputList)
        },
        "greater_equal": {
            "op": Op.GREATER_EQUAL,
            "operands": (2, 0),
            "build_fcn": (build_comparison, TosaTensorGen.tgBroadcastFuzz, None),
            "types": TYPE_FI32,
            "error_if_validators": (TosaErrorValidator.evWrongInputType, TosaErrorValidator.evWrongOutputType,
            TosaErrorValidator.evWrongInputList, TosaErrorValidator.evWrongOutputList)
        },
        "greater": {
            "op": Op.GREATER,
            "operands": (2, 0),
            "build_fcn": (build_comparison, TosaTensorGen.tgBroadcastFuzz, None),
            "types": TYPE_FI32,
            "error_if_validators": (TosaErrorValidator.evWrongInputType, TosaErrorValidator.evWrongOutputType,
            TosaErrorValidator.evWrongInputList, TosaErrorValidator.evWrongOutputList)
        },
        # Reduction operators
        "reduce_all": {
            "op": Op.REDUCE_ALL,
            "operands": (1, 0),
            "build_fcn": (build_reduce, TosaTensorGen.tgBasic, TosaArgGen.agAxis),
            "types": TYPE_BOOL,
            "error_if_validators": (TosaErrorValidator.evAxisLargerRank, TosaErrorValidator.evAxisSmallerZero, TosaErrorValidator.evShapeOfAxisNotOne,
            TosaErrorValidator.evWrongInputType, TosaErrorValidator.evWrongOutputType, TosaErrorValidator.evWrongRank,
            TosaErrorValidator.evWrongInputList, TosaErrorValidator.evWrongOutputList)
        },
        "reduce_any": {
            "op": Op.REDUCE_ANY,
            "operands": (1, 0),
            "build_fcn": (build_reduce, TosaTensorGen.tgBasic, TosaArgGen.agAxis),
            "types": TYPE_BOOL,
            "error_if_validators": (TosaErrorValidator.evAxisLargerRank, TosaErrorValidator.evAxisSmallerZero, TosaErrorValidator.evShapeOfAxisNotOne,
            TosaErrorValidator.evWrongInputType, TosaErrorValidator.evWrongOutputType, TosaErrorValidator.evWrongRank,
            TosaErrorValidator.evWrongInputList, TosaErrorValidator.evWrongOutputList)
        },
        "reduce_max": {
            "op": Op.REDUCE_MAX,
            "operands": (1, 0),
            "build_fcn": (build_reduce, TosaTensorGen.tgBasic, TosaArgGen.agAxis),
            "types": TYPE_INT_FP,
            "error_if_validators": (TosaErrorValidator.evAxisLargerRank, TosaErrorValidator.evAxisSmallerZero, TosaErrorValidator.evShapeOfAxisNotOne,
            TosaErrorValidator.evWrongInputType, TosaErrorValidator.evWrongOutputType, TosaErrorValidator.evWrongRank,
            TosaErrorValidator.evWrongInputList, TosaErrorValidator.evWrongOutputList)
        },
        "reduce_min": {
            "op": Op.REDUCE_MAX,
            "operands": (1, 0),
            "build_fcn": (build_reduce, TosaTensorGen.tgBasic, TosaArgGen.agAxis),
            "types": TYPE_INT_FP,
            "error_if_validators": (TosaErrorValidator.evAxisLargerRank, TosaErrorValidator.evAxisSmallerZero, TosaErrorValidator.evShapeOfAxisNotOne,
            TosaErrorValidator.evWrongInputType, TosaErrorValidator.evWrongOutputType, TosaErrorValidator.evWrongRank,
            TosaErrorValidator.evWrongInputList, TosaErrorValidator.evWrongOutputList)
        },
        "reduce_product": {
            "op": Op.REDUCE_PRODUCT,
            "operands": (1, 0),
            "build_fcn": (build_reduce, TosaTensorGen.tgBasic, TosaArgGen.agAxis),
            "types": TYPE_FP,
            "error_if_validators": (TosaErrorValidator.evAxisLargerRank, TosaErrorValidator.evAxisSmallerZero, TosaErrorValidator.evShapeOfAxisNotOne,
            TosaErrorValidator.evWrongInputType, TosaErrorValidator.evWrongOutputType, TosaErrorValidator.evWrongRank,
            TosaErrorValidator.evWrongInputList, TosaErrorValidator.evWrongOutputList)
        },
        "reduce_sum": {
            "op": Op.REDUCE_SUM,
            "operands": (1, 0),
            "build_fcn": (build_reduce, TosaTensorGen.tgBasic, TosaArgGen.agAxis),
            "types": TYPE_FI32,
            "error_if_validators": (TosaErrorValidator.evAxisLargerRank, TosaErrorValidator.evAxisSmallerZero, TosaErrorValidator.evShapeOfAxisNotOne,
            TosaErrorValidator.evWrongInputType, TosaErrorValidator.evWrongOutputType, TosaErrorValidator.evWrongRank,
            TosaErrorValidator.evWrongInputList, TosaErrorValidator.evWrongOutputList)
        },
        # Data layout operators
        "concat": {
            "op": Op.CONCAT,
            "operands": (2, 0),
            "build_fcn": (build_concat, TosaTensorGen.tgConcat, TosaArgGen.agAxis),
            "types": TYPE_FIB,
            "error_if_validators": (TosaErrorValidator.evAxisLargerRank, TosaErrorValidator.evAxisSmallerZero, TosaErrorValidator.evConcatInputRankMismatch,
            TosaErrorValidator.evConcatShapeSumMismatch, TosaErrorValidator.evConcatInputDimMismatch, TosaErrorValidator.evWrongInputType,
            TosaErrorValidator.evWrongOutputType, TosaErrorValidator.evWrongOutputList)
        },
        "pad": {
            "op": Op.PAD,
            "operands": (1, 0),
            "rank": (1, 5),
            "build_fcn": (build_pad, TosaTensorGen.tgBasic, TosaArgGen.agPad),
            "qgen": TosaQuantGen.qgPad,
            "types": TYPE_FIB,
            "error_if_validators": (TosaErrorValidator.evInputZeroPointNotZero, TosaErrorValidator.evWrongInputType, TosaErrorValidator.evPadSmallerZero,
            TosaErrorValidator.evWrongOutputType, TosaErrorValidator.evWrongInputList, TosaErrorValidator.evWrongOutputList)
        },
        "reshape": {
            "op": Op.RESHAPE,
            "operands": (1, 0),
            "build_fcn": (build_reshape, TosaTensorGen.tgBasic, TosaArgGen.agReshape),
            "types": TYPE_FIB,
            "error_if_validators": (TosaErrorValidator.evTensorSizeInputOutputMismatch, TosaErrorValidator.evWrongInputType, TosaErrorValidator.evWrongOutputType,
            TosaErrorValidator.evWrongInputList, TosaErrorValidator.evWrongOutputList)
        },
        "reverse": {
            "op": Op.REVERSE,
            "operands": (1, 0),
            "build_fcn": (build_reverse, TosaTensorGen.tgBasic, TosaArgGen.agAxis),
            "types": TYPE_FIB,
            "error_if_validators": (TosaErrorValidator.evAxisSmallerZero, TosaErrorValidator.evAxisLargerRank, TosaErrorValidator.evWrongInputType,
            TosaErrorValidator.evWrongOutputType, TosaErrorValidator.evWrongInputList, TosaErrorValidator.evWrongOutputList)
        },
        "slice": {
            "op": Op.SLICE,
            "operands": (1, 0),
            "rank": (1, 4),
            "build_fcn": (build_slice, TosaTensorGen.tgBasic, TosaArgGen.agSlice),
            "types": TYPE_FIB,
            "error_if_validators": (TosaErrorValidator.evStartSmallerZero, TosaErrorValidator.evSizeSmallerEqualZero, TosaErrorValidator.evStartSizeOutsideBounds,
            TosaErrorValidator.evSizeOutputShapeMismatch, TosaErrorValidator.evInputSizeStartLengthMismatch, TosaErrorValidator.evWrongRank,
            TosaErrorValidator.evWrongInputType, TosaErrorValidator.evWrongOutputType, TosaErrorValidator.evWrongInputList, TosaErrorValidator.evWrongOutputList)
        },
        "tile": {
            "op": Op.TILE,
            "operands": (1, 0),
            "build_fcn": (build_tile, TosaTensorGen.tgBasic, TosaArgGen.agTile),
            "types": TYPE_FIB,
            "error_if_validators": (TosaErrorValidator.evWrongInputType, TosaErrorValidator.evWrongOutputType,
            TosaErrorValidator.evWrongInputList, TosaErrorValidator.evWrongOutputList)
        },
        "transpose": {
            "op": Op.TRANSPOSE,
            "operands": (1, 0),
            "rank": (1, 4),
            "build_fcn": (
                build_transpose,
                TosaTensorGen.tgBasic,
                TosaArgGen.agTranspose,
            ),
            "types": TYPE_FIB,
            "error_if_validators": (TosaErrorValidator.evIndexOutsideBounds, TosaErrorValidator.evIndexUsedTwice, TosaErrorValidator.evWrongRank,
            TosaErrorValidator.evWrongInputType, TosaErrorValidator.evWrongOutputType, TosaErrorValidator.evWrongInputList, TosaErrorValidator.evWrongOutputList)
        },
        # Data nodes
        "const": {
            "op": Op.CONST,
            "operands": (0, 1),
            "build_fcn": (build_const, TosaTensorGen.tgBasic, None),
            "types": TYPE_FIB,
        },
        "identity": {
            "op": Op.IDENTITY,
            "operands": (1, 0),
            "build_fcn": (build_unary, TosaTensorGen.tgBasic, None),
            "types": TYPE_FIB,
        },
        # Scatter/Gather
        "gather": {
            "op": Op.GATHER,
            # Only specify 'values' tensor here. 'indices' is generated in op building stage
            "operands": (1, 0),
            "rank": (3, 3),
            "build_fcn": (build_gather, TosaTensorGen.tgBasic, None),
            "types": TYPE_INT_FP,
            "error_if_validators": (TosaErrorValidator.evWrongInputType, TosaErrorValidator.evWrongOutputType,
            TosaErrorValidator.evWrongInputList, TosaErrorValidator.evWrongOutputList)
        },
        "scatter": {
            "op": Op.SCATTER,
            # Only specify 'values_in' tensor here.
            #'indices' and 'input' are generated in op building stage
            "operands": (2, 0),
            "rank": (3, 3),
            "build_fcn": (build_scatter, TosaTensorGen.tgScatter, None),
            "types": TYPE_INT_FP,
            "error_if_validators": (TosaErrorValidator.evWrongInputType, TosaErrorValidator.evWrongOutputType,
            TosaErrorValidator.evWrongInputList, TosaErrorValidator.evWrongOutputList)
        },
        # Image operations
        "resize": {
            "op": Op.RESIZE,
            "operands": (1, 0),
            "rank": (4, 4),
            "build_fcn": (build_resize, TosaTensorGen.tgNHWC, TosaArgGen.agResize),
            "types": [DType.INT8, DType.INT16, DType.FLOAT],
            "invalid_test_validators": (TosaInvalidValidator.ivWrongDataTypeOrModeResize, TosaInvalidValidator.ivBadStride),
            "error_if_validators": (TosaErrorValidator.evMaxDimExceeded, TosaErrorValidator.evStrideSmallerEqualZero, TosaErrorValidator.evStrideLargerDimension,
            TosaErrorValidator.evStrideLargerEqualMax, TosaErrorValidator.evOffsetSmallerEqualMin, TosaErrorValidator.evOffsetLargerEqualMax,
            TosaErrorValidator.evShiftNotZero, TosaErrorValidator.evShiftSmallerOne, TosaErrorValidator.evShiftLargerEleven, TosaErrorValidator.evWrongInputType,
            TosaErrorValidator.evWrongOutputType, TosaErrorValidator.evWrongRank, TosaErrorValidator.evWrongInputList, TosaErrorValidator.evWrongOutputList,
            TosaErrorValidator.evBatchMismatch, TosaErrorValidator.evChannelMismatch)
        },
        # Type conversion
        "cast": {
            "op": Op.CAST,
            "operands": (1, 0),
            "build_fcn": (build_cast, TosaTensorGen.tgBasic, TosaArgGen.agCast),
            "types": [DType.FLOAT, DType.INT8, DType.INT16, DType.INT32, DType.BOOL],
            "error_if_validators": (TosaErrorValidator.evWrongInputType, TosaErrorValidator.evWrongOutputType,
            TosaErrorValidator.evWrongInputList, TosaErrorValidator.evWrongOutputList)
        },
        "rescale": {
            "op": Op.RESCALE,
            "operands": (1, 0),
            "rank": (1,4),
            "build_fcn": (build_rescale, TosaTensorGen.tgBasic, TosaArgGen.agRescale),
            "types": [DType.UINT8, DType.INT8, DType.INT16, DType.INT32, DType.INT48],
            "error_if_validators": (TosaErrorValidator.evInputZeroPointNotZero, TosaErrorValidator.evOutputZeroPointNotZero, TosaErrorValidator.evScaleTrue,
            TosaErrorValidator.evScaleNotTrue, TosaErrorValidator.evWrongInputType, TosaErrorValidator.evWrongOutputType, TosaErrorValidator.evWrongRank,
            TosaErrorValidator.evWrongInputList, TosaErrorValidator.evWrongOutputList)
        },
        # Custom
        # Not implemented.
        # Control flow operators
        # Two varients of cond_if, one that generates one of two constant tensors (no
        # inputs to the basic blocks, one output) and another that either adds or subtracts two tensors
        # (two inputs to the basic blocks, one output)
        "cond_if_const": {
            "op": Op.COND_IF,
            "operands": (0, 2),
            "build_fcn": (
                build_cond_if_const,
                TosaTensorGen.tgBasic,
                TosaArgGen.agCondIf,
            ),
            "types": [DType.BOOL],
            "error_if_validators": (TosaErrorValidator.evOutputListThenGraphMismatch, TosaErrorValidator.evOutputListElseGraphMismatch)
        },
        "cond_if_binary": {
            "op": Op.COND_IF,
            "operands": (2, 0),
            "build_fcn": (
                build_cond_if_binary,
                TosaTensorGen.tgBasic,
                TosaArgGen.agCondIf,
            ),
            "types": TYPE_INT_FP,
            "error_if_validators": (TosaErrorValidator.evInputListThenGraphMismatch, TosaErrorValidator.evInputListElseGraphMismatch,
            TosaErrorValidator.evOutputListThenGraphMismatch, TosaErrorValidator.evOutputListElseGraphMismatch)
        },
        # while_loop
        "while_loop": {
            "op": Op.WHILE_LOOP,
            "operands": (0, 1),
            "build_fcn": (
                build_while_loop,
                TosaTensorGen.tgBasic,
                TosaArgGen.agWhileLoop,
            ),
            "types": [DType.INT32],
            "error_if_validators": (TosaErrorValidator.evInputListOutputListMismatch, TosaErrorValidator.evInputListCondGraphMismatch,
            TosaErrorValidator.evInputListBodyGraphInputMismatch, TosaErrorValidator.evInputListBodyGraphOutputMismatch,
            TosaErrorValidator.evCondGraphOutputNotMatchingBool)
        },
    }


class OutputShaper:
    # Methods in this class compute the expected output shape and datatype
    # for common classes of operations
    def __init__(self):
        pass

    # These methods return arguments that can be used for
    # creating a new output tensor
    @staticmethod
    def binaryBroadcastOp(ser, rng, a, b, error_name=None):
        if error_name != ErrorIf.RankMismatch:
            assert len(a.shape) == len(b.shape)
        assert a.dtype == b.dtype

        shape = []
        for i in range(len(a.shape)):
            if a.shape[i] == 1 and error_name == None:
                shape.append(b.shape[i])
            else:
                shape.append(a.shape[i])

        if error_name == ErrorIf.WrongOutputType:
            all_dtypes = [DType.INT8, DType.INT16, DType.INT32, DType.INT48, DType.FLOAT]
            wrong_dtypes = list(set(all_dtypes) - set([a.dtype]))
            outputDType = rng.choice(wrong_dtypes)
        else:
            outputDType = a.dtype

        return ser.addOutput(shape, outputDType)

    @staticmethod
    def binaryNonBroadcastOp(ser, a, b):
        assert len(a.shape) == len(b.shape)
        assert a.dtype == b.dtype

        shape = []
        for i in range(len(a.shape)):
            assert a.shape[i] == b.shape[i]
            shape.append(a.shape[i])

        return ser.addOutput(shape, a.dtype)

    @staticmethod
    def unaryOp(ser, rng, a, error_name=None):
        if error_name == ErrorIf.WrongOutputType:
            all_dtypes = [DType.INT8, DType.INT16, DType.INT32, DType.INT48, DType.FLOAT]
            wrong_dtypes = list(set(all_dtypes) - set([a.dtype]))
            outputDType = rng.choice(wrong_dtypes)
        else:
            outputDType = a.dtype

        return ser.addOutput(a.shape, outputDType)

    @staticmethod
    def selectOp(ser, rng, cond, a, b, error_name=None):
        assert len(a.shape) == len(b.shape) and len(a.shape) == len(cond.shape)
        assert a.dtype == b.dtype

        shape = []
        for i in range(len(a.shape)):
            shape.append(max(cond.shape[i], a.shape[i], b.shape[i]))

        if error_name == ErrorIf.WrongOutputType:
            all_dtypes = [DType.INT8, DType.INT16, DType.INT32, DType.INT48, DType.FLOAT]
            wrong_dtypes = list(set(all_dtypes) - set([a.dtype]))
            outputDType = rng.choice(wrong_dtypes)
        else:
            outputDType = a.dtype

        return ser.addOutput(shape, outputDType)

    @staticmethod
    def binaryComparisonOp(ser, rng, a, b , error_name=None):
        assert len(a.shape) == len(b.shape)
        assert a.dtype == b.dtype

        # Do broadcast
        shape = []
        for i in range(len(a.shape)):
            if a.shape[i] == 1:
                shape.append(b.shape[i])
            else:
                shape.append(a.shape[i])

        if error_name == ErrorIf.WrongOutputType:
            wrong_dtypes = [DType.INT8, DType.INT16, DType.INT32, DType.INT48, DType.FLOAT]
            outputDType = rng.choice(wrong_dtypes)
        else:
            outputDType = DType.BOOL

        return ser.addOutput(shape, outputDType)

    @staticmethod
    def reduceOp(ser, rng, a, axis, error_name=None):
        shape = a.shape.copy()
        if error_name not in [ErrorIf.AxisSmallerZero, ErrorIf.AxisLargerRank, ErrorIf.ShapeOfAxisNotOne]:
            shape[axis] = 1
        if error_name == ErrorIf.ShapeOfAxisNotOne and shape[axis] == 1:
            shape[axis] = rng.integers(2, 10)

        if error_name == ErrorIf.WrongOutputType:
            all_dtypes = [DType.INT8, DType.INT16, DType.INT32, DType.INT48, DType.FLOAT]
            wrong_dtypes = list(set(all_dtypes) - set([a.dtype]))
            outputDType = rng.choice(wrong_dtypes)
        else:
            outputDType = a.dtype

        return ser.addOutput(shape, outputDType)

    @staticmethod
    def argmaxOp(ser, rng, a, axis, error_name=None):
        shape = a.shape.copy()

        if error_name not in [ErrorIf.AxisSmallerZero, ErrorIf.AxisLargerRank]:
            del shape[axis]

        if error_name == ErrorIf.ArgmaxOutputRankMismatch:
            remove = rng.choice([True, False])
            if remove and len(shape) > 1:
                del shape[0]
            else:
                shape.append(1)
        elif error_name == ErrorIf.ArgmaxOutputShapeMismatch:
            for i in range(len(shape)):
                shape[i] = shape[i] + rng.integers(1, 10)

        if error_name == ErrorIf.WrongOutputType:
            all_dtypes = [DType.INT8, DType.INT16, DType.INT32, DType.INT48, DType.FLOAT]
            wrong_dtypes = list(set(all_dtypes) - set([DType.INT32]))
            outputDType = rng.choice(wrong_dtypes)
        else:
            outputDType = DType.INT32

        return ser.addOutput(shape, outputDType)

    @staticmethod
    def conv2dOp(ser, ifm, filter, strides, padding, dilations):

        # IFM:    NHWC
        # Filter: OHWI
        # OFM:    NHWC

        if len(padding) == 2:
            # Expand padding to 4 parameters in the case of transpose_conv2d
            # From H,W to T,B,L,R
            padding = [padding[0], padding[0], padding[1], padding[1]]

        h = (
            ifm.shape[1]
            - filter.shape[1]
            - (filter.shape[1] - 1) * (dilations[0] - 1)
            + padding[0]
            + padding[1]
        ) // strides[0] + 1

        w = (
            ifm.shape[2]
            - filter.shape[2]
            - (filter.shape[2] - 1) * (dilations[1] - 1)
            + padding[2]
            + padding[3]
        ) // strides[1] + 1

        ofm_shape = [ifm.shape[0], h, w, filter.shape[0]]

        if ifm.dtype == DType.INT8:
            out_dtype = DType.INT32
        elif ifm.dtype == DType.INT16:
            out_dtype = DType.INT48
        elif ifm.dtype == DType.FLOAT:
            out_dtype = DType.FLOAT
        else:
            raise Exception("Unsupported input dtype: {}".format(ifm.dtype))

        return ser.addOutput(ofm_shape, out_dtype)

    @staticmethod
    def conv3dOp(ser, ifm, filter, strides, padding, dilations):

        # IFM:    NDHWC
        # Filter: ODHWI
        # OFM:    NDHWC

        d = (
            ifm.shape[1]
            - filter.shape[1]
            - (filter.shape[1] - 1) * (dilations[0] - 1)
            + padding[0]
            + padding[1]
        ) // strides[0] + 1

        h = (
            ifm.shape[2]
            - filter.shape[2]
            - (filter.shape[2] - 1) * (dilations[1] - 1)
            + padding[2]
            + padding[3]
        ) // strides[1] + 1

        w = (
            ifm.shape[3]
            - filter.shape[3]
            - (filter.shape[3] - 1) * (dilations[2] - 1)
            + padding[4]
            + padding[5]
        ) // strides[2] + 1

        ofm_shape = [ifm.shape[0], d, h, w, filter.shape[0]]

        if ifm.dtype == DType.INT8:
            out_dtype = DType.INT32
        elif ifm.dtype == DType.INT16:
            out_dtype = DType.INT48
        elif ifm.dtype == DType.FLOAT:
            out_dtype = DType.FLOAT
        else:
            raise Exception("Unsupported input dtype: {}".format(ifm.dtype))

        return ser.addOutput(ofm_shape, out_dtype)

    @staticmethod
    def depthwiseConv2dOp(ser, ifm, filter, strides, padding, dilations):
        # IFM:    NHWC
        # Filter: HWCM
        # OFM:    NHW C*M
        h = (
            ifm.shape[1]
            - filter.shape[0]
            - (filter.shape[0] - 1) * (dilations[0] - 1)
            + padding[0]
            + padding[1]
        ) // strides[0] + 1

        w = (
            ifm.shape[2]
            - filter.shape[1]
            - (filter.shape[1] - 1) * (dilations[1] - 1)
            + padding[2]
            + padding[3]
        ) // strides[1] + 1

        ofm_shape = [ifm.shape[0], h, w, filter.shape[2] * filter.shape[3]]

        if ifm.dtype == DType.INT8:
            out_dtype = DType.INT32
        elif ifm.dtype == DType.INT16:
            out_dtype = DType.INT48
        elif ifm.dtype == DType.FLOAT:
            out_dtype = DType.FLOAT
        else:
            raise Exception("Unsupported input dtype: {}".format(ifm.dtype))

        return ser.addOutput(ofm_shape, out_dtype)

    @staticmethod
    def pool2dOp(ser, rng, ifm, kernel, stride, pad, error_name=None):
        # input: NHWC
        if stride[0] <= 0 or stride[1] <= 0 or min(pad) < 0:
            # If an incorrect stride is used set dimensions to 1, test is invalid anyway.
            h = 1
            w = 1
        else:
            h = (ifm.shape[1] + pad[0] + pad[1] + stride[0] - kernel[0]) // stride[0]
            w = (ifm.shape[2] + pad[2] + pad[3] + stride[1] - kernel[1]) // stride[1]

        if error_name == ErrorIf.PoolingOutputShapeMismatch:
            choices = [1, 2, 3, 4, 5]
            h = h + rng.choice(choices)
            w = w + rng.choice(choices)

        ofm_shape = [ifm.shape[0], h, w, ifm.shape[3]]

        if error_name == ErrorIf.WrongOutputType:
            all_dtypes = [DType.INT8, DType.INT16, DType.INT32, DType.INT48, DType.FLOAT]
            wrong_dtypes = list(set(all_dtypes) - set([ifm.dtype]))
            outputDType = rng.choice(wrong_dtypes)
        else:
            outputDType = ifm.dtype

        return ser.addOutput(ofm_shape, outputDType)

    @staticmethod
    def fullyConnectedOp(ser, rng, input, filter, error_name=None):
        # input: N, IC
        # filter: OC, IC
        # output: N, OC

        output_shape = [input.shape[0], filter.shape[0]]

        if error_name == ErrorIf.WrongOutputType:
            if input.dtype == DType.INT8:
                incorrect_types = (DType.INT4, DType.INT8, DType.INT16, DType.INT48, DType.FLOAT)
            elif input.dtype == DType.INT16:
                incorrect_types = (DType.INT4, DType.INT8, DType.INT16, DType.INT32, DType.FLOAT)
            elif input.dtype == DType.FLOAT:
                incorrect_types = (DType.INT4, DType.INT8, DType.INT16, DType.INT32, DType.INT48)
            out_dtype = rng.choice(a=incorrect_types)
        elif input.dtype == DType.INT8:
            out_dtype = DType.INT32
        elif input.dtype == DType.INT16:
            out_dtype = DType.INT48
        elif input.dtype == DType.FLOAT:
            out_dtype = DType.FLOAT
        elif error_name == ErrorIf.WrongInputType:
            # Pick some potentially correct output dtype if input type is incorrect
            out_dtype = DType.INT32
        else:
            raise Exception("Unsupported input dtype: {}".format(input.dtype))

        return ser.addOutput(output_shape, out_dtype)

    @staticmethod
    def matmulOp(ser, rng, a, b, error_name=None):
        # a: N, H, C
        # b: N, C, W
        # out: N, H, W

        output_shape = [a.shape[0], a.shape[1], b.shape[2]]

        if error_name == ErrorIf.WrongOutputType:
            if a.dtype == DType.INT8:
                incorrect_types = (DType.INT4, DType.INT8, DType.INT16, DType.INT48, DType.FLOAT)
            elif a.dtype == DType.INT16:
                incorrect_types = (DType.INT4, DType.INT8, DType.INT16, DType.INT32, DType.FLOAT)
            elif a.dtype == DType.FLOAT:
                incorrect_types = (DType.INT4, DType.INT8, DType.INT16, DType.INT32, DType.INT48)
            out_dtype = rng.choice(a=incorrect_types)
        elif a.dtype == DType.INT8:
            out_dtype = DType.INT32
        elif a.dtype == DType.INT16:
            out_dtype = DType.INT48
        elif a.dtype == DType.FLOAT:
            out_dtype = DType.FLOAT
        elif error_name == ErrorIf.WrongInputType:
            # Pick some potentially correct output dtype if input type is incorrect
            out_dtype = DType.INT32
        else:
            raise Exception("Unsupported input dtype for matmul: {}".format(a.dtype))

        return ser.addOutput(output_shape, out_dtype)

    @staticmethod
    def concatOp(ser, rng, axis, *a, error_name=None):
        input1 = a[0]
        remaining_inputs = a[1:]

        # calculate the output shape, if possible, otherwise just use the first input shape
        output_shape = input1.shape.copy()
        if not (
            # unable to concat tensors of different ranks
            error_name == ErrorIf.ConcatInputRankMismatch
            # unable to concat tensors along an invalid axis
            or error_name in [ErrorIf.AxisLargerRank, ErrorIf.AxisSmallerZero]
        ):
            for tensor in remaining_inputs:
                output_shape[axis] += tensor.shape[axis]

        if error_name == ErrorIf.ConcatShapeSumMismatch:
            output_shape[axis] += rng.integers(5, 10)

        if error_name == ErrorIf.WrongOutputType:
            all_dtypes = {DType.INT8, DType.INT16, DType.INT32, DType.INT48, DType.FLOAT}
            wrong_dtypes = list(all_dtypes - set([input1.dtype]))
            outputDType = rng.choice(wrong_dtypes)
        else:
            outputDType = input1.dtype

        return ser.addOutput(output_shape, outputDType)

    @staticmethod
    def padOp(ser, rng, a, padding, error_name=None):

        output_shape = a.shape.copy()

        for i in range(len(output_shape)):
            output_shape[i] = padding[i][0] + padding[i][1] + output_shape[i]

        # Fix negative output shape if error_if test causes it
        if error_name == ErrorIf.PadSmallerZero and min(output_shape) < 1:
            output_shape = [i if i >= 1 else 1 for i in output_shape]

        if error_name == ErrorIf.WrongOutputType:
            all_dtypes = [DType.INT8, DType.INT16, DType.INT32, DType.INT48, DType.FLOAT]
            wrong_dtypes = list(set(all_dtypes) - set([a.dtype]))
            outputDType = rng.choice(wrong_dtypes)
        else:
            outputDType = a.dtype

        return ser.addOutput(output_shape, outputDType)

    @staticmethod
    def reshapeOp(ser, rng, a, shape, error_name=None):
        output_shape = shape.copy()

        totalElements = 1
        for i in a.shape:
            totalElements *= i

        # If there are any -1 elements, figure out what that dimension must be
        totalOutputElements = 1
        for i in output_shape:
            if i != -1:
                totalOutputElements *= i

        # And fill it in
        for i in range(len(output_shape)):
            if output_shape[i] == -1:
                output_shape[i] = totalElements // totalOutputElements

        if error_name == ErrorIf.TensorSizeInputOutputMismatch:
            for i in range(len(output_shape)):
                output_shape[i] = output_shape[i] + rng.integers(1, 10)

        if error_name == ErrorIf.WrongOutputType:
            all_dtypes = [DType.INT8, DType.INT16, DType.INT32, DType.INT48, DType.FLOAT]
            wrong_dtypes = list(set(all_dtypes) - set([a.dtype]))
            outputDType = rng.choice(wrong_dtypes)
        else:
            outputDType = a.dtype

        return ser.addOutput(output_shape, outputDType)

    @staticmethod
    def sliceOp(ser, rng, a, start, size, error_name=None):

        if error_name == ErrorIf.WrongOutputType:
            all_dtypes = [DType.INT8, DType.INT16, DType.INT32, DType.INT48, DType.FLOAT]
            wrong_dtypes = list(set(all_dtypes) - set([a.dtype]))
            outputDType = rng.choice(wrong_dtypes)
        else:
            outputDType = a.dtype

        if error_name == ErrorIf.SizeOutputShapeMismatch:
            output_shape = size.copy()
            for index in range(len(output_shape)):
                if output_shape[index] <= 2:
                    output_shape[index] = output_shape[index] + rng.choice([1, 2])
                else:
                    output_shape[index] = output_shape[index] + rng.choice([-2, -1, 1, 2])
        else:
            output_shape = size.copy()

        return ser.addOutput(output_shape, outputDType)

    @staticmethod
    def tileOp(ser, rng, a, multiples, error_name=None):

        output_shape = a.shape.copy()
        assert len(multiples) == len(output_shape)

        for i in range(len(output_shape)):
            output_shape[i] = a.shape[i] * multiples[i]

        if error_name == ErrorIf.WrongOutputType:
            all_dtypes = [DType.INT8, DType.INT16, DType.INT32, DType.INT48, DType.FLOAT]
            wrong_dtypes = list(set(all_dtypes) - set([a.dtype]))
            outputDType = rng.choice(wrong_dtypes)
        else:
            outputDType = a.dtype

        return ser.addOutput(output_shape, outputDType)

    @staticmethod
    def transposeOp(ser, rng, a, perms, error_name=None):
        output_shape = a.shape.copy()

        assert len(perms) == len(output_shape)

        if error_name == ErrorIf.IndexOutsideBounds:
            for i in range(len(output_shape)):
                output_shape[i] = a.shape[0]
        else:
            for i in range(len(output_shape)):
                output_shape[i] = a.shape[perms[i]]

        if error_name == ErrorIf.WrongOutputType:
            all_dtypes = [DType.INT8, DType.INT16, DType.INT32, DType.INT48, DType.FLOAT]
            wrong_dtypes = list(set(all_dtypes) - set([a.dtype]))
            outputDType = rng.choice(wrong_dtypes)
        else:
            outputDType = a.dtype

        return ser.addOutput(output_shape, outputDType)

    @staticmethod
    def gatherOp(ser, rng, values, indices, error_name=None):
        assert len(values.shape) == 3
        assert len(indices.shape) == 2
        assert values.shape[0] == indices.shape[0]

        output_shape = [values.shape[0], indices.shape[1], values.shape[2]]

        if error_name == ErrorIf.WrongOutputType:
            all_dtypes = [DType.INT8, DType.INT16, DType.INT32, DType.INT48, DType.FLOAT]
            wrong_dtypes = list(set(all_dtypes) - set([values.dtype]))
            outputDType = rng.choice(wrong_dtypes)
        else:
            outputDType = values.dtype

        return ser.addOutput(output_shape, outputDType)

    @staticmethod
    def scatterOp(ser, rng, values_in, indices, input, error_name=None):
        assert len(values_in.shape) == 3
        assert len(indices.shape) == 2
        assert len(input.shape) == 3
        assert values_in.shape[0] == indices.shape[0]  # N
        assert input.shape[1] == indices.shape[1]  # W
        assert values_in.shape[2] == input.shape[2]  # C

        output_shape = values_in.shape

        if error_name == ErrorIf.WrongOutputType:
            all_dtypes = [DType.INT8, DType.INT16, DType.INT32, DType.INT48, DType.FLOAT]
            wrong_dtypes = list(set(all_dtypes) - set([values_in.dtype]))
            outputDType = rng.choice(wrong_dtypes)
        else:
            outputDType = values_in.dtype

        return ser.addOutput(output_shape, outputDType)

    @staticmethod
    def tableOp(ser, rng, input, error_name=None):
        # Same shape as the input, dtype dependent on input dtype
        if error_name != ErrorIf.WrongInputType:
            assert input.dtype == DType.INT16 or input.dtype == DType.INT8
        output_dtype = DType.INT32 if input.dtype == DType.INT16 else DType.INT8
        if error_name == ErrorIf.WrongOutputType:
            wrong_dtypes = [DType.INT8, DType.INT16, DType.INT32, DType.INT48, DType.FLOAT]
            wrong_dtypes.remove(output_dtype)
            output_dtype = rng.choice(wrong_dtypes)
        return ser.addOutput(input.shape, output_dtype)

    @staticmethod
    def resizeOp(
        serializer,
        rng,
        input,
        mode,
        stride,
        offset,
        shift,
        stride_fp,
        offset_fp,
        output_dims,
        input_dtype,
        output_dtype,
        error_name = None
    ):
        if error_name == ErrorIf.WrongRank:
            output_dims = [input.shape[0], output_dims[0], output_dims[0], input.shape[0]]
        else:
            if error_name == ErrorIf.BatchMismatch:
                output_dims = [input.shape[0] + rng.integers(1, 10), output_dims[0], output_dims[1], input.shape[3]]
            elif error_name == ErrorIf.ChannelMismatch:
                output_dims = [input.shape[0], output_dims[0], output_dims[1], input.shape[3] + rng.integers(1, 10)]
            else:
                output_dims = [input.shape[0], output_dims[0], output_dims[1], input.shape[3]]

        return serializer.addOutput(output_dims, output_dtype)

    @staticmethod
    def typeConversionOp(ser, rng, val, out_dtype, error_name=None):
        return ser.addOutput(val.shape, out_dtype)

    @staticmethod
    def transposeConv2DOp(ser, ifm, output_shape):
        if ifm.dtype == DType.INT8:
            out_dtype = DType.INT32
        elif ifm.dtype == DType.INT16:
            out_dtype = DType.INT48
        elif ifm.dtype == DType.FLOAT:
            out_dtype = DType.FLOAT
        else:
            raise Exception("Unsupported input dtype: {}".format(ifm.dtype))

        return ser.addOutput(output_shape, out_dtype)

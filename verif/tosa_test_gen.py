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

# Convenience variables to the flatc-generated types that should be enums, but aren't
DType = tosa.DType.DType()
Op = tosa.Op.Op()
ResizeMode = tosa.ResizeMode.ResizeMode()


class TosaQuantGen:
    """QuantizedInfo random generator helper functions.  Specify with 'qgen': in the operator defintion"""

    def __init__(self):
        pass

    @staticmethod
    def getQinfo(testGen, dtype):
        if dtype == DType.INT8:
            return testGen.randInt(-128, 128)
        if dtype == DType.UINT8:
            return testGen.randInt(0, 256)
        return 0

    @staticmethod
    def qgUnary(testGen, op, dtype):
        qinfo = ts.TosaSerializerQuantInfo()
        qinfo.UnaryQuantInfo(
            TosaQuantGen.getQinfo(testGen, dtype), TosaQuantGen.getQinfo(testGen, dtype)
        )
        return qinfo

    @staticmethod
    def qgConv(testGen, op, dtype_or_dtypeList):
        qinfo = ts.TosaSerializerQuantInfo()
        if isinstance(dtype_or_dtypeList, list):
            # a list of [input, weights, accumulator] dtypes
            dtypeList = dtype_or_dtypeList
        else:
            # an int, [input, weights, accumulator] dtypes are the same
            dtypeList = [dtype_or_dtypeList] * 3
        input_zp = TosaQuantGen.getQinfo(testGen, dtypeList[0])
        weights_zp = TosaQuantGen.getQinfo(testGen, dtypeList[1])
        qinfo.ConvQuantInfo(input_zp, weights_zp)
        return qinfo

    @staticmethod
    def qgMatmul(testGen, op, dtype):
        qinfo = ts.TosaSerializerQuantInfo()
        qinfo.MatMulQuantInfo(
            TosaQuantGen.getQinfo(testGen, dtype), TosaQuantGen.getQinfo(testGen, dtype)
        )
        return qinfo

    @staticmethod
    def qgPad(testGen, op, dtype):
        qinfo = ts.TosaSerializerQuantInfo()
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
    def tgBasic(testGen, opName, rank):
        pl, const = opName["operands"]
        shape = testGen.makeShape(rank)

        shape_list = []
        for i in range(pl + const):
            shape_list.append(shape.copy())

        return shape_list

    @staticmethod
    def tgNHWC(testGen, opName, rank):
        pl, const = opName["operands"]

        assert rank == 4

        shape = testGen.makeShape(rank)

        # Constrict the batch size?
        if testGen.args.max_batch_size:
            shape[0] = (shape[0] % testGen.args.max_batch_size) + 1

        shape_list = []
        for i in range(pl + const):
            shape_list.append(shape.copy())

        return shape_list

    @staticmethod
    def tgScatter(testGen, opName, rank):
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
    def tgBroadcastFuzz(testGen, op, rank):
        shape = testGen.makeShape(rank)

        pl, const = op["operands"]

        shape_list = []

        # Choose one of the inputs to broadcast
        bcast_idx = testGen.randInt(0, pl + const)
        for i in range(pl + const):
            shape_bcast = shape.copy()

            # If the chosen input, pick a random index to broadcast
            if i == bcast_idx:
                fuzz_idx = testGen.randInt(0, rank)
                shape_bcast[fuzz_idx] = 1

            shape_list.append(shape_bcast)

        return shape_list

    @staticmethod
    def tgConv2D(testGen, op, rank):
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
    def tgTransposeConv2D(testGen, op, rank):
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
    def tgDepthwiseConv2D(testGen, op, rank):
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
    def tgFullyConnected(testGen, op, rank):
        pl, const = op["operands"]

        assert rank == 2

        input_shape = testGen.makeShape(rank)
        filter_oc = testGen.rng.integers(
            low=testGen.args.tensor_shape_range[0],
            high=testGen.args.tensor_shape_range[1],
            size=1,
        )[0]
        filter_shape = np.asarray([filter_oc, input_shape[1]])

        bias_shape = np.asarray([filter_oc])

        return [input_shape, filter_shape, bias_shape]

    @staticmethod
    def tgMatmul(testGen, op, rank):
        pl, const = op["operands"]

        assert rank == 3
        assert pl == 2 and const == 0

        a_shape = testGen.makeShape(rank)
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
    def tgConcat(testGen, opName, rank):
        pl, const = opName["operands"]
        shape = testGen.makeShape(rank)

        # Create extra tensors to concat.
        # Take into account value of pl when getting maximum number of concats
        num_tensors = testGen.randInt(0, 4)
        shape_list = []
        for i in range(pl + const + num_tensors):
            shape_list.append(shape.copy())

        return shape_list

    @staticmethod
    def tgConcatConstInput(testGen, shapeList, axis):
        # Split concat shape along axis to allow for multiple const inputs
        # without making too many large tensors
        shape = shapeList[0]
        if len(shapeList) == 2 or shape[axis] < len(shapeList):
            return shapeList

        new_shapeList = [shape.copy()]
        length_on_axis = shape[axis]
        remaining_length = length_on_axis
        for i in range(len(shapeList)-2):
            # Calculate split on axis and remaining value
            split_shape_val = int(shape[axis] / 2)
            remaining_length = remaining_length - split_shape_val

            # Append new shape, and set remaining shape
            shape[axis] = split_shape_val
            new_shapeList.append(shape.copy())
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
    def agNone(testGen, opName, shapeList, dtype):
        """A trivial argument generator for operators that don't take any
        non-tensor arguments"""
        return [("", [])]

    @staticmethod
    def agAxis(testGen, opName, shapeList, dtype):
        """Build the axis argument for operators that take a single axis"""
        axes = []

        shape = shapeList[0]

        for a in range(0, len(shape)):
            axes.append(("axis{}".format(a), [a]))
        return axes

    @staticmethod
    def agConv2D(testGen, opName, shapeList, dtype):
        arg_list = []

        ifm_shape = shapeList[0]
        filter_shape = shapeList[1]

        # Must be rank 4
        assert len(ifm_shape) == 4
        assert len(filter_shape) == 4

        maxStride = testGen.args.max_conv_stride
        maxPadding = testGen.args.max_conv_padding + 1
        maxDilation = testGen.args.max_conv_dilation

        # Strides, padding, dilations
        for stride in range(0, maxStride ** 2):
            for padding in range(0, (maxPadding) ** 4):
                for dilation in range(0, maxDilation ** 2):

                    s = [stride // maxStride + 1, stride % maxStride + 1]
                    p = [
                        (padding // (maxPadding * 4)) % maxPadding,
                        (padding // (maxPadding * 2)) % maxPadding,
                        (padding // (maxPadding * 1)) % maxPadding,
                        padding % maxPadding,
                    ]
                    d = [dilation // maxDilation + 1, dilation % maxDilation + 1]

                    # 4 padding parameters for regular conv2d
                    arg_list.append(
                        (
                            "st{}{}_pad{}{}{}{}_dilat{}{}".format(
                                s[0], s[1], p[0], p[1], p[2], p[3], d[0], d[1]
                            ),
                            [s, p, d],
                        )
                    )
        return arg_list

    @staticmethod
    def agTransposeConv2D(testGen, opName, shapeList, dtype):
        arg_list = []

        ifm_shape = shapeList[0]
        filter_shape = shapeList[1]

        # Must be rank 4
        assert len(ifm_shape) == 4
        assert len(filter_shape) == 4

        maxStride = testGen.args.max_conv_stride
        maxPadding = testGen.args.max_conv_padding + 1
        maxDilation = testGen.args.max_conv_dilation

        # Strides, padding, dilations
        for stride in range(0, maxStride ** 2):
            for out_padding in range(0, (maxPadding) ** 2):
                for dilation in range(0, maxDilation ** 2):

                    s = [stride // maxStride + 1, stride % maxStride + 1]
                    p = [
                        (out_padding // (maxPadding * 1)) % maxPadding,
                        out_padding % maxPadding,
                    ]
                    d = [dilation // maxDilation + 1, dilation % maxDilation + 1]

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

                    # Output shape
                    os = [ifm_shape[0], oh, ow, filter_shape[0]]

                    arg_list.append(
                        (
                            "st{}{}_outpad{}{}_dilat{}{}_os{}x{}x{}x{}".format(
                                s[0],
                                s[1],
                                p[0],
                                p[1],
                                d[0],
                                d[1],
                                os[0],
                                os[1],
                                os[2],
                                os[3],
                            ),
                            [s, p, d, os],
                        )
                    )

        return arg_list

    @staticmethod
    def agPad(testGen, opName, shapeList, dtype):
        arg_list = []
        rank = len(shapeList[0])

        # Exhaustively test combinations of padding on each side of each dimension
        # - the range of padding values is defined by pad_min and pad_max
        # - for padding >9, the name format needs to be more distinctive
        pad_min, pad_max = 0, 1
        pad_values = [x for x in range(pad_min, pad_max + 1)]
        axis_pad_values = [x for x in itertools.product(pad_values, pad_values)]
        shape_pad_values = itertools.product(*([axis_pad_values] * rank))

        for paddings in shape_pad_values:
            name = "pad"
            for r in range(rank):
                before, after = paddings[r]
                name = f"{name}{before}{after}"
            arg_list.append((name, [np.array(paddings)]))

        return arg_list

    @staticmethod
    def agPooling(testGen, opName, shapeList, dtype):
        arg_list = []

        shape = shapeList[0]
        assert len(shape) == 4

        maxStride = testGen.args.max_pooling_stride
        maxKernel = testGen.args.max_pooling_kernel
        maxPadding = testGen.args.max_pooling_padding + 1

        for kernel in range(0, maxKernel ** 2):
            for stride in range(0, maxStride ** 2):
                for padding in range(0, maxPadding ** 4):
                    s = [stride // maxStride + 1, stride % maxStride + 1]
                    k = [(kernel // maxKernel) + 2, (kernel % maxKernel) + 2]
                    p = [
                        (padding // (maxPadding * 4)) % maxPadding,
                        (padding // (maxPadding * 2)) % maxPadding,
                        (padding // (maxPadding * 1)) % maxPadding,
                        padding % maxPadding,
                    ]

                    arg_list.append(
                        (
                            "st{}{}_kern{}{}_pad{}{}{}{}".format(
                                s[0], s[1], k[0], k[1], p[0], p[1], p[2], p[3]
                            ),
                            [s, p, k],
                        )
                    )
        return arg_list

    @staticmethod
    def agCast(testGen, opName, shapeList, inDtype):
        arg_list = []

        # Enumerate the output types here
        if inDtype == DType.INT8:
            dtypeList = [DType.BOOL, DType.INT16, DType.INT32, DType.FLOAT]
        elif inDtype == DType.INT16:
            dtypeList = [DType.BOOL, DType.INT8, DType.INT32, DType.FLOAT]
        elif inDtype == DType.INT32:
            dtypeList = [DType.BOOL, DType.INT8, DType.INT16, DType.FLOAT]
        elif inDtype == DType.BOOL:
            dtypeList = [DType.INT8, DType.INT16, DType.INT32]
        elif inDtype == DType.FLOAT:
            dtypeList = [DType.INT8, DType.INT16, DType.INT32]
        else:
            raise Exception("Unexpected input dtype: {}".format(inDtype))

        for dtype in dtypeList:
            arg_list.append(("out{}".format(DTypeNames[dtype]), [dtype]))

        return arg_list

    @staticmethod
    def agRescale(testGen, opName, shapeList, inDtype):
        arg_list = []

        # Enumerate the output types here
        for dtype in [DType.UINT8, DType.INT8, DType.INT16, DType.INT32]:
            if inDtype == DType.UINT8 and dtype != DType.INT8:
                # The only output dtype for UINT8 is INT8, skip all other combinations
                continue
            if inDtype != DType.INT8 and dtype == DType.UINT8:
                # The only input dtype for UINT8 is INT8, skip all other combinations
                continue

            for scale32 in [False, True]:
                for double_round in [False, True]:
                    for per_channel in [False, True]:

                        if inDtype == DType.INT48 and scale32:
                            # Illegal condition.  Must be scale32=False
                            continue
                        if double_round and not scale32:
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
    def agMul(testGen, opName, shapeList, dtype):
        arg_list = []

        if dtype is DType.INT32:
            for p in range(testGen.args.num_rand_permutations):

                shift = testGen.randInt(0, 32)

                arg_list.append(("perm{}_shift{}".format(p, shift), [shift]))
        else:
            arg_list.append(("perm0_shift0", [0]))

        return arg_list

    @staticmethod
    def agArithmeticRightShift(testGen, opName, shapeList, dtype):
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
    def agReshape(testGen, opName, shapeList, dtype):
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
    def agTranspose(testGen, opName, shapeList, dtype):
        arg_list = []

        ifm_shape = shapeList[0]

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
    def agSlice(testGen, opName, shapeList, dtype):
        arg_list = []

        ifm_shape = shapeList[0]
        rank = len(ifm_shape)

        for p in range(testGen.args.num_rand_permutations):
            begin = []
            size = []

            valid = True

            for i in range(rank):
                if ifm_shape[i] > 1:
                    begin.append(testGen.randInt(0, ifm_shape[i]))
                    size.append(testGen.randInt(0, ifm_shape[i] - begin[i]))

                    # Invalid slice size?
                    if size[i] == 0:
                        valid = False
                else:
                    begin.append(0)
                    size.append(1)

            if valid:
                arg_list.append(("perm{}".format(p), [begin, size]))
        return arg_list

    @staticmethod
    def agTile(testGen, opName, shapeList, dtype):
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
    def agResize(testGen, opName, shapeList, dtype):
        arg_list = []

        ifm_shape = shapeList[0]

        for m in [ResizeMode.NEAREST, ResizeMode.BILINEAR]:

            # Exclude illegal {mode, type} configurations.  Pick legal output types
            if m == ResizeMode.NEAREST and dtype == DType.INT8:
                outputDTypeList = [DType.INT8]
            elif m == ResizeMode.NEAREST and dtype == DType.INT16:
                outputDTypeList = [DType.INT16]
            elif m == ResizeMode.BILINEAR and dtype == DType.INT8:
                outputDTypeList = [DType.INT32]
            elif m == ResizeMode.BILINEAR and dtype == DType.INT16:
                outputDTypeList = [DType.INT48]
            elif dtype == DType.FLOAT:
                outputDTypeList = [DType.FLOAT]
            else:
                continue

            for outputDType in outputDTypeList:
                for perm in range(testGen.args.num_rand_permutations):

                    # Randomly generate legal output dimensions and shift
                    # and then compute the stride and offset based on them
                    output_dims = [testGen.randInt(1), testGen.randInt(1)]
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
                        arg_list.append(
                            (
                                "mode{}_odim{}x{}_out{}_st{:.2f}x{:.2f}_off{:.2f}x{:.2f}".format(
                                    "N" if m == ResizeMode.NEAREST else "B",
                                    output_dims[0],
                                    output_dims[1],
                                    testGen.typeStr(outputDType),
                                    stride_fp[0],
                                    stride_fp[1],
                                    offset_fp[0],
                                    offset_fp[1],
                                ),
                                [
                                    m,
                                    stride,
                                    offset,
                                    shift,
                                    stride_fp,
                                    offset_fp,
                                    output_dims,
                                    dtype,
                                    outputDType,
                                ],
                            )
                        )
                    else:
                        shift = 11
                        unit = float(1 << shift)
                        stride_y = int(round(fp_stride_y * unit))
                        stride_x = int(round(fp_stride_x * unit))
                        offset_y = int(round(fp_offset_y * unit))
                        offset_x = int(round(fp_offset_x * unit))

                        while (
                            stride_y >= 32768
                            or stride_x >= 32768
                            or offset_y >= 32768
                            or offset_x >= 32768
                            or offset_y < -32768
                            or offset_x < -32768
                        ):
                            shift = shift - 1
                            unit = float(1 << shift)
                            stride_y = int(round(fp_stride_y * unit))
                            stride_x = int(round(fp_stride_x * unit))
                            offset_y = int(round(fp_offset_y * unit))
                            offset_x = int(round(fp_offset_x * unit))

                        stride = [stride_y, stride_x]
                        offset = [offset_y, offset_x]

                        stride_fp = [0.0, 0.0]
                        offset_fp = [0.0, 0.0]

                        arg_list.append(
                            (
                                "mode{}_shift{}_odim{}x{}_out{}_st{}x{}_off{}x{}".format(
                                    "N" if m == ResizeMode.NEAREST else "B",
                                    shift,
                                    output_dims[0],
                                    output_dims[1],
                                    testGen.typeStr(outputDType),
                                    stride[0],
                                    stride[1],
                                    offset[0],
                                    offset[1],
                                ),
                                [
                                    m,
                                    stride,
                                    offset,
                                    shift,
                                    stride_fp,
                                    offset_fp,
                                    output_dims,
                                    dtype,
                                    outputDType,
                                ],
                            )
                        )

        return arg_list

    def agCondIf(testGen, opName, shapeList, dtype):
        # CondIf generates the condition values here.
        # Convert to tensors in the build function, along with the
        # then and else blocks
        arg_list = []

        for c in [False, True]:
            arg_list.append(("cond{}".format(int(c)), [c]))

        return arg_list

    def agWhileLoop(testGen, opName, shapeList, dtype):
        # While loop: 0 iterations, 1, more than 1
        arg_list = []

        for iter in [0, 1, 4]:
            arg_list.append(("iter{}".format(iter), [iter]))

        return arg_list

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
        else:
            raise Exception("Unknown dtype, cannot convert to string: {}".format(t))

    # Argument generators
    # Returns a list of tuples (stringDescriptor, [build_fcn_arg_list])
    # Where the string descriptor is used to generate the test name and
    # The build_fcn_arg_list is expanded and passed to the operator test
    # build function

    def build_unary(self, op, a, qinfo=None):
        result_tens = OutputShaper.unaryOp(self.ser, a)
        self.ser.addOperator(op, [a.name], [result_tens.name], None, qinfo)
        return result_tens

    def build_binary_broadcast(self, op, a, b):
        result_tens = OutputShaper.binaryBroadcastOp(self.ser, a, b)
        self.ser.addOperator(op, [a.name, b.name], [result_tens.name])
        return result_tens

    def build_binary_nonbroadcast(self, op, a, b):
        result_tens = OutputShaper.binaryNonBroadcastOp(self.ser, a, b)
        self.ser.addOperator(op, [a.name, b.name], [result_tens.name])
        return result_tens

    def build_arithmetic_right_shift(self, op, a, b, round):
        result_tens = OutputShaper.binaryBroadcastOp(self.ser, a, b)

        attr = ts.TosaSerializerAttribute()
        attr.ArithmeticRightShiftAttribute(round)

        self.ser.addOperator(op, [a.name, b.name], [result_tens.name], attr)
        return result_tens

    def build_mul(self, op, a, b, shift):
        result_tens = OutputShaper.binaryBroadcastOp(self.ser, a, b)

        # Special for multiply:
        # Force the result to INT32 for INT types
        if a.dtype != DType.FLOAT:
            result_tens.setDtype(DType.INT32)

        attr = ts.TosaSerializerAttribute()
        attr.MulAttribute(shift)

        self.ser.addOperator(op, [a.name, b.name], [result_tens.name], attr)
        return result_tens

    def build_table(self, op, a):
        # Constant size depending on type, random values
        if a.dtype == DType.INT16:
            table_dtype = DType.INT16
            table_arr = self.getRandTensor([513], table_dtype)
        else:
            assert a.dtype == DType.INT8
            table_dtype = DType.INT8
            table_arr = self.getRandTensor([256], table_dtype)

        table_tens = self.ser.addConst(table_arr.shape, table_dtype, table_arr)
        result_tens = OutputShaper.tableOp(self.ser, a, table_dtype)
        self.ser.addOperator(op, [a.name, table_tens.name], [result_tens.name], None)

        return result_tens

    def build_select(self, op, cond, a, b):
        result_tens = OutputShaper.selectOp(self.ser, cond, a, b)
        self.ser.addOperator(op, [cond.name, a.name, b.name], [result_tens.name])
        return result_tens

    def build_comparison(self, op, a, b):
        result_tens = OutputShaper.binaryComparisonOp(self.ser, a, b)
        self.ser.addOperator(op, [a.name, b.name], [result_tens.name])
        return result_tens

    def build_argmax(self, op, a, axis):
        result_tens = OutputShaper.argmaxOp(self.ser, a, axis)

        attr = ts.TosaSerializerAttribute()
        attr.AxisAttribute(axis)

        self.ser.addOperator(op, [a.name], [result_tens.name], attr)
        return result_tens

    def build_pool2d(self, op, input, stride, pad, kernel, qinfo=None):
        result_tens = OutputShaper.pool2dOp(self.ser, input, kernel, stride, pad)

        attr = ts.TosaSerializerAttribute()
        attr.Pool2dAttribute(kernel, stride, pad)

        self.ser.addOperator(op, [input.name], [result_tens.name], attr, qinfo)
        return result_tens

    def build_conv2d(self, op, ifm, filter, bias, strides, padding, dilations, qinfo):
        assert len(padding) == 4
        result_tens = OutputShaper.conv2dOp(
            self.ser, ifm, filter, strides, padding, dilations
        )

        attr = ts.TosaSerializerAttribute()
        attr.Conv2dAttribute(padding, strides, dilations)

        self.ser.addOperator(
            op, [ifm.name, filter.name, bias.name], [result_tens.name], attr, qinfo
        )
        return result_tens

    def build_transpose_conv2d(
        self, op, ifm, filter, bias, stride, outpad, dilation, output_shape, qinfo
    ):
        assert len(outpad) == 2
        result_tens = OutputShaper.transposeConv2DOp(self.ser, ifm, output_shape)

        attr = ts.TosaSerializerAttribute()
        attr.TransposeConv2DAttribute(outpad, stride, dilation, output_shape)

        self.ser.addOperator(
            op, [ifm.name, filter.name, bias.name], [result_tens.name], attr, qinfo
        )
        return result_tens

    def build_depthwise_conv2d(
        self, op, ifm, filter, bias, strides, padding, dilations, qinfo
    ):
        result_tens = OutputShaper.depthwiseConv2dOp(
            self.ser, ifm, filter, strides, padding, dilations
        )

        attr = ts.TosaSerializerAttribute()
        attr.Conv2dAttribute(padding, strides, dilations)

        self.ser.addOperator(
            op, [ifm.name, filter.name, bias.name], [result_tens.name], attr, qinfo
        )
        return result_tens

    def build_fully_connected(self, op, ifm, filter, bias, qinfo):
        result_tens = OutputShaper.fullyConnectedOp(self.ser, ifm, filter)

        self.ser.addOperator(
            op, [ifm.name, filter.name, bias.name], [result_tens.name], None, qinfo
        )
        return result_tens

    def build_matmul(self, op, a, b, qinfo):
        result_tens = OutputShaper.matmulOp(self.ser, a, b)
        self.ser.addOperator(op, [a.name, b.name], [result_tens.name], None, qinfo)
        return result_tens

    def build_reduce(self, op, a, axis):
        result_tens = OutputShaper.reduceOp(self.ser, a, axis)

        attr = ts.TosaSerializerAttribute()
        attr.AxisAttribute(axis)

        self.ser.addOperator(op, [a.name], result_tens.name, attr)
        return result_tens

    def build_clamp(self, op, a):
        result_tens = OutputShaper.unaryOp(self.ser, a)

        attr = ts.TosaSerializerAttribute()
        v = [self.getRandNumberDType(a.dtype), self.getRandNumberDType(a.dtype)]

        if a.dtype == DType.FLOAT:
            attr.ClampAttribute(0, 0, min(v), max(v))
        else:
            attr.ClampAttribute(min(v), max(v), 0, 0)

        self.ser.addOperator(op, [a.name], [result_tens.name], attr)
        return result_tens

    def build_leaky_relu(self, op, a):
        result_tens = OutputShaper.unaryOp(self.ser, a)
        attr = ts.TosaSerializerAttribute()

        attr.LeakyReluAttribute(self.getRandNumberDType(DType.FLOAT))

        self.ser.addOperator(op, [a.name], [result_tens.name], attr)
        return result_tens

    # Needs an additional type/input
    def build_prelu(self, op, a):
        result_tens = OutputShaper.unaryOp(self.ser, a)

        self.ser.addOperator(op, [a.name], [result_tens.name])
        return result_tens

    def build_relun(self, op, a):
        result_tens = OutputShaper.unaryOp(self.ser, a)

        attr = ts.TosaSerializerAttribute()

        if a.dtype == DType.FLOAT:
            attr.ReluNAttribute(0, self.getRandNumberDType(a.dtype))
        else:
            attr.ReluNAttribute(self.getRandNumberDType(a.dtype), 0)

        self.ser.addOperator(op, [a.name], [result_tens.name], attr)
        return result_tens

    def build_sigmoid(self, op, a):
        result_tens = OutputShaper.unaryOp(self.ser, a)
        self.ser.addOperator(op, [a.name], [result_tens.name])
        return result_tens

    def build_tanh(self, op, a):
        result_tens = OutputShaper.unaryOp(self.ser, a)
        self.ser.addOperator(op, [a.name], [result_tens.name])
        return result_tens

    def build_concat(self, op, *a):
        assert (type(a[-1]) == int)

        # To store variable length list of input tensors we need to store axis along with it
        axis = a[-1]
        a = a[:-1]

        result_tens = OutputShaper.concatOp(self.ser, axis, *a)

        attr = ts.TosaSerializerAttribute()
        attr.AxisAttribute(axis)

        input_tensor_names = []
        for tensor in a:
            input_tensor_names.append(tensor.name)

        self.ser.addOperator(op, input_tensor_names, [result_tens.name], attr)

    def build_pad(self, op, a, padding, qinfo):
        result_tens = OutputShaper.padOp(self.ser, a, padding)

        # Need to turn the padding array into a TOSA tensor here.
        # This is one of the few tensor operands that does not get
        # randomly generated
        padding_tens = self.ser.addConst(padding.shape, DType.INT32, padding)

        self.ser.addOperator(
            op, [a.name, padding_tens.name], [result_tens.name], None, qinfo
        )

    def build_reshape(self, op, a, newShape):
        result_tens = OutputShaper.reshapeOp(self.ser, a, newShape)

        attr = ts.TosaSerializerAttribute()
        attr.ReshapeAttribute(newShape)

        self.ser.addOperator(op, [a.name], [result_tens.name], attr)
        return result_tens

    def build_reverse(self, op, a, axis):
        result_tens = OutputShaper.unaryOp(self.ser, a)

        attr = ts.TosaSerializerAttribute()
        attr.AxisAttribute(axis)

        self.ser.addOperator(op, [a.name], [result_tens.name], attr)
        return result_tens

    def build_transpose(self, op, a, perms):
        result_tens = OutputShaper.transposeOp(self.ser, a, perms)

        perms_tens = self.ser.addConst([len(perms)], DType.INT32, np.int32(perms))

        self.ser.addOperator(op, [a.name, perms_tens.name], [result_tens.name])
        return result_tens

    def build_slice(self, op, a, begin, size):
        result_tens = OutputShaper.sliceOp(self.ser, a, begin, size)

        attr = ts.TosaSerializerAttribute()
        attr.SliceAttribute(begin, size)

        self.ser.addOperator(op, [a.name], [result_tens.name], attr)
        return result_tens

    def build_tile(self, op, a, multiples):
        result_tens = OutputShaper.tileOp(self.ser, a, multiples)

        attr = ts.TosaSerializerAttribute()
        attr.TileAttribute(multiples)

        self.ser.addOperator(op, [a.name], [result_tens.name], attr)
        return result_tens

    def build_gather(self, op, values):

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

        result_tens = OutputShaper.gatherOp(self.ser, values, indicies)

        self.ser.addOperator(op, [values.name, indicies.name], [result_tens.name])

        return result_tens

    def build_scatter(self, op, values_in, input):

        # Create a new indicies tensor
        # here with data that doesn't exceed the dimensions of the values_in tensor

        K = values_in.shape[1]  # K
        W = input.shape[1]  # W
        indicies_arr = np.int32(
            self.rng.integers(low=0, high=K, size=[values_in.shape[0], W])
        )  # (N, W)
        indicies = self.ser.addConst(indicies_arr.shape, DType.INT32, indicies_arr)

        result_tens = OutputShaper.scatterOp(self.ser, values_in, indicies, input)

        self.ser.addOperator(
            op, [values_in.name, indicies.name, input.name], [result_tens.name]
        )

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
    ):
        result_tens = OutputShaper.resizeOp(
            self.ser,
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
        )

        attr = ts.TosaSerializerAttribute()

        attr.ResizeAttribute(
            output_dims, stride, offset, shift, stride_fp, offset_fp, mode
        )

        self.ser.addOperator(op, [input.name], [result_tens.name], attr)
        return result_tens

    def build_identityn(self, op, val, val2):

        result_tens = OutputShaper.unaryOp(self.ser, val)
        result_tens2 = OutputShaper.unaryOp(self.ser, val2)
        self.ser.addOperator(
            op, [val.name, val2.name], [result_tens.name, result_tens2.name]
        )
        return result_tens

    def build_placeholder(self, op, val):
        # Add an identity op to avoid warning in the reference model
        return self.build_unary(Op.IDENTITY, val)

    # Type Conversion
    def build_cast(self, op, val, out_dtype):
        result_tens = OutputShaper.typeConversionOp(self.ser, val, out_dtype)
        self.ser.addOperator(op, [val.name], [result_tens.name])
        return result_tens

    def build_rescale(self, op, val, out_dtype, scale32, double_round, per_channel):
        result_tens = OutputShaper.typeConversionOp(self.ser, val, out_dtype)

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
        else:
            input_zp = 0

        if out_dtype == DType.INT8:
            output_zp = self.randInt(-128, 128)
            out_type_width = out_type_width + 1
        elif out_dtype == DType.UINT8:
            output_zp = self.randInt(0, 256)
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

        self.ser.addOperator(op, [val.name], [result_tens.name], attr)
        return result_tens

    def build_cond_if_const(self, op, then_tens, else_tens, cond):
        # For cond_if with constants, we're supplied with then/else tensors that we ignore
        # (except for the generated shap) and the condition.  Build Then/Else blocks
        # and fill them with const nodes for the body.

        # Condition tensor
        cond_tens = self.ser.addConst([], DType.BOOL, [cond])

        # Make then/else tensors
        out_shape = then_tens.shape
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
        self.ser.addOperator(op, [cond_tens.name], [result_tens.name], attr)

        self.ser.startBasicBlock(then_block)
        # Build the actual then/else tensors inside their blocks
        then_tens = self.ser.addConst(out_shape, DType.INT32, then_arr)
        self.ser.addOutputTensor(then_tens)

        self.ser.startBasicBlock(else_block)
        else_tens = self.ser.addConst(out_shape, DType.INT32, else_arr)
        self.ser.addOutputTensor(else_tens)

        return result_tens

    def build_cond_if_binary(self, op, a, b, cond):
        # For cond_if with a binary op in the then/else blocks, take a and b and
        # alternately add or subtract them based on the condition

        # Condition tensor
        cond_tens = self.ser.addConst([], DType.BOOL, [cond])

        result_tens = self.ser.addOutput(a.shape, a.dtype)
        self.ser.currBasicBlock.addOutput(result_tens.name)

        # Create the attribute with the names of the then/else blocks
        then_block = "THEN_BLOCK"
        else_block = "ELSE_BLOCK"
        attr = ts.TosaSerializerAttribute()
        attr.CondIfAttribute(then_block, else_block)

        # Finally, build the op and the two blocks
        self.ser.addOperator(
            op, [cond_tens.name, a.name, b.name], [result_tens.name], attr
        )

        self.ser.startBasicBlock(then_block)
        self.ser.addInputTensor(a)
        self.ser.addInputTensor(b)
        then_tens = self.ser.addOutput(a.shape, a.dtype)
        self.ser.addOperator(Op.ADD, [a.name, b.name], [then_tens.name])

        self.ser.startBasicBlock(else_block)
        self.ser.addInputTensor(a)
        self.ser.addInputTensor(b)
        else_tens = self.ser.addOutput(a.shape, a.dtype)
        self.ser.addOperator(Op.SUB, [a.name, b.name], [else_tens.name])

        return result_tens

    def build_while_loop(self, op, a, iter_val):
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
        acc_out = self.ser.addIntermediate(acc.shape, acc.dtype)

        # While_loop operator
        self.ser.addOperator(
            op,
            [iter.name, a.name, acc.name],
            [iter_out.name, a_out.name, acc_out.name],
            attr,
        )
        self.ser.addOutputTensor(acc_out)

        # COND block (input: iter, output: cond_tens )
        self.ser.startBasicBlock(cond_block)
        self.ser.addInputTensor(iter)
        self.ser.addInputTensor(a)
        self.ser.addInputTensor(acc)
        zero_tens = self.ser.addConst([], DType.INT32, [np.int32(0)])
        cond_tens = self.ser.addOutput([], DType.BOOL)
        self.ser.addOperator(Op.GREATER, [iter.name, zero_tens.name], [cond_tens.name])

        # BODY block (input: a, acc, iter, output: a, acc, iter)
        # Note that local intermediate tensors need to be declared here for the outputs
        self.ser.startBasicBlock(body_block)
        self.ser.addInputTensor(iter)
        self.ser.addInputTensor(a)
        self.ser.addInputTensor(acc)
        one_tens = self.ser.addConst([], DType.INT32, [np.int32(1)])
        iter_body_out = self.ser.addIntermediate(iter.shape, iter.dtype)
        acc_body_out = self.ser.addIntermediate(acc.shape, acc.dtype)
        self.ser.addOperator(Op.ADD, [a.name, acc.name], [acc_body_out.name])
        self.ser.addOperator(Op.SUB, [iter.name, one_tens.name], [iter_body_out.name])
        self.ser.addOutputTensor(iter_body_out)
        self.ser.addOutputTensor(a)
        self.ser.addOutputTensor(acc_body_out)

        return acc_out

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

        # Generate the lists of arguments
        rmin, rmax = op["rank"]

        # Create a default testing rank range, 1-4 inclusive to keep test sizes reasonably small.
        default_test_rank_range = range(1, 5)

        # Test list consists of a tuple of:
        # (opName, testNameStr, dtype, shapeList, argumentsList)
        testList = []

        if not shapeFilter:
            shapeFilter = [None]

        # Positive test loop
        if testType in ['positive', 'both']:
            for r in range(rmin, rmax + 1):

                # Filter out the rank?
                if rankFilter is not None and r not in rankFilter:
                    continue
                if (
                    rankFilter is None
                    and shapeFilter[0] is None
                    and r not in default_test_rank_range
                ):
                    continue

                for t in op["types"]:

                    # Filter tests based on dtype?
                    if dtypeFilter is not None:
                        if not (
                            t in dtypeFilter
                            or (isinstance(t, list) and t[0] in dtypeFilter)
                        ):
                            continue

                    # Create the placeholder and const tensors
                    for shape in shapeFilter:
                        # A None shape chooses a random shape of a given rank

                        # Filter out by rank
                        if shape is not None and len(shape) != r:
                            continue

                        self.setTargetShape(shape)
                        shapeList = tgen_fcn(self, op, r)

                        shapeStr = self.shapeStr(shapeList[0])
                        typeStr = self.typeStr(t)

                        # Argument lists consists of tuples of the (str, []) string representation and the build function argument list
                        argList = []
                        if agen_fcn:
                            argList = agen_fcn(self, opName, shapeList, t)
                        else:
                            argList = [("", [])]

                        for argStr, args in argList:
                            if argStr:
                                testStr = "{}_{}_{}_{}".format(
                                    opName, shapeStr, typeStr, argStr
                                )
                            else:
                                testStr = "{}_{}_{}".format(opName, shapeStr, typeStr)

                            testList.append((opName, testStr, t, shapeList, args))

        # Remove tests which are expected to fail but don't correlate to a ERROR_IF statement
        if "invalid_test_validators" in op:
            invalid_test_validators = op["invalid_test_validators"]
            clean_testList = []
            for test in testList:
                for validator_fcn in invalid_test_validators:
                    remove_test = False
                    if validator_fcn(opName=test[0], input_dtype=test[2], shapeList=test[3], args=test[4]):
                        remove_test = True
                if not remove_test:
                    clean_testList.append(test)
            testList = clean_testList

        # Reset RNG so both positive and negative tests are reproducible
        self.resetRNG()
        # Negative test loop
        if testType in ['negative', 'both']:
            print("Negative tests unsupported")

        return testList

    def serializeTest(self, opName, testStr, dtype_or_dtypeList, shapeList, testArgs):
        try:
            op = self.TOSA_OP_LIST[opName]
        except KeyError as e:
            raise Exception("Cannot find op with name {}".format(opName))

        # Create a serializer
        self.createSerializer(opName, testStr)

        build_fcn, tgen_fcn, agen_fcn = op["build_fcn"]
        pCount, cCount = op["operands"]
        num_operands = pCount + cCount

        if isinstance(dtype_or_dtypeList, list):
            dtypeList = dtype_or_dtypeList
        elif op['op'] == Op.CONCAT:
            dtypeList = [dtype_or_dtypeList] * len(shapeList)
        else:
            dtypeList = [dtype_or_dtypeList] * (num_operands)

        if op['op'] != Op.CONCAT:
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

        if (op["op"] == Op.ADD or op["op"] == Op.SUB) and dtypeList[0] == DType.INT32:
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
        elif op["op"] == Op.INTDIV:
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

            shapeList = TosaTensorGen.tgConcatConstInput(self, shapeList, testArgs[0])
            tens.extend(
                self.buildPlaceholderTensors(shapeList[0:count], dtypeList[0:count])
            )
            tens.extend(self.buildConstTensors(shapeList[count:], dtypeList[count:]))
        else:
            tens.extend(
                self.buildPlaceholderTensors(shapeList[0:pCount], dtypeList[0:pCount])
            )
            tens.extend(self.buildConstTensors(shapeList[pCount:], dtypeList[pCount:]))

        if qgen is not None:
            qinfo = qgen(self, op, dtype_or_dtypeList)
        else:
            qinfo = None

        try:
            if qinfo is not None:
                resultName = build_fcn(self, op["op"], *tens, *testArgs, qinfo)
            else:
                resultName = build_fcn(self, op["op"], *tens, *testArgs)
        except TypeError as e:
            print(
                "build_fcn: {}\nTensors: {}\nArgs: {}\n".format(
                    build_fcn, tens, testArgs
                )
            )
            raise e

        # Save the serialized test
        self.serialize("test")

    def createDynamicOpLists(self):

        # Dynamically create op lists for convolutions with a list of kernel sizes
        KERNELS = [[1, 1], [2, 2], [3, 3], [5, 5], [3, 1], [1, 3]]

        for k in KERNELS:
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

    TYPE_CONV2D = [
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
            "build_fcn": (build_argmax, TosaTensorGen.tgBasic, TosaArgGen.agAxis),
            "types": TYPE_NARROW_INT_FP,
        },
        "avg_pool2d": {
            "op": Op.AVG_POOL2D,
            "operands": (1, 0),
            "rank": (4, 4),
            "build_fcn": (build_pool2d, TosaTensorGen.tgNHWC, TosaArgGen.agPooling),
            "qgen": TosaQuantGen.qgUnary,
            "types": TYPE_NARROW_INT_FP,
            "invalid_test_validators": (TosaInvalidValidator.ivHeightWidthSmallerZero,)
        },
        # Templated operator.  Filled in by createDynamicOpLists
        "conv2d_TEMPLATE": {
            "op": Op.CONV2D,
            "operands": (1, 2),
            "rank": (4, 4),
            "build_fcn": (build_conv2d, TosaTensorGen.tgConv2D, TosaArgGen.agConv2D),
            "qgen": TosaQuantGen.qgConv,
            "types": TYPE_CONV2D,
            "invalid_test_validators": (TosaInvalidValidator.ivHeightWidthSmallerZero,),
            "template": True,
        },
        # Conv3d TBD
        # Templated operator.  Filled in by createDynamicOpLists
        "depthwise_conv2d_TEMPLATE": {
            "op": Op.DEPTHWISE_CONV2D,
            "operands": (1, 2),
            "filter": [1, 1],
            "rank": (4, 4),
            "build_fcn": (
                build_depthwise_conv2d,
                TosaTensorGen.tgDepthwiseConv2D,
                TosaArgGen.agConv2D,
            ),
            "qgen": TosaQuantGen.qgConv,
            "types": TYPE_CONV2D,
            "invalid_test_validators": (TosaInvalidValidator.ivHeightWidthSmallerZero,),
            "template": True,
        },
        "fully_connected": {
            "op": Op.FULLY_CONNECTED,
            "operands": (1, 2),
            "rank": (2, 2),
            "build_fcn": (build_fully_connected, TosaTensorGen.tgFullyConnected, None),
            "qgen": TosaQuantGen.qgConv,
            "types": TYPE_CONV2D,
        },
        "matmul": {
            "op": Op.MATMUL,
            "operands": (2, 0),
            "rank": (3, 3),
            "build_fcn": (build_matmul, TosaTensorGen.tgMatmul, None),
            "qgen": TosaQuantGen.qgMatmul,
            "types": TYPE_NARROW_INT_FP,
        },
        "max_pool2d": {
            "op": Op.MAX_POOL2D,
            "operands": (1, 0),
            "rank": (4, 4),
            "build_fcn": (build_pool2d, TosaTensorGen.tgNHWC, TosaArgGen.agPooling),
            "types": TYPE_NARROW_INT_FP,
            "invalid_test_validators": (TosaInvalidValidator.ivHeightWidthSmallerZero,)
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
            "types": TYPE_CONV2D,
            "invalid_test_validators": (TosaInvalidValidator.ivNonPositiveOutputShape,),
            "template": True,
        },
        # Activation functions
        "clamp": {
            "op": Op.CLAMP,
            "operands": (1, 0),
            "build_fcn": (build_clamp, TosaTensorGen.tgBasic, None),
            "types": TYPE_NARROW_INT_FP,
        },
        "relun": {
            "op": Op.RELUN,
            "operands": (1, 0),
            "build_fcn": (build_relun, TosaTensorGen.tgBasic, None),
            "types": TYPE_FI32,
        },
        "sigmoid": {
            "op": Op.SIGMOID,
            "operands": (1, 0),
            "build_fcn": (build_sigmoid, TosaTensorGen.tgBasic, None),
            "types": TYPE_FP,
        },
        "tanh": {
            "op": Op.TANH,
            "operands": (1, 0),
            "build_fcn": (build_tanh, TosaTensorGen.tgBasic, None),
            "types": TYPE_FP,
        },
        # Elementwise Binary Operators
        "add": {
            "op": Op.ADD,
            "operands": (2, 0),
            "build_fcn": (build_binary_broadcast, TosaTensorGen.tgBroadcastFuzz, None),
            "types": TYPE_FI32,
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
        },
        "bitwise_and": {
            "op": Op.BITWISE_AND,
            "operands": (2, 0),
            "build_fcn": (build_binary_broadcast, TosaTensorGen.tgBroadcastFuzz, None),
            "types": TYPE_INT,
        },
        "bitwise_or": {
            "op": Op.BITWISE_OR,
            "operands": (2, 0),
            "build_fcn": (build_binary_broadcast, TosaTensorGen.tgBroadcastFuzz, None),
            "types": TYPE_INT,
        },
        "bitwise_xor": {
            "op": Op.BITWISE_XOR,
            "operands": (2, 0),
            "build_fcn": (build_binary_broadcast, TosaTensorGen.tgBroadcastFuzz, None),
            "types": TYPE_INT,
        },
        "intdiv": {
            "op": Op.INTDIV,
            "operands": (2, 0),
            "build_fcn": (build_binary_broadcast, TosaTensorGen.tgBroadcastFuzz, None),
            "types": [DType.INT32],
        },
        "logical_and": {
            "op": Op.LOGICAL_AND,
            "operands": (2, 0),
            "build_fcn": (build_binary_broadcast, TosaTensorGen.tgBroadcastFuzz, None),
            "types": TYPE_BOOL,
        },
        "logical_left_shift": {
            "op": Op.LOGICAL_LEFT_SHIFT,
            "operands": (2, 0),
            "build_fcn": (build_binary_broadcast, TosaTensorGen.tgBroadcastFuzz, None),
            "types": TYPE_INT,
        },
        "logical_right_shift": {
            "op": Op.LOGICAL_RIGHT_SHIFT,
            "operands": (2, 0),
            "build_fcn": (build_binary_broadcast, TosaTensorGen.tgBroadcastFuzz, None),
            "types": TYPE_INT,
        },
        "logical_or": {
            "op": Op.LOGICAL_OR,
            "operands": (2, 0),
            "build_fcn": (build_binary_broadcast, TosaTensorGen.tgBroadcastFuzz, None),
            "types": TYPE_BOOL,
        },
        "logical_xor": {
            "op": Op.LOGICAL_XOR,
            "operands": (2, 0),
            "build_fcn": (build_binary_broadcast, TosaTensorGen.tgBroadcastFuzz, None),
            "types": TYPE_BOOL,
        },
        "maximum": {
            "op": Op.MAXIMUM,
            "operands": (2, 0),
            "build_fcn": (build_binary_broadcast, TosaTensorGen.tgBroadcastFuzz, None),
            "types": TYPE_FI32,
        },
        "minimum": {
            "op": Op.MINIMUM,
            "operands": (2, 0),
            "build_fcn": (build_binary_broadcast, TosaTensorGen.tgBroadcastFuzz, None),
            "types": TYPE_FI32,
        },
        "mul": {
            "op": Op.MUL,
            "operands": (2, 0),
            "build_fcn": (build_mul, TosaTensorGen.tgBroadcastFuzz, TosaArgGen.agMul),
            "types": TYPE_INT_FP,
        },
        "pow": {
            "op": Op.POW,
            "operands": (2, 0),
            "build_fcn": (build_binary_broadcast, TosaTensorGen.tgBasic, None),
            "types": TYPE_FP,
        },
        "sub": {
            "op": Op.SUB,
            "operands": (2, 0),
            "build_fcn": (build_binary_broadcast, TosaTensorGen.tgBroadcastFuzz, None),
            "types": TYPE_FI32,
        },
        "table": {
            "op": Op.TABLE,
            # Use the automatic generation functions to create the input array
            # but create the table tensor in the build function, as it may be
            # a different type from the input
            "operands": (1, 0),
            "build_fcn": (build_table, TosaTensorGen.tgBasic, None),
            "types": [DType.INT8, DType.INT16],
        },
        # Elementwise Unary operators
        "abs": {
            "op": Op.ABS,
            "operands": (1, 0),
            "build_fcn": (build_unary, TosaTensorGen.tgBasic, None),
            "types": TYPE_FI32,
        },
        "bitwise_not": {
            "op": Op.BITWISE_NOT,
            "operands": (1, 0),
            "build_fcn": (build_unary, TosaTensorGen.tgBasic, None),
            "types": TYPE_INT,
        },
        "ceil": {
            "op": Op.CEIL,
            "operands": (1, 0),
            "build_fcn": (build_unary, TosaTensorGen.tgBasic, None),
            "types": TYPE_FP,
        },
        "clz": {
            "op": Op.CLZ,
            "operands": (1, 0),
            "build_fcn": (build_unary, TosaTensorGen.tgBasic, None),
            "types": [DType.INT32],
        },
        "exp": {
            "op": Op.EXP,
            "operands": (1, 0),
            "build_fcn": (build_unary, TosaTensorGen.tgBasic, None),
            "types": TYPE_FP,
        },
        "floor": {
            "op": Op.FLOOR,
            "operands": (1, 0),
            "build_fcn": (build_unary, TosaTensorGen.tgBasic, None),
            "types": TYPE_FP,
        },
        "log": {
            "op": Op.LOG,
            "operands": (1, 0),
            "build_fcn": (build_unary, TosaTensorGen.tgBasic, None),
            "types": TYPE_FP,
        },
        "logical_not": {
            "op": Op.LOGICAL_NOT,
            "operands": (1, 0),
            "build_fcn": (build_unary, TosaTensorGen.tgBasic, None),
            "types": TYPE_BOOL,
        },
        "negate": {
            "op": Op.NEGATE,
            "operands": (1, 0),
            "build_fcn": (build_unary, TosaTensorGen.tgBasic, None),
            "qgen": TosaQuantGen.qgUnary,
            "types": TYPE_INT_FP,
        },
        "reciprocal": {
            "op": Op.RECIPROCAL,
            "operands": (1, 0),
            "build_fcn": (build_unary, TosaTensorGen.tgBasic, None),
            "types": TYPE_FP,
        },
        "rsqrt": {
            "op": Op.RSQRT,
            "operands": (1, 0),
            "build_fcn": (build_unary, TosaTensorGen.tgBasic, None),
            "types": TYPE_FP,
        },
        # Elementwise Ternary operators
        "select": {
            "op": Op.SELECT,
            "operands": (3, 0),
            "build_fcn": (build_select, TosaTensorGen.tgBroadcastFuzz, None),
            "types": TYPE_FIB,
        },
        # Comparison operators
        "equal": {
            "op": Op.EQUAL,
            "operands": (2, 0),
            "build_fcn": (build_comparison, TosaTensorGen.tgBroadcastFuzz, None),
            "types": TYPE_FI32,
        },
        "greater_equal": {
            "op": Op.GREATER_EQUAL,
            "operands": (2, 0),
            "build_fcn": (build_comparison, TosaTensorGen.tgBroadcastFuzz, None),
            "types": TYPE_FI32,
        },
        "greater": {
            "op": Op.GREATER,
            "operands": (2, 0),
            "build_fcn": (build_comparison, TosaTensorGen.tgBroadcastFuzz, None),
            "types": TYPE_FI32,
        },
        # Reduction operators
        "reduce_all": {
            "op": Op.REDUCE_ALL,
            "operands": (1, 0),
            "build_fcn": (build_reduce, TosaTensorGen.tgBasic, TosaArgGen.agAxis),
            "types": TYPE_BOOL,
        },
        "reduce_any": {
            "op": Op.REDUCE_ANY,
            "operands": (1, 0),
            "build_fcn": (build_reduce, TosaTensorGen.tgBasic, TosaArgGen.agAxis),
            "types": TYPE_BOOL,
        },
        "reduce_max": {
            "op": Op.REDUCE_MAX,
            "operands": (1, 0),
            "build_fcn": (build_reduce, TosaTensorGen.tgBasic, TosaArgGen.agAxis),
            "types": TYPE_INT_FP,
        },
        "reduce_min": {
            "op": Op.REDUCE_MAX,
            "operands": (1, 0),
            "build_fcn": (build_reduce, TosaTensorGen.tgBasic, TosaArgGen.agAxis),
            "types": TYPE_INT_FP,
        },
        "reduce_product": {
            "op": Op.REDUCE_PRODUCT,
            "operands": (1, 0),
            "build_fcn": (build_reduce, TosaTensorGen.tgBasic, TosaArgGen.agAxis),
            "types": TYPE_FP,
        },
        "reduce_sum": {
            "op": Op.REDUCE_SUM,
            "operands": (1, 0),
            "build_fcn": (build_reduce, TosaTensorGen.tgBasic, TosaArgGen.agAxis),
            "types": TYPE_FI32,
        },
        # Data layout operators
        "concat": {
            "op": Op.CONCAT,
            "operands": (2, 0),
            "build_fcn": (build_concat, TosaTensorGen.tgConcat, TosaArgGen.agAxis),
            "types": TYPE_FIB,
        },
        "pad": {
            "op": Op.PAD,
            "operands": (1, 0),
            "build_fcn": (build_pad, TosaTensorGen.tgBasic, TosaArgGen.agPad),
            "qgen": TosaQuantGen.qgPad,
            "types": TYPE_FIB,
        },
        "reshape": {
            "op": Op.RESHAPE,
            "operands": (1, 0),
            "build_fcn": (build_reshape, TosaTensorGen.tgBasic, TosaArgGen.agReshape),
            "types": TYPE_FIB,
        },
        "reverse": {
            "op": Op.REVERSE,
            "operands": (1, 0),
            "build_fcn": (build_reverse, TosaTensorGen.tgBasic, TosaArgGen.agAxis),
            "types": TYPE_FIB,
        },
        "slice": {
            "op": Op.SLICE,
            "operands": (1, 0),
            "build_fcn": (build_slice, TosaTensorGen.tgBasic, TosaArgGen.agSlice),
            "types": TYPE_FIB,
        },
        "tile": {
            "op": Op.TILE,
            "operands": (1, 0),
            "build_fcn": (build_tile, TosaTensorGen.tgBasic, TosaArgGen.agTile),
            "types": TYPE_FIB,
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
        },
        # Data nodes
        "const": {
            "op": Op.CONST,
            "operands": (1, 0),
            "build_fcn": (build_placeholder, TosaTensorGen.tgBasic, None),
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
        },
        "scatter": {
            "op": Op.SCATTER,
            # Only specify 'values_in' tensor here.
            #'indices' and 'input' are generated in op building stage
            "operands": (2, 0),
            "rank": (3, 3),
            "build_fcn": (build_scatter, TosaTensorGen.tgScatter, None),
            "types": TYPE_INT_FP,
        },
        # Image operations
        "resize": {
            "op": Op.RESIZE,
            "operands": (1, 0),
            "rank": (4, 4),
            "build_fcn": (build_resize, TosaTensorGen.tgNHWC, TosaArgGen.agResize),
            "types": [DType.INT8, DType.INT16, DType.FLOAT],
            "invalid_test_validators": (TosaInvalidValidator.ivWrongDataTypeOrModeResize, TosaInvalidValidator.ivBadStride)
        },
        # Type conversion
        "cast": {
            "op": Op.CAST,
            "operands": (1, 0),
            "build_fcn": (build_cast, TosaTensorGen.tgBasic, TosaArgGen.agCast),
            "types": [DType.FLOAT, DType.INT8, DType.INT16, DType.INT32, DType.BOOL],
        },
        "rescale": {
            "op": Op.RESCALE,
            "operands": (1, 0),
            "build_fcn": (build_rescale, TosaTensorGen.tgBasic, TosaArgGen.agRescale),
            "types": [DType.UINT8, DType.INT8, DType.INT16, DType.INT32, DType.INT48],
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
        },
        "cond_if_binary": {
            "op": Op.COND_IF,
            "operands": (2, 0),
            "build_fcn": (
                build_cond_if_binary,
                TosaTensorGen.tgBasic,
                TosaArgGen.agCondIf,
            ),
            "types": TYPE_FI32,
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
    def binaryBroadcastOp(ser, a, b):
        assert len(a.shape) == len(b.shape)
        assert a.dtype == b.dtype

        shape = []
        for i in range(len(a.shape)):
            if a.shape[i] == 1:
                shape.append(b.shape[i])
            else:
                shape.append(a.shape[i])

        return ser.addOutput(shape, a.dtype)

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
    def unaryOp(ser, a):
        return ser.addOutput(a.shape, a.dtype)

    @staticmethod
    def selectOp(ser, cond, a, b):
        assert len(a.shape) == len(b.shape) and len(a.shape) == len(cond.shape)
        assert a.dtype == b.dtype

        shape = []
        for i in range(len(a.shape)):
            shape.append(max(cond.shape[i], a.shape[i], b.shape[i]))

        return ser.addOutput(shape, a.dtype)

    @staticmethod
    def binaryComparisonOp(ser, a, b):
        assert len(a.shape) == len(b.shape)
        assert a.dtype == b.dtype

        # Do broadcast
        shape = []
        for i in range(len(a.shape)):
            if a.shape[i] == 1:
                shape.append(b.shape[i])
            else:
                shape.append(a.shape[i])

        # Force the output type to bool
        return ser.addOutput(shape, DType.BOOL)

    @staticmethod
    def reduceOp(ser, a, axis):

        shape = a.shape.copy()

        shape[axis] = 1

        return ser.addOutput(shape, a.dtype)

    @staticmethod
    def argmaxOp(ser, a, axis):
        shape = a.shape.copy()
        del shape[axis]
        return ser.addOutput(shape, DType.INT32)

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
    def pool2dOp(ser, ifm, kernel, stride, pad):
        # input: NHWC
        h = (ifm.shape[1] + pad[0] + pad[1] + stride[0] - kernel[0]) // stride[0]
        w = (ifm.shape[2] + pad[2] + pad[3] + stride[1] - kernel[1]) // stride[1]

        ofm_shape = [ifm.shape[0], h, w, ifm.shape[3]]
        return ser.addOutput(ofm_shape, ifm.dtype)

    @staticmethod
    def fullyConnectedOp(ser, input, filter):
        # input: N, IC
        # filter: OC, IC
        # output: N, OC

        output_shape = [input.shape[0], filter.shape[0]]

        if input.dtype == DType.INT8:
            out_dtype = DType.INT32
        elif input.dtype == DType.INT16:
            out_dtype = DType.INT48
        elif input.dtype == DType.FLOAT:
            out_dtype = DType.FLOAT
        else:
            raise Exception("Unsupported input dtype: {}".format(input.dtype))

        return ser.addOutput(output_shape, out_dtype)

    @staticmethod
    def matmulOp(ser, a, b):
        # a: N, H, C
        # b: N, C, W
        # out: N, H, W

        output_shape = [a.shape[0], a.shape[1], b.shape[2]]

        if a.dtype == DType.INT8:
            out_dtype = DType.INT32
        elif a.dtype == DType.INT16:
            out_dtype = DType.INT48
        elif a.dtype == DType.FLOAT:
            out_dtype = DType.FLOAT
        else:
            raise Exception("UNsupported input dtype for matmul: {}".format(a.dtype))

        return ser.addOutput(output_shape, out_dtype)

    @staticmethod
    def concatOp(ser, axis, *a):
        input1 = a[0]
        remaining_inputs = a[1:]

        output_shape = input1.shape.copy()

        output_shape[axis] = input1.shape[axis]

        for tensor in remaining_inputs:
            output_shape[axis] += tensor.shape[axis]

        return ser.addOutput(output_shape, input1.dtype)

    @staticmethod
    def padOp(ser, a, padding):

        output_shape = a.shape.copy()

        for i in range(len(output_shape)):
            output_shape[i] = padding[i][0] + padding[i][1] + output_shape[i]

        return ser.addOutput(output_shape, a.dtype)

    @staticmethod
    def reshapeOp(ser, a, shape):
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

        return ser.addOutput(output_shape, a.dtype)

    @staticmethod
    def sliceOp(ser, a, begin, size):

        output_shape = size.copy()
        return ser.addOutput(output_shape, a.dtype)

    @staticmethod
    def tileOp(ser, a, multiples):

        output_shape = a.shape.copy()
        assert len(multiples) == len(output_shape)

        for i in range(len(output_shape)):
            output_shape[i] = a.shape[i] * multiples[i]

        return ser.addOutput(output_shape, a.dtype)

    @staticmethod
    def transposeOp(ser, a, perms):
        output_shape = a.shape.copy()
        assert len(perms) == len(output_shape)

        for i in range(len(output_shape)):
            output_shape[i] = a.shape[perms[i]]

        return ser.addOutput(output_shape, a.dtype)

    @staticmethod
    def gatherOp(ser, values, indices):
        assert len(values.shape) == 3
        assert len(indices.shape) == 2
        assert values.shape[0] == indices.shape[0]

        output_shape = [values.shape[0], indices.shape[1], values.shape[2]]

        return ser.addOutput(output_shape, values.dtype)

    @staticmethod
    def scatterOp(ser, values_in, indices, input):
        assert len(values_in.shape) == 3
        assert len(indices.shape) == 2
        assert len(input.shape) == 3
        assert values_in.shape[0] == indices.shape[0]  # N
        assert input.shape[1] == indices.shape[1]  # W
        assert values_in.shape[2] == input.shape[2]  # C

        output_shape = values_in.shape

        return ser.addOutput(output_shape, values_in.dtype)

    @staticmethod
    def tableOp(ser, input, table_dtype):
        # Same shape as the input, but dtype dependent on table dtype
        assert table_dtype == DType.INT16 or table_dtype == DType.INT8
        output_dtype = DType.INT32 if table_dtype == DType.INT16 else DType.INT8
        return ser.addOutput(input.shape, output_dtype)

    @staticmethod
    def resizeOp(
        ser,
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
    ):

        output_dims = [input.shape[0], output_dims[0], output_dims[1], input.shape[3]]

        return ser.addOutput(output_dims, output_dtype)

    @staticmethod
    def typeConversionOp(ser, val, out_dtype):
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

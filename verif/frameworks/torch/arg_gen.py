# Copyright (c) 2024-2025, ARM Limited.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
from frameworks.test_gen_utils import DTYPE_ATTRIBUTES


class ArgGen:
    """Argument generator functions.  These functions take a shape and dtype to
    create arguments for an operator. Methods are prefixed with 'ag' to make
    search easy."""

    def __init__(self):
        pass

    def typeStr(dtype):
        if dtype in DTYPE_ATTRIBUTES:
            return DTYPE_ATTRIBUTES[dtype]["str"]
        else:
            raise Exception("Unknown dtype, cannot convert to string: {}".format(dtype))

    @staticmethod
    def agNone(op, shapes, rng):
        """A trivial argument generator for operators that only take tensor
        operands"""
        return [("", [])]

    # Build the axis argument for operators where we want to iterate over N axes
    # as an argument
    @staticmethod
    def agAxes(op, shapes, rng):
        axes = []
        if shapes == ():
            axes.append(["_axis_0", [0]])
            return axes

        for i in range(-len(shapes), len(shapes), 1):
            if i >= 0:
                axes.append(["_axis_{}".format(i), [i]])
            else:
                axes.append(["_axis_m{}".format(-i), [i]])
        return axes

    # Build the axis LIST argument for operators that take an axis list.
    # This builds a list of each axis individually, plus one element
    # that contains a list of all axes.  Note that we need to pack the list in
    # an additional list so that it isn't exploded when being passed to the
    # build_operator function.
    # tensor_arg_count not used
    def agAxesList(op, shapes, rng):
        axes = ArgGen.agAxes(op, shapes, rng)
        axes_list = []
        for desc, a in axes:
            axes_list.append([desc, [a]])

        axes_list.append(["_axisall", [list(range(len(shapes)))]])
        axes_list.append(["_axisall_none", [None]])
        return axes_list

    def agAxesListKeepdims(op, shapes, rng):
        axes = ArgGen.agAxes(op, shapes, rng)
        axes_list = []
        for desc, a in axes:
            axes_list.append([desc + "_keep0", [a, False]])
            if (a[0] >= 0 and shapes[a[0]] != 1) or (
                a[0] < 0 and shapes[len(shapes) + a[0]] != 1
            ):
                axes_list.append([desc + "_keep1", [a, True]])

        axes_list.append(["_axisall_keep0", [list(range(len(shapes))), False]])
        axes_list.append(["_axisall_keep0_none", [None, False]])
        # another instance where the reduce gets optimized out.
        if len(shapes) != 1:
            axes_list.append(["_axisall_keep1", [list(range(len(shapes))), True]])
            axes_list.append(["_axisall_keep1_none", [None, True]])
        return axes_list

    # conv2d argument generators
    def agConv2d(op, shapes, rng):
        arg_list = []

        # Must be rank 4
        if len(shapes) < 4:
            return arg_list

        filter_h, filter_w = op["filter"]

        # strides, padding, dilations,
        for stride_h in [1, 2]:
            for stride_w in [1, 2]:
                for padding_h in [0, 1]:
                    for padding_w in [0, 1]:
                        for dilation_h in [1, 2]:
                            for dilation_w in [1, 2]:
                                # Stride should not be larger than kernel size
                                if (stride_h > filter_h) or (stride_w > filter_w):
                                    continue

                                # Pad should be at most half of effective kernel size
                                if (padding_h > filter_h // 2) or (
                                    padding_w > filter_w // 2
                                ):
                                    continue

                                # Output shape = (W + 2P - D(F - 1) - 1) / S + 1
                                # W: input shape, P: padding, D: dilation, F: filter size, S: stride
                                h_out_numerator = (
                                    shapes[2]
                                    + 2 * padding_h
                                    - dilation_h * (filter_h - 1)
                                    - 1
                                )
                                w_out_numerator = (
                                    shapes[3]
                                    + 2 * padding_w
                                    - dilation_w * (filter_w - 1)
                                    - 1
                                )

                                # Disqualify argument combinations that would cause an illegal
                                # convolution
                                if (h_out_numerator < 0) or (w_out_numerator < 0):
                                    continue

                                # Dilation must evenly divide the tensor. Some of our inputs
                                # intentionally use odd-sized tensors.
                                if (
                                    shapes[2] % dilation_h != 0
                                    or shapes[3] % dilation_w != 0
                                ):
                                    continue

                                arg_list.append(
                                    [
                                        "_st{}{}_pad{}{}_dilat{}{}".format(
                                            stride_h,
                                            stride_w,
                                            padding_h,
                                            padding_w,
                                            dilation_h,
                                            dilation_w,
                                        ),
                                        [
                                            [stride_h, stride_w],
                                            [padding_h, padding_w],
                                            [dilation_h, dilation_w],
                                        ],
                                    ]
                                )
        return arg_list

    def agPooling(op, shapes, rng):
        arg_list = []

        # Must be rank 4
        if len(shapes) < 4:
            return arg_list

        filter_h, filter_w = op["filter"]

        for stride_h in [1, 2]:
            for stride_w in [1, 2]:
                for padding_h in [0, 1]:
                    for padding_w in [0, 1]:
                        # Stride should not be larger than kernel size
                        if (stride_h > filter_h) or (stride_w > filter_w):
                            continue

                        # Pad should be at most half of effective kernel size
                        if (padding_h > filter_h // 2) or (padding_w > filter_w // 2):
                            continue

                        # Disqualify argument combinations that would cause
                        # an illegal convolution
                        if (padding_h == 1) and (shapes[2] < filter_h):
                            continue

                        if (padding_w == 1) and (shapes[3] < filter_w):
                            continue

                        if (padding_h == 0) and (shapes[2] <= filter_h):
                            continue

                        if (padding_w == 0) and (shapes[3] <= filter_w):
                            continue

                        arg_list.append(
                            [
                                "_st{}{}_pad{}{}".format(
                                    stride_h, stride_w, padding_h, padding_w
                                ),
                                [
                                    [stride_h, stride_w],
                                    [filter_h, filter_w],
                                    [padding_h, padding_w],
                                ],
                            ]
                        )
        return arg_list

    def getFactors(val, start=1):
        factors = []
        for i in range(start, int(np.sqrt(val))):
            if (val % i) == 0:
                factors.append(i)

        return factors

    def agFloat(op, shapes, rng):
        args = []

        i = 0
        for alpha in np.float32(rng.random(size=2)):
            args.append(["_{}".format(i), [alpha]])

        return args

    def agFill(op, shapes, rng):
        values = []
        for i in range(4):
            value = rng.integers(0, 10, dtype=np.int32)
            values.append(["_value{}".format(value), [shapes, value]])
        return values

    def getValuesToSum(total, rng):
        # Get a list of random integers that sum up to 'total'
        vals = []

        # np.random.randint() min and max to be different, so if the remainder
        # is 1, give up
        while total > 1:
            vals.append(rng.integers(1, total))
            total = total - vals[-1]

        if total == 1:
            vals.append(1)

        return vals

    def getAdaptivePoolingOutput(n):
        # Get a list of output size for adaptive pooling
        divisors = [1]

        for i in range(n // 2, 1, -1):
            if n % i == 0:
                divisors.append(i)
                break

        divisors.append(n)

        return divisors

    def agAdaptivePooling(op, shapes, rng):
        arg_list = []

        # Must be rank 4
        if len(shapes) < 4:
            return arg_list

        input_h, input_w = shapes[-2:]

        for out_h in ArgGen.getAdaptivePoolingOutput(input_h):
            for out_w in ArgGen.getAdaptivePoolingOutput(input_w):
                arg_list.append(
                    [
                        "_out{}x{}".format(out_h, out_w),
                        [
                            [out_h, out_w],
                        ],
                    ]
                )

        return arg_list

    def agSqueezeDim(op, shapes, rng):
        """Generates arguments for squeeze/unsqueeze dim ops"""
        arg_list = []

        if len(shapes) > 1:
            for dimnum, dim in enumerate(shapes):
                if dim == 1:
                    arg_list.append(["_dim" + str(dimnum), [dimnum]])
                    arg_list.append(["_negdim" + str(dimnum), [dimnum - len(shapes)]])

        return arg_list

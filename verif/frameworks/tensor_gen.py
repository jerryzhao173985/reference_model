# Copyright (c) 2020-2023, ARM Limited.
# SPDX-License-Identifier: Apache-2.0
import enum

import numpy as np
import tensorflow as tf

# FIXME: replace hardcoded '* 2' with random integers, where possible

# The scaling factor for random numbers generated in input tensors.  The
# random numbers are calculated as:
# (np.random.rand() - RAND_SHIFT_FACTOR) * RAND_SCALE_FACTOR
# FIXME: improve range here
RAND_SCALE_FACTOR = 4.0
# Amount to add to random numbers
RAND_SHIFT_FACTOR = 0.5

RAND_INT_MIN = -128
RAND_INT_MAX = 128


class ElemSignedness(enum.Enum):
    ALL_RANGE = 1
    POSITIVE = 2
    NEGATIVE = 3


class TGen:
    """A collection of functions to build tensor value arguments for an operator"""

    def __init__(self):
        pass

    @staticmethod
    def getRand(shape, dtype, rng, elem_signedness=ElemSignedness.ALL_RANGE):
        if elem_signedness == ElemSignedness.POSITIVE:
            RAND_SHIFT_FACTOR = 0
        elif elem_signedness == ElemSignedness.NEGATIVE:
            RAND_SHIFT_FACTOR = 1
        else:
            RAND_SHIFT_FACTOR = 0.5

        if dtype == tf.float32:
            return np.float32(
                (rng.random(size=shape) - RAND_SHIFT_FACTOR) * RAND_SCALE_FACTOR
            )
        if dtype == tf.float16:
            return np.float16(
                (rng.random(size=shape) - RAND_SHIFT_FACTOR) * RAND_SCALE_FACTOR
            )
        if dtype == tf.int32:
            return np.int32(
                rng.integers(low=RAND_INT_MIN, high=RAND_INT_MAX, size=shape)
            )
        if dtype == tf.uint32:
            return np.uint32(rng.integers(low=0, high=RAND_INT_MAX, size=shape))
        if dtype == tf.bool:
            return np.bool_(rng.choice(a=[False, True], size=shape))
        if dtype == tf.complex64:
            return TGen.getRand(shape, np.float32, rng) + 1j * TGen.getRand(
                shape, np.float32, rng
            )

        raise Exception("Unsupported type: {}".format(dtype))

    @staticmethod
    def tgBasicPositive(op, shape, dtype, rng, elem_signedness=ElemSignedness.POSITIVE):
        return TGen.tgBasic(op, shape, dtype, rng, elem_signedness)

    @staticmethod
    def tgBasic(op, shape, dtype, rng, elem_signedness=ElemSignedness.ALL_RANGE):
        # Build random tensor placeholder node args of a given shape
        pl, const = op["operands"]

        tf_placeholders = []
        tf_consts = []

        for i in range(pl):
            tf_placeholders.append(
                (
                    "placeholder_{}".format(i),
                    TGen.getRand(shape, dtype, rng, elem_signedness),
                )
            )

        for i in range(const):
            tf_consts.append(
                ("const_{}".format(i), TGen.getRand(shape, dtype, rng, elem_signedness))
            )

        return tf_placeholders, tf_consts

    @staticmethod
    def tgBFuzz(op, shape, dtype, rng, fuzzed=[]):
        # Build random tensor placeholder node args of a given shape, optionally
        # fuzzing the arguments with random 1's to force broadcasting

        pl, const = op["operands"]

        assert const == 0

        fuzz_arg = rng.integers(0, pl + const)
        fuzz_idx = rng.integers(0, len(shape))

        tf_placeholders = []
        tf_consts = []
        for i in range(pl):
            if not fuzzed and i == fuzz_arg:
                # Insert the broadcast in one dimension index
                s_fuzz = list(shape)
                s_fuzz[fuzz_idx] = 1
                s_fuzz = tuple(s_fuzz)
                i_shape = s_fuzz
                # Record the fuzzed index.
                fuzzed.append(i)
            else:
                i_shape = shape
            tf_placeholders.append(
                ("placeholder_{}".format(i), TGen.getRand(i_shape, dtype, rng))
            )

        return tf_placeholders, tf_consts

    @staticmethod
    def tgConvCommon(op, ifm_shape, filter_shape, out_channels, dtype, rng):

        # Take the shape and generate an input and filter
        tf_placeholders = []
        tf_consts = []
        tf_placeholders.append(("placeholder_0", TGen.getRand(ifm_shape, dtype, rng)))
        tf_consts.append(("const_0", TGen.getRand(filter_shape, dtype, rng)))

        try:
            bias = op["bias"]
        except KeyError:
            bias = False

        if bias:
            # bias is 1D and size == output channels
            bias_shape = (out_channels,)
            tf_consts.append(("const_1", TGen.getRand(bias_shape, dtype, rng)))

        return tf_placeholders, tf_consts

    @staticmethod
    def tgConv2d(op, ifm_shape, dtype, rng):

        # Require rank 4 shape
        if len(ifm_shape) != 4:
            return [], []

        filter_h, filter_w = op["filter"]

        # TODO: Hard-code the test by making the OFM depth 2x the IFM depth.
        # Could randomize this in the future.
        out_channels = ifm_shape[3] * 2
        filter_shape = (filter_h, filter_w, ifm_shape[3], out_channels)

        return TGen.tgConvCommon(op, ifm_shape, filter_shape, out_channels, dtype, rng)

    @staticmethod
    def tgDepthwiseConv2d(op, ifm_shape, dtype, rng):

        # Require rank 4 shape
        if len(ifm_shape) != 4:
            return [], []

        filter_h, filter_w = op["filter"]

        # TODO: Hard-code the test by making the channel_multiplier=2.
        # Could randomize this in the future.
        filter_shape = (filter_h, filter_w, ifm_shape[3], 2)
        out_channels = ifm_shape[3] * 2

        return TGen.tgConvCommon(op, ifm_shape, filter_shape, out_channels, dtype, rng)

    @staticmethod
    def tgTransposeConv2d(op, ifm_shape, dtype, rng):

        # Require rank 4 shape
        if len(ifm_shape) != 4:
            return [], []

        filter_h, filter_w = op["filter"]

        # TODO: Hard-code the test by making the IFM depth 2x the OFM depth.
        # Could randomize this in the future.
        out_channels = ifm_shape[3] * 2
        filter_shape = (filter_h, filter_w, out_channels, ifm_shape[3])

        return TGen.tgConvCommon(op, ifm_shape, filter_shape, out_channels, dtype, rng)

    @staticmethod
    def tgConv3d(op, ifm_shape, dtype, rng):

        # Require rank 5 shape
        if len(ifm_shape) != 5:
            return [], []

        filter_d, filter_h, filter_w = op["filter"]

        # TODO: Hard-code the test by making the OFM depth 2x the IFM depth.
        # Could randomize this in the future.
        in_channels = ifm_shape[4]
        out_channels = in_channels * 2
        filter_shape = (filter_d, filter_h, filter_w, in_channels, out_channels)

        return TGen.tgConvCommon(op, ifm_shape, filter_shape, out_channels, dtype, rng)

    @staticmethod
    def tgPooling(op, shapes, dtype, rng):
        # Pooling does nothing special except filter out non-rank-4 tensors
        if len(shapes) != 4:
            return [], []

        return TGen.tgBasic(op, shapes, dtype, rng)

    @staticmethod
    def tgMatmul(op, ifm_shape, dtype, rng):
        # Take the shape and generate an input and filter
        tf_placeholders = []
        tf_consts = []

        if len(ifm_shape) < 2:
            return [], []

        # For ifm_shape = [..., N, K]
        # Generate rhs tensor with shape [..., K x (2 * N)]
        tf_placeholders.append(("placeholder_0", TGen.getRand(ifm_shape, dtype, rng)))

        shape_rhs = list(ifm_shape)
        shape_rhs[-2] = ifm_shape[-1]
        shape_rhs[-1] = ifm_shape[-2] * 2
        tf_placeholders.append(
            (
                "placeholder_1",
                TGen.getRand(shape_rhs, dtype, rng),
            )
        )

        return tf_placeholders, tf_consts

    @staticmethod
    def tgOneHot(op, shape, dtype, rng):
        # Build random tensor placeholder node args of a given shape
        pl, const = op["operands"]

        assert pl == 3 and const == 1

        tf_placeholders = []
        tf_consts = []

        # depth
        depth = np.int32(rng.integers(low=1, high=32, size=None))
        tf_consts.append(("const_0", depth))

        # indices
        indices = np.int32(rng.integers(low=0, high=depth, size=shape))
        tf_placeholders.append(("placeholder_0", indices))

        # on_value
        tf_placeholders.append(("placeholder_1", TGen.getRand(None, dtype, rng)))

        # off_value
        tf_placeholders.append(("placeholder_2", TGen.getRand(None, dtype, rng)))

        return tf_placeholders, tf_consts

    @staticmethod
    def tgSelect(op, shape, dtype, rng):
        # Build random tensor placeholder node args of a given shape
        pl, const = op["operands"]
        assert pl == 3 and const == 0

        tf_placeholders = []
        tf_consts = []

        # selector
        tf_placeholders.append(("placeholder_0", TGen.getRand(None, tf.bool, rng)))
        # inputs
        tf_placeholders.append(("placeholder_1", TGen.getRand(shape, dtype, rng)))
        tf_placeholders.append(("placeholder_2", TGen.getRand(shape, dtype, rng)))

        return tf_placeholders, tf_consts

    @staticmethod
    def tgRecurrent(op, ifm_shape, dtype, rng):
        # Require rank 3 shape for recurrent networks
        if len(ifm_shape) != 3:
            return [], []
        pl, const = op["operands"]

        tf_placeholders = []
        tf_consts = []

        for i in range(pl):
            tf_placeholders.append(
                ("placeholder_{}".format(i), TGen.getRand(ifm_shape, dtype, rng))
            )

        for i in range(const):
            tf_consts.append(
                ("const_{}".format(i), TGen.getRand(ifm_shape, dtype, rng))
            )

        return tf_placeholders, tf_consts

    @staticmethod
    def tgRFFT2d(op, shape, dtype, rng):
        # Require rank 3 shape
        if len(shape) != 3:
            return [], []

        return TGen.tgBasic(op, shape, dtype, rng)

    @staticmethod
    def tgComplexComponents(op, shape, dtype, rng):
        # Temporarily require up to rank 3 shape, due to
        # slice maximum rank limitiation.
        if len(shape) > 3:
            return [], []

        return TGen.tgBasic(op, shape, dtype, rng)

    @staticmethod
    def tgBroadcastTo(op, shape, dtype, rng):

        pl, const = op["operands"]

        assert pl == 1
        assert const == 1

        tf_placeholders = []
        tf_consts = []

        shape_list = list(shape)
        t_shape_list = []
        s_shape_list = []
        for i in range(len(shape)):
            dim = shape_list[i]
            if rng.integers(0, 1) == 0:
                # append dim in s_shape_list, and 1 in t_shape_list unless it is still empty
                s_shape_list.append(dim)
                if len(t_shape_list) > 0:
                    t_shape_list.append(1)
            else:
                # append 1 in s_shape_list, and dim in t_shape_list
                s_shape_list.append(1)
                t_shape_list.append(dim)

        # if t_shape_list is empty, then insert 1
        if len(t_shape_list) == 0:
            t_shape_list.append(1)

        tf_placeholders.append(
            ("placeholder_0", TGen.getRand(tuple(t_shape_list), dtype, rng))
        )

        tf_consts.append(("shape", tuple(s_shape_list)))

        return tf_placeholders, tf_consts

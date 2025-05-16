# Copyright (c) 2024-2025, ARM Limited.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import torch
from frameworks.test_gen_utils import ElemSignedness
from frameworks.test_gen_utils import RAND_INT_MAX
from frameworks.test_gen_utils import RAND_INT_MIN
from frameworks.test_gen_utils import RAND_SCALE_FACTOR

# FIXME: replace hardcoded '* 2' with random integers, where possible


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

        if dtype == torch.float32:
            if shape != ():
                rfloat32 = np.float32(
                    (rng.random(size=shape) - RAND_SHIFT_FACTOR) * RAND_SCALE_FACTOR
                )
            else:
                rfloat32 = np.float32(rng.random())
            return (
                torch.from_numpy(rfloat32)
                if shape != ()
                else torch.tensor(rfloat32, dtype=torch.float32)
            )
        if dtype == torch.int32:
            rint32 = np.int32(
                rng.integers(low=RAND_INT_MIN, high=RAND_INT_MAX, size=shape)
            )
            return (
                torch.from_numpy(rint32)
                if shape != ()
                else torch.tensor(rint32, dtype=torch.int32)
            )
        if dtype == torch.bool:
            return torch.from_numpy(np.bool_(rng.choice(a=[False, True], size=shape)))

        raise Exception("Unsupported type: {}".format(dtype))

    @staticmethod
    def tgBasicPositive(op, shape, dtype, rng, elem_signedness=ElemSignedness.POSITIVE):
        return TGen.tgBasic(op, shape, dtype, rng, elem_signedness)

    @staticmethod
    def tgBasic(op, shape, dtype, rng, elem_signedness=ElemSignedness.ALL_RANGE):
        # Build random tensor placeholder node args of a given shape
        pl, const = op["operands"]

        torch_placeholders = []
        torch_consts = []

        for i in range(pl):
            torch_placeholders.append(
                (
                    "placeholder_{}".format(i),
                    TGen.getRand(shape, dtype, rng, elem_signedness),
                )
            )

        for i in range(const):
            torch_consts.append(
                ("const_{}".format(i), TGen.getRand(shape, dtype, rng, elem_signedness))
            )

        return torch_placeholders, torch_consts

    @staticmethod
    def tgBFuzz(op, shape, dtype, rng, for_tflite_converter=True):
        # Build random tensor placeholder node args of a given shape, optionally
        # fuzzing the arguments with random 1's to force broadcasting

        pl, const = op["operands"]

        assert const == 0

        if not for_tflite_converter:
            fuzz_arg = rng.integers(0, pl + const)
            fuzz_idx = rng.integers(0, len(shape))

        torch_placeholders = []
        torch_consts = []

        for i in range(pl):
            if not for_tflite_converter and i == fuzz_arg:
                # Insert the broadcast in one dimension index
                s_fuzz = list(shape)
                s_fuzz[fuzz_idx] = 1
                s_fuzz = tuple(s_fuzz)
                i_shape = s_fuzz
            else:
                i_shape = shape

            torch_placeholders.append(
                ("placeholder_{}".format(i), TGen.getRand(i_shape, dtype, rng))
            )

        return torch_placeholders, torch_consts

    @staticmethod
    def tgConvCommon(op, ifm_shape, filter_shape, out_channels, dtype, rng):
        # Take the shape and generate an input and filter
        torch_placeholders = []
        torch_consts = []
        torch_placeholders.append(
            ("placeholder_0", TGen.getRand(ifm_shape, dtype, rng))
        )
        torch_consts.append(("const_0", TGen.getRand(filter_shape, dtype, rng)))

        try:
            bias = op["bias"]
        except KeyError:
            bias = False

        if bias:
            # bias is 1D and size == output channels
            bias_shape = (out_channels,)
            torch_consts.append(("const_1", TGen.getRand(bias_shape, dtype, rng)))

        return torch_placeholders, torch_consts

    @staticmethod
    def tgConv2d(op, ifm_shape, dtype, rng):
        # Require rank 4 shape
        if len(ifm_shape) != 4:
            return [], []

        filter_h, filter_w = op["filter"]

        # TODO: Hard-code the test by making the OFM depth 2x the IFM depth.
        # Could randomize this in the future.
        # Use NCHW data format for PyTorch to calculate filter_shape
        out_channels = ifm_shape[1] * 2
        filter_shape = (out_channels, ifm_shape[1], filter_h, filter_w)

        return TGen.tgConvCommon(op, ifm_shape, filter_shape, out_channels, dtype, rng)

    @staticmethod
    def tgPooling(op, shapes, dtype, rng):
        # Pooling does nothing special except filter out non-rank-4 tensors
        if len(shapes) != 4:
            return [], []

        return TGen.tgBasic(op, shapes, dtype, rng)

    @staticmethod
    def tgMm(op, ifm_shapes, dtype, rng):
        # Take the shape and generate an input and filter
        torch_placeholders = []
        torch_consts = []

        torch_placeholders.append(
            ("placeholder_0", TGen.getRand(ifm_shapes[0], dtype, rng))
        )

        torch_placeholders.append(
            ("placeholder_1", TGen.getRand(list(ifm_shapes[1]), dtype, rng))
        )

        return torch_placeholders, torch_consts

    @staticmethod
    def tgMatmul(op, ifm_shape, dtype, rng):
        # Take the shape and generate an input and filter
        torch_placeholders = []
        torch_consts = []

        if len(ifm_shape) < 2:
            return [], []

        # For ifm_shape = [..., N, K]
        # Generate rhs tensor with shape [..., K x (2 * N)]
        torch_placeholders.append(
            ("placeholder_0", TGen.getRand(ifm_shape, dtype, rng))
        )

        shape_rhs = list(ifm_shape)
        shape_rhs[-2] = ifm_shape[-1]
        shape_rhs[-1] = ifm_shape[-2] * 2
        torch_placeholders.append(
            (
                "placeholder_1",
                TGen.getRand(shape_rhs, dtype, rng),
            )
        )

        return torch_placeholders, torch_consts

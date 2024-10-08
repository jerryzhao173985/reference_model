# Copyright (c) 2021-2024, ARM Limited.
# SPDX-License-Identifier: Apache-2.0
import itertools
import logging
import math
from copy import deepcopy

import generator.tosa_utils as gtu
import numpy as np
from conformance.tosa_profiles import TosaProfiles
from generator.tosa_error_if import ErrorIf
from generator.tosa_error_if import TosaErrorIfArgGen
from serializer.tosa_serializer import DTypeNames
from tosa.DType import DType
from tosa.NanPropagationMode import NanPropagationMode
from tosa.Op import Op
from tosa.ResizeMode import ResizeMode

# DTypeNames, DType, Op and ResizeMode are convenience variables to the
# flatc-generated types that should be enums, but aren't

logging.basicConfig()
logger = logging.getLogger("tosa_verif_build_tests")


class TosaQuantGen:
    """QuantizedInfo random generator helper functions.

    Specify with 'qgen': in the operator defintion.
    """

    def __init__(self):
        pass

    @staticmethod
    def getZeroPoint(rng, zeropoint, dtype, error_name=None):
        if dtype == DType.INT8:
            if zeropoint is not None:
                return min(127, max(-128, zeropoint))
            return rng.randInt(-128, 128)
        elif dtype == DType.UINT8:
            if zeropoint is not None:
                return min(255, max(0, zeropoint))
            return rng.randInt(0, 256)
        elif error_name in [
            ErrorIf.InputZeroPointNotZero,
            ErrorIf.WeightZeroPointNotZero,
            ErrorIf.OutputZeroPointNotZero,
        ]:
            zero_point = rng.randInt(-128, 128)
            if zero_point == 0:
                zero_point = 1
            return zero_point
        return 0

    @staticmethod
    def qgUnary(rng, zeropoint, op, dtype, error_name=None):
        if error_name == ErrorIf.InputZeroPointNotZero:
            qinfo = [
                TosaQuantGen.getZeroPoint(rng, zeropoint, dtype, error_name),
                TosaQuantGen.getZeroPoint(rng, zeropoint, dtype),
            ]
        elif error_name == ErrorIf.OutputZeroPointNotZero:
            qinfo = [
                TosaQuantGen.getZeroPoint(rng, zeropoint, dtype),
                TosaQuantGen.getZeroPoint(rng, zeropoint, dtype, error_name),
            ]
        else:
            qinfo = [
                TosaQuantGen.getZeroPoint(rng, zeropoint, dtype),
                TosaQuantGen.getZeroPoint(rng, zeropoint, dtype),
            ]
        return qinfo

    @staticmethod
    def qgConv(rng, zeropoint, op, dtype_or_dtypeList, error_name=None):
        if isinstance(dtype_or_dtypeList, list):
            # a list of [input, weights, output] dtypes
            dtypeList = dtype_or_dtypeList
        else:
            # an int, [input, weights, output] dtypes are the same
            dtypeList = [dtype_or_dtypeList] * 3

        if error_name == ErrorIf.InputZeroPointNotZero:
            qinfo = [
                TosaQuantGen.getZeroPoint(rng, zeropoint, dtypeList[0], error_name),
                TosaQuantGen.getZeroPoint(rng, zeropoint, dtypeList[1]),
            ]
        elif error_name == ErrorIf.WeightZeroPointNotZero:
            qinfo = [
                TosaQuantGen.getZeroPoint(rng, zeropoint, dtypeList[0]),
                TosaQuantGen.getZeroPoint(rng, zeropoint, dtypeList[1], error_name),
            ]
        else:
            qinfo = [
                TosaQuantGen.getZeroPoint(rng, zeropoint, dtypeList[0]),
                TosaQuantGen.getZeroPoint(rng, zeropoint, dtypeList[1]),
            ]
        return qinfo

    @staticmethod
    def qgMatmul(rng, zeropoint, op, dtype, error_name=None):
        if error_name == ErrorIf.InputZeroPointNotZero:
            qinfo = [
                TosaQuantGen.getZeroPoint(rng, zeropoint, dtype, error_name),
                TosaQuantGen.getZeroPoint(rng, zeropoint, dtype, error_name),
            ]
        else:
            qinfo = [
                TosaQuantGen.getZeroPoint(rng, zeropoint, dtype),
                TosaQuantGen.getZeroPoint(rng, zeropoint, dtype),
            ]
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
        logger.debug(
            f"computeMultiplierAndShift: scalefp={scaleFp} scaleBits={scaleBits} m={m} mult={multiplier} shift={shift}"
        )

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
    data operands for the operator.

    The actual random data is generated separately for each test.
    """

    def __init__(self):
        pass

    @staticmethod
    def _get_basic_shapes(testGen, rng, num_shapes, rank, error_name=None):
        shape = testGen.makeShape(rng, rank)
        # Constrict the overall size of the shape when creating ERROR_IF tests
        if error_name:
            shape = TosaErrorIfArgGen.eiRestrictDimensions(shape)

        shape_list = []
        for i in range(num_shapes):
            shape_list.append(shape.copy())

            # Generates an input rank mismatch for operators with more than one input
            if error_name == ErrorIf.RankMismatch:
                if rank == 1 and i != 1:
                    shape = testGen.makeShape(rng, rank + rng.choice([1, 2, 3]))
                elif i != 1:
                    shape = testGen.makeShape(rng, rank + rng.choice([-1, 1]))

        return shape_list

    @staticmethod
    def tgBasic(testGen, rng, op, rank, error_name=None):
        pl, const = op["operands"]
        return TosaTensorGen._get_basic_shapes(
            testGen, rng, pl + const, rank, error_name
        )

    @staticmethod
    def tgNHWC(testGen, rng, op, rank, error_name=None):
        pl, const = op["operands"]

        if error_name != ErrorIf.WrongRank:
            assert rank == 4

        shape = testGen.makeShape(rng, rank)
        shape = testGen.constrictBatchSize(shape)

        # Constrict the overall size of the shape when creating ERROR_IF tests
        if error_name and error_name != ErrorIf.MaxDimExceeded:
            shape = TosaErrorIfArgGen.eiRestrictDimensions(shape)

        shape_list = []
        for i in range(pl + const):
            shape_list.append(shape.copy())

        return shape_list

    @staticmethod
    def tgGather(testGen, rng, opName, rank, error_name=None):
        pl, const = opName["operands"]

        assert pl == 2
        assert const == 0
        if error_name != ErrorIf.WrongRank:
            assert rank == 3

        values_shape = testGen.makeShape(rng, rank)
        values_shape = testGen.constrictBatchSize(values_shape)

        N = values_shape[0]
        W = testGen.makeDimension(rng)
        indices_shape = [N, W]

        shape_list = [values_shape, indices_shape]
        return shape_list

    @staticmethod
    def tgScatter(testGen, rng, opName, rank, error_name=None):
        pl, const = opName["operands"]

        assert pl == 3
        assert const == 0
        if error_name != ErrorIf.WrongRank:
            assert rank == 3

        values_in_shape = testGen.makeShape(rng, rank)
        values_in_shape = testGen.constrictBatchSize(values_in_shape)

        N = values_in_shape[0]
        K = values_in_shape[1]
        C = values_in_shape[2]

        # Make sure W is not greater than K, as we can only write each output index
        # once (having a W greater than K means that you have to repeat a K index)
        W_min = min(testGen.args.tensor_shape_range[0], K)
        W_max = min(testGen.args.tensor_shape_range[1], K)
        W = rng.randInt(W_min, W_max) if W_min < W_max else W_min

        input_shape = [N, W, C]

        shape_list = []
        shape_list.append(values_in_shape)
        shape_list.append([N, W])  # indices
        shape_list.append(input_shape)

        return shape_list

    @staticmethod
    def _get_broadcast_shapes(testGen, rng, num_shapes, rank, error_name=None):
        if rank == 0:
            # No broadcasting possible for rank 0
            return [[]] * num_shapes

        shape = testGen.makeShape(rng, rank)
        # Do not broadcast for some tests
        if error_name is None and rng.randInt(high=100) < 10:
            return [shape] * num_shapes
        shape_list = []

        # Choose any one of the inputs to broadcast
        # Note for ERRORS: Simplifies OutputShaper code if we don't change first shape
        bcast_idx = rng.randInt(0 if error_name is None else 1, num_shapes)
        fuzz_idx = rng.randInt(0, rank)

        for i in range(num_shapes):
            shape_bcast = shape.copy()

            # To test broadcasting, the chosen fuzz index dimension should not be 1
            if shape_bcast[fuzz_idx] == 1:
                shape_bcast[fuzz_idx] += 1

            # If the chosen input, pick a random index to broadcast
            if i == bcast_idx:
                if error_name == ErrorIf.RankMismatch:
                    # Add one rank to the shape (or more for rank of 1)
                    extra_ranks = rng.choice([1, 2, 3]) if rank == 1 else 1
                    shape_bcast = np.concatenate(
                        (shape_bcast, testGen.makeShape(rng, extra_ranks))
                    )
                    if rank != 1:
                        # Either keep the extra rank, or remove it
                        new_len = rng.choice([-2, len(shape_bcast)])
                        shape_bcast = shape_bcast[:new_len]
                elif error_name == ErrorIf.BroadcastShapesMismatch:
                    shape_bcast[fuzz_idx] += 2
                else:
                    shape_bcast[fuzz_idx] = 1

            shape_list.append(shape_bcast)

        return shape_list

    @staticmethod
    def tgBroadcastFuzz(testGen, rng, op, rank, error_name=None):
        assert (
            op.get("broadcastable_inputs", 0) > 0
        ), "No broadcastable inputs supported"
        pl, const = op["operands"]
        num_shapes = pl + const
        assert (
            op.get("broadcastable_inputs", 0) == num_shapes
        ), "Mismatch between inputs and expected broadcastable shapes"
        return TosaTensorGen._get_broadcast_shapes(
            testGen, rng, num_shapes, rank, error_name
        )

    @staticmethod
    def tgMul(testGen, rng, op, rank, error_name=None):
        # Get broadcast shapes for the first 2 inputs as the 3rd is shift
        shape_list = TosaTensorGen._get_broadcast_shapes(
            testGen, rng, 2, rank, error_name
        )
        # Add a single dimension tensor for shift
        shape_list.append([1])
        return shape_list

    @staticmethod
    def tgConv2D(testGen, rng, op, rank, error_name=None):
        pl, const = op["operands"]

        if error_name != ErrorIf.WrongRank:
            assert rank == 4

        # IFM dimensions are NHWC
        ifm_shape = testGen.makeShape(rng, rank)
        ifm_shape = testGen.constrictBatchSize(ifm_shape)

        # Constrict the overall size of the shape when creating ERROR_IF tests
        if error_name:
            ifm_shape = TosaErrorIfArgGen.eiRestrictDimensions(
                ifm_shape, max_dim=24, max_items=10000
            )

        # Get the filter height/width from the operator parameters
        filter_hw = op["filter"]

        # Generate a random OFM depth
        ofm_depth = testGen.makeDimension(rng)

        # The filter dimensions are OHWI
        filter_shape = np.asarray([ofm_depth, filter_hw[0], filter_hw[1], ifm_shape[3]])

        # The bias is OC or 1 if broadcastable
        try:
            if op["broadcastable_bias"]:
                if rng.choice([True, False]):
                    ofm_depth = 1
        except KeyError:
            pass
        bias_shape = np.asarray([ofm_depth])

        return [ifm_shape, filter_shape, bias_shape]

    @staticmethod
    def tgConv3D(testGen, rng, op, rank, error_name=None):
        pl, const = op["operands"]

        if error_name != ErrorIf.WrongRank:
            assert rank == 5

        # IFM dimensions are NDHWC
        ifm_shape = testGen.makeShape(rng, rank)
        ifm_shape = testGen.constrictBatchSize(ifm_shape)

        # Constrict the overall size of the shape when creating ERROR_IF tests
        if error_name:
            ifm_shape = TosaErrorIfArgGen.eiRestrictDimensions(
                ifm_shape, max_dim=24, max_items=10000
            )

        # Get the filter depth/height/width from the operator parameters
        filter_dhw = op["filter"]

        # Generate a random OFM channel
        ofm_channel = testGen.makeDimension(rng)

        # The filter dimensions are ODHWI
        filter_shape = np.asarray(
            [ofm_channel, filter_dhw[0], filter_dhw[1], filter_dhw[2], ifm_shape[4]]
        )

        # The bias is OC
        bias_shape = np.asarray([ofm_channel])

        return [ifm_shape, filter_shape, bias_shape]

    @staticmethod
    def tgTransposeConv2D(testGen, rng, op, rank, error_name=None):
        pl, const = op["operands"]

        if error_name != ErrorIf.WrongRank:
            assert rank == 4

        # IFM dimensions are NHWC
        ifm_shape = testGen.makeShape(rng, rank)
        ifm_shape = testGen.constrictBatchSize(ifm_shape)

        # Constrict the overall size of the shape when creating ERROR_IF tests
        if error_name:
            ifm_shape = TosaErrorIfArgGen.eiRestrictDimensions(
                ifm_shape, max_dim=24, max_items=10000
            )

        # Get the filter height/width from the operator parameters
        filter_hw = op["filter"]

        # Generate a random OFM depth
        ofm_depth = testGen.makeDimension(rng)

        # The filter dimensions are OHWI
        filter_shape = np.asarray([ofm_depth, filter_hw[0], filter_hw[1], ifm_shape[3]])

        # The bias is OC
        bias_shape = np.asarray([ofm_depth])

        return [ifm_shape, filter_shape, bias_shape]

    @staticmethod
    def tgDepthwiseConv2D(testGen, rng, op, rank, error_name=None):
        pl, const = op["operands"]

        if error_name != ErrorIf.WrongRank:
            assert rank == 4
        assert pl == 1 and const == 2

        # IFM dimensions are NHWC
        ifm_shape = testGen.makeShape(rng, rank)
        ifm_shape = testGen.constrictBatchSize(ifm_shape)

        # Constrict the overall size of the shape when creating ERROR_IF tests
        if error_name:
            ifm_shape = TosaErrorIfArgGen.eiRestrictDimensions(
                ifm_shape, max_dim=24, max_items=10000
            )

        # Get the filter height/width from the operator parameters
        # Filter is KH, HW, C, M
        filter_hw = op["filter"]

        # Generate a random OFM depth, but don't let it get too big because
        # the output depth is M * C
        filter_m = (
            testGen.makeDimension(rng) % (testGen.args.tensor_shape_range[1] // 4)
        ) + 1

        # The filter dimensions are HWCM
        filter_shape = np.asarray([filter_hw[0], filter_hw[1], ifm_shape[3], filter_m])

        # The bias is M * C
        bias_shape = np.asarray([ifm_shape[3] * filter_m])

        return [ifm_shape, filter_shape, bias_shape]

    @staticmethod
    def tgFFT2d(testGen, rng, op, rank, error_name=None):
        pl, const = op["operands"]

        if error_name != ErrorIf.WrongRank:
            assert rank == 3
        assert pl == 2 and const == 0

        # IFM dimensions are NHW
        ifm_shape = testGen.makeShape(rng, rank)

        # Select nearest lower power of two from input height and width
        ifm_shape[1] = 2 ** int(math.log(ifm_shape[1], 2))
        ifm_shape[2] = 2 ** int(math.log(ifm_shape[2], 2))

        # Constrict the overall size of the shape when creating ERROR_IF tests
        if error_name:
            ifm_shape = TosaErrorIfArgGen.eiRestrictDimensions(ifm_shape)

        # Generate an invalid kernel that is not a power of two
        if error_name == ErrorIf.KernelNotPowerOfTwo:
            inc_h = 2 if ifm_shape[1] == 1 else 1
            inc_w = 2 if ifm_shape[2] == 1 else 1
            inc_choices = [(inc_h, 0), (0, inc_w), (inc_h, inc_w)]
            selected_inc = rng.choice(inc_choices)
            ifm_shape[1] += selected_inc[0]
            ifm_shape[2] += selected_inc[1]

        ifm_shape = testGen.constrictBatchSize(ifm_shape)

        ifm_shapes = [ifm_shape.copy(), ifm_shape.copy()]
        if error_name == ErrorIf.FFTInputShapeMismatch:
            modify_shape = rng.choice([0, 1])
            # Only modify kernel (H, W)
            modify_dim = rng.choice([1, 2])
            ifm_shapes[modify_shape][modify_dim] *= 2

        return [ifm_shapes[0], ifm_shapes[1]]

    @staticmethod
    def tgRFFT2d(testGen, rng, op, rank, error_name=None):
        pl, const = op["operands"]

        if error_name != ErrorIf.WrongRank:
            assert rank == 3
        assert pl == 1 and const == 0

        # IFM dimensions are NHW
        ifm_shape = testGen.makeShape(rng, rank)

        # Select nearest lower power of two from input height and width
        ifm_shape[1] = 2 ** int(math.log(ifm_shape[1], 2))
        ifm_shape[2] = 2 ** int(math.log(ifm_shape[2], 2))

        # Constrict the overall size of the shape when creating ERROR_IF tests
        if error_name:
            ifm_shape = TosaErrorIfArgGen.eiRestrictDimensions(ifm_shape)

        # Generate an invalid kernel that is not a power of two
        if error_name == ErrorIf.KernelNotPowerOfTwo:
            # We must increment by 2 if current size is 1
            inc_h = 2 if ifm_shape[1] == 1 else 1
            inc_w = 2 if ifm_shape[2] == 1 else 1
            inc_choices = [(inc_h, 0), (0, inc_w), (inc_h, inc_w)]
            selected_inc = rng.choice(inc_choices)
            ifm_shape[1] += selected_inc[0]
            ifm_shape[2] += selected_inc[1]

        ifm_shape = testGen.constrictBatchSize(ifm_shape)

        return [ifm_shape]

    @staticmethod
    def tgMatmul(testGen, rng, op, rank, error_name=None):
        pl, const = op["operands"]

        if error_name != ErrorIf.WrongRank:
            assert rank == 3
        assert pl == 2 and const == 0

        a_shape = testGen.makeShape(rng, rank)

        # Constrict the overall size of the shape when creating ERROR_IF tests
        if error_name:
            a_shape = TosaErrorIfArgGen.eiRestrictDimensions(a_shape)

        # Get a random number for b_oc even if target shape is defined
        b_oc = np.int32(
            rng.integers(
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
    def tgConcat(testGen, rng, op, rank, error_name=None):
        pl, const = op["operands"]
        shape = testGen.makeShape(rng, rank)

        # Create extra tensors to concat.
        # Take into account value of pl when getting maximum number of concats
        num_tensors = rng.randInt(0, 4)
        shape_list = []
        for i in range(pl + const + num_tensors):
            if error_name == ErrorIf.ConcatInputRankMismatch and i != 0:
                remove = rng.choice([True, False])
                wrongShape = shape.copy()

                if remove and len(shape) > 1:
                    wrongShape = wrongShape[1:]
                else:
                    wrongShape = list(wrongShape)
                    wrongShape.append(rng.integers(1, 10))

                shape_list.append(wrongShape)
            else:
                shape_list.append(shape.copy())

        return shape_list

    @staticmethod
    def tgConcatConstInput(rng, shapeList, axis, error_name=None):
        if error_name in [
            ErrorIf.AxisSmallerZero,
            ErrorIf.AxisLargerRank,
            ErrorIf.ConcatInputRankMismatch,
        ]:
            return shapeList

        # Split concat shape along axis to allow for multiple const inputs
        # without making too many large tensors
        if len(shapeList) == 2 or shapeList[0][axis] < len(shapeList):
            # If axis can't be split we still need to invalidate other dimensions
            if error_name == ErrorIf.ConcatInputDimMismatch:
                for shape in shapeList[1:]:
                    # Negative test shapeLists are created individually for each test,
                    # so no need to copy the shape before altering it.
                    shape[(axis + 1) % len(shape)] += rng.integers(5, 10)
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
                shape[(axis + 1) % len(shape)] += rng.integers(5, 10)
            else:
                shape[axis] = remaining_length

            if i == len(shapeList) - 3:
                new_shapeList.append(shape.copy())

        return new_shapeList

    @staticmethod
    def tgWhileLoop(testGen, rng, op, rank, error_name=None):
        pl, const = op["operands"]
        assert pl == 2 and const == 3, "Unsupported tensors for WHILE_LOOP"

        # Get a tensor for the body of the loop - used as a constant and an input
        body_shape = TosaTensorGen._get_basic_shapes(testGen, rng, 1, rank, error_name)[
            0
        ]

        # Default iteration value shape is rank 0
        iter_shape = []

        # Create the shape list for body and iteration tensors
        # pl_body_tens, pl_iter_tens, const_body_tens, const_iterchk_tens, const_itersub_tens
        return [body_shape, iter_shape, body_shape, iter_shape, iter_shape]


class TosaTensorValuesGen:
    """Tensor Value generators create the random data for each tensor in each test."""

    def __init__(self):
        pass

    class TVGInfo:
        """Enhanced tensor values information including data gen dict."""

        def __init__(self, tensorList, dataGenDict):
            self.tensorList = tensorList
            self.dataGenDict = dataGenDict

    # Default high value for random numbers
    TVG_FLOAT_HIGH_VALUE = {
        DType.FP32: (1 << 128) - (1 << (127 - 23)),
        DType.FP16: (1 << 16) - (1 << (15 - 10)),
        DType.BF16: (1 << 128) - (1 << (127 - 7)),
        DType.FP8E4M3: 448,
        DType.FP8E5M2: 57344,
    }

    # Default lowest normal values for random numbers
    TVG_FLOAT_LOW_VALUE = {
        DType.FP32: np.exp2(-126),
        DType.FP16: np.exp2(-14),
        DType.BF16: np.exp2(-126),
        DType.FP8E4M3: np.exp2(-9),
        DType.FP8E5M2: np.exp2(-16),
    }

    @staticmethod
    def _get_data_range(rng, dtype, highValueLookup, lowValueLookup=None):
        # Return a tuple of (low,high) data range values for the given data
        # type using a combination of per operator table limits, data limits
        # and user supplied ranges for FP numbers
        if dtype in highValueLookup:
            type_range = rng.dTypeRange(dtype, high_inclusive=True)
            high_val = highValueLookup[dtype]
            if lowValueLookup is not None and dtype in lowValueLookup:
                low_val = lowValueLookup[dtype]
            else:
                low_val = -high_val
            # Set the values to something that won't produce infinity whilst
            # respecting the default ranges if more/less than the low/high
            # values
            data_range = (
                max(low_val, type_range[0]),
                min(high_val, type_range[1]),
            )
            if data_range[0] > data_range[1]:
                # Invalid data range from low to high created due to user
                # constraints revert to using internal ranges as they are
                # known to work
                logger.info(
                    f"Using safe data range ({low_val} to {high_val}) instead of supplied ({type_range[0]} to {type_range[1]})"
                )
                data_range = (low_val, high_val)
            return data_range
        return None

    @staticmethod
    def _get_special_test_set(opInfo, argsDict):
        if opInfo["op"] == Op.CAST and not gtu.dtypeIsFloat(argsDict["out_type"]):
            return gtu.SpecialTestSet.CAST_FP_TO_INT
        return gtu.SpecialTestSet.DEFAULT

    @staticmethod
    def tvgLazyGenDefault(
        testGen, rng, opName, dtypeList, shapeList, argsDict, error_name=None
    ):
        def tensor_is_variable(pCount, idx):
            if dtypeList[idx] == DType.SHAPE:
                # Shapes must always be CONST_SHAPEs
                return False

            # Determine if the tensor is constant or variable (a placeholder)
            if testGen.args.random_const_inputs:
                # Choose type of tensor biased by defaults
                percentage = rng.randInt(0, 100)
                variable = (idx < pCount and percentage < 70) or (
                    idx >= pCount and percentage >= 70
                )
            else:
                # Use default set up of constants versus inputs for the op
                variable = idx < pCount
            return variable

        def get_data_range(idx):
            round_mode = False
            data_range = None
            if "data_range" in argsDict:
                data_range = argsDict["data_range"]
            elif (
                "data_range_list" in argsDict
                and argsDict["data_range_list"][idx] is not None
            ):
                data_range = argsDict["data_range_list"][idx]["range"]
                round_mode = argsDict["data_range_list"][idx].get("round", False)
            return (data_range, round_mode)

        # Variable inputs versus constants
        pCount, cCount = testGen.TOSA_OP_LIST[opName]["operands"]
        if "p_count" in argsDict:
            # Override for operators like CONCAT
            pCount = argsDict["p_count"]
            cCount = argsDict["c_count"]
        assert pCount + cCount == len(
            shapeList
        ), "Placeholders & Constant tensors must match shapes list"

        # Check if we need to create the serialization tensors
        serialize_data = argsDict.get("serialization", True)
        if not serialize_data:
            # Create data storage area in the argsDict
            argsDict["tensor_data"] = []

        tens_ser_list = []

        # Retrieve any fixed data tensors
        fixed_data_tensors = argsDict.get("fixed_data", [None] * len(shapeList))
        assert len(fixed_data_tensors) == len(
            shapeList
        ), "Fixed data list must match shapes list"

        if (
            error_name is not None
            or not gtu.dtypeIsSupportedByDataGen(dtypeList[0])
            or "data_gen" not in testGen.TOSA_OP_LIST[opName]
        ):
            # Fall back to internal data gen when dealing with unsupported types or ops
            for idx, info in enumerate(zip(shapeList, dtypeList)):
                data_range, round_mode = get_data_range(idx)
                shape, dtype = info
                if data_range is not None and not gtu.dtypeIsFloat(dtype):
                    # Change from inclusive to exclusive range
                    data_range = (data_range[0], data_range[1] + 1)

                if fixed_data_tensors[idx] is not None:
                    size = gtu.product(shape)
                    data = np.array(fixed_data_tensors[idx])
                    if size > 1 and data.size == 1:
                        # Broadcast the data to the correct size
                        data = np.broadcast_to(data, shape)
                    assert (
                        data.size == size
                    ), "Fixed data length does not match tensor size"
                    if dtype == DType.SHAPE:
                        arr = np.int64(data)
                    elif dtype == DType.INT8:
                        arr = np.int8(data)
                    elif dtype == DType.INT16:
                        arr = np.int16(data)
                    else:
                        # Treat data as int32
                        arr = np.int32(data)
                else:
                    arr = rng.randTensor(shape, dtype, data_range)
                if round_mode:
                    arr = np.round(arr)

                if serialize_data:
                    if tensor_is_variable(pCount, idx):
                        tens_ser_list.append(
                            testGen.ser.addPlaceholder(shape, dtype, arr)
                        )
                    else:
                        tens_ser_list.append(testGen.ser.addConst(shape, dtype, arr))
                else:
                    # We will do the serialization later
                    argsDict["tensor_data"].append(
                        {"dtype": dtype, "shape": shape, "data": arr}
                    )
            return TosaTensorValuesGen.TVGInfo(tens_ser_list, None)

        # Create data generator meta-data
        tens_data = {
            "version": "0.1",
            "tensors": {},
        }
        dg_tens_meta = tens_data["tensors"]

        fp_special_info = {}
        fp_special_info["start_idx"] = int(rng.randInt())
        fp_special_info["special_test_set"] = TosaTensorValuesGen._get_special_test_set(
            testGen.TOSA_OP_LIST[opName], argsDict
        ).name

        if argsDict["dg_type"] == gtu.DataGenType.FP_SPECIAL:
            broadcastable_inputs = testGen.TOSA_OP_LIST[opName].get(
                "broadcastable_inputs", 0
            )
            if broadcastable_inputs > 0:
                shapes_set = {tuple(x) for x in shapeList[:broadcastable_inputs]}
                assert len(shapes_set) == 1, "Broadcast shapes found in FP special test"

        for idx, shape in enumerate(shapeList):
            tens_meta = {}
            dtype = dtypeList[idx]

            if fixed_data_tensors[idx] is not None:
                dg_type = gtu.DataGenType.FIXED_DATA
            else:
                dg_type = argsDict["dg_type"]

            operand_idx_str = "operand" + str(idx)
            dg_override = testGen.TOSA_OP_LIST[opName].get("data_gen_override", {})
            dg_type = dg_override.get(operand_idx_str, dg_type)

            tens_meta["generator"] = gtu.DataGenType(dg_type).name
            tens_meta["data_type"] = gtu.DTYPE_ATTRIBUTES[dtype]["json"]
            tens_meta["shape"] = [int(i) for i in shape]
            tens_meta["input_pos"] = idx
            tens_meta["op"] = testGen.getOperatorNameStr(opName).upper()

            variable = tensor_is_variable(pCount, idx)

            if variable:
                tens_meta["input_type"] = "VARIABLE"
            else:
                tens_meta["input_type"] = "CONSTANT"

            if dg_type == gtu.DataGenType.FIXED_DATA:
                info = {}
                info["data"] = [int(i) for i in fixed_data_tensors[idx]]
                tens_meta["fixed_data_info"] = info

            elif dg_type == gtu.DataGenType.PSEUDO_RANDOM:
                info = {}
                info["rng_seed"] = rng.getDataGenSeed(idx)

                data_range, round_mode = get_data_range(idx)
                if data_range is None:
                    data_range = rng.dTypeRange(dtype, high_inclusive=True)
                info["range"] = [str(v) for v in data_range]
                if round_mode:
                    info["round"] = round_mode
                tens_meta["pseudo_random_info"] = info

            elif dg_type == gtu.DataGenType.DOT_PRODUCT:
                info = {}
                info["s"] = argsDict["s"]
                info["ks"] = int(argsDict["ks"])
                if "acc_type" in argsDict:
                    # Convert type number into JSON name
                    info["acc_type"] = gtu.DTYPE_ATTRIBUTES[argsDict["acc_type"]][
                        "json"
                    ]
                if "kernel" in argsDict:
                    info["kernel"] = [int(k) for k in argsDict["kernel"]]
                if "axis" in argsDict:
                    info["axis"] = int(argsDict["axis"])
                if "otherInputType" in argsDict:
                    info["otherInputType"] = gtu.DTYPE_ATTRIBUTES[
                        argsDict["otherInputType"]
                    ]["json"]
                tens_meta["dot_product_info"] = info

            elif dg_type == gtu.DataGenType.FULL_RANGE:
                info = {}
                info["start_val"] = int(
                    rng.randInt(0, gtu.DTYPE_ATTRIBUTES[dtype]["fullset"])
                )
                tens_meta["full_range_info"] = info

            elif dg_type == gtu.DataGenType.FP_SPECIAL:
                # Each tensor has its own seed
                fp_special_info_tensor = fp_special_info.copy()
                fp_special_info_tensor["rng_seed"] = rng.getDataGenSeed(idx)
                tens_meta["fp_special_info"] = fp_special_info_tensor

            else:
                # TODO - other data gen type
                assert False, "TODO: support other data gen types"

            # Using the finished generate config meta data - generate the data if
            # needed and assign a tensor name from the serializer

            # Need to generate data when not lazy or for the bias tensor as we need
            # to work out if the bias data is non-zero for compliance
            if not testGen.args.lazy_data_gen or (
                idx == 2 and dg_type == gtu.DataGenType.DOT_PRODUCT
            ):
                # Give this tensor a temporary name until we get one from the serializer
                temp_name = f"placeholder_{idx}"
                dg_tens_meta[temp_name] = tens_meta

                # Create data now using the temporary name to access meta details
                data = testGen.dgl.get_tensor_data(temp_name, tens_data)
                if tens_meta["data_type"] == "SHAPE":
                    # Tensor type SHAPE and Numpy file type must be the same
                    data = np.int64(data)
                # Remove the item as we will give it the correct name later
                del dg_tens_meta[temp_name]

            if idx == 2 and dg_type == gtu.DataGenType.DOT_PRODUCT:
                # The KS value used by compliance verification is altered when the
                # bias data is non-zero, store this in ksb_increment for tensorComplianceMetaData
                argsDict["ksb_increment"] = 1 if max(abs(data)) > 0.0 else 0

            if testGen.args.lazy_data_gen:
                data = None

            if serialize_data:
                if variable:
                    tens = testGen.ser.addPlaceholder(shape, dtype, data)
                else:
                    tens = testGen.ser.addConst(shape, dtype, data)

                tens_ser_list.append(tens)
                # Add the meta data to the list using the serializer tensor name
                dg_tens_meta[tens.name] = tens_meta
            else:
                # We will do the serialization later
                tdata = {
                    "dtype": dtype,
                    "shape": shape,
                    "data": data,
                    "meta": tens_meta,
                }
                argsDict["tensor_data"].append(tdata)

        return TosaTensorValuesGen.TVGInfo(tens_ser_list, tens_data)

    @staticmethod
    def tvgNegate(
        testGen, rng, opName, dtypeList, shapeList, argsDict, error_name=None
    ):
        if dtypeList[0] == DType.INT32 and error_name is None:
            # Integer test
            op = testGen.TOSA_OP_LIST[opName]
            pCount, cCount = op["operands"]
            assert (
                pCount == 1 and cCount == 0
            ), "Op.NEGATE must have 1 placeholders, 0 consts"
            # Must create tensors with values within accumulator (int32) negatable
            # range
            max_val = (1 << 31) - 1
            min_val = -max_val
            arr = rng.randTensor(
                shapeList[0], dtypeList[0], data_range=(min_val, (max_val + 1))
            )
            tens_ser_list = []
            tens_ser_list.append(
                testGen.ser.addPlaceholder(shapeList[0], dtypeList[0], arr)
            )
            return TosaTensorValuesGen.TVGInfo(tens_ser_list, None)
        else:
            # ERROR_IF or floating point test
            return TosaTensorValuesGen.tvgLazyGenDefault(
                testGen, rng, opName, dtypeList, shapeList, argsDict, error_name
            )

    # Set the ADD/SUB data range to half the largest value to avoid infinities
    TVG_FLOAT_HIGH_VALUE_ADDSUB = {
        DType.FP32: (TVG_FLOAT_HIGH_VALUE[DType.FP32] / 2),
        DType.FP16: (TVG_FLOAT_HIGH_VALUE[DType.FP16] / 2),
        DType.BF16: (TVG_FLOAT_HIGH_VALUE[DType.BF16] / 2),
        DType.FP8E4M3: (TVG_FLOAT_HIGH_VALUE[DType.FP8E4M3] / 2),
        DType.FP8E5M2: (TVG_FLOAT_HIGH_VALUE[DType.FP8E5M2] / 2),
    }

    @staticmethod
    def tvgAddSub(
        testGen, rng, opName, dtypeList, shapeList, argsDict, error_name=None
    ):
        if dtypeList[0] in (DType.INT32, DType.SHAPE) and error_name is None:
            # Make sure the integer operation does not cause value saturation - where
            # the number wraps due to limited number of bits to store the answer
            op = testGen.TOSA_OP_LIST[opName]
            pCount, cCount = op["operands"]
            assert (
                pCount == 2 and cCount == 0
            ), "Op.ADD / Op.SUB must have 2 placeholders, 0 consts"
            tens_ser_list = []
            add = op["op"] == Op.ADD
            data_range = None  # Use default
            a_arr = rng.randTensor(shapeList[0], dtypeList[0], data_range)
            b_arr = rng.randTensor(shapeList[1], dtypeList[1], data_range)
            if add:
                res_arr = np.add(a_arr, b_arr, dtype=np.int64)
            else:
                res_arr = np.subtract(a_arr, b_arr, dtype=np.int64)

            # Work out the saturation limits
            max_i32 = (1 << 31) - 1
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
                        assert (
                            dim == 1
                        ), "Op.ADD / SUB dimension must be 1 or matching to be broadcastable"
                        b_unsat_arr = np.amin(b_unsat_arr, axis=axis, keepdims=True)

            if (sat_min_arr != 0).any():
                # Clip values that cause saturation
                b_unsat_arr = np.subtract(b_unsat_arr, sat_min_arr, dtype=np.int32)
                # Reduce axes in unsaturated tensor to match original tensor
                for axis, dim in enumerate(b_arr.shape):
                    if dim != b_unsat_arr.shape[axis]:
                        assert (
                            dim == 1
                        ), "Op.ADD / SUB dimension must be 1 or matching to be broadcastable"
                        b_unsat_arr = np.amax(b_unsat_arr, axis=axis, keepdims=True)

            tens_ser_list.append(
                testGen.ser.addPlaceholder(shapeList[0], dtypeList[0], a_arr)
            )
            tens_ser_list.append(
                testGen.ser.addPlaceholder(shapeList[1], dtypeList[1], b_unsat_arr)
            )

            return TosaTensorValuesGen.TVGInfo(tens_ser_list, None)
        else:
            # ERROR_IF or floating point test
            data_range = TosaTensorValuesGen._get_data_range(
                rng, dtypeList[0], TosaTensorValuesGen.TVG_FLOAT_HIGH_VALUE_ADDSUB
            )
            if data_range:
                argsDict["data_range"] = data_range

            return TosaTensorValuesGen.tvgLazyGenDefault(
                testGen, rng, opName, dtypeList, shapeList, argsDict, error_name
            )

    @staticmethod
    def tvgCondIf(
        testGen, rng, opName, dtypeList, shapeList, argsDict, error_name=None
    ):
        dtype = dtypeList[0]
        if opName == "cond_if_const":
            # We don't want to serialize the tensors until build_cond_if_const
            argsDict["serialization"] = False
        else:
            assert opName == "cond_if_binary", f"Unexpected COND_IF op {opName}"
            if dtype == DType.INT32:
                # Limit data range to avoid saturation in add/sub
                argsDict["data_range"] = rng.dTypeRange(DType.INT16)
            elif dtype == DType.INT16:
                # Limit logical shift values
                argsDict["data_range"] = (0, 15)
            elif dtype == DType.INT8:
                # Limit logical shift values
                argsDict["data_range"] = (0, 7)
            elif not gtu.dtypeIsFloat(dtype):
                assert (
                    False
                ), f"No COND_IF binary generation support for Dtype: {testGen.typeStr(dtype)}"

        return TosaTensorValuesGen.tvgLazyGenDefault(
            testGen, rng, opName, dtypeList, shapeList, argsDict, error_name
        )

    @staticmethod
    def tvgWhileLoop(
        testGen, rng, opName, dtypeList, shapeList, argsDict, error_name=None
    ):
        def scale_add_values_to_iterations(iterations, values):
            # Avoid exceeding maximum values by scaling to number of iterations,
            # so that the ADD operation does not saturate
            if iterations > 1:
                new_values = []
                for index in range(len(values)):
                    new_values.append(values[index] // iterations)
                return tuple(new_values)
            return values

        dtype = dtypeList[0]
        iterations = argsDict["iterations"]
        body_data_range = None

        if gtu.dtypeIsFloat(dtype):
            # Create a range using a higher limit table of high value scaled to iterations
            add_op = True
            assert (
                dtype in TosaTensorValuesGen.TVG_FLOAT_HIGH_VALUE
            ), "Unsupported FP type in WHILE_LOOP values gen"
            high_value = TosaTensorValuesGen.TVG_FLOAT_HIGH_VALUE[dtype]

            high_value = scale_add_values_to_iterations(iterations, [high_value])[0]
            body_data_range = TosaTensorValuesGen._get_data_range(
                rng, dtype, {dtype: high_value}
            )
        elif dtype == DType.INT32:
            # Scale the data range to allow multiple ADDs without saturation
            add_op = True
            body_data_range = scale_add_values_to_iterations(
                iterations, rng.dTypeRange(dtype)
            )
        elif dtype in (DType.INT8, DType.INT16):
            # Allow any positive values for LOGICAL_RIGHT_SHIFT
            add_op = False
            body_data_range = (0, rng.dTypeRange(dtype)[1])

        assert (
            body_data_range is not None
        ), "Unsupported data type for WHILE_LOOP values generation"

        # Sort out data types, data ranges and fixed data
        shapes = len(shapeList)
        assert shapes == 5, "Unexpected tensor list in WHILE_LOOP values generation"
        data_range_list = [None] * shapes
        fixed_data = [None] * shapes

        # pl_body_tens
        if add_op:
            # ADD Accumulator set to zero
            fixed_data[0] = np.array([0], dtype=np.int32)
        else:
            # Tensor to perform LOGICAL_RIGHT_SHIFT on
            data_range_list[0] = {"range": body_data_range}

        # pl_iter_tens - iterations tensor
        dtypeList[1] = DType.INT32
        fixed_data[1] = np.array([argsDict["iterations"]], dtype=np.int32)

        # const_body_tens
        if add_op:
            # Tensor to ADD to accumulator per loop
            data_range_list[2] = {"range": body_data_range}
        else:
            # Tensor of shift values set to 1
            fixed_data[2] = np.array([1], dtype=np.int32)

        # const_iterchk_tens - iteraions value to check against (0)
        dtypeList[3] = DType.INT32
        fixed_data[3] = np.array([0], dtype=np.int32)

        # const_itersub_tens - value to subtract from iterations (1)
        dtypeList[4] = DType.INT32
        fixed_data[4] = np.array([1], dtype=np.int32)

        argsDict["data_range_list"] = data_range_list
        argsDict["fixed_data"] = fixed_data

        return TosaTensorValuesGen.tvgLazyGenDefault(
            testGen, rng, opName, dtypeList, shapeList, argsDict, error_name
        )

    @staticmethod
    def tvgArithmeticRightShift(
        testGen, rng, opName, dtypeList, shapeList, argsDict, error_name=None
    ):
        op = testGen.TOSA_OP_LIST[opName]
        pCount, cCount = op["operands"]
        # Force value of operand[1] to be within [0, num_bits]
        assert (
            pCount == 2 and cCount == 0
        ), "Op.ArithmeticRightShift must have 2 placeholders, 0 consts"

        tens_ser_list = []
        for idx, shape in enumerate(shapeList[:]):
            if idx == 1:
                if dtypeList[idx] == DType.INT8:
                    arr = rng.randTensor(shape, dtypeList[idx], data_range=(0, 8))
                elif dtypeList[idx] == DType.INT16:
                    arr = rng.randTensor(shape, dtypeList[idx], data_range=(0, 16))
                elif dtypeList[idx] == DType.INT32:
                    arr = rng.randTensor(shape, dtypeList[idx], data_range=(0, 32))
                elif error_name == ErrorIf.WrongInputType:
                    arr = rng.randTensor(shape, DType.INT32, data_range=(0, 8))
                else:
                    raise Exception("OpArithmeticRightShift: invalid input dtype")
            else:
                arr = rng.randTensor(shape, dtypeList[idx])
            tens_ser_list.append(testGen.ser.addPlaceholder(shape, dtypeList[idx], arr))

        return TosaTensorValuesGen.TVGInfo(tens_ser_list, None)

    @staticmethod
    def tvgReshape(
        testGen, rng, opName, dtypeList, shapeList, argsDict, error_name=None
    ):
        dtypeList[1] = DType.SHAPE
        # Check for rank 0 shapes
        size_shape = len(argsDict["new_shape"])
        shapeList[1] = [size_shape] if size_shape > 0 else []
        # Create a new list for the pre-generated data in argsDict["fixed_data"]
        argsDict["fixed_data"] = [
            None,
            argsDict["new_shape"] if size_shape > 0 else [0],
        ]

        return TosaTensorValuesGen.tvgLazyGenDefault(
            testGen, rng, opName, dtypeList, shapeList, argsDict, error_name
        )

    @staticmethod
    def tvgRescale(
        testGen, rng, opName, dtypeList, shapeList, argsDict, error_name=None
    ):
        scale32 = argsDict["scale"]
        multiplier_arr = argsDict["multiplier"]
        shift_arr = argsDict["shift"]

        if scale32:
            dtypeList[1] = DType.INT32
        else:
            dtypeList[1] = DType.INT16
        shapeList[1] = [len(multiplier_arr)]
        dtypeList[2] = DType.INT8
        shapeList[2] = [len(shift_arr)]
        # Create a new list for the pre-generated data in argsDict["fixed_data"]
        argsDict["fixed_data"] = [None, multiplier_arr, shift_arr]

        return TosaTensorValuesGen.tvgLazyGenDefault(
            testGen, rng, opName, dtypeList, shapeList, argsDict, error_name
        )

    @staticmethod
    def tvgPad(testGen, rng, opName, dtypeList, shapeList, argsDict, error_name=None):
        # argsDict["pad"] is 2D array, need to flatten it to get list of values
        pad_values = argsDict["pad"].flatten()
        dtypeList[1] = DType.SHAPE
        shapeList[1] = [len(pad_values)]
        # Create a new list for the pre-generated data in argsDict["fixed_data"]
        argsDict["fixed_data"] = [None, pad_values]

        return TosaTensorValuesGen.tvgLazyGenDefault(
            testGen, rng, opName, dtypeList, shapeList, argsDict, error_name
        )

    @staticmethod
    def tvgSlice(testGen, rng, opName, dtypeList, shapeList, argsDict, error_name=None):
        dtypeList[1] = DType.SHAPE
        shapeList[1] = [len(argsDict["start"])]
        dtypeList[2] = DType.SHAPE
        shapeList[2] = [len(argsDict["size"])]
        # Create a new list for the pre-generated data in argsDict["fixed_data"]
        argsDict["fixed_data"] = [None, argsDict["start"], argsDict["size"]]

        return TosaTensorValuesGen.tvgLazyGenDefault(
            testGen, rng, opName, dtypeList, shapeList, argsDict, error_name
        )

    @staticmethod
    def tvgTile(testGen, rng, opName, dtypeList, shapeList, argsDict, error_name=None):
        dtypeList[1] = DType.SHAPE
        shapeList[1] = [len(argsDict["multiples"])]
        argsDict["fixed_data"] = [None, argsDict["multiples"]]

        return TosaTensorValuesGen.tvgLazyGenDefault(
            testGen, rng, opName, dtypeList, shapeList, argsDict, error_name
        )

    @staticmethod
    def tvgSelect(
        testGen, rng, opName, dtypeList, shapeList, argsDict, error_name=None
    ):
        # Set datatype of condition tensor to boolean
        dtypeList[0] = DType.BOOL

        return TosaTensorValuesGen.tvgLazyGenDefault(
            testGen, rng, opName, dtypeList, shapeList, argsDict, error_name
        )

    @staticmethod
    def tvgIntDiv(
        testGen, rng, opName, dtypeList, shapeList, argsDict, error_name=None
    ):
        if error_name is None:
            op = testGen.TOSA_OP_LIST[opName]
            pCount, cCount = op["operands"]
            assert (
                pCount == 2 and cCount == 0
            ), "Op.INTDIV must have 2 placeholders, 0 consts"

            tens_ser_list = []

            # Two invalid cases for Op.INTDIV:
            # 1. divisor == 0
            # 2. dividend == -(1<<31) and divisor == -1
            while True:
                dividend_arr = rng.randTensor(shapeList[0], dtypeList[0])
                divisor_arr = rng.randTensor(shapeList[1], dtypeList[1])

                if (divisor_arr == 0).any():
                    continue

                if (dividend_arr == -(2**31)).any() and (divisor_arr == -1).any():
                    continue

                break

            tens_ser_list.append(
                testGen.ser.addPlaceholder(shapeList[0], dtypeList[0], dividend_arr)
            )
            tens_ser_list.append(
                testGen.ser.addPlaceholder(shapeList[1], dtypeList[1], divisor_arr)
            )

            return TosaTensorValuesGen.TVGInfo(tens_ser_list, None)
        else:
            return TosaTensorValuesGen.tvgLazyGenDefault(
                testGen, rng, opName, dtypeList, shapeList, argsDict, error_name
            )

    # Set the MUL data range to the square root of the largest value
    # to avoid infinities
    TVG_FLOAT_HIGH_VALUE_MUL = {
        DType.FP32: math.sqrt(TVG_FLOAT_HIGH_VALUE[DType.FP32]),
        DType.FP16: math.sqrt(TVG_FLOAT_HIGH_VALUE[DType.FP16]),
        DType.BF16: math.sqrt(TVG_FLOAT_HIGH_VALUE[DType.BF16]),
        DType.FP8E4M3: math.sqrt(TVG_FLOAT_HIGH_VALUE[DType.FP8E4M3]),
        DType.FP8E5M2: math.sqrt(TVG_FLOAT_HIGH_VALUE[DType.FP8E5M2]),
    }

    @staticmethod
    def tvgMul(testGen, rng, opName, dtypeList, shapeList, argsDict, error_name=None):
        # Need to supply shift tensor for MUL
        dtypeList[2] = DType.INT8
        shapeList[2] = [] if error_name != ErrorIf.InputRank0WrongRank else [1]

        if error_name is not None or dtypeList[0] in (
            DType.FP16,
            DType.BF16,
            DType.FP32,
        ):
            # ERROR_IF or floating point test
            data_range = TosaTensorValuesGen._get_data_range(
                rng, dtypeList[0], TosaTensorValuesGen.TVG_FLOAT_HIGH_VALUE_MUL
            )
            if data_range:
                argsDict["data_range"] = data_range

            # Create a new list for the pre-generated data in argsDict["fixed_data"]
            argsDict["fixed_data"] = [None, None, [argsDict["shift"]]]

            return TosaTensorValuesGen.tvgLazyGenDefault(
                testGen, rng, opName, dtypeList, shapeList, argsDict, error_name
            )
        else:
            op = testGen.TOSA_OP_LIST[opName]
            pCount, cCount = op["operands"]

            tens_ser_list = []

            # Make sure multiply result in int32 range
            if dtypeList[0] == DType.SHAPE:
                shift = 0
            else:
                shift = argsDict["shift"]

            np_type = np.int32
            if dtypeList[0] == DType.INT8:
                num_bits = 8
                np_type = np.int8
            elif dtypeList[0] == DType.INT16:
                num_bits = 16
                np_type = np.int16
            elif dtypeList[0] in (DType.INT32, DType.SHAPE):
                num_bits = 32
                # np_type is not used for DType.SHAPE so leave as np.int32
            elif error_name == ErrorIf.WrongInputType:
                num_bits = 8
            else:
                raise Exception(
                    f"OpMul: invalid input dtype {gtu.DTYPE_ATTRIBUTES[dtypeList[0]]['str']}"
                )

            for idx, shape in enumerate(shapeList[:]):
                if dtypeList[idx] == DType.SHAPE:
                    low = testGen.args.tensor_shape_range[0]
                    high = testGen.args.tensor_shape_range[1]
                else:
                    low = -(2 ** (num_bits - 1))
                    high = (2 ** (num_bits - 1)) - 1

                a_arr = rng.randTensor(
                    shapeList[0], DType.INT32, data_range=(low, high)
                )
                b_arr = rng.randTensor(
                    shapeList[1], DType.INT32, data_range=(low, high)
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

                if (result_arr > -(2**31)).all() and (
                    result_arr <= ((2**31) - 1)
                ).all():
                    break

                i = i + 1
                a_arr = a_arr // 2
                b_arr = b_arr // 2

            # MUL with 3 inputs (3rd is shift)
            tens_ser_list.append(
                testGen.ser.addPlaceholder(
                    shapeList[0], dtypeList[0], a_arr.astype(np_type)
                )
            )
            tens_ser_list.append(
                testGen.ser.addPlaceholder(
                    shapeList[1], dtypeList[1], b_arr.astype(np_type)
                )
            )
            tens_ser_list.append(
                testGen.ser.addPlaceholder(shapeList[2], dtypeList[2], np.int8([shift]))
            )

            return TosaTensorValuesGen.TVGInfo(tens_ser_list, None)

    @staticmethod
    def tvgConcat(
        testGen, rng, opName, dtypeList, shapeList, argsDict, error_name=None
    ):
        count = len(shapeList) - testGen.args.num_const_inputs_concat
        if count < 1:
            count = 1
        if testGen.args.num_const_inputs_concat == 0:
            count = len(shapeList)

        shapeList = TosaTensorGen.tgConcatConstInput(
            rng, shapeList, argsDict["axis"], error_name
        )

        # Override default pCount/cCount for operator
        argsDict["p_count"] = count
        argsDict["c_count"] = len(shapeList) - count

        return TosaTensorValuesGen.tvgLazyGenDefault(
            testGen, rng, opName, dtypeList, shapeList, argsDict, error_name
        )

    @staticmethod
    def tvgLogicalShift(
        testGen, rng, opName, dtypeList, shapeList, argsDict, error_name=None
    ):
        op = testGen.TOSA_OP_LIST[opName]
        pCount, cCount = op["operands"]
        assert (
            pCount == 2 and cCount == 0
        ), "Op.LOGICAL_LEFT_SHIFT or Op.LOGICAL_RIGHT_SHIFT must have 2 placeholders, 0 consts"
        values_arr = rng.randTensor(shapeList[0], dtypeList[0])

        if dtypeList[0] == DType.INT32:
            shift_max = 32
        elif dtypeList[0] == DType.INT16:
            shift_max = 16
        elif dtypeList[0] == DType.INT8:
            shift_max = 8
        else:
            shift_max = 32
        shift_arr = rng.randTensor(
            shapeList[1], dtypeList[0], data_range=(0, shift_max)
        )

        tens_ser_list = []
        tens_ser_list.append(
            testGen.ser.addPlaceholder(shapeList[0], dtypeList[0], values_arr)
        )
        tens_ser_list.append(
            testGen.ser.addPlaceholder(shapeList[1], dtypeList[1], shift_arr)
        )

        return TosaTensorValuesGen.TVGInfo(tens_ser_list, None)

    @staticmethod
    def tvgEqual(testGen, rng, opName, dtypeList, shapeList, argsDict, error_name=None):
        if error_name is None and not gtu.dtypeIsSupportedByDataGen(dtypeList[0]):
            # Integer
            op = testGen.TOSA_OP_LIST[opName]
            pCount, cCount = op["operands"]
            assert (
                pCount == 2 and cCount == 0
            ), "Op.EQUAL must have 2 placeholders, 0 consts"

            a_arr = rng.randTensor(shapeList[0], dtypeList[0])
            b_arr = rng.randTensor(shapeList[1], dtypeList[1])

            # Using random numbers means that it will be very unlikely that
            # there are any matching (equal) values, therefore force that
            # there are twice the number of matching values as the tensor rank
            for num in range(0, len(shapeList[0]) * 2):
                a_index = []
                b_index = []
                # Choose an index in each axis for the whole shape
                for axis in range(0, len(shapeList[0])):
                    # Index can be up to the largest dimension in both shapes
                    index = np.int32(
                        rng.integers(0, max(shapeList[0][axis], shapeList[1][axis]))
                    )
                    # Reduce the index down to a shape's dim for broadcasting
                    a_index.append(min(shapeList[0][axis] - 1, index))
                    b_index.append(min(shapeList[1][axis] - 1, index))

                a_arr[tuple(a_index)] = b_arr[tuple(b_index)]

            tens_ser_list = []
            tens_ser_list.append(
                testGen.ser.addPlaceholder(shapeList[0], dtypeList[0], a_arr)
            )
            tens_ser_list.append(
                testGen.ser.addPlaceholder(shapeList[1], dtypeList[1], b_arr)
            )
            return TosaTensorValuesGen.TVGInfo(tens_ser_list, None)
        else:
            # ERROR_IF or floating point test
            return TosaTensorValuesGen.tvgLazyGenDefault(
                testGen, rng, opName, dtypeList, shapeList, argsDict, error_name
            )

    @staticmethod
    def tvgReduceSum(
        testGen, rng, opName, dtypeList, shapeList, argsDict, error_name=None
    ):
        dtype = dtypeList[0]
        if dtype == DType.INT32:
            op = testGen.TOSA_OP_LIST[opName]
            pCount, cCount = op["operands"]
            assert (
                pCount == 1 and cCount == 0
            ), "Op.REDUCE_SUM must have 1 placeholders, 0 consts"
            # Limit values so that the sum cannot exceed the range of an int32 during
            # summation of any axis
            range_val = int((1 << 31) / max(shapeList[0]))
            values_arr = rng.randTensor(
                shapeList[0], dtype, data_range=(-range_val, range_val)
            )
            tens_ser_list = []
            tens_ser_list.append(
                testGen.ser.addPlaceholder(shapeList[0], dtype, values_arr)
            )
            return TosaTensorValuesGen.TVGInfo(tens_ser_list, None)
        else:
            # ERROR_IF or dot product floating point test
            if (
                error_name is None
                and argsDict["dg_type"] != gtu.ComplianceMode.DOT_PRODUCT
            ):
                # Limit ranges for (non error & non compliance) tests by using
                # values that can be summed on any axis to not hit infinity
                highval_lookup = {
                    dtype: TosaTensorValuesGen.TVG_FLOAT_HIGH_VALUE[dtype]
                    / max(shapeList[0])
                }
                data_range = TosaTensorValuesGen._get_data_range(
                    rng, dtype, highval_lookup
                )
                assert data_range is not None
                argsDict["data_range"] = data_range

            return TosaTensorValuesGen.tvgLazyGenDefault(
                testGen, rng, opName, dtypeList, shapeList, argsDict, error_name
            )

    @staticmethod
    def tvgReduceProduct(
        testGen, rng, opName, dtypeList, shapeList, argsDict, error_name=None
    ):
        dtype = dtypeList[0]
        if error_name is None:
            # Limit ranges for (non error) tests by using
            # values that can be multiplied on any axis to not hit infinity
            highval_lookup = {
                dtype: math.pow(
                    TosaTensorValuesGen.TVG_FLOAT_HIGH_VALUE[dtype],
                    1 / max(shapeList[0]),
                )
            }
            data_range = TosaTensorValuesGen._get_data_range(rng, dtype, highval_lookup)
            assert data_range is not None
            argsDict["data_range"] = data_range

        return TosaTensorValuesGen.tvgLazyGenDefault(
            testGen, rng, opName, dtypeList, shapeList, argsDict, error_name
        )

    @staticmethod
    def tvgTable(testGen, rng, opName, dtypeList, shapeList, argsDict, error_name=None):
        # Use supported type for table data on ERROR_IF
        dtypeList[1] = (
            dtypeList[0] if error_name != ErrorIf.WrongInputType else DType.INT8
        )

        table_values = argsDict["table"]
        shapeList[1] = [len(table_values)]
        argsDict["fixed_data"] = [None, table_values]

        return TosaTensorValuesGen.tvgLazyGenDefault(
            testGen, rng, opName, dtypeList, shapeList, argsDict, error_name
        )

    @staticmethod
    def tvgResize(
        testGen, rng, opName, dtypeList, shapeList, argsDict, error_name=None
    ):
        data_range = TosaTensorValuesGen._get_data_range(
            rng,
            dtypeList[0],
            TosaTensorValuesGen.TVG_FLOAT_HIGH_VALUE,
        )
        if data_range:
            argsDict["data_range"] = data_range
            # Needed for compliance
            argsDict["max_abs_value"] = data_range[1]

        scale_values = argsDict["scale"]
        offset_values = argsDict["offset"]
        border_values = argsDict["border"]
        dtypeList[1] = DType.SHAPE
        dtypeList[2] = DType.SHAPE
        dtypeList[3] = DType.SHAPE
        shapeList[1] = [len(scale_values)]
        shapeList[2] = [len(offset_values)]
        shapeList[3] = [len(border_values)]
        argsDict["fixed_data"] = [None, scale_values, offset_values, border_values]

        return TosaTensorValuesGen.tvgLazyGenDefault(
            testGen, rng, opName, dtypeList, shapeList, argsDict, error_name
        )

    # Set the POW exponent high data range
    TVG_FLOAT_HIGH_VALUE_POW_EXP = {
        DType.FP32: 10.0,
        DType.FP16: 10.0,
        DType.BF16: 10.0,
        DType.FP8E4M3: 10.0,
        DType.FP8E5M2: 10.0,
    }
    # POW highest base value (within a safe margin of error) that can be raised
    # to +ve exponent that doesn't become Infinity
    TVG_FLOAT_HIGH_VALUE_POW_BASE = {
        DType.FP32: math.floor(
            math.pow(
                TVG_FLOAT_HIGH_VALUE[DType.FP32],
                1.0 / TVG_FLOAT_HIGH_VALUE_POW_EXP[DType.FP32],
            )
        ),
        DType.FP16: math.floor(
            math.pow(
                TVG_FLOAT_HIGH_VALUE[DType.FP16],
                1.0 / TVG_FLOAT_HIGH_VALUE_POW_EXP[DType.FP16],
            )
        ),
        DType.BF16: math.floor(
            math.pow(
                TVG_FLOAT_HIGH_VALUE[DType.BF16],
                1.0 / TVG_FLOAT_HIGH_VALUE_POW_EXP[DType.BF16],
            )
        ),
        DType.FP8E4M3: math.floor(
            math.pow(
                TVG_FLOAT_HIGH_VALUE[DType.FP8E4M3],
                1.0 / TVG_FLOAT_HIGH_VALUE_POW_EXP[DType.FP8E4M3],
            )
        ),
        DType.FP8E5M2: math.floor(
            math.pow(
                TVG_FLOAT_HIGH_VALUE[DType.FP8E5M2],
                1.0 / TVG_FLOAT_HIGH_VALUE_POW_EXP[DType.FP8E5M2],
            )
        ),
    }
    # POW lowest base value (within a safe margin of error) that can be raised
    # to -ve exponent that doesn't become Infinity
    TVG_FLOAT_LOW_VALUE_POW_BASE = {
        DType.FP32: math.ceil(
            math.pow(
                1.0 / TVG_FLOAT_HIGH_VALUE[DType.FP32],
                1.0 / TVG_FLOAT_HIGH_VALUE_POW_EXP[DType.FP32],
            )
            * 1000
        )
        / 1000,
        DType.FP16: math.ceil(
            math.pow(
                1.0 / TVG_FLOAT_HIGH_VALUE[DType.FP16],
                1.0 / TVG_FLOAT_HIGH_VALUE_POW_EXP[DType.FP16],
            )
            * 1000
        )
        / 1000,
        DType.BF16: math.ceil(
            math.pow(
                1.0 / TVG_FLOAT_HIGH_VALUE[DType.BF16],
                1.0 / TVG_FLOAT_HIGH_VALUE_POW_EXP[DType.BF16],
            )
            * 1000
        )
        / 1000,
        DType.FP8E4M3: math.ceil(
            math.pow(
                1.0 / TVG_FLOAT_HIGH_VALUE[DType.FP8E4M3],
                1.0 / TVG_FLOAT_HIGH_VALUE_POW_EXP[DType.FP8E4M3],
            )
            * 1000
        )
        / 1000,
        DType.FP8E5M2: math.ceil(
            math.pow(
                1.0 / TVG_FLOAT_HIGH_VALUE[DType.FP8E5M2],
                1.0 / TVG_FLOAT_HIGH_VALUE_POW_EXP[DType.FP8E5M2],
            )
            * 1000
        )
        / 1000,
    }

    @staticmethod
    def tvgPow(testGen, rng, opName, dtypeList, shapeList, argsDict, error_name=None):
        if error_name is not None:
            return TosaTensorValuesGen.tvgLazyGenDefault(
                testGen, rng, opName, dtypeList, shapeList, argsDict, error_name
            )
        dtype = dtypeList[0]
        # Different value ranges for each test set for POW
        # Default to test set 0 for FP Special tests as the ranges will be ignored
        test_set = argsDict.get("s", 0)
        if test_set == 0:
            # Positive base with fractional exponent
            base_range = TosaTensorValuesGen._get_data_range(
                rng,
                dtype,
                TosaTensorValuesGen.TVG_FLOAT_HIGH_VALUE_POW_BASE,
                TosaTensorValuesGen.TVG_FLOAT_LOW_VALUE_POW_BASE,
            )
            exp_range = TosaTensorValuesGen._get_data_range(
                rng, dtype, TosaTensorValuesGen.TVG_FLOAT_HIGH_VALUE_POW_EXP
            )
            exp_round = False
        else:
            # Integer exponent
            exp_range = TosaTensorValuesGen._get_data_range(
                rng, dtype, TosaTensorValuesGen.TVG_FLOAT_HIGH_VALUE_POW_EXP
            )
            exp_round = True
            if test_set == 1:
                # Positive base
                base_range = TosaTensorValuesGen._get_data_range(
                    rng,
                    dtype,
                    TosaTensorValuesGen.TVG_FLOAT_HIGH_VALUE_POW_BASE,
                    TosaTensorValuesGen.TVG_FLOAT_LOW_VALUE_POW_BASE,
                )
            else:
                assert test_set == 2
                # Negative base
                # Supply new look up tables with negative values
                base_range = TosaTensorValuesGen._get_data_range(
                    rng,
                    dtype,
                    {dtype: -TosaTensorValuesGen.TVG_FLOAT_LOW_VALUE_POW_BASE[dtype]},
                    {dtype: -TosaTensorValuesGen.TVG_FLOAT_HIGH_VALUE_POW_BASE[dtype]},
                )

        data_range_list = (
            {
                "range": base_range,
            },
            {
                "range": exp_range,
                "round": exp_round,
            },
        )
        argsDict["data_range_list"] = data_range_list
        return TosaTensorValuesGen.tvgLazyGenDefault(
            testGen, rng, opName, dtypeList, shapeList, argsDict, error_name
        )

    @staticmethod
    def tvgLogRsqrt(
        testGen, rng, opName, dtypeList, shapeList, argsDict, error_name=None
    ):
        # LOG & RSQRT data range from lowest expressible positive number to
        # largest to avoid NaNs
        data_range = TosaTensorValuesGen._get_data_range(
            rng,
            dtypeList[0],
            TosaTensorValuesGen.TVG_FLOAT_HIGH_VALUE,
            TosaTensorValuesGen.TVG_FLOAT_LOW_VALUE,
        )
        if data_range:
            argsDict["data_range"] = data_range

        return TosaTensorValuesGen.tvgLazyGenDefault(
            testGen, rng, opName, dtypeList, shapeList, argsDict, error_name
        )

    # Set the EXP data range to the log of the largest to smallest values
    # to avoid infinities or making the result zero
    TVG_FLOAT_HIGH_VALUE_EXP = {
        DType.FP32: math.log(TVG_FLOAT_HIGH_VALUE[DType.FP32]),
        DType.FP16: math.log(TVG_FLOAT_HIGH_VALUE[DType.FP16]),
        DType.BF16: math.log(TVG_FLOAT_HIGH_VALUE[DType.BF16]),
        DType.FP8E4M3: math.log(TVG_FLOAT_HIGH_VALUE[DType.FP8E4M3]),
        DType.FP8E5M2: math.log(TVG_FLOAT_HIGH_VALUE[DType.FP8E5M2]),
    }
    TVG_FLOAT_LOW_VALUE_EXP = {
        DType.FP32: math.log(TVG_FLOAT_LOW_VALUE[DType.FP32]),
        DType.FP16: math.log(TVG_FLOAT_LOW_VALUE[DType.FP16]),
        DType.BF16: math.log(TVG_FLOAT_LOW_VALUE[DType.BF16]),
        DType.FP8E4M3: math.log(TVG_FLOAT_LOW_VALUE[DType.FP8E4M3]),
        DType.FP8E5M2: math.log(TVG_FLOAT_LOW_VALUE[DType.FP8E5M2]),
    }

    @staticmethod
    def tvgExp(testGen, rng, opName, dtypeList, shapeList, argsDict, error_name=None):
        data_range = TosaTensorValuesGen._get_data_range(
            rng,
            dtypeList[0],
            TosaTensorValuesGen.TVG_FLOAT_HIGH_VALUE_EXP,
            TosaTensorValuesGen.TVG_FLOAT_LOW_VALUE_EXP,
        )
        if data_range:
            argsDict["data_range"] = data_range

        return TosaTensorValuesGen.tvgLazyGenDefault(
            testGen, rng, opName, dtypeList, shapeList, argsDict, error_name
        )

    @staticmethod
    def tvgCast(testGen, rng, opName, dtypeList, shapeList, argsDict, error_name=None):
        in_dtype = dtypeList[0]
        out_dtype = argsDict["out_type"]
        # Create look up to limit input tensor to output type maximums to avoid
        # FP infinities and saturation of integers
        out_range = rng.dTypeRange(out_dtype, high_inclusive=True)
        highval_lookup = {in_dtype: out_range[1]}
        data_range = TosaTensorValuesGen._get_data_range(
            rng,
            in_dtype,
            highval_lookup,
        )

        assert data_range is not None
        argsDict["data_range"] = data_range

        return TosaTensorValuesGen.tvgLazyGenDefault(
            testGen, rng, opName, dtypeList, shapeList, argsDict, error_name
        )

    @staticmethod
    def tvgGather(
        testGen, rng, opName, dtypeList, shapeList, argsDict, error_name=None
    ):
        K = shapeList[0][1]

        # Fix the type of the indices tensor
        dtypeList[1] = DType.INT32

        dtype = dtypeList[0]
        if not gtu.dtypeIsSupportedByDataGen(dtype):
            # Test unsupported by data generator
            op = testGen.TOSA_OP_LIST[opName]
            pCount, cCount = op["operands"]
            assert (
                pCount == 2 and cCount == 0
            ), "Op.GATHER must have 2 placeholders, 0 consts"

            tens_ser_list = []
            for idx, shape in enumerate(shapeList):
                dtype = dtypeList[idx]
                if idx != 1:
                    arr = rng.randTensor(shape, dtype)
                    tens_ser_list.append(testGen.ser.addPlaceholder(shape, dtype, arr))
                else:
                    # Limit data range of indices tensor upto K (exclusive)
                    arr = rng.randTensor(shape, dtype, (0, K))
                    # To match old functionality - create indices as CONST
                    tens_ser_list.append(testGen.ser.addConst(shape, dtype, arr))

            return TosaTensorValuesGen.TVGInfo(tens_ser_list, None)

        else:
            # ERROR_IF or floating point test
            # Use inclusive values upto index K for indices tensor
            data_range_list = (
                {"range": None},
                {"range": (0, K - 1)},
            )
            argsDict["data_range_list"] = data_range_list

            return TosaTensorValuesGen.tvgLazyGenDefault(
                testGen, rng, opName, dtypeList, shapeList, argsDict, error_name
            )

    @staticmethod
    def tvgScatter(
        testGen, rng, opName, dtypeList, shapeList, argsDict, error_name=None
    ):
        K = shapeList[0][1]
        W = shapeList[2][1]

        # Work out an indices tensor here with data that doesn't exceed the
        # dimension K of the values_in tensor and does NOT repeat the same K
        # location as needed by the spec:
        # "It is not permitted to repeat the same output index within a single
        # SCATTER operation and so each output index occurs at most once."
        assert K >= W, "Op.SCATTER W must be smaller or equal to K"

        # Fix the type of the indices tensor
        dtypeList[1] = DType.INT32

        dtype = dtypeList[0]
        if not gtu.dtypeIsSupportedByDataGen(dtype):
            # Test unsupported by data generator
            op = testGen.TOSA_OP_LIST[opName]
            pCount, cCount = op["operands"]
            assert (
                pCount == 3 and cCount == 0
            ), "Op.SCATTER must have 3 placeholders, 0 consts"

            tens_ser_list = []
            for idx, shape in enumerate(shapeList):
                dtype = dtypeList[idx]
                if idx != 1:
                    arr = rng.randTensor(shape, dtype)
                    tens_ser_list.append(testGen.ser.addPlaceholder(shape, dtype, arr))
                else:
                    # Create the indices array
                    assert dtype == DType.INT32, "Op.SCATTER unexpected indices type"
                    arr = []
                    for n in range(shape[0]):
                        # Get a shuffled list of output indices (0 to K-1) and
                        # limit length to W
                        arr.append(rng.permutation(K)[:W])
                    indices_arr = np.array(arr, dtype=np.int32)  # (N, W)
                    # To match old functionality - create indices as CONST
                    tens_ser_list.append(
                        testGen.ser.addConst(shape, dtype, indices_arr)
                    )

            return TosaTensorValuesGen.TVGInfo(tens_ser_list, None)

        else:
            # ERROR_IF or floating point test
            # Use inclusive values upto index K for indices tensor
            data_range_list = (
                {"range": None},
                {"range": (0, K - 1)},
                {"range": None},
            )
            argsDict["data_range_list"] = data_range_list

            return TosaTensorValuesGen.tvgLazyGenDefault(
                testGen, rng, opName, dtypeList, shapeList, argsDict, error_name
            )


class TosaArgGen:
    """Argument generators create exhaustive or random lists of attributes for
    operators that take attributes or other parameters.

    The return value is a list of (descriptive_name, [arglist]) tuples where
    the descriptive_name is appended to the test name and the arglist is expanded
    as arguments to the operator build function.
    """

    def __init__(self):
        pass

    @staticmethod
    def _add_data_generators(testGen, opName, shapeList, dtype, arg_list, error_name):
        """Add extra tests for each type of data generator for this op."""
        op = testGen.TOSA_OP_LIST[opName]

        if (
            error_name is None
            and "data_gen" in op
            and gtu.dtypeIsSupportedByDataGen(dtype)
        ):
            dataGenTypesList = testGen.TOSA_OP_LIST[opName]["data_gen"].get(
                dtype, (gtu.DataGenType.PSEUDO_RANDOM,)
            )
        else:
            # Error test or No data generator types listed - assume random
            dataGenTypesList = (gtu.DataGenType.PSEUDO_RANDOM,)

        def check_min_size(opName, shape, min_size, reason):
            # Check tensor size meets minimum requirements
            tensor_size = gtu.product(shape)
            if tensor_size < min_size:
                shape_info = " ({})".format(shape)
                logger.info(
                    f"Skipping {opName}{shape_info} as tensor data size too small for {reason} values {tensor_size} < {min_size}"
                )
                return False
            return True

        def update_data_gen(testGen, opName, dtype, dgt_remove):
            # Remove special data generator to limit number of tests
            assert "data_gen" in testGen.TOSA_OP_LIST[opName]
            assert dtype in testGen.TOSA_OP_LIST[opName]["data_gen"]
            data_gen = testGen.TOSA_OP_LIST[opName]["data_gen"].copy()
            dgt_list = list(data_gen[dtype])
            dgt_list.remove(dgt_remove)
            data_gen[dtype] = tuple(dgt_list)
            testGen.TOSA_OP_LIST[opName]["data_gen"] = data_gen

        # Expand arg list with other data generator types
        new_arg_list = []
        for dg_type in dataGenTypesList:
            only_one = False  # Check for only one dg type test
            for arg_str, args_dict in arg_list:
                gen_args_dict = args_dict.copy()
                # Only create one test by default - no sets of tests
                num_test_sets = 0

                if dg_type == gtu.DataGenType.PSEUDO_RANDOM:
                    if error_name is None:
                        num_test_sets = args_dict.get("num_test_sets", 0)

                elif dg_type == gtu.DataGenType.DOT_PRODUCT:
                    # Extra tests for each dot product test set
                    dot_products = args_dict["dot_products"]
                    if dot_products < testGen.TOSA_MI_DOT_PRODUCT_MIN:
                        shape_info = (
                            " ({})".format(testGen.shapeStr(args_dict["shape"]))
                            if "shape" in args_dict
                            else ""
                        )
                        logger.info(
                            f"Skipping {opName}{shape_info} {gtu.DTYPE_ATTRIBUTES[dtype]['json']} dot product test as too few calculations {dot_products} < {testGen.TOSA_MI_DOT_PRODUCT_MIN}"
                        )
                        continue
                    # KS and acc_type is required by all dot product generators
                    assert "ks" in args_dict
                    assert "acc_type" in args_dict

                    num_test_sets = testGen.TOSA_MI_DOT_PRODUCT_TEST_SETS

                elif dg_type == gtu.DataGenType.FULL_RANGE:
                    if testGen.args.no_special_tests:
                        continue
                    if not check_min_size(
                        opName,
                        shapeList[0],
                        gtu.DTYPE_ATTRIBUTES[dtype]["fullset"],
                        "full range of",
                    ):
                        continue
                    # Large enough tensor data size for full range, add full test
                    arg_str = f"{arg_str}_full" if arg_str else "full"
                    gen_args_dict["tags"] = args_dict.get("tags", []) + [
                        "non_finite_fp_data"
                    ]
                    # Create only one special test per data type, unless the op
                    # explicitly marks it allows more than one.
                    allow_multiple_special_tests = op.get(
                        "allow_multiple_special_tests", False
                    )
                    only_one = not allow_multiple_special_tests

                elif dg_type == gtu.DataGenType.FP_SPECIAL:
                    if testGen.args.no_special_tests:
                        continue
                    if not check_min_size(
                        opName,
                        shapeList[0],
                        testGen.TOSA_FP_SPECIAL_MIN_SIZE,
                        "FP special",
                    ):
                        continue
                    broadcastable_inputs = testGen.TOSA_OP_LIST[opName].get(
                        "broadcastable_inputs", 0
                    )
                    if broadcastable_inputs > 0:
                        shapes_set = {
                            tuple(x) for x in shapeList[:broadcastable_inputs]
                        }
                        if len(shapes_set) != 1:
                            logger.info(
                                f"Changing {opName} input shapes {shapes_set} - broadcasting incompatable with FP special test"
                            )
                            broadcasted_shape = np.int32(
                                np.broadcast_shapes(*shapeList[:broadcastable_inputs])
                            )
                            # Change shape list in place to propagate changes
                            for idx in range(broadcastable_inputs):
                                shapeList[idx] = broadcasted_shape

                    arg_str = f"{arg_str}_fs" if arg_str else "fs"
                    gen_args_dict["tags"] = args_dict.get("tags", []) + [
                        "non_finite_fp_data"
                    ]
                    # Create only one special test per data type, unless the op
                    # explicitly marks it requires more than one.
                    allow_multiple_special_tests = op.get(
                        "allow_multiple_special_tests", False
                    )
                    only_one = not allow_multiple_special_tests

                gen_args_dict["dg_type"] = dg_type

                if num_test_sets > 0:
                    for s in range(0, num_test_sets):
                        set_arg_str = f"{arg_str}_s{s}" if arg_str else f"s{s}"
                        set_args_dict = gen_args_dict.copy()
                        set_args_dict["s"] = s
                        new_arg_list.append((set_arg_str, set_args_dict))
                else:
                    # Default is a single test
                    new_arg_list.append((arg_str, gen_args_dict))

                if only_one:
                    # Skip all remaining tests and remove this data generator
                    update_data_gen(testGen, opName, dtype, dg_type)
                    break

        return new_arg_list

    @staticmethod
    def _append_nan_mode(rng, arg_list):
        new_arg_list = []
        mode_to_str = {
            NanPropagationMode.PROPAGATE: "modeP",
            NanPropagationMode.IGNORE: "modeI",
        }

        for arg_str, args_dict in arg_list:
            nan_mode = rng.choice(
                [NanPropagationMode.PROPAGATE, NanPropagationMode.IGNORE]
            )
            mode_str = mode_to_str[nan_mode]
            separator = "" if len(arg_str) == 0 else "_"
            new_arg_str = separator.join((arg_str, mode_str))
            new_args_dict = deepcopy(args_dict)
            new_args_dict["nan_mode"] = nan_mode
            new_arg_list.append((new_arg_str, new_args_dict))

        return new_arg_list

    @staticmethod
    def agNone(testGen, rng, opName, shapeList, dtype, error_name=None):
        """A trivial argument generator for operators that don't take any
        non-tensor arguments"""
        arg_list = TosaArgGen._add_data_generators(
            testGen,
            opName,
            shapeList,
            dtype,
            [("", {})],
            error_name,
        )

        if gtu.dtypeIsFloat(dtype) and gtu.has_nan_mode_by_name(opName):
            arg_list = TosaArgGen._append_nan_mode(rng, arg_list)

        # Return list of tuples: (arg_str, args_dict)
        return arg_list

    @staticmethod
    def agPow(testGen, rng, opName, shapeList, dtype, error_name=None):
        """Pow operator needs different test sets to cover random numbers
        without creating NaNs or Infs"""
        arg_list = TosaArgGen._add_data_generators(
            testGen,
            opName,
            shapeList,
            dtype,
            [("", {"num_test_sets": 3})],
            error_name,
        )
        # Return list of tuples: (arg_str, args_dict)
        return arg_list

    @staticmethod
    def agAxis(testGen, rng, opName, shapeList, dtype, error_name=None):
        """Build the axis argument for operators that take a single axis"""
        arg_list = []
        shape = shapeList[0]

        if error_name == ErrorIf.AxisSmallerZero:
            # Set too small axis
            axes = [rng.integers(-5, 0)]
        elif error_name == ErrorIf.AxisLargerRank:
            # Set too large axis
            axes = [rng.integers(len(shape) + 1, len(shape) + 10)]
        else:
            # Create tests for each dimension
            axes = range(0, len(shape))

        opid = testGen.TOSA_OP_LIST[opName]["op"]

        for a in axes:
            args_dict = {"axis": int(a)}
            if opid == Op.REDUCE_SUM:
                output_shape = shape.copy()
                if error_name is None:
                    # It only matters that we calculate the dot_products correctly
                    # for non error_if tests as they should never be run
                    output_shape[a] = 1
                args_dict["dot_products"] = gtu.product(output_shape)
                args_dict["shape"] = shape
                args_dict["ks"] = int(shape[a]) if a >= 0 and a < len(shape) else 1
                args_dict["acc_type"] = dtype if dtype != DType.BF16 else DType.FP32

            arg_list.append(("axis{}".format(a), args_dict))

        arg_list = TosaArgGen._add_data_generators(
            testGen,
            opName,
            shapeList,
            dtype,
            arg_list,
            error_name,
        )

        if gtu.dtypeIsFloat(dtype) and gtu.has_nan_mode_by_name(opName):
            arg_list = TosaArgGen._append_nan_mode(rng, arg_list)

        # Return list of tuples: (arg_str, args_dict)
        return arg_list

    @staticmethod
    def _calculate_sparsity(num_tests, sparsity_factor):
        sparsity = num_tests // sparsity_factor + 1
        # If there are only a small number of tests, just select them all
        if sparsity < 13:
            sparsity = 1
        # To get a variety of parameter combinations sparsity should not be a
        # multiple of 2, 3 or 5
        while sparsity % 2 == 0 or sparsity % 3 == 0 or sparsity % 5 == 0:
            sparsity += 1
        return sparsity

    # Maximum number of error_if variants to produce
    MAX_TESTS_ERROR_IFS = 3

    @staticmethod
    def agConv(testGen, rng, opName, shapeList, dtypes, error_name=None):
        # Used by CONV2D, CONV3D and DEPTHWISE_CONV2D
        arg_list = []

        # Shape: Batches, (Depth), Height, Width, Channels
        ifm_shape = shapeList[0]
        # Shape: (OFM channels), (KD), KH, KW, IFM channels
        filter_shape = shapeList[1]

        accum_dtypes = gtu.get_conv_accum_dtypes_from_tgTypes(dtypes)

        if error_name == ErrorIf.WrongAccumulatorType:
            accum_dtypes = (
                [DType.BF16] if gtu.dtypeIsFloat(dtypes[0]) else [DType.INT16]
            )

        # For op type checks
        op = testGen.TOSA_OP_LIST[opName]

        # Check the rank
        rank = 5 if op["op"] == Op.CONV3D else 4
        if error_name != ErrorIf.WrongRank:
            assert len(ifm_shape) == rank
            assert len(filter_shape) == rank

        # kernel rank omits channels
        k_rank = rank - 2
        k_pos = 0 if op["op"] == Op.DEPTHWISE_CONV2D else 1
        k_shape = tuple(filter_shape[k_pos : (k_pos + k_rank)])
        # compliance size - KS
        k_size = gtu.product(k_shape)
        if not op["op"] == Op.DEPTHWISE_CONV2D:
            k_size *= ifm_shape[-1]

        def get_conv_output_info(p, s, d, fix_up_padding=False):
            # Work out remainders and output dimensions with an
            # option to adjust paddings to create a valid operation
            nonlocal ifm_shape, k_shape, error_name, k_rank
            if fix_up_padding:
                p = list(p)  # Make paddings editable
            outputs_no_stride = []
            remainders = []
            outputs = []
            for index in range(k_rank):
                pad_offset = index * 2
                fixed = False
                # Fix up pad values to produce valid conv2d
                while not fixed:
                    # Output dimension without being adjusted for stride
                    output_no_stride = (
                        ifm_shape[index + 1]
                        - 1
                        + p[pad_offset]
                        + p[pad_offset + 1]
                        - (k_shape[index] - 1) * d[index]
                    )
                    # Tensor left over after applying striding
                    remainder = output_no_stride % s[index]
                    if not fix_up_padding:
                        # Just want remainders and outputs
                        break
                    if output_no_stride <= 0:
                        p[pad_offset + 1] += abs(output_no_stride) + 1
                        continue
                    if error_name == ErrorIf.ConvOutputShapeNonInteger:
                        if remainder:
                            # Conditions to trigger the test
                            fixed = True
                        else:
                            p[pad_offset + 1] += 1
                    else:
                        if remainder:
                            # Stride will be negative for StrideSmallerOne
                            assert remainder > 0 or (
                                error_name == ErrorIf.StrideSmallerOne and remainder < 0
                            )
                            p[pad_offset + 1] += abs(remainder)
                        else:
                            fixed = True
                outputs_no_stride.append(output_no_stride)
                remainders.append(remainder)
                # Output dimension taking in to account stride
                outputs.append((output_no_stride // s[index]) + 1)

            if fix_up_padding:
                p = tuple(p)  # Make the paddings read-only
                assert min(outputs_no_stride) > 0, "Fix up did not work!"
            return p, remainders, outputs, outputs_no_stride

        # ERROR_IF and float conv2d tests are the most at risk of failing to generate any legal arguments
        fix_up_padding = error_name is not None or (
            gtu.dtypeIsFloat(dtypes[0]) and op["op"] == Op.CONV2D
        )
        # Allow any size of output dimension
        max_dim_size = None
        # Include all tests by default
        sparsity = 1

        # Work out padding, strides and dilation ranges depending on
        # error and arguments
        if error_name in (
            ErrorIf.PadSmallerZero,
            ErrorIf.StrideSmallerOne,
            ErrorIf.DilationSmallerOne,
        ):
            # Use specific invalid value(s)
            if error_name == ErrorIf.PadSmallerZero:
                # Create negative paddings but with positive opposite paddings
                neg_pad = rng.choice(range(-5, 0))
                p_vals = [neg_pad, abs(neg_pad)]
            else:
                p_vals = [0, 0]
            if error_name == ErrorIf.StrideSmallerOne:
                # Can't use stride=0, as it is used to derive output shape, as a divisor
                s_vals = [rng.choice(range(-5, 0))]
            else:
                s_vals = [1]
            if error_name == ErrorIf.DilationSmallerOne:
                d_vals = [rng.choice(range(-5, 1))]
            else:
                d_vals = [1]
            paddings = {tuple(p_vals) * k_rank}
            strides = {tuple(s_vals) * k_rank}
            dilations = {tuple(d_vals) * k_rank}

            fix_up_padding = True  # Need to fix up paddings to be valid

        elif testGen.args.level8k and error_name is None:
            # Only test 8k levels boundaries
            bigStride = testGen.TOSA_8K_LEVEL_MAX_STRIDE
            bigKernel = testGen.TOSA_8K_LEVEL_MAX_KERNEL
            bigPadding = bigKernel

            dilation_shape = [1] * k_rank
            pad_shape = [0] * k_rank * 2
            if op["op"] == Op.CONV3D:
                # Small stride apart from for big kernel (see below) to keep
                # tensor size/calculation small
                stride_shape = [1] * k_rank
                for idx in range(k_rank):
                    pad_offset = idx * 2
                    if k_shape[idx] == bigKernel:
                        # Padding shape needs to account for tensor shape
                        pad_shape[pad_offset] = bigPadding - ifm_shape[idx + 1]
                        pad_shape[pad_offset + 1] = bigPadding - dilation_shape[idx] + 1
                        # Big stride to reduce output size
                        stride_shape[idx] = bigKernel
                    else:
                        # Account for kernel size
                        pad_shape[pad_offset] = k_shape[idx] - 1
            else:
                # Always have a large stride with extra padding and dilation to keep
                # tensor calculation reasonable
                stride_shape = [bigKernel] * k_rank
                for idx in range(k_rank):
                    # Dilation shape must account for kernel size
                    dilation_shape[idx] = bigKernel // k_shape[idx]
                    # Padding shape needs to accommodate tensor/kernel & dilation
                    pad_offset = idx * 2
                    pad_shape[pad_offset] = bigPadding - ifm_shape[idx + 1]
                    pad_shape[pad_offset + 1] = bigPadding - dilation_shape[idx] + 1

            strides = {tuple(stride_shape)}
            dilations = {tuple(dilation_shape)}
            paddings = {tuple(pad_shape)}
            # Create a limit for the output dimensions size
            max_dim_size = testGen.TOSA_8K_LEVEL_MAX_KERNEL

            # Currently allow all combinations that are reasonable size
            sparsity = 1
        else:
            # Generate comprehensive argument lists
            p_vals = [x for x in range(0, testGen.args.max_conv_padding + 1)]
            paddings = {x for x in itertools.product(*([p_vals] * k_rank * 2))}
            # Stride must be greater than 1 to force non-integer error
            startStride = 1 if error_name != ErrorIf.ConvOutputShapeNonInteger else 2
            s_vals = [x for x in range(startStride, testGen.args.max_conv_stride + 1)]
            d_vals = [x for x in range(1, testGen.args.max_conv_dilation + 1)]

            strides = {x for x in itertools.product(*([s_vals] * k_rank))}
            dilations = {x for x in itertools.product(*([d_vals] * k_rank))}

            if error_name is None and testGen.args.oversize:
                # add some oversize argument values
                if max(ifm_shape) < 64:
                    bigPadding = 9
                    paddings.update(
                        {
                            x
                            for x in itertools.product(
                                *([[0, bigPadding]] * (k_rank * 2))
                            )
                        }
                    )
                bigStride = 8
                strides.update(
                    {x for x in itertools.product(*([[1, bigStride]] * k_rank))}
                )
                bigDilation = 7
                dilations.update(
                    {x for x in itertools.product(*([[1, bigDilation]] * k_rank))}
                )

            if error_name is None:
                # There are too many parameter combinations, so generate them sparsely,
                sparsity_factor = 120
                sparsity = TosaArgGen._calculate_sparsity(
                    len(paddings) * len(strides) * len(dilations), sparsity_factor
                )

        # Run through all the argument options creating valid test cases
        more_tests = True
        n = 0
        for a in accum_dtypes:
            for s in sorted(list(strides)):
                for p in sorted(list(paddings)):
                    for d in sorted(list(dilations)):
                        if more_tests and (n % sparsity == 0):
                            (
                                p,
                                remainders,
                                outputs,
                                outputs_no_stride,
                            ) = get_conv_output_info(p, s, d, fix_up_padding)
                            # Following is like checking each dimension N:
                            # (ifm_shape[N+1] - 1 + p[N*2] + p[N*2+1]) > d[N] * (k_shape[N] - 1)
                            if min(outputs_no_stride) <= 0:
                                # Not a valid operation
                                n += 1  # Increment count of tests
                                continue

                            if (
                                # the parameters must produce integer exact output
                                error_name != ErrorIf.ConvOutputShapeNonInteger
                                and max(remainders) == 0
                            ) or (
                                error_name == ErrorIf.ConvOutputShapeNonInteger
                                and max(remainders) > 0
                            ):
                                if (
                                    max_dim_size is not None
                                    and max(outputs) >= max_dim_size
                                ):
                                    # Test will consume too much memory - skip it
                                    logger.debug(
                                        "agConv: Convolution output too big - skipped"
                                    )
                                    continue

                                # Compliance - number of dot product calculations
                                if op["op"] == Op.DEPTHWISE_CONV2D:
                                    # N*OH*OW*C*M
                                    dots = gtu.product(
                                        (ifm_shape[0], *outputs, *filter_shape[2:])
                                    )
                                else:
                                    # N*OH*OW*OC or N*OD*OH*OW*OC
                                    dots = gtu.product(
                                        (ifm_shape[0], *outputs, filter_shape[0])
                                    )
                                if gtu.dtypeIsFloat(dtypes[0]):
                                    local_bound = rng.choice((False, True))
                                else:
                                    local_bound = False
                                args_dict = {
                                    "acc_type": a,
                                    "stride": s,
                                    "pad": p,
                                    "dilation": d,
                                    "kernel": k_shape,
                                    "ks": k_size,
                                    "dot_products": dots,
                                    "shape": ifm_shape,
                                    "local_bound": local_bound,
                                }

                                # Support for larger values than 9 needs different delimiter
                                delim = "" if max(s + p + d) <= 9 else "x"
                                arg_list.append(
                                    (
                                        "acc{}_st{}_pad{}_dilat{}_lclbnd{}".format(
                                            testGen.typeStr(a),
                                            delim.join([str(x) for x in s]),
                                            delim.join([str(x) for x in p]),
                                            delim.join([str(x) for x in d]),
                                            "1" if local_bound else "0",
                                        ),
                                        args_dict,
                                    )
                                )
                                if (
                                    error_name
                                    and len(arg_list) >= TosaArgGen.MAX_TESTS_ERROR_IFS
                                ):
                                    # Found enough errors
                                    logger.debug(
                                        f"Skipping creating more conv error tests for {error_name}"
                                    )
                                    more_tests = False
                        n += 1

        arg_list = TosaArgGen._add_data_generators(
            testGen,
            opName,
            shapeList,
            dtypes[0],
            arg_list,
            error_name,
        )
        # Return list of tuples: (arg_str, args_dict)
        return arg_list

    @staticmethod
    def agMatMul(testGen, rng, opName, shapeList, dtypes, error_name=None):
        if isinstance(dtypes, (list, tuple)):
            dtype = dtypes[0]
        else:
            dtype = dtypes

        # Get valid accumulate type(s)
        if dtype == DType.INT8:
            accum_dtypes = [DType.INT32]
        elif dtype == DType.INT16:
            accum_dtypes = [DType.INT48]
        elif dtype == DType.FP16:
            accum_dtypes = [DType.FP16, DType.FP32]
        elif dtype == DType.BF16:
            accum_dtypes = [DType.FP32]
        elif dtype == DType.FP32:
            accum_dtypes = [DType.FP32]
        elif dtype == DType.FP8E4M3 or dtype == DType.FP8E5M2:
            accum_dtypes = [DType.FP16, DType.FP32]
        elif error_name is None:
            assert False, f"Invalid I/O DType for MatMul: {DTypeNames[dtype]}"

        if error_name == ErrorIf.WrongOutputType:
            # Get incorrect output dtype for ErrorIf case
            accum_dtypes = [gtu.get_wrong_output_type(opName, rng, dtype)]
        elif error_name == ErrorIf.WrongInputType:
            # Pick some potentially correct output dtype if input type is incorrect
            accum_dtypes = [DType.INT32]

        # Get valid data type list based on supported profile/extension
        supported = testGen.args.profile + testGen.args.extension
        dtypeList = []

        if dtype == DType.INT8:
            if TosaProfiles.TosaProINT in supported:
                dtypeList.append([DType.INT8, DType.INT8])
        elif dtype == DType.INT16:
            if TosaProfiles.TosaExtInt16 in supported:
                dtypeList.append([DType.INT16, DType.INT16])
        elif dtype == DType.FP16:
            if TosaProfiles.TosaProFP in supported:
                dtypeList.append([DType.FP16, DType.FP16])
        elif dtype == DType.BF16:
            if TosaProfiles.TosaExtBF16 in supported:
                dtypeList.append([DType.BF16, DType.BF16])
        elif dtype == DType.FP32:
            if TosaProfiles.TosaProFP in supported:
                dtypeList.append([DType.FP32, DType.FP32])
        elif dtype == DType.FP8E4M3:
            if TosaProfiles.TosaExtFP8E4M3 in supported:
                if dtypes[1] == DType.FP8E4M3:
                    dtypeList.append([DType.FP8E4M3, DType.FP8E4M3])
                elif (
                    dtypes[1] == DType.FP8E5M2
                    and TosaProfiles.TosaExtFP8E5M2 in supported
                ):
                    dtypeList.append([DType.FP8E4M3, DType.FP8E5M2])
        elif dtype == DType.FP8E5M2:
            if TosaProfiles.TosaExtFP8E5M2 in supported:
                if dtypes[1] == DType.FP8E5M2:
                    dtypeList.append([DType.FP8E5M2, DType.FP8E5M2])
                elif (
                    dtypes[1] == DType.FP8E4M3
                    and TosaProfiles.TosaExtFP8E4M3 in supported
                ):
                    dtypeList.append([DType.FP8E5M2, DType.FP8E4M3])
        elif error_name == ErrorIf.WrongInputType:
            dtypeList.append([DType.BOOL, DType.BOOL])
        elif error_name is None:
            assert False, f"Invalid I/O DType for MatMul: {DTypeNames[dtype]}"

        # Set up compliance info
        args_dict = {
            "ks": int(shapeList[0][2]),  # Set KS = C, from input A (N,H,C)
            # Set dot_products = N*H*W
            "dot_products": gtu.product(
                (shapeList[0][0], shapeList[0][1], shapeList[1][2])
            ),
            "shape": shapeList[0],
        }

        # Create arg tuple of string and dict
        arg_list = []
        for dt in dtypeList:
            for a in accum_dtypes:
                d = args_dict.copy()
                d["acc_type"] = a
                d["otherInputType"] = int(dt[1])
                arg_list.append((f"acc{testGen.typeStr(a)}", d))

        arg_list = TosaArgGen._add_data_generators(
            testGen,
            opName,
            shapeList,
            dtype,
            arg_list,
            error_name,
        )
        # Return list of tuples: (arg_str, args_dict)
        return arg_list

    @staticmethod
    def agTransposeConv2D(testGen, rng, opName, shapeList, dtypes, error_name=None):
        arg_list = []

        if testGen.args.level8k and error_name is not None:
            # Don't produce negative large tests
            return arg_list

        ifm_shape = shapeList[0]
        filter_shape = shapeList[1]

        accum_dtypes = gtu.get_conv_accum_dtypes_from_tgTypes(dtypes)

        if error_name == ErrorIf.WrongAccumulatorType:
            accum_dtypes = (
                [DType.BF16] if gtu.dtypeIsFloat(dtypes[0]) else [DType.INT16]
            )

        # Must be rank 4
        if error_name != ErrorIf.WrongRank:
            assert len(ifm_shape) == 4
            assert len(filter_shape) == 4

        k_shape = tuple(filter_shape[1:3])
        # compliance size - KS
        k_size = gtu.product((*k_shape, ifm_shape[3]))

        if not testGen.args.level8k:
            # Generate comprehensive argument lists
            # - except for named errors, which use specific invalid value(s)
            smallest_padding_size = -min(k_shape[0], k_shape[1]) + 1
            if error_name == ErrorIf.PadLargerEqualKernel:
                max_filter_size = -max(k_shape[0], k_shape[1])
                p_vals = [rng.choice(range(max_filter_size - 10, max_filter_size))]
            else:
                p_vals = [
                    x
                    for x in range(
                        smallest_padding_size, testGen.args.max_conv_padding + 1
                    )
                ]
            paddings = {x for x in itertools.product(*([p_vals] * 4))}
            if error_name == ErrorIf.StrideSmallerOne:
                # Can't use stride=0, as it is used to derive output shape, as a divisor
                s_vals = [rng.choice(range(-5, 0))]
            else:
                s_vals = [x for x in range(1, testGen.args.max_conv_stride + 1)]
            strides = {x for x in itertools.product(*([s_vals] * 2))}

            if not error_name and testGen.args.oversize:
                # add some oversize argument values
                if max(ifm_shape) < 64:
                    bigPadding = 9
                    paddings.update(
                        {
                            x
                            for x in itertools.product(
                                *([[smallest_padding_size, bigPadding]] * 4)
                            )
                        }
                    )
                bigStride = 8
                strides.update({x for x in itertools.product(*([[1, bigStride]] * 2))})

            # There are too many parameter combinations, so generate them sparsely,
            # very sparse for negative tests
            sparsity_factor = 2 if error_name else 10
            sparsity = len(paddings) * len(strides) // sparsity_factor + 1
            # If there are only a small number of tests, just select them all
            if sparsity < 13:
                sparsity = 1
            # To get a variety of parameter combinations sparsity should not be a
            # multiple of 2, 3 or 5
            while sparsity % 2 == 0 or sparsity % 3 == 0 or sparsity % 5 == 0:
                sparsity += 1
        else:
            # Only test 8k levels boundaries
            bigStride = testGen.TOSA_8K_LEVEL_MAX_STRIDE
            bigKernel = testGen.TOSA_8K_LEVEL_MAX_KERNEL
            bigPadding = bigKernel

            pad_shape = [0] * (len(k_shape) * 2)
            stride_shape = [1] * len(k_shape)
            # The point at which input dimension combined with the stride will
            # create large output sizes!
            LARGE_SIZE = 2
            for idx in range(len(k_shape)):
                pad_offset = idx * 2
                if k_shape[idx] == bigKernel:
                    # Set large stride
                    stride_shape[idx] = bigKernel
                    # Use negative output padding to reduce shape size
                    pad_shape[pad_offset] = -(bigPadding - 1)
                    if ifm_shape[idx + 1] > LARGE_SIZE:
                        pad_shape[pad_offset + 1] = -(bigPadding - 1)
                else:
                    # The other dimension should be the bigKernel
                    alt_idx = 1 - idx
                    if (
                        k_shape[alt_idx] == bigKernel
                        and ifm_shape[alt_idx + 1] < LARGE_SIZE
                    ):
                        # As the input is small, the large stride won't
                        # affect the output so we can add some padding
                        pad_shape[pad_offset + 1] = bigPadding

            strides = {tuple(stride_shape)}
            paddings = {tuple(pad_shape)}

            # Currently allow all combinations that are reasonable size
            sparsity = 1

        n = 0
        for a in accum_dtypes:
            for s in sorted(list(strides)):
                for p in sorted(list(paddings)):
                    if n % sparsity == 0:
                        # Determine the output shape
                        oh = (ifm_shape[1] - 1) * s[0] + p[0] + p[1] + k_shape[0]
                        ow = (ifm_shape[2] - 1) * s[1] + p[2] + p[3] + k_shape[1]
                        os = [ifm_shape[0], oh, ow, filter_shape[0]]

                        if gtu.dtypeIsFloat(dtypes[0]):
                            local_bound = rng.choice((False, True))
                        else:
                            local_bound = False

                        # N*OH*OW*OC
                        dots = gtu.product((ifm_shape[0], oh, ow, filter_shape[0]))
                        args_dict = {
                            "acc_type": a,
                            "stride": s,
                            "pad": p,
                            "kernel": k_shape,
                            "ks": k_size,
                            "dot_products": dots,
                            "shape": ifm_shape,
                            "out_shape": os,
                            "local_bound": local_bound,
                        }

                        # Support for larger values than 9 needs different delimiter
                        delim = "" if max(s + p) <= 9 else "x"
                        arg_list.append(
                            (
                                "acc{}_st{}_pad{}_os{}_lclbnd{}".format(
                                    testGen.typeStr(a),
                                    delim.join([str(x) for x in s]),
                                    delim.join([str(x) for x in p]),
                                    "x".join([str(x) for x in os]),
                                    "1" if local_bound else "0",
                                ),
                                args_dict,
                            )
                        )
                    n += 1

        arg_list = TosaArgGen._add_data_generators(
            testGen,
            opName,
            shapeList,
            dtypes[0],
            arg_list,
            error_name,
        )
        # Return list of tuples: (arg_str, args_dict)
        return arg_list

    @staticmethod
    def agPad(testGen, rng, opName, shapeList, dtype, error_name=None):
        rank = len(shapeList[0])

        if error_name is None and testGen.args.oversize:
            pad_values = [6, 7, 10, 13]
        elif error_name == ErrorIf.PadSmallerZero:
            pad_values = [x for x in range(-2, 0)]
        else:
            # Exhaustively test combinations of padding on each side of each dimension
            # - the range of padding values is defined by pad_min and pad_max
            pad_min, pad_max = 0, 1
            pad_values = [x for x in range(pad_min, pad_max + 1)]

        # Calculate pad combinations
        axis_pad_values = [x for x in itertools.product(pad_values, pad_values)]
        shape_pad_values = itertools.product(*([axis_pad_values] * rank))

        if dtype in [DType.BOOL, DType.INT8, DType.INT16, DType.INT32]:
            pad_const_int = rng.randNumberDType(dtype)
            pad_const_fp = 0
        elif gtu.dtypeIsFloat(dtype):
            pad_const_int = 0
            pad_const_fp = rng.randNumberDType(dtype)
        else:
            assert error_name == ErrorIf.WrongInputType
            pad_const_int = 0
            pad_const_fp = 0

        list_shape_pad_values = list(shape_pad_values)
        # If we are producing tests for rank 6 or greater use sparsity
        if len(list_shape_pad_values) > 1024:
            sparsity_factor = 2 if error_name else 120
            sparsity = TosaArgGen._calculate_sparsity(
                len(list_shape_pad_values), sparsity_factor
            )
        else:
            sparsity = 1

        # Build arg list
        arg_list = []
        for n, paddings in enumerate(list_shape_pad_values):
            paddings = list(paddings)
            args_valid = True

            if error_name == ErrorIf.PadSmallerZero:
                # Prevent negative output shapes while ensuring still testing for negative padding
                for i in range(rank):
                    dim_after_padding = (
                        paddings[i][0] + paddings[i][1] + shapeList[0][i]
                    )
                    if dim_after_padding < 1:
                        paddings[i] = (0, 0)
                if all([p > -1 for p in paddings[i]]):
                    args_valid = False
            if args_valid and n % sparsity == 0:
                # Work out name
                pad_list = []
                for r in range(rank):
                    pad_list.extend(paddings[r])

                delim = "" if max(pad_list) <= 9 else "x"
                name = "pad{}".format(delim.join([str(x) for x in pad_list]))

                args_dict = {
                    "pad": np.array(paddings),
                    "pad_const_int": pad_const_int,
                    "pad_const_fp": pad_const_fp,
                }
                arg_list.append((name, args_dict))

        if error_name == ErrorIf.PadSmallerZero and len(arg_list) == 0:
            logger.debug(
                f"agPad: No PadSmallerZero ErrorIf test created for input shape: {shapeList[0]}"
            )

        arg_list = TosaArgGen._add_data_generators(
            testGen,
            opName,
            shapeList,
            dtype,
            arg_list,
            error_name,
        )

        # Return list of tuples: (arg_str, args_dict)
        return arg_list

    @staticmethod
    def agPooling(testGen, rng, opName, shapeList, dtype, error_name=None):
        arg_list = []

        shape = shapeList[0]
        if error_name != ErrorIf.WrongRank:
            assert len(shape) == 4

        test_level8k = testGen.args.level8k and error_name is None

        startStride = 1 if error_name != ErrorIf.PoolingOutputShapeNonInteger else 2
        startKernel = 2
        startPad = 0
        if not test_level8k:
            # Generate comprehensive argument lists
            p_vals = [x for x in range(startPad, testGen.args.max_pooling_padding + 1)]
            paddings = {x for x in itertools.product(*([p_vals] * 4))}
            # Stride must be greater than 1 to force non-integer error
            s_vals = [
                x for x in range(startStride, testGen.args.max_pooling_stride + 1)
            ]
            strides = {x for x in itertools.product(*([s_vals] * 2))}
            k_vals = [
                x for x in range(startKernel, testGen.args.max_pooling_kernel + 1)
            ]
            kernels = {x for x in itertools.product(*([k_vals] * 2))}
            max_dim_size = None
        else:
            # Only test 8k levels
            bigStride = testGen.TOSA_8K_LEVEL_MAX_STRIDE
            bigKernel = testGen.TOSA_8K_LEVEL_MAX_KERNEL
            strides = {(1, bigStride), (bigStride, 4)}
            kernels = {(1, bigKernel), (bigKernel, 3)}
            paddings = set()
            for s in sorted(list(strides)):
                for k in sorted(list(kernels)):
                    padding = []
                    for idx in range(len(k)):
                        total_padding = s[idx] - shape[idx + 1] + k[idx]
                        while total_padding < 0:
                            # Must meet: shape + padding > kernel
                            total_padding += s[idx]
                        if total_padding < k[idx]:
                            padding.extend([0, total_padding])
                        else:
                            # Note this may produce padding >= k[idx] which is not
                            # allowed - but will be ignored in the creation loop below
                            padding.extend([k[idx] - 1, total_padding - (k[idx] - 1)])
                    paddings.add(tuple(padding))
            # Create a limit for the output dimensions size
            max_dim_size = testGen.TOSA_8K_LEVEL_MAX_KERNEL

        if opName == "max_pool2d":
            accum_dtypes = [None]  # max_pool has no accumulate dtype
        elif dtype == DType.INT8 or dtype == DType.INT16:
            accum_dtypes = [DType.INT32]
        elif dtype == DType.FP16:
            accum_dtypes = [DType.FP16, DType.FP32]
        elif dtype == DType.BF16 or dtype == DType.FP32:
            accum_dtypes = [DType.FP32]
        elif dtype == DType.FP8E4M3 or dtype == DType.FP8E5M2:
            accum_dtypes = [DType.FP16]
        elif error_name is None:
            assert False, f"Invalid I/O DType for pooling: {DTypeNames[dtype]}"
        else:
            # Set to something for the ErrorIf case which has
            # incorrect input data-type
            accum_dtypes = [DType.INT32]

        if error_name == ErrorIf.WrongAccumulatorType:
            accum_dtypes = list(gtu.usableDTypes(excludes=accum_dtypes))

        if not test_level8k:
            if testGen.args.oversize:
                # add some oversize argument values
                bigStride = 7
                bigKernel = 9
                strides.update(
                    {x for x in itertools.product(*([[startStride, bigStride]] * 2))}
                )
                kernels.update(
                    {x for x in itertools.product(*([[startKernel, bigKernel]] * 2))}
                )
                if max(shape) < 64:
                    # padding must be less than the kernel size
                    bigPadding = bigKernel - 1
                    paddings.update(
                        {x for x in itertools.product(*([[startPad, bigPadding]] * 4))}
                    )

            if error_name:
                # Cycle through all error_if tests but we only keep the first few
                sparsity = 1
            else:
                # There are too many parameter combinations, so generate them sparsely
                sparsity_factor = 500
                sparsity = (
                    len(paddings) * len(strides) * len(kernels) // sparsity_factor + 1
                )
        else:
            # We have already limited test output combinations for 8k tests
            sparsity = 1

        arg_str = (
            "acc{}_st{}_kern{}_pad{}"
            if accum_dtypes[0] is not None
            else "st{}_kern{}_pad{}"
        )

        def get_arg_list_element(accum, stride, pad, kern, dot_products=0, shape=[]):
            # Return tuple containing the formatted argument string and
            # the corresponding argument values in a dictionary

            # Support for larger values than 9 needs different delimiter
            delim = "" if max(stride + kern + pad) <= 9 else "x"
            arg_str_elems = [
                delim.join([str(x) for x in stride]),
                delim.join([str(x) for x in kern]),
                delim.join([str(x) for x in pad]),
            ]
            args_dict = {
                "stride": stride,
                "pad": pad,
                "kernel": kern,
                "dot_products": dot_products,  # Ignored for error tests
                "shape": shape,
                "ks": gtu.product(kern),  # avg_pool2d: KS = KX*KY
            }

            if accum is not None:
                arg_str_elems.insert(0, testGen.typeStr(accum))
                args_dict["acc_type"] = accum
            return (arg_str.format(*arg_str_elems), args_dict)

        more_tests = True
        n = 0
        for a in accum_dtypes:
            for s in sorted(list(strides)):
                for p in sorted(list(paddings)):
                    for k in sorted(list(kernels)):
                        if error_name in [
                            ErrorIf.StrideSmallerOne,
                            ErrorIf.KernelSmallerOne,
                            ErrorIf.PadSmallerZero,
                            ErrorIf.PadLargerEqualKernel,
                        ]:
                            sNew, pNew, kNew = TosaErrorIfArgGen.eiPoolingErrorIf(
                                rng, error_name, s, p, k
                            )
                            if None not in [sNew, pNew, kNew] and n % sparsity == 0:
                                arg_list.append(
                                    get_arg_list_element(a, sNew, pNew, kNew, shape)
                                )
                        elif (
                            more_tests
                            and n % sparsity == 0
                            # padding must not exceed the kernel size
                            and p[0] < k[0]
                            and p[1] < k[0]
                            and p[2] < k[1]
                            and p[3] < k[1]
                            # the padded shape must exceed the kernel size
                            and (shape[1] + p[0] + p[1]) > k[0]
                            and (shape[2] + p[2] + p[3]) > k[1]
                        ):
                            partial_h = shape[1] + p[0] + p[1] - k[0]
                            partial_w = shape[2] + p[2] + p[3] - k[1]
                            remainder_h = partial_h % s[0]
                            remainder_w = partial_w % s[1]
                            output_h = partial_h // s[0] + 1
                            output_w = partial_w // s[1] + 1
                            logger.debug(
                                f"agPooling: {shape} remainder=({remainder_h}, {remainder_w}) output=({output_h}, {output_w})"
                            )
                            if (
                                # the parameters must produce integer exact output
                                error_name != ErrorIf.PoolingOutputShapeNonInteger
                                and remainder_h == 0
                                and remainder_w == 0
                            ) or (
                                error_name == ErrorIf.PoolingOutputShapeNonInteger
                                and (remainder_h != 0 or remainder_w != 0)
                            ):
                                if (
                                    max_dim_size is not None
                                    and max(output_h, output_w) > max_dim_size
                                ):
                                    # Test will consume too much memory - skip it
                                    continue
                                # Dot products = N*OH*OW*C
                                dp = gtu.product(
                                    (shape[0], output_h, output_w, shape[3])
                                )
                                arg_list.append(
                                    get_arg_list_element(a, s, p, k, dp, shape)
                                )
                                if (
                                    error_name
                                    and len(arg_list) >= TosaArgGen.MAX_TESTS_ERROR_IFS
                                ):
                                    # Found enough errors
                                    logger.debug(
                                        f"Skipping creating more pooling error tests for {error_name}"
                                    )
                                    more_tests = False

                        n += 1

        # Now add data generator types
        arg_list = TosaArgGen._add_data_generators(
            testGen,
            opName,
            shapeList,
            dtype,
            arg_list,
            error_name,
        )

        if gtu.dtypeIsFloat(dtype) and gtu.has_nan_mode_by_name(opName):
            arg_list = TosaArgGen._append_nan_mode(rng, arg_list)

        # Return list of tuples: (arg_str, args_dict)
        return arg_list

    @staticmethod
    def agCast(testGen, rng, opName, shapeList, inDtype, error_name=None):
        arg_list = []

        supported = testGen.args.profile + testGen.args.extension
        dtypeList = []

        # Enumerate the output types here
        if error_name == ErrorIf.WrongOutputType:
            dtypeList = TosaErrorIfArgGen.eiCastErrorIf(inDtype)
        elif inDtype in (DType.INT8, DType.INT16, DType.INT32):
            if TosaProfiles.TosaProINT in supported:
                # Get the common list of output types without the input type
                outDtypes = [DType.BOOL, DType.INT8, DType.INT16, DType.INT32]
                outDtypes.remove(inDtype)
                dtypeList.extend(outDtypes)
            if TosaProfiles.TosaProFP in supported:
                dtypeList.extend([DType.FP16, DType.FP32])
            if TosaProfiles.TosaExtBF16 in supported:
                dtypeList.extend([DType.BF16])
        elif inDtype == DType.BOOL:
            if TosaProfiles.TosaProINT in supported:
                dtypeList.extend([DType.INT8, DType.INT16, DType.INT32])
        elif inDtype == DType.FP16:
            if TosaProfiles.TosaProFP in supported:
                dtypeList.extend([DType.INT8, DType.INT16, DType.INT32, DType.FP32])
            if TosaProfiles.TosaExtFP8E4M3 in supported:
                dtypeList.extend([DType.FP8E4M3])
            if TosaProfiles.TosaExtFP8E5M2 in supported:
                dtypeList.extend([DType.FP8E5M2])
        elif inDtype == DType.BF16:
            if TosaProfiles.TosaExtBF16 in supported:
                dtypeList.extend([DType.INT8, DType.INT16, DType.INT32, DType.FP32])
                # Need EXT-BF16 and the EXT-FP8 extensions
                if TosaProfiles.TosaExtFP8E4M3 in supported:
                    dtypeList.extend([DType.FP8E4M3])
                if TosaProfiles.TosaExtFP8E5M2 in supported:
                    dtypeList.extend([DType.FP8E5M2])
        elif inDtype == DType.FP32:
            if TosaProfiles.TosaProFP in supported:
                dtypeList.extend([DType.INT8, DType.INT16, DType.INT32, DType.FP16])
            if TosaProfiles.TosaExtBF16 in supported:
                dtypeList.extend([DType.BF16])
            if TosaProfiles.TosaExtFP8E4M3 in supported:
                dtypeList.extend([DType.FP8E4M3])
            if TosaProfiles.TosaExtFP8E5M2 in supported:
                dtypeList.extend([DType.FP8E5M2])
        elif inDtype == DType.FP8E4M3:
            if TosaProfiles.TosaExtFP8E4M3 in supported:
                dtypeList.extend([DType.FP16, DType.FP32])
                if TosaProfiles.TosaExtBF16 in supported:
                    dtypeList.extend([DType.BF16])
        elif inDtype == DType.FP8E5M2:
            if TosaProfiles.TosaExtFP8E5M2 in supported:
                dtypeList.extend([DType.FP16, DType.FP32])
                if TosaProfiles.TosaExtBF16 in supported:
                    dtypeList.extend([DType.BF16])
        elif error_name == ErrorIf.WrongInputType:
            # Pick some potentially correct output type for incorrect input type
            dtypeList = [DType.BOOL, DType.INT8, DType.INT16, DType.FP32]
        else:
            raise Exception("Unexpected input dtype: {}".format(inDtype))

        for dtype in dtypeList:
            arg_list.append(
                ("out{}".format(testGen.typeStr(dtype)), {"out_type": dtype})
            )

        # Now add data generator types
        arg_list = TosaArgGen._add_data_generators(
            testGen,
            opName,
            shapeList,
            inDtype,
            arg_list,
            error_name,
        )

        return arg_list

    @staticmethod
    def agRescale(testGen, rng, opName, shapeList, inDtype, error_name=None):
        arg_list = []

        # Enumerate the output types here
        for outDtype in [
            DType.UINT8,
            DType.INT8,
            DType.INT16,
            DType.INT32,
            DType.UINT16,
        ]:
            if (
                outDtype in [DType.UINT8, DType.INT8, DType.UINT16]
                and error_name == ErrorIf.OutputZeroPointNotZero
            ):
                continue
            if (
                outDtype != DType.UINT16
                and error_name == ErrorIf.U16OutputZeroPointNotValid
            ) or (
                inDtype != DType.UINT16
                and error_name == ErrorIf.U16InputZeroPointNotValid
            ):
                # ErrorIfs only valid with UINT16
                continue
            if (
                inDtype == DType.UINT8
                and outDtype not in [DType.INT8, DType.INT16]
                and error_name != ErrorIf.WrongOutputType
            ):
                # The only output dtypes for UINT8 are INT8/INT16, skip all others
                continue
            if (
                inDtype not in [DType.INT8, DType.INT16]
                and outDtype == DType.UINT8
                and error_name != ErrorIf.WrongOutputType
            ):
                # The only input dtypes for UINT8 are INT8/INT16, skip all others
                continue
            if (
                inDtype == DType.UINT16
                and outDtype != DType.INT16
                and error_name != ErrorIf.WrongOutputType
            ):
                # The only output dtype for UINT16 is INT16, skip all others
                continue
            if (
                inDtype != DType.INT16
                and outDtype == DType.UINT16
                and error_name != ErrorIf.WrongOutputType
            ):
                # The only input dtype for UINT16 is INT16, skip all others
                continue
            if (
                error_name == ErrorIf.WrongOutputType
                and not TosaErrorIfArgGen.eiRescaleWrongOutputType(inDtype, outDtype)
            ):
                continue

            for scale32 in [False, True]:
                if error_name == ErrorIf.ScaleTrue and not scale32:
                    continue
                elif error_name == ErrorIf.ScaleNotTrue and scale32:
                    continue
                for double_round in [False, True]:
                    if error_name == ErrorIf.ScaleNotTrue and not double_round:
                        continue
                    # Per_channel is only valid with rank > 0
                    pc_options = (False, True) if len(shapeList[0]) > 0 else (False,)
                    for per_channel in pc_options:
                        if (
                            inDtype == DType.INT48
                            and scale32
                            and error_name != ErrorIf.ScaleTrue
                        ):
                            # Illegal condition.  Must be scale32=False
                            continue
                        if (
                            double_round
                            and not scale32
                            and error_name != ErrorIf.ScaleNotTrue
                        ):
                            # Illegal condition.  ERROR_IF(!scale32 && double_round)
                            continue

                        if per_channel:
                            nc = shapeList[0][-1]
                        else:
                            nc = 1

                        in_type_width = gtu.dtypeWidth(inDtype)
                        out_type_width = gtu.dtypeWidth(outDtype)

                        # Calculate scale based on:
                        # scale = a *(2^output_width)/(2^input_width))

                        a = np.float32(rng.random(size=[nc]))
                        scale_arr = a * np.float32(
                            (1 << out_type_width) / (1 << in_type_width)
                        )

                        if scale32:
                            # Cap the scaling at 2^31 - 1 for scale32
                            scale_arr = np.clip(
                                scale_arr, 1.0 / (1 << 31), (1 << 31) - 1
                            )
                        else:
                            # Cap the scaling at 2^15 - 1 for scale16
                            scale_arr = np.clip(scale_arr, 1.0 / (1 << 31), 32767.0)

                        logger.debug(
                            f"agRescale: {out_type_width} {in_type_width} -> {scale_arr}"
                        )

                        multiplier_arr = np.int32(np.zeros(shape=[nc]))
                        shift_arr = np.int32(np.zeros(shape=[nc]))
                        for i in range(nc):
                            (
                                multiplier_arr[i],
                                shift_arr[i],
                            ) = TosaQuantGen.computeMultiplierAndShift(
                                scale_arr[i], scale32
                            )

                        arg_list.append(
                            (
                                "out{}_sc{}_dr{}_pc{}".format(
                                    testGen.typeStr(outDtype),
                                    int(scale32),
                                    int(double_round),
                                    int(per_channel),
                                ),
                                {
                                    "output_dtype": outDtype,
                                    "scale": scale32,
                                    "double_round": double_round,
                                    "per_channel": per_channel,
                                    "multiplier": multiplier_arr,
                                    "shift": shift_arr,
                                },
                            )
                        )

        arg_list = TosaArgGen._add_data_generators(
            testGen,
            opName,
            shapeList,
            inDtype,
            arg_list,
            error_name,
        )
        # Return list of tuples: (arg_str, args_dict)
        return arg_list

    @staticmethod
    def agMul(testGen, rng, opName, shapeList, dtype, error_name=None):
        arg_list = []

        if dtype is DType.INT32:
            for p in range(testGen.args.num_rand_permutations):
                shift = rng.randInt(0, 32)
                arg_list.append(("perm{}_shift{}".format(p, shift), {"shift": shift}))
        else:
            arg_list.append(("perm0_shift0", {"shift": 0}))

        arg_list = TosaArgGen._add_data_generators(
            testGen,
            opName,
            shapeList,
            dtype,
            arg_list,
            error_name,
        )
        # Return list of tuples: (arg_str, args_dict)
        return arg_list

    @staticmethod
    def agArithmeticRightShift(testGen, rng, opName, shapeList, dtype, error_name=None):
        arg_list = []

        for round in (True, False):
            args_dict = {
                "round": round,
            }
            arg_list.append((f"round{round}", args_dict))

        arg_list = TosaArgGen._add_data_generators(
            testGen,
            opName,
            shapeList,
            dtype,
            arg_list,
            error_name,
        )
        # Return list of tuples: (arg_str, args_dict)
        return arg_list

    @staticmethod
    def agFFT2d(testGen, rng, opName, shapeList, dtype, error_name=None):
        arg_list = []

        shape = shapeList[0]
        dot_products = gtu.product(shape)
        ks = 2 * shape[1] * shape[2]  # 2*H*W
        for inverse in (True, False):
            args_dict = {
                "dot_products": dot_products,
                "shape": shape,
                "ks": ks,
                "acc_type": dtype,
                "inverse": inverse,
            }
            arg_list.append((f"inverse{inverse}", args_dict))

        arg_list = TosaArgGen._add_data_generators(
            testGen,
            opName,
            shapeList,
            dtype,
            arg_list,
            error_name,
        )
        # Return list of tuples: (arg_str, args_dict)
        return arg_list

    @staticmethod
    def agRFFT2d(testGen, rng, opName, shapeList, dtype, error_name=None):
        arg_list = []

        shape = shapeList[0]
        dot_products = gtu.product(shape)
        ks = shape[1] * shape[2]  # H*W
        args_dict = {
            "dot_products": dot_products,
            "shape": shape,
            "ks": ks,
            "acc_type": dtype,
        }
        arg_list.append(("", args_dict))

        arg_list = TosaArgGen._add_data_generators(
            testGen,
            opName,
            shapeList,
            dtype,
            arg_list,
            error_name,
        )
        # Return list of tuples: (arg_str, args_dict)
        return arg_list

    # Helper function for reshape.  Gets some factors of a larger number.
    @staticmethod
    def getFactors(val, start=1):
        factors = []

        for i in range(start, int(np.sqrt(val)) + 1):
            if (val % i) == 0:
                factors.append(i)

        # Valid factor is the number its self
        factors.append(val)

        return factors

    @staticmethod
    def agReshape(testGen, rng, opName, shapeList, dtype, error_name=None):
        arg_list = []

        origShape = shapeList[0]
        totalElements = gtu.product(origShape)
        factors = TosaArgGen.getFactors(totalElements)

        # Find new shapes up to the number of permutations asked for
        # This code is NOT fast.  Fortunately, the numbers are fairly small.
        for p in range(testGen.args.num_rand_permutations):
            if totalElements > 1:
                # Can't rescale to a rank 0 with more than one element
                startRank = 1
            else:
                assert totalElements == 1
                startRank = 0

            # Rank from 0/1 to MAX_TENSOR_RANK
            newRank = rng.randInt(startRank, (gtu.MAX_TENSOR_RANK + 1))

            # escape_counter limits the generation of new shapes to a reasonable time
            for escape_counter in range(100):
                # Generate the new shape of the chosen new rank
                newShape = []
                remainingElements = totalElements
                shuffledFactors = rng.permutation(factors)
                for i in range(1, newRank):
                    # pick rank-1 factors
                    newShape.append(shuffledFactors[0])
                    remainingElements = remainingElements // shuffledFactors[0]
                    shuffledFactors = rng.permutation(
                        TosaArgGen.getFactors(remainingElements)
                    )
                if newRank > 0:
                    newShape.append(remainingElements)

                # Check for duplicates
                duplicate = False
                for name, args_dict in arg_list:
                    if args_dict["new_shape"] == newShape:
                        duplicate = True
                        break

                if not duplicate:
                    outShape = testGen.shapeStr(newShape)
                    arg_list.append(
                        (
                            "perm{}_rank{}_out{}".format(p, newRank, outShape),
                            {"new_shape": newShape},
                        )
                    )
                    # Found an output shape for this permutation
                    break

        # Now add data generator types
        arg_list = TosaArgGen._add_data_generators(
            testGen,
            opName,
            shapeList,
            dtype,
            arg_list,
            error_name,
        )

        return arg_list

    @staticmethod
    def agTranspose(testGen, rng, opName, shapeList, dtype, error_name=None):
        arg_list = []

        ifm_shape = shapeList[0]

        if error_name == ErrorIf.IndexOutsideBounds:
            incorrect_large_index = range(len(ifm_shape) + 1, 2 * len(ifm_shape) + 1)
            incorrect_small_index = range(-len(ifm_shape), 0)
            permutations = [p for p in itertools.permutations(incorrect_large_index)]
            permutations.extend(
                [p for p in itertools.permutations(incorrect_small_index)]
            )
        elif error_name == ErrorIf.IndexUsedTwice:
            # Create list with a duplicated index
            perm_range = list(range(len(ifm_shape)))
            index_choice = rng.choice(range(len(perm_range)))
            perm_range[(index_choice + 1) % len(perm_range)] = perm_range[index_choice]
            permutations = [p for p in itertools.permutations(perm_range)]

        else:
            # Get all permutations
            permutations = [p for p in itertools.permutations(range(len(ifm_shape)))]

        # Limit to possible permutations from shape dimension or argument setting
        limit = min(len(permutations), testGen.args.num_rand_permutations)

        # Get random permutation generator that uses all permutations
        random_permutations = rng.permutation(permutations)

        # Create list of required amount of permutations
        arg_list = [
            ("perm{}".format(p), {"perms": random_permutations[p].tolist()})
            for p in range(limit)
        ]
        # Now add data generator types
        arg_list = TosaArgGen._add_data_generators(
            testGen,
            opName,
            shapeList,
            dtype,
            arg_list,
            error_name,
        )
        # Return list of tuples: (arg_str, args_dict)
        return arg_list

    @staticmethod
    def agSlice(testGen, rng, opName, shapeList, dtype, error_name=None):
        arg_list = []

        ifm_shape = shapeList[0]
        rank = len(ifm_shape)

        for p in range(testGen.args.num_rand_permutations):
            start = []
            size = []

            for i in range(rank):
                if ifm_shape[i] > 1:
                    # Start from 0 to dimension size - 1 to leave room for slice of 1
                    start.append(rng.randInt(0, ifm_shape[i]))
                    # Size from 1 up to rest of room (dimension size - start)
                    size.append(rng.randInt(1, ifm_shape[i] + 1 - start[i]))

                    # Should never hit an invalid slice size
                    assert size[i] > 0 and (size[i] + start[i]) <= ifm_shape[i]
                else:
                    start.append(0)
                    size.append(1)

            # If ERROR_IF test required then incorrect start, size will be returned
            start, size = TosaErrorIfArgGen.eiSliceErrorIf(
                rng, error_name, ifm_shape, start, size
            )
            append = True
            for _, d in arg_list:
                if d["start"] == start and d["size"] == size:
                    # Already have a test for this
                    append = False
            if append:
                arg_list.append(("perm{}".format(p), {"start": start, "size": size}))

        # Now add data generator types
        arg_list = TosaArgGen._add_data_generators(
            testGen,
            opName,
            shapeList,
            dtype,
            arg_list,
            error_name,
        )
        # Return list of tuples: (arg_str, args_dict)
        return arg_list

    @staticmethod
    def agTile(testGen, rng, opName, shapeList, dtype, error_name=None):
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
                    # Multiple of 1 if ifm_shape dimension is large to reduce
                    # tensor size
                    multiples.append(1)
                elif max(ifm_shape) > 1000:
                    multiples.append(2)
                else:
                    multiples.append(rng.randInt(1, 4))
            arg_list.append(("perm{}".format(p), {"multiples": multiples}))

        # Now add data generator types
        arg_list = TosaArgGen._add_data_generators(
            testGen,
            opName,
            shapeList,
            dtype,
            arg_list,
            error_name,
        )
        # Return list of tuples: (arg_str, args_dict)
        return arg_list

    @staticmethod
    def agResize(testGen, rng, opName, shapeList, dtype, error_name=None):
        arg_list = []
        ifm_shape = shapeList[0]

        def get_aspect_ratio_resize_params():
            common_aspect_ratios = ((3, 2), (16, 9), (4, 3))
            aspect_ratio = rng.choice(common_aspect_ratios)
            invert = rng.choice((False, True))
            letterbox = rng.choice((False, True))

            scale_y_n = aspect_ratio[0] if invert else aspect_ratio[1]
            scale_x_n = aspect_ratio[1] if invert else aspect_ratio[0]
            scale_y_d = scale_x_d = 1
            offset_x = offset_y = 0

            if letterbox:
                max_border = scale_y_n
                border_y = rng.randInt(low=0, high=max_border)
                border_x = 0
            else:
                # Pillarboxing
                border_y = 0
                max_border = scale_x_n
                border_x = rng.randInt(low=0, high=max_border)

            scale = (scale_y_n, scale_y_d, scale_x_n, scale_x_d)
            offset = (offset_y, offset_x)
            border = (border_y, border_x)

            return scale, offset, border

        def get_upscale_downscale_params():
            valid_params = False
            while not valid_params:
                upscale = rng.choice((False, True))

                # True if sampling begins from (0,0). Otherwise (-0.5,-0.5)
                origin_sampling = rng.choice((False, True))

                if upscale:
                    shift = rng.randInt(low=1, high=4)
                    scale_x_d = scale_y_d = 1
                    scale_x_n = scale_y_n = (
                        1 << shift if origin_sampling else 2 << shift
                    )
                    border_x = border_y = 0 if origin_sampling else (1 << shift) - 1
                    offset_x = offset_y = 0 if origin_sampling else -(1 << shift) + 1
                else:
                    scale_x_n = 1
                    scale_y_n = 1

                    # Return list of valid scale_*_d values (max value 4) given input dim shape
                    def get_valid_denom(ifm_dim):
                        return [x for x in range(1, 5) if ifm_dim % x == 1]

                    # Generate list of valid downscale values and choose one randomly
                    valid_scale_y_ds = get_valid_denom(ifm_shape[1])
                    valid_scale_x_ds = get_valid_denom(ifm_shape[2])

                    if not valid_scale_y_ds and not valid_scale_x_ds:
                        # Bad parameters, skip
                        continue

                    if not valid_scale_y_ds:
                        scale_y_d = 1
                    else:
                        scale_y_d = rng.choice(valid_scale_y_ds)

                    if not valid_scale_x_ds:
                        scale_x_d = 1
                    else:
                        scale_x_d = rng.choice(valid_scale_x_ds)

                    border_x = border_y = 0
                    offset_y = rng.randInt(0, 16 * scale_y_n)
                    offset_x = rng.randInt(0, 16 * scale_x_n)
                valid_params = True

            scale = (scale_y_n, scale_y_d, scale_x_n, scale_x_d)
            offset = (offset_y, offset_x)
            border = (border_y, border_x)
            return scale, offset, border

        def get_rand_params():
            def fix_scale_to_max_scale(scale_n, scale_d, max_scale):
                scale = scale_n / scale_d
                if scale > max_scale:
                    factor = scale / max_scale
                    new_scale_d = math.ceil(scale_d * factor)
                    assert scale_n / new_scale_d <= max_scale
                    scale_d = new_scale_d
                return scale_d

            # Scale
            scale_y_n = rng.randInt(low=1, high=(1 << 11))
            scale_x_n = rng.randInt(low=1, high=(1 << 11))

            scale_y_d = rng.randInt(low=1, high=(16 * scale_y_n))
            scale_x_d = rng.randInt(low=1, high=(16 * scale_x_n))

            scale_y_d = fix_scale_to_max_scale(
                scale_y_n, scale_y_d, testGen.TOSA_8K_LEVEL_MAX_SCALE
            )
            scale_x_d = fix_scale_to_max_scale(
                scale_x_n, scale_x_d, testGen.TOSA_8K_LEVEL_MAX_SCALE
            )

            # Offsets and border within the scale
            offset_y = rng.randInt(low=-scale_y_n, high=(16 * scale_y_n))
            offset_x = rng.randInt(low=-scale_x_n, high=(16 * scale_x_n))
            border_y = rng.randInt(low=(-16 * scale_y_n), high=scale_y_n)
            border_x = rng.randInt(low=(-16 * scale_x_n), high=scale_x_n)

            scale = (scale_y_n, scale_y_d, scale_x_n, scale_x_d)
            offset = (offset_y, offset_x)
            border = (border_y, border_x)
            return scale, offset, border

        def get_level_8k_params():
            # Create 64x scale - 64/1 to 2048/32
            scale_d = rng.randInt(
                low=1, high=(1 << 11) / testGen.TOSA_8K_LEVEL_MAX_SCALE
            )
            scale_n = scale_d * testGen.TOSA_8K_LEVEL_MAX_SCALE
            # Create half to fifth scaling
            scale_d_alt = rng.randInt(low=2, high=6)
            scale_n_alt = 1
            switch = rng.choice((False, True))
            if switch:
                scale = (scale_n_alt, scale_d_alt, scale_n, scale_d)
            else:
                scale = (scale_n, scale_d, scale_n_alt, scale_d_alt)

            offset_y = rng.choice((-scale[0], 0, (16 * scale[0]) - 1))
            offset_x = rng.choice((-scale[2], 0, (16 * scale[2]) - 1))
            offset = (offset_y, offset_x)
            border_y = rng.choice((-16 * scale[0], 0, scale[0] - 1))
            border_x = rng.choice((-16 * scale[2], 0, scale[2] - 1))
            border = (border_y, border_x)
            return scale, offset, border

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
            elif dtype == DType.FP16:
                outputDTypeList = [DType.FP16]
            elif dtype == DType.BF16:
                outputDTypeList = [DType.BF16]
            elif dtype == DType.FP32:
                outputDTypeList = [DType.FP32]
            elif dtype == DType.FP8E4M3:
                outputDTypeList = [DType.FP8E4M3]
            elif dtype == DType.FP8E5M2:
                outputDTypeList = [DType.FP8E5M2]
            elif error_name == ErrorIf.WrongInputType:
                # If an incorrect input type is used then we set a 'correct'
                # output type to avoid other errors
                outputDTypeList = [DType.INT8, DType.INT16, DType.INT32]
            else:
                continue

            arg_str = "mode{}_out{}_sc{}x{}x{}x{}_off{}x{}_bor{}x{}"

            for outputDType in outputDTypeList:
                perm = 0
                while perm < testGen.args.num_rand_permutations:
                    # Random choice of type of params we are testing
                    if not testGen.args.level8k:
                        _rnd_param_fn = rng.choice(
                            (
                                get_rand_params,
                                get_upscale_downscale_params,
                                get_aspect_ratio_resize_params,
                            )
                        )
                        scale, offset, border = _rnd_param_fn()
                    else:
                        scale, offset, border = get_level_8k_params()

                    # Expand params for bounds-checking
                    (scale_y_n, scale_y_d, scale_x_n, scale_x_d) = scale
                    (offset_y, offset_x) = offset
                    (border_y, border_x) = border

                    # Make sure output dimensions OH and OW are integers
                    partial_output_y = (
                        (ifm_shape[1] - 1) * scale_y_n - offset_y + border_y
                    )
                    partial_output_x = (
                        (ifm_shape[2] - 1) * scale_x_n - offset_x + border_x
                    )
                    if error_name == ErrorIf.ResizeOutputShapeNonInteger:
                        # Look for non-integer test
                        if (
                            partial_output_y % scale_y_d == 0
                            and partial_output_x % scale_x_d == 0
                        ):
                            # Skip this test as it doesn't produce NonInteger output
                            if perm > 0:
                                perm += 1
                            continue
                    else:
                        # Alter the scaling factors to make the output integer
                        while partial_output_y % scale_y_d != 0:
                            scale_y_d -= 1
                        while partial_output_x % scale_x_d != 0:
                            scale_x_d -= 1
                        # Make sure we are still within max scaling
                        if (
                            scale_y_n / scale_y_d
                        ) > testGen.TOSA_8K_LEVEL_MAX_SCALE or (
                            scale_x_n / scale_x_d
                        ) > testGen.TOSA_8K_LEVEL_MAX_SCALE:
                            # Skip the test as it is using too large a scaling factor
                            if perm > 0:
                                perm += 1
                            continue

                    output_y = partial_output_y // scale_y_d + 1
                    output_x = partial_output_x // scale_x_d + 1

                    if (
                        output_y >= testGen.args.max_resize_output_dim
                        or output_x >= testGen.args.max_resize_output_dim
                    ) and error_name is None:
                        # Skip positive test if output dim will be too high
                        # Avoid high test latency and OOM issues
                        if not testGen.args.level8k or perm > 0:
                            perm += 1
                        continue

                    if (
                        output_y <= 0
                        or output_y >= gtu.MAX_RESIZE_DIMENSION
                        or output_x <= 0
                        or output_x >= gtu.MAX_RESIZE_DIMENSION
                    ):
                        # Output dimensions out of scope
                        if error_name is not None and perm > 0:
                            # As long as we have one ERROR_IF test, don't worry
                            # about creating all the other permutations
                            perm += 1
                        continue

                    if error_name == ErrorIf.ResizeOutputShapeMismatch and (
                        (
                            output_y + scale_y_d >= gtu.MAX_RESIZE_DIMENSION
                            and output_y - scale_y_d < 1
                        )
                        or (
                            output_x + scale_x_d >= gtu.MAX_RESIZE_DIMENSION
                            and output_x - scale_x_d < 1
                        )
                    ):
                        # Can't create a negative test with these params as it
                        # will create invalid output size
                        if perm > 0:
                            perm += 1
                        continue

                    scale = [scale_y_n, scale_y_d, scale_x_n, scale_x_d]
                    offset = [offset_y, offset_x]
                    border = [border_y, border_x]

                    # Common for all data types
                    if error_name is not None:
                        (
                            scale,
                            offset,
                            border,
                            outputDTypeNew,
                        ) = TosaErrorIfArgGen.eiResizeErrorIf(
                            rng,
                            error_name,
                            mode,
                            dtype,
                            shapeList,
                            outputDType,
                            scale,
                            offset,
                            border,
                        )
                    else:
                        outputDTypeNew = outputDType

                    arg_to_append = (
                        arg_str.format(
                            "N" if mode == ResizeMode.NEAREST else "B",
                            testGen.typeStr(outputDTypeNew),
                            scale[0],
                            scale[1],
                            scale[2],
                            scale[3],
                            offset[0],
                            offset[1],
                            border[0],
                            border[1],
                        ),
                        {
                            "mode": mode,
                            "scale": scale,
                            "offset": offset,
                            "border": border,
                            "output_dtype": outputDTypeNew,
                        },
                    )
                    if arg_to_append in arg_list:
                        # Skip already generated test params
                        continue

                    # Valid permutation
                    perm += 1
                    arg_list.append(arg_to_append)

        # Now add data generator types
        arg_list = TosaArgGen._add_data_generators(
            testGen,
            opName,
            shapeList,
            dtype,
            arg_list,
            error_name,
        )
        # Return list of tuples: (arg_str, args_dict)
        return arg_list

    @staticmethod
    def agTable(testGen, rng, opName, shapeList, dtype, error_name=None):
        arg_list = []

        if dtype == DType.INT8:
            table = np.int32(rng.integers(low=-128, high=128, size=[256])).tolist()
        else:  # INT16
            table = np.int32(rng.integers(low=-32768, high=32768, size=[513])).tolist()
            # Make sure all slopes are within REQUIRE min/max 16-bit int
            for idx in range(len(table) - 1):
                slope = table[idx + 1] - table[idx]
                # Alter the next table entry to force the slope to be ok
                if slope > 32767:
                    table[idx + 1] -= slope - 32767
                if slope < -32768:
                    table[idx + 1] -= slope + 32768
                slope = table[idx + 1] - table[idx]
                assert slope <= 32767 and slope >= -32768
        arg_list.append(
            (
                "",
                {"table": table},
            )
        )
        # Now add data generator types
        arg_list = TosaArgGen._add_data_generators(
            testGen,
            opName,
            shapeList,
            dtype,
            arg_list,
            error_name,
        )
        # Return list of tuples: (arg_str, args_dict)
        return arg_list

    def agCondIf(testGen, rng, opName, shapeList, dtype, error_name=None):
        # CondIf generates the condition values here.
        # Convert to tensors in the build function, along with the
        # then and else blocks
        arg_list = []

        for c in [False, True]:
            arg_list.append(("cond{}".format(int(c)), {"condition": c}))

        # Now add data generator types
        arg_list = TosaArgGen._add_data_generators(
            testGen,
            opName,
            shapeList,
            dtype,
            arg_list,
            error_name,
        )
        # Return list of tuples: (arg_str, args_dict)
        return arg_list

    def agWhileLoop(testGen, rng, opName, shapeList, dtype, error_name=None):
        # While loop: 0 iterations, 1, more than 1
        arg_list = []

        for iterations in [0, 1, 4]:
            arg_list.append(("iter{}".format(iterations), {"iterations": iterations}))

        # Now add data generator types
        arg_list = TosaArgGen._add_data_generators(
            testGen,
            opName,
            shapeList,
            dtype,
            arg_list,
            error_name,
        )
        # Return list of tuples: (arg_str, args_dict)
        return arg_list

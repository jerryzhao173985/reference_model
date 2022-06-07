# Copyright (c) 2021-2022, ARM Limited.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
from generator.tosa_utils import product
from generator.tosa_utils import usableDTypes
from generator.tosa_utils import valueToName
from tosa.DType import DType
from tosa.Op import Op
from tosa.ResizeMode import ResizeMode


class ErrorIf(object):
    MaxDimExceeded = "MaxDimExceeded"
    StrideSmallerEqualZero = "StrideSmallerEqualZero"
    StrideLargerEqualMax = "StrideLargerEqualMax"
    StrideLargerDimension = "StrideLargerDimension"
    OffsetSmallerEqualMin = "OffsetSmallerEqualMin"
    OffsetLargerEqualMax = "OffsetLargerEqualMax"
    ShiftNotZero = "ShiftNotZero"
    ShiftSmallerOne = "ShiftSmallerOne"
    ShiftLargerEleven = "ShiftLargerEleven"
    WrongInputType = "WrongInputType"
    WrongOutputType = "WrongOutputType"
    WrongInputList = "WrongInputList"
    WrongOutputList = "WrongOutputList"
    WrongRank = "WrongRank"
    BatchMismatch = "BatchMismatch"
    ChannelMismatch = "ChannelMismatch"
    RankMismatch = "RankMismatch"
    DimensionMismatch = "DimensionMismatch"
    InputZeroPointNotZero = "InputZeroPointNotZero"
    WeightZeroPointNotZero = "WeightZeroPointNotZero"
    OutputZeroPointNotZero = "OutputZeroPointNotZero"
    AxisSmallerZero = "AxisSmallerZero"
    AxisLargerRank = "AxisLargerRank"
    ArgmaxOutputShapeMismatch = "ArgmaxOutputShapeMismatch"
    ArgmaxOutputRankMismatch = "ArgmaxOutputRankMismatch"
    ShapeOfAxisNotOne = "ShapeOfAxisNotOne"
    KernelSmallerOne = "KernelSmallerOne"
    StrideSmallerOne = "StrideSmallerOne"
    DilationSmallerOne = "DilationSmallerOne"
    PadSmallerZero = "PadSmallerZero"
    PadLargerEqualKernel = "PadLargerEqualKernel"
    PoolingOutputShapeMismatch = "PoolingOutputShapeMismatch"
    PoolingOutputShapeNonInteger = "PoolingOutputShapeNonInteger"
    ConvOutputShapeMismatch = "ConvOutputShapeMismatch"
    ConvOutputShapeNonInteger = "ConvOutputShapeNonInteger"
    ScaleNotTrue = "ScaleNotTrue"
    ScaleTrue = "ScaleTrue"
    TensorSizeInputOutputMismatch = "TensorSizeInputOutputMismatch"
    StartSmallerZero = "StartSmallerZero"
    SizeSmallerEqualZero = "SizeSmallerEqualZero"
    StartSizeOutsideBounds = "StartSizeOutsideBounds"
    SizeOutputShapeMismatch = "SizeOutputShapeMismatch"
    InputSizeStartLengthMismatch = "InputSizeStartLengthMismatch"
    IndexOutsideBounds = "IndexOutsideBounds"
    IndexUsedTwice = "IndexUsedTwice"
    MaxSmallerMin = "MaxSmallerMin"
    ConcatInputRankMismatch = "ConcatInputRankMismatch"
    ConcatInputDimMismatch = "ConcatInputDimMismatch"
    ConcatShapeSumMismatch = "ConcatShapeSumMismatch"
    CondIfInputListThenGraphMismatch = "CondIfInputListThenGraphMismatch"
    CondIfInputListElseGraphMismatch = "CondIfInputListElseGraphMismatch"
    CondIfOutputListThenGraphMismatch = "CondIfOutputListThenGraphMismatch"
    CondIfOutputListElseGraphMismatch = "CondIfOutputListElseGraphMismatch"
    InputListOutputListMismatch = "InputListOutputListMismatch"
    InputListCondGraphMismatch = "InputListCondGraphMismatch"
    InputListBodyGraphInputMismatch = "InputListBodyGraphInputMismatch"
    InputListBodyGraphOutputMismatch = "InputListBodyGraphOutputMismatch"
    CondGraphOutputNotMatchingBool = "CondGraphOutputNotMatchingBool"
    U16InputZeroPointNotValid = "U16InputZeroPointNotValid"
    U16OutputZeroPointNotValid = "U16OutputZeroPointNotValid"


class TosaErrorIfArgGen:
    @staticmethod
    def eiResizeErrorIf(
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
        offset_fp,
    ):

        if outputDType == DType.FLOAT:
            if error_name == ErrorIf.StrideSmallerEqualZero:
                stride_fp = testGen.rng.random(size=[2]) - 2
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
                    stride = [
                        (16 >> -shift) - 1,
                        (16 >> -shift) - 1,
                    ]  # avoids other ERROR_IF checks
                    offset = [
                        (16 >> -shift) - 1,
                        (16 >> -shift) - 1,
                    ]  # avoids other ERROR_IF checks
                else:
                    stride = [
                        (16 << shift) - 1,
                        (16 << shift) - 1,
                    ]  # avoids other ERROR_IF checks
                    offset = [
                        (16 << shift) - 1,
                        (16 << shift) - 1,
                    ]  # avoids other ERROR_IF checks
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
                incorrect_types = (
                    DType.INT4,
                    DType.INT16,
                    DType.INT32,
                    DType.INT48,
                    DType.FLOAT,
                )
            elif mode == ResizeMode.NEAREST and dtype == DType.INT16:
                incorrect_types = (
                    DType.INT4,
                    DType.INT8,
                    DType.INT32,
                    DType.INT48,
                    DType.FLOAT,
                )
            elif mode == ResizeMode.BILINEAR and dtype == DType.INT8:
                incorrect_types = (
                    DType.INT4,
                    DType.INT8,
                    DType.INT16,
                    DType.INT48,
                    DType.FLOAT,
                )
            elif mode == ResizeMode.BILINEAR and dtype == DType.INT16:
                incorrect_types = (
                    DType.INT4,
                    DType.INT8,
                    DType.INT16,
                    DType.INT32,
                    DType.FLOAT,
                )
            elif dtype == DType.FLOAT:
                incorrect_types = (
                    DType.INT4,
                    DType.INT8,
                    DType.INT16,
                    DType.INT32,
                    DType.INT48,
                )
            outputDType = testGen.rng.choice(a=incorrect_types)

        return shift, stride, stride_fp, offset, offset_fp, outputDType

    @staticmethod
    def eiPoolingErrorIf(testGen, error_name, stride, pad, kernel):
        if (
            error_name == ErrorIf.StrideSmallerOne
            # padding must not exceed the kernel size
            and pad[0] < kernel[0]
            and pad[1] < kernel[0]
            and pad[2] < kernel[1]
            and pad[3] < kernel[1]
        ):
            wrongStride = (
                testGen.rng.choice([0, -1, -2, -3]),
                testGen.rng.choice([0, -1, -2, -3]),
            )
            return wrongStride, pad, kernel
        elif error_name == ErrorIf.PadSmallerZero:
            wrongPad = (
                testGen.rng.choice([-1, -2, -3]),
                testGen.rng.choice([-1, -2, -3]),
                testGen.rng.choice([-1, -2, -3]),
                testGen.rng.choice([-1, -2, -3]),
            )
            return stride, wrongPad, kernel
        elif error_name == ErrorIf.KernelSmallerOne:
            wrongKernel = (
                testGen.rng.choice([0, -1, -2, -3]),
                testGen.rng.choice([0, -1, -2, -3]),
            )
            return stride, pad, wrongKernel
        elif error_name == ErrorIf.PadLargerEqualKernel:
            wrongPad = (
                testGen.rng.choice([kernel[0], kernel[0] + 1, kernel[0] + 2]),
                testGen.rng.choice([kernel[0], kernel[0] + 1, kernel[0] + 2]),
                testGen.rng.choice([kernel[1], kernel[1] + 1, kernel[1] + 2]),
                testGen.rng.choice([kernel[1], kernel[1] + 1, kernel[1] + 2]),
            )
            return stride, wrongPad, kernel
        else:
            return None, None, None

    @staticmethod
    def eiRescaleWrongOutputType(input_dtype, output_dtype):
        if input_dtype == DType.INT8:
            if output_dtype not in [DType.UINT8, DType.INT8, DType.INT16, DType.INT32]:
                return True
        elif input_dtype == DType.INT16:
            if output_dtype not in [
                DType.UINT8,
                DType.INT8,
                DType.UINT16,
                DType.INT16,
                DType.INT32,
            ]:
                return True
        elif input_dtype == DType.INT32:
            if output_dtype not in [DType.INT8, DType.INT16, DType.INT32]:
                return True
        elif input_dtype == DType.INT48:
            if output_dtype not in [DType.INT8, DType.INT16, DType.INT32]:
                return True
        elif input_dtype == DType.UINT8:
            if output_dtype not in [DType.INT8, DType.INT16]:
                return True
        elif input_dtype == DType.UINT16:
            if output_dtype != DType.INT16:
                return True
        return False

    @staticmethod
    def eiInvalidateInputOutputList(testGen, error_name, input_list, output_list):
        # Mess up input/output tensors for ERROR_IF checks
        if error_name == "WrongInputList":
            add_input = testGen.rng.choice([True, False])
            if add_input:
                input_list.append("eiDummyInput")
            else:
                input_list = input_list[:-1]
        elif error_name == "WrongOutputList":
            add_output = testGen.rng.choice([True, False])
            if add_output:
                output_list.append("eiDummyOutput")
            else:
                output_list = []
        return input_list, output_list

    @staticmethod
    def eiRestrictDimensions(shape, max_dim=32, max_items=100000):
        """Restrict the dimensions and overall size of a shape to
        max_dim and max_items.
        """
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
                newStart.append(input_shape[i] - 1)
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
        """Check ERROR_IF statements are caught and set the expected result.

        Args:
            serializer: the serializer to set the expected result in
            validator_fcns: a sequence of validator functions to verify the result
            error_name: the name of the ERROR_IF condition to check for
            kwargs: keyword arguments for the validator functions
        Returns:
            True if the result matches the expected result; otherwise False
        """
        overall_result = True
        for val_fcn in validator_fcns:
            val_result = val_fcn(True, **kwargs)
            validator_name = val_result["error_name"]
            error_result = val_result["error_result"]
            error_reason = val_result["error_reason"]

            # expect an error IFF the error_name and validator_name match
            expected_result = error_result == (error_name == validator_name)
            overall_result &= expected_result

            if expected_result and error_result:
                serializer.setExpectedReturnCode(2, True, desc=error_reason)
            elif error_result:  # and not expected_result
                print(
                    f"Unexpected ERROR_IF: Op: {valueToName(Op, kwargs['op']['op'])}"
                    f" Expected: {error_name}, Got: {validator_name}"
                )
            elif not expected_result:  # and not error_result
                print(
                    f"Missed ERROR_IF: Op: {valueToName(Op, kwargs['op']['op'])}"
                    f" Expected: {error_name}"
                )

            if not expected_result:
                for k, v in sorted(kwargs.items()):
                    if k != "op":
                        if k.endswith("dtype"):
                            v = valueToName(DType, v)
                        print(f"  {k} = {v}")

        return overall_result

    @staticmethod
    def evWrongInputType(check=False, **kwargs):
        error_result = False

        # Find the unsupported input data types
        op = kwargs["op"]
        input_dtypes = op["types"]
        allowed_input_dtypes = {
            t[0] if isinstance(t, list) else t for t in input_dtypes
        }
        wrong_input_dtypes = list(usableDTypes(excludes=allowed_input_dtypes))

        if op["op"] == Op.CLAMP:
            wrong_input_dtypes.remove(DType.INT48)

        if check:
            input_dtype = kwargs["input_dtype"]
            if input_dtype not in allowed_input_dtypes:
                error_result = True

        info_dict = {
            "error_name": ErrorIf.WrongInputType,
            "error_result": error_result,
            "error_reason": "Input data type not supported for this operator",
            "param_reqs": {"rank": None, "dtype": wrong_input_dtypes, "shape": None},
        }
        return info_dict

    @staticmethod
    def evWrongOutputType(check=False, **kwargs):
        error_result = False

        if check:
            input_dtype = kwargs["input_dtype"]
            output_dtype = kwargs["output_dtype"]
            op = kwargs["op"]

            if op["op"] == Op.RESIZE:
                mode = kwargs["mode"]
                if (
                    (
                        mode == ResizeMode.NEAREST
                        and input_dtype == DType.INT8
                        and output_dtype != DType.INT8
                    )
                    or (
                        mode == ResizeMode.NEAREST
                        and input_dtype == DType.INT16
                        and output_dtype != DType.INT16
                    )
                    or (
                        mode == ResizeMode.BILINEAR
                        and input_dtype == DType.INT8
                        and output_dtype != DType.INT32
                    )
                    or (
                        mode == ResizeMode.BILINEAR
                        and input_dtype == DType.INT16
                        and output_dtype != DType.INT48
                    )
                    or (input_dtype == DType.FLOAT and output_dtype != DType.FLOAT)
                ):
                    error_result = True

            elif op["op"] == Op.RESCALE:
                error_result = TosaErrorIfArgGen.eiRescaleWrongOutputType(
                    input_dtype, output_dtype
                )

            elif op["op"] in [Op.FULLY_CONNECTED, Op.MATMUL]:
                if (
                    (input_dtype == DType.INT8 and output_dtype != DType.INT32)
                    or (input_dtype == DType.INT16 and output_dtype != DType.INT48)
                    or (input_dtype == DType.FLOAT and output_dtype != DType.FLOAT)
                ):
                    error_result = True

            elif op["op"] == Op.ARGMAX:
                if (
                    input_dtype in [DType.INT8, DType.INT16, DType.FLOAT]
                    and output_dtype != DType.INT32
                ):
                    error_result = True

            elif op["op"] == Op.MUL:
                if input_dtype != DType.FLOAT and output_dtype != DType.INT32:
                    error_result = True
                elif input_dtype == DType.FLOAT and output_dtype != DType.FLOAT:
                    error_result = True

            elif op["op"] == Op.TABLE:
                if input_dtype == DType.INT8 and output_dtype != DType.INT8:
                    error_result = True
                elif input_dtype == DType.INT16 and output_dtype != DType.INT32:
                    error_result = True

            elif op["op"] in [Op.EQUAL, Op.GREATER_EQUAL, Op.GREATER]:
                if output_dtype != DType.BOOL:
                    error_result = True

            elif op["op"] == Op.CAST:
                if (
                    (
                        input_dtype == DType.BOOL
                        and output_dtype not in [DType.INT8, DType.INT16, DType.INT32]
                    )
                    or (
                        input_dtype == DType.INT8
                        and output_dtype
                        not in [DType.BOOL, DType.INT16, DType.INT32, DType.FLOAT]
                    )
                    or (
                        input_dtype == DType.INT16
                        and output_dtype
                        not in [DType.BOOL, DType.INT8, DType.INT32, DType.FLOAT]
                    )
                    or (
                        input_dtype == DType.INT32
                        and output_dtype
                        not in [DType.BOOL, DType.INT8, DType.INT16, DType.FLOAT]
                    )
                    or (
                        input_dtype == DType.FLOAT
                        and output_dtype not in [DType.INT8, DType.INT16, DType.INT32]
                    )
                ):
                    error_result = True

            elif op["op"] in {
                Op.CONV2D,
                Op.CONV3D,
                Op.DEPTHWISE_CONV2D,
                Op.TRANSPOSE_CONV2D,
            }:
                if (
                    input_dtype == DType.INT8
                    and output_dtype != DType.INT32
                    or input_dtype == DType.INT16
                    and output_dtype != DType.INT48
                    or input_dtype == DType.FLOAT
                    and output_dtype != DType.FLOAT
                ):
                    error_result = True
                # invalid input types are ignored, to avoid reporting multiple errors

            else:
                if output_dtype != input_dtype:
                    error_result = True

        info_dict = {
            "error_name": ErrorIf.WrongOutputType,
            "error_result": error_result,
            "error_reason": (
                "Output data type not supported for this configuration of operator"
            ),
            "param_reqs": {"rank": None, "dtype": None, "shape": None},
        }
        return info_dict

    @staticmethod
    def evWrongRank(check=False, **kwargs):
        all_ranks = (1, 2, 3, 4, 5)

        # Make a list of incorrect ranks
        assert "op" in kwargs
        op = kwargs["op"]
        rmin, rmax = op["rank"]
        rank_range = range(rmin, rmax + 1)
        incorrect_ranks = list(set(all_ranks) - set(rank_range))
        # Remove small incorrect ranks to avoid index errors
        incorrect_ranks = [rank for rank in incorrect_ranks if rank > rmin]
        # Set minimum incorrect rank to 3 to avoid index error
        if op["op"] in [Op.RESIZE]:
            incorrect_ranks = [3, 5]
        elif op["op"] in [Op.TRANSPOSE]:
            incorrect_ranks = [7, 8]
        elif op["op"] in [Op.CONV3D]:
            incorrect_ranks = [6, 7]

        error_name = ErrorIf.WrongRank
        param_reqs = {"rank": incorrect_ranks, "dtype": None, "shape": None}
        error_result = False
        error_reason = "Rank not supported for this operator"

        if check:
            input_shape = kwargs["input_shape"]

            if (
                op["op"] in [Op.RESIZE, Op.AVG_POOL2D, Op.MAX_POOL2D]
                and len(input_shape) != 4
            ):
                error_result = True
            elif op["op"] == Op.FULLY_CONNECTED and len(input_shape) != 2:
                error_result = True
            elif op["op"] == Op.MATMUL and len(input_shape) != 3:
                error_result = True
            else:
                if len(input_shape) not in rank_range:
                    error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs,
        }
        return info_dict

    @staticmethod
    def evWrongInputList(check=False, **kwargs):
        error_name = ErrorIf.WrongInputList
        param_reqs = {"rank": None, "dtype": None, "shape": None}
        error_result = False
        error_reason = "Op input list does not match expected input"

        if check:
            op = kwargs["op"]
            input_list = kwargs["input_list"]
            num_operands = kwargs["num_operands"]
            if op["op"] in [Op.SCATTER, Op.GATHER]:
                # SCATTER/GATHER add an indices input tensor in their build functions
                num_operands += 1
            if len(input_list) != num_operands:
                error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs,
        }
        return info_dict

    @staticmethod
    def evWrongOutputList(check=False, **kwargs):
        error_name = ErrorIf.WrongOutputList
        param_reqs = {"rank": None, "dtype": None, "shape": None}
        error_result = False
        error_reason = "Op output list does not match expected output"

        if check:
            output_list = kwargs["output_list"]
            # Note this will be incorrect if an operator returns more than one output
            if len(output_list) != 1:
                error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs,
        }
        return info_dict

    @staticmethod
    def evMaxDimExceeded(check=False, **kwargs):
        error_name = ErrorIf.MaxDimExceeded
        param_reqs = {
            "rank": [4, 4],
            "dtype": [DType.INT8],
            "shape": [[1, 16584, 5, 1], [1, 2, 16499, 4]],
        }
        error_result = False
        error_reason = (
            "At least one maximum dimension is greater than or equal to 16384"
        )

        if check:
            input_shape = kwargs["input_shape"]
            output_shape = kwargs["output_shape"]  # Note this is just (OH, OW)
            if (
                (input_shape[1] >= 16384)
                or (input_shape[2] >= 16384)
                or (output_shape[0] >= 16384)
                or (output_shape[1] >= 16384)
            ):
                error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs,
        }
        return info_dict

    @staticmethod
    def evBatchMismatch(check=False, **kwargs):
        error_name = ErrorIf.BatchMismatch
        param_reqs = {"rank": [4, 4], "dtype": None, "shape": None}
        error_result = False
        error_reason = "Input batch size not equal to output batch size"

        assert "op" in kwargs
        op = kwargs["op"]
        rmin, rmax = op["rank"]
        rank_range = range(rmin, rmax + 1)

        if check:
            input_shape = kwargs["input_shape"]
            output_shape = kwargs[
                "result_tensor"
            ].shape  # Note this is just (N, OH, OW, C)

            if (len(input_shape) in rank_range) and (input_shape[0] != output_shape[0]):
                error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs,
        }
        return info_dict

    @staticmethod
    def evChannelMismatch(check=False, **kwargs):
        error_name = ErrorIf.ChannelMismatch
        param_reqs = {"rank": [4, 4], "dtype": None, "shape": None}
        error_result = False
        error_reason = "Input channel size not equal to output channel size"

        assert "op" in kwargs
        op = kwargs["op"]
        rmin, rmax = op["rank"]
        rank_range = range(rmin, rmax + 1)

        if check:
            input_shape = kwargs["input_shape"]
            output_shape = kwargs[
                "result_tensor"
            ].shape  # Note this is just (N, OH, OW, C)
            if (len(input_shape) in rank_range) and (input_shape[3] != output_shape[3]):
                error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs,
        }
        return info_dict

    @staticmethod
    def evStrideSmallerEqualZero(check=False, **kwargs):
        error_name = ErrorIf.StrideSmallerEqualZero
        param_reqs = {"rank": None, "dtype": None, "shape": None}
        error_result = False
        error_reason = "Stride value smaller than or equal zero"

        if check:
            input_dtype = kwargs["input_dtype"]
            output_dtype = kwargs["output_dtype"]
            if input_dtype != DType.FLOAT and output_dtype == DType.FLOAT:
                stride = kwargs["stride"]  # Work around wrong input/output type tests
            elif output_dtype == DType.FLOAT:
                stride = kwargs["stride_fp"]
            elif input_dtype == DType.FLOAT and output_dtype != DType.FLOAT:
                stride = kwargs[
                    "stride_fp"
                ]  # Work around wrong input/output type tests
            else:
                stride = kwargs["stride"]

            if min(stride) <= 0:
                error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs,
        }
        return info_dict

    @staticmethod
    def evStrideLargerEqualMax(check=False, **kwargs):
        error_name = ErrorIf.StrideLargerEqualMax
        param_reqs = {"rank": None, "dtype": [DType.INT8, DType.INT16], "shape": None}
        error_result = False
        error_reason = "Stride value larger than or equal to maximum value"

        if check:
            shift = kwargs["shift"]
            input_dtype = kwargs["input_dtype"]
            stride = kwargs["stride"]
            if input_dtype in [DType.INT8, DType.INT16]:
                if shift >= 0 and (
                    stride[0] >= (16 << shift) or stride[1] >= (16 << shift)
                ):
                    error_result = True
                elif shift < 0 and (
                    stride[0] >= (16 >> -shift) or stride[1] >= (16 >> -shift)
                ):
                    error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs,
        }
        return info_dict

    @staticmethod
    def evStrideLargerDimension(check=False, **kwargs):
        error_name = ErrorIf.StrideLargerDimension
        param_reqs = {"rank": None, "dtype": [DType.FLOAT], "shape": None}
        error_result = False
        error_reason = "Stride value larger than or equal to H/W dimension"

        if check:
            shape = kwargs["input_shape"]
            input_dtype = kwargs["input_dtype"]
            stride = kwargs["stride_fp"]

            if (
                input_dtype == DType.FLOAT
                and (stride[0] > shape[1])
                or (stride[1] > shape[2])
            ):
                error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs,
        }
        return info_dict

    @staticmethod
    def evOffsetSmallerEqualMin(check=False, **kwargs):
        error_name = ErrorIf.OffsetSmallerEqualMin
        param_reqs = {"rank": None, "dtype": [DType.INT8, DType.INT16], "shape": None}
        error_result = False
        error_reason = "Offset value smaller than or equal to minimum value"

        if check:
            shift = kwargs["shift"]
            output_dtype = kwargs["output_dtype"]
            if output_dtype == DType.FLOAT:
                offset = kwargs["offset_fp"]
            else:
                offset = kwargs["offset"]

            if shift >= 0 and (
                offset[0] <= (-16 << shift) or offset[1] <= (-16 << shift)
            ):
                error_result = True
            elif shift < 0 and (
                offset[0] <= (-16 >> -shift) or offset[1] <= (-16 >> -shift)
            ):
                error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs,
        }
        return info_dict

    @staticmethod
    def evOffsetLargerEqualMax(check=False, **kwargs):
        error_name = ErrorIf.OffsetLargerEqualMax
        param_reqs = {"rank": None, "dtype": [DType.INT8, DType.INT16], "shape": None}
        error_result = False
        error_reason = "Offset value larger than or equal to maximum value"

        if check:
            shift = kwargs["shift"]
            output_dtype = kwargs["output_dtype"]
            if output_dtype == DType.FLOAT:
                offset = kwargs["offset_fp"]
            else:
                offset = kwargs["offset"]

            if shift >= 0:
                if offset[0] >= (16 << shift) or offset[1] >= (16 << shift):
                    error_result = True

            if shift >= 0 and (
                offset[0] >= (16 << shift) or offset[1] >= (16 << shift)
            ):
                error_result = True
            elif shift < 0 and (
                offset[0] >= (16 >> -shift) or offset[1] >= (16 >> -shift)
            ):
                error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs,
        }
        return info_dict

    @staticmethod
    def evShiftNotZero(check=False, **kwargs):
        error_name = ErrorIf.ShiftNotZero
        param_reqs = {"rank": None, "dtype": [DType.FLOAT], "shape": None}
        error_result = False
        error_reason = "Shift value must be zero for float input"

        if check:
            shift = kwargs["shift"]
            input_dtype = kwargs["input_dtype"]
            output_dtype = kwargs["output_dtype"]
            if (
                input_dtype == DType.FLOAT
                and output_dtype == DType.FLOAT
                and shift != 0
            ):
                error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs,
        }
        return info_dict

    @staticmethod
    def evShiftSmallerOne(check=False, **kwargs):
        error_name = ErrorIf.ShiftSmallerOne
        param_reqs = {"rank": None, "dtype": [DType.INT8, DType.INT16], "shape": None}
        error_result = False
        error_reason = "Shift value smaller than one"

        if check:
            shift = kwargs["shift"]
            input_dtype = kwargs["input_dtype"]
            output_dtype = kwargs["output_dtype"]
            if shift < 1 and input_dtype != DType.FLOAT and output_dtype != DType.FLOAT:
                error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs,
        }
        return info_dict

    @staticmethod
    def evShiftLargerEleven(check=False, **kwargs):
        error_name = ErrorIf.ShiftLargerEleven
        param_reqs = {"rank": None, "dtype": [DType.INT8, DType.INT16], "shape": None}
        error_result = False
        error_reason = "Shift value larger than eleven"

        if check:
            shift = kwargs["shift"]
            if shift > 11:
                error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs,
        }
        return info_dict

    @staticmethod
    def evRankMismatch(check=False, **kwargs):
        error_name = ErrorIf.RankMismatch
        param_reqs = {"rank": None, "dtype": None, "shape": None}
        error_result = False
        error_reason = "Input Rank does not match output rank"

        if check:
            input1_shape = kwargs["input1"].shape
            input2_shape = kwargs["input2"].shape
            # In case of SELECT op
            input3_shape = (
                kwargs["input3"].shape if "input3" in kwargs else input2_shape
            )
            output_shape = kwargs["result_tensor"].shape
            if (
                (len(input1_shape) != len(output_shape))
                or (len(input2_shape) != len(output_shape))
                or (len(input3_shape) != len(output_shape))
            ):
                error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs,
        }
        return info_dict

    @staticmethod
    def evDimensionMismatch(check=False, **kwargs):
        error_name = ErrorIf.DimensionMismatch
        param_reqs = {"rank": None, "dtype": None, "shape": None}
        error_result = False
        error_reason = "Input Dimensions do not match output"

        if check:
            input1_shape = kwargs["input1"].shape
            input2_shape = kwargs["input2"].shape
            # In case of SELECT op
            input3_shape = (
                kwargs["input3"].shape if "input3" in kwargs else input2_shape
            )
            output_shape = kwargs["result_tensor"].shape
            for i in range(
                min(len(input1_shape), len(input2_shape), len(input3_shape))
            ):
                if (
                    (input1_shape[i] != 1 and input1_shape[i] != output_shape[i])
                    or (input2_shape[i] != 1 and input2_shape[i] != output_shape[i])
                    or (input3_shape[i] != 1 and input3_shape[i] != output_shape[i])
                ):
                    error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs,
        }
        return info_dict

    @staticmethod
    def _getZeroPoint(qinfo, index):
        """Return zero point value from quantization info.

        Generally input_zp is index 0, output_zp is index 1
        """
        return qinfo[index]

    @staticmethod
    def evInputZeroPointNotZero(check=False, **kwargs):
        op = kwargs["op"]
        error_result = False

        # Quantizable types
        qTypes = (DType.INT8, DType.UINT8, DType.UINT16)

        # This does not apply to quantizable types
        inputDtypes = [
            dtype
            for dtype in op["types"]
            if (isinstance(dtype, list) and dtype[0] not in qTypes)
            or (not isinstance(dtype, list) and dtype not in qTypes)
        ]

        if check:
            input_dtype = kwargs["input_dtype"]
            input_zero_point = TosaErrorValidator._getZeroPoint(kwargs["qinfo"], 0)
            if op["op"] == Op.MATMUL:
                input2_zero_point = TosaErrorValidator._getZeroPoint(kwargs["qinfo"], 1)
                for dtype, zp in (
                    (kwargs["input_dtype"], input_zero_point),
                    (kwargs["input2_dtype"], input2_zero_point),
                ):
                    if dtype not in qTypes and zp != 0:
                        error_result = True
                        break
            else:
                error_result = input_dtype not in qTypes and input_zero_point != 0

        info_dict = {
            "error_name": ErrorIf.InputZeroPointNotZero,
            "error_result": error_result,
            "error_reason": "Input DType not INT8 and zero point not 0",
            "param_reqs": {"rank": None, "dtype": inputDtypes, "shape": None},
        }
        return info_dict

    @staticmethod
    def evWeightZeroPointNotZero(check=False, **kwargs):
        op = kwargs["op"]

        # exclude inputs with INT8 weights
        inputDtypes = [
            t for t in op["types"] if not isinstance(t, list) or t[1] != DType.INT8
        ]

        error_name = ErrorIf.WeightZeroPointNotZero
        param_reqs = {"rank": None, "dtype": inputDtypes, "shape": None}
        error_result = False
        error_reason = "Weight DType not INT8 and zero point not 0"

        if check:
            weight_dtype = kwargs["weight_dtype"]
            weight_zero_point = TosaErrorValidator._getZeroPoint(kwargs["qinfo"], 1)
            if weight_dtype != DType.INT8 and weight_zero_point != 0:
                error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs,
        }
        return info_dict

    @staticmethod
    def evOutputZeroPointNotZero(check=False, **kwargs):
        op = kwargs["op"]
        inputDtypes = [
            t for t in op["types"] if t not in [DType.INT8, DType.UINT8, DType.UINT16]
        ]

        error_name = ErrorIf.OutputZeroPointNotZero
        param_reqs = {"rank": None, "dtype": inputDtypes, "shape": None}
        error_result = False
        error_reason = "Output DType not INT8 and zero point not 0"

        if check:
            input_dtype = kwargs["input_dtype"]
            output_dtype = kwargs["output_dtype"]
            output_zero_point = TosaErrorValidator._getZeroPoint(kwargs["qinfo"], 1)
            if op["op"] == Op.AVG_POOL2D:
                if input_dtype != DType.INT8 and output_zero_point != 0:
                    error_result = True
            elif (
                output_dtype not in [DType.INT8, DType.UINT8, DType.UINT16]
                and output_zero_point != 0
            ):
                error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs,
        }
        return info_dict

    @staticmethod
    def evU16InputZeroPointNotValid(check=False, **kwargs):
        error_name = ErrorIf.U16InputZeroPointNotValid
        param_reqs = {"rank": None, "dtype": [DType.UINT16], "shape": None}
        error_result = False
        error_reason = "Input DType is UINT16 and zero point not 0 or 32678"

        if check:
            input_dtype = kwargs["input_dtype"]
            input_zero_point = TosaErrorValidator._getZeroPoint(kwargs["qinfo"], 0)
            error_result = input_dtype == DType.UINT16 and input_zero_point not in [
                0,
                32768,
            ]

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs,
        }
        return info_dict

    @staticmethod
    def evU16OutputZeroPointNotValid(check=False, **kwargs):
        error_name = ErrorIf.U16OutputZeroPointNotValid
        param_reqs = {"rank": None, "dtype": None, "shape": None}
        error_result = False
        error_reason = "Output DType is UINT16 and zero point not 0 or 32678"

        if check:
            output_dtype = kwargs["output_dtype"]
            output_zero_point = TosaErrorValidator._getZeroPoint(kwargs["qinfo"], 1)

            error_result = output_dtype == DType.UINT16 and output_zero_point not in [
                0,
                32768,
            ]

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs,
        }
        return info_dict

    @staticmethod
    def evAxisSmallerZero(check=False, **kwargs):
        error_name = ErrorIf.AxisSmallerZero
        param_reqs = {"rank": None, "dtype": None, "shape": None}
        error_result = False
        error_reason = "Axis smaller than zero"

        if check:
            axis = kwargs["axis"]
            if axis < 0:
                error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs,
        }
        return info_dict

    @staticmethod
    def evAxisLargerRank(check=False, **kwargs):
        error_name = ErrorIf.AxisLargerRank
        param_reqs = {"rank": None, "dtype": None, "shape": None}
        error_result = False
        error_reason = "Axis larger than rank"

        if check:
            axis = kwargs["axis"]
            shape = kwargs["input_shape"]
            if axis > len(shape):
                error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs,
        }
        return info_dict

    @staticmethod
    def evShapeOfAxisNotOne(check=False, **kwargs):
        error_name = ErrorIf.ShapeOfAxisNotOne
        param_reqs = {"rank": None, "dtype": None, "shape": None}
        error_result = False
        error_reason = "shape[axis] is not equal to 1"

        if check:
            axis = kwargs["axis"]
            shape = kwargs["output_shape"]
            if (0 <= axis < len(shape)) and shape[axis] != 1:
                error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs,
        }
        return info_dict

    @staticmethod
    def evPadSmallerZero(check=False, **kwargs):
        error_name = ErrorIf.PadSmallerZero
        param_reqs = {"rank": None, "dtype": None, "shape": None}
        error_result = False
        error_reason = "At least one pad is smaller than zero"

        if check:
            op = kwargs["op"]
            pad = kwargs["pad"]
            if op["op"] == Op.PAD:
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
            "param_reqs": param_reqs,
        }
        return info_dict

    @staticmethod
    def evPadLargerEqualKernel(check=False, **kwargs):
        error_name = ErrorIf.PadLargerEqualKernel
        param_reqs = {"rank": None, "dtype": None, "shape": None}
        error_result = False
        error_reason = "At least one pad is larger than kernel dimension"

        if check:
            pad = kwargs["pad"]
            kernel = kwargs["kernel"]
            if min(pad) > 0 and min(kernel) > 1:
                if (
                    pad[0] >= kernel[0]
                    or pad[1] >= kernel[0]
                    or pad[2] >= kernel[1]
                    or pad[3] >= kernel[1]
                ):
                    error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs,
        }
        return info_dict

    @staticmethod
    def checkPoolingParams(kernel, stride, pad):
        return (
            min(kernel) >= 1
            and min(stride) >= 1
            and min(pad) >= 0
            and not (
                pad[0] >= kernel[0]
                or pad[1] >= kernel[0]
                or pad[2] >= kernel[1]
                or pad[3] >= kernel[1]
            )
        )

    @staticmethod
    def evPoolingOutputShapeMismatch(check=False, **kwargs):
        error_name = ErrorIf.PoolingOutputShapeMismatch
        param_reqs = {"rank": None, "dtype": None, "shape": None}
        error_result = False
        error_reason = (
            "Mismatch between output shape provided and expected output shape"
        )

        if check:
            pad = kwargs["pad"]
            pad_top, pad_bottom, pad_left, pad_right = pad[0], pad[1], pad[2], pad[3]

            kernel = kwargs["kernel"]
            kernel_y, kernel_x = kernel[0], kernel[1]

            input_shape = kwargs["input_shape"]
            IH, IW = input_shape[1], input_shape[2]

            output_shape = kwargs["output_shape"]
            OH, OW = output_shape[1], output_shape[2]

            stride = kwargs["stride"]
            stride_y, stride_x = stride[0], stride[1]

            # calculate correct height, width dimensions
            if stride_x != 0 and stride_y != 0:
                y_correct = ((IH + pad_top + pad_bottom - kernel_y) // stride_y) + 1
                x_correct = ((IW + pad_left + pad_right - kernel_x) // stride_x) + 1

            # ensure parameters are valid
            params_valid = TosaErrorValidator.checkPoolingParams(kernel, stride, pad)

            if params_valid and (OH != y_correct or OW != x_correct):
                error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs,
        }
        return info_dict

    @staticmethod
    def evPoolingOutputShapeNonInteger(check=False, **kwargs):
        error_name = ErrorIf.PoolingOutputShapeNonInteger
        param_reqs = {"rank": None, "dtype": None, "shape": None}
        error_result = False
        error_reason = "Parameters do not yield exact integer output dimensions"

        if check:
            pad = kwargs["pad"]
            pad_top, pad_bottom, pad_left, pad_right = pad[0], pad[1], pad[2], pad[3]

            kernel = kwargs["kernel"]
            kernel_y, kernel_x = kernel[0], kernel[1]

            input_shape = kwargs["input_shape"]
            IH, IW = input_shape[1], input_shape[2]

            stride = kwargs["stride"]
            stride_y, stride_x = stride[0], stride[1]

            # calculate remainder of height, width dimensions
            if stride_x != 0 and stride_y != 0:
                y_remainder = (IH + pad_top + pad_bottom - kernel_y) % stride_y
                x_remainder = (IW + pad_left + pad_right - kernel_x) % stride_x

            # ensure parameters are valid
            params_valid = TosaErrorValidator.checkPoolingParams(kernel, stride, pad)
            if params_valid and (y_remainder != 0 or x_remainder != 0):
                error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs,
        }
        return info_dict

    @staticmethod
    def checkConvParams(weight_shape, stride, pad, dilation):
        return (
            # Check kernel sizes
            min(weight_shape[1:-1]) >= 1
            and min(stride) >= 1
            and min(pad) >= 0
            and (dilation is None or min(dilation) >= 1)
        )

    @staticmethod
    def evConvOutputShapeMismatch(check=False, **kwargs):
        error_name = ErrorIf.ConvOutputShapeMismatch
        param_reqs = {"rank": None, "dtype": None, "shape": None}
        error_result = False
        error_reason = (
            "Mismatch between output shape provided and expected output shape"
        )

        if check:
            op = kwargs["op"]
            pad = kwargs["pad"]
            weight_shape = kwargs["weight_shape"]
            input_shape = kwargs["input_shape"]
            output_shape = kwargs["output_shape"]
            dilation = kwargs["dilation"] if op["op"] != Op.TRANSPOSE_CONV2D else None
            stride = kwargs["stride"]

            kernel_offset = 0 if op["op"] == Op.DEPTHWISE_CONV2D else 1

            # calculate correct dimensions
            dims_correct = []
            if min(stride) > 0:
                for index in range(len(stride)):
                    pad_offset = index * 2
                    if op["op"] == Op.TRANSPOSE_CONV2D:
                        dims_correct.append(
                            (input_shape[index + 1] - 1) * stride[index]
                            - pad[pad_offset]
                            - pad[pad_offset + 1]
                            + weight_shape[index + kernel_offset]
                        )
                    else:
                        dims_correct.append(
                            (
                                input_shape[index + 1]
                                - 1
                                + pad[pad_offset]
                                + pad[pad_offset + 1]
                                - (weight_shape[index + kernel_offset] - 1)
                                * dilation[index]
                            )
                            // stride[index]
                            + 1
                        )

            # ensure parameters are valid
            params_valid = TosaErrorValidator.checkConvParams(
                weight_shape, stride, pad, dilation
            )

            if params_valid and output_shape[1:-1] != dims_correct:
                error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs,
        }
        return info_dict

    @staticmethod
    def evConvOutputShapeNonInteger(check=False, **kwargs):
        error_name = ErrorIf.ConvOutputShapeNonInteger
        param_reqs = {"rank": None, "dtype": None, "shape": None}
        error_result = False
        error_reason = "Parameters do not yield exact integer output dimensions"

        if check:
            op = kwargs["op"]
            pad = kwargs["pad"]
            weight_shape = kwargs["weight_shape"]
            input_shape = kwargs["input_shape"]
            dilation = kwargs["dilation"]
            stride = kwargs["stride"]

            kernel_offset = 0 if op["op"] == Op.DEPTHWISE_CONV2D else 1

            # calculate correct height, width dimensions
            remainders = []
            if min(stride) > 0:
                for index in range(len(stride)):
                    pad_offset = index * 2
                    remainders.append(
                        (
                            input_shape[index + 1]
                            - 1
                            + pad[pad_offset]
                            + pad[pad_offset + 1]
                            - (weight_shape[index + kernel_offset] - 1)
                            * dilation[index]
                        )
                        % stride[index]
                    )

            # ensure parameters are valid
            params_valid = TosaErrorValidator.checkConvParams(
                weight_shape, stride, pad, dilation
            )
            if params_valid and max(remainders) > 0:
                error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs,
        }
        return info_dict

    @staticmethod
    def evArgmaxOutputShapeMismatch(check=False, **kwargs):
        error_name = ErrorIf.ArgmaxOutputShapeMismatch
        param_reqs = {"rank": [2, 4], "dtype": None, "shape": None}
        error_result = False
        error_reason = (
            "Mismatch between output shape provided and expected output shape"
        )

        if check:
            output_shape = kwargs["output_shape"]
            input_shape = kwargs["input_shape"]
            axis = kwargs["axis"]

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
            "param_reqs": param_reqs,
        }
        return info_dict

    @staticmethod
    def evArgmaxOutputRankMismatch(check=False, **kwargs):
        error_name = ErrorIf.ArgmaxOutputRankMismatch
        param_reqs = {"rank": None, "dtype": None, "shape": None}
        error_result = False
        error_reason = (
            "Mismatch between output shape provided and expected output shape"
        )

        if check:
            output_shape = kwargs["output_shape"]
            input_shape = kwargs["input_shape"]
            axis = kwargs["axis"]
            valid_params = axis >= 0 and axis < len(input_shape)

            if valid_params and (len(input_shape) - 1) != len(output_shape):
                error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs,
        }
        return info_dict

    @staticmethod
    def evKernelSmallerOne(check=False, **kwargs):
        error_name = ErrorIf.KernelSmallerOne
        param_reqs = {"rank": None, "dtype": None, "shape": None}
        error_result = False
        error_reason = "At least one kernel dimension is smaller than zero"

        if check:
            kernel = kwargs["kernel"]
            if min(kernel) < 1:
                error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs,
        }
        return info_dict

    @staticmethod
    def evStrideSmallerOne(check=False, **kwargs):
        error_name = ErrorIf.StrideSmallerOne
        param_reqs = {"rank": None, "dtype": None, "shape": None}
        error_result = False
        error_reason = "At least one stride dimension is smaller than zero"

        if check:
            stride = kwargs["stride"]
            if min(stride) < 1:
                error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs,
        }
        return info_dict

    @staticmethod
    def evDilationSmallerOne(check=False, **kwargs):
        error_result = check and min(kwargs["dilation"]) < 1
        return {
            "error_name": ErrorIf.DilationSmallerOne,
            "error_reason": "At least one dilation is smaller than one",
            "param_reqs": {"rank": None, "dtype": None, "shape": None},
            "error_result": error_result,
        }

    @staticmethod
    def evScaleTrue(check=False, **kwargs):
        error_name = ErrorIf.ScaleTrue
        param_reqs = {"rank": None, "dtype": [DType.INT48], "shape": None}
        error_result = False
        error_reason = "Scale set to true but input type is INT48"

        if check:
            input_dtype = kwargs["input_dtype"]
            scale32 = kwargs["scale32"]
            if scale32 and input_dtype == DType.INT48:
                error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs,
        }
        return info_dict

    @staticmethod
    def evScaleNotTrue(check=False, **kwargs):
        error_name = ErrorIf.ScaleNotTrue
        param_reqs = {"rank": None, "dtype": None, "shape": None}
        error_result = False
        error_reason = "Scale set to false but double round set to true"

        if check:
            scale32 = kwargs["scale32"]
            double_round = kwargs["double_round"]
            if not scale32 and double_round:
                error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs,
        }
        return info_dict

    @staticmethod
    def evTensorSizeInputOutputMismatch(check=False, **kwargs):
        error_name = ErrorIf.TensorSizeInputOutputMismatch
        param_reqs = {"rank": None, "dtype": None, "shape": None}
        error_result = False
        error_reason = "Input tensor size does not match output tensor size"

        if check:
            input_shape = kwargs["input_shape"]
            output_shape = kwargs["output_shape"]
            input_size = np.prod(input_shape)
            output_size = np.prod(output_shape)
            if input_size != output_size:
                error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs,
        }
        return info_dict

    @staticmethod
    def evStartSmallerZero(check=False, **kwargs):
        error_name = ErrorIf.StartSmallerZero
        param_reqs = {"rank": None, "dtype": None, "shape": None}
        error_result = False
        error_reason = "Starting point smaller than zero"

        if check:
            input_shape = kwargs["input_shape"]
            start = kwargs["start"]
            rank = len(input_shape)
            if len(start) == rank:
                for index in range(rank):
                    if start[index] < 0:
                        error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs,
        }
        return info_dict

    @staticmethod
    def evSizeSmallerEqualZero(check=False, **kwargs):
        error_name = ErrorIf.SizeSmallerEqualZero
        param_reqs = {"rank": None, "dtype": None, "shape": None}
        error_result = False
        error_reason = "Size smaller than or equal to zero"

        if check:
            input_shape = kwargs["input_shape"]
            size = kwargs["size"]
            rank = len(input_shape)
            if len(size) == rank:
                for index in range(rank):
                    if size[index] <= 0:
                        error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs,
        }
        return info_dict

    @staticmethod
    def evStartSizeOutsideBounds(check=False, **kwargs):
        error_name = ErrorIf.StartSizeOutsideBounds
        param_reqs = {"rank": None, "dtype": None, "shape": None}
        error_result = False
        error_reason = "starting point plus size larger than input dimension"

        if check:
            input_shape = kwargs["input_shape"]
            start = kwargs["start"]
            size = kwargs["size"]
            rank = len(input_shape)
            if len(start) == rank and len(size) == rank:
                for index in range(rank):
                    if start[index] + size[index] > input_shape[index]:
                        error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs,
        }
        return info_dict

    @staticmethod
    def evSizeOutputShapeMismatch(check=False, **kwargs):
        error_name = ErrorIf.SizeOutputShapeMismatch
        param_reqs = {"rank": None, "dtype": None, "shape": None}
        error_result = False
        error_reason = "Size does not match output dimension"

        if check:
            input_shape = kwargs["input_shape"]
            output_shape = kwargs["output_shape"]
            size = kwargs["size"]
            rank = len(input_shape)
            if len(size) == rank:
                for index in range(rank):
                    if size[index] != output_shape[index]:
                        error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs,
        }
        return info_dict

    @staticmethod
    def evInputSizeStartLengthMismatch(check=False, **kwargs):
        error_name = ErrorIf.InputSizeStartLengthMismatch
        param_reqs = {"rank": None, "dtype": None, "shape": None}
        error_result = False
        error_reason = "rank of input not equal to length of start or size"

        if check:
            input_shape = kwargs["input_shape"]
            start = kwargs["start"]
            size = kwargs["size"]
            rank = len(input_shape)
            if rank != len(start) or rank != len(size):
                error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs,
        }
        return info_dict

    @staticmethod
    def evIndexOutsideBounds(check=False, **kwargs):
        error_name = ErrorIf.IndexOutsideBounds
        param_reqs = {"rank": None, "dtype": None, "shape": None}
        error_result = False
        error_reason = "Index outside of allowed bounds"

        if check:
            input_shape = kwargs["input_shape"]
            perms = kwargs["perms"]
            rank = len(input_shape)

            for index in perms:
                if index < 0 or index > rank:
                    error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs,
        }
        return info_dict

    @staticmethod
    def evIndexUsedTwice(check=False, **kwargs):
        error_name = ErrorIf.IndexUsedTwice
        param_reqs = {"rank": [2, 4], "dtype": None, "shape": None}
        error_result = False
        error_reason = "Index used multiple times"

        if check:
            perms = kwargs["perms"]

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
            "param_reqs": param_reqs,
        }
        return info_dict

    @staticmethod
    def evMaxSmallerMin(check=False, **kwargs):
        error_name = ErrorIf.MaxSmallerMin
        param_reqs = {"rank": [2, 4], "dtype": None, "shape": None}
        error_result = False
        error_reason = "Max value smaller than min value"

        if check:
            max_val = kwargs["max_val"]
            min_val = kwargs["min_val"]
            if max_val < min_val:
                error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs,
        }
        return info_dict

    @staticmethod
    def evConcatInputRankMismatch(check=False, **kwargs):
        error_name = ErrorIf.ConcatInputRankMismatch
        param_reqs = {"rank": [2, 4], "dtype": None, "shape": None}
        error_result = False
        error_reason = "Input ranks are not identical"

        if check:
            inputs = kwargs["inputs"]
            input_shape = kwargs["input_shape"]
            for input in inputs:
                if len(input.shape) != len(input_shape):
                    error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs,
        }
        return info_dict

    @staticmethod
    def evConcatInputDimMismatch(check=False, **kwargs):
        error_name = ErrorIf.ConcatInputDimMismatch
        param_reqs = {"rank": [2, 4], "dtype": None, "shape": None}
        error_result = False
        error_reason = "Input dimensions differ on too many axes"

        if check:
            inputs = kwargs["inputs"]
            input_shape = kwargs["input_shape"]
            axis = kwargs["axis"]

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
            "param_reqs": param_reqs,
        }
        return info_dict

    @staticmethod
    def evConcatShapeSumMismatch(check=False, **kwargs):
        error_name = ErrorIf.ConcatShapeSumMismatch
        param_reqs = {"rank": [2, 4], "dtype": None, "shape": None}
        error_result = False
        error_reason = "Sum of dimensions on axis not equal to output dimension"

        if check:
            inputs = kwargs["inputs"]
            input_shape = kwargs["input_shape"]
            output_shape = kwargs["output_shape"]
            axis = kwargs["axis"]

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
            "param_reqs": param_reqs,
        }
        return info_dict

    @staticmethod
    def evInputListThenGraphMismatch(check=False, **kwargs):
        error_name = ErrorIf.CondIfInputListThenGraphMismatch
        param_reqs = {"rank": None, "dtype": None, "shape": None}
        error_result = False
        error_reason = "Input list shape does not match then-graph shape"

        if check:
            a = kwargs["a"]
            b = kwargs["b"]
            basicBlocks = kwargs["basicBlocks"]
            then_block = basicBlocks[1]
            then_inputs = then_block.inputs
            then_tens = then_block.tensors
            if (a.shape != then_tens[then_inputs[0]].shape) or (
                b.shape != then_tens[then_inputs[1]].shape
            ):
                error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs,
        }
        return info_dict

    @staticmethod
    def evInputListElseGraphMismatch(check=False, **kwargs):
        error_name = ErrorIf.CondIfInputListElseGraphMismatch
        param_reqs = {"rank": None, "dtype": None, "shape": None}
        error_result = False
        error_reason = "Input list shape does not match else-graph shape"

        if check:
            a = kwargs["a"]
            b = kwargs["b"]
            basicBlocks = kwargs["basicBlocks"]
            else_block = basicBlocks[2]
            else_inputs = else_block.inputs
            else_tens = else_block.tensors
            if (a.shape != else_tens[else_inputs[0]].shape) or (
                b.shape != else_tens[else_inputs[1]].shape
            ):
                error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs,
        }
        return info_dict

    @staticmethod
    def evOutputListThenGraphMismatch(check=False, **kwargs):
        error_name = ErrorIf.CondIfOutputListThenGraphMismatch
        param_reqs = {"rank": None, "dtype": None, "shape": None}
        error_result = False
        error_reason = "Output list shape does not match then-graph shape"

        if check:
            basicBlocks = kwargs["basicBlocks"]
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
            "param_reqs": param_reqs,
        }
        return info_dict

    @staticmethod
    def evOutputListElseGraphMismatch(check=False, **kwargs):
        error_name = ErrorIf.CondIfOutputListElseGraphMismatch
        param_reqs = {"rank": None, "dtype": None, "shape": None}
        error_result = False
        error_reason = "Output list shape does not match else-graph shape"

        if check:
            basicBlocks = kwargs["basicBlocks"]
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
            "param_reqs": param_reqs,
        }
        return info_dict

    @staticmethod
    def evInputListOutputListMismatch(check=False, **kwargs):
        error_name = ErrorIf.InputListOutputListMismatch
        param_reqs = {"rank": None, "dtype": None, "shape": None}
        error_result = False
        error_reason = "Input list does not match output list"

        if check:
            basicBlocks = kwargs["basicBlocks"]
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
            "param_reqs": param_reqs,
        }
        return info_dict

    @staticmethod
    def evInputListCondGraphMismatch(check=False, **kwargs):
        error_name = ErrorIf.InputListCondGraphMismatch
        param_reqs = {"rank": None, "dtype": None, "shape": None}
        error_result = False
        error_reason = "Input list does not match cond graph"

        if check:
            basicBlocks = kwargs["basicBlocks"]
            while_block = basicBlocks[0]
            while_inputs = while_block.inputs
            while_tens = while_block.tensors
            cond_block = basicBlocks[1]
            cond_inputs = cond_block.inputs
            cond_tens = cond_block.tensors
            if (
                while_tens[while_inputs[0]].shape != cond_tens[cond_inputs[0]].shape
            ) or (while_tens[while_inputs[1]].shape != cond_tens[cond_inputs[2]].shape):
                error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs,
        }
        return info_dict

    @staticmethod
    def evInputListBodyGraphInputMismatch(check=False, **kwargs):
        error_name = ErrorIf.InputListBodyGraphInputMismatch
        param_reqs = {"rank": None, "dtype": None, "shape": None}
        error_result = False
        error_reason = "Input list does not match body graph input"

        if check:
            basicBlocks = kwargs["basicBlocks"]
            while_block = basicBlocks[0]
            while_inputs = while_block.inputs
            while_tens = while_block.tensors
            body_block = basicBlocks[2]
            body_outputs = body_block.inputs
            body_tens = body_block.tensors
            if (
                while_tens[while_inputs[0]].shape != body_tens[body_outputs[0]].shape
            ) or (
                while_tens[while_inputs[1]].shape != body_tens[body_outputs[2]].shape
            ):
                error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs,
        }
        return info_dict

    @staticmethod
    def evInputListBodyGraphOutputMismatch(check=False, **kwargs):
        error_name = ErrorIf.InputListBodyGraphOutputMismatch
        param_reqs = {"rank": None, "dtype": None, "shape": None}
        error_result = False
        error_reason = "Input list does not match body graph output"

        if check:
            basicBlocks = kwargs["basicBlocks"]
            while_block = basicBlocks[0]
            while_inputs = while_block.inputs
            while_tens = while_block.tensors
            body_block = basicBlocks[2]
            body_outputs = body_block.outputs
            body_tens = body_block.tensors
            if (
                while_tens[while_inputs[0]].shape != body_tens[body_outputs[0]].shape
            ) or (
                while_tens[while_inputs[1]].shape != body_tens[body_outputs[2]].shape
            ):
                error_result = True
        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs,
        }
        return info_dict

    @staticmethod
    def evCondGraphOutputNotMatchingBool(check=False, **kwargs):
        error_name = ErrorIf.CondGraphOutputNotMatchingBool
        param_reqs = {"rank": None, "dtype": None, "shape": None}
        error_result = False
        error_reason = "Cond graph output is not a match list of booleans"

        if check:
            basicBlocks = kwargs["basicBlocks"]
            cond_block = basicBlocks[1]
            cond_outputs = cond_block.outputs
            cond_tens = cond_block.tensors
            if cond_tens[cond_outputs[0]].dtype != DType.BOOL:
                error_result = True

        info_dict = {
            "error_name": error_name,
            "error_result": error_result,
            "error_reason": error_reason,
            "param_reqs": param_reqs,
        }
        return info_dict


class TosaInvalidValidator:
    @staticmethod
    def ivWrongDataTypeOrModeResize(**kwargs):
        input_dtype = kwargs["input_dtype"]
        args = kwargs["args"]
        mode = args[0]
        output_dtype = args[8]

        if mode == ResizeMode.BILINEAR:
            # Invalid output data type / Invalid input datatype
            return (
                not (input_dtype == DType.INT8 and output_dtype == DType.INT32)
                or not (input_dtype == DType.INT16 and output_dtype == DType.INT48)
                or not (input_dtype == DType.FLOAT and output_dtype == DType.FLOAT)
                or (input_dtype not in [DType.INT8, DType.INT32, DType.FLOAT])
            )
        elif mode == ResizeMode.NEAREST:
            # Invalid output data type / Invalid input datatype
            return (input_dtype != output_dtype) or (
                input_dtype not in [DType.INT8, DType.INT32, DType.FLOAT]
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
    def ivHeightWidthInvalid(**kwargs):
        opName = kwargs["opName"]

        inputShapes = kwargs["shapeList"]
        input_shape = inputShapes[0]

        args = kwargs["args"]
        strides = args[0]
        padding = args[1]

        if opName.endswith("pool2d"):
            # avg_pool2d, max_pool2d
            kernel_shape = args[2]
            h = (
                input_shape[1] + padding[0] + padding[1] + strides[0] - kernel_shape[0]
            ) // strides[0]
            w = (
                input_shape[2] + padding[2] + padding[3] + strides[1] - kernel_shape[1]
            ) // strides[1]
            # return True if any dimension is < 1
            return h < 1 or w < 1

        if opName.startswith("transpose_conv2d"):
            # transpose_conv2d
            output_shape = args[2]
            filter_shape = inputShapes[1]
            kernel_shape = filter_shape[1:-1]

            def get_out_size(in_size, stride, kernel_size, out_pad, in_pad):
                """Calculate the transpose_conv2d output size for a dimension.

                Args:
                    in_size: the input size - int
                    stride: the stride - int
                    kernel_size: the kernel size - int
                    out_pad: the output padding - int
                    in_pad: the input padding - int

                Returns:
                    the output size
                """
                return (in_size - 1) * stride + kernel_size - in_pad - out_pad

            for pad_h, pad_w in (
                (kernel_shape[0] - 1, kernel_shape[1] - 1),  # FULL padding
                (kernel_shape[0] // 2, kernel_shape[1] // 2),  # SAME padding
                (0, 0),  # VALID padding
            ):
                h = get_out_size(
                    input_shape[1],
                    strides[0],
                    kernel_shape[0],
                    padding[0],
                    pad_h,
                )
                w = get_out_size(
                    input_shape[2],
                    strides[1],
                    kernel_shape[1],
                    padding[1],
                    pad_w,
                )
                if output_shape[1] == h and output_shape[2] == w:
                    return False

            # output shape does not match the expected shape for any padding option
            return True

        if "conv2d" in opName or "conv3d" in opName:
            # conv2d, conv3d, depthwise_conv2d
            dilations = args[2]
            filter_shape = inputShapes[1]
            kernel_shape = (
                filter_shape[0:2]
                if opName.startswith("depthwise_conv2d")
                else filter_shape[1:-1]
            )

            for i in range(len(kernel_shape)):
                dim = (
                    input_shape[i + 1]
                    - kernel_shape[i]
                    - (kernel_shape[i] - 1) * (dilations[i] - 1)
                    + padding[i * 2 + 0]
                    + padding[i * 2 + 1]
                ) // strides[i] + 1
                # return True if any dimension is < 1
                if dim < 1:
                    return True
            return False

        assert False, f"Unrecognized Op: {opName}"

    @staticmethod
    def ivNonPositiveOutputShape(**kwargs):
        args = kwargs["args"]
        output_shape = args[2]
        if output_shape[1] <= 0 or output_shape[2] <= 0:
            # Negative output shape
            return True
        return False

# Copyright (c) 2021, ARM Limited.
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
    PadSmallerZero = "PadSmallerZero"
    PadLargerEqualKernel = "PadLargerEqualKernel"
    PoolingOutputShapeMismatch = "PoolingOutputShapeMismatch"
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



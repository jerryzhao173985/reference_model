// Copyright (c) 2024, ARM Limited.
//
//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at
//
//         http://www.apache.org/licenses/LICENSE-2.0
//
//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.

#include "generate_special.h"
#include "generate_special_utils.h"

namespace
{

const TestValues conditionalOpsTestVals{
    { SValue(SVE::Inf), SValue(SVE::Inf) },      { -SValue(SVE::Inf), -SValue(SVE::Inf) },
    { SValue(SVE::Inf), -SValue(SVE::Inf) },     { -SValue(SVE::Inf), SValue(SVE::Inf) },
    { -SValue(SVE::Zero), SValue(SVE::Zero) },   { SValue(SVE::Zero), -SValue(SVE::Zero) },
    { SValue(SVE::NaN), SValue(SVE::RndFloat) }, { SValue(SVE::RndFloat), SValue(SVE::NaN) },
    { SValue(SVE::NaN), SValue(SVE::Inf) },      { SValue(SVE::Inf), SValue(SVE::NaN) },
    { SValue(SVE::NaN), -SValue(SVE::Inf) },     { -SValue(SVE::Inf), SValue(SVE::NaN) },
    { SValue(SVE::NaN), SValue(SVE::NaN) },
};

const TestValues addTestVals{ { SValue(SVE::RndFloat, SVE::ULPMax, SVE::Max), SValue(SVE::Max) },
                              { -SValue(SVE::Max), -SValue(SVE::RndFloat, SVE::ULPMax, SVE::Max) },
                              { SValue(SVE::Inf), -SValue(SVE::Inf) },
                              { SValue(SVE::Inf), SValue(SVE::Inf) },
                              { -SValue(SVE::Inf), -SValue(SVE::Inf) },
                              { SValue(SVE::Inf), SValue(SVE::RndFloat) },
                              { SValue(SVE::RndFloat), -SValue(SVE::Inf) },
                              { SValue(SVE::NaN), SValue(SVE::RndFloat) },
                              { SValue(SVE::RndFloat), SValue(SVE::NaN) } };

const TestValues subTestVals{ { SValue(SVE::Max), -SValue(SVE::RndFloat, SVE::ULPMax, SVE::Max) },
                              { -SValue(SVE::Max), SValue(SVE::RndFloat, SVE::ULPMax, SVE::Max) },
                              { SValue(SVE::Inf), SValue(SVE::Inf) },
                              { -SValue(SVE::Inf), -SValue(SVE::Inf) },
                              { SValue(SVE::Inf), -SValue(SVE::Inf) },
                              { -SValue(SVE::Inf), SValue(SVE::Inf) },
                              { SValue(SVE::Inf), SValue(SVE::RndFloat) },
                              { -SValue(SVE::Inf), SValue(SVE::RndFloat) },
                              { SValue(SVE::RndFloat), SValue(SVE::Inf) },
                              { SValue(SVE::RndFloat), -SValue(SVE::Inf) },
                              { SValue(SVE::NaN), SValue(SVE::RndFloat) },
                              { SValue(SVE::RndFloat), SValue(SVE::NaN) } };

const TestValues mulTestVals{ { SValue(SVE::Max), SValue(SVE::RndFloat, SVE::Two, SVE::Max) },
                              { -SValue(SVE::Max), -SValue(SVE::RndFloat, SVE::Two, SVE::Max) },
                              { -SValue(SVE::Max), SValue(SVE::RndFloat, SVE::Two, SVE::Ten) },
                              { SValue(SVE::Max), -SValue(SVE::RndFloat, SVE::Two, SVE::Ten) },
                              { SValue(SVE::Inf), SValue(SVE::Zero) },
                              { -SValue(SVE::Inf), SValue(SVE::Zero) },
                              { SValue(SVE::Inf), -SValue(SVE::Zero) },
                              { -SValue(SVE::Inf), -SValue(SVE::Zero) },
                              { SValue(SVE::Inf), SValue(SVE::Inf) },
                              { -SValue(SVE::Inf), -SValue(SVE::Inf) },
                              { SValue(SVE::Inf), -SValue(SVE::Inf) },
                              { -SValue(SVE::Inf), SValue(SVE::Inf) },
                              { SValue(SVE::NaN), SValue(SVE::RndFloat) },
                              { SValue(SVE::RndFloat), SValue(SVE::NaN) } };

const TestValues powTestVals{ { -SValue(SVE::RndFloat, SVE::Min, SVE::Max), SValue(SVE::Euler) },
                              { -SValue(SVE::RndFloat, SVE::Min, SVE::Max), SValue(SVE::Pythagoras) },
                              { SValue(SVE::Max), SValue(SVE::RndFloat, SVE::Two, SVE::Max) },
                              { -SValue(SVE::Max), SValue(SVE::RndOddInteger, SVE::One, SVE::Ten) },
                              { -SValue(SVE::Max), SValue(SVE::RndEvenInteger, SVE::One, SVE::Ten) },
                              { SValue(SVE::Zero), SValue(SVE::One) },
                              { -SValue(SVE::Zero), SValue(SVE::One) },
                              { SValue(SVE::Zero), SValue(SVE::Two) },
                              { -SValue(SVE::Zero), SValue(SVE::Two) },
                              /* TODO: Missing infinity tests - need spec clarification */
                              { SValue(SVE::NaN), SValue(SVE::RndFloat) },
                              { SValue(SVE::RndFloat), SValue(SVE::NaN) } };

const TestValues minMaxTestVals{ { SValue(SVE::Zero), -SValue(SVE::Zero) },
                                 { SValue(SVE::Inf), -SValue(SVE::Inf) },
                                 { SValue(SVE::Min), -SValue(SVE::Min) },
                                 { SValue(SVE::Max), -SValue(SVE::Max) },
                                 /* TODO: Add denorm numbers - need spec clarification */
                                 { SValue(SVE::RndFloat), SValue(SVE::NaN) },
                                 { SValue(SVE::NaN), -SValue(SVE::RndFloat) } };

const TestValues castTestVals{
    { SValue(SVE::Zero) },
    { -SValue(SVE::Zero) },
    { SValue(SVE::Inf) },
    { -SValue(SVE::Inf) },
    { SValue(SVE::Min) },
    { -SValue(SVE::Min) },
    { SValue(SVE::Max) },
    { -SValue(SVE::Max) },
    { SValue(SVE::NaN) },
    // Values for testing overflows. We add one for each possible target type because we don't
    // really know the output type in the generator library
    { SValue(SVE::AboveMaxFP8E4M3) },
    { SValue(SVE::AboveMaxFP8E5M2) },
    { SValue(SVE::AboveMaxBF16) },
    { SValue(SVE::AboveMaxFP16) },
    { SValue(SVE::AboveMaxFP32) },
    { -SValue(SVE::AboveMaxFP8E4M3) },
    { -SValue(SVE::AboveMaxFP8E5M2) },
    { -SValue(SVE::AboveMaxBF16) },
    { -SValue(SVE::AboveMaxFP16) },
    { -SValue(SVE::AboveMaxFP32) },
    // TODO: Testing of cast underflows as part of a general improvement to underflow/overflow
    // analysis in the verification library.
};

const TestValues dotProductTestVals{
    { SValue(SVE::Zero), -SValue(SVE::Zero), SValue(SVE::Zero) },
    { SValue(SVE::Inf), -SValue(SVE::Inf), SValue(SVE::One) },
    { SValue(SVE::NaN), SValue(SVE::One), SValue(SVE::One) },
    { SValue(SVE::Min), SValue(SVE::Min), SValue(SVE::Zero) },
    { SValue(SVE::One), SValue(SVE::Min), SValue(SVE::Min) },
};

// NaN is unpredictable for casts to non-fp types, so we don't want to test it
const TestValues castFpToInt = {
    { SValue(SVE::Zero) },
    { -SValue(SVE::Zero) },
    { SValue(SVE::Inf) },
    { -SValue(SVE::Inf) },
    { SValue(SVE::Min) },
    { -SValue(SVE::Min) },
    { SValue(SVE::Max) },
    { -SValue(SVE::Max) },
    // Values for testing overflows. We add one for each possible target type because we don't
    // really know the output type in the generator library
    { SValue(SVE::AboveMaxINT8) },
    { SValue(SVE::AboveMaxINT16) },
    { SValue(SVE::AboveMaxINT32) },
    { SValue(SVE::BelowLowestINT8) },
    { SValue(SVE::BelowLowestINT16) },
    { SValue(SVE::BelowLowestINT32) },
};

// Maps operators to the operator-specific list of default values for
// FP_SPECIAL tests.
const std::map<Op, TestValues> opTestValues_fp = {
    { Op::Op_EQUAL, conditionalOpsTestVals },
    { Op::Op_GREATER, conditionalOpsTestVals },
    { Op::Op_GREATER_EQUAL, conditionalOpsTestVals },
    { Op::Op_ADD, addTestVals },
    { Op::Op_MAXIMUM, minMaxTestVals },
    { Op::Op_MINIMUM, minMaxTestVals },
    { Op::Op_MUL, mulTestVals },
    { Op::Op_POW, powTestVals },
    { Op::Op_SUB, subTestVals },
    { Op::Op_CONV2D, dotProductTestVals },
    { Op::Op_CONV3D, dotProductTestVals },
    { Op::Op_DEPTHWISE_CONV2D, dotProductTestVals },
    { Op::Op_TRANSPOSE_CONV2D, dotProductTestVals },
    { Op::Op_AVG_POOL2D, dotProductTestVals },
    { Op::Op_MATMUL, dotProductTestVals },
    { Op::Op_REDUCE_SUM, dotProductTestVals },
    { Op::Op_REDUCE_PRODUCT, dotProductTestVals },
};

// Values that will be picked up if the Op is not in opTestValues_fp and the
// conformance test does not have a specific SpecialTestSet assigned.
const TestValues defaultTestValues_fp{ { SValue(SVE::Zero) },       { -SValue(SVE::Zero) }, { SValue(SVE::Inf) },
                                       { -SValue(SVE::Inf) },       { SValue(SVE::Min) },   { -SValue(SVE::Min) },
                                       { SValue(SVE::Max) },        { -SValue(SVE::Max) },  { SValue(SVE::MinDenorm) },
                                       { -SValue(SVE::MinDenorm) }, { SValue(SVE::One) },   { -SValue(SVE::One) },
                                       { SValue(SVE::NaN) } };

// Maps SpecialTestSets to the list of values to be used for that test.
const std::map<SpecialTestSet, std::pair<TestValues, SpecialTestSetMode>> specialTestValues_fp = {
    { SpecialTestSet::CastFpToInt, { castFpToInt, SpecialTestSetMode::REPEAT_ALL_VALUES } }
};

}    // namespace

namespace TosaReference
{

// Define FP specialization of getSpecialConfig
template <>
SpecialGenProfile getSpecialConfig<SpecialConfig::FP>()
{
    static const SpecialGenProfile fpConfig = { opTestValues_fp, defaultTestValues_fp, specialTestValues_fp };
    return fpConfig;
};

}    // namespace TosaReference

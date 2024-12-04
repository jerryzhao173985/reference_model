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
const SValue FullRangeRndInteger{ SVE::RndInteger, SVE::Lowest, SVE::Max };

const TestValues binaryExtremesTestVals{

    { SValue(SVE::Max), SValue(SVE::Zero) },      { SValue(SVE::Zero), SValue(SVE::Max) },
    { SValue(SVE::Lowest), SValue(SVE::Zero) },   { SValue(SVE::Zero), SValue(SVE::Lowest) },
    { SValue(SVE::Zero), FullRangeRndInteger },   { FullRangeRndInteger, SValue(SVE::Zero) },
    { SValue(SVE::Max), FullRangeRndInteger },    { FullRangeRndInteger, SValue(SVE::Max) },
    { SValue(SVE::Lowest), FullRangeRndInteger }, { FullRangeRndInteger, SValue(SVE::Lowest) },
    { SValue(SVE::Zero), SValue(SVE::Zero) },
};

const TestValues shiftTestVals{

    { SValue(SVE::Max), SValue(SVE::RndInteger, SVE::Zero, SVE::MaxShift) },
    { SValue(SVE::Max), SValue(SVE::Zero) },
    { SValue(SVE::Max), SValue(SVE::MaxShift) },
    { SValue(SVE::Zero), SValue(SVE::RndInteger, SVE::Zero, SVE::MaxShift) },
    { SValue(SVE::Zero), SValue(SVE::Zero) },
    { SValue(SVE::Zero), SValue(SVE::MaxShift) },
    { SValue(SVE::Lowest), SValue(SVE::RndInteger, SVE::Zero, SVE::MaxShift) },
    { SValue(SVE::Lowest), SValue(SVE::Zero) },
    { SValue(SVE::Lowest), SValue(SVE::MaxShift) },
    { FullRangeRndInteger, SValue(SVE::Zero) },
    { FullRangeRndInteger, SValue(SVE::MaxShift) },
};

const TestValues mulTestVals{
    { SValue(SVE::Max), SValue(SVE::Zero) },
    { SValue(SVE::Max), SValue(SVE::One) },
    { SValue(SVE::Max), -SValue(SVE::One) },
    { SValue(SVE::Lowest), SValue(SVE::Zero) },
    { SValue(SVE::Lowest), SValue(SVE::One) },
    { SValue(SVE::Zero), SValue(SVE::Max) },
    { SValue(SVE::Zero), SValue(SVE::Lowest) },
    { SValue(SVE::Zero), FullRangeRndInteger },
    { SValue(SVE::One), SValue(SVE::Max) },
    { SValue(SVE::One), SValue(SVE::Lowest) },
    { SValue(SVE::One), FullRangeRndInteger },
    { -SValue(SVE::One), SValue(SVE::Max) },
    { -SValue(SVE::One), SValue(SVE::RndInteger, SVE::Zero, SVE::Max) },
    { -SValue(SVE::One), -SValue(SVE::RndInteger, SVE::Zero, SVE::Max) },
    { FullRangeRndInteger, SValue(SVE::Zero) },
    { FullRangeRndInteger, SValue(SVE::One) },
    // Some verbosity needed to avoid testing `lowest` * -1
    { SValue(SVE::RndInteger, SVE::Zero, SVE::Max), -SValue(SVE::One) },
    { -SValue(SVE::RndInteger, SVE::Zero, SVE::Max), -SValue(SVE::One) },
};

const TestValues addTestVals{
    { SValue(SVE::Max), SValue(SVE::Lowest) },
    { SValue(SVE::Max), -SValue(SVE::Max) },
    { -SValue(SVE::Max), SValue(SVE::Max) },
    { SValue(SVE::Max), SValue(SVE::Zero) },
    { SValue(SVE::Max), SValue(SVE::RndInteger, SVE::Lowest, SVE::Zero) },
    { SValue(SVE::Lowest), SValue(SVE::Max) },
    { SValue(SVE::Lowest), SValue(SVE::Zero) },
    { SValue(SVE::Lowest), SValue(SVE::RndInteger, SVE::Zero, SVE::Max) },
    { SValue(SVE::Zero), SValue(SVE::Max) },
    { SValue(SVE::Zero), SValue(SVE::Lowest) },
    { SValue(SVE::RndInteger, SVE::Lowest, SVE::Zero), SValue(SVE::Max) },
    { SValue(SVE::RndInteger, SVE::Zero, SVE::Max), SValue(SVE::Lowest) },
};

const TestValues subTestVals{
    { SValue(SVE::Max), SValue(SVE::Max) },
    { SValue(SVE::Max), SValue(SVE::Zero) },
    { SValue(SVE::Max), SValue(SVE::RndInteger, SVE::Zero, SVE::Max) },
    { SValue(SVE::Zero), SValue(SVE::Max) },
    { -SValue(SVE::One), SValue(SVE::Lowest) },
    { SValue(SVE::RndInteger, SVE::Zero, SVE::Max), SValue(SVE::Max) },
    { SValue(SVE::RndInteger, SVE::Lowest, SVE::Zero), SValue(SVE::Lowest) },
    { SValue(SVE::Lowest), SValue(SVE::Lowest) },
    { SValue(SVE::Lowest), SValue(SVE::Zero) },
    { SValue(SVE::Lowest), -SValue(SVE::One) },
    { SValue(SVE::Lowest), SValue(SVE::RndInteger, SVE::Lowest, SVE::Zero) },
};

const TestValues intDivTestVals{
    { SValue(SVE::Zero), SValue(SVE::Max) },
    { SValue(SVE::Zero), SValue(SVE::Lowest) },
    { SValue(SVE::Zero), FullRangeRndInteger },
    { SValue(SVE::Max), SValue(SVE::One) },
    { SValue(SVE::Max), SValue(SVE::RndInteger, SVE::One, SVE::Max) },
    { SValue(SVE::Max), -SValue(SVE::RndInteger, SVE::One, SVE::Max) },
    { SValue(SVE::Lowest), SValue(SVE::RndInteger, SVE::One, SVE::Max) },
    { SValue(SVE::Lowest), SValue(SVE::One) },
    // Avoid INTDIV(Lowest, -1) which overflows
    { -SValue(SVE::Max), -SValue(SVE::RndInteger, SVE::One, SVE::Max) },
    { FullRangeRndInteger, SValue(SVE::Max) },
    { FullRangeRndInteger, SValue(SVE::Lowest) },
};

// Avoid lowest value for int32 as will overflow
const TestValues absNegateTestVals{
    { SValue(SVE::Zero) }, { SValue(SVE::Max) },  { -SValue(SVE::Max) },
    { SValue(SVE::One) },  { -SValue(SVE::One) }, { SValue(SVE::RndSignInteger, SVE::Max, SVE::Max) },
};

// Maps operators to the operator-specific list of default values for
// INT_SPECIAL tests.
const std::map<Op, TestValues> opTestValues_int = { { Op_EQUAL, binaryExtremesTestVals },
                                                    { Op_GREATER, binaryExtremesTestVals },
                                                    { Op_GREATER_EQUAL, binaryExtremesTestVals },
                                                    { Op_BITWISE_AND, binaryExtremesTestVals },
                                                    { Op_BITWISE_OR, binaryExtremesTestVals },
                                                    { Op_BITWISE_XOR, binaryExtremesTestVals },
                                                    { Op_MAXIMUM, binaryExtremesTestVals },
                                                    { Op_MINIMUM, binaryExtremesTestVals },
                                                    { Op_LOGICAL_LEFT_SHIFT, shiftTestVals },
                                                    { Op_LOGICAL_RIGHT_SHIFT, shiftTestVals },
                                                    { Op_ARITHMETIC_RIGHT_SHIFT, shiftTestVals },
                                                    { Op_MUL, mulTestVals },
                                                    { Op_INTDIV, intDivTestVals },
                                                    { Op_ADD, addTestVals },
                                                    { Op_SUB, subTestVals },
                                                    { Op_ABS, absNegateTestVals },
                                                    { Op_NEGATE, absNegateTestVals } };

// Values that will be picked up if the Op is not in opTestValues_int and the
// conformance test does not have a specific SpecialTestSet assigned.
const TestValues defaultTestValues_int{
    { SValue(SVE::Zero) }, { SValue(SVE::Lowest) }, { SValue(SVE::Max) },    { -SValue(SVE::Max) },
    { SValue(SVE::One) },  { -SValue(SVE::One) },   { FullRangeRndInteger },
};

// Single value test sets
const TestValues allMaxValues           = { { SValue(SVE::Max) } };
const TestValues allLowestValues        = { { SValue(SVE::Lowest) } };
const TestValues allZeroes              = { { SValue(SVE::Zero) } };
const TestValues allSmallValues         = { { SValue(SVE::RndSignInteger, SVE::Two, SVE::Two) } };
const TestValues firstMaxThenZero       = { { SValue(SVE::Max) }, { SValue(SVE::Zero) } };
const TestValues firstLowestThenZero    = { { SValue(SVE::Lowest) }, { SValue(SVE::Zero) } };
const TestValues firstMaxThenMinusOne   = { { SValue(SVE::Max) }, { -SValue(SVE::One) } };
const TestValues firstLowestThenPlusOne = { { SValue(SVE::Lowest) }, { SValue(SVE::One) } };

// Maps SpecialTestSets to the list of values to be used for that test.
const std::map<SpecialTestSet, std::pair<TestValues, SpecialTestSetMode>> specialTestValues_int = {
    { SpecialTestSet::AllMaxValues, { allMaxValues, SpecialTestSetMode::REPEAT_ALL_VALUES } },
    { SpecialTestSet::AllLowestValues, { allLowestValues, SpecialTestSetMode::REPEAT_ALL_VALUES } },
    { SpecialTestSet::AllZeroes, { allZeroes, SpecialTestSetMode::REPEAT_ALL_VALUES } },
    { SpecialTestSet::AllSmallValues, { allSmallValues, SpecialTestSetMode::REPEAT_ALL_VALUES } },
    { SpecialTestSet::FirstMaxThenZeroes, { firstMaxThenZero, SpecialTestSetMode::REPEAT_LAST_VALUE } },
    { SpecialTestSet::FirstLowestThenZeroes, { firstLowestThenZero, SpecialTestSetMode::REPEAT_LAST_VALUE } },
    { SpecialTestSet::FirstMaxThenMinusOnes, { firstMaxThenMinusOne, SpecialTestSetMode::REPEAT_LAST_VALUE } },
    { SpecialTestSet::FirstLowestThenPlusOnes, { firstLowestThenPlusOne, SpecialTestSetMode::REPEAT_LAST_VALUE } },
};

}    // namespace

namespace TosaReference
{

// Define INT specialization of getSpecialConfig
template <>
SpecialGenProfile getSpecialConfig<SpecialConfig::INT>()
{
    static const SpecialGenProfile intConfig = { opTestValues_int, defaultTestValues_int, specialTestValues_int };
    return intConfig;
};

}    // namespace TosaReference

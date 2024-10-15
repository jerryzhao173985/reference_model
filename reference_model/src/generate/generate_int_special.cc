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

    { SValue(SVE::Max), SValue(SVE::Zero) },          { SValue(SVE::Zero), SValue(SVE::Max) },
    { SValue(SVE::Lowest), SValue(SVE::Zero) },       { SValue(SVE::Zero), SValue(SVE::Lowest) },
    { SValue(SVE::Zero), SValue(SVE::RndInteger) },   { SValue(SVE::RndInteger), SValue(SVE::Zero) },
    { SValue(SVE::Max), SValue(SVE::RndInteger) },    { SValue(SVE::RndInteger), SValue(SVE::Max) },
    { SValue(SVE::Lowest), SValue(SVE::RndInteger) }, { SValue(SVE::RndInteger), SValue(SVE::Lowest) },
    { SValue(SVE::Zero), SValue(SVE::Zero) },
};

// Maps operators to the operator-specific list of default values for
// INT_SPECIAL tests.
const std::map<Op, TestValues> opTestValues_int = {
    { Op_EQUAL, conditionalOpsTestVals },
    { Op_GREATER, conditionalOpsTestVals },
    { Op_GREATER_EQUAL, conditionalOpsTestVals },
};

// Values that will be picked up if the Op is not in opTestValues_int and the
// conformance test does not have a specific SpecialTestSet assigned.
const TestValues defaultTestValues_int{ { SValue(SVE::Zero) }, { SValue(SVE::Lowest) }, { SValue(SVE::Max) },
                                        { -SValue(SVE::Max) }, { SValue(SVE::One) },    { -SValue(SVE::One) } };

// Maps SpecialTestSets to the list of values to be used for that test.
const std::map<SpecialTestSet, TestValues> specialTestValues_int = {};

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

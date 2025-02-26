// Copyright (c) 2025 ARM Limited.
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

#include "cfloat.h"
#include "dtype.h"
#include "dtype_limits.h"
#include "half.hpp"

// Include this last because it redefines REQUIRE
#include "test_utils.h"

template <typename FP_TYPE>
void testSelectFpSpecial()
{
    constexpr DType dtype           = NativeType2DType<FP_TYPE>();
    constexpr TOSA_REF_TYPE refType = DType2RefType(dtype);

    const int TEST_VALS = 9;

    RefModelTestBuilder tb{};
    tb.addInput({ TEST_VALS }, DType_BOOL);
    tb.addInput({ TEST_VALS }, dtype);
    tb.addInput({ TEST_VALS }, dtype);
    tb.addOutput({ TEST_VALS }, dtype);

    TosaSelectAttribute attr{};
    tb.addOp(Op_SELECT, Attribute_SelectAttribute, &attr);

    tb.initializeRunner();

    const FP_TYPE lowest = DtypeLimits<refType>::lowest;
    const FP_TYPE max    = DtypeLimits<refType>::max;
    const FP_TYPE nan    = DtypeLimits<refType>::quiet_NaN;
    // NOTE: all types supported by SELECT support inf
    const FP_TYPE inf = DtypeLimits<refType>::infinity;

    const int8_t SELECT_INPUT2 = 1;
    const int8_t SELECT_INPUT3 = 0;

    // Alternate between selecting from input2 and input3
    std::vector<int8_t> selection = {
        SELECT_INPUT2, SELECT_INPUT3, SELECT_INPUT2, SELECT_INPUT3, SELECT_INPUT2,
        SELECT_INPUT3, SELECT_INPUT2, SELECT_INPUT3, SELECT_INPUT2,
    };
    std::vector<FP_TYPE> inVals2 = {
        FP_TYPE(0), lowest, inf, -inf, -inf, FP_TYPE(2), max, nan, nan,
    };
    std::vector<FP_TYPE> inVals3 = {
        FP_TYPE(2), lowest, inf, -inf, inf, lowest, FP_TYPE(0), FP_TYPE(2), max,
    };

    std::vector<FP_TYPE> expectedOut = {
        FP_TYPE(0),    // 2
        lowest,        // 3
        inf,           // 2
        -inf,          // 3
        -inf,          // 2
        lowest,        // 3
        max,           // 2
        FP_TYPE(2),    // 3
        nan,           // 2
    };

    tb.setInput(selection);
    tb.setInput(inVals2);
    tb.setInput(inVals3);

    REQUIRE(tb.run() == GraphStatus::TOSA_VALID);
    std::vector<FP_TYPE> actualOut = tb.getOutput<FP_TYPE>(0, /* size */ expectedOut.size());

    compareOutputSpecial<FP_TYPE>(expectedOut, actualOut);
}

TEST_SUITE("reference_model")
{
    TEST_CASE_TEMPLATE("SELECT FP SPECIAL", IN_TYPE, float, half, bfloat16)
    {
        testSelectFpSpecial<IN_TYPE>();
    }
}    // TEST_SUITE("reference_model")

// Copyright (c) 2024-2025 ARM Limited.
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
void testTanhSpecial(std::vector<FP_TYPE>& inVals, std::vector<FP_TYPE>& expectedOut)
{
    RefModelTestBuilder tb{};
    constexpr DType dtype = NativeType2DType<FP_TYPE>();
    const bool sizeMatch  = (inVals.size() == expectedOut.size());
    REQUIRE_MESSAGE(sizeMatch, "test construction error: size mismatch input: %d and output: %d", inVals.size(),
                    expectedOut.size());

    int32_t sz = static_cast<int32_t>(inVals.size());
    tb.addInput({ sz }, dtype);
    tb.addOutput({ sz }, dtype);

    TosaTanhAttribute attr{};
    tb.addOp(Op_TANH, Attribute_TanhAttribute, &attr);

    tb.initializeRunner();

    tb.setInput(inVals);
    REQUIRE(tb.run() == GraphStatus::TOSA_VALID);
    std::vector<FP_TYPE> actualOut = tb.getOutput<FP_TYPE>(0, expectedOut.size());

    compareOutputSpecial<FP_TYPE>(expectedOut, actualOut);
}

template <typename FP_TYPE>
void testSigmoidSpecial(std::vector<FP_TYPE>& inVals, std::vector<FP_TYPE>& expectedOut)
{
    RefModelTestBuilder tb{};
    constexpr DType dtype = NativeType2DType<FP_TYPE>();
    const bool sizeMatch  = (inVals.size() == expectedOut.size());
    REQUIRE_MESSAGE(sizeMatch, "test construction error: size mismatch input: %d and output: %d", inVals.size(),
                    expectedOut.size());

    int32_t sz = static_cast<int32_t>(inVals.size());
    tb.addInput({ sz }, dtype);
    tb.addOutput({ sz }, dtype);

    TosaSigmoidAttribute attr{};
    tb.addOp(Op_SIGMOID, Attribute_SigmoidAttribute, &attr);

    tb.initializeRunner();

    tb.setInput(inVals);
    REQUIRE(tb.run() == GraphStatus::TOSA_VALID);
    std::vector<FP_TYPE> actualOut = tb.getOutput<FP_TYPE>(0, /* size */ expectedOut.size());

    compareOutputSpecial<FP_TYPE>(expectedOut, actualOut);
}

TEST_SUITE("reference_model")
{
    TEST_CASE_TEMPLATE("Tanh FP special values", FP_TYPE, float, half, bfloat16)
    {
        SUBCASE("special behaviour")
        {
            INFO("This test is meant to catch special cases for tanh");
            constexpr DType dtype           = NativeType2DType<FP_TYPE>();
            constexpr TOSA_REF_TYPE refType = DType2RefType(dtype);
            const FP_TYPE inf               = DtypeLimits<refType>::infinity;
            const FP_TYPE nan               = DtypeLimits<refType>::quiet_NaN;

            std::vector<FP_TYPE> inVals      = { -inf, inf, -static_cast<FP_TYPE>(0), static_cast<FP_TYPE>(0), nan };
            std::vector<FP_TYPE> expectedOut = { static_cast<FP_TYPE>(-1), static_cast<FP_TYPE>(1),
                                                 -static_cast<FP_TYPE>(0), static_cast<FP_TYPE>(0), nan };

            testTanhSpecial(inVals, expectedOut);
        }
    }

    TEST_CASE_TEMPLATE("Sigmoid FP special values", FP_TYPE, float, half, bfloat16)
    {
        SUBCASE("special behaviour")
        {
            INFO("This test is meant to catch special cases for sigmoid");
            constexpr DType dtype           = NativeType2DType<FP_TYPE>();
            constexpr TOSA_REF_TYPE refType = DType2RefType(dtype);
            const FP_TYPE inf               = DtypeLimits<refType>::infinity;
            const FP_TYPE nan               = DtypeLimits<refType>::quiet_NaN;

            std::vector<FP_TYPE> inVals      = { -inf, inf, -static_cast<FP_TYPE>(0), static_cast<FP_TYPE>(0), nan };
            std::vector<FP_TYPE> expectedOut = { static_cast<FP_TYPE>(0), static_cast<FP_TYPE>(1),
                                                 static_cast<FP_TYPE>(0.5), static_cast<FP_TYPE>(0.5), nan };

            testSigmoidSpecial(inVals, expectedOut);
        }
    }
}    // TEST_SUITE("reference_model")

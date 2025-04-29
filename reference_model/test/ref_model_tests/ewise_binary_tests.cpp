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

// Include this last because it redefines REQUIRE
#include "test_utils.h"

template <typename FP_TYPE>
void testPow(std::vector<FP_TYPE>& inVals1, std::vector<FP_TYPE>& inVals2, std::vector<FP_TYPE>& expectedOut)
{
    RefModelTestBuilder tb{};
    constexpr DType dtype = NativeType2DType<FP_TYPE>();

    const bool sizeMatch = (inVals1.size() == inVals2.size()) && (inVals1.size() == expectedOut.size());
    REQUIRE_MESSAGE(sizeMatch, "test construction error: size mismatch input1: %d, input2: %d, and output: %d",
                    inVals1.size(), inVals2.size(), expectedOut.size());

    int32_t sz = static_cast<int32_t>(inVals1.size());
    tb.addInput({ sz }, dtype);
    tb.addInput({ sz }, dtype);
    tb.addOutput({ sz }, dtype);

    TosaPowAttribute attr{};
    tb.addOp(Op_POW, Attribute_PowAttribute, &attr);

    tb.initializeRunner();

    tb.setInput(inVals1);
    tb.setInput(inVals2);

    REQUIRE(tb.run() == GraphStatus::TOSA_VALID);
    std::vector<FP_TYPE> actualOut = tb.getOutput<FP_TYPE>(0, /* size */ expectedOut.size());

    compareOutput<FP_TYPE>(expectedOut, actualOut);
}

TEST_SUITE("reference_model")
{
    TEST_CASE_TEMPLATE("POW", FP_TYPE, float, float16, bfloat16)
    {
        SUBCASE("special behaviour")
        {

            constexpr DType dtype           = NativeType2DType<FP_TYPE>();
            constexpr TOSA_REF_TYPE refType = DType2RefType(dtype);
            const FP_TYPE max               = DtypeLimits<refType>::max;

            std::vector<FP_TYPE> inVals1 = {
                static_cast<FP_TYPE>(+0.0), static_cast<FP_TYPE>(+0.0),
                static_cast<FP_TYPE>(+0.0), static_cast<FP_TYPE>(-0.0),
                static_cast<FP_TYPE>(-0.0), static_cast<FP_TYPE>(15.0),
                static_cast<FP_TYPE>(2.0),  static_cast<FP_TYPE>(1.0),
                static_cast<FP_TYPE>(0.25), max,
            };

            std::vector<FP_TYPE> inVals2 = {
                static_cast<FP_TYPE>(1.0),
                static_cast<FP_TYPE>(13.0),
                static_cast<FP_TYPE>(0.5),
                static_cast<FP_TYPE>(0.25),
                max,
                static_cast<FP_TYPE>(+0.0),
                static_cast<FP_TYPE>(+0.0),
                static_cast<FP_TYPE>(-0.0),
                static_cast<FP_TYPE>(+0.0),
                static_cast<FP_TYPE>(-0.0),
            };

            std::vector<FP_TYPE> expectedOut = {
                // pow(+-0, y > 0)
                static_cast<FP_TYPE>(0.0),
                static_cast<FP_TYPE>(0.0),
                static_cast<FP_TYPE>(0.0),
                static_cast<FP_TYPE>(0.0),
                static_cast<FP_TYPE>(0.0),
                // pow(x > 0, +-0)
                static_cast<FP_TYPE>(1.0),
                static_cast<FP_TYPE>(1.0),
                static_cast<FP_TYPE>(1.0),
                static_cast<FP_TYPE>(1.0),
                static_cast<FP_TYPE>(1.0),
            };
            testPow<FP_TYPE>(inVals1, inVals2, expectedOut);
        }
    }
}    // TEST_SUITE("reference_model")

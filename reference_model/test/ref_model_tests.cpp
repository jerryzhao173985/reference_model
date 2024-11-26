// Copyright (c) 2024 ARM Limited.
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

template <typename IN_FP_TYPE, typename OUT_FP_TYPE>
void testCastFpSpecial()
{
    RefModelTestBuilder tb{};
    constexpr DType inDtype            = NativeType2Dtype<IN_FP_TYPE>();
    constexpr TOSA_REF_TYPE inReftype  = DType2RefType(inDtype);
    constexpr DType outDtype           = NativeType2Dtype<OUT_FP_TYPE>();
    constexpr TOSA_REF_TYPE outReftype = DType2RefType(outDtype);

    const int TEST_VALUES = 7;

    tb.addInput({ TEST_VALUES }, inDtype);
    tb.addOutput({ TEST_VALUES }, outDtype);

    tb.addOp(Op_CAST, Attribute_NONE, nullptr);

    tb.initializeRunner();

    const IN_FP_TYPE in_nan       = DtypeLimits<inReftype>::quiet_NaN;
    const IN_FP_TYPE in_inf       = DtypeLimits<inReftype>::has_infinity ? DtypeLimits<inReftype>::infinity : in_nan;
    const IN_FP_TYPE in_above_max = (double(DtypeLimits<inReftype>::max) > double(DtypeLimits<outReftype>::max) * 2)
                                        ? IN_FP_TYPE(IN_FP_TYPE(DtypeLimits<outReftype>::max) * IN_FP_TYPE(2))
                                        : in_inf;

    std::vector<IN_FP_TYPE> inVals = {
        in_inf,
        -in_inf,
        in_above_max,
        -in_above_max,
        in_nan,
        static_cast<IN_FP_TYPE>(+0.0),
        static_cast<IN_FP_TYPE>(-0.0),
    };

    const OUT_FP_TYPE out_nan = DtypeLimits<outReftype>::quiet_NaN;
    const OUT_FP_TYPE out_inf = (DtypeLimits<outReftype>::has_infinity && DtypeLimits<inReftype>::has_infinity)
                                    ? DtypeLimits<outReftype>::infinity
                                    : out_nan;

    std::vector<OUT_FP_TYPE> expectedOut = {
        out_inf, -out_inf, out_inf, -out_inf, out_nan, static_cast<OUT_FP_TYPE>(+0.0f), static_cast<OUT_FP_TYPE>(-0.0f)
    };

    tb.setInput(inVals);

    REQUIRE(tb.run() == GraphStatus::TOSA_VALID);
    std::vector<OUT_FP_TYPE> actualOut = tb.getOutput<OUT_FP_TYPE>(0, /* size */ expectedOut.size());

    compareOutputSpecial<OUT_FP_TYPE>(expectedOut, actualOut);
}

template <typename IN_TYPE>
void testArgmaxFpSpecial(bool propagate)
{
    constexpr DType inDtype           = NativeType2Dtype<IN_TYPE>();
    constexpr TOSA_REF_TYPE inReftype = DType2RefType(inDtype);

    const int TEST_ROWS    = 2;
    const int TEST_COLUMNS = 9;

    RefModelTestBuilder tb{};
    tb.addInput({ TEST_ROWS, TEST_COLUMNS }, inDtype);
    tb.addOutput({ TEST_COLUMNS }, DType_INT32);

    TosaAttributeBase* attr =
        new TosaAxisAttribute(/* axis */ 0, propagate ? NanPropagationMode_PROPAGATE : NanPropagationMode_IGNORE);
    tb.addOp(Op_ARGMAX, Attribute_AxisAttribute, attr);

    tb.initializeRunner();

    const IN_TYPE lowest = DtypeLimits<inReftype>::lowest;
    const IN_TYPE nan    = DtypeLimits<inReftype>::quiet_NaN;
    const IN_TYPE inf    = DtypeLimits<inReftype>::has_infinity ? DtypeLimits<inReftype>::infinity : nan;
    const bool hasInf    = DtypeLimits<inReftype>::has_infinity;

    // NOTE: Eigen stores Tensors in column-major order but we initialise them row-wise
    std::vector<IN_TYPE> inVals = { // Row 0
                                    IN_TYPE(0), lowest, inf, -inf, -inf, -inf, lowest, nan, nan,
                                    // Row 1
                                    IN_TYPE(2), lowest, inf, -inf, inf, lowest, IN_TYPE(0), IN_TYPE(0), nan
    };

    // (-inf, lowest) in fp8e4m3 is a complicated case because it becomes (NaN, lowest)
    // And so in that case the result depends on propagation
    std::vector<int32_t> expectedOut = {
        1,                                   // 0, 2
        0,                                   // lowest, lowest
        0,                                   // inf, inf
        0,                                   // -inf, -inf
        hasInf ? 1 : 0,                      // -inf, inf
        hasInf ? 1 : (propagate ? 0 : 1),    // -inf, lowest
        1,                                   // lowest, 0
        propagate ? 0 : 1,                   // nan, 0
        0                                    // nan, nan
    };
    tb.setInput(inVals);

    REQUIRE(tb.run() == GraphStatus::TOSA_VALID);
    std::vector<int32_t> actualOut = tb.getOutput<int32_t>(0, /* size */ expectedOut.size());

    compareOutput<int32_t>(expectedOut, actualOut);
}

TEST_SUITE("reference_model")
{
    TEST_CASE("CAST FP SPECIAL")
    {
        SUBCASE("fp32 -> fp16")
        {
            testCastFpSpecial<float, half>();
        }

        SUBCASE("fp32 -> bf16")
        {
            testCastFpSpecial<float, bfloat16>();
        }

        SUBCASE("fp32 -> fp8e4m3")
        {
            testCastFpSpecial<float, fp8_e4m3>();
        }

        SUBCASE("fp32 -> fp8e5m2")
        {
            testCastFpSpecial<float, fp8_e5m2>();
        }

        SUBCASE("fp16 -> fp32")
        {
            testCastFpSpecial<half, float>();
        }

        SUBCASE("fp16 -> fp8e4m3")
        {
            testCastFpSpecial<half, fp8_e4m3>();
        }

        SUBCASE("fp16 -> fp8e5m2")
        {
            testCastFpSpecial<half, fp8_e5m2>();
        }

        SUBCASE("bf16 -> fp32")
        {
            testCastFpSpecial<bfloat16, float>();
        }

        SUBCASE("bf16 -> fp8e4m3")
        {
            testCastFpSpecial<bfloat16, fp8_e4m3>();
        }

        SUBCASE("bf16 -> fp8e5m2")
        {
            testCastFpSpecial<bfloat16, fp8_e5m2>();
        }

        SUBCASE("fp8e5m2 -> fp32")
        {
            testCastFpSpecial<fp8_e5m2, float>();
        }

        SUBCASE("fp8e5m2 -> fp16")
        {
            testCastFpSpecial<fp8_e5m2, half>();
        }

        SUBCASE("fp8e5m2 -> bf16")
        {
            testCastFpSpecial<fp8_e5m2, bfloat16>();
        }

        SUBCASE("fp8e4m3 -> fp32")
        {
            testCastFpSpecial<fp8_e4m3, float>();
        }

        SUBCASE("fp8e4m3 -> fp16")
        {
            testCastFpSpecial<fp8_e4m3, half>();
        }

        SUBCASE("fp8e4m3 -> bf16")
        {
            testCastFpSpecial<fp8_e4m3, bfloat16>();
        }
    }

    TEST_CASE_TEMPLATE("ARGMAX FP SPECIAL", IN_TYPE, float, half, bfloat16, fp8_e5m2, fp8_e4m3)
    {
        SUBCASE("propagate")
        {
            testArgmaxFpSpecial<IN_TYPE>(true);
        }
        SUBCASE("ignore")
        {
            testArgmaxFpSpecial<IN_TYPE>(false);
        }
    }
}    // TEST_SUITE("reference_model")

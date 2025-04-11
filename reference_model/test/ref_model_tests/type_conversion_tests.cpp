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

template <typename IN_FP_TYPE, typename OUT_FP_TYPE>
void testCastFpSpecial()
{
    RefModelTestBuilder tb{};
    constexpr DType inDtype            = NativeType2DType<IN_FP_TYPE>();
    constexpr TOSA_REF_TYPE inReftype  = DType2RefType(inDtype);
    constexpr DType outDtype           = NativeType2DType<OUT_FP_TYPE>();
    constexpr TOSA_REF_TYPE outReftype = DType2RefType(outDtype);

    const int TEST_VALUES = 7;

    tb.addInput({ TEST_VALUES }, inDtype);
    tb.addOutput({ TEST_VALUES }, outDtype);

    TosaCastAttribute attr{};
    tb.addOp(Op_CAST, Attribute_CastAttribute, &attr);

    tb.initializeRunner();

    const IN_FP_TYPE in_nan = DtypeLimits<inReftype>::quiet_NaN;
    const IN_FP_TYPE in_inf = DtypeLimits<inReftype>::has_infinity ? DtypeLimits<inReftype>::infinity : in_nan;
    const IN_FP_TYPE in_above_max =
        (double(DtypeLimits<inReftype>::max) > double(DtypeLimits<outReftype>::max) * 2)
            ? ct::compat::cast<IN_FP_TYPE>(ct::compat::cast<IN_FP_TYPE>(DtypeLimits<outReftype>::max) * IN_FP_TYPE(2))
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

template <typename IN_TYPE, typename OUT_TYPE>
void testCast(IN_TYPE in, OUT_TYPE expected)
{
    RefModelTestBuilder tb{};
    constexpr DType inDtype  = NativeType2DType<IN_TYPE>();
    constexpr DType outDtype = NativeType2DType<OUT_TYPE>();

    const int TEST_VALUES = 1;

    tb.addInput({ TEST_VALUES }, inDtype);
    tb.addOutput({ TEST_VALUES }, outDtype);

    TosaCastAttribute attr{};
    tb.addOp(Op_CAST, Attribute_CastAttribute, &attr);

    tb.initializeRunner();

    std::vector<IN_TYPE> inVals = { in };

    std::vector<OUT_TYPE> expectedOut = { expected };

    tb.setInput(inVals);

    REQUIRE(tb.run() == GraphStatus::TOSA_VALID);
    std::vector<OUT_TYPE> actualOut = tb.getOutput<OUT_TYPE>(0, /* size */ expectedOut.size());

    compareOutput<OUT_TYPE>(expectedOut, actualOut);
}

template <typename IN_TYPE, typename OUT_TYPE>
void testRescale(IN_TYPE input,
                 OUT_TYPE expected_output,
                 int16_t multiplier,
                 int8_t shift,
                 IN_TYPE input_zp     = 0,
                 OUT_TYPE output_zp   = 0,
                 bool output_unsigned = false)
{
    RefModelTestBuilder tb{};
    constexpr DType inDtype  = NativeType2DType<IN_TYPE>();
    constexpr DType outDtype = NativeType2DType<OUT_TYPE>();

    // Inputs
    tb.addInput({ 1 }, inDtype);        // input tensor
    tb.addInput({ 1 }, DType_INT16);    // multiplier
    tb.addInput({ 1 }, DType_INT8);     // shift
    tb.addInput({ 1 }, inDtype);        // input_zp
    tb.addInput({ 1 }, outDtype);       // output_zp

    // Output
    tb.addOutput({ 1 }, outDtype);

    // Attributes
    auto attr = TosaRescaleAttribute(/* scale32 */ false, /* rounding_mode */ RoundingMode_SINGLE_ROUND,
                                     /* per_channel */ false, /* input_unsigned */ false,
                                     /* output_unsigned */ output_unsigned);
    tb.addOp(Op_RESCALE, Attribute_RescaleAttribute, &attr);
    tb.initializeRunner();

    std::vector<IN_TYPE> inputVal      = { input };
    std::vector<int16_t> multiplierVal = { multiplier };
    std::vector<int8_t> shiftVal       = { shift };
    std::vector<IN_TYPE> inputZpVal    = { input_zp };
    std::vector<OUT_TYPE> outputZpVal  = { output_zp };
    std::vector<OUT_TYPE> expectedVal  = { expected_output };

    // Set inputs
    tb.setInput(inputVal);
    tb.setInput(multiplierVal);
    tb.setInput(shiftVal);
    tb.setInput(inputZpVal);
    tb.setInput(outputZpVal);

    // Run and compare
    REQUIRE(tb.run() == GraphStatus::TOSA_VALID);
    auto actualOut = tb.getOutput<OUT_TYPE>(0, expectedVal.size());
    compareOutput<OUT_TYPE>(expectedVal, actualOut);
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

    TEST_CASE("CAST int overflow truncate")
    {
        SUBCASE("int32 -> int16")
        {

            // This case reproduces a user-found error
            testCast<int32_t, int16_t>(491392, 32640);
            // Other test cases
            testCast<int32_t, int16_t>(0x6e17a5, 0x17a5);
            testCast<int32_t, int16_t>(0x12150000, 0x0000);
            // -267124983 == std::bitcast<int32_t>(uint32_t(0xf013ff09))
            // -247 == std::bitcast<int16_t>(uint16_t(0xff09))
            testCast<int32_t, int16_t>(-267124983, -247);
        }

        SUBCASE("int32 -> int16 signflip")
        {
            // positive input becomes negative because it's a signless
            // operation
            // -1 == std::bitcast<int16_t>(uint16_t(0xffff))
            testCast<int32_t, int16_t>(0x01ffff, -1);
            // negative input becomes positive because it's a signless
            // operation
            // -1426046977 == std::bitcast<int32_t>(uint32_t(0xab003fff))
            testCast<int32_t, int16_t>(-1426046977, 0x3fff);
        }

        SUBCASE("int16 -> int8")
        {
            testCast<int16_t, int8_t>(0x0337, 0x37);
            // -17984 == std::bitcast<int16_t>(uint16_t(0xb9c0))
            // -64 == std::bitcast<int8_t>(uint8_t(0xc0))
            testCast<int16_t, int8_t>(-17984, -64);
        }
        SUBCASE("int16 -> int8 signflip")
        {
            // -93 == std::bitcast<int8_t>(uint8_t(0xa3))
            testCast<int16_t, int8_t>(0x01a3, -93);
            // -27 == std::bitcast<int8_t>(uint8_t(0xe5))
            testCast<int16_t, int8_t>(0x0ee5, -27);
            // -28617 == std::bitcast<int16_t>(uint16_t(0x9037))
            // 55 == std::bitcast<int8_t>(uint8_t(0x37))
            testCast<int16_t, int8_t>(-28617, 55);
        }
    }

    TEST_CASE("Cast FP to INT: Round to Nearest (Ties to Even)")
    {
        SUBCASE("Halfway cases (round to even)")
        {
            testCast<float, int8_t>(2.5f, 2);
            testCast<float, int8_t>(3.5f, 4);
            testCast<float, int8_t>(-2.5f, -2);
            testCast<float, int8_t>(-3.5f, -4);
        }
        SUBCASE("Non-halfway cases")
        {
            testCast<float, int8_t>(2.3f, 2);
            testCast<float, int8_t>(2.7f, 3);
            testCast<float, int8_t>(-2.3f, -2);
            testCast<float, int8_t>(-2.7f, -3);
        }
        SUBCASE("Boundary cases")
        {
            // exceeds max, clipped to int8 max
            testCast<float, int8_t>(128.0f, 127);
            // exceeds min, clipped to int8 min
            testCast<float, int8_t>(-129.0f, -128);
        }
    }

    TEST_CASE("RESCALE Op")
    {
        SUBCASE("Test value_extend64_impl of output_zp")
        {
            // Expect: (65 - 1) * 4 >> 2 + 32768 = 32832
            // this test catches reported bugs where output_zp was incorrectly using IN_TYPE
            testRescale<int8_t, uint16_t>(/* inputVals */ 65, /* expectedVals */ 32832,
                                          /* multiplier */ 4, /* shift */ 2,
                                          /* input_zp */ 1, /* output_zp */ 32768,
                                          /* output_unsigned */ true);
        }
    }
}    // TEST_SUITE("reference_model")

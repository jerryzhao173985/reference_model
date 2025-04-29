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

template <typename T>
void testReduceSpecial(tosa::Op reduceOp, bool propagate, std::vector<T>& in, std::vector<T>& expected)
{
    constexpr DType dtype = NativeType2DType<T>();

    RefModelTestBuilder tb{};
    tb.addInput({ static_cast<int32_t>(in.size()) }, dtype);
    tb.addOutput({ 1 }, dtype);

    std::unique_ptr<TosaAttributeBase> attr = nullptr;
    Attribute attrType                      = Attribute_NONE;
    switch (reduceOp)
    {
        case Op_REDUCE_MIN:
            attr     = std::make_unique<TosaReduceMinAttribute>(/* axis */ 0, propagate ? NanPropagationMode_PROPAGATE
                                                                                        : NanPropagationMode_IGNORE);
            attrType = Attribute_ReduceMinAttribute;
            break;
        case Op_REDUCE_MAX:
            attr     = std::make_unique<TosaReduceMaxAttribute>(/* axis */ 0, propagate ? NanPropagationMode_PROPAGATE
                                                                                        : NanPropagationMode_IGNORE);
            attrType = Attribute_ReduceMaxAttribute;
            break;
        default:
            REQUIRE_MESSAGE(false, "Unit test construction error: testReduceSpecial using unsupported op %s",
                            EnumNameOp(reduceOp));
    }
    tb.addOp(reduceOp, attrType, attr.get());

    tb.initializeRunner();

    tb.setInput(in);

    REQUIRE(tb.run() == GraphStatus::TOSA_VALID);
    std::vector<T> actualOut = tb.getOutput<T>(0, /* size */ expected.size());

    compareOutputSpecial<T>(expected, actualOut);
}

TEST_SUITE("reference_model")
{
    TEST_CASE_TEMPLATE("REDUCE_MAX FP_SPECIAL", FP_TYPE, float, float16, bfloat16)
    {
        constexpr DType dtype           = NativeType2DType<FP_TYPE>();
        constexpr TOSA_REF_TYPE refType = DType2RefType(dtype);

        FP_TYPE nan = DtypeLimits<refType>::quiet_NaN;
        FP_TYPE inf = DtypeLimits<refType>::infinity;
        FP_TYPE max = DtypeLimits<refType>::max;

        SUBCASE("propagate NaN over inf and max")
        {
            std::vector<FP_TYPE> in       = { max, inf, nan, max, inf };
            std::vector<FP_TYPE> expected = { nan };
            testReduceSpecial<FP_TYPE>(Op_REDUCE_MAX, /*propagate*/ true, in, expected);
        }

        SUBCASE("propagate NaN over finite values")
        {
            std::vector<FP_TYPE> in       = { FP_TYPE(0), FP_TYPE(12), nan, FP_TYPE(-5) };
            std::vector<FP_TYPE> expected = { nan };
            testReduceSpecial<FP_TYPE>(Op_REDUCE_MAX, /*propagate*/ true, in, expected);
        }

        SUBCASE("ignore NaN with all NaN is still NaN - size 1")
        {
            std::vector<FP_TYPE> in       = { nan };
            std::vector<FP_TYPE> expected = { nan };
            testReduceSpecial<FP_TYPE>(Op_REDUCE_MAX, /*propagate*/ false, in, expected);
        }

        SUBCASE("ignore NaN with all NaN is still NaN - size 6")
        {
            std::vector<FP_TYPE> in       = { nan, nan, nan, nan, nan, nan };
            std::vector<FP_TYPE> expected = { nan };
            testReduceSpecial<FP_TYPE>(Op_REDUCE_MAX, /*propagate*/ false, in, expected);
        }

        SUBCASE("inf is higher than max - size 2")
        {
            std::vector<FP_TYPE> in       = { max, inf };
            std::vector<FP_TYPE> expected = { inf };
            testReduceSpecial<FP_TYPE>(Op_REDUCE_MAX, /*propagate*/ true, in, expected);
        }

        SUBCASE("inf is higher than max - size 7")
        {
            std::vector<FP_TYPE> in       = { inf, max, FP_TYPE(0), max, max, FP_TYPE(-5), max };
            std::vector<FP_TYPE> expected = { inf };
            testReduceSpecial<FP_TYPE>(Op_REDUCE_MAX, /*propagate*/ true, in, expected);
        }
    }

    TEST_CASE_TEMPLATE("REDUCE_MIN FP_SPECIAL", FP_TYPE, float, float16, bfloat16)
    {
        constexpr DType dtype           = NativeType2DType<FP_TYPE>();
        constexpr TOSA_REF_TYPE refType = DType2RefType(dtype);

        FP_TYPE nan    = DtypeLimits<refType>::quiet_NaN;
        FP_TYPE inf    = DtypeLimits<refType>::infinity;
        FP_TYPE lowest = DtypeLimits<refType>::lowest;

        SUBCASE("propagate NaN over -inf")
        {
            std::vector<FP_TYPE> in       = { -inf, -inf, nan, -inf };
            std::vector<FP_TYPE> expected = { nan };
            testReduceSpecial<FP_TYPE>(Op_REDUCE_MIN, /*propagate*/ true, in, expected);
        }

        SUBCASE("propagate NaN over finite values")
        {
            std::vector<FP_TYPE> in       = { FP_TYPE(0), FP_TYPE(12), nan, FP_TYPE(-5) };
            std::vector<FP_TYPE> expected = { nan };
            testReduceSpecial<FP_TYPE>(Op_REDUCE_MIN, /*propagate*/ true, in, expected);
        }

        SUBCASE("ignore NaN with all NaN is still NaN - size 1")
        {
            std::vector<FP_TYPE> in       = { nan };
            std::vector<FP_TYPE> expected = { nan };
            testReduceSpecial<FP_TYPE>(Op_REDUCE_MIN, /*propagate*/ false, in, expected);
        }

        SUBCASE("ignore NaN with all NaN is still NaN - size 6")
        {
            std::vector<FP_TYPE> in       = { nan, nan, nan, nan, nan, nan };
            std::vector<FP_TYPE> expected = { nan };
            testReduceSpecial<FP_TYPE>(Op_REDUCE_MIN, /*propagate*/ false, in, expected);
        }

        SUBCASE("-inf is lower than lowest - size 2")
        {
            std::vector<FP_TYPE> in       = { lowest, -inf };
            std::vector<FP_TYPE> expected = { -inf };
            testReduceSpecial<FP_TYPE>(Op_REDUCE_MIN, /*propagate*/ true, in, expected);
        }

        SUBCASE("-inf is lower than lowest - size 7")
        {
            std::vector<FP_TYPE> in       = { -inf, lowest, FP_TYPE(0), lowest, lowest, FP_TYPE(-5), lowest };
            std::vector<FP_TYPE> expected = { -inf };
            testReduceSpecial<FP_TYPE>(Op_REDUCE_MIN, /*propagate*/ true, in, expected);
        }
    }
}    // TEST_SUITE("reference_model")

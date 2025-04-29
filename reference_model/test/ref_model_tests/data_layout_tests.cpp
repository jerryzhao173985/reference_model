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

// Include this last because it redefines REQUIRE
#include "test_utils.h"

template <typename T>
void testReshape(std::vector<T>& inVals, std::vector<int32_t>& oldShape, std::vector<int32_t>& newShape)
{
    constexpr DType dtype = NativeType2DType<T>();

    auto oldSize   = std::reduce(oldShape.begin(), oldShape.end(), 1, std::multiplies<int32_t>());
    auto newSize   = std::reduce(newShape.begin(), newShape.end(), 1, std::multiplies<int32_t>());
    bool sizeMatch = (oldSize == newSize) && (oldSize == static_cast<int64_t>(inVals.size()));

    REQUIRE_MESSAGE(sizeMatch,
                    "Test construction error: oldShape and newShape must match in size with inVals "
                    "but sizes are %d, %d and %d respectively",
                    oldSize, newSize, inVals.size());

    RefModelTestBuilder tb{};

    tb.addInput(oldShape, dtype);

    tb.addInputShape(static_cast<int32_t>(newShape.size()));

    tb.addOutput(newShape, dtype);

    TosaReshapeAttribute attr{};
    tb.addOp(Op_RESHAPE, Attribute_ReshapeAttribute, &attr);

    tb.initializeRunner();

    tb.setInput(inVals);

    std::vector<int64_t> newShape64{ newShape.begin(), newShape.end() };
    tb.setInput(newShape64);

    REQUIRE(tb.run() == GraphStatus::TOSA_VALID);
    std::vector<T> actualOut = tb.getOutput<T>(0, /* size */ inVals.size());

    compareOutput(inVals, actualOut);
}

TEST_SUITE("reference_model")
{
    TEST_CASE_TEMPLATE("RESHAPE", T, float, float16, bfloat16, fp8_e5m2, fp8_e4m3, int32_t, int16_t, int8_t)
    {

        SUBCASE("rank 1 -> rank 0")
        {
            std::vector<T> inVals      = { std::numeric_limits<T>::max() };
            std::vector<int32_t> rank1 = { 1 };
            std::vector<int32_t> rank0 = {};

            testReshape(inVals, rank1, rank0);
        }

        SUBCASE("rank 0 -> rank 1")
        {
            std::vector<T> inVals      = { std::numeric_limits<T>::lowest() };
            std::vector<int32_t> rank1 = { 1 };
            std::vector<int32_t> rank0 = {};

            testReshape(inVals, rank0, rank1);
        }

        SUBCASE("rank 4 -> rank 1")
        {
            std::vector<T> inVals = {
                T(0), T(0), T(15), T(12), T(55), T(20), T(-10), T(-105), T(55.5f), T(-1), T(-2.75f), T(-12.12f),
            };
            std::vector<int32_t> rank4 = { 2, 2, 1, 3 };
            std::vector<int32_t> rank1 = { 12 };

            testReshape(inVals, rank4, rank1);
        }

        SUBCASE("rank 1 -> rank 4")
        {
            std::vector<T> inVals = {
                T(0), T(0), T(15), T(12), T(55), T(20), T(-10), T(-105), T(55.5f), T(-1), T(-2.75f), T(-12.12f),
            };
            std::vector<int32_t> rank1 = { 12 };
            std::vector<int32_t> rank4 = { 2, 2, 1, 3 };

            testReshape(inVals, rank1, rank4);
        }
    }
}

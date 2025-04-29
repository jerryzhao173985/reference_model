// Copyright (c) 2025, ARM Limited.
//
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
#include "tensor.h"

#include <cmath>
#include <numeric>

// Include this last because it redefines REQUIRE
#include "test_utils.h"

#ifdef _WIN32
#include <io.h>
#endif

using namespace TosaReference;
using namespace tosa;

static const char* tensorDumpRefOutputsF32[] = {
    // Rank 0
    R"([ 42.00000 ])",
    // Rank 1
    R"([ 1.00000  2.00000  3.00000  4.00000 ])",
    // Rank 2
    R"([[ 1.00000  2.00000  3.00000 ]
[ 4.00000  5.00000  6.00000 ]
])",
    // Rank 3
    R"([[[ 1.00000  2.00000 ]
[ 3.00000  4.00000 ]
]
[[ 5.00000  6.00000 ]
[ 7.00000  8.00000 ]
]
])",
    // Rank 4
    R"([[[[ 1.00000  2.00000  3.00000 ]
]
[[ 4.00000  5.00000  6.00000 ]
]
]
])",
    // Rank 5
    R"([[[[[ 1.00000  2.00000  3.00000 ]
]
[[ 4.00000  5.00000  6.00000 ]
]
]
[[[ 7.00000  8.00000  9.00000 ]
]
[[ 10.00000  11.00000  12.00000 ]
]
]
]
])",
    // Rank 6
    R"([[[[[[ 1.00000  2.00000  3.00000 ]
[ 4.00000  5.00000  6.00000 ]
]
]
]
[[[[ 7.00000  8.00000  9.00000 ]
[ 10.00000  11.00000  12.00000 ]
]
]
]
]
])",
};

static const char* tensorDumpRefOutputsI32[] = {
    // Rank 0
    R"([ 42 ])",
    // Rank 1
    R"([ 1  2  3  4 ])",
    // Rank 2
    R"([[ 1  2  3 ]
[ 4  5  6 ]
])",
    // Rank 3
    R"([[[ 1  2 ]
[ 3  4 ]
]
[[ 5  6 ]
[ 7  8 ]
]
])",
    // Rank 4
    R"([[[[ 1  2  3 ]
]
[[ 4  5  6 ]
]
]
])",
    // Rank 5
    R"([[[[[ 1  2  3 ]
]
[[ 4  5  6 ]
]
]
[[[ 7  8  9 ]
]
[[ 10  11  12 ]
]
]
]
])",
    // Rank 6
    R"([[[[[[ 1  2  3 ]
[ 4  5  6 ]
]
]
]
[[[[ 7  8  9 ]
[ 10  11  12 ]
]
]
]
]
])",
};

// This test verifies that tensor contents are printed in the expected format by the dumpTensor() method.
// It compares the actual printed output of a tensor (of various shapes and types) with a known reference.
template <typename T>
void testTensorDumpCompare(int rank, const std::vector<int>& shape, DType dtype, const std::vector<T>& values)
{
    std::string tensorName = "TestTensor";

    Tensor* tensor = TensorFactory::newTensor(tensorName, dtype, shape, static_cast<uint32_t>(shape.size()));
    REQUIRE(tensor != nullptr);
    REQUIRE(tensor->allocate() == 0);

    const char* expectedOutputCStr = nullptr;

    switch (dtype)
    {
        case DType_FP32:
            expectedOutputCStr = tensorDumpRefOutputsF32[rank];
            REQUIRE(tensor->setTensorValueFloat(values.size(), reinterpret_cast<const float*>(values.data())) == 0);
            break;
        case DType_INT32:
            expectedOutputCStr = tensorDumpRefOutputsI32[rank];
            REQUIRE(tensor->setTensorValueInt32(values.size(), reinterpret_cast<const int32_t*>(values.data())) == 0);
            break;
        default:
            FAIL("Unsupported dtype for testTensorDumpCompare");
    }

    std::string expectedOutput(expectedOutputCStr);
    size_t bufferSize = expectedOutput.size() + 1;

#ifdef _WIN32
    // Use tmpfile and read contents back
    FILE* memStream = tmpfile();

    tensor->dumpTensor(memStream);
    fflush(memStream);
    fseek(memStream, 0, SEEK_SET);

    std::string actualOutput;
    actualOutput.resize(bufferSize - 1);
    size_t bytesRead = fread(&actualOutput[0], 1, actualOutput.size(), memStream);
    actualOutput.resize(bytesRead);
#else
    char* buffer = new char[bufferSize];
    memset(buffer, 0, bufferSize);

    // Use fmemopen to capture dumpTensor output into a buffer
    FILE* memStream = fmemopen(buffer, bufferSize, "w");
    REQUIRE(memStream != nullptr);
    tensor->dumpTensor(memStream);
    fclose(memStream);

    std::string actualOutput(buffer);

    delete[] buffer;
#endif

    auto trim = [](std::string& s) { s.erase(s.find_last_not_of(" \n\r\t") + 1); };
    trim(actualOutput);
    trim(expectedOutput);

    REQUIRE(actualOutput == expectedOutput);
}

TEST_SUITE("reference_model")
{
    TEST_CASE_TEMPLATE("tensor_dump_tests", TYPE, float, int32_t)
    {
        constexpr DType dtype = NativeType2DType<TYPE>();

        SUBCASE("Rank 0")
        {
            testTensorDumpCompare<TYPE>(0, {}, dtype, { 42 });
        }

        SUBCASE("Rank 1")
        {
            testTensorDumpCompare<TYPE>(1, { 4 }, dtype, { 1, 2, 3, 4 });
        }

        SUBCASE("Rank 2")
        {
            testTensorDumpCompare<TYPE>(2, { 2, 3 }, dtype, { 1, 2, 3, 4, 5, 6 });
        }

        SUBCASE("Rank 3")
        {
            testTensorDumpCompare<TYPE>(3, { 2, 2, 2 }, dtype, { 1, 2, 3, 4, 5, 6, 7, 8 });
        }

        SUBCASE("Rank 4")
        {
            testTensorDumpCompare<TYPE>(4, { 1, 2, 1, 3 }, dtype, { 1, 2, 3, 4, 5, 6 });
        }

        SUBCASE("Rank 5")
        {
            testTensorDumpCompare<TYPE>(5, { 1, 2, 2, 1, 3 }, dtype, { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });
        }

        SUBCASE("Rank 6")
        {
            testTensorDumpCompare<TYPE>(6, { 1, 2, 1, 1, 2, 3 }, dtype, { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });
        }
    }
}

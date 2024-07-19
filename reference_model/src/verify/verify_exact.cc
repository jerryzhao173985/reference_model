// Copyright (c) 2023, ARM Limited.
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

#include "func_debug.h"
#include "half.hpp"
#include "verifiers.h"
#include <cmath>

namespace
{
template <typename OutDtype>
bool exact_fp(const double& referenceValue, const OutDtype& implementationValue)
{
    return std::isnan(referenceValue) ? std::isnan(implementationValue) : (referenceValue == implementationValue);
}

template <typename OutDtype>
bool exact_int(const OutDtype& referenceValue, const OutDtype& implementationValue)
{
    return referenceValue == implementationValue;
}
}    // namespace

namespace TosaReference
{

bool verifyExact(const CTensor* referenceTensor, const CTensor* implementationTensor)
{
    // Validate that tensors are provided

    TOSA_REF_REQUIRE(referenceTensor != nullptr, "[E] Reference tensor is missing");
    TOSA_REF_REQUIRE(implementationTensor != nullptr, "[E] Implementation tensor is missing");

    // Get number of elements

    const auto elementCount =
        numElements(std::vector<int32_t>(referenceTensor->shape, referenceTensor->shape + referenceTensor->num_dims));
    TOSA_REF_REQUIRE(elementCount > 0, "[E] Invalid shape for reference tensor");

    const auto* refData_dbl = reinterpret_cast<const double*>(referenceTensor->data);

    TOSA_REF_REQUIRE(refData_dbl != nullptr, "[E] Missing data for reference");

    switch (implementationTensor->data_type)
    {
        case tosa_datatype_fp32_t: {
            const auto* impData = reinterpret_cast<const float*>(implementationTensor->data);
            TOSA_REF_REQUIRE(impData != nullptr, "[E] Missing data for implementation");
            auto result = std::equal(refData_dbl, std::next(refData_dbl, elementCount), impData,
                                     std::next(impData, elementCount), exact_fp<float>);
            return result;
        }
        case tosa_datatype_fp16_t: {
            const auto* impData = reinterpret_cast<const half_float::half*>(implementationTensor->data);
            TOSA_REF_REQUIRE(impData != nullptr, "[E] Missing data for implementation");
            auto result = std::equal(refData_dbl, std::next(refData_dbl, elementCount), impData,
                                     std::next(impData, elementCount), exact_fp<half_float::half>);
            return result;
        }
        case tosa_datatype_bf16_t: {
            const auto* impData = reinterpret_cast<const bf16*>(implementationTensor->data);
            TOSA_REF_REQUIRE(impData != nullptr, "[E] Missing data for implementation");
            auto result = std::equal(refData_dbl, std::next(refData_dbl, elementCount), impData,
                                     std::next(impData, elementCount), exact_fp<bf16>);
            return result;
        }
        case tosa_datatype_fp8e4m3_t: {
            const auto* impData = reinterpret_cast<const fp8e4m3*>(implementationTensor->data);
            TOSA_REF_REQUIRE(impData != nullptr, "[E] Missing data for implementation");
            auto result = std::equal(refData_dbl, std::next(refData_dbl, elementCount), impData,
                                     std::next(impData, elementCount), exact_fp<fp8e4m3>);
            return result;
        }
        case tosa_datatype_fp8e5m2_t: {
            const auto* impData = reinterpret_cast<const fp8e5m2*>(implementationTensor->data);
            TOSA_REF_REQUIRE(impData != nullptr, "[E] Missing data for implementation");
            auto result = std::equal(refData_dbl, std::next(refData_dbl, elementCount), impData,
                                     std::next(impData, elementCount), exact_fp<fp8e5m2>);
            return result;
        }

        case tosa_datatype_int32_t: {

            const auto* refData_int = reinterpret_cast<const int32_t*>(referenceTensor->data);

            TOSA_REF_REQUIRE(refData_int != nullptr, "[E] Missing data for reference");

            const auto* impData = reinterpret_cast<const int32_t*>(implementationTensor->data);
            TOSA_REF_REQUIRE(impData != nullptr, "[E] Missing data for implementation");

            return std::equal(refData_int, std::next(refData_int, elementCount), impData,
                              std::next(impData, elementCount), exact_int<int32_t>);
        }

        case tosa_datatype_int16_t: {

            const auto* refData_int = reinterpret_cast<const int16_t*>(referenceTensor->data);

            TOSA_REF_REQUIRE(refData_int != nullptr, "[E] Missing data for reference");

            const auto* impData = reinterpret_cast<const int16_t*>(implementationTensor->data);
            TOSA_REF_REQUIRE(impData != nullptr, "[E] Missing data for implementation");

            return std::equal(refData_int, std::next(refData_int, elementCount), impData,
                              std::next(impData, elementCount), exact_int<int16_t>);
        }

        case tosa_datatype_int8_t: {

            const auto* refData_int = reinterpret_cast<const int8_t*>(referenceTensor->data);

            TOSA_REF_REQUIRE(refData_int != nullptr, "[E] Missing data for reference");

            const auto* impData = reinterpret_cast<const int8_t*>(implementationTensor->data);
            TOSA_REF_REQUIRE(impData != nullptr, "[E] Missing data for implementation");

            return std::equal(refData_int, std::next(refData_int, elementCount), impData,
                              std::next(impData, elementCount), exact_int<int8_t>);
        }

        case tosa_datatype_int48_t: {

            const auto* refData_int = reinterpret_cast<const int64_t*>(referenceTensor->data);

            TOSA_REF_REQUIRE(refData_int != nullptr, "[E] Missing data for reference");

            const auto* impData = reinterpret_cast<const int64_t*>(implementationTensor->data);
            TOSA_REF_REQUIRE(impData != nullptr, "[E] Missing data for implementation");

            return std::equal(refData_int, std::next(refData_int, elementCount), impData,
                              std::next(impData, elementCount), exact_int<int64_t>);
        }

        case tosa_datatype_uint16_t: {

            const auto* refData_int = reinterpret_cast<const uint16_t*>(referenceTensor->data);

            TOSA_REF_REQUIRE(refData_int != nullptr, "[E] Missing data for reference");

            const auto* impData = reinterpret_cast<const uint16_t*>(implementationTensor->data);
            TOSA_REF_REQUIRE(impData != nullptr, "[E] Missing data for implementation");

            return std::equal(refData_int, std::next(refData_int, elementCount), impData,
                              std::next(impData, elementCount), exact_int<uint16_t>);
        }

        case tosa_datatype_uint8_t: {

            const auto* refData_int = reinterpret_cast<const uint8_t*>(referenceTensor->data);

            TOSA_REF_REQUIRE(refData_int != nullptr, "[E] Missing data for reference");

            const auto* impData = reinterpret_cast<const uint8_t*>(implementationTensor->data);
            TOSA_REF_REQUIRE(impData != nullptr, "[E] Missing data for implementation");

            return std::equal(refData_int, std::next(refData_int, elementCount), impData,
                              std::next(impData, elementCount), exact_int<uint8_t>);
        }
        case tosa_datatype_bool_t: {

            const auto* refData_bool = reinterpret_cast<const bool*>(referenceTensor->data);

            TOSA_REF_REQUIRE(refData_bool != nullptr, "[E] Missing data for reference");

            const auto* impData = reinterpret_cast<const bool*>(implementationTensor->data);
            TOSA_REF_REQUIRE(impData != nullptr, "[E] Missing data for implementation");

            return std::equal(refData_bool, std::next(refData_bool, elementCount), impData,
                              std::next(impData, elementCount), exact_int<bool>);
        }
        default:
            WARNING("[Verifier][E] Data-type not supported.");
            break;
    }

    return false;
}
}    // namespace TosaReference

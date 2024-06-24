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

#include <cfloat>

#include "half.hpp"
#include "verifiers.h"

namespace
{
template <typename Datatype>
bool compliant(const double& referenceValue, const Datatype& implementationValue)
{
    // Compliant when values are zero (maybe different sign)
    // OR both NaNs
    // OR have the same finiteness AND the same sign
    return (referenceValue == 0.0 && static_cast<double>(implementationValue) == 0.0) ||
           (std::isnan(referenceValue) && std::isnan(implementationValue)) ||
           (std::isnan(referenceValue) == std::isnan(implementationValue) &&
            std::isfinite(referenceValue) == std::isfinite(implementationValue) &&
            std::signbit(referenceValue) == std::signbit(implementationValue));
}

template <>
bool compliant(const double& referenceValue, const half_float::half& implementationValue)
{
    // Compliant when values are zero (maybe different sign)
    // OR both NaNs
    // OR have the same finiteness AND the same sign
    return (referenceValue == 0.0 && implementationValue == 0.0) ||
           (std::isnan(referenceValue) && half_float::isnan(implementationValue)) ||
           (std::isnan(referenceValue) == half_float::isnan(implementationValue) &&
            std::isfinite(referenceValue) == half_float::isfinite(implementationValue) &&
            std::signbit(referenceValue) == half_float::signbit(implementationValue));
}

template <typename Datatype>
bool verify(const double* refData,
            const double* refDataEnd,
            Datatype* impData,
            const int64_t elementCount,
            const std::vector<int32_t> refShape)
{
    auto pair = std::mismatch(refData, refDataEnd, impData, std::next(impData, elementCount),
                              [](const double& referenceValue, const Datatype& implementationValue) {
                                  return compliant(referenceValue, implementationValue);
                              });

    if (std::get<0>(pair) == refDataEnd)
    {
        // No mismatch found
        return true;
    }
    else
    {
        auto pos = TosaReference::indexToPosition(std::get<0>(pair) - refData, refShape);
        WARNING("[Verfier][FS] Location %s", TosaReference::positionToString(pos).c_str());
        return false;
    }
}
}    // namespace

namespace TosaReference
{

bool verifyFpSpecial(const CTensor* referenceTensor, const CTensor* implementationTensor)
{
    // Validate that tensors are provided
    TOSA_REF_REQUIRE(referenceTensor != nullptr, "[FS] Reference tensor is missing");
    TOSA_REF_REQUIRE(implementationTensor != nullptr, "[FS] Implementation tensor is missing");

    // Get number of elements
    const std::vector<int32_t> refShape(referenceTensor->shape, referenceTensor->shape + referenceTensor->num_dims);
    const auto elementCount = numElements(refShape);
    TOSA_REF_REQUIRE(elementCount > 0, "[FS] Invalid shape for reference tensor");

    TOSA_REF_REQUIRE(referenceTensor->data_type == tosa_datatype_fp64_t, "[FS] Reference tensor is not fp64");
    const auto* refData = reinterpret_cast<const double*>(referenceTensor->data);
    TOSA_REF_REQUIRE(refData != nullptr, "[FS] Missing data for reference");
    const auto* refDataEnd = std::next(refData, elementCount);

    switch (implementationTensor->data_type)
    {
        case tosa_datatype_fp32_t: {
            const auto* impData = reinterpret_cast<const float*>(implementationTensor->data);
            TOSA_REF_REQUIRE(impData != nullptr, "[FS] Missing data for implementation");

            return verify(refData, refDataEnd, impData, elementCount, refShape);
        }
        case tosa_datatype_fp16_t: {
            const auto* impData = reinterpret_cast<const half_float::half*>(implementationTensor->data);
            TOSA_REF_REQUIRE(impData != nullptr, "[FS] Missing data for implementation");

            return verify(refData, refDataEnd, impData, elementCount, refShape);
        }
        case tosa_datatype_bf16_t: {
            const auto* impData = reinterpret_cast<const bf16*>(implementationTensor->data);
            TOSA_REF_REQUIRE(impData != nullptr, "[FS] Missing data for implementation");

            return verify(refData, refDataEnd, impData, elementCount, refShape);
        }
        case tosa_datatype_fp8e4m3_t: {
            const auto* impData = reinterpret_cast<const fp8e4m3*>(implementationTensor->data);
            TOSA_REF_REQUIRE(impData != nullptr, "[FS] Missing data for implementation");

            return verify(refData, refDataEnd, impData, elementCount, refShape);
        }
        case tosa_datatype_fp8e5m2_t: {
            const auto* impData = reinterpret_cast<const fp8e5m2*>(implementationTensor->data);
            TOSA_REF_REQUIRE(impData != nullptr, "[FS] Missing data for implementation");

            return verify(refData, refDataEnd, impData, elementCount, refShape);
        }
        default:
            WARNING("[Verifier][FS] Data-type not supported.");
            break;
    }

    return false;
}
}    // namespace TosaReference

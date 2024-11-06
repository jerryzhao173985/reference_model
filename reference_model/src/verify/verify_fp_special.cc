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
bool compliant(const double& referenceValue, const double& boundValue, const Datatype& implementationValue)
{
    // Compliant when values are zero (maybe different sign)
    // OR implementation is NaN and either the reference or bound (with extra multiplies) is NaN
    // OR both are/not NaN AND have the same finiteness AND the same sign
    return (referenceValue == 0.0 && static_cast<double>(implementationValue) == 0.0) ||
           (std::isnan(implementationValue) && (std::isnan(referenceValue) || std::isnan(boundValue))) ||
           (std::isnan(referenceValue) == std::isnan(implementationValue) &&
            std::isfinite(referenceValue) == std::isfinite(implementationValue) &&
            std::signbit(referenceValue) == std::signbit(implementationValue));
}

template <>
bool compliant(const double& referenceValue, const double& boundValue, const half_float::half& implementationValue)
{
    // Compliant when values are zero (maybe different sign)
    // OR both NaNs
    // OR ref is not NaN but bound value is NaN
    // OR have the same finiteness AND the same sign
    return (referenceValue == 0.0 && implementationValue == 0.0) ||
           (std::isnan(referenceValue) && half_float::isnan(implementationValue)) ||
           (!std::isnan(referenceValue) && std::isnan(boundValue)) ||
           (std::isnan(referenceValue) == half_float::isnan(implementationValue) &&
            std::isfinite(referenceValue) == half_float::isfinite(implementationValue) &&
            std::signbit(referenceValue) == half_float::signbit(implementationValue));
}

template <typename Datatype>
bool verify(const double* refData,
            const double* refBndData,
            const Datatype* impData,
            const int64_t elementCount,
            const std::vector<int32_t> refShape)
{
    for (int64_t i = 0; i < elementCount; i++)
    {
        if (!compliant<Datatype>(refData[i], refBndData[i], impData[i]))
        {
            // mismatch found
            auto pos = TosaReference::indexToPosition(i, refShape);
            WARNING("[Verfier][FS] Location %s", TosaReference::positionToString(pos).c_str());
            return false;
        }
    }
    // No mismatch found
    return true;
}
}    // namespace

namespace TosaReference
{

bool verifyFpSpecial(const CTensor* referenceTensor, const CTensor* boundsTensor, const CTensor* implementationTensor)
{
    // Validate that tensors are provided
    TOSA_REF_REQUIRE(referenceTensor != nullptr, "[FS] Reference tensor is missing");
    TOSA_REF_REQUIRE(implementationTensor != nullptr, "[FS] Implementation tensor is missing");

    // Get number of elements
    const std::vector<int32_t> refShape(referenceTensor->shape, referenceTensor->shape + referenceTensor->num_dims);
    const auto elementCount = numElements(refShape);
    TOSA_REF_REQUIRE(elementCount > 0, "[FS] Invalid shape for reference tensor");

    TOSA_REF_REQUIRE(referenceTensor->data_type == tosa_datatype_fp64_t, "[FS] Reference tensor is not fp64");
    const auto* refData    = reinterpret_cast<const double*>(referenceTensor->data);
    const auto* refBndData = reinterpret_cast<const double*>(boundsTensor->data);
    TOSA_REF_REQUIRE(refData != nullptr && refBndData != nullptr, "[FS] Missing data for reference");

    switch (implementationTensor->data_type)
    {
        case tosa_datatype_fp32_t: {
            const auto* impData = reinterpret_cast<const float*>(implementationTensor->data);
            TOSA_REF_REQUIRE(impData != nullptr, "[FS] Missing data for implementation");

            return verify(refData, refBndData, impData, elementCount, refShape);
        }
        case tosa_datatype_fp16_t: {
            const auto* impData = reinterpret_cast<const half_float::half*>(implementationTensor->data);
            TOSA_REF_REQUIRE(impData != nullptr, "[FS] Missing data for implementation");

            return verify(refData, refBndData, impData, elementCount, refShape);
        }
        case tosa_datatype_bf16_t: {
            const auto* impData = reinterpret_cast<const bf16*>(implementationTensor->data);
            TOSA_REF_REQUIRE(impData != nullptr, "[FS] Missing data for implementation");

            return verify(refData, refBndData, impData, elementCount, refShape);
        }
        case tosa_datatype_fp8e4m3_t: {
            const auto* impData = reinterpret_cast<const fp8e4m3*>(implementationTensor->data);
            TOSA_REF_REQUIRE(impData != nullptr, "[FS] Missing data for implementation");

            return verify(refData, refBndData, impData, elementCount, refShape);
        }
        case tosa_datatype_fp8e5m2_t: {
            const auto* impData = reinterpret_cast<const fp8e5m2*>(implementationTensor->data);
            TOSA_REF_REQUIRE(impData != nullptr, "[FS] Missing data for implementation");

            return verify(refData, refBndData, impData, elementCount, refShape);
        }
        default:
            WARNING("[Verifier][FS] Data-type not supported.");
            break;
    }

    return false;
}
}    // namespace TosaReference

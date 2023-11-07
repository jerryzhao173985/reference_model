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

#include <cmath>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>

#include "verifiers.h"

namespace TosaReference
{

namespace
{
bool tosaCheckULP(float testValue, double referenceValue, double ulpNum)
{
    double errorBound = 0.0;
    if (std::isfinite(referenceValue) && std::abs(referenceValue) != 0.0)
    {
        // Make the sign of the reference value positive
        // and adjust the test value appropriately.
        if (referenceValue < 0)
        {
            referenceValue = -referenceValue;
            testValue      = -testValue;
        }
        // Find the exponent of the reference value.
        int32_t referenceExponent = ilog2(referenceValue);

        // Work out the values magnitude - by raising 2 to the power of the
        // exponent and taking the normalized minimum for denormal values
        const double referencePower2 = std::max(exp2(referenceExponent), AccPrecision<float>::normal_min);
        // Get the value of changing the last bit - by shifting the least significant bit to this magnitude
        // i.e. the ULP.
        double ulpValue = referencePower2 * exp2(-AccPrecision<float>::normal_frac);

        errorBound = ulpValue * ulpNum;
    }
    return tosaCheckFloatBound(testValue, referenceValue, errorBound);
}
}    // namespace

bool verifyULP(const CTensor* referenceTensor, const CTensor* implementationTensor, const UlpInfo& ulpInfo)
{
    // Validate that tensors are provided
    TOSA_REF_REQUIRE(referenceTensor != nullptr, "[ULP] Reference tensor is missing");
    TOSA_REF_REQUIRE(implementationTensor != nullptr, "[ULP] Implementation tensor is missing");

    // Get number of elements
    const std::vector<int32_t> refShape(referenceTensor->shape, referenceTensor->shape + referenceTensor->num_dims);
    const auto elementCount = numElements(refShape);
    TOSA_REF_REQUIRE(elementCount > 0, "[ULP] Invalid shape for reference tensor");

    const double ulp = ulpInfo.ulp;
    switch (implementationTensor->data_type)
    {
        case tosa_datatype_fp32_t: {
            const auto* refData = reinterpret_cast<const double*>(referenceTensor->data);
            TOSA_REF_REQUIRE(refData != nullptr, "[ULP] Missing data for reference");
            const auto* impData = reinterpret_cast<const float*>(implementationTensor->data);
            TOSA_REF_REQUIRE(impData != nullptr, "[ULP] Missing data for implementation");
            const auto* refDataEnd = std::next(refData, elementCount);
            // Use mismatch to get the location of the first unequal value
            auto pair = std::mismatch(refData, refDataEnd, impData, std::next(impData, elementCount),
                                      [ulp](const auto& referenceValue, const auto& implementationValue) {
                                          return tosaCheckULP(implementationValue, referenceValue, ulp);
                                      });
            if (std::get<0>(pair) == refDataEnd)
            {
                // No mismatch found
                return true;
            }
            else
            {
                auto pos = indexToPosition(std::get<0>(pair) - refData, refShape);
                WARNING("[Verfier][ULP] Location %s", positionToString(pos).c_str());
                return false;
            }
        }
        default:
            WARNING("[Verifier][ULP] Data-type not supported.");
            break;
    }

    return false;
}
}    // namespace TosaReference

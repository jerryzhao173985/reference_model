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

#include "verifiers.h"

namespace TosaReference
{

namespace
{
static_assert(std::numeric_limits<float>::is_iec559,
              "TOSA Reference Model has not been built with standard IEE574 32-bit float support; ULP based "
              "verifcation is invalid");
static_assert(std::numeric_limits<double>::is_iec559,
              "TOSA Reference Model has not been built with standard IEE574 64-bit float support; ULP based "
              "verifcation is invalid");

bool tosaCheckULP(double referenceValue, float testValue, double ulpNum)
{

    // Start by sanitizing the input.

    // The concept of ULP isn't defined for NaN's
    if (std::isnan(referenceValue) || std::isnan(testValue))
    {
        return false;
    }

    // Make the sign of the reference value positive
    // and adjust the test value appropriately.
    if (referenceValue < 0)
    {
        referenceValue = -referenceValue;
        testValue      = -testValue;
    }

    // At this point we are ready to calculate the ULP bounds for the reference value.
    double referenceMin, referenceMax;

    // If the reference is infinity e.g. the result of an overflow the test value must
    // be infinity of an appropriate sign.
    if (std::isinf(referenceValue))
    {
        // We already canonicalized the input such that the reference value is positive
        // so no need to check again here.
        referenceMin = std::numeric_limits<float>::infinity();
        referenceMax = std::numeric_limits<float>::infinity();
    }
    else if (referenceValue == 0)
    {
        // For zero we require that the results match exactly with the correct sign.
        referenceMin = 0;
        referenceMax = 0;
    }
    else
    {
        // Find the exponent of the reference value.
        int32_t referenceExponent = ilog2(referenceValue);

        // Work out the values magnitude - by raising 2 to the power of the
        // exponent and taking the normalized minimum for denormal values
        const double referencePower2 = std::max(exp2(referenceExponent), AccPrecision<float>::normal_min);
        // Get the value of changing the last bit - by shifting the least significant bit to this magnitude
        // i.e. the ULP.
        double ulpValue = referencePower2 * exp2(-AccPrecision<float>::normal_frac);

        // Scale by the number of ULPs requested by the user.
        referenceMax = referenceValue + ulpValue * ulpNum;
        referenceMin = referenceValue - ulpValue * ulpNum;

        // Handle the overflow cases.
        if (referenceMax > AccPrecision<float>::normal_max)
        {
            referenceMax = std::numeric_limits<float>::infinity();
        }

        if (referenceMin > AccPrecision<float>::normal_max)
        {
            referenceMin = std::numeric_limits<float>::infinity();
        }

        // And the underflow cases.
        if (referenceMax < AccPrecision<float>::normal_min)
        {
            referenceMax = AccPrecision<float>::normal_min;
        }

        if (referenceMin < AccPrecision<float>::normal_min)
        {
            referenceMin = 0.0;
        }
    }

    // And finally... Do the comparison.
    double testValue64 = static_cast<double>(testValue);
    bool withinUlp     = testValue64 >= referenceMin && testValue64 <= referenceMax;
    if (!withinUlp)
    {
        WARNING("[Verfier][ULP] value (%10f) is not in ULP %g range (%10f <= ref (%10f) <= %10f).", testValue64, ulpNum,
                referenceMin, referenceValue, referenceMax);
    }
    return withinUlp;
}
}    // namespace

bool verifyULP(const CTensor* referenceTensor, const CTensor* implementationTensor, const UlpInfo& ulpInfo)
{
    // Validate that tensors are provided
    TOSA_REF_REQUIRE(referenceTensor != nullptr, "[ULP] Reference tensor is missing");
    TOSA_REF_REQUIRE(implementationTensor != nullptr, "[ULP] Implementation tensor is missing");

    // Get number of elements
    const auto elementCount =
        numElements(std::vector<int32_t>(referenceTensor->shape, referenceTensor->shape + referenceTensor->num_dims));
    TOSA_REF_REQUIRE(elementCount > 0, "[ULP] Invalid shape for reference tensor");

    const double ulp = ulpInfo.ulp;
    switch (implementationTensor->data_type)
    {
        case tosa_datatype_fp32_t: {
            const auto* refData = reinterpret_cast<const double*>(referenceTensor->data);
            TOSA_REF_REQUIRE(refData != nullptr, "[ULP] Missing data for reference");
            const auto* impData = reinterpret_cast<const float*>(implementationTensor->data);
            TOSA_REF_REQUIRE(impData != nullptr, "[ULP] Missing data for implementation");
            return std::equal(refData, std::next(refData, elementCount), impData, std::next(impData, elementCount),
                              [ulp](const auto& referenceValue, const auto& implementationValue) {
                                  return tosaCheckULP(referenceValue, implementationValue, ulp);
                              });
        }
        default:
            WARNING("[Verifier][ULP] Data-type not supported.");
            break;
    }

    return false;
}
}    // namespace TosaReference

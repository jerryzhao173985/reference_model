// Copyright (c) 2023-2024, ARM Limited.
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

#include "half.hpp"
#include "verifiers.h"

namespace TosaReference
{

namespace
{
template <typename OutType>
double calcErrorBound(double referenceValue, double boundsValue, const void* cfgPtr)
{
    const auto cfg = reinterpret_cast<const UlpVerifyInfo*>(cfgPtr);
    unused(boundsValue);

    double errBound = 0.0;
    if (std::isfinite(referenceValue) && std::abs(referenceValue) != 0.0)
    {
        // Find the exponent of the reference value.
        int32_t refExponent = ilog2(std::abs(referenceValue));

        // Work out the values magnitude - by raising 2 to the power of the
        // exponent and taking the normalized minimum for denormal values
        const double refPower2 = std::max(exp2(refExponent), AccPrecision<OutType>::normal_min);
        // Get the value of changing the last bit - by shifting the least significant bit to this magnitude
        // i.e. the ULP.
        double ulpValue = refPower2 * exp2(-AccPrecision<OutType>::normal_frac);

        errBound = ulpValue * cfg->ulp;
    }
    return errBound;
}
}    // namespace

bool verifyULP(const CTensor* referenceTensor, const CTensor* implementationTensor, const UlpVerifyInfo& ulpInfo)
{
    // Validate that tensors are provided
    TOSA_REF_REQUIRE(referenceTensor != nullptr, "[ULP] Reference tensor is missing");
    TOSA_REF_REQUIRE(implementationTensor != nullptr, "[ULP] Implementation tensor is missing");

    const std::vector<int32_t> refShape(referenceTensor->shape, referenceTensor->shape + referenceTensor->num_dims);

    const auto* refData = reinterpret_cast<const double*>(referenceTensor->data);
    TOSA_REF_REQUIRE(refData != nullptr, "[ULP] Missing data for reference");

    const std::string modeStr = "ULP";

    switch (implementationTensor->data_type)
    {
        case tosa_datatype_fp32_t: {
            const auto* impData = reinterpret_cast<const float*>(implementationTensor->data);
            TOSA_REF_REQUIRE(impData != nullptr, "[ULP] Missing data for implementation");
            return validateData(refData, nullptr, impData, refShape, modeStr, &ulpInfo, &calcErrorBound<float>);
        }
        case tosa_datatype_fp16_t: {
            const auto* impData = reinterpret_cast<const half_float::half*>(implementationTensor->data);
            TOSA_REF_REQUIRE(impData != nullptr, "[ULP] Missing data for implementation");
            return validateData(refData, nullptr, impData, refShape, modeStr, &ulpInfo,
                                &calcErrorBound<half_float::half>);
        }
        default:
            WARNING("[Verifier][ULP] Data-type not supported.");
            break;
    }

    return false;
}
}    // namespace TosaReference


// Copyright (c) 2024-2025, ARM Limited.
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
#include <vector>

#include "verifiers.h"
#include "verify_utils.h"

namespace TosaReference
{

namespace
{
template <typename OutType>
ErrorBoundsRange calcErrorBounds(double referenceValue, double boundsValue, const void* cfgPtr)
{
    const auto cfg = reinterpret_cast<const RelativeVerifyInfo*>(cfgPtr);
    unused(boundsValue);

    double ulpValue = 0.0;
    if (std::isfinite(referenceValue) && std::abs(referenceValue) != 0.0)
    {
        // Work out the values magnitude - by raising 2 to the power of the
        // exponent
        const double refPower2 = std::max(exp2(ilog2(std::abs(referenceValue))), AccPrecision<OutType>::normal_min);
        // Get the value of changing the last bit - by shifting the least significant bit to this magnitude
        // i.e. the ULP.
        ulpValue = refPower2 * exp2(-AccPrecision<OutType>::normal_frac);
    }

    // For some cases the relative bound is too tight for dtypes like bf16 resize
    // Set an alternative bound for the error_bound to be cfg->ulpBound * ulpValue
    double error_bound = std::max(cfg->max * cfg->scale, cfg->ulpBound * ulpValue);

    return { error_bound, error_bound };
}
}    // namespace

bool verifyRelative(const CTensor* referenceTensor,
                    const CTensor* implementationTensor,
                    const RelativeVerifyInfo& rInfo)
{
    // Validate that tensors are provided
    TOSA_REF_REQUIRE(referenceTensor != nullptr, "[R] Reference tensor is missing");
    TOSA_REF_REQUIRE(implementationTensor != nullptr, "[R] Implementation tensor is missing");

    const std::vector<int32_t> refShape(referenceTensor->shape, referenceTensor->shape + referenceTensor->num_dims);

    const double* refData = reinterpret_cast<const double*>(referenceTensor->data);
    TOSA_REF_REQUIRE(refData != nullptr, "[R] Missing data for reference");

    const std::string modeStr = "R";

    switch (implementationTensor->data_type)
    {
        case tosa_datatype_fp32_t: {
            const auto* impData = reinterpret_cast<const float*>(implementationTensor->data);
            TOSA_REF_REQUIRE(impData != nullptr, "[R] Missing data for implementation");
            return validateData(refData, nullptr, impData, refShape, modeStr, &rInfo, &calcErrorBounds<float>);
        }
        case tosa_datatype_bf16_t: {
            const auto* impData = reinterpret_cast<const bf16*>(implementationTensor->data);
            TOSA_REF_REQUIRE(impData != nullptr, "[R] Missing data for implementation");
            return validateData(refData, nullptr, impData, refShape, modeStr, &rInfo, &calcErrorBounds<bf16>);
        }
        case tosa_datatype_fp8e4m3_t: {
            const auto* impData = reinterpret_cast<const fp8e4m3*>(implementationTensor->data);
            TOSA_REF_REQUIRE(impData != nullptr, "[R] Missing data for implementation");
            return validateData(refData, nullptr, impData, refShape, modeStr, &rInfo, &calcErrorBounds<fp8e4m3>);
        }
        case tosa_datatype_fp8e5m2_t: {
            const auto* impData = reinterpret_cast<const fp8e5m2*>(implementationTensor->data);
            TOSA_REF_REQUIRE(impData != nullptr, "[R] Missing data for implementation");
            return validateData(refData, nullptr, impData, refShape, modeStr, &rInfo, &calcErrorBounds<fp8e5m2>);
        }
        case tosa_datatype_fp16_t: {
            const auto* impData = reinterpret_cast<const float16*>(implementationTensor->data);
            TOSA_REF_REQUIRE(impData != nullptr, "[R] Missing data for implementation");
            return validateData(refData, nullptr, impData, refShape, modeStr, &rInfo, &calcErrorBounds<float16>);
        }
        default:
            WARNING("[Verifier][R] Data-type not supported.");
            break;
    }

    return false;
}
}    // namespace TosaReference

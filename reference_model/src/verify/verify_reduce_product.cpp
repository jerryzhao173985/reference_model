
// Copyright (c) 2023-2025, ARM Limited.
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
    const auto cfg = reinterpret_cast<const ReduceProductVerifyInfo*>(cfgPtr);
    unused(boundsValue);

    // ULPs for subnormal values are the ULP of the normal minimum
    const double referenceMagnitude = std::max(std::abs(referenceValue), double(std::numeric_limits<OutType>::min()));
    double bValue                   = referenceMagnitude *
                    (std::pow(1 + std::pow(2, -AccPrecision<OutType>::normal_frac - 1), cfg->numberOfProducts) - 1);
    return { bValue, bValue };
}
}    // namespace

bool verifyReduceProduct(const CTensor* referenceTensor,
                         const CTensor* implementationTensor,
                         const ReduceProductVerifyInfo& rpInfo)
{
    // Validate that tensors are provided
    TOSA_REF_REQUIRE(referenceTensor != nullptr, "[RP] Reference tensor is missing");
    TOSA_REF_REQUIRE(implementationTensor != nullptr, "[RP] Implementation tensor is missing");

    const std::vector<int32_t> refShape(referenceTensor->shape, referenceTensor->shape + referenceTensor->num_dims);

    const double* refData = reinterpret_cast<const double*>(referenceTensor->data);
    TOSA_REF_REQUIRE(refData != nullptr, "[RP] Missing data for reference");

    const std::string modeStr = "RP";

    switch (implementationTensor->data_type)
    {
        case tosa_datatype_fp32_t: {
            const auto* impData = reinterpret_cast<const float*>(implementationTensor->data);
            TOSA_REF_REQUIRE(impData != nullptr, "[RP] Missing data for implementation");
            return validateData(refData, nullptr, impData, refShape, modeStr, &rpInfo, &calcErrorBounds<float>);
        }
        case tosa_datatype_fp16_t: {
            const auto* impData = reinterpret_cast<const float16*>(implementationTensor->data);
            TOSA_REF_REQUIRE(impData != nullptr, "[RP] Missing data for implementation");
            return validateData(refData, nullptr, impData, refShape, modeStr, &rpInfo, &calcErrorBounds<float16>);
        }
        case tosa_datatype_bf16_t: {
            const auto* impData = reinterpret_cast<const bf16*>(implementationTensor->data);
            TOSA_REF_REQUIRE(impData != nullptr, "[RP] Missing data for implementation");
            return validateData(refData, nullptr, impData, refShape, modeStr, &rpInfo, &calcErrorBounds<bf16>);
        }
        case tosa_datatype_fp8e4m3_t: {
            const auto* impData = reinterpret_cast<const fp8e4m3*>(implementationTensor->data);
            TOSA_REF_REQUIRE(impData != nullptr, "[RP] Missing data for implementation");
            return validateData(refData, nullptr, impData, refShape, modeStr, &rpInfo, &calcErrorBounds<fp8e4m3>);
        }
        case tosa_datatype_fp8e5m2_t: {
            const auto* impData = reinterpret_cast<const fp8e5m2*>(implementationTensor->data);
            TOSA_REF_REQUIRE(impData != nullptr, "[RP] Missing data for implementation");
            return validateData(refData, nullptr, impData, refShape, modeStr, &rpInfo, &calcErrorBounds<fp8e5m2>);
        }
        default:
            WARNING("[Verifier][RP] Data-type not supported.");
            break;
    }

    return false;
}
}    // namespace TosaReference

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

#include "func_debug.h"
#include "half.hpp"
#include "verifiers.h"

namespace TosaReference
{

namespace
{
// TODO: Document the pre-conditions and expectations from arguments to this function
// Similarly, write about how `boundAsMagnitude` is meant to be understood
template <typename OutType>
double calcErrorBound(double referenceValue, double boundsValue, const void* cfgPtr)
{
    const auto cfg = reinterpret_cast<const AbsErrorVerifyInfo*>(cfgPtr);
    ASSERT_MSG(cfg != nullptr, "AbsErrorVerifyInfo is nullptr in calcErrorBound");

    double boundsMagnitude{ 0.0 };
    if (cfg->boundAsMagnitude)
    {
        // Special case for SIN/COS
        // use the input value (stored in the bounds tensor) as the magnitude and value
        boundsMagnitude = boundsValue;
        boundsValue     = 1.0;
    }
    else
    {
        // Use the referenceValue as the magnitude
        boundsMagnitude = referenceValue;
        // Add the base error bound to the bounds value
        boundsValue += cfg->baseBound;
    }

    double errorBound = 0.0;
    if (std::isfinite(boundsValue) || std::abs(boundsMagnitude) != 0.0)
    {
        // ULPs for subnormal values are the ULP of the normal minimum
        double valueBound = std::max(std::abs(boundsMagnitude), double(std::numeric_limits<OutType>::min()));

        if (cfg->lowerBound > 0)
        {
            valueBound = std::max(cfg->lowerBound / boundsValue, valueBound);
        }
        errorBound = exp2(static_cast<int32_t>(-AccPrecision<OutType>::normal_frac / cfg->normalDivisor)) * valueBound;
        errorBound *= boundsValue;
    }
    // TODO: If this function proves to no longer be generic,
    // calculate errorBound in this function and introduce
    // calcAbsErrorBound for all verify_*_error.cc variants
    return cfg->maxCompare > 0.0 ? std::max(errorBound, (cfg->maxCompare)) : errorBound;
}
}    // namespace

bool verifyAbsError(const CTensor* referenceTensor,
                    const CTensor* boundsTensor,
                    const CTensor* implementationTensor,
                    const AbsErrorVerifyInfo& aeInfo)
{
    // Validate that tensors are provided
    TOSA_REF_REQUIRE(referenceTensor != nullptr, "[AE] Reference tensor is missing");
    TOSA_REF_REQUIRE(boundsTensor != nullptr, "[AE] Reference bounds tensor is missing");
    TOSA_REF_REQUIRE(implementationTensor != nullptr, "[AE] Implementation tensor is missing");

    const std::vector<int32_t> refShape(referenceTensor->shape, referenceTensor->shape + referenceTensor->num_dims);

    const double* refData    = reinterpret_cast<const double*>(referenceTensor->data);
    const double* refBndData = reinterpret_cast<const double*>(boundsTensor->data);
    TOSA_REF_REQUIRE(refData != nullptr && refBndData != nullptr, "[AE] Missing data for reference or bounds tensors");

    const std::string modeStr = "AE";

    switch (implementationTensor->data_type)
    {
        case tosa_datatype_fp32_t: {
            const auto* impData = reinterpret_cast<const float*>(implementationTensor->data);
            TOSA_REF_REQUIRE(impData != nullptr, "[AE] Missing data for implementation");
            return validateData(refData, refBndData, impData, refShape, modeStr, &aeInfo, &calcErrorBound<float>);
        }
        case tosa_datatype_fp16_t: {
            const auto* impData = reinterpret_cast<const half_float::half*>(implementationTensor->data);
            TOSA_REF_REQUIRE(impData != nullptr, "[AE] Missing data for implementation");
            return validateData(refData, refBndData, impData, refShape, modeStr, &aeInfo,
                                &calcErrorBound<half_float::half>);
        }
        case tosa_datatype_bf16_t: {
            const auto* impData = reinterpret_cast<const bf16*>(implementationTensor->data);
            TOSA_REF_REQUIRE(impData != nullptr, "[AE] Missing data for implementation");
            return validateData(refData, refBndData, impData, refShape, modeStr, &aeInfo, &calcErrorBound<bf16>);
        }
        case tosa_datatype_fp8e4m3_t: {
            const auto* impData = reinterpret_cast<const fp8e4m3*>(implementationTensor->data);
            TOSA_REF_REQUIRE(impData != nullptr, "[AE] Missing data for implementation");
            return validateData(refData, refBndData, impData, refShape, modeStr, &aeInfo, &calcErrorBound<fp8e4m3>);
        }
        case tosa_datatype_fp8e5m2_t: {
            const auto* impData = reinterpret_cast<const fp8e5m2*>(implementationTensor->data);
            TOSA_REF_REQUIRE(impData != nullptr, "[AE] Missing data for implementation");
            return validateData(refData, refBndData, impData, refShape, modeStr, &aeInfo, &calcErrorBound<fp8e5m2>);
        }
        default:
            WARNING("[Verifier][AE] Data-type not supported.");
            break;
    }

    return false;
}
}    // namespace TosaReference

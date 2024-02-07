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
    const auto cfg = reinterpret_cast<const AbsErrorVerifyInfo*>(cfgPtr);

    double errorBound = 0.0;
    if (std::isfinite(referenceValue) && std::abs(referenceValue) != 0.0)
    {
        double valBound = std::abs(referenceValue) * boundsValue;
        if (cfg->lowerBound > 0)
        {
            valBound = std::max(cfg->lowerBound, valBound);
        }
        errorBound = exp2(-AccPrecision<OutType>::normal_frac) * valBound;
    }
    return errorBound;
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
        default:
            WARNING("[Verifier][AE] Data-type not supported.");
            break;
    }

    return false;
}
}    // namespace TosaReference

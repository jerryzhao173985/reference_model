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
double calcErrorBound(double referenceValue, double boundsValue, const void* cfgPtr)
{
    return 0.0;
}

template <typename OutDtype>
bool exact_fp(const double& referenceValue, const OutDtype& implementationValue)
{
    return std::isnan(referenceValue) ? std::isnan(implementationValue) : (referenceValue == implementationValue);
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
    const std::vector<int32_t> refShape(referenceTensor->shape, referenceTensor->shape + referenceTensor->num_dims);
    const auto elementCount = numElements(refShape);
    TOSA_REF_REQUIRE(elementCount > 0, "[E] Invalid shape for reference tensor");

    TOSA_REF_REQUIRE(referenceTensor->data_type == tosa_datatype_fp64_t, "[E] Reference tensor is not fp64");
    const auto* refData = reinterpret_cast<const double*>(referenceTensor->data);
    TOSA_REF_REQUIRE(refData != nullptr, "[E] Missing data for reference");

    const std::string modeStr = "E";

    switch (implementationTensor->data_type)
    {
        case tosa_datatype_fp32_t: {
            const auto* impData = reinterpret_cast<const float*>(implementationTensor->data);
            TOSA_REF_REQUIRE(impData != nullptr, "[E] Missing data for implementation");
            return validateData(refData, nullptr, impData, refShape, modeStr, nullptr, &calcErrorBound);
        }
        case tosa_datatype_fp16_t: {
            const auto* impData = reinterpret_cast<const half_float::half*>(implementationTensor->data);
            TOSA_REF_REQUIRE(impData != nullptr, "[E] Missing data for implementation");
            return validateData(refData, nullptr, impData, refShape, modeStr, nullptr, &calcErrorBound);
        }
        default:
            WARNING("[Verifier][E] Data-type not supported.");
            break;
    }

    return false;
}
}    // namespace TosaReference

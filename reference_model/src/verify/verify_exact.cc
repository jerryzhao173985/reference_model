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
#include "verifiers.h"
#include <cmath>

namespace TosaReference
{

bool verifyExact(const CTensor* referenceTensor, const CTensor* implementationTensor)
{
    // Validate that tensors are provided
    TOSA_REF_REQUIRE(referenceTensor != nullptr, "reference tensor is missing");
    TOSA_REF_REQUIRE(implementationTensor != nullptr, "implementation tensor is missing");

    // Get number of elements
    const auto elementCount =
        numElements(std::vector<int32_t>(referenceTensor->shape, referenceTensor->shape + referenceTensor->num_dims));
    TOSA_REF_REQUIRE(elementCount > 0, "invalid shape for reference tensor");

    switch (implementationTensor->data_type)
    {
        case tosa_datatype_fp32_t: {
            const auto* refData = reinterpret_cast<const float*>(referenceTensor->data);
            TOSA_REF_REQUIRE(refData != nullptr, "missing data for reference");
            const auto* impData = reinterpret_cast<const float*>(implementationTensor->data);
            TOSA_REF_REQUIRE(impData != nullptr, "missing data for implementation");
            return std::equal(refData, std::next(refData, elementCount), impData, std::next(impData, elementCount),
                              [](const auto& referenceValue, const auto& implementationValue) {
                                  return std::isnan(referenceValue) ? std::isnan(implementationValue)
                                                                    : (referenceValue == implementationValue);
                              });
        }
        default:
            WARNING("data-type not supported.");
            break;
    }

    return false;
}
}    // namespace TosaReference


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
#include <vector>

#include "verifiers.h"
#include "verify/verify_utils.h"

namespace
{

auto calculateError(uint64_t M, uint64_t N)
{
    return std::pow(1 + std::pow(2, -static_cast<int64_t>(M) - 1), N) - 1;
}

template <typename FP>
auto calculateTolerance(uint64_t M, uint64_t N, FP value)
{
    return std::abs(value) * calculateError(M, N);
}
}    // namespace

namespace TosaReference
{

bool verifyReduceProduct(const CTensor* referenceTensor, const CTensor* implementationTensor, uint64_t m, uint64_t n)
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
                              [m, n](const auto& referenceValue, const auto& implementationValue) {
                                  // Result overflows must be set to zero of the correct sign.
                                  if (std::isinf(implementationValue))
                                  {
                                      return implementationValue == referenceValue;
                                  }

                                  // Result underflows must be set to a zero of the correct sign.
                                  if (implementationValue == 0.f || implementationValue == -0.f)
                                  {
                                      return implementationValue == referenceValue;
                                  }

                                  // Otherwise we are in the normal range.
                                  const auto absoulteError = (referenceValue < implementationValue)
                                                                 ? implementationValue - referenceValue
                                                                 : referenceValue - implementationValue;
                                  return absoulteError <= calculateTolerance(m, n, implementationValue);
                              });
        }
        default:
            WARNING("tosa verifier: data-type not supported.");
            break;
    }

    return false;
}
}    // namespace TosaReference

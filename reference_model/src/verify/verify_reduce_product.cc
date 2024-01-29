
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
#include <vector>

#include "verifiers.h"
#include "verify/verify_utils.h"

namespace TosaReference
{

namespace
{
template <typename OutDtype>
bool validateData(const double* ref,
                  const OutDtype* imp,
                  const std::vector<int32_t>& shape,
                  const ReduceProductVerifyInfo& cfg)
{
    const size_t T = static_cast<size_t>(numElements(shape));
    TOSA_REF_REQUIRE(T > 0, "[RP] Invalid shape for reference tensor");

    for (size_t i = 0; i < T; ++i)
    {
        double errBound =
            std::abs(ref[i]) * (std::pow(1 + std::pow(2, -AccPrecision<OutDtype>::normal_frac - 1), cfg.n) - 1);
        bool valid = tosaCheckFloatBound(imp[i], ref[i], errBound);
        if (!valid)
        {
            auto pos = indexToPosition(i, shape);
            WARNING("[Verifier][RP] Location %s", positionToString(pos).c_str());
            return false;
        }
    }
    return true;
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

    switch (implementationTensor->data_type)
    {
        case tosa_datatype_fp32_t: {
            const auto* impData = reinterpret_cast<const float*>(implementationTensor->data);
            TOSA_REF_REQUIRE(impData != nullptr, "[RP] Missing data for implementation");
            return validateData(refData, impData, refShape, rpInfo);
        }
        case tosa_datatype_fp16_t: {
            const auto* impData = reinterpret_cast<const half_float::half*>(implementationTensor->data);
            TOSA_REF_REQUIRE(impData != nullptr, "[RP] Missing data for implementation");
            return validateData(refData, impData, refShape, rpInfo);
        }
        default:
            WARNING("[Verifier][RP] Data-type not supported.");
            break;
    }

    return false;
}
}    // namespace TosaReference

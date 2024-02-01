
// Copyright (c) 2024, ARM Limited.
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
                  const RelativeVerifyInfo& cfg)
{
    const size_t T = static_cast<size_t>(numElements(shape));
    TOSA_REF_REQUIRE(T > 0, "[R] Invalid shape for reference tensor");

    double errBound = cfg.max * cfg.scale;
    for (size_t i = 0; i < T; ++i)
    {
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

    switch (implementationTensor->data_type)
    {
        case tosa_datatype_fp32_t: {
            const auto* impData = reinterpret_cast<const float*>(implementationTensor->data);
            TOSA_REF_REQUIRE(impData != nullptr, "[R] Missing data for implementation");
            return validateData(refData, impData, refShape, rInfo);
        }
        case tosa_datatype_fp16_t: {
            const auto* impData = reinterpret_cast<const half_float::half*>(implementationTensor->data);
            TOSA_REF_REQUIRE(impData != nullptr, "[R] Missing data for implementation");
            return validateData(refData, impData, refShape, rInfo);
        }
        default:
            WARNING("[Verifier][R] Data-type not supported.");
            break;
    }

    return false;
}
}    // namespace TosaReference

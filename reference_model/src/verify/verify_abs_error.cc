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
template <typename OutDtype>
bool validateData(const double* ref, const double* bnd, const OutDtype* imp, const std::vector<int32_t>& shape)
{
    const size_t T = static_cast<size_t>(numElements(shape));
    TOSA_REF_REQUIRE(T > 0, "[AE] Invalid shape for reference tensor");

    for (size_t i = 0; i < T; ++i)
    {
        double errBound = std::abs(ref[i]) * exp2(-AccPrecision<OutDtype>::normal_frac) * bnd[i];
        bool valid      = tosaCheckFloatBound(imp[i], ref[i], errBound);
        if (!valid)
        {
            auto pos = indexToPosition(T, shape);
            WARNING("[Verifier][AE] Location %s", positionToString(pos).c_str());
            return false;
        }
    }
    return true;
}
}    // namespace
bool verifyAbsError(const CTensor* ref, const CTensor* refBnd, const CTensor* imp)
{
    // Validate that tensors are provided
    TOSA_REF_REQUIRE(ref != nullptr, "[AE] Reference tensor is missing");
    TOSA_REF_REQUIRE(refBnd != nullptr, "[AE] Reference bounds tensor is missing");
    TOSA_REF_REQUIRE(imp != nullptr, "[AE] Implementation tensor is missing");

    const std::vector<int32_t> refShape(ref->shape, ref->shape + ref->num_dims);

    const double* refData    = reinterpret_cast<const double*>(ref->data);
    const double* refBndData = reinterpret_cast<const double*>(refBnd->data);
    TOSA_REF_REQUIRE(refData != nullptr && refBndData != nullptr, "[AE] Missing data for reference or bounds tensors");

    switch (imp->data_type)
    {
        case tosa_datatype_fp32_t: {
            const auto* impData = reinterpret_cast<const float*>(imp->data);
            TOSA_REF_REQUIRE(impData != nullptr, "[AE] Missing data for implementation");
            return validateData(refData, refBndData, impData, refShape);
        }
        case tosa_datatype_fp16_t: {
            const auto* impData = reinterpret_cast<const half_float::half*>(imp->data);
            TOSA_REF_REQUIRE(impData != nullptr, "[AE] Missing data for implementation");
            return validateData(refData, refBndData, impData, refShape);
        }
        default:
            WARNING("[Verifier][AE] Data-type not supported.");
            break;
    }

    return false;
}
}    // namespace TosaReference

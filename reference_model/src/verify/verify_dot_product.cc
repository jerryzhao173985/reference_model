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

#include "func_debug.h"
#include "half.hpp"
#include "verifiers.h"

#include <cfloat>
#include <cmath>
#include <numeric>
#include <optional>
#include <type_traits>

namespace TosaReference
{
namespace
{
// Generic element validation function
template <typename AccType>
std::optional<double> validateElement(size_t index, double ref, double bnd, AccType imp, size_t KS)
{
    double err    = 0.0;
    bool is_valid = true;

    if (std::isinf(static_cast<AccType>(bnd * (1 + exp2(-1 - AccPrecision<AccType>::normal_frac)))))
    {
        // dot product can overflow and there is no accuracy limit
        is_valid = true;
        err      = 0.0;
    }
    else if (bnd == 0.0)
    {
        is_valid = (ref == 0.0) && (static_cast<double>(imp) == 0.0);
        if (!is_valid)
        {
            WARNING("[Verifier][DP] index %d: bound is zero, but ref (%.*g) or imp (%.*g) is not.", index, DBL_DIG, ref,
                    FLT_DIG, static_cast<double>(imp));
        }
        err = 0.0;
    }
    else
    {
        // 0.0 < bnd < infinity
        const double out_err_bnd =
            std::max(bnd * exp2(-1 - AccPrecision<AccType>::normal_frac), AccPrecision<AccType>::normal_min);
        const double imp_fp64 = static_cast<double>(imp);
        err                   = (imp_fp64 - ref) / out_err_bnd;
        is_valid              = std::abs(err) <= KS;
        if (!is_valid)
        {
            WARNING("[Verifier][DP] index %d: out_err (abs(%.*g)) is not within KS (%d).", index, DBL_DIG, err, KS);
        }
    }

    return is_valid ? std::optional(err) : std::nullopt;
}

// Dot Product data validation function
template <typename AccType>
bool validateDataDP(const double* referenceData,
                    const double* boundsData,
                    const AccType* implementationData,
                    const std::vector<int32_t>& shape,
                    const DotProductVerifyInfo& cfg)
{
    const size_t T = static_cast<size_t>(numElements(shape));
    TOSA_REF_REQUIRE(T > 0, "[DP] Invalid shape for reference tensor");

    const int32_t S = cfg.setNumber;
    // NOTE: KS in the compliance config MUST have already been updated to (KS + 1) if the bias
    // tensor is non-zero
    const int32_t KS = cfg.kernelSize;

    double out_err_sum   = 0.0;
    double out_err_sumsq = 0.0;

    for (size_t i = 0; i < T; ++i)
    {
        auto out_err = validateElement<AccType>(i, referenceData[i], boundsData[i], implementationData[i], KS);
        if (!out_err)
        {
            auto pos = indexToPosition(i, shape);
            TOSA_REF_REQUIRE(out_err, "[DP] Location %s: Data required to be zero or error within range",
                             positionToString(pos).c_str());
        }
        out_err_sum += out_err.value();
        out_err_sumsq += out_err.value() * out_err.value();
    }

    if (S >= 3 && S <= 5)
    {
        const double max_bias = 2 * sqrt(KS * T);
        // Check error bias magnitude for data sets S which are not positive biased
        TOSA_REF_REQUIRE(std::abs(out_err_sum) <= max_bias, "[DP] Bias magnitude (abs(%.*g)) is out of range (%.*g)",
                         DBL_DIG, out_err_sum, DBL_DIG, max_bias);
    }
    // Check error variance magnitude
    const double max_error = 0.4 * KS * T;
    TOSA_REF_REQUIRE(out_err_sumsq <= max_error, "[DP] Error variance magnitude (%.*g) is out of range (%.*g)", DBL_DIG,
                     out_err_sumsq, DBL_DIG, max_error);
    return true;
}
}    // namespace

bool verifyDotProduct(const CTensor* referenceTensor,
                      const CTensor* boundsTensor,
                      const CTensor* implementationTensor,
                      const DotProductVerifyInfo& dpInfo)
{
    // Validate that tensors are provided
    TOSA_REF_REQUIRE(referenceTensor != nullptr, "[DP] Reference tensor is missing");
    TOSA_REF_REQUIRE(boundsTensor != nullptr, "[DP] Reference bounds tensor is missing");
    TOSA_REF_REQUIRE(implementationTensor != nullptr, "[DP] Implementation tensor is missing");

    const std::vector<int32_t> refShape(referenceTensor->shape, referenceTensor->shape + referenceTensor->num_dims);

    const double* refData    = reinterpret_cast<const double*>(referenceTensor->data);
    const double* refBndData = reinterpret_cast<const double*>(boundsTensor->data);
    TOSA_REF_REQUIRE(refData != nullptr && refBndData != nullptr, "[DP] Missing data for reference or bounds tensors");

    switch (implementationTensor->data_type)
    {
        case tosa_datatype_fp32_t: {
            const float* impData = reinterpret_cast<const float*>(implementationTensor->data);
            TOSA_REF_REQUIRE(impData != nullptr, "[DP] Missing data for implementation");
            return validateDataDP(refData, refBndData, impData, refShape, dpInfo);
            break;
        }
        case tosa_datatype_fp16_t: {
            const half_float::half* impData = reinterpret_cast<const half_float::half*>(implementationTensor->data);
            TOSA_REF_REQUIRE(impData != nullptr, "[DP] Missing data for implementation");
            return validateDataDP(refData, refBndData, impData, refShape, dpInfo);
            break;
        }
        case tosa_datatype_bf16_t: {
            const bf16* impData = reinterpret_cast<const bf16*>(implementationTensor->data);
            TOSA_REF_REQUIRE(impData != nullptr, "[DP] Missing data for implementation");
            return validateDataDP(refData, refBndData, impData, refShape, dpInfo);
            break;
        }
        case tosa_datatype_fp8e4m3_t: {
            const fp8e4m3* impData = reinterpret_cast<const fp8e4m3*>(implementationTensor->data);
            TOSA_REF_REQUIRE(impData != nullptr, "[DP] Missing data for implementation");
            return validateDataDP(refData, refBndData, impData, refShape, dpInfo);
            break;
        }
        case tosa_datatype_fp8e5m2_t: {
            const fp8e5m2* impData = reinterpret_cast<const fp8e5m2*>(implementationTensor->data);
            TOSA_REF_REQUIRE(impData != nullptr, "[DP] Missing data for implementation");
            return validateDataDP(refData, refBndData, impData, refShape, dpInfo);
            break;
        }
        default: {
            WARNING("[Verifier][DP] Data-type not supported.");
            break;
        }
    }

    return false;
}

}    // namespace TosaReference

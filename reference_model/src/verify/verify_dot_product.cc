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
#include <numeric>
#include <optional>
#include <type_traits>

namespace TosaReference
{
namespace
{

// Accumulator precision
template <typename T>
struct AccPrecision;
#define two_m42 1.0 / (double)(((int64_t)1) << 42)    // 2^-42
template <>
struct AccPrecision<float>
{
    static constexpr double precision  = (double)(1 << 24);
    static constexpr double min_normal = two_m42 * two_m42 * two_m42;    // 2^-126
};
#undef two_m42

// Generic element validation function
template <typename AccType, typename std::enable_if_t<std::is_floating_point_v<AccType>, int> = 0>
std::optional<double> validateElement(double ref, double bnd, AccType imp, size_t KS)
{
    double err    = 0.0;
    bool is_valid = true;

    if (bnd == 0.0)
    {
        is_valid = (ref == 0.0) && (imp == 0.0);
        err      = 0.0;
    }
    else if (std::isinf(static_cast<AccType>(bnd)))
    {
        // dot product can overflow and there is no accuracy limit
        is_valid = true;
        err      = 0.0;
    }
    else
    {
        // 0.0 < bnd < infinity
        const double bnd_norm      = std::max(bnd, AccPrecision<AccType>::min_normal);
        const double imp_fp64      = static_cast<double>(imp);
        const double acc_prec_fp64 = AccPrecision<AccType>::precision;
        err                        = (imp_fp64 - ref) * acc_prec_fp64 / bnd_norm;
        is_valid                   = std::abs(err) <= KS;
    }

    return is_valid ? std::optional(err) : std::nullopt;
}

// Generic data validation function
template <typename AccType, typename std::enable_if_t<std::is_floating_point_v<AccType>, int> = 0>
bool validateData(const double* ref, const double* bnd, const AccType* imp, size_t T, const DotProductVerifyInfo& cfg)
{
    const int32_t S = cfg.s;
    // TODO - needed for other ops - (max_value(bias_abs) > 0) ? (KS + 1) : KS
    const int32_t KS = cfg.ks;

    double out_err_sum   = 0.0;
    double out_err_sumsq = 0.0;

    for (size_t i = 0; i < T; ++i)
    {
        auto out_err = validateElement<AccType>(ref[i], bnd[i], imp[i], KS);
        TOSA_REF_REQUIRE(out_err, "data required to be zero or error within range");
        out_err_sum += out_err.value();
        out_err_sumsq += out_err.value() * out_err.value();
    }

    if (S >= 3 && S <= 5)
    {
        // Check error bias magnitude for data sets S which are not positive biased
        TOSA_REF_REQUIRE(std::abs(out_err_sum) <= 2 * sqrt(KS * T), "bias magnitude is out of range");
    }
    // Check error variance magnitude
    TOSA_REF_REQUIRE(out_err_sumsq <= 0.4 * KS * T, "error variance magnitude is out of range");
    return true;
}
}    // namespace

bool verifyDotProduct(const CTensor* ref, const CTensor* refBnd, const CTensor* imp, const DotProductVerifyInfo& dpInfo)
{
    // Validate that tensors are provided
    TOSA_REF_REQUIRE(ref != nullptr, "reference tensor is missing");
    TOSA_REF_REQUIRE(refBnd != nullptr, "reference bounds tensor is missing");
    TOSA_REF_REQUIRE(imp != nullptr, "implementation tensor is missing");

    // Validate data-type
    TOSA_REF_REQUIRE(dpInfo.dataType == mapToDType(imp->data_type), "invalid data type in config");

    // Get number of dot-product elements
    const int64_t T = numElements(std::vector<int32_t>(ref->shape, ref->shape + ref->num_dims));
    TOSA_REF_REQUIRE(T > 0, "invalid shape for reference tensor");

    const double* refData    = reinterpret_cast<const double*>(ref->data);
    const double* refBndData = reinterpret_cast<const double*>(refBnd->data);
    TOSA_REF_REQUIRE(refData != nullptr && refBndData != nullptr, "missing data for reference or bounds tensors");

    switch (imp->data_type)
    {
        case tosa_datatype_fp32_t: {
            const float* impData = reinterpret_cast<const float*>(imp->data);
            TOSA_REF_REQUIRE(impData != nullptr, "missing data for implementation");
            return validateData(refData, refBndData, impData, static_cast<size_t>(T), dpInfo);
            break;
        }
        default:
            break;
    }

    return false;
}

}    // namespace TosaReference

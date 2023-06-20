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
//===----------------------------------------------------------------------===//
//
// Verification functionality as per TOSA Specification
// Output Verification : Section 1.8.2
//
//===----------------------------------------------------------------------===//

#include "verify.h"

#include <half.hpp>

#include <cmath>
#include <numeric>
#include <optional>
#include <type_traits>

#define REQUIRE(COND)                                                                                                  \
    if (!(COND))                                                                                                       \
    {                                                                                                                  \
        return false;                                                                                                  \
    }

namespace
{
// Accumulator precision
template <typename T>
struct AccPrecision;
template <>
struct AccPrecision<float>
{
    static constexpr double precision = (double)(1 << 24);
};
template <>
struct AccPrecision<half_float::half>
{
    static constexpr double precision = (double)(1 << 11);
};

// Generic element validation function
template <typename AccType, typename std::enable_if_t<std::is_floating_point_v<AccType>, int> = 0>
std::optional<double> validate_element(double ref, double bnd, AccType imp, size_t KS)
{
    double err    = 0.0;
    bool is_valid = true;

    if (bnd == 0.0)
    {
        is_valid = (ref == 0.0) && (imp == 0.0);
        err      = 0.0;
    }
    else
    {    // bnd > 0.0
        const double imp_fp64      = static_cast<double>(imp);
        const double acc_prec_fp64 = AccPrecision<AccType>::precision;
        err                        = (imp_fp64 - ref) * acc_prec_fp64 / bnd;
        is_valid                   = std::abs(err) <= KS;
    }

    return is_valid ? std::optional(err) : std::nullopt;
}

// Generic data validation function
template <typename AccType, typename std::enable_if_t<std::is_floating_point_v<AccType>, int> = 0>
bool validate_data(const double* ref, const double* bnd, const AccType* imp, size_t T, size_t KS, int32_t S)
{
    double out_err_sum   = 0.0;
    double out_err_sumsq = 0.0;

    for (size_t i = 0; i < T; ++i)
    {
        auto out_err = validate_element<AccType>(ref[i], bnd[i], imp[i], KS);
        REQUIRE(out_err);
        out_err_sum += out_err.value();
        out_err_sumsq += out_err.value() * out_err.value();
    }

    return tosa_validate_output_error(out_err_sum, out_err_sumsq, T, KS, S);
}

// Convert std::optional to CheckResult
CheckResult from_optional(const std::optional<double>& res)
{
    if (res)
        return { true, *res };
    else
        return { false, 0.0 };
}
}    // namespace

extern "C" {

CheckResult tosa_validate_element_accfp32(double ref, double bnd, float imp, size_t KS)
{
    auto err = validate_element<float>(ref, bnd, imp, KS);
    return from_optional(err);
}

bool tosa_validate_output_error(double err_sum, double err_sum_sq, size_t T, size_t KS, int S)
{
    if (S != 1 && S != 2)
    {
        // Check error bias magnitude for data sets S which are not positive biased
        REQUIRE(abs(err_sum) <= 2 * sqrt(KS * T));
    }
    // Check error variance magnitude
    REQUIRE(err_sum_sq <= 0.4 * KS * T);

    return true;
}

bool tosa_validate_data_fp32(const double* ref, const double* bnd, const float* imp, size_t T, size_t KS, int S)
{
    return validate_data<float>(ref, bnd, imp, T, KS, S);
}

} // extern "C"
#undef REQUIRE
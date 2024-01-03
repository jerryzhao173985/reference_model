
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

#ifndef VERIFY_UTILS_H_
#define VERIFY_UTILS_H_

#include "dtype.h"
#include "half.hpp"
#include "types.h"

#include <cstdint>
#include <optional>
#include <vector>

#define TOSA_REF_REQUIRE(COND, MESSAGE, ...)                                                                           \
    if (!(COND))                                                                                                       \
    {                                                                                                                  \
        WARNING("[Verifier]" MESSAGE ".", ##__VA_ARGS__);                                                              \
        return false;                                                                                                  \
    }

namespace TosaReference
{

// Name alias
using CTensor = tosa_tensor_t;

/// \brief Supported verification modes
enum class VerifyMode
{
    Unknown,
    Exact,
    Ulp,
    DotProduct,
    FpSpecial,
    ReduceProduct,
    AbsError
};

/// \brief ULP verification meta-data
struct UlpVerifyInfo
{
    UlpVerifyInfo() = default;

    double ulp;
};

/// \brief Dot-product verification meta-data
struct DotProductVerifyInfo
{
    DotProductVerifyInfo() = default;

    int32_t s;
    int32_t ks;
};

/// \brief reduce-product verification meta-data
struct ReduceProductVerifyInfo
{
    ReduceProductVerifyInfo() = default;

    int64_t m;
    int64_t n;
};

/// \brief abs-error verification meta-data
struct AbsErrorVerifyInfo
{
    AbsErrorVerifyInfo() = default;

    double lowerBound;
};

/// \brief Verification meta-data
struct VerifyConfig
{
    VerifyConfig() = default;

    VerifyMode mode;
    DType dataType;
    UlpVerifyInfo ulpInfo;
    DotProductVerifyInfo dotProductInfo;
    ReduceProductVerifyInfo reduceProductInfo;
    AbsErrorVerifyInfo absErrorInfo;
};

/// \brief Parse the verification config for a tensor when given in JSON form
std::optional<VerifyConfig> parseVerifyConfig(const char* tensorName, const char* configJson);

/// \brief Extract number of total elements
int64_t numElements(const std::vector<int32_t>& shape);

/// \brief Convert a flat index to a shape position
std::vector<int32_t> indexToPosition(int64_t index, const std::vector<int32_t>& shape);

/// \brief A string representing the shape or position
std::string positionToString(const std::vector<int32_t>& pos);

/// \brief Map API data-type to DType
DType mapToDType(tosa_datatype_t dataType);

/// \brief Return 2 to the power of N or -N
// For use during compile time - as no range check
constexpr double const_exp2(int32_t n)
{
    double v = 1.0;
    while (n > 0)
    {
        v = v * 2.0;
        n--;
    }
    while (n < 0)
    {
        v = v / 2.0;
        n++;
    }
    return v;
}

/// \brief Same as const_exp2 but with runtime range check of N
double exp2(int32_t n);

/// \brief Return the base-2 exponent of V
int32_t ilog2(double v);

/// \brief Accuracy precision information
template <typename T>
struct AccPrecision;
template <>
struct AccPrecision<float>
{
    static constexpr double normal_min   = const_exp2(-126);
    static constexpr double normal_max   = const_exp2(128) - const_exp2(127 - 23);
    static constexpr int32_t normal_frac = 23;
};
template <>
struct AccPrecision<half_float::half>
{
    static constexpr double normal_min   = const_exp2(-14);
    static constexpr double normal_max   = const_exp2(16) - const_exp2(15 - 10);
    static constexpr int32_t normal_frac = 7;
};

/// \brief Error bounds check for ULP and ABS_ERROR modes
template <typename OutType>
bool tosaCheckFloatBound(OutType testValue, double referenceValue, double errorBound);
};    // namespace TosaReference

#endif    // VERIFY_UTILS_H_

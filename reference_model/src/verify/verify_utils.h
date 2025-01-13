
// Copyright (c) 2023-2025, ARM Limited.
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

#include "cfloat.h"
#include "dtype.h"
#include "half.hpp"
#include "types.h"

#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#define TOSA_REF_REQUIRE(COND, MESSAGE, ...)                                                                           \
    if (!(COND))                                                                                                       \
    {                                                                                                                  \
        WARNING("[Verifier]" MESSAGE ".", ##__VA_ARGS__);                                                              \
        return false;                                                                                                  \
    }

using bf16    = ct::cfloat<int16_t, 8, true, true, true>;
using fp8e4m3 = ct::cfloat<int8_t, 4, true, true, false>;
using fp8e5m2 = ct::cfloat<int8_t, 5, true, true, true>;

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
    AbsError,
    Relative
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

    int32_t setNumber;
    int32_t kernelSizeBound;
};

/// \brief reduce-product verification meta-data
struct ReduceProductVerifyInfo
{
    ReduceProductVerifyInfo() = default;

    int64_t numberOfProducts;
};

/// \brief abs-error verification meta-data
struct AbsErrorVerifyInfo
{
    AbsErrorVerifyInfo() = default;

    double lowerBound{ 0.0 };
    double normalDivisor{ 1.0 };
    bool boundAsMagnitude{ false };
    double maxCompare{ 0.0 };    // One value to compare against, before deciding absolute error-bound
};

/// \brief relative verification meta-data
struct RelativeVerifyInfo
{
    RelativeVerifyInfo() = default;

    double max;
    double scale;
    double ulpBound;
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
    RelativeVerifyInfo relativeInfo;
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

/// \brief Set resultWarning and resultDifference for NaN values
///
void setNaNWarning(double testValue, double referenceValue, double& resultDifference, std::string& resultWarning);

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
    static constexpr int32_t normal_frac = 10;
};
template <>
struct AccPrecision<bf16>
{
    static constexpr double normal_min   = const_exp2(-126);
    static constexpr double normal_max   = const_exp2(128) - const_exp2(127 - 7);
    static constexpr int32_t normal_frac = 7;
};
template <>
struct AccPrecision<fp8e4m3>
{
    static constexpr double normal_min   = const_exp2(-6);
    static constexpr double normal_max   = 448;
    static constexpr int32_t normal_frac = 3;
};
template <>
struct AccPrecision<fp8e5m2>
{
    static constexpr double normal_min   = const_exp2(-14);
    static constexpr double normal_max   = 57344;
    static constexpr int32_t normal_frac = 2;
};

/// \brief Single value error bounds check for ULP, ABS_ERROR and other compliance modes
///
/// \param testValue        Implementation value
/// \param referenceValue   Reference value
/// \param errorBound       Positive error bound value
/// \param resultDifference Return: Difference between reference value and implementation value
/// \param resultWarning    Return: Warning message if implementation is outside error bounds
///
/// \return True if compliant else false
template <typename OutType>
bool tosaCheckFloatBound(
    OutType testValue, double referenceValue, double errorBound, double& resultDifference, std::string& resultWarning);

/// \brief Whole tensor checker for values inside error bounds
///
/// \param referenceData        Reference output tensor data
/// \param boundsData           Optional reference bounds tensor data
/// \param implementationData   Implementation output tensor data
/// \param shape                Tensor shape - all tensors must be this shape
/// \param modeStr              Short string indicating which compliance mode we are testing
/// \param cfgPtr               Pointer to this mode's configuration data, passed to the calcErrorBound()
/// \param calcErrorBound       Pointer to a function that can calculate the error bound per ref value
///
/// \return True if compliant else false
template <typename OutType>
bool validateData(const double* referenceData,
                  const double* boundsData,
                  const OutType* implementationData,
                  const std::vector<int32_t>& shape,
                  const std::string& modeStr,
                  const void* cfgPtr,
                  double (*calcErrorBound)(double referenceValue, double boundsValue, const void* cfgPtr));

// Unused arguments helper function
template <typename... Args>
inline void unused(Args&&...)
{}
};    // namespace TosaReference

#endif    // VERIFY_UTILS_H_

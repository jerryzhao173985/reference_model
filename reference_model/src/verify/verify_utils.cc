
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

#include "verify_utils.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cfloat>
#include <map>
#include <string>

namespace tosa
{

NLOHMANN_JSON_SERIALIZE_ENUM(DType,
                             { { DType::DType_UNKNOWN, "UNKNOWN" },
                               { DType::DType_BOOL, "BOOL" },
                               { DType::DType_INT4, "INT4" },
                               { DType::DType_INT8, "INT8" },
                               { DType::DType_INT16, "INT16" },
                               { DType::DType_INT32, "INT32" },
                               { DType::DType_INT48, "INT48" },
                               { DType::DType_FP16, "FP16" },
                               { DType::DType_BF16, "BF16" },
                               { DType::DType_FP32, "FP32" },
                               { DType::DType_FP8E4M3, "FP8E4M3" },
                               { DType::DType_FP8E5M2, "FP8E5M2" },
                               { DType::DType_UINT16, "UINT16" },
                               { DType::DType_UINT8, "UINT8" } })

}    // namespace tosa

namespace TosaReference
{

NLOHMANN_JSON_SERIALIZE_ENUM(VerifyMode,
                             {
                                 { VerifyMode::Unknown, "UNKNOWN" },
                                 { VerifyMode::Exact, "EXACT" },
                                 { VerifyMode::Ulp, "ULP" },
                                 { VerifyMode::DotProduct, "DOT_PRODUCT" },
                                 { VerifyMode::FpSpecial, "FP_SPECIAL" },
                                 { VerifyMode::ReduceProduct, "REDUCE_PRODUCT" },
                                 { VerifyMode::AbsError, "ABS_ERROR" },
                                 { VerifyMode::Relative, "RELATIVE" },
                             })

void from_json(const nlohmann::json& j, UlpVerifyInfo& ulpInfo)
{
    j.at("ulp").get_to(ulpInfo.ulp);
}

void from_json(const nlohmann::json& j, DotProductVerifyInfo& dotProductInfo)
{
    j.at("s").get_to(dotProductInfo.setNumber);
    j.at("ks").get_to(dotProductInfo.kernelSize);
}

void from_json(const nlohmann::json& j, ReduceProductVerifyInfo& reduceProduceInfo)
{
    j.at("n").get_to(reduceProduceInfo.numberOfProducts);
}

void from_json(const nlohmann::json& j, AbsErrorVerifyInfo& absErrorInfo)
{
    if (j.contains("lower_bound"))
    {
        j.at("lower_bound").get_to(absErrorInfo.lowerBound);
    }
    if (j.contains("normal_divisor"))
    {
        j.at("normal_divisor").get_to(absErrorInfo.normalDivisor);
    }
    if (j.contains("bound_as_magnitude"))
    {
        j.at("bound_as_magnitude").get_to(absErrorInfo.boundAsMagnitude);
    }
    if (j.contains("max_compare"))
    {
        j.at("max_compare").get_to(absErrorInfo.maxCompare);
    }
}

void from_json(const nlohmann::json& j, RelativeVerifyInfo& rInfo)
{
    j.at("max").get_to(rInfo.max);
    j.at("scale").get_to(rInfo.scale);
}

void from_json(const nlohmann::json& j, VerifyConfig& cfg)
{
    j.at("mode").get_to(cfg.mode);
    j.at("data_type").get_to(cfg.dataType);
    cfg.ulpInfo.ulp = 0;
    if (j.contains("ulp_info"))
    {
        j.at("ulp_info").get_to(cfg.ulpInfo);
    }
    cfg.dotProductInfo.setNumber  = 0;
    cfg.dotProductInfo.kernelSize = 0;
    if (j.contains("dot_product_info"))
    {
        j.at("dot_product_info").get_to(cfg.dotProductInfo);
    }
    cfg.reduceProductInfo.numberOfProducts = 0;
    if (j.contains("reduce_product_info"))
    {
        j.at("reduce_product_info").get_to(cfg.reduceProductInfo);
    }
    cfg.absErrorInfo.lowerBound       = 0;
    cfg.absErrorInfo.normalDivisor    = 1;
    cfg.absErrorInfo.boundAsMagnitude = false;
    if (j.contains("abs_error_info"))
    {
        j.at("abs_error_info").get_to(cfg.absErrorInfo);
    }
    cfg.relativeInfo.max   = 0;
    cfg.relativeInfo.scale = 0;
    if (j.contains("relative_info"))
    {
        j.at("relative_info").get_to(cfg.relativeInfo);
    }
}

std::optional<VerifyConfig> parseVerifyConfig(const char* tensorName, const char* json)
{
    if (!tensorName)
        return std::nullopt;

    auto jsonCfg = nlohmann::json::parse(json, nullptr, /* allow exceptions */ false);

    if (jsonCfg.is_discarded())
    {
        WARNING("[Verifier] Invalid json config.");
        return std::nullopt;
    }
    if (!jsonCfg.contains("tensors"))
    {
        WARNING("[Verifier] Missing tensors in json config.");
        return std::nullopt;
    }

    const auto& tensors = jsonCfg["tensors"];
    if (!tensors.contains(tensorName))
        if (!tensors.contains(tensorName))
        {
            WARNING("[Verifier] Missing tensor %s in json config.", tensorName);
            return std::nullopt;
        }
    const auto& namedTensor = tensors[tensorName];
    return namedTensor.get<VerifyConfig>();
}

int64_t numElements(const std::vector<int32_t>& shape)
{
    return std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<int64_t>());
}

std::vector<int32_t> indexToPosition(int64_t index, const std::vector<int32_t>& shape)
{
    // Shape is rank 0 (scalar).
    if (shape.size() == 0)
        return { 0 };

    std::vector<int32_t> pos;
    for (auto d = shape.end() - 1; d >= shape.begin(); --d)
    {
        pos.insert(pos.begin(), index % *d);
        index /= *d;
    }
    ASSERT_MSG(index == 0, "index too large for given shape")
    return pos;
}

std::string positionToString(const std::vector<int32_t>& pos)
{
    std::string str = "[";
    for (auto d = pos.begin(); d < pos.end(); ++d)
    {
        str.append(std::to_string(*d));
        if (pos.end() - d > 1)
        {
            str.append(",");
        }
    }
    str.append("]");
    return str;
}

DType mapToDType(tosa_datatype_t dataType)
{
    static std::map<tosa_datatype_t, DType> typeMap = {
        { tosa_datatype_bool_t, DType_BOOL },       { tosa_datatype_int4_t, DType_INT4 },
        { tosa_datatype_int8_t, DType_INT8 },       { tosa_datatype_uint16_t, DType_UINT16 },
        { tosa_datatype_int16_t, DType_INT16 },     { tosa_datatype_int32_t, DType_INT32 },
        { tosa_datatype_int48_t, DType_INT48 },     { tosa_datatype_fp16_t, DType_FP16 },
        { tosa_datatype_bf16_t, DType_BF16 },       { tosa_datatype_fp32_t, DType_FP32 },
        { tosa_datatype_shape_t, DType_SHAPE },     { tosa_datatype_fp8e4m3_t, DType_FP8E4M3 },
        { tosa_datatype_fp8e5m2_t, DType_FP8E5M2 }, { tosa_datatype_uint8_t, DType_UINT8 },
    };

    if (typeMap.count(dataType))
    {
        return typeMap[dataType];
    }

    return DType_UNKNOWN;
}

// Like const_exp2 but for use during runtime
double exp2(int32_t n)
{
    if (n < -1075)
    {
        return 0.0;    // smaller than smallest denormal
    }
    TOSA_REF_REQUIRE(n <= 1023, " Invalid exponent value (%d) in exp2", n);
    return const_exp2(n);
}

int32_t ilog2(double v)
{
    TOSA_REF_REQUIRE(0.0 < v && v < std::numeric_limits<double>::infinity(), " Value out of range (%g) in ilog2", v);
    int32_t n = 0;
    while (v >= 2.0)
    {
        v = v / 2.0;
        n++;
    }
    while (v < 1.0)
    {
        v = v * 2.0;
        n--;
    }
    return n;
}

void setNaNWarning(double testValue, double referenceValue, double& resultDifference, std::string& resultWarning)
{
    char buff[200];
    snprintf(buff, 200, "Non-matching NaN values - ref (%g) versus test (%g).", referenceValue, testValue);
    resultWarning.assign(buff);
    resultDifference = std::numeric_limits<double>::quiet_NaN();
}

static_assert(std::numeric_limits<float>::is_iec559,
              "TOSA Reference Model has not been built with standard IEEE 754 32-bit float support; Bounds based "
              "verification is invalid");
static_assert(std::numeric_limits<double>::is_iec559,
              "TOSA Reference Model has not been built with standard IEEE 754 64-bit float support; Bounds based "
              "verification is invalid");

template <typename OutType>
bool tosaCheckFloatBound(
    OutType testValue, double referenceValue, double errorBound, double& resultDifference, std::string& resultWarning)
{
    // Both must be NaNs to be correct
    if (std::isnan(referenceValue) || (std::isnan(testValue) && !std::is_same<OutType, fp8e4m3>::value))
    {
        if (std::isnan(referenceValue) && std::isnan(testValue))
        {
            resultDifference = 0.0;
            return true;
        }
        setNaNWarning(static_cast<double>(testValue), referenceValue, resultDifference, resultWarning);
        return false;
    }

    // Check the errorBound
    TOSA_REF_REQUIRE(errorBound >= 0.f, " Invalid error bound (%g), expected positive value", errorBound);
    if (!std::isfinite(errorBound))
    {
        // When the errorBound is infinite (or NaN) there is no valid check to perform
        // This can happen for example in POW or EXP when the resulting reference
        // value is infinite, but the bounds value determined by compliance is finite.
        // Multiplying these two together to produce the errorBound will create an
        // infinite.
        resultDifference = 0.0;
        return true;
    }

    // Make the sign of the reference value positive
    // and adjust the test value appropriately.
    if (referenceValue < 0)
    {
        referenceValue = -referenceValue;
        testValue      = -testValue;
    }

    // Scale by the number of ULPs requested by the user.
    double referenceMax = referenceValue + errorBound;
    double referenceMin = referenceValue - errorBound;

    // Handle the overflow cases.
    if (referenceMax > AccPrecision<OutType>::normal_max)
    {
        if (std::is_same<OutType, fp8e4m3>::value)
        {
            referenceMax = std::numeric_limits<OutType>::quiet_NaN();
        }
        else
        {
            referenceMax = std::numeric_limits<OutType>::infinity();
        }
    }

    if (referenceMin > AccPrecision<OutType>::normal_max)
    {
        if (std::is_same<OutType, fp8e4m3>::value)
        {
            referenceMin = std::numeric_limits<OutType>::quiet_NaN();
        }
        else
        {
            referenceMin = std::numeric_limits<OutType>::infinity();
        }
    }
    // And the underflow cases.
    if (referenceMax < AccPrecision<OutType>::normal_min)
    {
        referenceMax = AccPrecision<OutType>::normal_min;
    }

    if (referenceMin < AccPrecision<OutType>::normal_min)
    {
        if (!std::is_same<OutType, fp8e4m3>::value && !std::is_same<OutType, fp8e5m2>::value)
        {
            // Allow subnormal values to be flushed to zero for non FP8 types
            // Also support large error bounds where referenceMin is negative
            referenceMin = std::min(0.0, referenceMin);
        }
    }

    // And finally... Do the comparison.
    double testValue64 = static_cast<double>(testValue);
    bool withinBound   = false;
    // referenceMax and/or referenceMin can be NaN for fp8e4m3
    if (std::isnan(referenceMax) && std::isnan(referenceMin))
    {
        // this check should never really occur, just accept any value of fp8e4m3
        withinBound = true;
    }
    else if (std::isnan(referenceMax))
    {
        withinBound = std::isnan(testValue64) || testValue64 >= referenceMin;
    }
    else if (std::isnan(referenceMin))
    {
        withinBound = std::isnan(testValue64) || testValue64 <= referenceMax;
    }
    else
    {
        withinBound = testValue64 >= referenceMin && testValue64 <= referenceMax;
    }
    // handle resultDifference and resultWarning for NaN cases for fp8e4m3
    if (std::isnan(testValue64))
    {
        if (withinBound)
        {
            resultDifference = 0.0;
            return true;
        }
        setNaNWarning(testValue64, referenceValue, resultDifference, resultWarning);
        return false;
    }
    resultDifference = testValue64 - referenceValue;

    if (!withinBound)
    {
        char buff[300];
        snprintf(buff, 300,
                 "value %.*g has a difference of %.*g compared to an error bound of +/- %.*g (range: %.*g <= ref %.*g "
                 "<= %.*g).",
                 DBL_DIG, testValue64, DBL_DIG, resultDifference, DBL_DIG, errorBound, DBL_DIG, referenceMin, DBL_DIG,
                 referenceValue, DBL_DIG, referenceMax);
        resultWarning.assign(buff);
    }
    return withinBound;
}

template <typename OutType>
bool validateData(const double* referenceData,
                  const double* boundsData,
                  const OutType* implementationData,
                  const std::vector<int32_t>& shape,
                  const std::string& modeStr,
                  const void* cfgPtr,
                  double (*calcErrorBound)(double referenceValue, double boundsValue, const void* cfgPtr))
{
    const size_t T = static_cast<size_t>(numElements(shape));
    TOSA_REF_REQUIRE(T > 0, "Invalid shape for reference tensor");
    TOSA_REF_REQUIRE(referenceData != nullptr, "Missing data for reference tensor");
    TOSA_REF_REQUIRE(implementationData != nullptr, "Missing data for implementation tensor");
    // NOTE: Bounds data tensor is allowed to be null as it may not be needed
    if (modeStr != "E")
    {
        TOSA_REF_REQUIRE(cfgPtr != nullptr, "Missing config for validation");
    }
    TOSA_REF_REQUIRE(calcErrorBound != nullptr, "Missing error bound function validation");

    std::string warning, worstWarning;
    double worstDifference = 0.0;
    // Set to invalid index
    size_t worstIndex = T;
    bool compliant    = true;

    for (size_t i = 0; i < T; ++i)
    {
        double difference = 0.0;
        double boundVal   = (boundsData == nullptr) ? 0.0 : boundsData[i];
        double errBound   = calcErrorBound(referenceData[i], boundVal, cfgPtr);
        bool valid        = tosaCheckFloatBound(implementationData[i], referenceData[i], errBound, difference, warning);
        if (!valid)
        {
            compliant = false;
            if (std::isnan(difference) || std::abs(difference) > std::abs(worstDifference))
            {
                worstIndex      = i;
                worstDifference = difference;
                worstWarning.assign(warning);
                if (std::isnan(difference))
                {
                    // Worst case is difference in NaN
                    break;
                }
            }
            else if (std::abs(difference) == 0.0)
            {
                auto pos = indexToPosition(i, shape);
                WARNING("[Verifier][%s] Invalid error bound, no difference found. Location: %s", modeStr.c_str(),
                        positionToString(pos).c_str());
                return false;
            }
        }
    }
    if (!compliant)
    {
        auto pos = indexToPosition(worstIndex, shape);
        WARNING("[Verifier][%s] Largest deviance at location %s: %s", modeStr.c_str(), positionToString(pos).c_str(),
                worstWarning.c_str());
    }
    return compliant;
}

// Instantiate the needed check functions
template bool validateData(const double* referenceData,
                           const double* boundsData,
                           const float* implementationData,
                           const std::vector<int32_t>& shape,
                           const std::string& modeStr,
                           const void* cfgPtr,
                           double (*calcErrorBound)(double referenceValue, double boundsValue, const void* cfgPtr));
template bool validateData(const double* referenceData,
                           const double* boundsData,
                           const half_float::half* implementationData,
                           const std::vector<int32_t>& shape,
                           const std::string& modeStr,
                           const void* cfgPtr,
                           double (*calcErrorBound)(double referenceValue, double boundsValue, const void* cfgPtr));
template bool validateData(const double* referenceData,
                           const double* boundsData,
                           const bf16* implementationData,
                           const std::vector<int32_t>& shape,
                           const std::string& modeStr,
                           const void* cfgPtr,
                           double (*calcErrorBound)(double referenceValue, double boundsValue, const void* cfgPtr));
template bool validateData(const double* referenceData,
                           const double* boundsData,
                           const fp8e4m3* implementationData,
                           const std::vector<int32_t>& shape,
                           const std::string& modeStr,
                           const void* cfgPtr,
                           double (*calcErrorBound)(double referenceValue, double boundsValue, const void* cfgPtr));
template bool validateData(const double* referenceData,
                           const double* boundsData,
                           const fp8e5m2* implementationData,
                           const std::vector<int32_t>& shape,
                           const std::string& modeStr,
                           const void* cfgPtr,
                           double (*calcErrorBound)(double referenceValue, double boundsValue, const void* cfgPtr));

}    // namespace TosaReference

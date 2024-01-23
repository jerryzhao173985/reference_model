
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
#include <map>

namespace tosa
{

NLOHMANN_JSON_SERIALIZE_ENUM(DType,
                             {
                                 { DType::DType_UNKNOWN, "UNKNOWN" },
                                 { DType::DType_BOOL, "BOOL" },
                                 { DType::DType_INT4, "INT4" },
                                 { DType::DType_INT8, "INT8" },
                                 { DType::DType_INT16, "INT16" },
                                 { DType::DType_INT32, "INT32" },
                                 { DType::DType_INT48, "INT48" },
                                 { DType::DType_FP16, "FP16" },
                                 { DType::DType_BF16, "BF16" },
                                 { DType::DType_FP32, "FP32" },
                             })

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
                             })

void from_json(const nlohmann::json& j, UlpVerifyInfo& ulpInfo)
{
    j.at("ulp").get_to(ulpInfo.ulp);
}

void from_json(const nlohmann::json& j, DotProductVerifyInfo& dotProductInfo)
{
    j.at("s").get_to(dotProductInfo.s);
    j.at("ks").get_to(dotProductInfo.ks);
}

void from_json(const nlohmann::json& j, ReduceProductVerifyInfo& reduceProduceInfo)
{
    j.at("n").get_to(reduceProduceInfo.n);
}

void from_json(const nlohmann::json& j, AbsErrorVerifyInfo& absErrorInfo)
{
    if (j.contains("lower_bound"))
    {
        j.at("lower_bound").get_to(absErrorInfo.lowerBound);
    }
}

void from_json(const nlohmann::json& j, VerifyConfig& cfg)
{
    j.at("mode").get_to(cfg.mode);
    j.at("data_type").get_to(cfg.dataType);
    if (j.contains("ulp_info"))
    {
        j.at("ulp_info").get_to(cfg.ulpInfo);
    }
    if (j.contains("dot_product_info"))
    {
        j.at("dot_product_info").get_to(cfg.dotProductInfo);
    }
    if (j.contains("reduce_product_info"))
    {
        j.at("reduce_product_info").get_to(cfg.reduceProductInfo);
    }
    // Set up defaults for optional AbsErrorVerifyInfo
    cfg.absErrorInfo.lowerBound = 0;
    if (j.contains("abs_error_info"))
    {
        j.at("abs_error_info").get_to(cfg.absErrorInfo);
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
        { tosa_datatype_bool_t, DType_BOOL },   { tosa_datatype_int4_t, DType_INT4 },
        { tosa_datatype_int8_t, DType_INT8 },   { tosa_datatype_uint16_t, DType_UINT16 },
        { tosa_datatype_int16_t, DType_INT16 }, { tosa_datatype_int32_t, DType_INT32 },
        { tosa_datatype_int48_t, DType_INT48 }, { tosa_datatype_fp16_t, DType_FP16 },
        { tosa_datatype_bf16_t, DType_BF16 },   { tosa_datatype_fp32_t, DType_FP32 },
        { tosa_datatype_shape_t, DType_SHAPE },
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

static_assert(std::numeric_limits<float>::is_iec559,
              "TOSA Reference Model has not been built with standard IEEE 754 32-bit float support; Bounds based "
              "verification is invalid");
static_assert(std::numeric_limits<double>::is_iec559,
              "TOSA Reference Model has not been built with standard IEEE 754 64-bit float support; Bounds based "
              "verification is invalid");

template <typename OutType>
bool tosaCheckFloatBound(OutType testValue, double referenceValue, double errorBound)
{
    // Both must be NaNs to be correct
    if (std::isnan(referenceValue) || std::isnan(testValue))
    {
        if (std::isnan(referenceValue) && std::isnan(testValue))
        {
            return true;
        }
        WARNING("[Verifier][Bound] Non-matching NaN values - ref (%g) versus test (%g).", referenceValue, testValue);
        return false;
    }

    // Check the errorBound
    TOSA_REF_REQUIRE(errorBound >= 0.f, " Invalid error bound (%g)", errorBound);

    // Make the sign of the reference value positive
    // and adjust the test value appropriately.
    if (referenceValue < 0)
    {
        referenceValue = -referenceValue;
        testValue      = -testValue;
    }

    // At this point we are ready to calculate the ULP bounds for the reference value.
    double referenceMin, referenceMax;

    // If the reference is infinity e.g. the result of an overflow the test value must
    // be infinity of an appropriate sign.
    if (std::isinf(referenceValue))
    {
        // We already canonicalized the input such that the reference value is positive
        // so no need to check again here.
        referenceMin = std::numeric_limits<OutType>::infinity();
        referenceMax = std::numeric_limits<OutType>::infinity();
    }
    else if (referenceValue == 0)
    {
        // For zero we require that the results match exactly with the correct sign.
        referenceMin = 0;
        referenceMax = 0;
    }
    else
    {

        // Scale by the number of ULPs requested by the user.
        referenceMax = referenceValue + errorBound;
        referenceMin = referenceValue - errorBound;

        // Handle the overflow cases.
        if (referenceMax > AccPrecision<OutType>::normal_max)
        {
            referenceMax = std::numeric_limits<OutType>::infinity();
        }

        if (referenceMin > AccPrecision<OutType>::normal_max)
        {
            referenceMin = std::numeric_limits<OutType>::infinity();
        }

        // And the underflow cases.
        if (referenceMax < AccPrecision<OutType>::normal_min)
        {
            referenceMax = AccPrecision<OutType>::normal_min;
        }

        if (referenceMin < AccPrecision<OutType>::normal_min)
        {
            referenceMin = 0.0;
        }
    }

    // And finally... Do the comparison.
    double testValue64 = static_cast<double>(testValue);
    bool withinBound   = testValue64 >= referenceMin && testValue64 <= referenceMax;
    if (!withinBound)
    {
        WARNING("[Verifier][Bound] value %.20f is not in error bound %g range (%.20f <= ref %.20f <= %.20f).",
                testValue64, testValue64, errorBound, referenceMin, referenceValue, referenceValue, referenceMax);
    }
    return withinBound;
}

// Instantiate the needed check functions
template bool tosaCheckFloatBound(float testValue, double referenceValue, double errorBound);
template bool tosaCheckFloatBound(half_float::half testValue, double referenceValue, double errorBound);
}    // namespace TosaReference

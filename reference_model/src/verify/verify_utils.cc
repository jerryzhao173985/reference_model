
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

#include "verify_utils.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <map>

namespace tosa
{

NLOHMANN_JSON_SERIALIZE_ENUM(DType,
                             {
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
                                 { VerifyMode::Exact, "EXACT" },
                                 { VerifyMode::Ulp, "ULP" },
                                 { VerifyMode::DotProduct, "DOT_PRODUCT" },
                                 { VerifyMode::ReduceProduct, "REDUCE_PRODUCT" },
                                 { VerifyMode::FpSpecial, "FP_SPECIAL" },
                             })

void from_json(const nlohmann::json& j, UlpInfo& ulpInfo)
{
    j.at("ulp").get_to(ulpInfo.ulp);
}

void from_json(const nlohmann::json& j, DotProductVerifyInfo& dotProductInfo)
{
    j.at("data_type").get_to(dotProductInfo.dataType);
    j.at("s").get_to(dotProductInfo.s);
    j.at("ks").get_to(dotProductInfo.ks);
}

void from_json(const nlohmann::json& j, VerifyConfig& cfg)
{
    j.at("mode").get_to(cfg.mode);
    if (j.contains("ulp_info"))
    {
        j.at("ulp_info").get_to(cfg.ulpInfo);
    }
    if (j.contains("dot_product_info"))
    {
        j.at("dot_product_info").get_to(cfg.dotProductInfo);
    }
}

std::optional<VerifyConfig> parseVerifyConfig(const char* tensorName, const char* json)
{
    if (!tensorName)
        return std::nullopt;

    auto jsonCfg = nlohmann::json::parse(json, nullptr, /* allow exceptions */ false);

    if (jsonCfg.is_discarded())
        return std::nullopt;
    if (!jsonCfg.contains("tensors"))
        return std::nullopt;

    const auto& tensors = jsonCfg["tensors"];
    if (!tensors.contains(tensorName))
        return std::nullopt;

    const auto& namedTensor = tensors[tensorName];
    return namedTensor.get<VerifyConfig>();
}

int64_t numElements(const std::vector<int32_t>& shape)
{
    return std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<int64_t>());
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
}    // namespace TosaReference

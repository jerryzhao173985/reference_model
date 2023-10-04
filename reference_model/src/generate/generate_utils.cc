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

#include "generate_utils.h"

#include <nlohmann/json.hpp>

#include <algorithm>

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

NLOHMANN_JSON_SERIALIZE_ENUM(Op,
                             {
                                 { Op::Op_MATMUL, "MATMUL" },
                             })

}    // namespace tosa

namespace TosaReference
{

NLOHMANN_JSON_SERIALIZE_ENUM(GeneratorType,
                             {
                                 { GeneratorType::PseudoRandom, "PSEUDO_RANDOM" },
                                 { GeneratorType::DotProduct, "DOT_PRODUCT" },
                                 { GeneratorType::OpFullRange, "OP_FULL_RANGE" },
                                 { GeneratorType::OpBoundary, "OP_BOUNDARY" },
                                 { GeneratorType::OpSpecial, "OP_SPECIAL" },
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(InputType,
                             {
                                 { InputType::Variable, "VARIABLE" },
                                 { InputType::Constant, "CONSTANT" },
                             })

void from_json(const nlohmann::json& j, DotProductInfo& dotProductInfo)
{
    j.at("s").get_to(dotProductInfo.s);
    j.at("ks").get_to(dotProductInfo.ks);
    j.at("acc_type").get_to(dotProductInfo.accType);
    if (j.contains("kernel"))
    {
        j.at("kernel").get_to(dotProductInfo.kernel);
    }
    if (j.contains("axis"))
    {
        j.at("axis").get_to(dotProductInfo.axis);
    }
}

void from_json(const nlohmann::json& j, GenerateConfig& cfg)
{
    j.at("data_type").get_to(cfg.dataType);
    j.at("input_type").get_to(cfg.inputType);
    j.at("shape").get_to(cfg.shape);
    j.at("input_pos").get_to(cfg.inputPos);
    j.at("op").get_to(cfg.opType);
    j.at("generator").get_to(cfg.generatorType);
    if (j.contains("dot_product_info"))
    {
        j.at("dot_product_info").get_to(cfg.dotProductInfo);
    }
}

std::optional<GenerateConfig> parseGenerateConfig(const char* json, const char* tensorName)
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
    return namedTensor.get<GenerateConfig>();
}

int64_t numElementsFromShape(const std::vector<int32_t>& shape)
{
    // Rank 0 shapes have no entries and so this will return 1
    // Other ranked shapes will return the product of their dimensions
    return std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<int64_t>());
}

size_t elementSizeFromType(DType type)
{
    switch (type)
    {
        case DType::DType_BOOL:
        case DType::DType_UINT8:
        case DType::DType_INT8:
            return 1;
        case DType::DType_UINT16:
        case DType::DType_INT16:
        case DType::DType_FP16:
        case DType::DType_BF16:
            return 2;
        case DType::DType_INT32:
        case DType::DType_FP32:
            return 4;
        default:
            return 0;
    }
    return 0;
}
}    // namespace TosaReference

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

#include "generate_utils.h"

#include <nlohmann/json.hpp>

#include <algorithm>

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
                                 { DType::DType_SHAPE, "SHAPE" },
                                 { DType::DType_FP8E4M3, "FP8E4M3" },
                                 { DType::DType_FP8E5M2, "FP8E5M2" },
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(Op,
                             {
                                 { Op::Op_UNKNOWN, "UNKNOWN" },
                                 { Op::Op_ABS, "ABS" },
                                 { Op::Op_ADD, "ADD" },
                                 { Op::Op_ARGMAX, "ARGMAX" },
                                 { Op::Op_AVG_POOL2D, "AVG_POOL2D" },
                                 { Op::Op_CAST, "CAST" },
                                 { Op::Op_CEIL, "CEIL" },
                                 { Op::Op_CLAMP, "CLAMP" },
                                 { Op::Op_CONCAT, "CONCAT" },
                                 { Op::Op_CONST, "CONST" },
                                 { Op::Op_CONV2D, "CONV2D" },
                                 { Op::Op_COS, "COS" },
                                 { Op::Op_DEPTHWISE_CONV2D, "DEPTHWISE_CONV2D" },
                                 { Op::Op_CONV3D, "CONV3D" },
                                 { Op::Op_EQUAL, "EQUAL" },
                                 { Op::Op_ERF, "ERF" },
                                 { Op::Op_EXP, "EXP" },
                                 { Op::Op_FLOOR, "FLOOR" },
                                 { Op::Op_FFT2D, "FFT2D" },
                                 { Op::Op_GATHER, "GATHER" },
                                 { Op::Op_GREATER, "GREATER" },
                                 { Op::Op_GREATER_EQUAL, "GREATER_EQUAL" },
                                 { Op::Op_IDENTITY, "IDENTITY" },
                                 { Op::Op_LOG, "LOG" },
                                 { Op::Op_MATMUL, "MATMUL" },
                                 { Op::Op_MAXIMUM, "MAXIMUM" },
                                 { Op::Op_MAX_POOL2D, "MAX_POOL2D" },
                                 { Op::Op_MINIMUM, "MINIMUM" },
                                 { Op::Op_MUL, "MUL" },
                                 { Op::Op_NEGATE, "NEGATE" },
                                 { Op::Op_PAD, "PAD" },
                                 { Op::Op_POW, "POW" },
                                 { Op::Op_RECIPROCAL, "RECIPROCAL" },
                                 { Op::Op_REDUCE_MAX, "REDUCE_MAX" },
                                 { Op::Op_REDUCE_MIN, "REDUCE_MIN" },
                                 { Op::Op_REDUCE_PRODUCT, "REDUCE_PRODUCT" },
                                 { Op::Op_REDUCE_SUM, "REDUCE_SUM" },
                                 { Op::Op_RESHAPE, "RESHAPE" },
                                 { Op::Op_RESIZE, "RESIZE" },
                                 { Op::Op_REVERSE, "REVERSE" },
                                 { Op::Op_RFFT2D, "RFFT2D" },
                                 { Op::Op_RSQRT, "RSQRT" },
                                 { Op::Op_SCATTER, "SCATTER" },
                                 { Op::Op_SELECT, "SELECT" },
                                 { Op::Op_SIGMOID, "SIGMOID" },
                                 { Op::Op_SIN, "SIN" },
                                 { Op::Op_SLICE, "SLICE" },
                                 { Op::Op_SUB, "SUB" },
                                 { Op::Op_TANH, "TANH" },
                                 { Op::Op_TILE, "TILE" },
                                 { Op::Op_TRANSPOSE, "TRANSPOSE" },
                                 { Op::Op_TRANSPOSE_CONV2D, "TRANSPOSE_CONV2D" },
                             })

}    // namespace tosa

namespace TosaReference
{

NLOHMANN_JSON_SERIALIZE_ENUM(GeneratorType,
                             {
                                 { GeneratorType::Unknown, "UNKNOWN" },
                                 { GeneratorType::PseudoRandom, "PSEUDO_RANDOM" },
                                 { GeneratorType::DotProduct, "DOT_PRODUCT" },
                                 { GeneratorType::FullRange, "FULL_RANGE" },
                                 { GeneratorType::Boundary, "BOUNDARY" },
                                 { GeneratorType::FpSpecial, "FP_SPECIAL" },
                                 { GeneratorType::FixedData, "FIXED_DATA" },
                             })

// NOTE: This assumes it's VARIABLE if the InputType is not recognized
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

void from_json(const nlohmann::json& j, PseudoRandomInfo& pseudoRandomInfo)
{
    j.at("rng_seed").get_to(pseudoRandomInfo.rngSeed);
    if (j.contains("range"))
    {
        j.at("range").get_to(pseudoRandomInfo.range);
    }
    if (j.contains("round"))
    {
        j.at("round").get_to(pseudoRandomInfo.round);
    }
}

void from_json(const nlohmann::json& j, FixedDataInfo& fixedDataInfo)
{
    j.at("data").get_to(fixedDataInfo.data);
}

void from_json(const nlohmann::json& j, FullRangeInfo& fullRangeInfo)
{
    if (j.contains("start_val"))
    {
        j.at("start_val").get_to(fullRangeInfo.startVal);
    }
}

void from_json(const nlohmann::json& j, FpSpecialInfo& fpSpecialInfo)
{
    if (j.contains("start_idx"))
    {
        j.at("start_idx").get_to(fpSpecialInfo.startIndex);
    }
    if (j.contains("rng_seed"))
    {
        j.at("rng_seed").get_to(fpSpecialInfo.rngSeed);
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

    // Set up defaults for dotProductInfo
    cfg.dotProductInfo.s       = -1;
    cfg.dotProductInfo.ks      = -1;
    cfg.dotProductInfo.accType = DType_UNKNOWN;
    cfg.dotProductInfo.kernel  = std::vector<int32_t>();
    cfg.dotProductInfo.axis    = -1;
    if (j.contains("dot_product_info"))
    {
        j.at("dot_product_info").get_to(cfg.dotProductInfo);
    }

    // Set up defaults for pseudoRandomInfo
    cfg.pseudoRandomInfo.rngSeed = 0;
    cfg.pseudoRandomInfo.range   = std::vector<std::string>();
    cfg.pseudoRandomInfo.round   = false;
    if (j.contains("pseudo_random_info"))
    {
        j.at("pseudo_random_info").get_to(cfg.pseudoRandomInfo);
    }

    // Set up defaults for fixedDataInfo
    cfg.fixedDataInfo.data = std::vector<int32_t>();
    if (j.contains("fixed_data_info"))
    {
        j.at("fixed_data_info").get_to(cfg.fixedDataInfo);
    }

    //Set up defaults for fullRangeInfo
    cfg.fullRangeInfo.startVal = 0;
    if (j.contains("full_range_info"))
    {
        j.at("full_range_info").get_to(cfg.fullRangeInfo);
    }

    //Set up defaults for fpSpecialInfo
    cfg.fpSpecialInfo.startIndex = 0;
    cfg.fpSpecialInfo.rngSeed    = 0;
    if (j.contains("fp_special_info"))
    {
        j.at("fp_special_info").get_to(cfg.fpSpecialInfo);
    }
}

std::optional<GenerateConfig> parseGenerateConfig(const char* json, const char* tensorName)
{
    if (!tensorName)
        return std::nullopt;

    auto jsonCfg = nlohmann::json::parse(json, nullptr, /* allow exceptions */ false);

    if (jsonCfg.is_discarded())
    {
        WARNING("[Generator] Invalid json config.");
        return std::nullopt;
    }
    if (!jsonCfg.contains("tensors"))
    {
        WARNING("[Generator] Missing tensors in json config.");
        return std::nullopt;
    }
    const auto& tensors = jsonCfg["tensors"];
    if (!tensors.contains(tensorName))
    {
        WARNING("[Generator] Missing tensor %s in json config.", tensorName);
        return std::nullopt;
    }
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
        case DType::DType_FP8E4M3:
        case DType::DType_FP8E5M2:
            return 1;
        case DType::DType_UINT16:
        case DType::DType_INT16:
        case DType::DType_FP16:
        case DType::DType_BF16:
            return 2;
        case DType::DType_INT32:
        case DType::DType_FP32:
        case DType::DType_SHAPE:
            return 4;
        default:
            return 0;
    }
    return 0;
}
}    // namespace TosaReference

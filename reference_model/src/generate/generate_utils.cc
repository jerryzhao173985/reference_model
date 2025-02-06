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
                                 { DType::DType_UINT8, "UINT8" },
                                 { DType::DType_UINT16, "UINT16" },
                                 { DType::DType_FP16, "FP16" },
                                 { DType::DType_BF16, "BF16" },
                                 { DType::DType_FP32, "FP32" },
                                 { DType::DType_SHAPE, "SHAPE" },
                                 { DType::DType_FP8E4M3, "FP8E4M3" },
                                 { DType::DType_FP8E5M2, "FP8E5M2" },
                             })

// Operators follow the definition order in tosa_generated.h
// This should be updated when new operators are added.
NLOHMANN_JSON_SERIALIZE_ENUM(Op,
                             {
                                 { Op_UNKNOWN, "UNKNOWN" },
                                 { Op_ARGMAX, "ARGMAX" },
                                 { Op_AVG_POOL2D, "AVG_POOL2D" },
                                 { Op_CONV2D, "CONV2D" },
                                 { Op_CONV3D, "CONV3D" },
                                 { Op_DEPTHWISE_CONV2D, "DEPTHWISE_CONV2D" },
                                 { Op_MATMUL, "MATMUL" },
                                 { Op_MAX_POOL2D, "MAX_POOL2D" },
                                 { Op_TRANSPOSE_CONV2D, "TRANSPOSE_CONV2D" },
                                 { Op_CLAMP, "CLAMP" },
                                 { Op_SIGMOID, "SIGMOID" },
                                 { Op_TANH, "TANH" },
                                 { Op_ADD, "ADD" },
                                 { Op_ARITHMETIC_RIGHT_SHIFT, "ARITHMETIC_RIGHT_SHIFT" },
                                 { Op_BITWISE_AND, "BITWISE_AND" },
                                 { Op_BITWISE_OR, "BITWISE_OR" },
                                 { Op_BITWISE_XOR, "BITWISE_XOR" },
                                 { Op_INTDIV, "INTDIV" },
                                 { Op_LOGICAL_AND, "LOGICAL_AND" },
                                 { Op_LOGICAL_LEFT_SHIFT, "LOGICAL_LEFT_SHIFT" },
                                 { Op_LOGICAL_RIGHT_SHIFT, "LOGICAL_RIGHT_SHIFT" },
                                 { Op_LOGICAL_OR, "LOGICAL_OR" },
                                 { Op_LOGICAL_XOR, "LOGICAL_XOR" },
                                 { Op_MAXIMUM, "MAXIMUM" },
                                 { Op_MINIMUM, "MINIMUM" },
                                 { Op_MUL, "MUL" },
                                 { Op_POW, "POW" },
                                 { Op_SUB, "SUB" },
                                 { Op_TABLE, "TABLE" },
                                 { Op_ABS, "ABS" },
                                 { Op_BITWISE_NOT, "BITWISE_NOT" },
                                 { Op_CEIL, "CEIL" },
                                 { Op_CLZ, "CLZ" },
                                 { Op_EXP, "EXP" },
                                 { Op_FLOOR, "FLOOR" },
                                 { Op_LOG, "LOG" },
                                 { Op_LOGICAL_NOT, "LOGICAL_NOT" },
                                 { Op_NEGATE, "NEGATE" },
                                 { Op_RECIPROCAL, "RECIPROCAL" },
                                 { Op_RSQRT, "RSQRT" },
                                 { Op_SELECT, "SELECT" },
                                 { Op_EQUAL, "EQUAL" },
                                 { Op_GREATER, "GREATER" },
                                 { Op_GREATER_EQUAL, "GREATER_EQUAL" },
                                 { Op_REDUCE_ANY, "REDUCE_ANY" },
                                 { Op_REDUCE_ALL, "REDUCE_ALL" },
                                 { Op_REDUCE_MAX, "REDUCE_MAX" },
                                 { Op_REDUCE_MIN, "REDUCE_MIN" },
                                 { Op_REDUCE_PRODUCT, "REDUCE_PRODUCT" },
                                 { Op_REDUCE_SUM, "REDUCE_SUM" },
                                 { Op_CONCAT, "CONCAT" },
                                 { Op_PAD, "PAD" },
                                 { Op_RESHAPE, "RESHAPE" },
                                 { Op_REVERSE, "REVERSE" },
                                 { Op_SLICE, "SLICE" },
                                 { Op_TILE, "TILE" },
                                 { Op_TRANSPOSE, "TRANSPOSE" },
                                 { Op_GATHER, "GATHER" },
                                 { Op_SCATTER, "SCATTER" },
                                 { Op_RESIZE, "RESIZE" },
                                 { Op_CAST, "CAST" },
                                 { Op_RESCALE, "RESCALE" },
                                 { Op_CONST, "CONST" },
                                 { Op_IDENTITY, "IDENTITY" },
                                 { Op_COND_IF, "COND_IF" },
                                 { Op_WHILE_LOOP, "WHILE_LOOP" },
                                 { Op_FFT2D, "FFT2D" },
                                 { Op_RFFT2D, "RFFT2D" },
                                 { Op_ERF, "ERF" },
                                 { Op_CONST_SHAPE, "CONST_SHAPE" },
                                 { Op_COS, "COS" },
                                 { Op_SIN, "SIN" },
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
                                 { GeneratorType::Special, "SPECIAL" },
                                 { GeneratorType::FixedData, "FIXED_DATA" },
                             })

NLOHMANN_JSON_SERIALIZE_ENUM(SpecialTestSet,
                             {
                                 { SpecialTestSet::Default, "DEFAULT" },
                                 { SpecialTestSet::CastFpToInt, "CAST_FP_TO_INT" },
                                 { SpecialTestSet::AllMaxValues, "ALL_MAX_VALUES" },
                                 { SpecialTestSet::AllLowestValues, "ALL_LOWEST_VALUES" },
                                 { SpecialTestSet::AllZeroes, "ALL_ZEROES" },
                                 { SpecialTestSet::AllSmallValues, "ALL_SMALL_VALUES" },
                                 { SpecialTestSet::FirstMaxThenZeroes, "FIRST_MAX_THEN_ZEROES" },
                                 { SpecialTestSet::FirstLowestThenZeroes, "FIRST_LOWEST_THEN_ZEROES" },
                                 { SpecialTestSet::FirstMaxThenMinusOnes, "FIRST_MAX_THEN_MINUS_ONES" },
                                 { SpecialTestSet::FirstLowestThenPlusOnes, "FIRST_LOWEST_THEN_PLUS_ONES" },
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
    if (j.contains("otherInputType"))
    {
        j.at("otherInputType").get_to(dotProductInfo.otherInputType);
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

void from_json(const nlohmann::json& j, SpecialInfo& specialInfo)
{
    if (j.contains("start_idx"))
    {
        j.at("start_idx").get_to(specialInfo.startIndex);
    }

    if (j.contains("rng_seed"))
    {
        j.at("rng_seed").get_to(specialInfo.rngSeed);
    }

    if (j.contains("special_test_set"))
    {
        j.at("special_test_set").get_to(specialInfo.specialTestSet);
    }
    else
    {
        specialInfo.specialTestSet = SpecialTestSet::Default;
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
    if (j.contains("unsigned_data"))
    {
        j.at("unsigned_data").get_to(cfg.unsignedData);
    }
    else
    {
        cfg.unsignedData = false;
    }

    // Set up defaults for dotProductInfo
    cfg.dotProductInfo.s              = -1;
    cfg.dotProductInfo.ks             = -1;
    cfg.dotProductInfo.accType        = DType_UNKNOWN;
    cfg.dotProductInfo.kernel         = std::vector<int32_t>();
    cfg.dotProductInfo.axis           = -1;
    cfg.dotProductInfo.otherInputType = DType_UNKNOWN;
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
    cfg.specialInfo.startIndex = 0;
    cfg.specialInfo.rngSeed    = 0;
    if (j.contains("special_info"))
    {
        j.at("special_info").get_to(cfg.specialInfo);
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

size_t tensorSizeInBytesFromType(int64_t numElements, DType type)
{
    switch (type)
    {
        case DType::DType_BOOL:
        case DType::DType_UINT8:
        case DType::DType_INT8:
        case DType::DType_FP8E4M3:
        case DType::DType_FP8E5M2:
            return 1 * numElements;
        case DType::DType_UINT16:
        case DType::DType_INT16:
        case DType::DType_FP16:
        case DType::DType_BF16:
            return 2 * numElements;
        case DType::DType_INT32:
        case DType::DType_FP32:
        case DType::DType_SHAPE:
            return 4 * numElements;
        case DType::DType_INT4:
            // 2x INT4 elements per byte
            return (numElements + 1) >> 1;
        case DType::DType_INT48:
            return 6 * numElements;
        default:
            FATAL_ERROR("dtype is not supported");
            return 0;
    }
    return 0;
}

// Integer write value functions
template <typename StorageType, TOSA_REF_TYPE TosaRefType>
void writeValue(int64_t value, int64_t index, StorageType* data)
{
    data[index] = static_cast<StorageType>(value);
}

template <>
void writeValue<int8_t, TOSA_REF_TYPE_INT4>(int64_t value, int64_t index, int8_t* data)
{
    // Packed index
    const auto byte_idx = index >> 1;
    // Low or high part of the byte
    const auto byte_pos = index & 0x1;

    int8_t byte_half0, byte_half1;
    if (byte_pos == 0)
    {
        // overwrite low position
        byte_half0 = static_cast<int8_t>(value);
        byte_half1 = data[byte_idx];
    }
    else
    {
        // overwrite high position
        byte_half0 = data[byte_idx];
        byte_half1 = static_cast<int8_t>(value);
    }
    data[byte_idx] = (byte_half0 & 0xF) | ((byte_half1 & 0xF) << 4);
}

template <>
void writeValue<int8_t, TOSA_REF_TYPE_INT48>(int64_t value, int64_t index, int8_t* data)
{
    const auto byte_idx = index * 6;
    const auto val_u64  = static_cast<uint64_t>(value);
    for (auto i = 0; i < 6; ++i)
    {
        auto shift         = i * 8;
        data[byte_idx + i] = (val_u64 >> shift) & 0xFF;
    }
}

template <>
void writeValue<int8_t, TOSA_REF_TYPE_BOOL>(int64_t value, int64_t index, int8_t* data)
{
    // Make sure we truncate the values to valid boolean values
    data[index] = static_cast<bool>(value);
}

// Instantiate other needed writeValue functions
template void writeValue<int8_t, TOSA_REF_TYPE_INT8>(int64_t value, int64_t index, int8_t* data);
template void writeValue<int16_t, TOSA_REF_TYPE_INT16>(int64_t value, int64_t index, int16_t* data);
template void writeValue<int32_t, TOSA_REF_TYPE_INT32>(int64_t value, int64_t index, int32_t* data);
template void writeValue<uint8_t, TOSA_REF_TYPE_UINT8>(int64_t value, int64_t index, uint8_t* data);
template void writeValue<uint16_t, TOSA_REF_TYPE_UINT16>(int64_t value, int64_t index, uint16_t* data);
}    // namespace TosaReference

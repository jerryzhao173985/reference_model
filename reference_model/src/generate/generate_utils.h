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

#ifndef GENERATE_UTILS_H_
#define GENERATE_UTILS_H_

#include "dtype.h"

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

namespace TosaReference
{

/// \brief Supported generator types
enum class GeneratorType
{
    Unknown,
    PseudoRandom,
    DotProduct,
    OpFullRange,
    OpBoundary,
    OpSpecial,
};

/// \brief Supported input types
enum class InputType
{
    Variable,
    Constant,
};

/// \brief Dot-product generator meta-data
struct DotProductInfo
{
    DotProductInfo() = default;

    int32_t s;
    int32_t ks;
    DType accType;
    int32_t axis;
    std::array<int32_t, 2> kernel;
};

/// \brief Generator configuration
struct GenerateConfig
{
    GeneratorType generatorType;
    DType dataType;
    InputType inputType;
    std::vector<int32_t> shape;
    int32_t inputPos;
    tosa::Op opType;
    DotProductInfo dotProductInfo;
};

/// \brief Parse the generator config when given in JSON form
std::optional<GenerateConfig> parseGenerateConfig(const char* json, const char* tensorName);

/// \brief Extract number of total elements
int64_t numElementsFromShape(const std::vector<int32_t>& shape);

/// \brief Size in bytes of a given type
size_t elementSizeFromType(DType type);

};    // namespace TosaReference

#endif    // GENERATE_UTILS_H_

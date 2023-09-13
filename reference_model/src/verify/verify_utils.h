
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

#ifndef VERIFY_UTILS_H_
#define VERIFY_UTILS_H_

#include "dtype.h"
#include "types.h"

#include <cstdint>
#include <optional>
#include <vector>

#define TOSA_REF_REQUIRE(COND, MESSAGE)                                                                                \
    if (!(COND))                                                                                                       \
    {                                                                                                                  \
        WARNING(MESSAGE);                                                                                              \
        return false;                                                                                                  \
    }

namespace TosaReference
{

// Name alias
using CTensor = tosa_tensor_t;

/// \brief Supported verification modes
enum class VerifyMode
{
    Exact,
    Ulp,
    DotProduct,
    ReduceProduct,
    FpSpecial
};

/// \brief ULP verification meta-data
struct UlpInfo
{
    UlpInfo() = default;

    uint64_t ulp;
};

/// \brief Dot-product verification meta-data
struct DotProductVerifyInfo
{
    DotProductVerifyInfo() = default;

    DType dataType;
    int32_t s;
    int32_t ks;
};

/// \brief Verification meta-data
struct VerifyConfig
{
    VerifyConfig() = default;

    VerifyMode mode;
    UlpInfo ulpInfo;
    DotProductVerifyInfo dotProductInfo;
};

/// \brief Parse the verification config for a tensor when given in JSON form
std::optional<VerifyConfig> parseVerifyConfig(const char* tensorName, const char* configJson);

/// \brief Extract number of total elements
int64_t numElements(const std::vector<int32_t>& shape);

/// \brief Map API data-type to DType
DType mapToDType(tosa_datatype_t dataType);

};    // namespace TosaReference

#endif    // VERIFY_UTILS_H_

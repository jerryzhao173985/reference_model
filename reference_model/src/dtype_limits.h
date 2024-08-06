// Copyright (c) 2024, ARM Limited.
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

#include "dtype.h"
#include "half.hpp"

using half = half_float::half;

namespace TosaReference
{

template <TOSA_REF_TYPE type>
struct DtypeLimits;

template <>
struct DtypeLimits<TOSA_REF_TYPE_FP64>
{
    static constexpr double lowest       = std::numeric_limits<double>::lowest();
    static constexpr double max          = std::numeric_limits<double>::max();
    static constexpr double min          = std::numeric_limits<double>::min();
    static constexpr double denorm_min   = std::numeric_limits<double>::denorm_min();
    static constexpr double infinity     = std::numeric_limits<double>::infinity();
    static constexpr double low_extreme  = -infinity;
    static constexpr double high_extreme = infinity;
    static constexpr bool has_infinity   = true;
};

template <>
struct DtypeLimits<TOSA_REF_TYPE_FP32>
{
    static constexpr float lowest       = std::numeric_limits<float>::lowest();
    static constexpr float max          = std::numeric_limits<float>::max();
    static constexpr float min          = std::numeric_limits<float>::min();
    static constexpr float denorm_min   = std::numeric_limits<float>::denorm_min();
    static constexpr float infinity     = std::numeric_limits<float>::infinity();
    static constexpr float low_extreme  = -infinity;
    static constexpr float high_extreme = infinity;
    static constexpr bool has_infinity  = true;
};

template <>
struct DtypeLimits<TOSA_REF_TYPE_FP16>
{
    static constexpr half lowest       = std::numeric_limits<half>::lowest();
    static constexpr half max          = std::numeric_limits<half>::max();
    static constexpr half min          = std::numeric_limits<half>::min();
    static constexpr half denorm_min   = std::numeric_limits<half>::denorm_min();
    static constexpr half infinity     = std::numeric_limits<half>::infinity();
    static constexpr half low_extreme  = -infinity;
    static constexpr half high_extreme = infinity;
    static constexpr bool has_infinity = true;
};

template <>
struct DtypeLimits<TOSA_REF_TYPE_BF16>
{
    // We don't yet have an implementation of bf16 in the reference model. Return float.
    // - (2 ^ 8 - 1) * (2 ^ -7) * (2 & 127)
    static constexpr float lowest = -338953138925153547590470800371487866880.f;
    // (2 ^ 8 - 1) * (2 ^ -7) * (2 & 127)
    static constexpr float max = 338953138925153547590470800371487866880.f;
    // (2 ^ -126)
    static constexpr float min =
        1.1754943508222875079687365372222456778186655567720875215087517062784172594547271728515625e-38;
    // (2 ^ -133)
    static constexpr float denorm_min =
        9.18354961579912115600575419704879435795832466228193376178712270530013483949005603790283203125e-41;
    static constexpr float infinity     = std::numeric_limits<float>::infinity();
    static constexpr float low_extreme  = -infinity;
    static constexpr float high_extreme = infinity;
    static constexpr bool has_infinity  = true;
};

template <>
struct DtypeLimits<TOSA_REF_TYPE_INT32>
{
    static constexpr int32_t lowest       = std::numeric_limits<int32_t>::lowest();
    static constexpr int32_t max          = std::numeric_limits<int32_t>::max();
    static constexpr int32_t min          = std::numeric_limits<int32_t>::min();
    static constexpr int32_t denorm_min   = std::numeric_limits<int32_t>::denorm_min();
    static constexpr int32_t infinity     = std::numeric_limits<int32_t>::infinity();
    static constexpr int32_t low_extreme  = lowest;
    static constexpr int32_t high_extreme = max;
    static constexpr bool has_infinity    = false;
};

template <>
struct DtypeLimits<TOSA_REF_TYPE_INT16>
{
    static constexpr int16_t lowest       = std::numeric_limits<int16_t>::lowest();
    static constexpr int16_t max          = std::numeric_limits<int16_t>::max();
    static constexpr int16_t min          = std::numeric_limits<int16_t>::min();
    static constexpr int16_t denorm_min   = std::numeric_limits<int16_t>::denorm_min();
    static constexpr int16_t infinity     = std::numeric_limits<int16_t>::infinity();
    static constexpr int16_t low_extreme  = lowest;
    static constexpr int16_t high_extreme = max;
    static constexpr bool has_infinity    = false;
};

template <>
struct DtypeLimits<TOSA_REF_TYPE_INT8>
{
    static constexpr int8_t lowest       = std::numeric_limits<int8_t>::lowest();
    static constexpr int8_t max          = std::numeric_limits<int8_t>::max();
    static constexpr int8_t min          = std::numeric_limits<int8_t>::min();
    static constexpr int8_t denorm_min   = std::numeric_limits<int8_t>::denorm_min();
    static constexpr int8_t infinity     = std::numeric_limits<int8_t>::infinity();
    static constexpr int8_t low_extreme  = lowest;
    static constexpr int8_t high_extreme = max;
    static constexpr bool has_infinity   = false;
};

template <>
struct DtypeLimits<TOSA_REF_TYPE_BOOL>
{
    static constexpr bool lowest       = std::numeric_limits<bool>::lowest();
    static constexpr bool max          = std::numeric_limits<bool>::max();
    static constexpr bool min          = std::numeric_limits<bool>::min();
    static constexpr bool denorm_min   = std::numeric_limits<bool>::denorm_min();
    static constexpr bool infinity     = std::numeric_limits<bool>::infinity();
    static constexpr bool low_extreme  = lowest;
    static constexpr bool high_extreme = max;
    static constexpr bool has_infinity = false;
};

}    // namespace TosaReference

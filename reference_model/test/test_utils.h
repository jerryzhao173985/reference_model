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

#include "cfloat.h"
#include "half.hpp"
#include "tosa_generated.h"

using namespace ct;
using namespace tosa;

using fp8_e4m3 = cfloat_advanced<8, 4, FloatFeatures::HasNaN | FloatFeatures::HasDenorms>;
using fp8_e5m2 = cfloat_advanced<8, 5, float_support::AllFeats>;
using binary32 = cfloat_advanced<32, 8, float_support::AllFeats>;
using binary16 = cfloat_advanced<16, 5, float_support::AllFeats>;
using bfloat16 = cfloat_advanced<16, 8, float_support::AllFeats>;

template <typename T>
constexpr DType NativeType2Dtype()
{
    if constexpr (std::is_same<T, bool>::value)
        return DType_BOOL;

    if constexpr (std::is_same<T, int8_t>::value)
        return DType_INT8;

    if constexpr (std::is_same<T, uint8_t>::value)
        return DType_UINT8;

    if constexpr (std::is_same<T, int16_t>::value)
        return DType_INT16;

    if constexpr (std::is_same<T, uint16_t>::value)
        return DType_UINT16;

    if constexpr (std::is_same<T, int32_t>::value)
        return DType_INT32;

    if constexpr (std::is_same<T, binary16>::value)
        return DType_FP16;

    if constexpr (std::is_same<T, half_float::half>::value)
        return DType_FP16;

    if constexpr (std::is_same<T, float>::value)
        return DType_FP32;

    if constexpr (std::is_same<T, binary32>::value)
        return DType_FP32;

    if constexpr (std::is_same<T, bfloat16>::value)
        return DType_BF16;

    if constexpr (std::is_same<T, fp8_e5m2>::value)
        return DType_FP8E5M2;

    if constexpr (std::is_same<T, fp8_e4m3>::value)
        return DType_FP8E4M3;

    return DType_UNKNOWN;
}

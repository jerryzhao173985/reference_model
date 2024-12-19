// Copyright (c) 2024-2025, ARM Limited.
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
#include "generate.h"
#include "generate_utils.h"
#include "half.hpp"

#include <algorithm>
#include <array>
#include <iterator>
#include <type_traits>
#include <vector>

using namespace TosaReference;

namespace
{
template <typename StorageType, TOSA_REF_TYPE TosaRefType>
bool copyFixedDataFP(const int64_t elements,
                     const std::vector<int32_t> inData,
                     StorageType* outData,
                     bool broadcastMode)
{
    // Copy the input int32 data and cast it to the required float type

    if (broadcastMode)
    {
        for (auto t = 0; t < elements; t++)
        {
            outData[t] = static_cast<StorageType>(inData[0]);
        }
    }
    else
    {
        for (auto t = 0; t < elements; t++)
        {
            outData[t] = static_cast<StorageType>(inData[t]);
        }
    }
    return true;
}

template <typename StorageType, TOSA_REF_TYPE TosaRefType>
bool copyFixedDataINT(const int64_t elements,
                      const std::vector<int32_t> inData,
                      StorageType* outData,
                      bool broadcastMode)
{
    // Copy the input int32 data and write it out as the required integer value

    if (broadcastMode)
    {
        for (auto t = 0; t < elements; t++)
        {
            writeValue<StorageType, TosaRefType>(static_cast<int64_t>(inData[0]), t, outData);
        }
    }
    else
    {
        for (auto t = 0; t < elements; t++)
        {
            writeValue<StorageType, TosaRefType>(static_cast<int64_t>(inData[t]), t, outData);
        }
    }
    return true;
}
}    // namespace

namespace TosaReference
{

bool generateFixedData(const GenerateConfig& cfg, void* data, size_t size)
{
    // Check we support the operator
    if (cfg.opType == Op::Op_UNKNOWN)
    {
        WARNING("[Generator][FD] Unknown operator.");
        return false;
    }

    std::vector<int32_t> inData = cfg.fixedDataInfo.data;
    const auto T                = numElementsFromShape(cfg.shape);
    const int64_t inSize        = static_cast<int64_t>(inData.size());
    const bool broadcastMode    = (inSize == 1);
    // Check data size matches tensor size or it is 1 so that we can broadcast the values
    if (T != inSize && !broadcastMode)
    {
        WARNING("[Generator][FD] Given data size %d is not broadcastable or does not match output size %d.",
                inData.size(), T);
        return false;
    }

    switch (cfg.dataType)
    {
        case DType_INT48: {
            int8_t* outData = reinterpret_cast<int8_t*>(data);
            return copyFixedDataINT<int8_t, TOSA_REF_TYPE_INT48>(T, inData, outData, broadcastMode);
        }
        case DType_SHAPE:
            [[fallthrough]];
        case DType_INT32: {
            int32_t* outData = reinterpret_cast<int32_t*>(data);
            return copyFixedDataINT<int32_t, TOSA_REF_TYPE_INT32>(T, inData, outData, broadcastMode);
        }
        case DType_INT16: {
            int16_t* outData = reinterpret_cast<int16_t*>(data);
            return copyFixedDataINT<int16_t, TOSA_REF_TYPE_INT16>(T, inData, outData, broadcastMode);
        }
        case DType_INT8: {
            int8_t* outData = reinterpret_cast<int8_t*>(data);
            return copyFixedDataINT<int8_t, TOSA_REF_TYPE_INT8>(T, inData, outData, broadcastMode);
        }
        case DType_INT4: {
            int8_t* outData = reinterpret_cast<int8_t*>(data);
            return copyFixedDataINT<int8_t, TOSA_REF_TYPE_INT4>(T, inData, outData, broadcastMode);
        }
        case DType_BOOL: {
            int8_t* outData = reinterpret_cast<int8_t*>(data);
            return copyFixedDataINT<int8_t, TOSA_REF_TYPE_BOOL>(T, inData, outData, broadcastMode);
        }
        case DType_FP16: {
            half_float::half* outData = reinterpret_cast<half_float::half*>(data);
            return copyFixedDataFP<half_float::half, TOSA_REF_TYPE_FP16>(T, inData, outData, broadcastMode);
        }
        case DType_FP32: {
            float* outData = reinterpret_cast<float*>(data);
            return copyFixedDataFP<float, TOSA_REF_TYPE_FP32>(T, inData, outData, broadcastMode);
        }
        case DType_BF16: {
            bf16* outData = reinterpret_cast<bf16*>(data);
            return copyFixedDataFP<bf16, TOSA_REF_TYPE_BF16>(T, inData, outData, broadcastMode);
        }
        case DType_FP8E4M3: {
            fp8e4m3* outData = reinterpret_cast<fp8e4m3*>(data);
            return copyFixedDataFP<fp8e4m3, TOSA_REF_TYPE_FP8E4M3>(T, inData, outData, broadcastMode);
        }
        case DType_FP8E5M2: {
            fp8e5m2* outData = reinterpret_cast<fp8e5m2*>(data);
            return copyFixedDataFP<fp8e5m2, TOSA_REF_TYPE_FP8E5M2>(T, inData, outData, broadcastMode);
        }
        default:
            WARNING("[Generator][FD] Unsupported type %s.", EnumNameDType(cfg.dataType));
            return false;
    }
}
}    // namespace TosaReference

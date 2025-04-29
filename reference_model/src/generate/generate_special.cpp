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

#include "generate_special.h"
#include "dtype_limits.h"
#include "generate_special_utils.h"

#include <map>
#include <random>

using namespace TosaReference;

namespace TosaReference
{
template <typename StorageType, typename DataType, TOSA_REF_TYPE TosaRefType>
bool generate(const GenerateConfig& cfg, StorageType* data, size_t size, const SpecialGenProfile& specialGen)
{
    const SpecialInfo& fsinfo = cfg.specialInfo;
    uint8_t startIndex        = fsinfo.startIndex;

    // Unpack the specialGen struct for succinctness in the code that follows
    const auto& specialVals = specialGen.specialValues;
    const auto& opTestVals  = specialGen.opValues;
    TestValues vals         = specialGen.defaultValues;
    // Default mode is to repeat all values in the test values set
    SpecialTestSetMode mode = SpecialTestSetMode::REPEAT_ALL_VALUES;

    size_t inputIndex = 0;

    if (fsinfo.specialTestSet != SpecialTestSet::Default)
    {
        // Use a SpecialTestSet if present in the configuration.
        if (specialVals.find(fsinfo.specialTestSet) != specialVals.end())
        {
            vals = specialVals.at(fsinfo.specialTestSet).first;
            mode = specialVals.at(fsinfo.specialTestSet).second;
        }
        else
        {
            WARNING("[Generator][S] SpecialTestSet values are not defined.");
            return false;
        }
    }
    else if (opTestVals.find(cfg.opType) != opTestVals.end())
    {
        // When no special test set is defined, if an op has an entry in testValues
        // we use its op specific special test values, otherwise default values are used
        vals       = opTestVals.at(cfg.opType);
        inputIndex = static_cast<size_t>(cfg.inputPos);
    }

    auto rng     = RandomGen<DataType>(static_cast<uint64_t>(fsinfo.rngSeed));
    const auto T = numElementsFromShape(cfg.shape);
    // Make sure start index is within the number of elements
    // for modes like repeat last value
    startIndex = static_cast<uint8_t>(startIndex % T);
    for (int64_t t = 0; t < T; ++t)
    {
        int64_t valsIndex;
        switch (mode)
        {
            case SpecialTestSetMode::REPEAT_LAST_VALUE:
                // Repeat the last value in the test set
                valsIndex = t - startIndex;
                if (valsIndex < 0 || valsIndex >= static_cast<int64_t>(vals.size()))
                    valsIndex = static_cast<int64_t>(vals.size() - 1);
                break;
            case SpecialTestSetMode::REPEAT_ALL_VALUES:
                // Repeat all the values in the test set
                valsIndex = (t + startIndex) % static_cast<int64_t>(vals.size());
                break;
            default:
                ASSERT_MSG(false, "Unknown test set repeat mode");
                valsIndex = 0;
                break;
        }
        auto value = vals[static_cast<size_t>(valsIndex)].at(inputIndex).evaluate<TosaRefType, DataType>(rng);
        if constexpr (std::is_integral<StorageType>())
        {
            // Support for packed formats like INT4 & INT48
            writeValue<StorageType, TosaRefType>(static_cast<int64_t>(value), t, data);
        }
        else
        {
            data[t] = value;
        }
    }
    return true;
}

bool generateSpecial(const GenerateConfig& cfg, void* data, size_t size)
{
    // Check we support the operator
    if (cfg.opType == Op::Op_UNKNOWN)
    {
        WARNING("[Generator][S] Unknown operator.");
        return false;
    }

    switch (cfg.dataType)
    {
        case DType::DType_FP16: {
            float16* outData = reinterpret_cast<float16*>(data);
            return generate<float16, float16, TOSA_REF_TYPE_FP16>(cfg, outData, size,
                                                                  getSpecialConfig<SpecialConfig::FP>());
        }
        case DType::DType_FP32: {
            float* outData = reinterpret_cast<float*>(data);
            return generate<float, float, TOSA_REF_TYPE_FP32>(cfg, outData, size,
                                                              getSpecialConfig<SpecialConfig::FP>());
        }
        case DType::DType_BF16: {
            bf16* outData = reinterpret_cast<bf16*>(data);
            return generate<bf16, bf16, TOSA_REF_TYPE_BF16>(cfg, outData, size, getSpecialConfig<SpecialConfig::FP>());
        }
        case DType::DType_FP8E4M3: {
            fp8e4m3* outData = reinterpret_cast<fp8e4m3*>(data);
            return generate<fp8e4m3, fp8e4m3, TOSA_REF_TYPE_FP8E4M3>(cfg, outData, size,
                                                                     getSpecialConfig<SpecialConfig::FP>());
        }
        case DType::DType_FP8E5M2: {
            fp8e5m2* outData = reinterpret_cast<fp8e5m2*>(data);
            return generate<fp8e5m2, fp8e5m2, TOSA_REF_TYPE_FP8E5M2>(cfg, outData, size,
                                                                     getSpecialConfig<SpecialConfig::FP>());
        }
        case DType::DType_INT4: {
            int8_t* outData = reinterpret_cast<int8_t*>(data);
            return generate<int8_t, int8_t, TOSA_REF_TYPE_INT4>(cfg, outData, size,
                                                                getSpecialConfig<SpecialConfig::INT>());
        }
        case DType::DType_INT8: {
            if (cfg.unsignedData)
            {
                uint8_t* outData = reinterpret_cast<uint8_t*>(data);
                return generate<uint8_t, uint8_t, TOSA_REF_TYPE_UINT8>(cfg, outData, size,
                                                                       getSpecialConfig<SpecialConfig::INT>());
            }
            else
            {
                int8_t* outData = reinterpret_cast<int8_t*>(data);
                return generate<int8_t, int8_t, TOSA_REF_TYPE_INT8>(cfg, outData, size,
                                                                    getSpecialConfig<SpecialConfig::INT>());
            }
        }
        case DType::DType_INT16: {
            if (cfg.unsignedData)
            {
                uint16_t* outData = reinterpret_cast<uint16_t*>(data);
                return generate<uint16_t, uint16_t, TOSA_REF_TYPE_UINT16>(cfg, outData, size,
                                                                          getSpecialConfig<SpecialConfig::INT>());
            }
            else
            {
                int16_t* outData = reinterpret_cast<int16_t*>(data);
                return generate<int16_t, int16_t, TOSA_REF_TYPE_INT16>(cfg, outData, size,
                                                                       getSpecialConfig<SpecialConfig::INT>());
            }
        }
        case DType::DType_INT32: {
            int32_t* outData = reinterpret_cast<int32_t*>(data);
            return generate<int32_t, int32_t, TOSA_REF_TYPE_INT32>(cfg, outData, size,
                                                                   getSpecialConfig<SpecialConfig::INT>());
        }
        case DType::DType_INT48: {
            int8_t* outData = reinterpret_cast<int8_t*>(data);
            return generate<int8_t, int64_t, TOSA_REF_TYPE_INT48>(cfg, outData, size,
                                                                  getSpecialConfig<SpecialConfig::INT>());
        }
        case DType::DType_BOOL: {
            int8_t* outData = reinterpret_cast<int8_t*>(data);
            return generate<int8_t, int8_t, TOSA_REF_TYPE_BOOL>(cfg, outData, size,
                                                                getSpecialConfig<SpecialConfig::INT>());
        }
        default:
            WARNING("[Generator][S] Unsupported type %s.", EnumNameDType(cfg.dataType));
            return false;
    }
}
}    // namespace TosaReference

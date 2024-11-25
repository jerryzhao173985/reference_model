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

#include "generate_special.h"
#include "dtype_limits.h"
#include "generate_special_utils.h"
#include "half.hpp"

#include <map>
#include <random>

using namespace TosaReference;

namespace TosaReference
{
template <typename DataType>
bool generate(const TosaReference::GenerateConfig& cfg,
              DataType* data,
              size_t size,
              const SpecialGenProfile& specialGen)
{
    const TosaReference::SpecialInfo& fsinfo = cfg.specialInfo;
    uint8_t startIndex                       = fsinfo.startIndex;

    // Unpack the specialGen struct for succintness in the code that follows
    const auto& specialVals = specialGen.specialValues;
    const auto& opTestVals  = specialGen.opValues;
    TestValues vals         = specialGen.defaultValues;

    size_t inputIndex = 0;

    if (fsinfo.specialTestSet != SpecialTestSet::Default)
    {
        // Use a SpecialTestSet if present in the configuration.
        if (specialVals.find(fsinfo.specialTestSet) != specialVals.end())
        {
            vals = specialVals.at(fsinfo.specialTestSet);
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
        inputIndex = cfg.inputPos;
    }

    auto rng     = RandomGen<DataType>(fsinfo.rngSeed);
    const auto T = TosaReference::numElementsFromShape(cfg.shape);
    for (auto t = 0; t < T; ++t)
    {
        int valsIndex = (t + startIndex) % vals.size();
        data[t]       = vals[valsIndex].at(inputIndex).evaluate<DataType>(rng);
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
            half_float::half* outData = reinterpret_cast<half_float::half*>(data);
            return generate(cfg, outData, size, getSpecialConfig<SpecialConfig::FP>());
        }
        case DType::DType_FP32: {
            float* outData = reinterpret_cast<float*>(data);
            return generate(cfg, outData, size, getSpecialConfig<SpecialConfig::FP>());
        }
        case DType::DType_BF16: {
            bf16* outData = reinterpret_cast<bf16*>(data);
            return generate(cfg, outData, size, getSpecialConfig<SpecialConfig::FP>());
        }
        case DType::DType_FP8E4M3: {
            fp8e4m3* outData = reinterpret_cast<fp8e4m3*>(data);
            return generate(cfg, outData, size, getSpecialConfig<SpecialConfig::FP>());
        }
        case DType::DType_FP8E5M2: {
            fp8e5m2* outData = reinterpret_cast<fp8e5m2*>(data);
            return generate(cfg, outData, size, getSpecialConfig<SpecialConfig::FP>());
        }
        case DType::DType_INT8: {
            int8_t* outData = reinterpret_cast<int8_t*>(data);
            return generate(cfg, outData, size, getSpecialConfig<SpecialConfig::INT>());
        }
        case DType::DType_INT16: {
            int16_t* outData = reinterpret_cast<int16_t*>(data);
            return generate(cfg, outData, size, getSpecialConfig<SpecialConfig::INT>());
        }
        case DType::DType_INT32: {
            int32_t* outData = reinterpret_cast<int32_t*>(data);
            return generate(cfg, outData, size, getSpecialConfig<SpecialConfig::INT>());
        }
        case DType::DType_UINT8: {
            uint8_t* outData = reinterpret_cast<uint8_t*>(data);
            return generate(cfg, outData, size, getSpecialConfig<SpecialConfig::INT>());
        }
        case DType::DType_UINT16: {
            uint16_t* outData = reinterpret_cast<uint16_t*>(data);
            return generate(cfg, outData, size, getSpecialConfig<SpecialConfig::INT>());
        }
        default:
            WARNING("[Generator][S] Unsupported type.");
            return false;
    }
}
}    // namespace TosaReference

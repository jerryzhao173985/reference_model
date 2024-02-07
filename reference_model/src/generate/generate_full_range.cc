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

#include "generate_full_range.h"
#include "half.hpp"

namespace
{

template <typename DataType>
bool generate(const TosaReference::GenerateConfig& cfg, DataType* data, size_t size)
{
    const TosaReference::FullRangeInfo& frinfo = cfg.fullRangeInfo;
    DataType value                             = frinfo.startVal;

    const auto T = TosaReference::numElementsFromShape(cfg.shape);
    for (auto t = 0; t < T; ++t)
    {
        data[t] = value;
        value++;
    }
    return true;
}
}    // namespace

namespace TosaReference
{
bool generateFullRange(const GenerateConfig& cfg, void* data, size_t size)
{
    // Check we support the operator
    if (cfg.opType == Op::Op_UNKNOWN)
    {
        WARNING("[Generator][PR] Unknown operator.");
        return false;
    }

    switch (cfg.dataType)
    {
        case DType::DType_FP16: {
            uint16_t* outData = reinterpret_cast<uint16_t*>(data);
            return generate(cfg, outData, size);
        }
        default:
            WARNING("[Generator][PR] Unsupported type.");
            return false;
    }
}
}    // namespace TosaReference
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
#include "generate.h"
#include "generate_utils.h"

#include <algorithm>
#include <array>
#include <iterator>
#include <type_traits>
#include <vector>

namespace
{
template <typename OutType>
bool copyFixedData(const int64_t elements, const std::vector<int32_t> inData, OutType* outData)
{
    for (auto t = 0; t < elements; t++)
    {
        outData[t] = inData[t];
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
    const auto T                = TosaReference::numElementsFromShape(cfg.shape);
    if (T != static_cast<int64_t>(inData.size()))
    {
        WARNING("[Generator][FD] Given data size %d does not match output size %d.", inData.size(), T);
        return false;
    }

    switch (cfg.dataType)
    {
        case DType::DType_SHAPE: {
            int32_t* outData = reinterpret_cast<int32_t*>(data);
            return copyFixedData(T, inData, outData);
        }
        case DType::DType_INT8: {
            int8_t* outData = reinterpret_cast<int8_t*>(data);
            return copyFixedData(T, inData, outData);
        }
        default:
            WARNING("[Generator][FD] Unsupported type.");
            return false;
    }
}
}    // namespace TosaReference

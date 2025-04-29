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

#include "generate_full_range.h"
#include "cfloat.h"

using namespace std;

namespace
{
template <typename BinaryType>
void generateBinaryValues(const TosaReference::GenerateConfig& cfg, BinaryType* data, const int64_t elements)
{
    const TosaReference::FullRangeInfo& frinfo = cfg.fullRangeInfo;
    BinaryType value                           = static_cast<BinaryType>(frinfo.startVal);

    // Generate the full range of binary data values - wrapping as necessary
    // For floating point type representations this will enable testing of all values
    for (auto t = 0; t < elements; ++t)
    {
        data[t] = value;
        value++;
    }
}

template <typename DataType>
void zeroSubnorm(DataType* data, const int64_t elements)
{
    for (auto t = 0; t < elements; ++t)
    {
        if (fpclassify(data[t]) == FP_SUBNORMAL)
        {
            data[t] = static_cast<DataType>(0.0);
        }
    }
}
}    // namespace

namespace TosaReference
{
bool generateFullRange(const GenerateConfig& cfg, void* data, size_t size)
{
    // Check we support the operator
    if (cfg.opType == Op::Op_UNKNOWN)
    {
        WARNING("[Generator][FR] Unknown operator.");
        return false;
    }

    const auto elements = TosaReference::numElementsFromShape(cfg.shape);

    switch (cfg.dataType)
    {
        case DType::DType_FP16: {
            uint16_t* outBinaryData = reinterpret_cast<uint16_t*>(data);
            generateBinaryValues(cfg, outBinaryData, elements);

            // TODO: Re-enable subnorm testing
            // Skip sub-normal values as they are allowed to be flushed to zero
            // This is not currently supported by Conformance Testing
            float16* outData = reinterpret_cast<float16*>(data);
            zeroSubnorm(outData, elements);
            break;
        }
        case DType::DType_BF16: {
            uint16_t* outBinaryData = reinterpret_cast<uint16_t*>(data);
            generateBinaryValues(cfg, outBinaryData, elements);

            // TODO: Re-enable subnorm testing
            // Skip sub-normal values as they are allowed to be flushed to zero and
            // this is not currently supported by Conformance Testing
            bf16* outData = reinterpret_cast<bf16*>(data);
            zeroSubnorm(outData, elements);
            break;
        }
        case DType::DType_INT8: {
            uint8_t* outBinaryData = reinterpret_cast<uint8_t*>(data);
            generateBinaryValues(cfg, outBinaryData, elements);
            break;
        }
        case DType::DType_INT16: {
            uint16_t* outBinaryData = reinterpret_cast<uint16_t*>(data);
            generateBinaryValues(cfg, outBinaryData, elements);
            break;
        }
        default:
            WARNING("[Generator][FR] Unsupported type %s.", EnumNameDType(cfg.dataType));
            return false;
    }

    return true;
}
}    // namespace TosaReference

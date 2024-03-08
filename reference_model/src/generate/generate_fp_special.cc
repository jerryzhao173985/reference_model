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

#include "generate_fp_special.h"
#include "half.hpp"

#include <map>

namespace
{

class SpecialValue
{
public:
    enum SpecialValsEnum
    {
        Zero,
        Inf,
        NaN,
        Min,
        Max,
        One,
    };

    SpecialValue() = default;
    SpecialValue(SpecialValsEnum v)
        : value(v)
    {}
    operator SpecialValsEnum() const
    {
        return value;
    }
    SpecialValue& operator=(SpecialValsEnum v)
    {
        value = v;
        return *this;
    }
    bool operator==(const SpecialValsEnum v) const
    {
        return value == v;
    }
    bool operator!=(const SpecialValsEnum v) const
    {
        return value != v;
    }
    SpecialValue operator-()
    {
        negative = !negative;
        return *this;
    }

    template <typename DataType>
    DataType evaluate()
    {
        switch (value)
        {
            case Zero:
                return static_cast<DataType>(negative ? -0.0 : 0.0);
            case Inf:
                return negative ? -std::numeric_limits<DataType>::infinity()
                                : std::numeric_limits<DataType>::infinity();
            case NaN:
                return std::numeric_limits<DataType>::quiet_NaN();
            case Min:
                return negative ? -std::numeric_limits<DataType>::min() : std::numeric_limits<DataType>::min();
            case Max:
                return negative ? -std::numeric_limits<DataType>::max() : std::numeric_limits<DataType>::max();
            case One:
                return static_cast<DataType>(negative ? -1.0 : 1.0);
            default:
                WARNING("[Generator][FS] Uninitialised special value.");
                return static_cast<DataType>(0.0);
        }
    }

private:
    SpecialValsEnum value;
    bool negative = false;
};

/*
Test vals format

I: Number of inputs to an op - referenced by cfg.inputPos
T: Number of test cases defined for the op

vector of test inputs: {
    vector of values for test 0:   { valueForinputPos0, valueForinputPos1, ..., valueForinputPosI-1 },
    vector of values for test 1:   { valueForinputPos0, valueForinputPos1, ..., valueForinputPosI-1 },
    ...
    vector of values for test T-1: { valueForinputPos0, valueForinputPos1, ..., valueForinputPosI-1 },
}
*/
using TestValues = std::vector<std::vector<SpecialValue>>;

TestValues equalOpsTestVals{ { SpecialValue(SpecialValue::Zero), -SpecialValue(SpecialValue::Zero) },
                             { SpecialValue(SpecialValue::Inf), -SpecialValue(SpecialValue::Inf) } };

TestValues addTestVals{ { SpecialValue(SpecialValue::Max), SpecialValue(SpecialValue::One) },
                        { SpecialValue(SpecialValue::Inf), -SpecialValue(SpecialValue::Inf) } };

TestValues defaultTestVals{ { SpecialValue(SpecialValue::Zero) }, { -SpecialValue(SpecialValue::Zero) },
                            { SpecialValue(SpecialValue::Inf) },  { -SpecialValue(SpecialValue::Inf) },
                            { SpecialValue(SpecialValue::NaN) },  { SpecialValue(SpecialValue::Min) },
                            { SpecialValue(SpecialValue::Max) } };

std::map<Op, TestValues> testValues = { { Op::Op_EQUAL, equalOpsTestVals },
                                        { Op::Op_GREATER, equalOpsTestVals },
                                        { Op::Op_GREATER_EQUAL, equalOpsTestVals },
                                        { Op::Op_ADD, addTestVals } };

template <typename DataType>
bool generate(const TosaReference::GenerateConfig& cfg, DataType* data, size_t size)
{
    const TosaReference::FpSpecialInfo& fsinfo = cfg.fpSpecialInfo;
    uint8_t startIndex                         = fsinfo.startIndex;

    std::vector<DataType> values;
    auto testValuesResult = testValues.find(cfg.opType);
    TestValues opTestVals = defaultTestVals;
    size_t inputIndex     = 0;
    if (testValuesResult != testValues.end())
    {
        // When an op has an entry in testValues we use its op specific special test values, otherwise default values are used
        opTestVals = testValuesResult->second;
        inputIndex = cfg.inputPos;
    }

    for (std::vector<SpecialValue> inputs : opTestVals)
    {
        values.push_back(inputs[inputIndex].evaluate<DataType>());
    }

    const auto T = TosaReference::numElementsFromShape(cfg.shape);
    for (auto t = 0; t < T; ++t)
    {
        data[t] = values[(t + startIndex) % values.size()];
    }
    return true;
}
}    // namespace

namespace TosaReference
{
bool generateFpSpecial(const GenerateConfig& cfg, void* data, size_t size)
{
    // Check we support the operator
    if (cfg.opType == Op::Op_UNKNOWN)
    {
        WARNING("[Generator][FS] Unknown operator.");
        return false;
    }

    switch (cfg.dataType)
    {
        case DType::DType_FP16: {
            half_float::half* outData = reinterpret_cast<half_float::half*>(data);
            return generate(cfg, outData, size);
        }
        case DType::DType_FP32: {
            float* outData = reinterpret_cast<float*>(data);
            return generate(cfg, outData, size);
        }
        default:
            WARNING("[Generator][FS] Unsupported type.");
            return false;
    }
}
}    // namespace TosaReference
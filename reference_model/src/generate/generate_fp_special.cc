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
#include <random>

namespace
{
template <typename DataType>
class RandomGen
{
public:
    RandomGen(uint64_t seed)
        : _gen(seed)
    {}

    DataType getFloat(DataType min, DataType max)
    {
        if (!_rangeOk(min, max))
            return static_cast<DataType>(1.5);

        auto dis     = std::uniform_real_distribution<double>(static_cast<double>(min), static_cast<double>(max));
        DataType rnd = static_cast<DataType>(dis(_gen));
        return rnd;
    }

    DataType getInteger(DataType min, DataType max)
    {
        return _getTnteger(min, max, Any);
    }

    DataType getEvenInteger(DataType min, DataType max)
    {
        return _getTnteger(min, max, Even);
    }

    DataType getOddInteger(DataType min, DataType max)
    {
        return _getTnteger(min, max, Odd);
    }

private:
    enum _integerTypeEnum
    {
        Any,
        Even,
        Odd
    };

    bool _rangeOk(DataType min, DataType max)
    {
        if (min > max || min == max)
        {
            WARNING("[Generator][FS] Bad random range - min:%g max:%g", static_cast<double>(min),
                    static_cast<double>(max));
            return false;
        }
        return true;
    }

    bool _checkEven(DataType val)
    {
        if ((val != static_cast<DataType>(round(val))) || (val / 2.0 != round(val / 2.0)))
        {
            WARNING("[Generator][FS] Bad even integer generated - %g", static_cast<double>(val));
            return false;
        }
        return true;
    }

    bool _checkOdd(DataType val)
    {
        if ((val != static_cast<DataType>(round(val))) || (val / 2.0 == round(val / 2.0)))
        {
            WARNING("[Generator][FS] Bad odd integer generated - %g", static_cast<double>(val));
            return false;
        }
        return true;
    }

    DataType _getTnteger(DataType min, DataType max, _integerTypeEnum type)
    {
        // Set min/max to integers
        min = ceil(min);
        max = floor(max);

        if (!_rangeOk(min, max))
        {
            switch (type)
            {
                case Any:
                    return static_cast<DataType>(1.0);
                case Odd:
                    return static_cast<DataType>(3.0);
                case Even:
                    return static_cast<DataType>(2.0);
                default:
                    WARNING("[Generator][FS] Unsupported integer type.");
                    return static_cast<DataType>(0.0);
            }
        }

        switch (type)
        {
            case Any:
                return round(getFloat(min, max));
            case Odd: {
                DataType val =
                    static_cast<DataType>(round(round(getFloat(min, max - static_cast<DataType>(1.0))) / 2) * 2 + 1);
                // For large values of floating point (such as with FP8) there
                // may not be enough range to express an odd number - so check
                return _checkOdd(val) ? val : static_cast<DataType>(3.0);
            }
            case Even: {
                DataType val =
                    static_cast<DataType>(round(round(getFloat(min + static_cast<DataType>(1.0), max)) / 2) * 2);
                // For large values of floating point (such as with FP8) there
                // may not be enough range to express an even number - so check
                return _checkEven(val) ? val : static_cast<DataType>(2.0);
            }
            default:
                WARNING("[Generator][FS] Unsupported integer type.");
                return static_cast<DataType>(0.0);
        }
    }

    std::mt19937 _gen;
};

class SpecialValue
{
public:
    enum SpecialValsEnum
    {
        Zero,
        Inf,
        NaN,
        Min,    // Smallest positive normal floating point value
        Max,    // Largest positive floating point value
        One,
        Two,
        Ten,
        Euler,         // Floating point number
        Pythagorus,    // Floating point number
        MinDenorm,     // Smallest positive denormal floating point value
        ULPMax,        // To force overflows to infinity when added/subtracted
        RndFloat,
        RndInteger,
        RndEvenInteger,
        RndOddInteger
    };

    SpecialValue() = default;
    SpecialValue(SpecialValsEnum v)
        : _value(v)
    {}
    SpecialValue(SpecialValsEnum v, SpecialValsEnum rangeMin, SpecialValsEnum rangeMax)
        : _value(v)
        , _rangeMin(rangeMin)
        , _rangeMax(rangeMax)
    {}
    operator SpecialValsEnum() const
    {
        return _value;
    }
    SpecialValue& operator=(SpecialValsEnum v)
    {
        _value = v;
        return *this;
    }
    bool operator==(const SpecialValsEnum v) const
    {
        return _value == v;
    }
    bool operator!=(const SpecialValsEnum v) const
    {
        return _value != v;
    }
    SpecialValue operator-()
    {
        _negative = !_negative;
        return *this;
    }

    template <typename DataType>
    DataType evaluate(RandomGen<DataType> rng)
    {
        // Work out the simple values
        switch (_value)
        {
            case Zero:
            case Inf:
            case NaN:
            case Min:
            case Max:
            case One:
            case Two:
            case Ten:
            case Euler:
            case Pythagorus:
            case MinDenorm:
            case ULPMax:
                return _static_evaluate<DataType>(_value, _negative);
            default:
                // Handle the Random and unsupported cases below
                break;
        }
        // Must be random value, work out positive range
        auto min = _static_evaluate<DataType>(_rangeMin, false);
        auto max = _static_evaluate<DataType>(_rangeMax, false);

        DataType rnd;
        switch (_value)
        {
            case RndFloat:
                rnd = rng.getFloat(min, max);
                break;
            case RndInteger:
                rnd = rng.getInteger(min, max);
                break;
            case RndEvenInteger:
                rnd = rng.getEvenInteger(min, max);
                break;
            case RndOddInteger:
                rnd = rng.getOddInteger(min, max);
                break;
            default:
                WARNING("[Generator][FS] Unsupported special value type.");
                return static_cast<DataType>(0.0);
        }
        return _negative ? -rnd : rnd;
    }

private:
    template <typename DataType>
    DataType _static_evaluate(SpecialValsEnum v, bool negate)
    {
        // Work out the static value
        switch (v)
        {
            case Zero:
                return static_cast<DataType>(negate ? -0.0 : 0.0);
            case Inf:
                return negate ? -std::numeric_limits<DataType>::infinity() : std::numeric_limits<DataType>::infinity();
            case NaN:
                return std::numeric_limits<DataType>::quiet_NaN();
            case Min:
                return negate ? -std::numeric_limits<DataType>::min() : std::numeric_limits<DataType>::min();
            case Max:
                return negate ? -std::numeric_limits<DataType>::max() : std::numeric_limits<DataType>::max();
            case One:
                return static_cast<DataType>(negate ? -1.0 : 1.0);
            case Two:
                return static_cast<DataType>(negate ? -2.0 : 2.0);
            case Ten:
                return static_cast<DataType>(negate ? -10.0 : 10.0);
            case Euler:
                return static_cast<DataType>(negate ? -2.71828 : 2.71828);
            case Pythagorus:
                return static_cast<DataType>(negate ? -1.41421 : 1.41421);
            case MinDenorm:
                return negate ? -std::numeric_limits<DataType>::denorm_min()
                              : std::numeric_limits<DataType>::denorm_min();
            case ULPMax: {
                DataType max = std::numeric_limits<DataType>::max();
                DataType ulp = max - nextafter(max, static_cast<DataType>(0.0));
                return negate ? -ulp : ulp;
            }
            default:
                // Assumption that we only get called with a valid enum
                return static_cast<DataType>(0.0);
        }
    }

    SpecialValsEnum _value;
    SpecialValsEnum _rangeMin = SpecialValsEnum::One;
    SpecialValsEnum _rangeMax = SpecialValsEnum::Max;
    bool _negative            = false;
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
using SValue     = SpecialValue;
using SVE        = SpecialValue::SpecialValsEnum;

TestValues conditionalOpsTestVals{
    { SValue(SVE::Inf), SValue(SVE::Inf) },      { -SValue(SVE::Inf), -SValue(SVE::Inf) },
    { SValue(SVE::Inf), -SValue(SVE::Inf) },     { -SValue(SVE::Inf), SValue(SVE::Inf) },
    { -SValue(SVE::Zero), SValue(SVE::Zero) },   { SValue(SVE::Zero), -SValue(SVE::Zero) },
    { SValue(SVE::NaN), SValue(SVE::RndFloat) }, { SValue(SVE::RndFloat), SValue(SVE::NaN) },
    { SValue(SVE::NaN), SValue(SVE::Inf) },      { SValue(SVE::Inf), SValue(SVE::NaN) },
    { SValue(SVE::NaN), -SValue(SVE::Inf) },     { -SValue(SVE::Inf), SValue(SVE::NaN) },
    { SValue(SVE::NaN), SValue(SVE::NaN) },
};

TestValues addTestVals{ { SValue(SVE::RndFloat, SVE::ULPMax, SVE::Max), SValue(SVE::Max) },
                        { -SValue(SVE::Max), -SValue(SVE::RndFloat, SVE::ULPMax, SVE::Max) },
                        { SValue(SVE::Inf), -SValue(SVE::Inf) },
                        { SValue(SVE::Inf), SValue(SVE::Inf) },
                        { -SValue(SVE::Inf), -SValue(SVE::Inf) },
                        { SValue(SVE::Inf), SValue(SVE::RndFloat) },
                        { SValue(SVE::RndFloat), -SValue(SVE::Inf) },
                        { SValue(SVE::NaN), SValue(SVE::RndFloat) },
                        { SValue(SVE::RndFloat), SValue(SVE::NaN) } };

TestValues subTestVals{ { SValue(SVE::Max), -SValue(SVE::RndFloat, SVE::ULPMax, SVE::Max) },
                        { -SValue(SVE::Max), SValue(SVE::RndFloat, SVE::ULPMax, SVE::Max) },
                        { SValue(SVE::Inf), SValue(SVE::Inf) },
                        { -SValue(SVE::Inf), -SValue(SVE::Inf) },
                        { SValue(SVE::Inf), -SValue(SVE::Inf) },
                        { -SValue(SVE::Inf), SValue(SVE::Inf) },
                        { SValue(SVE::Inf), SValue(SVE::RndFloat) },
                        { -SValue(SVE::Inf), SValue(SVE::RndFloat) },
                        { SValue(SVE::RndFloat), SValue(SVE::Inf) },
                        { SValue(SVE::RndFloat), -SValue(SVE::Inf) },
                        { SValue(SVE::NaN), SValue(SVE::RndFloat) },
                        { SValue(SVE::RndFloat), SValue(SVE::NaN) } };

TestValues mulTestVals{ { SValue(SVE::Max), SValue(SVE::RndFloat, SVE::Two, SVE::Max) },
                        { -SValue(SVE::Max), -SValue(SVE::RndFloat, SVE::Two, SVE::Max) },
                        { -SValue(SVE::Max), SValue(SVE::RndFloat, SVE::Two, SVE::Ten) },
                        { SValue(SVE::Max), -SValue(SVE::RndFloat, SVE::Two, SVE::Ten) },
                        { SValue(SVE::Inf), SValue(SVE::Zero) },
                        { -SValue(SVE::Inf), SValue(SVE::Zero) },
                        { SValue(SVE::Inf), -SValue(SVE::Zero) },
                        { -SValue(SVE::Inf), -SValue(SVE::Zero) },
                        { SValue(SVE::Inf), SValue(SVE::Inf) },
                        { -SValue(SVE::Inf), -SValue(SVE::Inf) },
                        { SValue(SVE::Inf), -SValue(SVE::Inf) },
                        { -SValue(SVE::Inf), SValue(SVE::Inf) },
                        { SValue(SVE::NaN), SValue(SVE::RndFloat) },
                        { SValue(SVE::RndFloat), SValue(SVE::NaN) } };

TestValues powTestVals{ { -SValue(SVE::RndFloat, SVE::Min, SVE::Max), SValue(SVE::Euler) },
                        { -SValue(SVE::RndFloat, SVE::Min, SVE::Max), SValue(SVE::Pythagorus) },
                        { SValue(SVE::Max), SValue(SVE::RndFloat, SVE::Two, SVE::Max) },
                        { -SValue(SVE::Max), SValue(SVE::RndOddInteger, SVE::One, SVE::Ten) },
                        { -SValue(SVE::Max), SValue(SVE::RndEvenInteger, SVE::One, SVE::Ten) },
                        { SValue(SVE::Zero), SValue(SVE::One) },
                        { -SValue(SVE::Zero), SValue(SVE::One) },
                        { SValue(SVE::Zero), SValue(SVE::Two) },
                        { -SValue(SVE::Zero), SValue(SVE::Two) },
                        /* TODO: Missing infinity tests - need spec clarification */
                        { SValue(SVE::NaN), SValue(SVE::RndFloat) },
                        { SValue(SVE::RndFloat), SValue(SVE::NaN) } };

TestValues minMaxTestVals{ { SValue(SVE::Zero), -SValue(SVE::Zero) },
                           { SValue(SVE::Inf), -SValue(SVE::Inf) },
                           { SValue(SVE::Min), -SValue(SVE::Min) },
                           { SValue(SVE::Max), -SValue(SVE::Max) },
                           /* TODO: Add denorm numbers - need spec clarification */
                           { SValue(SVE::RndFloat), SValue(SVE::NaN) },
                           { SValue(SVE::NaN), -SValue(SVE::RndFloat) } };

TestValues defaultTestVals{ { SValue(SVE::Zero) },       { -SValue(SVE::Zero) }, { SValue(SVE::Inf) },
                            { -SValue(SVE::Inf) },       { SValue(SVE::Min) },   { -SValue(SVE::Min) },
                            { SValue(SVE::Max) },        { -SValue(SVE::Max) },  { SValue(SVE::MinDenorm) },
                            { -SValue(SVE::MinDenorm) }, { SValue(SVE::One) },   { -SValue(SVE::One) },
                            { SValue(SVE::NaN) } };

TestValues dotProductTestVals{
    { SValue(SVE::Zero), -SValue(SVE::Zero), SValue(SVE::Zero) },
    { SValue(SVE::Inf), -SValue(SVE::Inf), SValue(SVE::One) },
    { SValue(SVE::NaN), SValue(SVE::One), SValue(SVE::One) },
    { SValue(SVE::Min), SValue(SVE::Min), SValue(SVE::Zero) },
    { SValue(SVE::One), SValue(SVE::Min), SValue(SVE::Min) },
};

std::map<Op, TestValues> testValues = {
    { Op::Op_EQUAL, conditionalOpsTestVals },
    { Op::Op_GREATER, conditionalOpsTestVals },
    { Op::Op_GREATER_EQUAL, conditionalOpsTestVals },
    { Op::Op_ADD, addTestVals },
    { Op::Op_MAXIMUM, minMaxTestVals },
    { Op::Op_MINIMUM, minMaxTestVals },
    { Op::Op_MUL, mulTestVals },
    { Op::Op_POW, powTestVals },
    { Op::Op_SUB, subTestVals },
    { Op::Op_CONV2D, dotProductTestVals },
    { Op::Op_CONV3D, dotProductTestVals },
    { Op::Op_DEPTHWISE_CONV2D, dotProductTestVals },
    { Op::Op_TRANSPOSE_CONV2D, dotProductTestVals },
    { Op::Op_AVG_POOL2D, dotProductTestVals },
    { Op::Op_MATMUL, dotProductTestVals },
    { Op::Op_REDUCE_SUM, dotProductTestVals },
    { Op::Op_REDUCE_PRODUCT, dotProductTestVals },
};

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

    auto rng = RandomGen<DataType>(fsinfo.rngSeed);
    for (std::vector<SpecialValue> inputs : opTestVals)
    {
        DataType val;
        val = inputs[inputIndex].evaluate<DataType>(rng);
        values.push_back(val);
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
        case DType::DType_BF16: {
            bf16* outData = reinterpret_cast<bf16*>(data);
            return generate(cfg, outData, size);
        }
        case DType::DType_FP8E4M3: {
            fp8e4m3* outData = reinterpret_cast<fp8e4m3*>(data);
            return generate(cfg, outData, size);
        }
        case DType::DType_FP8E5M2: {
            fp8e5m2* outData = reinterpret_cast<fp8e5m2*>(data);
            return generate(cfg, outData, size);
        }
        default:
            WARNING("[Generator][FS] Unsupported type.");
            return false;
    }
}
}    // namespace TosaReference
